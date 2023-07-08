"""
This code implements outlier-free point set registration as torch function

"""
import time
import torch
import open3d as o3d
from pytorch3d import transforms, ops

from utils.evaluation_metrics import translation_error, rotation_matrix_error
from utils.math_utils import make_se3_batched


def wahba(source_points, target_points, device_=None):
    """
    inputs:
    source_points: torch.tensor of shape (B, 3, N)
    target_points: torch.tensor of shape (B, 3, N)

    where
        B = batch size
        N = number of points in each point set

    output:
    R   : torch.tensor of shape (B, 3, 3)
    """
    batch_size = source_points.shape[0]

    if device_ == None:
        device_ = source_points.device

    mat = target_points @ source_points.transpose(-1, -2)  # (B, 3, 3)
    U, S, Vh = torch.linalg.svd(mat)

    D = torch.eye(3).to(device=device_).to(dtype=source_points.dtype)  # (3, 3)
    D = D.unsqueeze(0)  # (1, 3, 3)
    D = D.repeat(batch_size, 1, 1)  # (B, 3, 3)

    D[:, 2, 2] = torch.linalg.det(U) * torch.linalg.det(Vh)

    return U @ D @ Vh  # (B, 3, 3)


def solve_outlierfree_reg(source_points, target_points, weights=None, device_=None):
    """ Solve outlier free registration problem """
    N = source_points.shape[-1]
    if device_ == None:
        device_ = source_points.device

    if weights == None:
        ttype = target_points.dtype
        weights = torch.ones((1, N), device=device_).to(dtype=ttype)

    source_points_ave = torch.einsum('bdn,ln->bd', source_points, weights) / weights.sum()  # (B, 3)
    target_points_ave = torch.einsum('bdn,ln->bd', target_points, weights) / weights.sum()  # (B, 3)

    # getting the rotation
    source_points_centered = source_points - source_points_ave.unsqueeze(-1)  # (B, 3, N)
    target_points_centered = target_points - target_points_ave.unsqueeze(-1)  # (B, 3, N)

    source_points_centered = torch.einsum('bdn,ln->bdn', source_points_centered, weights)  # (B, 3, N)
    target_points_centered = torch.einsum('bdn,ln->bdn', target_points_centered, weights)  # (B, 3, N)
    rotation = wahba(source_points=source_points_centered, target_points=target_points_centered)

    # getting the translation
    translation = target_points_ave.unsqueeze(-1) - rotation @ source_points_ave.unsqueeze(-1)

    return rotation, translation


def solve_registration(source_points,
                       target_points,
                       noise_bound,
                       sample_size=4,
                       max_iters=5000,
                       use_icp=False,
                       use_ransac=False,
                       icp_method='o3d_p2p',
                       icp_threshold=0.05,
                       input_pc=None,
                       model_pc=None,
                       device_=None):
    """ Use RANSAC w/ solvers to solve robust registration problems
    """
    batch_size, N = source_points.shape[0], source_points.shape[-1]
    if device_ == None:
        device_ = source_points.device

    def ideal_ransac_iter(desired_probability, outlier_probability, sample_size):
        n = torch.ceil(torch.log(1 - desired_probability)
                       / torch.log(1 - (1 - outlier_probability) ** sample_size + 1e-10))
        n[n < 0] = torch.inf
        return n

    def residual_fun(R_est, t_est, source_pts, target_pts):
        """ Calculate reprojected residuals """
        transformed_model_est = R_est @ source_pts + t_est
        residuals = torch.linalg.norm(transformed_model_est - target_pts, axis=1)
        return residuals

    start_time = time.time()
    if use_ransac:
        print("Use RANSAC.")
        best_result = None
        best_inlier_ratio = torch.zeros(batch_size, device=device_)
        best_inliers = torch.zeros((batch_size, N), device=device_, dtype=torch.bool)
        desired_probabilities = torch.ones(batch_size, device=device_) * 0.99
        continue_ransac_iters_flags = torch.ones(batch_size, device=device_)
        total_iters = torch.zeros(batch_size, device=device_, dtype=torch.int)
        for iter in range(max_iters):
            sampled_indices = torch.multinomial(input=torch.ones(N),
                                                num_samples=sample_size,
                                                replacement=False)

            # run solver
            result = solve_outlierfree_reg(source_points=source_points[:, :, sampled_indices],
                                           target_points=target_points[:, :, sampled_indices],
                                           device_=device_)

            if best_result is None:
                best_result = result

            # build consensus set
            c_residuals = residual_fun(result[0], result[1], source_points, target_points)
            inlier_indices = torch.le(c_residuals, noise_bound)
            num_inliers = torch.count_nonzero(inlier_indices, dim=-1)
            inlier_ratio = num_inliers / N

            best_inlier_update_flags = inlier_ratio > best_inlier_ratio

            # update best result
            best_result[0][best_inlier_update_flags, ...] = result[0][best_inlier_update_flags, ...]
            best_result[1][best_inlier_update_flags, ...] = result[1][best_inlier_update_flags, ...]

            # update best inlier ratio
            best_inlier_ratio[best_inlier_update_flags] = inlier_ratio[best_inlier_update_flags]

            # update best inlier indices
            best_inliers[best_inlier_update_flags, ...] = inlier_indices[best_inlier_update_flags, ...]

            # update max iters
            est_outlier_ratio = 1 - inlier_ratio
            adapt_max_itr = ideal_ransac_iter(desired_probabilities, est_outlier_ratio, sample_size)
            adapt_max_itr[adapt_max_itr > max_iters] = max_iters

            # update continue ransac flags & break if all not continue
            to_be_stopped = adapt_max_itr < iter + 1
            total_iters[to_be_stopped] = iter + 1
            continue_ransac_iters_flags[to_be_stopped] = False
            if not torch.any(continue_ransac_iters_flags):
                break

        # final solve
        R_est, t_est = [], []
        for b in range(batch_size):
            c_inliers = best_inliers[b, ...]
            cR, ct = solve_outlierfree_reg(source_points=source_points[b, :, c_inliers].unsqueeze(0),
                                           target_points=target_points[b, :, c_inliers].unsqueeze(0),
                                           device_=device_)
            R_est.append(cR.squeeze(0))
            t_est.append(ct.squeeze(0))
        batched_arun_R, batched_arun_t = torch.stack(R_est), torch.stack(t_est)
        arun_residuals = residual_fun(batched_arun_R, batched_arun_t, source_points, target_points)
        arun_costs = torch.mean(arun_residuals, dim=-1)
    else:
        print("Not using RANSAC.")
        R_est, t_est = [], []
        for b in range(batch_size):
            cR, ct = solve_outlierfree_reg(source_points=source_points[b, :, :].unsqueeze(0),
                                           target_points=target_points[b, :, :].unsqueeze(0),
                                           device_=device_)
            R_est.append(cR.squeeze(0))
            t_est.append(ct.squeeze(0))
        batched_arun_R, batched_arun_t = torch.stack(R_est), torch.stack(t_est)
        arun_residuals = residual_fun(batched_arun_R, batched_arun_t, source_points, target_points)
        arun_costs = torch.mean(arun_residuals, dim=-1)

    # icp solve
    if use_icp and icp_method == "pytorch3d":
        init_T = ops.points_alignment.SimilarityTransform(R=batched_arun_R,
                                                          T=batched_arun_t.squeeze(-1),
                                                          s=torch.ones(batched_arun_t.shape[0], ).to(device_))
        sol = ops.iterative_closest_point(X=model_pc.transpose(1, 2),
                                          Y=input_pc.transpose(1, 2),
                                          init_transform=init_T,
                                          max_iterations=500,
                                          verbose=False)

        batched_icp_R, batched_icp_t = sol.RTs.R, sol.RTs.T.unsqueeze(-1)
        icp_residuals = residual_fun(batched_icp_R, batched_icp_t, source_points, target_points)
        icp_costs = torch.mean(icp_residuals, dim=-1)
    elif use_icp and icp_method == "o3d_p2p":
        print("Use Open3D P2P ICP")
        init_T = make_se3_batched(batched_arun_R, batched_arun_t).cpu().numpy()
        batched_icp_R, batched_icp_t = [], []
        for b in range(batch_size):
            src_points = model_pc[b, ...].T.cpu().numpy()
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(src_points)

            tgt_points = input_pc[b, ...].T.cpu().numpy()
            tgt_pcd = o3d.geometry.PointCloud()
            tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)

            icp_sol = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, icp_threshold, init_T[b, ...],
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            cR, ct = icp_sol.transformation[:3, :3], icp_sol.transformation[:3, -1]
            batched_icp_R.append(torch.tensor(cR))
            batched_icp_t.append(torch.tensor(ct))
        batched_icp_R, batched_icp_t = torch.stack(batched_icp_R).to(device_), torch.stack(batched_icp_t).to(
            device_).unsqueeze(-1)
        icp_residuals = residual_fun(batched_icp_R, batched_icp_t, source_points, target_points)
        icp_costs = torch.mean(icp_residuals, dim=-1)
    elif use_icp and icp_method == "o3d_p2l":
        print("Use Open3D P2L ICP")
        init_T = make_se3_batched(batched_arun_R, batched_arun_t).cpu().numpy()
        batched_icp_R, batched_icp_t = [], []
        for b in range(batch_size):
            src_points = model_pc[b, ...].T.cpu().numpy()
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(src_points)

            tgt_points = input_pc[b, ...].T.cpu().numpy()
            tgt_pcd = o3d.geometry.PointCloud()
            tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)
            tgt_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=15))

            loss = o3d.pipelines.registration.TukeyLoss(k=noise_bound)
            p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

            icp_sol = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, noise_bound * 10, init_T[b, ...],
                p2l)
            cR, ct = icp_sol.transformation[:3, :3], icp_sol.transformation[:3, -1]
            batched_icp_R.append(torch.tensor(cR))
            batched_icp_t.append(torch.tensor(ct))
        batched_icp_R, batched_icp_t = torch.stack(batched_icp_R).to(device_), torch.stack(batched_icp_t).to(
            device_).unsqueeze(-1)
        icp_residuals = residual_fun(batched_icp_R, batched_icp_t, source_points, target_points)
        icp_costs = torch.mean(icp_residuals, dim=-1)

    end_time = time.time()
    print(f"RANSAC (3D REG) solver time: {end_time - start_time}")

    # return the result payload
    result_payload = {
        # meta data
        'solver_func': f'ransac_reg',

        #
        # Estimated
        #
        # transformation data
        'R_arun': batched_arun_R,
        't_arun': batched_arun_t,

        # cost and residuals
        'final_arun_cost': arun_costs,
        'final_arun_residuals': arun_residuals,

        # timing data
        'solver_time': end_time - start_time,
        'itr': total_iters,

        # inlier data
        "ransac_inliers": best_inliers,
        "weights": best_inliers,
    }

    if use_icp:
        result_payload['R_icp'] = batched_icp_R
        result_payload['t_icp'] = batched_icp_t
        result_payload['final_icp_cost'] = icp_costs
        result_payload['final_icp_residuals'] = icp_residuals

    return result_payload


class PointSetRegistration:
    def __init__(self, source_points):
        super().__init__()
        """
        source_points   : torch.tensor of shape (1, 3, N)
        
        """

        self.source_points = source_points

    def forward(self, target_points, weights=None, device_=None):
        """
        inputs:
        target_points   : torch.tensor of shape (B, 3, N)

        output:
        R   : torch.tensor of shape (B, 3, 3)
        t   : torch.tensor of shape (B, 3, 1)
        """
        batch_size = target_points.shape[0]

        source_points = self.source_points.repeat(batch_size, 1, 1)

        batch_size, d, N = source_points.shape

        if device_ == None:
            device_ = source_points.device

        if weights == None:
            ttype = target_points.dtype
            weights = torch.ones((1, N), device=device_).to(dtype=ttype)

        # if torch.any(torch.isnan(target_points)):
        #    logging.error(f"target_points : {target_points}")

        source_points_ave = torch.einsum('bdn,ln->bd', source_points, weights) / weights.sum()  # (B, 3)
        target_points_ave = torch.einsum('bdn,ln->bd', target_points, weights) / weights.sum()  # (B, 3)

        # if torch.any(torch.isnan(target_points_ave)):
        #    logging.error(f"target_points ave weighted: {target_points_ave}")

        # getting the rotation
        source_points_centered = source_points - source_points_ave.unsqueeze(-1)  # (B, 3, N)
        target_points_centered = target_points - target_points_ave.unsqueeze(-1)  # (B, 3, N)

        # if torch.any(torch.isnan(target_points_centered)):
        #    logging.error(f"target_points centered: {target_points_centered}")

        source_points_centered = torch.einsum('bdn,ln->bdn', source_points_centered, weights)  # (B, 3, N)
        target_points_centered = torch.einsum('bdn,ln->bdn', target_points_centered, weights)  # (B, 3, N)

        # if torch.any(torch.isnan(target_points_centered)):
        #    logging.error(f"target_points centered: {target_points_centered}")

        rotation = wahba(source_points=source_points_centered, target_points=target_points_centered)

        # getting the translation
        translation = target_points_ave.unsqueeze(-1) - rotation @ source_points_ave.unsqueeze(-1)

        return rotation, translation


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    B = 2
    N = 10
    d = 3

    source_points = torch.rand(1, d, N).to(device=device)
    print(source_points.shape)
    registration_fxn = PointSetRegistration(source_points=source_points)

    rotation = transforms.random_rotations(B).to(device=device)

    target_points = rotation @ source_points
    target_points += 0.01 * torch.rand(size=target_points.shape).to(device=device)
    target_points.requires_grad = True

    print("target_points.shape", target_points.shape)
    print('-' * 40)
    print("Testing wahba()")
    print('-' * 40)
    source_points_repeat = source_points.repeat(B, 1, 1)
    start = time.process_time()
    rotation_est = wahba(source_points=source_points_repeat - source_points_repeat.mean(-1).unsqueeze(-1),
                         target_points=target_points - target_points.mean(-1).unsqueeze(-1))
    end = time.process_time()
    print("Output shape: ", rotation_est.shape)

    err = rotation_matrix_error(rotation, rotation_est)
    print("Rotation error: ", err.mean())
    print("Time for wahba: ", 1000 * (end - start) / B, ' ms')

    loss = err.sum()
    loss.backward()
    print("Target point gradient: ", target_points.grad)

    print('-' * 40)
    print("Testing point_set_registration()")
    print('-' * 40)

    B = 20
    N = 10
    d = 3

    source_points = torch.rand(1, d, N).to(device=device)

    rotation = transforms.random_rotations(B).to(device=device)
    translation = torch.rand(B, d, 1).to(device=device)

    target_points = rotation @ source_points + translation
    target_points += 0.1 * torch.rand(size=target_points.shape).to(device=device)
    target_points.requires_grad = True

    start = time.process_time()
    rotation_est, translation_est = registration_fxn.forward(target_points=target_points)
    end = time.process_time()

    print("Output rotation shape: ", rotation.shape)
    print("Output translation shape: ", translation.shape)

    err_rot = rotation_error(rotation, rotation_est)
    err_trans = translation_error(translation, translation_est)
    print("Rotation error: ", err_rot.mean())
    print("Translation error: ", err_trans.mean())
    print("Time for point_set_registration: ", 1000 * (end - start) / B, ' ms')

    loss = 100 * err_rot.sum() + 100 * err_trans.sum()
    loss.backward()
    print("Target point gradient: ", target_points.grad)
