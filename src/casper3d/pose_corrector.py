"""
This implements the keypoint correction with registration, pace, as (1) a class and (2) as an AbstractDeclarativeNode

"""
import logging
# import cvxpy as cp
import numpy as np
# import open3d as o3d
# import os
import time
import torch
# import torch.nn as nn
from pytorch3d import ops, transforms
from scipy import optimize

from casper3d.model_gen import ModelFromShape
from casper3d.pace import PACEmodule
from casper3d.point_set_registration import PointSetRegistration
from datasets.shapenet import SE3PointCloud, SE3nIsotropicShapePointCloud, DepthPC
from utils.ddn.node import ParamDeclarativeFunction
from utils.evaluation_metrics import shape_error, translation_error, rotation_euler_error
from utils.math_utils import line_search_wolfe
from utils.visualization_utils import display_two_pcs, update_pos_tensor_to_keypoint_markers


# ToDo: don't use self.--- inside the `while iter < max_iterations' loop.

# From C-3PO
def registration_eval(R, R_, t, t_):
    """
    inputs:
    R, R_   : torch.tensor of shape (B, 3, 3)
    t, t_   : torch.tensor of shape (B, 3, 1)

    output:
    loss    : torch.tensor of shape (B, 1)
    """

    return rotation_euler_error(R, R_) + translation_error(t, t_)


def pose_perturbation(b=1, var=0.8, fra=0.2):
    """
    inputs:
    b   : int   :  batch size

    output:
    R   : torch.tensor (B, 3, 3) : batch of rotation matrices
    """
    R = transforms.random_rotation(b)

    return R

lossMSE = torch.nn.MSELoss(reduction="none")

class pose_corrector_reg:
    def __init__(
        self,
        cad_models,
        kappa=10.0,
        theta=50.0,
        algo="torch-gd-accel",
        animation_update=False,
        max_solve_iters=1000,
        chamfer_max_loss=False,
        chamfer_clamped=True,
        chamfer_clamp_thres=0.1,
        solve_tol=1e-12,
        log_loss_traj=False,
        vis=None,
    ):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (1, 3, n)
        model_features  : torch.tensor of shape (1, d, n)  
        algo            : 'torch-gd-accel'
        """

        self.cad_models = cad_models
        self.device_ = cad_models.device
        self.animation_update = animation_update
        self.vis = vis

        # solve / objective function settings
        self.algo = algo
        self.kappa = kappa
        self.theta = theta
        self.solve_tol = solve_tol
        self.max_solve_iters = max_solve_iters
        self.chamfer_max_loss = chamfer_max_loss
        self.chamfer_clamped = chamfer_clamped
        self.chamfer_clamp_thres = chamfer_clamp_thres
        self.log_loss_traj = log_loss_traj
        if self.log_loss_traj:
            self.loss_traj = []
        else:
            self.loss_traj = None

        self.markers = None

        self.point_set_registration_fn = PointSetRegistration(source_points=self.cad_models)

        # configure the chamfer loss function
        if chamfer_clamped:
            self.custom_chamfer_loss = lambda pc, pc_: self.chamfer_loss_clamped(
                pc, pc_, thres=self.chamfer_clamp_thres, max_loss=self.chamfer_max_loss
            )
        else:
            self.custom_chamfer_loss = lambda pc, pc_: self.chamfer_loss(pc, pc_, max_loss=self.chamfer_max_loss)

        logging.info(
            f"Corrector using chamfer loss with "
            f"clamped={self.chamfer_clamped}, "
            f"thres={self.chamfer_clamp_thres}, "
            f"max_loss={self.chamfer_max_loss}, "
            f"max_iters={self.max_solve_iters}, "
            f"solve_tol={self.solve_tol}."
        )

    def chamfer_loss(self, pc, pc_, pc_padding=None, max_loss=False):
        """
        inputs:
        pc  : torch.tensor of shape (B, 3, n)
        pc_ : torch.tensor of shape (B, 3, m)
        pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
        max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

        output:
        loss    : (B, 1)
            returns max_loss if max_loss is true
        """

        if pc_padding == None:
            batch_size, _, n = pc.shape
            device_ = pc.device

            # computes a padding by flagging zero vectors in the input point cloud.
            pc_padding = (pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3

        sq_dist, _, _ = ops.knn_points(
            torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False
        )
        # dist (B, n, 1): distance from point in X to the nearest point in Y

        sq_dist = sq_dist.squeeze(-1) * torch.logical_not(pc_padding)
        a = torch.logical_not(pc_padding)

        ## use pytorch to compute distance
        # sq_dist_torch = torch.cdist(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), compute_mode='donot_use_mm_for_euclid_dist').topk(k=1, dim=2, largest=False, sorted=False).values
        # sq_dist_torch = sq_dist_torch * sq_dist_torch
        # sq_dist_torch = sq_dist_torch.squeeze(-1) * torch.logical_not(pc_padding)
        # breakpoint()

        if max_loss:
            loss = sq_dist.max(dim=1)[0]
            # loss_torch = sq_dist_torch.max()
            # logging.debug(f"chamfer loss torch3d: {loss.item()}")
            # logging.debug(f"chamfer loss pytorch: {loss_torch.item()}")
            # breakpoint()
        else:
            loss = sq_dist.sum(dim=1) / a.sum(dim=1)

        return loss.unsqueeze(-1)

    def chamfer_loss_clamped(self, pc, pc_, thres, pc_padding=None, max_loss=False):
        """

        Args:
            pc: torch.tensor of shape (B, 3, n)
            pc_: torch.tensor of shape (B, 3, m)
            thres: threshold on the distance; below which we consider as inliers and above which we take a constant cost
            pc_padding: torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
            max_loss: indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

        Returns:

        """
        if pc_padding == None:
            batch_size, _, n = pc.shape
            device_ = pc.device

            # computes a padding by flagging zero vectors in the input point cloud.
            pc_padding = (pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3

        sq_dist, _, _ = ops.knn_points(
            torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False
        )
        # dist (B, n, 1): distance from point in X to the nearest point in Y

        # sq_dist = sq_dist.squeeze(-1) * torch.logical_not(pc_padding)
        sq_dist = sq_dist.squeeze(-1)
        # thresholding mask
        thres_mask = torch.le(sq_dist, thres**2)
        aa = torch.logical_and(torch.logical_not(pc_padding), thres_mask)

        sq_dist = sq_dist * aa

        if max_loss:
            loss = sq_dist.max(dim=1)[0]
        else:
            loss = sq_dist.sum(dim=1) / aa.sum(dim=1)

        return loss.unsqueeze(-1)

    def set_markers(self, markers):
        self.markers = markers

    def objective(self, input_point_cloud, rotation, translation, correction_rotation, correction_translation):
        """
        inputs:
        detected_features  : torch.tensor of shape (B, d, m)
        rotation           : torch.tensor of shape (B, 3, 3)
        translation        : torch.tensor of shape (B, 3, 1)
        correction_rotation: torch.tensor of shape (B, 3, 3)
        correction_translation: torch.tensor of shape (B, 3, 1)

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """

        R = correction_rotation @ rotation
        t = translation + correction_translation

        model_estimate = R @ self.cad_models + t

        # see the init function for the settings of this chamfer loss
        loss_pc = self.custom_chamfer_loss(input_point_cloud, model_estimate)

        return self.kappa * loss_pc

    def solve(self, input_point_cloud, rotation, translation):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """

        if self.algo == "torch-linesearch-wolfe":
            # steepest descent with wolfe condition line search
            # correction = self.batch_gradient_descent_wolfe(
            #     detected_keypoints, input_point_cloud, max_iterations=self.max_solve_iters, tol=self.solve_tol
            # )
            print("ERROR: NOT IMPLEMENTED")
            correction_rotation, correction_translation = None, None

        elif self.algo == "torch-gd-accel":
            # fixed step deepest descent
            correction_rotation, correction_translation = self.batch_accel_gd(
                input_point_cloud, rotation, translation, max_iterations=self.max_solve_iters, tol=self.solve_tol
            )
        elif self.algo == "torch-gd":
            # fixed step deepest descent
            # correction = self.batch_gradient_descent(
            #     detected_keypoints,
            #     input_point_cloud,
            #     max_iterations=self.max_solve_iters,
            #     tol=self.solve_tol,
            #     accelerate=False,
            # )
            print("ERROR: NOT IMPLEMENTED")
            correction_rotation, correction_translation = None, None

        elif self.algo == "scipy-tr":
            # correction = self.scipy_trust_region(detected_keypoints, input_point_cloud)
            print("ERROR: NOT IMPLEMENTED")
            correction_rotation, correction_translation = None, None

        else:
            raise NotImplementedError

        return correction_rotation, correction_translation, None

    def scipy_trust_region(self, input_point_cloud, rotation, translation, lr=0.1, num_steps=20):
        #TODO: Not Implemented.
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """

        N = detected_keypoints.shape[-1]
        batch_size = detected_keypoints.shape[0]
        correction = torch.zeros_like(detected_keypoints)
        device_ = input_point_cloud.device

        with torch.enable_grad():

            for batch in range(batch_size):

                kp = detected_keypoints[batch, ...]
                pc = input_point_cloud[batch, ...]
                kp = kp.clone().detach().to("cpu").numpy()
                pc = pc.clone().detach().to("cpu").numpy()

                batch_correction_init = 0.001 * np.random.rand(3 * N)
                fun = lambda x: self.objective_numpy(detected_keypoints=kp, input_point_cloud=pc, correction=x)

                # Note: tried other methods and trust-constr works the best
                result = optimize.minimize(
                    fun=fun, x0=batch_correction_init, method="trust-constr"
                )  # Note: tried, best so far. Promising visually. Misses orientation a few times. Faster than 'Powell'.

                batch_correction = torch.from_numpy(result.x).to(torch.float)
                batch_correction = batch_correction.reshape(3, N)

                correction[batch, ...] = batch_correction

        return correction.to(device=device_)

    def batch_gradient_descent(
        self, input_point_cloud, rotation, translation, lr=0.1, max_iterations=1000, tol=1e-12, accelerate=False, gamma=0.1
    ):
        #TODO: Not Implemented.
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """

        def _get_objective_jacobian(fun, x):

            torch.set_grad_enabled(True)
            batch_size = x.shape[0]
            dfdcorrection = torch.zeros_like(x)

            # Do not set create_graph=True in jacobian. It will slow down computation substantially.
            dfdcorrectionX = torch.autograd.functional.jacobian(fun, x)
            b = range(batch_size)
            dfdcorrection[b, ...] = dfdcorrectionX[b, 0, b, ...]

            return dfdcorrection

        N = detected_keypoints.shape[-1]

        correction = torch.zeros_like(detected_keypoints)
        y = correction.clone()
        y_prev = y.clone()

        f = lambda x: self.objective(detected_keypoints, input_point_cloud, x)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr
        if self.log_loss_traj:
            self.loss_traj.append([])

        iter = 0
        obj_ = f(correction)
        if self.log_loss_traj:
            self.loss_traj[-1].append(obj_)

        flag = torch.ones_like(obj_).to(dtype=torch.bool)
        # flag_idx = flag.nonzero()
        flag = flag.unsqueeze(-1).repeat(1, 3, N)
        while iter < max_iterations:

            iter += 1
            if self.vis is not None and self.animation_update and self.markers is not None:
                self.markers = update_pos_tensor_to_keypoint_markers(
                    self.vis, detected_keypoints + correction, self.markers
                )
                print("ATTEMPTED TO UPDATE VIS")
            obj = obj_

            dfdcorrection = _get_objective_jacobian(f, correction)
            if not accelerate:
                correction -= lr * dfdcorrection * flag
            else:
                y -= lr * dfdcorrection * flag
                correction = (1 - gamma) * y + gamma * y_prev
                y_prev = y.clone()

            obj_ = f(correction)

            if self.log_loss_traj:
                self.loss_traj[-1].append(obj_)

            if (obj - obj_).abs().max() < tol:
                break
            else:
                flag = (obj - obj_).abs() > tol
                # flag_idx = flag.nonzero()
                flag = flag.unsqueeze(-1).repeat(1, 3, N)

        logging.debug(f"Solver done. Final iter: {iter}")
        self.iters = iter

        return correction

    def _get_objective_jacobian(self, fun, x):

        torch.set_grad_enabled(True)
        batch_size = x.shape[0]
        dfdcorrection = torch.zeros_like(x)

        # Do not set create_graph=True in jacobian. It will slow down computation substantially.
        dfdcorrectionX = torch.autograd.functional.jacobian(fun, x)
        b = range(batch_size)
        dfdcorrection[b, ...] = dfdcorrectionX[b, 0, b, ...]

        return dfdcorrection

    def batch_accel_gd(self, input_point_cloud, rotation, translation, lr=0.1, max_iterations=1000, tol=1e-12, gamma=0.1):
        """Steepest descent with Nesterov acceleration

        See Nocedal & Wright, eq. 3.6(a) & (b)

        inputs:
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        rotation            : torch.tensor of shape (B, 3, 3)
        translation         : torch.tensor of shape (B, 3, 1)

        outputs:
        correction_rotation : torch.tensor of shape (B, 3, 3)
        correction_translation: torch.tensor of shape (B, 3, 1)

        """

        B, d, m = detected_features.shape
        device = detected_features.device
        correction = torch.zeros_like(detected_features)

        # wrapper for function and gradient function
        f = lambda x: self.objective(detected_features, input_point_cloud, x)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr
        # create a new trajectory
        if self.log_loss_traj:
            self.loss_traj.append([])

        # batch-wise convergence flags; terminate (stop descenting correction) if true
        flag = torch.ones((B, 1), dtype=torch.bool, device=device)
        flag = flag.unsqueeze(-1).repeat(1, 3, m)

        # calculate initial obj value (this stores the current value)
        obj_ = f(correction)
        obj = torch.tensor(float("inf"), device=device)

        # prepare variables
        y = correction.clone()
        y_prev = y.clone()

        iter = 0
        while iter < max_iterations:
            iter += 1
            if self.log_loss_traj:
                self.loss_traj[-1].append(obj_)

            # using steepest descent, descent direction = -gradient
            # dfdcorrection size: (B, 3, num keypoints)
            dfdcorrection = self._get_objective_jacobian(f, correction)

            # gradient descent
            y = correction - lr * dfdcorrection * flag

            # momentum
            correction = y + gamma * (y - y_prev)

            # update y
            y_prev = y.clone()

            # update objective value
            obj_ = f(correction)

            if (obj - obj_).abs().max() < tol:
                break
            else:
                flag = (obj - obj_).abs() > tol
                flag = flag.unsqueeze(-1).repeat(1, d, m)

            if self.vis is not None and self.animation_update and self.markers is not None:
                self.markers = update_pos_tensor_to_keypoint_markers(
                    self.vis, detected_features + correction, self.markers
                )
                print("ATTEMPTED TO UPDATE VIS")

            # save old obj value for convergence check
            obj = torch.clone(obj_)

        logging.debug(f"Solver (w/ NAGD) done. Final iter: {iter}")
        self.iters = iter

        return correction

    def batch_gradient_descent_wolfe(
        self, detected_keypoints, input_point_cloud, lr=0.1, max_iterations=1000, tol=1e-12
    ):
        """Steepest descent with weak Wolfe condition

        See Nocedal & Wright, eq. 3.6(a) & (b)

        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """

        def _get_objective_jacobian(fun, x):

            torch.set_grad_enabled(True)
            batch_size = x.shape[0]
            dfdcorrection = torch.zeros_like(x)

            # Do not set create_graph=True in jacobian. It will slow down computation substantially.
            dfdcorrectionX = torch.autograd.functional.jacobian(fun, x)
            b = range(batch_size)
            dfdcorrection[b, ...] = dfdcorrectionX[b, 0, b, ...]

            return dfdcorrection

        N = detected_keypoints.shape[-1]
        B = detected_keypoints.shape[0]
        device = detected_keypoints.device
        correction = torch.zeros_like(detected_keypoints)

        # wrapper for function and gradient function
        f = lambda x: self.objective(detected_keypoints, input_point_cloud, x)
        f_grad = lambda x: _get_objective_jacobian(f, x)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr
        # create a new trajectory
        if self.log_loss_traj:
            self.loss_traj.append([])

        # batch-wise convergence flags; terminate (stop descenting correction) if true
        flag = torch.ones((B, 1), dtype=torch.bool, device=device)
        # flag_idx = flag.nonzero()
        # flag = flag.unsqueeze(-1).repeat(1, 3, N)

        # calculate initial obj value (this stores the current value)
        obj_ = f(correction)

        iter = 0
        while iter < max_iterations:
            iter += 1
            if self.log_loss_traj:
                self.loss_traj[-1].append(obj_)

            # using steepest descent, descent direction = -gradient
            # dfdcorrection size: (B, 3, num keypoints)
            dfdcorrection = _get_objective_jacobian(f, correction)

            # save old obj value for convergence check
            obj = torch.clone(obj_)

            # search for the correct step length
            alpha, correction, obj_, gradalpha, fail, beta = line_search_wolfe(
                correction, obj_, dfdcorrection, -dfdcorrection, f, f_grad, flag=flag
            )

            # correction size: (B, 3, num keypoints)
            # if not fail:
            #    correction -= alpha * dfdcorrection * flag
            # else:
            #    correction -= lr * dfdcorrection * flag
            if (obj - obj_).abs().max() < tol:
                break
            else:
                flag = (obj - obj_).abs() > tol
                # flag_idx = flag.nonzero()
                # flag = flag.unsqueeze(-1).repeat(1, 3, N)

            if self.vis is not None and self.animation_update and self.markers is not None:
                self.markers = update_pos_tensor_to_keypoint_markers(
                    self.vis, detected_keypoints + correction, self.markers
                )
                print("ATTEMPTED TO UPDATE VIS")

        logging.debug(f"Solver (w/ Wolfe line search) done. Final iter: {iter}")
        self.iters = iter

        return correction

    def gradient(self, detected_features, input_point_cloud, y=None, v=None, ctx=None):

        if v == None:
            v = torch.ones_like(detected_features)

        # v = gradient of ML loss with respect to correction.
        # Therefore, the gradient to backpropagate is -v for detected_features.
        # We don't backpropagate gradient with respect to the input_point_cloud
        return (-v, None)


if __name__ == "__main__":

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print("device is ", device)
    print("-" * 20)

    model = torch.rand(1, 3, 1000)
    model_features = torch.rand(1, 10, 1000)

    R = transforms.random_rotation()
    R = R.unsqueeze(0)
    t = 10*torch.rand(1, 3, 1)

    input_pc = R @ (model + 0.01*torch.rand(1, 3, 1000)) + t
    # background = 10*torch.rand(1, 3, 100)
    # input_pc = torch.cat((input_pc, background), -1)

    err = 0.1*torch.rand(1, 10, 1000)
    input_features = model_features + err
    # background_fratures_ = torch.rand(1, 10, 100)
    # input_features = torch.cat((input_features, background_fratures_), -1)

    corrector = densefeat_corrector_reg(cad_models=model, model_features=model_features,
                                        kappa=10.0, algo='torch-gd-accel',
                                        chamfer_max_loss=False, chamfer_clamped=True, chamfer_clamp_thres=0.1)

    corrector_node = ParamDeclarativeFunction(corrector)
    registration_node = PointSetRegistration(source_points=model)

    correction = corrector_node.forward(input_features, input_pc)
    points = corrector.matching(input_features+correction, input_pc)
    R_, t_ = registration_node.forward(points)

    print(correction.shape)

    R = R.squeeze(0)
    R_ = R_.squeeze(0)
    print("Rotation error: ", torch.norm(R.T @ R_ - torch.eye(3), 2))
    print("Translation error: ", torch.norm(t-t_, 2))
    print("Correction error: ", torch.norm(correction - err, 2))
    breakpoint()


