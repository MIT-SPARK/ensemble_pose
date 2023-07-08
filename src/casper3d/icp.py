import torch
import torch.nn as nn
import open3d as o3d

from utils.difficp import ICP6DoF
from utils.math_utils import make_se3_batched


class ICP_o3d(nn.Module):
    def __init__(self, iters_max, corr_threshold):
        super().__init__()
        self.iters_max = iters_max
        self.threshold = corr_threshold

    def forward(
        self,
        pc_src: torch.Tensor,
        pc_dest: torch.Tensor,
        init_R: torch.Tensor,
        init_t: torch.tensor,
        pc_dest_is_valid_mask=None,
    ):
        """

        Args:
            pc_src: B, 3, M
            pc_dest: B, 3, M; pc_dest = R @ pc_src + t
            is_valid_mask: (B, N) mask equals True if the point is a valid point

        Returns:

        """
        B = pc_dest.size(dim=0)
        device = pc_src.device

        if pc_dest_is_valid_mask == None:
            # computes a padding by flagging zero vectors in the input point cloud.
            pc_dest_is_valid_mask = (
                pc_dest != torch.zeros(3, 1).to(device=device)
            ).sum(dim=1) == 3

        R = torch.zeros((B, 3, 3), device=pc_src.device)
        t = torch.zeros((B, 3, 1), device=pc_src.device)
        mse = torch.zeros((B, 1), device=pc_src.device)
        init_T = make_se3_batched(init_R, init_t).cpu().numpy()

        # only one cad model, so do it outside the loop
        pc_src_np = pc_src.cpu().numpy()[0, ...].T
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(pc_src_np)
        for b in range(B):
            c_pc_dest = pc_dest[b, :3, pc_dest_is_valid_mask[b, ...]].transpose(0, 1).cpu().numpy()
            dest_pcd = o3d.geometry.PointCloud()
            dest_pcd.points = o3d.utility.Vector3dVector(c_pc_dest)

            if c_pc_dest.shape[0] != 0:
                icp_sol = o3d.pipelines.registration.registration_icp(
                    src_pcd,
                    dest_pcd,
                    self.threshold,
                    init_T[b, ...],
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )
                cR, ct = icp_sol.transformation[:3, :3], icp_sol.transformation[:3, -1]
                mse[b] = icp_sol.inlier_rmse
            else:
                cR, ct = init_T[b, :3, :3], init_T[b, :3, -1]
                mse[b] = torch.tensor(float('nan'))

            R[b, :, :] = torch.tensor(cR).to(pc_src.device)
            t[b, :, :] = torch.tensor(ct).reshape(3, 1).to(pc_src.device)

        return R, t, mse


class ICP(nn.Module):
    def __init__(
        self,
        iters_max,
        mse_threshold,
        corr_threshold,
        solver_type,
        corr_type,
        dist_type,
        differentiable,
    ):
        """Construct a differentiable ICP model

        Args:
            iters_max:
            mse_threshold:
            corr_threshold:
            solver_type:
            corr_type:
            dist_type:
            differentiable:
        """
        super().__init__()
        self.icp = ICP6DoF(
            iters_max=iters_max,
            mse_threshold=mse_threshold,
            corr_threshold=corr_threshold,
            solver_type=solver_type,
            corr_type=corr_type,
            dist_type=dist_type,
            differentiable=True,
        )
        return

    def forward(
        self, pc_src: torch.Tensor, pc_dest: torch.Tensor, pc_dest_is_valid_mask=None
    ):
        """

        Args:
            pc_src: B, 3, M
            pc_dest: B, 3, M; pc_dest = R @ pc_src + t
            is_valid_mask: (B, N) mask equals True if the point is a valid point

        Returns:

        """
        B = pc_src.size(dim=0)
        device = pc_src.device

        if pc_dest_is_valid_mask == None:
            # computes a padding by flagging zero vectors in the input point cloud.
            pc_dest_is_valid_mask = (
                pc_dest != torch.zeros(3, 1).to(device=device)
            ).sum(dim=1) == 3

        R = torch.zeros((B, 3, 3), device=pc_src.device)
        t = torch.zeros((B, 3, 1), device=pc_src.device)
        mse = torch.zeros((B, 1), device=pc_src.device)
        for b in range(B):
            c_pc_src = torch.transpose(pc_src[b, :, :], 0, 1)
            c_pc_dest = torch.transpose(
                pc_dest[b, :, pc_dest_is_valid_mask[b, :]], 0, 1
            )
            T_new, _, c_mse = self.icp(c_pc_src, c_pc_dest)
            R[b, :, :] = T_new[:3, :3]
            t[b, :, :] = T_new[:3, 3].unsqueeze(1)
            mse[b] = c_mse

        return R, t, mse
