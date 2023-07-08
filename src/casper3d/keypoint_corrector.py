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
from pytorch3d import ops
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


# From C-3PO
def pace_eval(R, R_, t, t_, c, c_):
    """
    inputs:
    R, R_   : torch.tensor of shape (B, 3, 3)
    t, t_   : torch.tensor of shape (B, 3, 1)

    output:
    loss    : torch.tensor of shape (B, 1)
    """

    return rotation_euler_error(R, R_) + translation_error(t, t_) + shape_error(c, c_)


# From C-3PO
def keypoint_perturbation(keypoints_true, var=0.8, fra=0.2):
    """
    inputs:
    keypoints_true  :  torch.tensor of shape (B, 3, N)
    var             :  float
    fra             :  float    : used if type == 'sporadic'

    output:
    detected_keypoints  : torch.tensor of shape (B, 3, N)
    """
    device_ = keypoints_true.device

    mask = (torch.rand(size=keypoints_true.shape).to(device=device_) < fra).int().float()
    detected_keypoints = keypoints_true + var * (torch.rand(size=keypoints_true.shape).to(device=device_) - 0.5) * mask

    return detected_keypoints


# From C-3PO
class kp_corrector_reg:
    def __init__(
        self,
        cad_models,
        model_keypoints,
        model_id=None,
        theta=50.0,
        kappa=10.0,
        algo="torch-gd",
        animation_update=False,
        max_solve_iters=1000,
        chamfer_max_loss=True,
        chamfer_clamped=False,
        chamfer_clamp_thres=0.1,
        solve_tol=1e-4,
        log_loss_traj=False,
        vis=None,
    ):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (1, 3, m)
        model_keypoints : torch.tensor of shape (1, 3, N)
        algo            : 'scipy-tr' or 'torch-gd' or 'torch-gd-accel'
        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        self.model_id = model_id
        self.device_ = model_keypoints.device
        self.animation_update = animation_update
        self.vis = vis

        # solve / objective function settings
        self.algo = algo
        self.theta = theta
        self.kappa = kappa
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

        self.point_set_registration_fn = PointSetRegistration(source_points=self.model_keypoints)

        logging.info(f"Chamfer clamp thres: {self.chamfer_clamp_thres}")
        logging.info(f"Chamfer max loss: {self.chamfer_max_loss}")
        # configure the chamfer loss function
        if chamfer_clamped:
            logging.info("Using clamped chamfer in corrector.")
            self.custom_chamfer_loss = lambda pc, pc_: self.chamfer_loss_clamped(
                pc, pc_, thres=self.chamfer_clamp_thres, max_loss=self.chamfer_max_loss
            )
        else:
            logging.info("Using regular chamfer in corrector.")
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

        # thresholding: clamp the square distances
        sq_dist.clamp_(max=thres**2)

        # aa: =1 if the point is valid (lower than the threshold)
        aa = torch.logical_not(pc_padding)

        sq_dist = sq_dist * aa
        div_factor = aa.sum(dim=1).float()

        # avoid divide by zero
        div_factor += 1.0e-7

        if max_loss:
            loss = sq_dist.max(dim=1)[0]
        else:
            loss = sq_dist.sum(dim=1) / div_factor

        return loss.unsqueeze(-1)

    def set_markers(self, markers):
        self.markers = markers

    def keypoint_loss(self, kp, kp_):
        """
        kp  : torch.tensor of shape (B, 3, N)
        kp_ : torch.tensor of shape (B, 3, N)
        returns Tensor with length B
        """

        lossMSE = torch.nn.MSELoss(reduction="none")

        return lossMSE(kp, kp_).sum(1).mean(1).unsqueeze(-1)

    def objective(self, detected_keypoints, input_point_cloud, correction):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        correction          : torch.tensor of shape (B, 3, N)

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """
        if torch.any(torch.isnan(detected_keypoints)):
            logging.warning(f"NaNs in detected keypoints when calculating corrector objective.")
        if torch.any(torch.isnan(correction)):
            logging.warning(f"NaNs in correction when calculating corrector objective.")
            A = torch.isnan(correction)
            logging.warning(f"nan indices: {torch.argwhere(A)}")

        R, t = self.point_set_registration_fn.forward(torch.nan_to_num(detected_keypoints + correction))
        model_estimate = R @ self.cad_models + t
        keypoint_estimate = R @ self.model_keypoints + t

        # see the init function for the settings of this chamfer loss
        loss_pc = self.custom_chamfer_loss(input_point_cloud, model_estimate)

        loss_kp = self.keypoint_loss(kp=detected_keypoints + correction, kp_=keypoint_estimate)
        #logging.debug(f"loss_pc: {loss_pc[0].item()}, loss_kp: {loss_kp[0].item()}")

        return self.kappa * loss_pc + self.theta * loss_kp

    def objective_numpy(self, detected_keypoints, input_point_cloud, correction):
        """
        inputs:
        detected_keypoints  : numpy.ndarray of shape (3, N)
        input_point_cloud   : numpy.ndarray of shape (3, m)
        correction          : numpy.ndarray of shape (3*N,)

        output:
        loss    : numpy.ndarray of shape (1,)
        """
        N = detected_keypoints.shape[-1]
        correction = correction.reshape(3, N)

        detected_keypoints = torch.from_numpy(detected_keypoints).unsqueeze(0).to(torch.float)
        input_point_cloud = torch.from_numpy(input_point_cloud).unsqueeze(0).to(torch.float)
        correction = torch.from_numpy(correction).unsqueeze(0).to(torch.float)

        loss = self.objective(
            detected_keypoints=detected_keypoints.to(device=self.device_),
            input_point_cloud=input_point_cloud.to(device=self.device_),
            correction=correction.to(device=self.device_),
        )

        return loss.squeeze(0).to("cpu").numpy()

    def solve(self, detected_keypoints, input_point_cloud):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """

        if self.algo == "torch-linesearch-wolfe":
            # steepest descent with wolfe condition line search
            correction = self.batch_gradient_descent_wolfe(
                detected_keypoints, input_point_cloud, max_iterations=self.max_solve_iters, tol=self.solve_tol
            )
        elif self.algo == "torch-gd-accel":
            # fixed step deepest descent
            correction = self.batch_accel_gd(
                detected_keypoints, input_point_cloud, max_iterations=self.max_solve_iters, tol=self.solve_tol
            )
        elif self.algo == "torch-gd":
            # fixed step deepest descent
            correction = self.batch_gradient_descent(
                detected_keypoints,
                input_point_cloud,
                max_iterations=self.max_solve_iters,
                tol=self.solve_tol,
                accelerate=False,
            )
        elif self.algo == "scipy-tr":
            correction = self.scipy_trust_region(detected_keypoints, input_point_cloud)
        else:
            raise NotImplementedError
        return correction, None

    def scipy_trust_region(self, detected_keypoints, input_point_cloud, lr=0.1, num_steps=20):
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
        self, detected_keypoints, input_point_cloud, lr=0.1, max_iterations=1000, tol=1e-12, accelerate=False, gamma=0.1
    ):
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

    def batch_accel_gd(self, detected_keypoints, input_point_cloud, lr=0.1, max_iterations=1000, tol=1e-12, gamma=0.1):
        """Steepest descent with Nesterov acceleration

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
        if torch.all(torch.isnan(detected_keypoints)):
            logging.warning(f"Corrector ({self.model_id}) has all NaN keypoints. Aborting.")
            return correction

        f = lambda x: self.objective(detected_keypoints, input_point_cloud, x)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr
        # create a new trajectory
        if self.log_loss_traj:
            self.loss_traj.append([])

        # batch-wise convergence flags; terminate (stop descenting correction) if true
        flag = torch.ones((B, 1), dtype=torch.bool, device=device)
        flag = flag.unsqueeze(-1).repeat(1, 3, N)

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
                self.loss_traj[-1].append(obj_.detach().cpu())

            # using steepest descent, descent direction = -gradient
            # dfdcorrection size: (B, 3, num keypoints)
            dfdcorrection = _get_objective_jacobian(f, correction)
            if torch.all(torch.isnan(dfdcorrection)):
                logging.warning(f"Corrector ({self.model_id}) Jacobians are all NaNs at iter={iter}.")
                break

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
                flag = flag.unsqueeze(-1).repeat(1, 3, N)

            if self.vis is not None and self.animation_update and self.markers is not None:
                self.markers = update_pos_tensor_to_keypoint_markers(
                    self.vis, detected_keypoints + correction, self.markers
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

    def gradient(self, detected_keypoints, input_point_cloud, y=None, v=None, ctx=None):

        if v == None:
            v = torch.ones_like(detected_keypoints)

        # v = gradient of ML loss with respect to correction.
        # Therefore, the gradient to backpropagate is -v for detected_keypoints.
        # We don't backpropagate gradient with respect to the input_point_cloud
        return (-v, None)


# SDF-based C-3PO for testing
class kp_corrector_reg_sdf:
    """Corrector class for testing SDF"""

    def __init__(
        self,
        sdf_model,
        model_keypoints,
        theta=50.0,
        kappa=10.0,
        algo="torch",
        animation_update=False,
        sdf_max_loss=False,
        vis=None,
    ):
        """

        Args:
            sdf_models: list of one ModelSDF instances (wrapped by param declrative function)
            model_keypoints: torch.tensor of shape (1, 3, N)
            theta:
            kappa:
            algo: 'scipy' or 'torch'
            animation_update:
            vis:
        """
        self.shape_model = sdf_model
        self.model_keypoints = model_keypoints
        self.theta = theta
        self.kappa = kappa
        self.algo = algo
        self.device_ = model_keypoints.device
        self.animation_update = animation_update
        self.sdf_max_loss = sdf_max_loss
        self.vis = vis

        self.markers = None

        self.point_set_registration_fn = PointSetRegistration(source_points=self.model_keypoints)

    def set_markers(self, markers):
        self.markers = markers

    def sdf_loss(self, pc, sdf_func, pc_padding=None, max_loss=False):
        """Loss function based on SDF

        Args:
            pc: torch.tensor of shape (B, 3, n)
            sdf_func: a ParamDeclarativeFunction that returns SDF values; takes in (B, 3, n) and returns (B, 1, n)
            max_loss: indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

        Returns:
            loss: torch.tensor of shape (B, 1)
        """
        if pc_padding == None:
            batch_size, _, n = pc.shape
            device_ = pc.device

            # computes a padding by flagging zero vectors in the input point cloud.
            pc_padding = (pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3

        # sdf/sdf_sq is of shape (B, 1, n)
        sdf = sdf_func.forward(pc)
        sdf_sq = torch.square(sdf)
        sdf_sq = sdf_sq.squeeze(1) * torch.logical_not(pc_padding)
        if max_loss:
            loss = sdf_sq.max(dim=1)[0]
        else:
            a = torch.logical_not(pc_padding)
            loss = sdf_sq.sum(dim=1) / a.sum(dim=1)

        return loss.unsqueeze(-1)

    def objective(self, detected_keypoints, input_point_cloud, correction):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        correction          : torch.tensor of shape (B, 3, N)

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """
        R, t = self.point_set_registration_fn.forward(detected_keypoints + correction)

        # sdf loss: calculated by transforming input point cloud back to model frame,
        # and query sdf model
        p_model = torch.transpose(R, 1, 2) @ (input_point_cloud - t)
        loss_sdf = self.sdf_loss(p_model, self.shape_model, max_loss=self.sdf_max_loss)

        # keypoint loss
        keypoint_estimate = R @ self.model_keypoints + t
        loss_kp = self.keypoint_loss(kp=detected_keypoints + correction, kp_=keypoint_estimate)

        return self.kappa * loss_sdf + self.theta * loss_kp

    def keypoint_loss(self, kp, kp_):
        """
        kp  : torch.tensor of shape (B, 3, N)
        kp_ : torch.tensor of shape (B, 3, N)
        returns Tensor with length B
        """

        lossMSE = torch.nn.MSELoss(reduction="none")

        return lossMSE(kp, kp_).sum(1).mean(1).unsqueeze(-1)

    def objective_numpy(self, detected_keypoints, input_point_cloud, correction):
        """
        inputs:
        detected_keypoints  : numpy.ndarray of shape (3, N)
        input_point_cloud   : numpy.ndarray of shape (3, m)
        correction          : numpy.ndarray of shape (3*N,)

        output:
        loss    : numpy.ndarray of shape (1,)
        """
        N = detected_keypoints.shape[-1]
        correction = correction.reshape(3, N)

        detected_keypoints = torch.from_numpy(detected_keypoints).unsqueeze(0).to(torch.float)
        input_point_cloud = torch.from_numpy(input_point_cloud).unsqueeze(0).to(torch.float)
        correction = torch.from_numpy(correction).unsqueeze(0).to(torch.float)

        loss = self.objective(
            detected_keypoints=detected_keypoints.to(device=self.device_),
            input_point_cloud=input_point_cloud.to(device=self.device_),
            correction=correction.to(device=self.device_),
        )

        return loss.squeeze(0).to("cpu").numpy()

    def solve(self, detected_keypoints, input_point_cloud):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """

        if self.algo == "scipy":
            correction = self.scipy_trust_region(detected_keypoints, input_point_cloud)
        elif self.algo == "torch":
            correction = self.batch_gradient_descent(detected_keypoints, input_point_cloud)
        else:
            raise NotImplementedError
        return correction, None

    def scipy_trust_region(self, detected_keypoints, input_point_cloud, lr=0.1, num_steps=20):
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

    def batch_gradient_descent(self, detected_keypoints, input_point_cloud, lr=0.1, max_iterations=1000, tol=1e-12):
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

        f = lambda x: self.objective(detected_keypoints, input_point_cloud, x)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr

        iter = 0
        obj_ = f(correction)
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
            correction -= lr * dfdcorrection * flag

            obj_ = f(correction)

            if (obj - obj_).abs().max() < tol:
                break
            else:
                flag = (obj - obj_).abs() > tol
                # flag_idx = flag.nonzero()
                flag = flag.unsqueeze(-1).repeat(1, 3, N)

        logging.debug(f"Solver done. Final iter: {iter}")
        self.iters = iter
        return correction

    def gradient(self, detected_keypoints, input_point_cloud, y=None, v=None, ctx=None):

        if v == None:
            v = torch.ones_like(detected_keypoints)

        # v = gradient of ML loss with respect to correction.
        # Therefore, the gradient to backpropagate is -v for detected_keypoints.
        # We don't backpropagate gradient with respect to the input_point_cloud
        return (-v, None)


class multi_thres_kp_corrector:
    def __init__(
        self,
        cad_models,
        model_keypoints,
        theta=50.0,
        kappa=10.0,
        algo="torch-multithres-gd-accel",
        animation_update=False,
        max_solve_iters=100,
        chamfer_max_loss=False,
        # chamfer_clamped=False,
        chamfer_clamp_thres=0.1,
        chamfer_clamp_thres_list=(5, 2, 1),
        solve_tol=1e-4,
        log_loss_traj=False,
        vis=None,
    ):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (1, 3, m)
        model_keypoints : torch.tensor of shape (1, 3, N)
        algo            : 'torch-multithres-gd-accel'
        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        self.device_ = model_keypoints.device
        self.animation_update = animation_update
        self.vis = vis

        # solve / objective function settings
        self.algo = algo
        self.theta = theta
        self.kappa = kappa
        self.solve_tol = solve_tol
        self.max_solve_iters = max_solve_iters
        self.chamfer_max_loss = chamfer_max_loss
        self.chamfer_clamp_thres_list = chamfer_clamp_thres_list
        self.chamfer_clamp_thres = chamfer_clamp_thres
        self.log_loss_traj = log_loss_traj
        if self.log_loss_traj:
            self.loss_traj = []
        else:
            self.loss_traj = None

        self.markers = None

        self.point_set_registration_fn = PointSetRegistration(source_points=self.model_keypoints)

        logging.info(
            f"Corrector using chamfer loss with "
            f"thres={self.chamfer_clamp_thres}, "
            f"max_loss={self.chamfer_max_loss}, "
            f"max_iters={self.max_solve_iters}, "
            f"solve_tol={self.solve_tol}."
        )

    def set_markers(self, markers):
        self.markers = markers

    def custom_chamfer_loss(self, pc, pc_, weights=None, thres=None, pc_padding=None, max_loss=False):
        """
        inputs:
        pc  : torch.tensor of shape (B, 3, n)
        pc_ : torch.tensor of shape (B, 3, m)
        weights : torch.tensor of shape (B, n)  : weights for each point in pc, in [0, 1], indicating if it is an inlier or outlier
        pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
        max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

        output:
        loss    : (B, 1)
            returns max_loss if max_loss is true
        """
        max_loss = self.chamfer_max_loss

        sq_dist, _, _ = ops.knn_points(
            torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False
        )
        # sq_dist (B, n, 1): distance from point in X to the nearest point in Y

        sq_dist = sq_dist.squeeze(-1)

        if pc_padding == None:
            batch_size, _, n = pc.shape
            device_ = pc.device

            # computes a padding by flagging zero vectors in the input point cloud.
            pc_padding = (pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3
            aa = torch.logical_not(pc_padding)
            sq_dist = sq_dist * aa
            thres_mask = torch.ones_like(sq_dist).to(dtype=torch.bool)

        if thres is not None:
            thres_mask = torch.le(sq_dist, thres**2)
            sq_dist_inliers = sq_dist * thres_mask
        else:
            sq_dist_inliers = sq_dist

        if weights is None:
            wt_sq_dist = sq_dist_inliers
        else:
            wt_sq_dist = weights * sq_dist_inliers

        if max_loss:
            loss = wt_sq_dist.max(dim=1)[0]
            # loss_torch = sq_dist_torch.max()
            # logging.debug(f"chamfer loss torch3d: {loss.item()}")
            # logging.debug(f"chamfer loss pytorch: {loss_torch.item()}")
            # breakpoint()
        else:
            loss = wt_sq_dist.sum(dim=1) / thres_mask.sum(dim=1)

        return loss.unsqueeze(-1), sq_dist

    def keypoint_loss(self, kp, kp_):
        """
        kp  : torch.tensor of shape (B, 3, N)
        kp_ : torch.tensor of shape (B, 3, N)
        returns Tensor with length B
        """
        lossMSE = torch.nn.MSELoss(reduction="none")

        return lossMSE(kp, kp_).sum(1).mean(1).unsqueeze(-1)

    def objective(self, detected_keypoints, input_point_cloud, correction, threshold):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        correction          : torch.tensor of shape (B, 3, N)

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """

        R, t = self.point_set_registration_fn.forward(detected_keypoints + correction)
        model_estimate = R @ self.cad_models + t
        keypoint_estimate = R @ self.model_keypoints + t

        # see the init function for the settings of this chamfer loss
        loss_pc, _ = self.custom_chamfer_loss(
            pc=input_point_cloud, pc_=model_estimate, thres=threshold, max_loss=self.chamfer_max_loss
        )

        loss_kp = self.keypoint_loss(kp=detected_keypoints + correction, kp_=keypoint_estimate)
        # logging.debug(f"loss_pc: {loss_pc.item()}, loss_kp: {loss_kp.item()}")

        return self.kappa * loss_pc + self.theta * loss_kp

    def solve(self, detected_keypoints, input_point_cloud):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """

        correction = torch.zeros_like(detected_keypoints)

        for thres_factor in self.chamfer_clamp_thres_list:
            correction = self.batch_accel_gd(
                detected_keypoints + correction,
                input_point_cloud,
                thres=thres_factor * self.chamfer_clamp_thres,
                max_iterations=self.max_solve_iters,
                tol=self.solve_tol,
            )

            # if torch.any(torch.isnan(correction)):
            #     breakpoint()

        return correction, None

    def batch_accel_gd(
        self, detected_keypoints, input_point_cloud, thres, lr=0.1, max_iterations=1000, tol=1e-12, gamma=0.1
    ):
        """Steepest descent with Nesterov acceleration

        See Nocedal & Wright, eq. 3.6(a) & (b)

        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """

        N = detected_keypoints.shape[-1]
        B = detected_keypoints.shape[0]
        device = detected_keypoints.device
        correction = torch.zeros_like(detected_keypoints)

        # wrapper for function and gradient function
        f = lambda x: self.objective(detected_keypoints, input_point_cloud, x, thres)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr
        # create a new trajectory
        if self.log_loss_traj:
            self.loss_traj.append([])

        # batch-wise convergence flags; terminate (stop descenting correction) if true
        flag = torch.ones((B, 1), dtype=torch.bool, device=device)
        flag = flag.unsqueeze(-1).repeat(1, 3, N)

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
                flag = flag.unsqueeze(-1).repeat(1, 3, N)

            if self.vis is not None and self.animation_update and self.markers is not None:
                self.markers = update_pos_tensor_to_keypoint_markers(
                    self.vis, detected_keypoints + correction, self.markers
                )
                print("ATTEMPTED TO UPDATE VIS")

            # save old obj value for convergence check
            obj = torch.clone(obj_)

        logging.debug(f"Solver (w/ NAGD) done. Final iter: {iter}")
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

    def gradient(self, detected_keypoints, input_point_cloud, y=None, v=None, ctx=None):

        if v == None:
            v = torch.ones_like(detected_keypoints)

        # v = gradient of ML loss with respect to correction.
        # Therefore, the gradient to backpropagate is -v for detected_keypoints.
        # We don't backpropagate gradient with respect to the input_point_cloud
        return (-v, None)


class robust_kp_corrector:
    def __init__(
        self,
        cad_models,
        model_keypoints,
        theta=50.0,
        kappa=10.0,
        chamfer_max_loss=False,
        chamfer_clamp_thres=0.1,
        algo="torch-gnc-gm",
        gnc_max_solve_iters=10,
        corrector_max_solve_iters=100,
        solve_tol=1e-12,
        animation_update=False,
        log_loss_traj=False,
        vis=None,
    ):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (1, 3, m)
        model_keypoints : torch.tensor of shape (1, 3, N)
        chamfer_loss_type : 'max' or 'sum'
        algo            : 'torch-gnc-gm' or 'torch-gnc-tls'
        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        self.device_ = model_keypoints.device
        self.animation_update = animation_update
        self.vis = vis

        # solve / objective function settings
        self.algo = algo
        self.theta = theta
        self.kappa = kappa
        self.solve_tol = solve_tol
        self.gnc_max_solve_iters = gnc_max_solve_iters
        self.corrector_max_solve_iters = corrector_max_solve_iters
        self.chamfer_max_loss = chamfer_max_loss
        self.chamfer_clamp_thres = chamfer_clamp_thres

        self.log_loss_traj = log_loss_traj
        if self.log_loss_traj:
            self.loss_traj = []
        else:
            self.loss_traj = None

        self.markers = None

        self.point_set_registration_fn = PointSetRegistration(source_points=self.model_keypoints)

        logging.info(
            f"Corrector using chamfer loss with "
            f"thres={self.chamfer_clamp_thres}, "
            f"max_loss={self.chamfer_max_loss}, "
            f"max_iters={self.gnc_max_solve_iters}, "
            f"solve_tol={self.solve_tol}."
        )

    def set_markers(self, markers):
        self.markers = markers

    def custom_chamfer_loss(self, pc, pc_, weights=None, thres=None, pc_padding=None, max_loss=False):
        """
        inputs:
        pc  : torch.tensor of shape (B, 3, n)
        pc_ : torch.tensor of shape (B, 3, m)
        weights : torch.tensor of shape (B, n)  : weights for each point in pc, in [0, 1], indicating if it is an inlier or outlier
        pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
        max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

        output:
        loss    : (B, 1)
            returns max_loss if max_loss is true
        """
        max_loss = self.chamfer_max_loss

        sq_dist, _, _ = ops.knn_points(
            torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False
        )
        # sq_dist (B, n, 1): distance from point in X to the nearest point in Y

        sq_dist = sq_dist.squeeze(-1)

        if pc_padding == None:
            batch_size, _, n = pc.shape
            device_ = pc.device

            # computes a padding by flagging zero vectors in the input point cloud.
            pc_padding = (pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3
            aa = torch.logical_not(pc_padding)
            sq_dist = sq_dist * aa
            thres_mask = torch.ones_like(sq_dist).to(dtype=torch.bool)

        if thres is not None:
            thres_mask = torch.le(sq_dist, thres**2)
            sq_dist_inliers = sq_dist * thres_mask
        else:
            sq_dist_inliers = sq_dist

        if weights is None:
            wt_sq_dist = sq_dist_inliers
        else:
            wt_sq_dist = weights * sq_dist_inliers

        if max_loss:
            loss = wt_sq_dist.max(dim=1)[0]
            # loss_torch = sq_dist_torch.max()
            # logging.debug(f"chamfer loss torch3d: {loss.item()}")
            # logging.debug(f"chamfer loss pytorch: {loss_torch.item()}")
            # breakpoint()
        else:
            loss = wt_sq_dist.sum(dim=1) / thres_mask.sum(dim=1)

        return loss.unsqueeze(-1), sq_dist

    def keypoint_loss(self, kp, kp_):
        """
        kp  : torch.tensor of shape (B, 3, N)
        kp_ : torch.tensor of shape (B, 3, N)
        returns Tensor with length B
        """
        lossMSE = torch.nn.MSELoss(reduction="none")

        return lossMSE(kp, kp_).sum(1).mean(1).unsqueeze(-1)

    def solve(self, detected_keypoints, input_point_cloud):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """

        correction = self.robust_solver(
            detected_keypoints, input_point_cloud,
            max_iterations=self.gnc_max_solve_iters, tol=self.solve_tol
        )

        return correction, None

    def robust_solver(self, detected_keypoints, input_point_cloud, max_iterations=1000, tol=1e-12):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        algo                : 'torch-gnc-gm' or 'torch-gnc-tls'

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """

        B, _, m = input_point_cloud.shape

        f = lambda x: self.objective_weighted(detected_keypoints, input_point_cloud, x, None)

        # initialization
        correction = torch.zeros_like(detected_keypoints)
        weights = torch.ones(B, m).to(device=self.device_)

        obj_, sq_dist_ = f(correction)

        r_sq_max = torch.max(sq_dist_, 1)[0].unsqueeze(-1)  # should be of shape (B, 1)
        sq_thres = self.chamfer_clamp_thres**2
        if self.algo == "torch-gnc-gm":
            parameter_mu = 2 * r_sq_max / sq_thres

        elif self.algo == "torch-gnc-tls":
            parameter_mu = sq_thres / (2 * r_sq_max)
            _gnc_tls_stopping_cost = torch.zeros(1, 1).to(device=self.device_)

        else:
            print("ERROR: parameter_mu not initialized.")
            parameter_mu = None

        iter = 0
        stop_criteria = torch.zeros_like(obj_).to(dtype=torch.bool)  # all false
        # flag = torch.ones_like(obj_).to(dtype=torch.bool)
        # flag_idx = flag.nonzero()
        # flag = flag.unsqueeze(-1).repeat(1, 3, N)

        # gnc iterations
        while iter < max_iterations:
            # print(iter)
            iter += 1
            if self.vis is not None and self.animation_update and self.markers is not None:
                self.markers = update_pos_tensor_to_keypoint_markers(
                    self.vis, detected_keypoints + correction, self.markers
                )
                print("ATTEMPTED TO UPDATE VIS")

            # update correction
            correction = self.solve_weighted_relaxation(detected_keypoints, input_point_cloud, weights)
            obj, sq_dist = f(correction)

            # update weights
            weights = self.update_weights(weights, sq_dist, parameter_mu)

            # update parameter_mu
            if self.algo == "torch-gnc-gm":
                parameter_mu = torch.max(torch.ones_like(parameter_mu).to(device=self.device_), parameter_mu / 1.4)
                stop_criteria = (
                    torch.tensor((parameter_mu <= 1)).clone().detach().to(dtype=torch.bool).to(device=self.device_)
                )

            elif self.algo == "torch-gnc-tls":
                parameter_mu = parameter_mu * 1.4

                gnc_tls_stopping_cost = torch.sum(weights * sq_dist, 1)
                stop_criteria = (_gnc_tls_stopping_cost - gnc_tls_stopping_cost) < tol
                _gnc_tls_stopping_cost = gnc_tls_stopping_cost
            else:
                print("ERROR: parameter_mu update not implemented.")
                parameter_mu = None

            # tol termination criteria
            # if (obj-obj_).abs().max() < tol or torch.all(stop_criteria):
            if (obj - obj_).abs().max() < tol:
                break
            # else:
            #     flag = (obj-obj_).abs() > tol
            # flag_idx = flag.nonzero()
            # flag = flag.unsqueeze(-1).repeat(1, 3, N)

        return correction

    def objective_weighted(self, detected_keypoints, input_point_cloud, correction, weights=None):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        correction          : torch.tensor of shape (B, 3, N)

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """

        R, t = self.point_set_registration_fn.forward(detected_keypoints + correction)
        model_estimate = R @ self.cad_models + t
        keypoint_estimate = R @ self.model_keypoints + t

        # see the init function for the settings of this chamfer loss
        loss_pc, sq_dist = self.custom_chamfer_loss(input_point_cloud, model_estimate, weights)
        loss_kp = self.keypoint_loss(kp=detected_keypoints + correction, kp_=keypoint_estimate)
        # logging.debug(f"loss_pc: {loss_pc.item()}, loss_kp: {loss_kp.item()}")

        return self.kappa * loss_pc + self.theta * loss_kp, sq_dist

    def solve_weighted_relaxation(self, detected_keypoints, input_point_cloud, weights):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """
        correction = self.batch_accel_gd(
            detected_keypoints, input_point_cloud, weights,
            max_iterations=self.corrector_max_solve_iters, tol=self.solve_tol
        )

        return correction

    def batch_accel_gd(
        self, detected_keypoints, input_point_cloud, weights, lr=0.1, max_iterations=1000, tol=1e-12, gamma=0.1
    ):
        """Steepest descent with Nesterov acceleration

        See Nocedal & Wright, eq. 3.6(a) & (b)

        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """

        N = detected_keypoints.shape[-1]
        B = detected_keypoints.shape[0]
        device = detected_keypoints.device
        correction = torch.zeros_like(detected_keypoints)

        # wrapper for function and gradient function
        f = lambda x: self.objective_weighted(detected_keypoints, input_point_cloud, x, weights)[0]

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr
        # create a new trajectory
        if self.log_loss_traj:
            self.loss_traj.append([])

        # batch-wise convergence flags; terminate (stop descenting correction) if true
        flag = torch.ones((B, 1), dtype=torch.bool, device=device)
        flag = flag.unsqueeze(-1).repeat(1, 3, N)

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
                flag = flag.unsqueeze(-1).repeat(1, 3, N)

            if self.vis is not None and self.animation_update and self.markers is not None:
                self.markers = update_pos_tensor_to_keypoint_markers(
                    self.vis, detected_keypoints + correction, self.markers
                )
                print("ATTEMPTED TO UPDATE VIS")

            # save old obj value for convergence check
            obj = torch.clone(obj_)

        logging.debug(f"Solver (w/ NAGD) done. Final iter: {iter}")
        self.iters = iter

        return correction

    def update_weights(self, weights, sq_dist, parameter_mu):
        """
        :param weights: torch.tensor of shape (B, n)
        :param sq_dist: torch.tensor of shape (B, n)

        """

        if self.algo == "torch-gnc-gm":

            _const = parameter_mu * (self.chamfer_clamp_thres**2)
            _weights = (_const / (sq_dist + _const)) ** 2

        elif self.algo == "torch-gnc-tls":

            _thL = (parameter_mu / (1 + parameter_mu)) * (self.chamfer_clamp_thres**2)
            _thU = ((parameter_mu + 1) / parameter_mu) * (self.chamfer_clamp_thres**2)

            _ones = torch.ones_like(weights)
            _weights = torch.zeros_like(weights)
            _function = -parameter_mu + torch.sqrt(
                parameter_mu * (parameter_mu + 1)
            ) * self.chamfer_clamp_thres / torch.sqrt(sq_dist)

            _weights = _ones * (torch.le(sq_dist, _thL)) + _function * (
                torch.gt(sq_dist, _thL) * torch.le(sq_dist, _thU)
            )
            # breakpoint()

        else:
            print("ERROR: weights update not implemented.")
            _weights = None

        return _weights

    def gradient(self, detected_keypoints, input_point_cloud, y=None, v=None, ctx=None):

        if v == None:
            v = torch.ones_like(detected_keypoints)

        # v = gradient of ML loss with respect to correction.
        # Therefore, the gradient to backpropagate is -v for detected_keypoints.
        # We don't backpropagate gradient with respect to the input_point_cloud
        return (-v, None)

    def _get_objective_jacobian(self, fun, x):

        torch.set_grad_enabled(True)
        batch_size = x.shape[0]
        dfdcorrection = torch.zeros_like(x)

        # Do not set create_graph=True in jacobian. It will slow down computation substantially.
        dfdcorrectionX = torch.autograd.functional.jacobian(fun, x)
        b = range(batch_size)
        dfdcorrection[b, ...] = dfdcorrectionX[b, 0, b, ...]

        return dfdcorrection


if __name__ == "__main__":

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print("device is ", device)
    print("-" * 20)

    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model

    ####################################################################################################################
    print("-" * 40)
    print("Verifying kp_corrector_reg() with SE3PointCloud(dataset) and keypoint_perturbation(): ")

    B = 10
    se3_dataset = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=500, dataset_len=1000)
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=B, shuffle=False)

    model_keypoints = se3_dataset._get_model_keypoints()  # (1, 3, N)
    cad_models = se3_dataset._get_cad_models()  # (1, 3, m)
    model_keypoints = model_keypoints.to(device=device)
    cad_models = cad_models.to(device=device)

    # define the keypoint corrector
    # option 1: use the backprop through all the iterations of optimization
    # corrector = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)  #Note: DO NOT USE THIS.
    # option 2: use autograd computed gradient for backprop.
    corrector_node = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)
    corrector = ParamDeclarativeFunction(problem=corrector_node)

    print("model_keypoints shape", model_keypoints.shape)
    point_set_reg = PointSetRegistration(source_points=model_keypoints)

    for i, data in enumerate(se3_dataset_loader):

        input_point_cloud, keypoints_true, rotation_true, translation_true = data

        input_point_cloud = input_point_cloud.to(device=device)
        keypoints_true = keypoints_true.to(device=device)
        rotation_true = rotation_true.to(device=device)
        translation_true = translation_true.to(device=device)

        # generating perturbed keypoints
        # keypoints_true = rotation_true @ model_keypoints + translation_true
        # detected_keypoints = keypoints_true
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, var=0.8, fra=1.0)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using point set registration on perturbed keypoints
        start = time.perf_counter()
        R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
        end = time.perf_counter()
        print("Naive registration time: ", 1000 * (end - start) / B, " ms")
        # model_estimate = R_naive @ cad_models + t_naive
        # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate)

        # # estimate model: using the keypoint corrector
        detected_keypoints.requires_grad = True
        start = time.perf_counter()
        correction = corrector.forward(detected_keypoints, input_point_cloud)
        end = time.perf_counter()
        print("Corrector time: ", 1000 * (end - start) / B, " ms")
        #

        loss = torch.norm(correction, p=2) ** 2
        loss = loss.sum()
        print("Testing backward: ")
        loss.backward()
        print("Shape of detected_keypoints.grad: ", detected_keypoints.grad.shape)
        print("Sum of abs() of all elements in the detected_keypoints.grad: ", detected_keypoints.grad.abs().sum())
        #

        # correction = torch.zeros_like(correction)
        R, t = point_set_reg.forward(target_points=detected_keypoints + correction)
        # model_estimate = R @ cad_models + t
        # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate)

        # evaluate the two metrics
        print(
            "Evaluation error (wo correction): ",
            registration_eval(R_naive, rotation_true, t_naive, translation_true).mean(),
        )
        print("Evaluation error (w correction): ", registration_eval(R, rotation_true, t, translation_true).mean())
        # the claim is that with the correction we can

        if i >= 3:
            break

    print("-" * 40)

    ####################################################################################################################
    print("-" * 40)
    print("Verifying kp_corrector_reg() with DepthPC(dataset) and keypoint_perturbation(): ")

    B = 10
    depth_dataset = DepthPC(class_id=class_id, model_id=model_id, n=500, num_of_points_to_sample=1000, dataset_len=100)
    depth_dataset_loader = torch.utils.data.DataLoader(depth_dataset, batch_size=B, shuffle=False)

    model_keypoints = depth_dataset._get_model_keypoints()  # (1, 3, N)
    cad_models = depth_dataset._get_cad_models()  # (1, 3, m)
    model_keypoints = model_keypoints.to(device=device)
    cad_models = cad_models.to(device=device)

    # define the keypoint corrector
    # option 1: use the backprop through all the iterations of optimization
    # corrector = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)  #Note: DO NOT USE THIS.
    # option 2: use autograd computed gradient for backprop.
    corrector_node = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)
    corrector = ParamDeclarativeFunction(problem=corrector_node)

    point_set_reg = PointSetRegistration(source_points=model_keypoints)

    for i, data in enumerate(depth_dataset_loader):

        input_point_cloud, keypoints_true, rotation_true, translation_true = data

        input_point_cloud = input_point_cloud.to(device=device)
        keypoints_true = keypoints_true.to(device=device)
        rotation_true = rotation_true.to(device=device)
        translation_true = translation_true.to(device=device)

        # generating perturbed keypoints
        # detected_keypoints = keypoints_true
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, var=0.8, fra=1.0)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using point set registration on perturbed keypoints
        start = time.process_time()
        R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
        end = time.process_time()
        print("Naive registration time: ", 1000 * (end - start) / B, " ms")
        # model_estimate = R_naive @ cad_models + t_naive
        # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate)

        # estimate model: using the keypoint corrector
        detected_keypoints.requires_grad = True
        start = time.process_time()
        correction = corrector.forward(detected_keypoints, input_point_cloud)
        end = time.process_time()
        print("Corrector time: ", 1000 * (end - start) / B, " ms")

        #
        loss = torch.norm(correction, p=2) ** 2
        loss = loss.sum()
        print("Testing backward: ")
        loss.backward()
        print("Shape of detected_keypoints.grad: ", detected_keypoints.grad.shape)
        print("Sum of abs() of all elements in the detected_keypoints.grad: ", detected_keypoints.grad.abs().sum())
        #

        # correction = torch.zeros_like(correction)
        R, t = point_set_reg.forward(target_points=detected_keypoints + correction)
        # model_estimate = R @ cad_models + t

        # evaluate the two metrics
        print(
            "Evaluation error (wo correction): ",
            registration_eval(R_naive, rotation_true, t_naive, translation_true).mean(),
        )
        print("Evaluation error (w correction): ", registration_eval(R, rotation_true, t, translation_true).mean())
        # the claim is that with the correction we can

        if i >= 3:
            break

    print("-" * 40)

    ###################################################################################################################
    print("-" * 40)
    print("Verifying kp_corrector_pace() with SE3nIsotropicShapePointCloud(dataset) and keypoint_perturbation(): ")

    B = 1
    se3_dataset = SE3nIsotropicShapePointCloud(
        class_id=class_id, model_id=model_id, num_of_points=500, dataset_len=1000
    )
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=B, shuffle=False)

    model_keypoints = se3_dataset._get_model_keypoints()  # (1, 3, N)
    cad_models = se3_dataset._get_cad_models()  # (1, 3, m)
    model_keypoints = model_keypoints.to(device=device)
    cad_models = cad_models.to(device=device)

    # define the keypoint corrector
    # option 1: use the backprop through all the iterations of optimization
    # corrector = kp_corrector_pace(cad_models=cad_models, model_keypoints=model_keypoints) #Note: DO NOT USE THIS.
    # option 2: use autograd computed gradient for backprop.
    corrector_node = kp_corrector_pace(cad_models=cad_models, model_keypoints=model_keypoints)
    corrector = ParamDeclarativeFunction(problem=corrector_node)

    pace = PACEmodule(model_keypoints=model_keypoints)
    modelgen = ModelFromShape(cad_models=cad_models, model_keypoints=model_keypoints)

    for i, data in enumerate(se3_dataset_loader):

        input_point_cloud, keypoints_true, rotation_true, translation_true, shape_true = data

        input_point_cloud = input_point_cloud.to(device=device)
        keypoints_true = keypoints_true.to(device=device)
        rotation_true = rotation_true.to(device=device)
        translation_true = translation_true.to(device=device)
        shape_true = shape_true.to(device=device)

        # generating perturbed keypoints
        # keypoints_true = rotation_true @ model_keypoints + translation_true
        # detected_keypoints = keypoints_true
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, var=0.8, fra=0.2)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using point set registration on perturbed keypoints
        start = time.perf_counter()
        R_naive, t_naive, c_naive = pace(detected_keypoints)
        end = time.perf_counter()
        print("Naive pace time: ", 1000 * (end - start) / B, " ms")
        keypoint_naive, model_naive = modelgen.forward(shape=c_naive)
        model_naive = R_naive @ model_naive + t_naive
        keypoint_naive = R_naive @ keypoint_naive + t_naive
        display_two_pcs(pc1=input_point_cloud, pc2=model_naive)

        # # estimate model: using the keypoint corrector
        detected_keypoints.requires_grad = True
        start = time.perf_counter()
        correction = corrector.forward(detected_keypoints, input_point_cloud)
        end = time.perf_counter()
        print("Corrector with pace time: ", 1000 * (end - start) / B, " ms")
        #

        loss = torch.norm(correction, p=2) ** 2
        loss = loss.sum()
        print("Testing backward: ")
        loss.backward()
        print("Shape of detected_keypoints.grad: ", detected_keypoints.grad.shape)
        print("Sum of abs() of all elements in the detected_keypoints.grad: ", detected_keypoints.grad.abs().sum())
        #

        # correction = torch.zeros_like(correction)
        R, t, c = pace.forward(detected_keypoints + correction)
        end = time.perf_counter()
        print("Naive registration time: ", 1000 * (end - start) / B, " ms")
        keypoints, model = modelgen.forward(shape=c)
        model = R_naive @ model + t_naive
        keypoints = R_naive @ keypoints + t_naive
        # model_estimate = R @ cad_models + t
        display_two_pcs(pc1=input_point_cloud, pc2=model)

        # evaluate the two metrics
        print(
            "Evaluation error (wo correction): ",
            pace_eval(R_naive, rotation_true, t_naive, translation_true, c_naive, shape_true).mean(),
        )
        print(
            "Evaluation error (w correction): ", pace_eval(R, rotation_true, t, translation_true, c, shape_true).mean()
        )
        # the claim is that with the correction we can

        if i >= 3:
            break

    print("-" * 40)
