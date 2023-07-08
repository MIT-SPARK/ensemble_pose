import csv
import os
import pickle
import random
import sys
import open3d
import time
import torch
from pathlib import Path
from datetime import datetime
from pytorch3d import ops
import numpy as np

BASE_DIR = Path(__file__).parent.parent.parent
# sys.path.append("../../")

# from datasets.shapenet import DepthPC
# from datasets.shapenet import PCD_FOLDER_NAME as KEYPOINTNET_PCD_FOLDER_NAME

from casper3d.keypoint_corrector import kp_corrector_reg, robust_kp_corrector, multi_thres_kp_corrector
from casper3d.point_set_registration import PointSetRegistration, solve_registration
from casper3d.certifiability import certifiability
from casper3d.keypoint_corrector import keypoint_perturbation

from utils.ddn.node import ParamDeclarativeFunction
from utils.visualization_utils import display_two_pcs, temp_expt_1_viz
from utils.evaluation_metrics import chamfer_dist, translation_error, rotation_error
from utils.math_utils import set_all_random_seeds

from datasets.bop import get_bop_object_images_dataset, get_keypoints, get_cad_model_mesh, get_depth_and_pose
from datasets.bop import BOP_NUM_OBJECTS
from utils.general import o3dpc_to_tensor


def get_sq_distances(X, Y):
    """
    inputs:
    X   : torch.tensor of shape (B, 3, n)
    Y   : torch.tensor of shape (B, 3, m)

    outputs:
    sq_dist_xy  : torch.tensor of shape (B, n)  : for every point in X, the sq. distance to the closest point in Y
    sq_dist_yz  : torch.tensor of shape (B, m)  : for every point in Y, the sq. distance to the closest point in X
    """

    sq_dist_xy, _, _ = ops.knn_points(torch.transpose(X, -1, -2), torch.transpose(Y, -1, -2), K=1, return_sorted=False)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist_yx, yx_nn_idxs, _ = ops.knn_points(torch.transpose(Y, -1, -2), torch.transpose(X, -1, -2), K=1,
                                               return_sorted=False)
    # dist (B, n, 1): distance from point in Y to the nearest point in X

    return sq_dist_xy, sq_dist_yx, yx_nn_idxs


def get_kp_sq_distances(kp, kp_):
    sq_dist = ((kp - kp_) ** 2).sum(dim=1)
    return sq_dist  # check output dimensions


def sample_depth_pc(pc, num_points):
    """
    :pc torch.tensor of shape (b, 3, m)
    :num_points a number
    :returns: torch.tensor of shape (b, 3, num_points) or (b, 3, m), whichever is smaller
    """

    tol = 1e-3
    idx_ = torch.where(torch.abs(pc) < tol, True, False)
    idx_nonzero = torch.logical_not(torch.all(idx_, 1))  # idx=True for non-zero points in pc. shape (b, m)

    _num_points = torch.sum(idx_nonzero, -1)  # number of points in the segmented object. shape (b,)

    b, _, n = pc.shape
    _list_n = torch.arange(n)

    sampled_pc = torch.zeros(b, 3, num_points).to(device=pc.device)

    for i in range(b):

        m = _num_points[i]
        if m >= num_points:
            aa = _list_n[idx_nonzero[i]]
            _sample_indices = torch.multinomial(aa.to(dtype=torch.float), num_points)
            sample_indices = aa[_sample_indices]

        else:
            _sample_indices = _list_n[idx_nonzero[i]]
            aa = _list_n[torch.logical_not(idx_nonzero[i])]
            _sample_indices_ = torch.multinomial(aa.to(dtype=torch.float), num_points - m)
            sample_indices = _list_n[_sample_indices]
            sample_indices_ = aa[_sample_indices_]
            sample_indices = torch.hstack([sample_indices, sample_indices_])

        sampled_pc[i, :, :] = pc[i, :, sample_indices]

    return sampled_pc.to(dtype=torch.float64)


def convert_to_fixed_sized_pc(pc, num_points):
    """
    :pc torch.tensor of shape (3, m)
    :n int
    :returns: torch.tensor of shape (3, n)
    """
    n = num_points
    m = pc.shape[-1]

    if m > n:
        idx = torch.randperm(m)
        point_cloud = pc[:, idx[:n]]
        padding = torch.zeros(size=(n,), dtype=torch.bool)

    elif m < n:

        pc_pad = torch.zeros(3, n - m)
        point_cloud = torch.cat([pc, pc_pad], dim=1)
        padding1 = torch.zeros(size=(m,), dtype=torch.bool)
        padding2 = torch.ones(size=(n - m,), dtype=torch.bool)
        padding = torch.cat([padding1, padding2], dim=0)
        # Write code to pad pc with (n-m) zeros

    else:
        point_cloud = pc
        padding = torch.zeros(size=(n,), dtype=torch.bool)

    return point_cloud, padding


class Experiment:
    def __init__(self, ds_name, object_label, num_points, num_iterations, kp_noise_var_range,
                 kp_noise_fra=0.2,
                 clamp_thres=0.1,
                 epsilon=0.02,
                 theta=50.0, kappa=10.0,
                 device='cpu',
                 algo='torch-gd',
                 do_certification=False,
                 corrector_max_solve_iters=100,
                 visible_fraction_lb=0.9,
                 visible_fraction_ub=1.0
                 ):
        super().__init__()
        # ToDo: epsilon, clamp_thres are absolutes or relative to diameter. make it consistent or use different name.

        # dataset parameters
        self.ds_name = ds_name
        self.object_label = object_label
        self.num_points = num_points  # num_points to sample on the segmented depth point cloud
        self.visible_fraction_ub = visible_fraction_ub  # visible fraction parameter
        self.visible_fraction_lb = visible_fraction_lb  # visible fraction parameter

        # experiment parameters
        self.num_iterations = num_iterations  # results averaged over num_iterations
        self.kp_noise_fra = kp_noise_fra  # added keypoint noise parameter
        self.kp_noise_var_range = kp_noise_var_range  # added keypoint noise parameter
        self.do_certification = do_certification  # certify or not
        self.algo = algo  # algo used: 'scipy-tr', 'torch-gd', 'torch-gd-accel', 'torch-gnc-gm', 'torch-gnc-tlr'

        # model parameters
        self.epsilon = epsilon  # certification parameter
        self.theta = theta  # corrector loss function parameter
        self.kappa = kappa  # corrector loss function parameter
        self.corrector_max_solve_iters = corrector_max_solve_iters  # corrector parameter
        self.chamfer_clamp_thres_factor = clamp_thres
        self.gnc_max_solve_iters = 20  # fixed
        self.chamfer_clamp_thres_list = (5, 2, 1)  # fixed

        # experiment name
        self.name = 'keypoint_corrector_analysis.' + ds_name + '.' + object_label

        # device
        self.device_ = device

        # setting up the dataset and dataloader
        self.ds = get_bop_object_images_dataset(ds_name, object_label,
                                                visible_fraction_lb=self.visible_fraction_lb,
                                                visible_fraction_ub=self.visible_fraction_ub)

        self.dl = torch.utils.data.DataLoader(self.ds, batch_size=self.num_iterations, shuffle=False)

        # setting up c3po++
        # extracting keypoints
        self.model_keypoints = get_keypoints(ds_name=ds_name,
                                             object_label=object_label)
        self.model_keypoints = self.model_keypoints.unsqueeze(0)
        self.model_keypoints = self.model_keypoints.to(device=self.device_)

        # extracting cad model
        self.cad_model_mesh = get_cad_model_mesh(ds_name=ds_name,
                                                 object_label=object_label)
        pc = self.cad_model_mesh.sample_points_poisson_disk(number_of_points=1000, init_factor=5)
        self.cad_model = o3dpc_to_tensor(pc)
        self.cad_model = self.cad_model.unsqueeze(0)
        self.cad_model = self.cad_model.to(device=self.device_)
        self.cad_model = self.cad_model.to(dtype=torch.double)

        # extracting object diameter
        self.diameter = np.linalg.norm(np.asarray(self.cad_model_mesh.get_max_bound())
                                       - np.asarray(self.cad_model_mesh.get_min_bound()))

        # setting up the certifier
        self.certify = certifiability(epsilon=epsilon * self.diameter,
                                      clamp_thres=clamp_thres * self.diameter)

        # setting up the corrector
        if self.algo.split('-')[1] == 'gnc':
            corrector_node = robust_kp_corrector(cad_models=self.cad_model, model_keypoints=self.model_keypoints,
                                                 theta=self.theta, kappa=self.kappa,
                                                 algo=self.algo,
                                                 gnc_max_solve_iters=self.gnc_max_solve_iters,
                                                 corrector_max_solve_iters=self.corrector_max_solve_iters,
                                                 chamfer_max_loss=False,
                                                 chamfer_clamp_thres=self.chamfer_clamp_thres_factor * self.diameter)
        elif self.algo.split('-')[1] == 'multithres':
            print("here")
            corrector_node = multi_thres_kp_corrector(cad_models=self.cad_model, model_keypoints=self.model_keypoints,
                                                      theta=self.theta, kappa=self.kappa,
                                                      max_solve_iters=self.corrector_max_solve_iters,
                                                      chamfer_max_loss=False,
                                                      chamfer_clamp_thres_list=self.chamfer_clamp_thres_list,
                                                      chamfer_clamp_thres=self.chamfer_clamp_thres_factor * self.diameter)
        else:
            corrector_node = kp_corrector_reg(cad_models=self.cad_model, model_keypoints=self.model_keypoints,
                                              theta=self.theta, kappa=self.kappa,
                                              max_solve_iters=self.corrector_max_solve_iters,
                                              chamfer_max_loss=False,
                                              chamfer_clamped=True,
                                              chamfer_clamp_thres=self.chamfer_clamp_thres_factor * self.diameter)

        self.corrector = ParamDeclarativeFunction(problem=corrector_node)

        # setting up point set registration
        self.point_set_registration = PointSetRegistration(source_points=self.model_keypoints)

        # setting up experiment parameters and data for saving
        self.data = dict()
        self.parameters = dict()

        # ToDo: Need to update this list. New parameters have been added since.
        self.parameters['ds_name'] = self.ds_name
        self.parameters['object_label'] = self.object_label
        self.parameters['num_points'] = self.num_points
        self.parameters['num_iterations'] = self.num_iterations
        self.parameters['kp_noise_fra'] = self.kp_noise_fra
        self.parameters['kp_noise_var_range'] = self.kp_noise_var_range
        self.parameters['certify'] = self.certify
        self.parameters['theta'] = self.theta
        self.parameters['kappa'] = self.kappa
        self.parameters['name'] = self.name
        self.parameters['diameter'] = self.diameter
        self.parameters['chamfer_clamp_thres_factor'] = self.chamfer_clamp_thres_factor
        self.parameters['algo'] = algo

    def _single_loop(self, kp_noise_var, visualization=False):

        # experiment data
        rotation_err_naive = torch.zeros(self.num_iterations, 1).to(device=self.device_)
        rotation_err_corrector = torch.zeros(self.num_iterations, 1).to(device=self.device_)
        translation_err_naive = torch.zeros(self.num_iterations, 1).to(device=self.device_)
        translation_err_corrector = torch.zeros(self.num_iterations, 1).to(device=self.device_)
        if self.do_certification:
            certi_naive = torch.zeros((self.num_iterations, 1), dtype=torch.bool).to(device=self.device_)
            certi_corrector = torch.zeros((self.num_iterations, 1), dtype=torch.bool).to(device=self.device_)

        sqdist_input_naiveest = []
        sqdist_input_correctorest = []
        sqdist_input_icp = []

        sqdist_kp_naiveest = []
        sqdist_kp_correctorest = []
        sqdist_kp_icp = []

        pc_padding_masks = []

        chamfer_pose_naive_to_gt_pose_list = []
        chamfer_pose_corrected_to_gt_pose_list = []
        chamfer_pose_icp_to_gt_pose_list = []

        # experiment loop
        for i, data in enumerate(self.dl):

            # extracting data
            rgb, mask, obs = data
            scene_depth_pc, object_depth_pc, object_pose, visible_fraction = get_depth_and_pose(rgb, mask, obs,
                                                                                                self.object_label)

            # extracting rotation, translation, and putting data on gpu
            rotation_true = torch.from_numpy(object_pose[:, :3, :3])
            translation_true = torch.from_numpy(object_pose[:, :3, 3:])
            rotation_true = rotation_true.to(device=self.device_)
            translation_true = translation_true.to(device=self.device_)
            object_depth_pc = object_depth_pc.to(device=self.device_)

            # sampling the depth pc to self.num_points
            input_point_cloud = sample_depth_pc(object_depth_pc, self.num_points)

            # computing gt keypoints using gt pose
            keypoints_true = rotation_true @ self.model_keypoints + translation_true

            # generating perturbed keypoints
            detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true,
                                                       fra=self.kp_noise_fra, var=kp_noise_var * self.diameter)
            # detected_keypoints = detected_keypoints.to(dtype=torch.double)
            detected_keypoints = detected_keypoints.to(device=self.device_)

            # visualizing input pc, gt keypoints, and detected keypoints
            if visualization:
                print("Visualizing: input point cloud (grey), gt keypoints (red), detected keypoints (green)")
                temp_expt_1_viz(cad_models=input_point_cloud, model_keypoints=detected_keypoints,
                                gt_keypoints=keypoints_true)
                # visualize_torch_model_n_keypoints(cad_models=input_point_cloud, model_keypoints=detected_keypoints)

            # estimate model: naive point set registration
            start_inner = time.perf_counter()
            R_naive, t_naive = self.point_set_registration.forward(target_points=detected_keypoints)
            model_estimate_naive = R_naive @ self.cad_model + t_naive
            keypoint_estimate_naive = R_naive @ self.model_keypoints + t_naive
            if visualization:
                # print("Displaying input and naive model estimate: ")
                print("Visualizing: input point cloud (grey), gt keypoints (red), naive model kp estimates (green)")
                temp_expt_1_viz(cad_models=input_point_cloud, model_keypoints=keypoint_estimate_naive,
                                gt_keypoints=keypoints_true)
                # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate_naive)

            # estimate model: keypoint corrector
            corrector_start = time.perf_counter()
            correction = self.corrector.forward(detected_keypoints, input_point_cloud)
            # correction = torch.zeros_like(correction)
            corrector_end = time.perf_counter()
            R, t = self.point_set_registration.forward(target_points=detected_keypoints + correction)

            # if torch.any(torch.isnan(t)):
            #     breakpoint()

            model_estimate = R @ self.cad_model + t
            keypoint_estimate = R @ self.model_keypoints + t
            if visualization:
                # print("Displaying input and corrector model estimate: ")
                print("Visualizing: input point cloud (grey), gt keypoints (red), corrector kp estimates (green)")
                temp_expt_1_viz(cad_models=input_point_cloud, model_keypoints=keypoint_estimate,
                                gt_keypoints=keypoints_true)
                # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate_naive)
            end_inner = time.perf_counter()
            print(f"Inner time: {end_inner - start_inner}")

            # other methods to evaluate
            # RANSAC + ICP
            batched_source_points = self.point_set_registration.source_points.repeat(detected_keypoints.shape[0], 1, 1)
            ransac_reg_result = solve_registration(
                source_points=batched_source_points,
                target_points=detected_keypoints,
                noise_bound=self.chamfer_clamp_thres_factor * self.diameter,
                icp_threshold=0.05 * self.diameter,
                sample_size=4,
                max_iters=500,
                use_icp=True,
                input_pc=input_point_cloud,
                model_pc=self.cad_model.expand(input_point_cloud.shape[0], -1, -1),
                device_=self.device_)

            # metrics for ransac+icp
            rotation_err_icp = rotation_error(rotation_true, ransac_reg_result['R_icp'])
            translation_err_icp = translation_error(translation_true, ransac_reg_result['t_icp'])

            model_estimate_icp = ransac_reg_result['R_icp'] @ self.cad_model + ransac_reg_result['t_icp']
            keypoint_estimate_icp = ransac_reg_result['R_icp'] @ self.model_keypoints + ransac_reg_result['t_icp']

            # evaluate the two metrics
            rotation_err_naive = rotation_error(rotation_true, R_naive)
            rotation_err_corrector = rotation_error(rotation_true, R)
            translation_err_naive = translation_error(translation_true, t_naive)
            translation_err_corrector = translation_error(translation_true, t)

            # saving sq distances for certification analysis
            sq_dist_input_naive = get_sq_distances(X=input_point_cloud, Y=model_estimate_naive)
            sq_dist_input_corrector = get_sq_distances(X=input_point_cloud, Y=model_estimate)
            sq_dist_input_icp = get_sq_distances(X=input_point_cloud, Y=model_estimate_icp)
            sqdist_input_naiveest.append(sq_dist_input_naive)
            sqdist_input_correctorest.append(sq_dist_input_corrector)
            sqdist_input_icp.append(sq_dist_input_icp)

            sq_dist_kp_naive = get_kp_sq_distances(kp=keypoint_estimate_naive, kp_=detected_keypoints)
            sq_dist_kp_corrector = get_kp_sq_distances(kp=keypoint_estimate, kp_=detected_keypoints + correction)
            sq_dist_kp_icp = get_kp_sq_distances(kp=keypoint_estimate_icp, kp_=detected_keypoints)
            sqdist_kp_naiveest.append(sq_dist_kp_naive)
            sqdist_kp_correctorest.append(sq_dist_kp_corrector)
            sqdist_kp_icp.append(sq_dist_kp_icp)

            pc_padding = ((input_point_cloud == torch.zeros(3, 1).to(device=self.device_)).sum(dim=1) == 3)
            pc_padding_masks.append(pc_padding)

            model_true = rotation_true @ self.cad_model + translation_true
            chamfer_pose_naive_to_gt_pose = chamfer_dist(model_estimate_naive, model_true, max_loss=False)
            chamfer_pose_corrected_to_gt_pose = chamfer_dist(model_estimate, model_true, max_loss=False)
            chamfer_pose_naive_to_gt_pose_list.append(chamfer_pose_naive_to_gt_pose)
            chamfer_pose_corrected_to_gt_pose_list.append(chamfer_pose_corrected_to_gt_pose)
            chamfer_pose_icp_to_gt_pose = chamfer_dist(model_estimate_icp, model_true, max_loss=False)
            chamfer_pose_icp_to_gt_pose_list.append(chamfer_pose_icp_to_gt_pose)

            # certification
            certi_naive, certi_corrector, certi_icp = None, None, None
            if self.do_certification:
                certi, _ = self.certify.forward(X=input_point_cloud, Z=model_estimate_naive,
                                                kp=keypoint_estimate_naive, kp_=detected_keypoints)
                # certi_naive[i] = certi
                certi_naive = certi

                certi, _ = self.certify.forward(X=input_point_cloud, Z=model_estimate,
                                                kp=keypoint_estimate, kp_=detected_keypoints + correction)
                # certi_corrector[i] = certi
                certi_corrector = certi

                # certi icp
                certi, _ = self.certify.forward(X=input_point_cloud, Z=model_estimate_icp,
                                                kp=keypoint_estimate_icp, kp_=detected_keypoints)
                # certi_corrector[i] = certi
                certi_icp = certi

            # collapse all icp related into one dictionary
            icp_result = {**ransac_reg_result,
                          'rotation_err_icp': rotation_err_icp,
                          'translation_err_icp': translation_err_icp,
                          'certi_icp': certi_icp,
                          'sqdist_input_icp': sqdist_input_icp,
                          'sqdist_kp_icp': sqdist_kp_icp,
                          'chamfer_pose_icp_to_gt_pose_list': chamfer_pose_icp_to_gt_pose_list,
                          }

            if visualization and i >= 5:
                break
            if visualization and i < 5:
                continue
            else:
                break

        if self.do_certification:
            return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
                certi_naive, certi_corrector, sqdist_input_naiveest, sqdist_input_correctorest, sqdist_kp_naiveest, \
                sqdist_kp_correctorest, pc_padding_masks, chamfer_pose_naive_to_gt_pose_list, chamfer_pose_corrected_to_gt_pose_list, \
                icp_result
        else:
            return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
                sqdist_input_naiveest, sqdist_input_correctorest, sqdist_kp_naiveest, \
                sqdist_kp_correctorest, pc_padding_masks, chamfer_pose_naive_to_gt_pose_list, chamfer_pose_corrected_to_gt_pose_list, \
                icp_result

    def execute(self):

        rotation_err_naive = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(device=self.device_)
        translation_err_naive = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(device=self.device_)

        rotation_err_corrector = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(device=self.device_)
        translation_err_corrector = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(
            device=self.device_)

        rotation_err_icp = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(device=self.device_)
        translation_err_icp = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(
            device=self.device_)

        if self.do_certification:
            certi_naive = torch.zeros(size=(len(self.kp_noise_var_range), self.num_iterations), dtype=torch.bool).to(
                device=self.device_)
            certi_corrector = torch.zeros(size=(len(self.kp_noise_var_range), self.num_iterations),
                                          dtype=torch.bool).to(device=self.device_)
            certi_icp = torch.zeros(size=(len(self.kp_noise_var_range), self.num_iterations),
                                    dtype=torch.bool).to(device=self.device_)

        sqdist_input_naiveest = []
        sqdist_input_correctorest = []
        sqdist_input_icp = []
        sqdist_kp_naiveest = []
        sqdist_kp_correctorest = []
        sqdist_kp_icp = []
        pc_padding_masks = []
        chamfer_pose_naive_to_gt_pose_list = []
        chamfer_pose_corrected_to_gt_pose_list = []
        chamfer_pose_icp_to_gt_pose_list = []

        for i, kp_noise_var in enumerate(self.kp_noise_var_range):

            print("-" * 40)
            print("Testing at kp_noise_var: ", kp_noise_var)
            print("-" * 40)

            start = time.perf_counter()
            if self.do_certification:
                Rerr_naive, Rerr_corrector, terr_naive, terr_corrector, \
                    c_naive, c_corrector, sqdist_in_naive, sq_dist_in_corrector, sqdist_kp_naive, \
                    sqdist_kp_corrector, pc_padding_mask, chamfer_pose_naive_to_gt_pose, \
                    chamfer_pose_corrected_to_gt_pose, icp_result = self._single_loop(kp_noise_var=kp_noise_var)
            else:
                Rerr_naive, Rerr_corrector, terr_naive, terr_corrector, \
                    sqdist_in_naive, sq_dist_in_corrector, sqdist_kp_naive, \
                    sqdist_kp_corrector, pc_padding_mask, chamfer_pose_naive_to_gt_pose, \
                    chamfer_pose_corrected_to_gt_pose, icp_result = self._single_loop(kp_noise_var=kp_noise_var)

            end = time.perf_counter()
            print("Time taken: ", (end - start) / 60, ' min')

            rotation_err_naive[i, ...] = Rerr_naive.squeeze(-1)
            rotation_err_corrector[i, ...] = Rerr_corrector.squeeze(-1)
            rotation_err_icp[i, ...] = icp_result['rotation_err_icp'].squeeze(-1)

            translation_err_naive[i, ...] = terr_naive.squeeze(-1)
            translation_err_corrector[i, ...] = terr_corrector.squeeze(-1)
            translation_err_icp[i, ...] = icp_result['translation_err_icp'].squeeze(-1)

            if self.do_certification:
                certi_naive[i, ...] = c_naive.squeeze(-1)
                certi_corrector[i, ...] = c_corrector.squeeze(-1)
                certi_icp[i, ...] = icp_result['certi_icp'].squeeze(-1)

            sqdist_input_naiveest.append(sqdist_in_naive)
            sqdist_input_correctorest.append(sq_dist_in_corrector)
            sqdist_input_icp.append(icp_result['sqdist_input_icp'])

            sqdist_kp_naiveest.append(sqdist_kp_naive)
            sqdist_kp_correctorest.append(sqdist_kp_corrector)
            sqdist_kp_icp.append(icp_result['sqdist_kp_icp'])

            pc_padding_masks.append(pc_padding_mask)

            chamfer_pose_naive_to_gt_pose_list.append(chamfer_pose_naive_to_gt_pose)
            chamfer_pose_corrected_to_gt_pose_list.append(chamfer_pose_corrected_to_gt_pose)
            chamfer_pose_icp_to_gt_pose_list.append(icp_result['chamfer_pose_icp_to_gt_pose_list'])

        self.data['rotation_err_naive'] = rotation_err_naive
        self.data['rotation_err_corrector'] = rotation_err_corrector
        self.data['rotation_err_icp'] = rotation_err_icp
        self.data['translation_err_naive'] = translation_err_naive
        self.data['translation_err_corrector'] = translation_err_corrector
        self.data['translation_err_icp'] = translation_err_icp
        if self.do_certification:
            self.data['certi_naive'] = certi_naive
            self.data['certi_corrector'] = certi_corrector
            self.data['certi_icp'] = certi_icp
        self.data['sqdist_input_naiveest'] = sqdist_input_naiveest
        self.data['sqdist_input_correctorest'] = sqdist_input_correctorest
        self.data['sqdist_input_icp'] = sqdist_input_icp
        self.data['sqdist_kp_naiveest'] = sqdist_kp_naiveest
        self.data['sqdist_kp_correctorest'] = sqdist_kp_correctorest
        self.data['sqdist_kp_icp'] = sqdist_kp_icp
        self.data['pc_padding_masks'] = pc_padding_masks
        self.data['chamfer_pose_naive_to_gt_pose_list'] = chamfer_pose_naive_to_gt_pose_list
        self.data['chamfer_pose_corrected_to_gt_pose_list'] = chamfer_pose_corrected_to_gt_pose_list
        self.data['chamfer_pose_icp_to_gt_pose_list'] = chamfer_pose_icp_to_gt_pose_list

        if self.do_certification:
            return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
                certi_naive, certi_corrector, sqdist_input_naiveest, sqdist_input_correctorest, \
                sqdist_kp_naiveest, sqdist_kp_correctorest, pc_padding_masks, chamfer_pose_naive_to_gt_pose_list, \
                chamfer_pose_corrected_to_gt_pose_list
        else:
            return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
                sqdist_input_naiveest, sqdist_input_correctorest, \
                sqdist_kp_naiveest, sqdist_kp_correctorest, pc_padding_masks, chamfer_pose_naive_to_gt_pose_list, \
                chamfer_pose_corrected_to_gt_pose_list

    def execute_n_save(self, location):

        # execute the experiment
        self.execute()

        # saving the experiment and data
        # location = BASE_DIR / 'local_data' / 'corrector_analysis' / self.ds_name / self.object_label
        # if not location.exists():
        #     os.makedirs(location)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = timestamp + '_' + self.algo + '_experiment.pickle'

        file = open(location / filename, 'wb')
        pickle.dump([self.parameters, self.data], file)
        file.close()

        return location / filename, filename


def run_experiments_on(ds_name,
                       object_label,
                       kp_noise_fra=0.2,
                       visible_fraction_lb=0.9,
                       visible_fraction_ub=1.0,
                       algo='torch-gd-accel',
                       epsilon=0.02,
                       clamp_thres=0.1,
                       kp_noise_var_range=[0.1, 1.55, 0.1],
                       corrector_max_solve_iters=100,
                       only_visualize=False,
                       do_certification=False):
    # model parameters
    num_points = 1200  # fixed

    # averaging over
    sample_size = 100  # fixed

    # kp_noise parameters
    kp_noise_var_range = torch.arange(kp_noise_var_range[0], kp_noise_var_range[1], kp_noise_var_range[2])

    # corrector loss function parameters
    theta = 50.0  # fixed
    kappa = 10.0  # fixed

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device is: ", device)

    # print
    print("-" * 40)
    print("Experiment: ")
    print("dataset: ", ds_name)
    print("object: ", object_label)
    print("kp_noise_fra: ", kp_noise_fra)
    print("algo: ", algo)
    print("-" * 40)

    # setup the experiment
    expt = Experiment(ds_name=ds_name,
                      object_label=object_label,
                      num_points=num_points,
                      clamp_thres=clamp_thres,
                      num_iterations=sample_size,
                      kp_noise_var_range=kp_noise_var_range,
                      kp_noise_fra=kp_noise_fra,
                      algo=algo,
                      epsilon=epsilon,
                      theta=theta, kappa=kappa, device=device, do_certification=do_certification,
                      corrector_max_solve_iters=corrector_max_solve_iters,
                      visible_fraction_lb=visible_fraction_lb,
                      visible_fraction_ub=visible_fraction_ub)

    if only_visualize:
        while True:
            kp_noise_var = float(input('Enter noise variance parameter: '))
            expt._single_loop(kp_noise_var=kp_noise_var, visualization=True)
            flag = input('Do you want to try another variance? (y/n): ')
            if flag == 'n':
                break
    else:

        location = BASE_DIR / 'local_data' / 'corrector_analysis' / ds_name / object_label
        if not location.exists():
            os.makedirs(location)

        filename, _expt_filename = expt.execute_n_save(location)

        # experiment data
        expt = dict()
        expt['ds_name'] = ds_name
        expt['object_label'] = object_label
        expt['kp_noise_type'] = 'sporadic'
        expt['kp_noise_fra'] = kp_noise_fra
        expt['filename'] = filename
        expt['num_iterations'] = corrector_max_solve_iters  # ToDo: make this terminology consistent.
        expt['algo'] = algo

        # breakpoint()
        # expt_filename = _expt_filename[:-7] + '.csv'
        expt_filename = str(filename)[:-7] + '.csv'
        field_names = ['ds_name', 'object_label', 'kp_noise_type', 'kp_noise_fra', 'filename', 'num_iterations', 'algo']

        fp = open(expt_filename, 'a')
        dict_writer = csv.DictWriter(fp, field_names)
        dict_writer.writerow(expt)
        fp.close()

        return filename


def choose_models(ds_name, use_random=False):
    """
    """
    num_objects = BOP_NUM_OBJECTS[ds_name.split('.')[0]]
    object_labels = ["obj_0000" + "%02d" % (x + 1) for x in range(num_objects)]

    if use_random:
        object_labels = np.random.choice(object_labels)

    return object_labels


def run_full_experiment(ds_names=['ycbv'],
                        kp_noise_fra=0.8,
                        algo='torch-gd-accel',
                        epsilon=0.02,
                        clamp_thres=0.1,
                        corrector_max_solve_iters=100,
                        visible_fraction_ub=1.0,
                        visible_fraction_lb=0.9,
                        kp_noise_var_range=[0.1, 1.55, 0.1],
                        use_random=False,
                        object_labels=None,
                        do_certification=False,
                        only_visualize=False):
    filename_list = []

    for ds_name in ds_names:

        if object_labels is None:
            object_labels = choose_models(ds_name=ds_name, use_random=use_random)

        for object_label in object_labels:
            rng_seed = 0
            set_all_random_seeds(rng_seed)
            filename = run_experiments_on(ds_name=ds_name,
                                          object_label=object_label,
                                          kp_noise_fra=kp_noise_fra,
                                          algo=algo,
                                          epsilon=epsilon,
                                          clamp_thres=clamp_thres,
                                          kp_noise_var_range=kp_noise_var_range,
                                          corrector_max_solve_iters=corrector_max_solve_iters,
                                          only_visualize=only_visualize,
                                          visible_fraction_ub=visible_fraction_ub,
                                          visible_fraction_lb=visible_fraction_lb,
                                          do_certification=do_certification)
            filename_list.append(filename)

    return filename_list


if __name__ == "__main__":
    # set random seeds for reproducibility
    rng_seed = 0
    set_all_random_seeds(rng_seed)

    # select dataset name
    ds_names = ['ycbv.train.real', 'tless.primesense.train']

    # select algo
    # algo = 'scipy-tr'
    # algo = 'torch-gd'
    algo = 'torch-gd-accel'
    # algo = 'torch-gnc-gm'
    # algo = 'torch-gnc-tls'

    run_full_experiment(ds_names=ds_names,
                        kp_noise_fra=0.8,
                        algo=algo,
                        visible_fraction_ub=1.0,
                        visible_fraction_lb=0.9,
                        object_labels=None,
                        use_random=False,
                        do_certification=False)

    ################## old
    # ds_name = 'ycbv.pbr'
    # ds_name = 'tless.pbr'
    # ds_name = 'tless.primesense.train'

    # object_label = 'obj_000002'
    # run_experiments_on(ds_name=ds_name, object_label=object_label,
    #                    kp_noise_fra=0.8,
    #                    only_visualize=False, do_certification=False)
