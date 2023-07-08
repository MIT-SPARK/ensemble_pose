"""
This code implements supervised training of keypoint detector in simulation.

It can use registration during supervised training.

"""
import copy

import logging
import trimesh
import numpy as np
import pandas as pd
import shutil
import os
import pickle
from tqdm import tqdm
import open3d as o3d
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from datasets import ycbv, tless
from proposed_model import PointsRegressionModel as ProposedModel
from proposed_model import (
    load_c3po_model,
    load_c3po_cad_models,
    load_c3po_cad_meshes,
    load_all_cad_models,
    load_batch_renderer,
    load_certifier,
    load_cosypose_coarse_refine_model,
)
from utils.general import TrackingMeter
from utils.loss_functions import bounded_avg_kpt_distance_loss, half_chamfer_loss_clamped
import utils.visualization_utils as vutils
from utils.visualization_utils import display_results
from utils.torch_utils import cast2cuda
import utils.math_utils as mutils
from utils.torch_utils import get_grad_norm, del_tensor_dictionary, del_tensor_iterable
from utils.math_utils import sq_half_chamfer_dists_clamped, sq_half_chamfer_dists


from self_supervised_single_obj_training import certified_loss, collate_c3po_cosypose
from training_utils import *


def get_chamfer_dists_stats(chamfer_dists_data):
    mean = torch.mean(chamfer_dists_data, dim=1, keepdim=False)
    median = torch.median(chamfer_dists_data, dim=1, keepdim=False)[0]
    max = torch.max(chamfer_dists_data, dim=1, keepdim=False)[0]
    min = torch.min(chamfer_dists_data, dim=1, keepdim=False)[0]
    return mean, median, min, max


def update_metric_stats(method_name, chamfer_dists_data, pylds, metric_name="adds"):
    c3po_cmean, c3po_cmedian, c3po_cmin, c3po_cmax = get_chamfer_dists_stats(chamfer_dists_data)

    pylds[f"{method_name}_{metric_name}_mean"] = c3po_cmean.cpu().tolist()
    pylds[f"{method_name}_{metric_name}_median"] = c3po_cmedian.cpu().tolist()
    pylds[f"{method_name}_{metric_name}_max"] = c3po_cmax.cpu().tolist()
    pylds[f"{method_name}_{metric_name}_min"] = c3po_cmin.cpu().tolist()
    return pylds


def cert_one_batch_outputs(
    cert_model,
    object_id,
    input_imgs,
    input_pc,
    detected_masks,
    K,
    centroids,
    R_gt,
    t_gt,
    cad_models,
    model_outputs,
    pc_eps,
    pc_clamp_thres,
    mask_eps,
    tb_writer,
    tb_global_step,
    device,
    cfg,
):
    B = input_pc.shape[0]
    obj_infos = [dict(name=object_id) for i in range(B)]

    # each entry in this list contains:
    # method type:
    # c3po_cert
    pylds = {}
    if "c3po" in model_outputs.keys():
        out_c3po = model_outputs["c3po"]
        cert_c3po_pc = cert_model.certify_pc(
            input_pc=input_pc, predicted_pc=out_c3po[0], eps=pc_eps, clamp_thres=pc_clamp_thres
        )
        cert_c3po_mask, c3po_mask_scores, _ = cert_model.certify_mask(
            detected_masks=detected_masks,
            predicted_poses=mutils.make_se3_batched(out_c3po[2], out_c3po[3] + centroids),
            obj_infos=obj_infos,
            K=K,
            resolution=input_imgs.shape[-2:],
            eps=mask_eps,
        )

        if cfg["visualization"]["certifier_pc"]["c3po"]:
            logging.info("Visualizing point clouds for C3PO certification.")
            for b in range(B):
                logging.info(f"Certificate value: {cert_c3po_pc[b]}")
                vutils.visualize_pcs([input_pc[b, ...], out_c3po[0][b, ...]], colors=[[0, 1, 0], [1, 0, 0]])

        pylds["c3po_R_est"] = out_c3po[2].cpu().tolist()
        pylds["c3po_t_est"] = (out_c3po[3] + centroids).squeeze(-1).cpu().tolist()
        pylds["cert_c3po_pc"] = cert_c3po_pc.cpu().tolist()
        pylds["cert_c3po_mask"] = cert_c3po_mask.cpu().tolist()
        pylds["c3po_mask_scores"] = c3po_mask_scores.cpu().tolist()

    if "cosypose" in model_outputs.keys():
        out_csy = model_outputs["cosypose"]
        csy_preds = out_csy[0]
        csy_predicted_pc = csy_preds.poses[:, :3, :3] @ cad_models + csy_preds.poses[:, :3, -1][..., None] - centroids

        cert_csy_pc = cert_model.certify_pc(
            input_pc=input_pc, predicted_pc=csy_predicted_pc, eps=pc_eps, clamp_thres=pc_clamp_thres
        )
        if cfg["visualization"]["certifier_pc"]["cosypose"]:
            logging.info("Visualizing point clouds for Cosypose certification.")
            for b in range(B):
                logging.info(f"Certificate value: {cert_csy_pc[b]}")
                vutils.visualize_pcs(
                    [input_pc[b, ...], csy_predicted_pc[b, ...]],
                    colors=[[0, 1, 0], [1, 0, 0]],
                )

        cert_csy_mask, csy_mask_scores, csy_pred_masks = cert_model.certify_mask(
            detected_masks=detected_masks,
            predicted_poses=csy_preds.poses,
            obj_infos=obj_infos,
            K=K,
            resolution=input_imgs.shape[-2:],
            eps=mask_eps,
        )
        if cfg["visualization"]["certifier_rendered_mask"]["cosypose"]:
            logging.info("Visualizing rendered mask for Cosypose certification.")
            # visualizing detected & predicted masks
            vutils.visualize_det_and_pred_masks(
                rgbs=input_imgs,
                batch_im_id=csy_preds.infos["batch_im_id"],
                det_masks=detected_masks,
                pred_masks=csy_pred_masks,
                cert_scores=csy_mask_scores,
                show=True,
            )

        pylds["cosypose_R_est"] = csy_preds.poses[:, :3, :3].cpu().tolist()
        pylds["cosypose_t_est"] = csy_preds.poses[:, :3, -1].cpu().tolist()
        pylds["cert_cosypose_pc"] = cert_csy_pc.cpu().tolist()
        pylds["cert_cosypose_mask"] = cert_csy_mask.cpu().tolist()
        pylds["cosypose_mask_scores"] = csy_mask_scores.cpu().tolist()

    # joint strategy
    if len(model_outputs.keys()) == 2 and "c3po" in model_outputs.keys() and "cosypose" in model_outputs.keys():
        # select joint certifiable
        cert_csy = torch.logical_and(cert_csy_pc, cert_csy_mask)
        cert_c3po = torch.logical_and(cert_c3po_pc, cert_c3po_mask)
        cert_total = torch.logical_or(cert_c3po, cert_csy)
        cert_both = torch.logical_and(cert_csy, cert_c3po)
        cert_only_one = torch.logical_xor(cert_csy, cert_c3po)

        # for two methods both certifiable:
        # select c3po
        pylds["use_c3po"] = torch.logical_or(cert_both, cert_c3po).cpu().tolist()
        pylds["use_cosypose"] = (cert_csy * cert_only_one).cpu().tolist()

    pylds["method_name"] = ["_".join(sorted(cfg["models_to_use"]))] * B
    return pylds


def eval_one_batch_outputs(
    cert_model,
    object_id,
    input_imgs,
    input_pc,
    detected_masks,
    K,
    centroids,
    R_gt,
    t_gt,
    cad_models,
    model_outputs,
    pc_eps,
    pc_clamp_thres,
    mask_eps,
    tb_writer,
    tb_global_step,
    device,
    cfg,
):
    B = input_pc.shape[0]
    obj_infos = [dict(name=object_id) for i in range(B)]
    gt_pc = R_gt.float() @ cad_models + t_gt.float().squeeze(-1)[..., None]

    # each entry in this list contains:
    # method type:
    # c3po_cert
    pylds = {}
    if "c3po" in model_outputs.keys():
        out_c3po = model_outputs["c3po"]
        cert_c3po_pc = cert_model.certify_pc(
            input_pc=input_pc, predicted_pc=out_c3po[0], eps=pc_eps, clamp_thres=pc_clamp_thres
        )
        cert_c3po_mask, c3po_mask_scores, _ = cert_model.certify_mask(
            detected_masks=detected_masks,
            predicted_poses=mutils.make_se3_batched(out_c3po[2], out_c3po[3] + centroids),
            obj_infos=obj_infos,
            K=K,
            resolution=input_imgs.shape[-2:],
            eps=mask_eps,
        )

        if cfg["visualization"]["certifier_pc"]["c3po"]:
            logging.info("Visualizing point clouds for C3PO certification.")
            for b in range(B):
                logging.info(f"Certificate value: {cert_c3po_pc[b]}")
                vutils.visualize_pcs(
                    [input_pc[b, ...], out_c3po[0][b, ...], gt_pc[b, ...]], colors=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]
                )

        c3po_chamfer_dists = torch.sqrt(mutils.sq_half_chamfer_dists(gt_pc, out_c3po[0]))
        c3po_chamfer_dists_clamped = torch.clamp(c3po_chamfer_dists, max=pc_clamp_thres)

        pylds = update_metric_stats("c3po", c3po_chamfer_dists, pylds, metric_name="chamfer")
        pylds = update_metric_stats("c3po", c3po_chamfer_dists_clamped, pylds, metric_name="chamfer_clamped")
        pylds["cert_c3po_pc"] = cert_c3po_pc.cpu().tolist()
        pylds["cert_c3po_mask"] = cert_c3po_mask.cpu().tolist()
        pylds["c3po_mask_scores"] = c3po_mask_scores.cpu().tolist()

    if "cosypose" in model_outputs.keys():
        out_csy = model_outputs["cosypose"]
        csy_preds = out_csy[0]
        csy_predicted_pc = csy_preds.poses[:, :3, :3] @ cad_models + csy_preds.poses[:, :3, -1][..., None] - centroids

        cert_csy_pc = cert_model.certify_pc(
            input_pc=input_pc, predicted_pc=csy_predicted_pc, eps=pc_eps, clamp_thres=pc_clamp_thres
        )
        if cfg["visualization"]["certifier_pc"]["cosypose"]:
            logging.info("Visualizing point clouds for Cosypose certification.")
            for b in range(B):
                logging.info(f"Certificate value: {cert_csy_pc[b]}")
                vutils.visualize_pcs(
                    [input_pc[b, ...], csy_predicted_pc[b, ...], gt_pc[b, ...]],
                    colors=[[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                )

        cert_csy_mask, csy_mask_scores, csy_pred_masks = cert_model.certify_mask(
            detected_masks=detected_masks,
            predicted_poses=csy_preds.poses,
            obj_infos=obj_infos,
            K=K,
            resolution=input_imgs.shape[-2:],
            eps=mask_eps,
        )
        if cfg["visualization"]["certifier_rendered_mask"]["cosypose"]:
            logging.info("Visualizing rendered mask for Cosypose certification.")
            # visualizing detected & predicted masks
            vutils.visualize_det_and_pred_masks(
                rgbs=input_imgs,
                batch_im_id=csy_preds.infos["batch_im_id"],
                det_masks=detected_masks,
                pred_masks=csy_pred_masks,
                cert_scores=csy_mask_scores,
                show=True,
            )

        csy_chamfer_dists = torch.sqrt(mutils.sq_half_chamfer_dists(gt_pc, csy_predicted_pc))
        csy_chamfer_dists_clamped = torch.clamp(csy_chamfer_dists, max=pc_clamp_thres)
        pylds = update_metric_stats("cosypose", csy_chamfer_dists, pylds, metric_name="chamfer")
        pylds = update_metric_stats("cosypose", csy_chamfer_dists_clamped, pylds, metric_name="chamfer_clamped")
        pylds["cert_cosypose_pc"] = cert_csy_pc.cpu().tolist()
        pylds["cert_cosypose_mask"] = cert_csy_mask.cpu().tolist()
        pylds["cosypose_mask_scores"] = csy_mask_scores.cpu().tolist()

    # joint strategy
    if len(model_outputs.keys()) == 2 and "c3po" in model_outputs.keys() and "cosypose" in model_outputs.keys():
        # select joint certifiable
        cert_csy = torch.logical_and(cert_csy_pc, cert_csy_mask)
        cert_c3po = torch.logical_and(cert_c3po_pc, cert_c3po_mask)
        cert_total = torch.logical_or(cert_c3po, cert_csy)
        cert_both = torch.logical_and(cert_csy, cert_c3po)
        cert_only_one = torch.logical_xor(cert_csy, cert_c3po)

        # get the final chamfer dists
        # for only one method certifiable:
        # 0. mask c3po using cert_c3po mask; mask csy using cert_csy mask
        # 1. xor to get only one certifiable's net list
        cert_c3po_only_chamfer = (cert_c3po * cert_only_one).unsqueeze(-1) * c3po_chamfer_dists
        cert_csy_only_chamfer = (cert_csy * cert_only_one).unsqueeze(-1) * csy_chamfer_dists

        # for two methods both certifiable:
        # select c3po
        both_chamfer = cert_both.unsqueeze(-1) * c3po_chamfer_dists
        final_chamfer = cert_c3po_only_chamfer + cert_csy_only_chamfer + both_chamfer
        pylds = update_metric_stats("joint", final_chamfer, pylds=pylds, metric_name="chamfer")
        pylds["use_c3po"] = torch.logical_or(cert_both, cert_c3po).cpu().tolist()
        pylds["use_cosypose"] = (cert_csy * cert_only_one).cpu().tolist()

    pylds["method_name"] = ["_".join(sorted(cfg["models_to_use"]))] * B
    return pylds


def eval(
    eval_loader,
    object_id,
    object_diameter,
    models,
    cert_model,
    device,
    object_ds,
    cad_mesh,
    tensorboard_writer,
    cad_models=None,
    visualize=False,
    save_render=True,
    cfg=None,
):
    """Main evaluation function"""
    # set all models to eval
    for _, m in models.items():
        m.eval()

    pylds = []
    with torch.no_grad():
        nf = 1.0
        if cfg["training"]["normalize_pc"]:
            nf = object_diameter

        cert_pc_eps = cfg["certifier"]["epsilon_pc"][object_id] * object_diameter
        cert_pc_clamp_thres = cfg["certifier"]["clamp_thres"][object_id] * object_diameter
        cert_mask_eps = cfg["certifier"]["epsilon_mask"][object_id]

        for i, data in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            pc, kp, centroids = data["c3po"][0].to(device), data["c3po"][1].to(device), data["c3po"][4].to(device)
            R_gt, t_gt = data["c3po"][2].to(device), data["c3po"][3].to(device)
            pc_invalid_mask = (pc[:, :3, :] == torch.zeros(3, 1).to(device=pc.device)).sum(dim=1) == 3
            if torch.any(pc_invalid_mask):
                raise ValueError("Point cloud contains zero paddings!")

            if cfg["training"]["normalize_pc"]:
                kp_gt = kp
            else:
                kp_gt = kp / object_diameter

            model_outputs = {}
            if "c3po" in models.keys():
                # c3po forward pass
                # note: predicted keypoints are in the centered frame
                model_outputs["c3po"] = models["c3po"](object_id, pc)

                if visualize:
                    logging.info("Visualizing predictions and keypoint annotations for c3po.")
                    # visualize: input point cloud, ground truth keypoint annotations, predicted keypoints
                    kp_pred = model_outputs["c3po"][1] / object_diameter
                    pc_gt = (R_gt.float() @ cad_models + t_gt.float()[..., None]) / object_diameter

                    # load mesh
                    transformed_meshes = []
                    T_est = mutils.make_se3_batched(model_outputs["c3po"][2], model_outputs["c3po"][3])
                    for b in range(kp_pred.shape[0]):
                        c_mesh = copy.deepcopy(cad_mesh)
                        c_mesh = c_mesh.apply_transform(T_est[b, ...].cpu().numpy())
                        c_mesh.vertices /= object_diameter
                        transformed_meshes.append(c_mesh)

                    vutils.visualize_gt_and_pred_keypoints_w_trimesh(
                        #pc_gt,
                        pc[:, :3, :],
                        # kp_gt=kp_gt,
                        #meshes=transformed_meshes,
                        #pc_gt=pc_gt,
                        # kp_pred=kp_pred,
                        radius=0.03,
                        save_render=save_render,
                        save_render_path="./renders",
                        render_name=f"{i}_c3po_render",
                    )

            # cosypose forward pass
            imgs, K, cosypose_det = (
                data["cosypose"][0].to(device),
                data["cosypose"][1].to(device),
                data["cosypose"][2].to(device),
            )

            if "cosypose" in models.keys():
                model_outputs["cosypose"] = models["cosypose"](
                    images=imgs,
                    K=K,
                    detections=cosypose_det,
                    TCO=None,
                    n_coarse_iterations=cfg["cosypose_coarse_refine"]["coarse"]["n_iterations"],
                    n_refiner_iterations=cfg["cosypose_coarse_refine"]["refiner"]["n_iterations"],
                    pc_centered_normalized=pc[:, :3, :],
                    pc_centroids=centroids,
                )

                if visualize:
                    logging.info("Visualizing predictions and keypoint annotations for cosypose.")
                    # visualize: input point cloud, ground truth keypoint annotations, predicted keypoints

                    # load mesh
                    transformed_meshes = []
                    T_est = mutils.make_se3_batched(
                        model_outputs["cosypose"][0].poses[:, :3, :3], model_outputs["cosypose"][0].poses[:, :3, -1]
                    )
                    T_est[:, :3, -1] -= centroids.squeeze(-1)
                    for b in range(pc.shape[0]):
                        c_mesh = copy.deepcopy(cad_mesh)
                        c_mesh = c_mesh.apply_transform(T_est[b, ...].cpu().numpy())
                        c_mesh.vertices /= object_diameter
                        transformed_meshes.append(c_mesh)

                    pc_gt = (R_gt.float() @ cad_models + t_gt.float()[..., None]) / object_diameter
                    vutils.visualize_gt_and_pred_keypoints_w_trimesh(
                        #pc_gt,
                        pc[:, :3, :],
                        # kp_gt=kp_gt,
                        #meshes=transformed_meshes,
                        # kp_pred=kp_pred,
                        #pc_gt=pc_gt,
                        radius=0.03,
                        save_render=save_render,
                        save_render_path="./renders",
                        render_name=f"{i}_cosypose_render",
                    )

            # certification
            batch_pylds = eval_one_batch_outputs(
                cert_model=cert_model,
                # inputs
                object_id=object_id,
                input_imgs=imgs,
                input_pc=pc[:, :3, :] * nf,
                detected_masks=cosypose_det._tensors["masks"],
                K=K,
                centroids=centroids,
                cad_models=cad_models,
                R_gt=R_gt,
                t_gt=t_gt,
                # outputs
                model_outputs=model_outputs,
                # parameters
                pc_eps=cert_pc_eps,
                pc_clamp_thres=cert_pc_clamp_thres,
                mask_eps=cert_mask_eps,
                tb_writer=None,
                tb_global_step=None,
                device=device,
                cfg=cfg,
            )
            pylds.extend(pd.DataFrame(batch_pylds).to_dict("records"))

        del pc, kp, centroids, batch_pylds, imgs, K, model_outputs
        torch.cuda.empty_cache()

    return pylds


def eval_nongt_detections(
    eval_loader,
    object_id,
    object_diameter,
    models,
    cert_model,
    device,
    object_ds,
    cad_mesh,
    tensorboard_writer,
    cad_models=None,
    visualize=False,
    save_render=True,
    cfg=None,
):
    """Main evaluation function"""
    # set all models to eval
    for _, m in models.items():
        m.eval()

    pylds = []
    with torch.no_grad():
        nf = 1.0
        if cfg["training"]["normalize_pc"]:
            nf = object_diameter

        cert_pc_eps = cfg["certifier"]["epsilon_pc"][object_id] * object_diameter
        cert_pc_clamp_thres = cfg["certifier"]["clamp_thres"][object_id] * object_diameter
        cert_mask_eps = cfg["certifier"]["epsilon_mask"][object_id]

        for i, data in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            pc, centroids = data["c3po"][0].to(device), data["c3po"][4].to(device)
            pc_invalid_mask = (pc[:, :3, :] == torch.zeros(3, 1).to(device=pc.device)).sum(dim=1) == 3
            if torch.any(pc_invalid_mask):
                raise ValueError("Point cloud contains zero paddings!")

            model_outputs = {}
            if "c3po" in models.keys():
                # c3po forward pass
                # note: predicted keypoints are in the centered frame
                model_outputs["c3po"] = models["c3po"](object_id, pc)

                if visualize:
                    logging.info("Visualizing predictions and keypoint annotations for c3po.")
                    # visualize: input point cloud, ground truth keypoint annotations, predicted keypoints
                    kp_pred = model_outputs["c3po"][1] / object_diameter

                    # load mesh
                    transformed_meshes = []
                    T_est = mutils.make_se3_batched(model_outputs["c3po"][2], model_outputs["c3po"][3])
                    for b in range(kp_pred.shape[0]):
                        c_mesh = copy.deepcopy(cad_mesh)
                        c_mesh = c_mesh.apply_transform(T_est[b, ...].cpu().numpy())
                        c_mesh.vertices /= object_diameter
                        transformed_meshes.append(c_mesh)

                    vutils.visualize_gt_and_pred_keypoints_w_trimesh(
                        pc[:, :3, :],
                        # kp_gt=kp_gt,
                        meshes=transformed_meshes,
                        kp_pred=kp_pred,
                        radius=0.03,
                    )

            # cosypose forward pass
            imgs, K, cosypose_det = (
                data["cosypose"][0].to(device),
                data["cosypose"][1].to(device),
                data["cosypose"][2].to(device),
            )

            if "cosypose" in models.keys():
                model_outputs["cosypose"] = models["cosypose"](
                    images=imgs,
                    K=K,
                    detections=cosypose_det,
                    TCO=None,
                    n_coarse_iterations=cfg["cosypose_coarse_refine"]["coarse"]["n_iterations"],
                    n_refiner_iterations=cfg["cosypose_coarse_refine"]["refiner"]["n_iterations"],
                    pc_centered_normalized=pc[:, :3, :],
                    pc_centroids=centroids,
                )

                if visualize:
                    logging.info("Visualizing predictions and keypoint annotations for cosypose.")
                    # visualize: input point cloud, ground truth keypoint annotations, predicted keypoints

                    # load mesh
                    transformed_meshes = []
                    T_est = mutils.make_se3_batched(
                        model_outputs["cosypose"][0].poses[:, :3, :3], model_outputs["cosypose"][0].poses[:, :3, -1]
                    )
                    T_est[:, :3, -1] -= centroids.squeeze(-1)
                    for b in range(pc.shape[0]):
                        c_mesh = copy.deepcopy(cad_mesh)
                        c_mesh = c_mesh.apply_transform(T_est[b, ...].cpu().numpy())
                        c_mesh.vertices /= object_diameter
                        transformed_meshes.append(c_mesh)

                    vutils.visualize_gt_and_pred_keypoints_w_trimesh(
                        pc[:, :3, :],
                        # kp_gt=kp_gt,
                        meshes=transformed_meshes,
                        # kp_pred=kp_pred,
                        radius=0.03,
                    )

            # certification
            batch_pylds = cert_one_batch_outputs(
                cert_model=cert_model,
                # inputs
                object_id=object_id,
                input_imgs=imgs,
                input_pc=pc[:, :3, :] * nf,
                detected_masks=cosypose_det._tensors["masks"],
                K=K,
                centroids=centroids,
                cad_models=cad_models,
                R_gt=None,
                t_gt=None,
                # outputs
                model_outputs=model_outputs,
                # parameters
                pc_eps=cert_pc_eps,
                pc_clamp_thres=cert_pc_clamp_thres,
                mask_eps=cert_mask_eps,
                tb_writer=None,
                tb_global_step=None,
                device=device,
                cfg=cfg,
            )

            batch_pylds["scene_id"] = data["metadata"]["scene_id"]
            batch_pylds["view_id"] = data["metadata"]["view_id"]
            batch_pylds["category_id"] = data["metadata"]["category_id"]
            batch_pylds["det_scores"] = data["metadata"]["det_scores"]
            pylds.extend(pd.DataFrame(batch_pylds).to_dict("records"))

        del pc, centroids, batch_pylds, imgs, K, model_outputs
        torch.cuda.empty_cache()

    return pylds


def eval_detector(cfg, model_id="obj_000001", **kwargs):
    """Main training function"""
    dataset = cfg["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running self supervised training: {datetime.now()}")
    logging.info(f"device is {device}")
    torch.cuda.empty_cache()
    mutils.set_all_random_seeds(42)

    # object datasets
    all_cad_models = load_all_cad_models(device=device, cfg=cfg)
    object_ds, mesh_db_batched = load_objects(cfg)
    _, _, original_cad_model, original_model_keypoints, obj_diameter = load_c3po_cad_models(
        model_id, device, output_unit="m", cfg=cfg
    )

    # create dataset and dataloader
    # note: validation set dataloader has a subset random sampler
    ds_train, ds_val, ds_iter_train, ds_iter_val = None, None, None, None
    if cfg["training"]["dataloader_type"] == "single_obj_pc_img":
        collate_f = lambda x: collate_c3po_cosypose(x, device=device, cfg=cfg)
        ds_train, ds_val, ds_iter_train, ds_iter_val = load_single_obj_pc_img_dataset(
            object_id=model_id, obj_diameter=obj_diameter, collate_fn=collate_f, cfg=cfg
        )

    # load batch renderer
    batch_renderer = load_batch_renderer(cfg)

    # certifier
    cert_model = load_certifier(all_cad_models, batch_renderer, cfg)

    # models, optimizers and schedulers
    models = {}
    if "c3po" in set(cfg["models_to_use"]):
        if len(set(cfg["models_to_use"])) == 1:
            cfg["c3po"]["load_pretrained_weights"] = True
        models["c3po"] = load_c3po_model(
            model_id=model_id,
            cad_models=original_cad_model,
            model_keypoints=original_model_keypoints,
            object_diameter=obj_diameter,
            device=device,
            cfg=cfg,
        )

    if "cosypose_coarse_refine" in set(cfg["models_to_use"]):
        models["cosypose"] = load_cosypose_coarse_refine_model(
            batch_renderer=batch_renderer,
            mesh_db_batched=mesh_db_batched,
            model_id=model_id,
            cad_models=original_cad_model,
            model_keypoints=original_model_keypoints,
            object_diameter=obj_diameter,
            cfg_dict=cfg,
        )

    if len(cfg["models_to_use"]) > 1:
        logging.info("Logging parallel model weights.")
        ckpt_path = kwargs["checkpoint_path"]
        logging.info(f"Loading checkpoint from {ckpt_path}")
        save = torch.load(ckpt_path)
        for k in models.keys():
            models[k].load_state_dict(save[k]["state_dict"])

    # tensorboard loss writer
    tb_writer = SummaryWriter(os.path.join(cfg["training"]["tb_log_dir"], cfg["dataset"], cfg["timestamp"]))

    # evaluation
    visualize = False
    save_render = False
    if visualize or save_render:
        _, original_cad_mesh = load_c3po_cad_meshes(model_id, device, output_unit="m", cfg=cfg)
    else:
        original_cad_mesh = None

    if cfg["detector"]["det_type"] == "gt":
        logging.info("Evaluating using GT detections.")
        eval_results = eval(
            ds_iter_train,
            object_id=model_id,
            object_diameter=obj_diameter,
            models=models,
            cert_model=cert_model,
            cad_models=original_cad_model,
            object_ds=object_ds,
            cad_mesh=original_cad_mesh,
            device=device,
            tensorboard_writer=tb_writer,
            visualize=visualize,
            save_render=save_render,
            cfg=cfg,
        )
    else:
        logging.info("Evaluating using non-GT detections.")
        eval_results = eval_nongt_detections(
            ds_iter_train,
            object_id=model_id,
            object_diameter=obj_diameter,
            models=models,
            cert_model=cert_model,
            cad_models=original_cad_model,
            object_ds=object_ds,
            cad_mesh=original_cad_mesh,
            device=device,
            tensorboard_writer=tb_writer,
            visualize=visualize,
            save_render=save_render,
            cfg=cfg,
        )

    # save results
    model_save_dir = cfg["save_folder"]
    save_name = f"{model_id}_{'_'.join(sorted(cfg['models_to_use']))}_eval_data.pkl"
    with open(os.path.join(model_save_dir, save_name), "wb") as outp:
        pickle.dump(eval_results, outp, pickle.HIGHEST_PROTOCOL)
