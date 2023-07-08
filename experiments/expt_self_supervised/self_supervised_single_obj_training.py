"""
This code implements supervised training of keypoint detector in simulation.

It can use registration during supervised training.

"""
import logging
import shutil
import os
import pickle
from tqdm import tqdm
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from datasets import ycbv, tless
from proposed_model import PointsRegressionModel as ProposedModel
from proposed_model import (
    load_c3po_model,
    load_c3po_cad_models,
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

from training_utils import *


def certified_loss(
    cert_model,
    object_id,
    input_imgs,
    input_pc,
    detected_masks,
    K,
    centroids,
    cad_models,
    out_c3po,
    out_csy,
    pc_eps,
    pc_clamp_thres,
    mask_eps,
    tb_writer,
    tb_global_step,
    cfg,
):
    """Helper function to certify C3PO and Cosypose

    Args:
        cert_model:
        out_c3po:
        out_csy:
        cfg:
    """
    # Certification pseudocode
    #
    # Input:
    #       (gt) segmentation mask
    #       intrinsics
    #       cad models
    #       outputs from C3PO (predicted keypoints, rotation, translation estimation)
    #       outputs from CosyPose (predicted rotation, translation)
    # 1. certify C3PO with point cloud certification
    #    - histogram based certifier
    #    - returns a batch of (0,1) that indicates certification successes
    # 2. certify C3PO with the projection certification
    #    - half IOU based certifier
    #    - returns a batch of (0,1) that indicates certification successes
    # 3. repeat the same for CosyPose
    B = out_c3po[0].shape[0]
    pc_len = input_pc.shape[-1]
    csy_preds = out_csy[0]
    obj_infos = [dict(name=csy_preds.infos.loc[i, "label"]) for i in range(len(csy_preds))]
    device = out_c3po[0].device
    csy_predicted_pc = csy_preds.poses[:, :3, :3] @ cad_models + csy_preds.poses[:, :3, -1][..., None] - centroids
    assert len(obj_infos) == B

    # c3po certification
    with torch.no_grad():
        cert_c3po_pc = cert_model.certify_pc(
            input_pc=input_pc, predicted_pc=out_c3po[0], eps=pc_eps, clamp_thres=pc_clamp_thres
        )
        if cfg["visualization"]["certifier_pc"]["c3po"]:
            logging.info("Visualizing point clouds for C3PO certification.")
            for b in range(B):
                logging.info(f"Certificate value: {cert_c3po_pc[b]}")
                vutils.visualize_pcs([input_pc[b, ...], out_c3po[0][b, ...]], colors=[[0, 1, 0], [1, 0, 0]])

        cert_c3po_mask, c3po_mask_scores, c3po_pred_masks = cert_model.certify_mask(
            detected_masks=detected_masks,
            predicted_poses=mutils.make_se3_batched(out_c3po[2], out_c3po[3] + centroids),
            obj_infos=obj_infos,
            K=K,
            resolution=input_imgs.shape[-2:],
            eps=mask_eps,
        )
        if cfg["visualization"]["certifier_rendered_mask"]["c3po"]:
            logging.info("Visualizing rendered mask for C3PO certification.")
            # visualizing detected & predicted masks
            vutils.visualize_det_and_pred_masks(
                rgbs=input_imgs,
                batch_im_id=csy_preds.infos["batch_im_id"],
                det_masks=detected_masks,
                pred_masks=c3po_pred_masks,
                cert_scores=c3po_mask_scores,
                show=True,
            )

        # cosypose certification
        cert_csy_pc = cert_model.certify_pc(
            input_pc=input_pc, predicted_pc=csy_predicted_pc, eps=pc_eps, clamp_thres=pc_clamp_thres
        )
        if cfg["visualization"]["certifier_pc"]["cosypose"]:
            logging.info("Visualizing point clouds for Cosypose certification.")
            for b in range(B):
                logging.info(f"Certificate value: {cert_csy_pc[b]}")
                vutils.visualize_pcs([input_pc[b, ...], csy_predicted_pc[b, ...]], colors=[[0, 1, 0], [1, 0, 0]])

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

        # Certification outputs
        #      cosy pose  |  c3po
        # -------------------------
        # 3D  |  (B,1)    | (B,1)
        # 2D  |  (B,1)    | (B,1)
        #
        # 4. AND reduction along dim=1
        #  cosy pose  |  c3po
        # --------------------
        #   c_a =(B,1)     |  c_b=(B,1)
        # 5.
        # c_a: mask on certifiable cosypose results
        # c_b: mask on certifiable c3po results
        # loss = c_a * self-sup-loss(cosypose outputs * c_a)
        #       + c_a * sup-loss(cosypose outputs * c_a, c3po outputs * c_a)
        #       + c_b * self-sup-loss(c3po outputs * c_b)
        #       + c_b * sup-loss(c3po outputs * c_b, cosypose outputs * c_b)
        # where self-supervised = half-chamfer, supervised = chamfer
        cert_csy = torch.logical_and(cert_csy_pc, cert_csy_mask)
        cert_c3po = torch.logical_and(cert_c3po_pc, cert_c3po_mask)
        cert_c3po_2d_count, cert_c3po_3d_count = torch.sum(cert_c3po_mask), torch.sum(cert_c3po_pc)
        cert_csy_2d_count, cert_csy_3d_count = torch.sum(cert_csy_mask), torch.sum(cert_csy_pc)
        cert_csy_count = torch.sum(cert_csy)
        cert_c3po_count = torch.sum(cert_c3po)
        cert_batch_size = torch.sum(torch.logical_or(cert_c3po, cert_csy))

    # loss components
    # 1. self-supervised half chamfer loss
    # 2. supervised chamfer loss
    self_sup_csy_loss, pseudo_gt_c3po_loss, self_sup_c3po_loss, pseudo_gt_csy_loss = (
        torch.zeros(1, device=device, requires_grad=True),
        torch.zeros(1, device=device, requires_grad=True),
        torch.zeros(1, device=device, requires_grad=True),
        torch.zeros(1, device=device, requires_grad=True),
    )

    if cert_c3po_count > 0:
        self_sup_c3po_loss = (
            torch.sum(
                cert_c3po * sq_half_chamfer_dists_clamped(input_pc, out_c3po[0], pc_clamp_thres).sum(dim=1) / pc_len
            )
            / cert_c3po_count
        )

        pseudo_c3po_gt = out_c3po[0].detach()
        pseudo_gt_csy_loss = (
            torch.sum(
                cert_c3po
                * (
                    sq_half_chamfer_dists(csy_predicted_pc, pseudo_c3po_gt) / csy_predicted_pc.shape[-1]
                    + sq_half_chamfer_dists(pseudo_c3po_gt, csy_predicted_pc) / pseudo_c3po_gt.shape[-1]
                ).sum(dim=1)
            )
            / cert_c3po_count
        )

        if cfg["visualization"]["certifier_loss_calc"]:
            for b in range(B):
                if cert_c3po[b]:
                    logging.info(
                        "Visualizing point clouds for pseudo_gt_csy_loss (C3PO certified output supervised Cosypose)."
                    )
                    vutils.visualize_pcs(
                        [
                            pseudo_c3po_gt[b, ...],
                            input_pc[b, ...],
                            csy_predicted_pc[b, ...],
                        ],
                        colors=[[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                    )

    if cert_csy_count > 0:
        self_sup_csy_loss = (
            torch.sum(
                cert_csy * sq_half_chamfer_dists_clamped(input_pc, csy_predicted_pc, pc_clamp_thres).sum(dim=1) / pc_len
            )
            / cert_csy_count
        )

        psuedo_csy_gt = csy_predicted_pc.detach()
        pseudo_gt_c3po_loss = (
            torch.sum(
                cert_csy
                * (
                    sq_half_chamfer_dists(out_c3po[0], psuedo_csy_gt) / out_c3po[0].shape[-1]
                    + sq_half_chamfer_dists(psuedo_csy_gt, out_c3po[0]) / psuedo_csy_gt.shape[-1]
                ).sum(dim=1)
            )
            / cert_csy_count
        )

        if cfg["visualization"]["certifier_loss_calc"]:
            for b in range(B):
                if cert_csy[b]:
                    logging.info(
                        "Visualizing point clouds for pseudo_gt_c3po_loss (Cosypose certified output supervised C3PO)."
                    )
                    vutils.visualize_pcs(
                        [psuedo_csy_gt[b, ...], input_pc[b, ...], out_c3po[0][b, ...]],
                        colors=[[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                    )

    # 3. c3po kpt correction loss
    correction_MSE_c3po = torch.nn.MSELoss(reduction="none")
    if out_c3po[4] is None:
        c3po_correction_loss = torch.zeros_like(self_sup_c3po_loss).mean()
    else:
        c3po_correction_loss = torch.mean(
            cert_c3po * correction_MSE_c3po(out_c3po[4], torch.zeros_like(out_c3po[4])).sum(dim=1).mean(dim=1)
        )

    # 4. cosypose kpt correction loss
    correction_MSE_cosypose = torch.nn.MSELoss(reduction="none")
    if "correction" not in out_csy[1].keys():
        csy_correction_loss = torch.zeros(1, device=device, requires_grad=True)
    else:
        csy_correction_loss = torch.mean(
            cert_csy
            * correction_MSE_cosypose(out_csy[1]["correction"], torch.zeros_like(out_csy[1]["correction"]))
            .sum(dim=1)
            .mean(dim=1)
        )

    loss = (
        cfg["training"]["loss_weights"]["c3po"]["self_sup"] * self_sup_c3po_loss
        + cfg["training"]["loss_weights"]["c3po"]["pseudo_gt_sup"] * pseudo_gt_c3po_loss
        + cfg["training"]["loss_weights"]["c3po"]["correction"] * c3po_correction_loss
        + cfg["training"]["loss_weights"]["cosypose"]["self_sup"] * self_sup_csy_loss
        + cfg["training"]["loss_weights"]["cosypose"]["pseudo_gt_sup"] * pseudo_gt_csy_loss
        + cfg["training"]["loss_weights"]["cosypose"]["correction"] * csy_correction_loss
    )

    # tensorboard logging
    if tb_writer is not None:
        tb_writer.add_scalars(
            f"Loss/self-supervised/train/{object_id}",
            {
                "c3po_self_sup": self_sup_c3po_loss.item(),
                "c3po_cert_sup": pseudo_gt_c3po_loss.item(),
                "cosypose_self_sup": self_sup_csy_loss.item(),
                "cosypose_cert_sup": pseudo_gt_csy_loss.item(),
                "c3po_correction": c3po_correction_loss.item(),
                "csy_correction": csy_correction_loss.item(),
                "total": loss.item(),
            },
            tb_global_step,
        )

        tb_writer.add_scalars(
            f"FractionCert/train/{object_id}",
            {
                "c3po-2d": cert_c3po_2d_count.item() / B,
                "c3po-3d": cert_c3po_3d_count.item() / B,
                "csy-2d": cert_csy_2d_count.item() / B,
                "csy-3d": cert_csy_3d_count.item() / B,
                "c3po": cert_c3po_count.item() / B,
                "cosypose": cert_csy_count.item() / B,
                "total": cert_batch_size.item() / B,
            },
            global_step=tb_global_step,
        )

    pyld = dict(
        c3po_cert={
            "pc": cert_c3po_pc,
            "mask": cert_c3po_mask,
            "combined": cert_c3po,
        },
        cosypose_cert={
            "pc": cert_csy_pc,
            "mask": cert_csy_mask,
            "combined": cert_csy,
        },
    )

    return pyld, cert_batch_size.item(), loss


def collate_c3po_cosypose(data, device, cfg):
    """Custom collate function to prepare data from ObjectPoseDataset for C3PO & Cosypose"""
    # each entry in data: rgb, filtered mask, pyld
    # pyld keys: see get_obj_data_from_scene_frame_index in pose.py
    batch_size = len(data)
    h, w = data[0][0].shape[1], data[0][0].shape[2]

    # build gt detections
    seg_outputs = []
    if cfg["detector"]["det_type"] == "gt":
        for bid in range(batch_size):
            # note: each entry only contains one object detection
            # we put them into lists just to be compatible with cosypose's input conventions
            c_bboxes = [torch.as_tensor(data[bid][-1]["obj"]["bbox"]).float()]
            c_labels = [
                torch.as_tensor(cfg["detector"]["label_to_category_id"][data[bid][-1]["obj"]["name"]]).to(torch.uint8)
            ]
            c_scores = [torch.as_tensor(1.0, dtype=torch.float)]
            # c_masks: (1, 1, h, w)
            c_masks = torch.as_tensor(data[bid][1]).to(torch.uint8).unsqueeze(0).unsqueeze(0)
            # c_pcs shape: (B, 3, N)
            seg_outputs.append(dict(boxes=c_bboxes, labels=c_labels, scores=c_scores, masks=c_masks))
    else:
        for bid in range(batch_size):
            c_bboxes = [torch.as_tensor(data[bid][-1]["bbox"]).float()]
            c_labels = [torch.as_tensor(data[bid][-1]["category_id"]).to(torch.uint8)]
            c_scores = [torch.as_tensor(data[bid][-1]["score"]).float()]
            c_masks = torch.as_tensor(data[bid][1]).to(torch.uint8).unsqueeze(0).unsqueeze(0)
            seg_outputs.append(dict(boxes=c_bboxes, labels=c_labels, scores=c_scores, masks=c_masks))

    # to cosypose format
    cosypose_input_detections = build_cosypose_detections(
        img_height=h,
        img_width=w,
        seg_outputs=seg_outputs,
        cfg=cfg,
        output_masks=True,
        output_pcs=False,
        detection_th=None,
        one_instance_per_class=False,
    )

    # batch the images
    images = torch.stack([x[0] for x in data]).float() / 255.0
    Ks = torch.stack([x[-1]["K"] for x in data])

    # prepare C3PO batched data
    centered_normalized_pc = torch.stack([x[-1]["centered_normalized_pc"] for x in data])
    centroids = torch.stack([x[-1]["centroid"] for x in data])

    # only when using gt
    if cfg["detector"]["det_type"] == "gt":
        centered_normalized_kpts = torch.stack([x[-1]["centered_normalized_kpts"] for x in data])
        cent_R_cad = torch.stack([x[-1]["cent_R_cad"] for x in data])
        cent_t_cad = torch.stack([x[-1]["cent_t_cad"] for x in data])
    else:
        centered_normalized_kpts, cent_R_cad, cent_t_cad = None, None, None

    # annotate meta information
    metadata = {}
    if "scene_id" in data[0][-1].keys():
        scene_ids, view_ids, cat_ids, det_scores = [], [], [], []
        for bid in range(batch_size):
            scene_ids.append(data[bid][-1]["scene_id"])
            view_ids.append(data[bid][-1]["view_id"])
            cat_ids.append(data[bid][-1]["category_id"])
            det_scores.append(data[bid][-1]["score"])
        metadata["scene_id"] = scene_ids
        metadata["view_id"] = view_ids
        metadata["category_id"] = cat_ids
        metadata["det_scores"] = det_scores

    pyld = dict(
        metadata=metadata,
        cosypose=(images, Ks, cosypose_input_detections),
        c3po=(centered_normalized_pc, centered_normalized_kpts, cent_R_cad, cent_t_cad, centroids),
    )
    return pyld


def train_one_epoch(
    training_loader,
    object_id,
    mesh_db_batched,
    cad_models_db,
    models,
    cert_model,
    optimizers,
    current_epoch_num,
    tensorboard_writer,
    visualize=False,
    device=None,
    cfg=None,
):
    running_loss = 0.0
    epoch_size = len(training_loader)
    object_diameter = models["c3po"].object_diameter
    nf = 1.0
    if cfg["training"]["normalize_pc"]:
        nf = object_diameter

    cert_pc_eps = cfg["certifier"]["epsilon_pc"][object_id] * object_diameter
    cert_pc_clamp_thres = cfg["certifier"]["clamp_thres"][object_id] * object_diameter
    cert_mask_eps = cfg["certifier"]["epsilon_mask"][object_id]

    multi_zero_grad(optimizers)
    for i, data in enumerate(training_loader):
        # Zero your gradients for every batch!
        # c3po forward pass
        # note: predicted keypoints are in the centered frame
        pc, centroids = data["c3po"][0].to(device), data["c3po"][4].to(device)
        pc_invalid_mask = (pc[:, :3, :] == torch.zeros(3, 1).to(device=pc.device)).sum(dim=1) == 3
        if torch.any(pc_invalid_mask):
            del pc, centroids, pc_invalid_mask
            logging.warning("Point cloud contains zero paddings encountered in training. Skipping batch.")
            continue

        out_c3po = models["c3po"](object_id, pc)

        # cosypose forward pass
        imgs, K, cosypose_det = (
            data["cosypose"][0].to(device),
            data["cosypose"][1].to(device),
            data["cosypose"][2].to(device),
        )
        out_cp = models["cosypose"](
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
            logging.info("Visualizing predictions and keypoint annotations.")
            # visualize: input point cloud, ground truth keypoint annotations, predicted keypoints
            kp = data["c3po"][1]
            if cfg["training"]["normalize_pc"]:
                kp_gt = kp
            else:
                kp_gt = kp / object_diameter
            kp_pred = out_c3po[1] / object_diameter
            vutils.visualize_gt_and_pred_keypoints(pc[:, :3, :], kp_gt, kp_pred=kp_pred)

        # certification
        cert_results, cert_batch_size, loss = certified_loss(
            cert_model=cert_model,
            # inputs
            object_id=object_id,
            input_imgs=imgs,
            input_pc=pc[:, :3, :] * nf,
            detected_masks=cosypose_det._tensors["masks"],
            K=K,
            centroids=centroids,
            cad_models=models["c3po"].cad_models,
            # outputs
            out_c3po=out_c3po,
            out_csy=out_cp,
            # parameters
            pc_eps=cert_pc_eps,
            pc_clamp_thres=cert_pc_clamp_thres,
            mask_eps=cert_mask_eps,
            tb_writer=tensorboard_writer,
            tb_global_step=current_epoch_num * epoch_size + i,
            cfg=cfg,
        )

        if cert_batch_size != 0:
            loss.backward()
            multi_step(optimizers)

        tensorboard_writer.add_scalar(
            tag=f"GradNorm/train/c3po/{object_id}",
            scalar_value=get_grad_norm(models["c3po"].parameters()),
            global_step=current_epoch_num * epoch_size + i,
        )
        tensorboard_writer.add_scalar(
            tag=f"GradNorm/train/cosypose/{object_id}",
            scalar_value=get_grad_norm(models["cosypose"].parameters()),
            global_step=current_epoch_num * epoch_size + i,
        )

        multi_zero_grad(optimizers)

        # Gather data and report
        running_loss += loss.detach().item()  # Note: the output of supervised_loss is already averaged over batch_size
        if i % 10 == 0:
            print(
                f"Batch {(i + 1)} / {len(training_loader)}, loss: {loss.item()} cert %:{cert_batch_size / pc.shape[0]}"
            )

        # delete variables
        del_tensor_iterable(out_c3po)
        del_tensor_iterable(out_cp)
        del_tensor_dictionary(cert_results)
        del_tensor_dictionary(cosypose_det._tensors)
        del imgs, K, cosypose_det, pc, centroids, out_cp, out_c3po, cert_results, cert_batch_size, loss, pc_invalid_mask
        torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)

    return ave_tloss


# Validation code
def validate(validation_loader, object_id, models, cert_model, device, tensorboard_writer, visualize=False, cfg=None):
    # We don't need gradients on to do reporting.
    with torch.no_grad():
        running_vloss, running_cert_batch_frac = 0.0, 0.0
        object_diameter = models["c3po"].object_diameter
        nf = 1.0
        if cfg["training"]["normalize_pc"]:
            nf = object_diameter

        cert_pc_eps = cfg["certifier"]["epsilon_pc"][object_id] * object_diameter
        cert_pc_clamp_thres = cfg["certifier"]["clamp_thres"][object_id] * object_diameter
        cert_mask_eps = cfg["certifier"]["epsilon_mask"][object_id]

        for i, data in tqdm(enumerate(validation_loader)):
            # c3po forward pass
            # note: predicted keypoints are in the centered frame
            pc, centroids = data["c3po"][0].to(device), data["c3po"][4].to(device)
            pc_invalid_mask = (pc[:, :3, :] == torch.zeros(3, 1).to(device=pc.device)).sum(dim=1) == 3
            if torch.any(pc_invalid_mask):
                raise ValueError("Point cloud contains zero paddings!")

            out_c3po = models["c3po"](object_id, pc)

            # cosypose forward pass
            imgs, K, cosypose_det = (
                data["cosypose"][0].to(device),
                data["cosypose"][1].to(device),
                data["cosypose"][2].to(device),
            )
            out_cp = models["cosypose"](
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
                logging.info("Visualizing predictions and keypoint annotations.")
                # visualize: input point cloud, ground truth keypoint annotations, predicted keypoints
                kp = data["c3po"][1].to(device),
                if cfg["training"]["normalize_pc"]:
                    kp_gt = kp
                else:
                    kp_gt = kp / object_diameter
                kp_pred = out_c3po[1] / object_diameter
                vutils.visualize_gt_and_pred_keypoints(pc[:, :3, :], kp_gt, kp_pred=kp_pred)

            # certification
            cert_results, cert_batch_size, vloss = certified_loss(
                cert_model=cert_model,
                # inputs
                object_id=object_id,
                input_imgs=imgs,
                input_pc=pc[:, :3, :] * nf,
                detected_masks=cosypose_det._tensors["masks"],
                K=K,
                centroids=centroids,
                cad_models=models["c3po"].cad_models,
                # outputs
                out_c3po=out_c3po,
                out_csy=out_cp,
                # parameters
                pc_eps=cert_pc_eps,
                pc_clamp_thres=cert_pc_clamp_thres,
                mask_eps=cert_mask_eps,
                tb_writer=None,
                tb_global_step=None,
                cfg=cfg,
            )

            running_vloss += vloss.item()
            running_cert_batch_frac += cert_batch_size / pc.shape[0]

        del pc, centroids, vloss, cert_results, imgs, K, out_cp, out_c3po
        torch.cuda.empty_cache()

        avg_vloss = running_vloss / (i + 1)
        avg_cert_frac = running_cert_batch_frac / (i + 1)

    return avg_vloss, avg_cert_frac


def train_with_self_supervision(
    synth_supervised_train_loader=None,
    validation_loader=None,
    mesh_db_batched=None,
    cad_models_db=None,
    models=None,
    cert_model=None,
    object_id=None,
    optimizers=None,
    schedulers=None,
    start_epoch=0,
    model_save_folder=None,
    best_model_save_fname=None,
    train_loss_save_fname=None,
    val_loss_save_fname=None,
    cert_save_fname=None,
    tensorboard_writer=None,
    device=None,
    cfg=None,
):
    num_epochs = cfg["training"]["n_epochs"]

    train_loss = TrackingMeter()
    val_loss = TrackingMeter()

    # delete logs folder
    shutil.rmtree(os.path.join(model_save_folder, "logs"), ignore_errors=True)

    for epoch in range(start_epoch, num_epochs):
        logging.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logging.info(
            f"Current LR C3PO: {schedulers['c3po'].get_last_lr()}, Cosypose {schedulers['cosypose'].get_last_lr()}"
        )

        # training
        for _, m in models.items():
            m.train(True)
        logging.info(f"Training with self-supervision on synt. data (single object, object ID: {object_id}): ")
        avg_loss_self_supervised = train_one_epoch(
            training_loader=synth_supervised_train_loader,
            models=models,
            cert_model=cert_model,
            object_id=object_id,
            optimizers=optimizers,
            current_epoch_num=epoch,
            mesh_db_batched=mesh_db_batched,
            cad_models_db=cad_models_db,
            tensorboard_writer=tensorboard_writer,
            visualize=cfg["visualization"]["c3po_outputs"],
            device=device,
            cfg=cfg,
        )
        train_loss.add_item(avg_loss_self_supervised)

        # validation
        for _, m in models.items():
            m.train(False)
        logging.info("Run validation.")
        avg_vloss, avg_cert_frac = validate(
            validation_loader=validation_loader,
            object_id=object_id,
            models=models,
            cert_model=cert_model,
            tensorboard_writer=tensorboard_writer,
            visualize=cfg["visualization"]["c3po_outputs"],
            device=device,
            cfg=cfg,
        )
        val_loss.add_item(avg_vloss)

        logging.info(f"\nLOSS self-supervised train {avg_loss_self_supervised}")
        logging.info(f"\nLOSS valid {avg_vloss}")

        # update tensorboard for validation
        tensorboard_writer.add_scalar(
            tag=f"Loss/self-supervised/val/{object_id}", scalar_value=avg_vloss, global_step=epoch
        )
        tensorboard_writer.add_scalar(
            tag=f"FractionCert/self-supervised/val/{object_id}", scalar_value=avg_cert_frac, global_step=epoch
        )

        # Saving the model every epoch
        save_pyld = {"epoch": epoch}
        for k in models.keys():
            save_pyld[k] = {}
            save_pyld[k]["state_dict"] = models[k].state_dict()
            save_pyld[k]["scheduler_state_dict"] = schedulers[k].state_dict()
            save_pyld[k]["optimizer_state_dict"] = optimizers[k].state_dict()
        torch.save(
            save_pyld,
            os.path.join(model_save_folder, f"_epoch_{epoch + 1}{best_model_save_fname}"),
        )

        multi_step(schedulers)

        with open(os.path.join(model_save_folder, train_loss_save_fname), "wb") as outp:
            pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_save_folder, val_loss_save_fname), "wb") as outp:
            pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

        # copy tb logs and make an archive
        shutil.copytree(tensorboard_writer.log_dir, os.path.join(model_save_folder, "logs"), dirs_exist_ok=True)

    return train_loss, val_loss


def train_detector(cfg, detector_type="pointnet", model_id="obj_000001", use_corrector=False, **kwargs):
    """Main training function"""
    dataset = cfg["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running self supervised training: {datetime.now()}")
    logging.info(f"device is {device}")
    torch.cuda.empty_cache()

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

    # load model
    batch_renderer = load_batch_renderer(cfg)

    # models, optimizers and schedulers
    models, optimizers, schedulers = {}, {}, {}
    models["c3po"] = load_c3po_model(
        model_id=model_id,
        cad_models=original_cad_model,
        model_keypoints=original_model_keypoints,
        object_diameter=obj_diameter,
        device=device,
        cfg=cfg,
    )
    models["cosypose"] = load_cosypose_coarse_refine_model(
        batch_renderer=batch_renderer,
        mesh_db_batched=mesh_db_batched,
        model_id=model_id,
        cad_models=original_cad_model,
        model_keypoints=original_model_keypoints,
        object_diameter=obj_diameter,
        cfg_dict=cfg,
    )
    optimizers["c3po"], schedulers["c3po"] = make_sgd_optimizer_scheduler(models["c3po"], cfg["c3po"])
    optimizers["cosypose"], schedulers["cosypose"] = make_sgd_optimizer_scheduler(
        models["cosypose"], cfg["cosypose_coarse_refine"]
    )

    # certifier
    cert_model = load_certifier(all_cad_models, batch_renderer, cfg)

    # model save locations
    model_save_dir = cfg["save_folder"]
    best_model_save_fname = f"_self_supervised_single_obj_joint.pth.tar"
    train_loss_save_fname = "_sstrain_loss.pkl"
    val_loss_save_fname = "_ssval_loss.pkl"
    cert_save_fname = "_certi_all_batches.pkl"

    if kwargs["resume_run"]:
        ckpt_path = kwargs["checkpoint_path"]
        logging.info(f"Loading checkpoint from {ckpt_path}")
        save = torch.load(ckpt_path)

        for k in models.keys():
            models[k].load_state_dict(save[k]["state_dict"])
            schedulers[k].load_state_dict(save[k]["scheduler_state_dict"])
            optimizers[k].load_state_dict(save[k]["optimizer_state_dict"])

        start_epoch = save["epoch"] + 1
    else:
        start_epoch = 0

    # tensorboard loss writer
    tb_writer = SummaryWriter(os.path.join(cfg["training"]["tb_log_dir"], cfg["dataset"], cfg["timestamp"]))

    # training
    train_loss, val_loss = train_with_self_supervision(
        synth_supervised_train_loader=ds_iter_train,
        validation_loader=ds_iter_val,
        mesh_db_batched=mesh_db_batched,
        models=models,
        cert_model=cert_model,
        object_id=model_id,
        schedulers=schedulers,
        start_epoch=start_epoch,
        optimizers=optimizers,
        model_save_folder=model_save_dir,
        best_model_save_fname=best_model_save_fname,
        train_loss_save_fname=train_loss_save_fname,
        val_loss_save_fname=val_loss_save_fname,
        cert_save_fname=cert_save_fname,
        tensorboard_writer=tb_writer,
        device=device,
        cfg=cfg,
    )
