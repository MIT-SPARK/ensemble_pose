"""
This code implements supervised and self-supervised training, and validation, for keypoint detector with registration.
It uses registration during supervised training. It uses registration plus corrector during self-supervised training.

"""

import os
import pickle
import torch.utils.data
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# experiment specific imports
from proposed_model import (
    MultiModel,
    load_segmentation_model,
    load_multi_model,
    load_certifier,
    load_all_cad_models,
    load_batch_renderer,
)
from training_utils import *
from utils.loss_functions import (
    self_supervised_training_loss as self_supervised_loss,
    self_supervised_validation_loss as validation_loss,
)
from utils.math_utils import depth_to_point_cloud_batched
from utils.torch_utils import cast2cuda


def load_objects(cfg_dict):
    """Load objects for this dataset"""
    cfg = CosyposeCfg()
    cfg.load_from_dict(cfg_dict)

    if cfg.object_set == "tless":
        object_ds_name, urdf_ds_name = "tless.bop", "tless.cad"
    else:
        object_ds_name, urdf_ds_name = "ycbv.bop-compat.eval", "ycbv"

    object_ds = make_object_dataset(object_ds_name, bop_ds_dir=Path(cfg_dict["bop_ds_dir"]))
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    mesh_db_batched = mesh_db.batched(n_sym=cfg.n_symmetries_batch).cuda().float()
    return object_ds, mesh_db_batched


def load_rand_objs_datasets(cfg):
    """Load relevant datasets"""
    cfg_training = cfg["training"]

    scene_ds_train = make_scene_dataset(cfg["train_ds_name"], bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)
    scene_ds_val = make_scene_dataset(cfg["val_ds_name"], bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)

    ds_kwargs = dict(
        resize=cfg_training["input_resize"],
        rgb_augmentation=cfg_training["rgb_augmentation"],
        background_augmentation=cfg_training["background_augmentation"],
        min_area=cfg_training["min_area"],
        gray_augmentation=cfg_training["gray_augmentation"],
    )
    ds_train = PoseDataset(scene_ds_train, **ds_kwargs)
    ds_val = PoseDataset(scene_ds_val, **ds_kwargs)

    train_sampler = PartialSampler(ds_train, epoch_size=cfg_training["epoch_size"])
    ds_iter_train = torch.utils.data.DataLoader(
        ds_train,
        sampler=train_sampler,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        collate_fn=ds_train.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)

    val_sampler = PartialSampler(ds_val, epoch_size=int(0.1 * cfg_training["epoch_size"]))
    ds_iter_val = torch.utils.data.DataLoader(
        ds_val,
        sampler=val_sampler,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        collate_fn=ds_val.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    ds_iter_val = MultiEpochDataLoader(ds_iter_val)

    return ds_train, ds_val, ds_iter_train, ds_iter_val


def load_frame_objs_datasets(cfg):
    """Load data loaders that return frames directly"""
    cfg_training = cfg["training"]

    scene_ds_train = make_scene_dataset(cfg["train_ds_name"], bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)
    scene_ds_val = make_scene_dataset(cfg["val_ds_name"], bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)

    if cfg["dataset"] == "ycbv":
        all_model_names = list(bop_constants.YCBV.keys())
    elif cfg["dataset"] == "tless":
        all_model_names = list(bop_constants.TLESS.keys())
    else:
        raise NotImplementedError
    ds_kwargs = dict(
        all_model_names=all_model_names,
        pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
        min_area=cfg_training["min_area"],
    )
    ds_train = FramePoseDataset(scene_ds_train, **ds_kwargs)
    ds_val = FramePoseDataset(scene_ds_val, **ds_kwargs)

    ds_iter_train = torch.utils.data.DataLoader(
        ds_train,
        shuffle=False,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        collate_fn=ds_train.collate_fn,
        drop_last=False,
        pin_memory=True,
    )

    # sub-sample validation set according to params provided
    # equally spaced sampling
    val_indices = torch.linspace(
        start=0, end=ds_val.__len__(), steps=cfg["training"]["val_ds_size"], dtype=torch.int64
    ).tolist()
    ds_iter_val = torch.utils.data.DataLoader(
        ds_val,
        shuffle=False,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        collate_fn=ds_val.collate_fn,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        drop_last=False,
        pin_memory=True,
    )

    return ds_train, ds_val, ds_iter_train, ds_iter_val


def build_multi_model_inputs(
    available_modules=None, images=None, K=None, labels=None, TCO=None, n_iterations=None, masked_pcs=None
):
    """Make inputs to Multimodel

    Args:
        model_ids: batched model_ids
        masked_pcs: batched point clouds
    """
    inputs = {}
    if "c3po_multi" in available_modules:
        inputs["c3po_multi"] = dict(model_ids=labels, input_point_clouds=masked_pcs)
    if "c3po" in available_modules:
        inputs["c3po"] = dict(model_ids=labels[0], input_point_clouds=masked_pcs)
    if "cosypose" in available_modules:
        inputs["cosypose"] = dict(images=images, K=K, labels=labels, TCO=TCO, n_iterations=n_iterations)
    return inputs


def self_supervised_train_one_epoch(training_loader, mesh_db_batched, seg_model, model, optimizer, device, cfg):
    raise NotImplementedError("Deprecated!")
    running_loss = 0.0
    fra_certi_track = []

    for i, data in enumerate(training_loader):
        optimizer.zero_grad()

        # data is a PoseData
        # prepare data and parameters
        batch_size, _, h, w = data.images.shape
        images = cast2cuda(data.images).float() / 255.0  # (B, 3, H, W)
        K = cast2cuda(data.K).float()  # (B, 3, 3)
        TCO_gt = cast2cuda(data.TCO).float()  # (B, 4, 4)
        bboxes = cast2cuda(data.bboxes).float()  # (B, 4)
        depths = cast2cuda(data.depths).float()  # (B, H, W)
        labels = np.array([obj["name"] for obj in data.objects])  # (B,)
        ids_in_seg = np.array([obj["id_in_segm"] for obj in data.objects])  # (B,)

        # perform segmentation
        if cfg["seg_type"] == "gt":
            # masks size: (B, H, W)
            # note: masks actually contains all the objects in the frame, not only the objects of interest
            masks = data.masks
        else:
            seg_predictions = seg_model(images=images)
            raise NotImplementedError

        # mask out the depth point cloud
        # obtain per-batch object mask
        filtered_masks = torch.zeros_like(masks, dtype=torch.bool)
        for b in range(batch_size):
            filtered_masks[b, :, :] = masks[b, :, :] == ids_in_seg[b]

        # visualize masks
        if cfg["visualization"]["rgb_segmentation"]:
            logging.info("Drawing segmentations and bboxes")
            vutils.visualize_batched_bop_masks(rgbs=images, bboxes=bboxes, masks=filtered_masks, show=True)

        batched_pcs = depth_to_point_cloud_batched(
            depths,
            K,
            x_index=2,
            y_index=1,
            mask=filtered_masks.to(device),
            pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
            device=device,
        )

        # visualize cropped point clouds
        if cfg["visualization"]["pc_segmentation"]:
            logging.info("Drawing masked point clouds")
            scene_pcs = depth_to_point_cloud_batched(
                depths,
                K,
                x_index=2,
                y_index=1,
                mask=None,
                pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
                device=device,
            )
            vutils.visualize_batched_bop_point_clouds(
                scene_pcs=scene_pcs,
                masked_scene_pcs=batched_pcs,
                batched_obj_labels=labels,
                mesh_db=mesh_db_batched,
                Ts=TCO_gt,
            )

        # forward pass on the MultiModel
        # make sure the order is consistent with the configuration yaml file's spec
        # c3po model inputs:
        # - input point cloud
        # cosypose model inputs:
        # - images, K, labels, TCO, n_iterations
        # TODO: Use coarse + refine
        inputs = build_multi_model_inputs(
            available_modules=model.available_models,
            images=images,
            K=K,
            labels=labels,
            TCO=TCO_gt,
            n_iterations=cfg["cosypose"]["n_iterations"],
            masked_pcs=batched_pcs,
        )
        outputs = model(inputs)
        breakpoint()

        # Certification
        certi = certify(
            input_point_cloud=input_point_cloud,
            predicted_point_cloud=predicted_point_cloud,
            corrected_keypoints=corrected_keypoints,
            predicted_model_keypoints=predicted_model_keypoints,
            epsilon=cfg["training"]["cert_epsilon"],
        )
        certi = certi.squeeze(-1)  # (B,)

        # Compute the loss and its gradients
        loss, pc_loss, kp_loss, fra_cert = self_supervised_loss(
            input_point_cloud=input_point_cloud,
            predicted_point_cloud=predicted_point_cloud,
            keypoint_correction=correction,
            certi=certi,
            theta=cfg["training"]["loss_theta"],
        )
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1 == 0:
            logging.info(f"Batch {(i + 1)} loss: {loss.item()} pc loss: {pc_loss.item()} kp loss: {kp_loss.item()}")
            logging.info(f"Batch {(i + 1)} fra cert: {fra_cert.item()}")

        fra_certi_track.append(fra_cert)

        del input_point_cloud, predicted_point_cloud, correction

    ave_tloss = running_loss / (i + 1)

    return ave_tloss, fra_certi_track


def self_supervised_train_one_epoch_frame_objs_rgb_only(
    training_loader,
    mesh_db_batched,
    seg_model,
    model,
    cert_model,
    optimizer,
    per_object_train_flags,
    tensorboard_writer,
    device,
    cfg,
):
    depth_method_name = "c3po_multi"

    # storing losses and fraction certifiable for each method & objects
    running_loss = dict(c3po_multi={k: 0.0 for k in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()})
    fra_certi_track = dict(c3po_multi={k: [] for k in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()})
    per_object_global_steps = dict(c3po_multi={k: 0 for k in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()})
    trained_object_ids = set()

    for i, data in enumerate(tqdm(training_loader)):
        optimizer.zero_grad()

        # data is a PoseData
        # prepare data and parameters
        batch_size, _, h, w = data.images.shape
        images = cast2cuda(data.images).float() / 255.0  # (B, 3, H, W)
        K = cast2cuda(data.K).float()  # (B, 3, 3)

        if cfg["detector"]["det_type"] == "gt":
            # cosypose detections: list of dictionaries, with the following keys:
            # - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
            #           ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            # - labels (Int64Tensor[N]): the predicted labels for each image
            # - scores (Tensor[N]): the scores or each prediction
            # - masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to
            #           obtain the final segmentation masks, the soft masks can be thresholded, generally
            #           with a value of 0.5 (mask >= 0.5)
            gt_seg_outputs = []
            for bid in range(batch_size):
                c_bboxes = [
                    cast2cuda(torch.as_tensor(data.objects[obj_idx]["bbox"])).float()
                    for obj_idx in data.frame_to_objects_index[bid]
                ]
                c_labels = [
                    torch.as_tensor(
                        cfg["detector"]["label_to_category_id"][data.objects[obj_idx]["name"]], dtype=torch.int64
                    )
                    for obj_idx in data.frame_to_objects_index[bid]
                ]
                c_scores = [torch.as_tensor(1.0, dtype=torch.float) for _ in data.frame_to_objects_index[bid]]
                c_masks = torch.stack(
                    [
                        torch.as_tensor(data.objects[obj_idx]["obj_mask"]).to(torch.uint8)
                        for obj_idx in data.frame_to_objects_index[bid]
                    ]
                ).unsqueeze(1)
                # c_pcs shape: (B, 3, N)
                c_pcs = torch.stack(
                    [
                        torch.as_tensor(data.objects[obj_idx]["point_cloud"]).to(torch.uint8)
                        for obj_idx in data.frame_to_objects_index[bid]
                    ]
                )
                gt_seg_outputs.append(
                    dict(boxes=c_bboxes, labels=c_labels, scores=c_scores, masks=c_masks, point_clouds=c_pcs)
                )

            cosypose_input_detections = build_cosypose_detections(
                img_height=h,
                img_width=w,
                seg_outputs=gt_seg_outputs,
                cfg=cfg,
                output_masks=True,
                output_pcs=True,
                detection_th=None,
                one_instance_per_class=False,
                device=device
            )
        else:
            raise NotImplementedError

        # forward pass on the MultiModel
        # make sure the order is consistent with the configuration yaml file's spec
        # c3po model inputs:
        # - input point cloud
        # NOTE: Check the input/model object point clouds' scale
        # make sure to normalize input for C3PO
        inputs = dict(
            cosypose_coarse_refine=dict(
                images=images,
                K=K,
                detections=cosypose_input_detections,
                TCO=None,  # if TCO=None, then cosypose will use a default canonical initial pose
                n_coarse_iterations=cfg["cosypose_coarse_refine"]["coarse"]["n_iterations"],
                n_refiner_iterations=cfg["cosypose_coarse_refine"]["refiner"]["n_iterations"],
            ),
        )

        # outputs format:
        # c3po: a dictionary with keys = object names, values = tuples containing:
        #       predicted_pc, corrected_kpts, R, t, correction, predicted_model_kpts
        outputs = model(**inputs)

        manage_visualization(
            data=data, model=model, model_inputs=inputs, model_outputs=outputs, mesh_db=mesh_db_batched, cfg=cfg
        )

        breakpoint()
        # certification
        certified_outputs = cert_model.certify(
            inputs,
            outputs,
            K=torch.stack([data.K[obj["frame_id_in_batch"], ...] for obj in data.objects]).to(device=device),
            resolution=data.images[0, ...].shape[-2:],
        )
        breakpoint()

        # loss (for each object)
        # TODO: New self-supervised loss
        #for obj_label in certified_outputs[depth_method_name].keys():
        #    per_object_global_steps[depth_method_name][obj_label] += 1

        #    predicted_point_cloud, _, _, _, keypoint_correction, _ = outputs[depth_method_name][obj_label]
        #    loss, pc_loss, kp_loss, fra_cert = self_supervised_loss(
        #        input_point_cloud=inputs[depth_method_name]["object_batched_pcs"][obj_label],
        #        predicted_point_cloud=predicted_point_cloud,
        #        keypoint_correction=keypoint_correction,
        #        certi=certified_outputs[depth_method_name][obj_label],
        #        theta=cfg["training"]["loss_theta"],
        #    )
        #    # only backpropagate the ones we are still training
        #    if per_object_train_flags[depth_method_name][obj_label]:
        #        loss.backward()

        #    # Gather data and report
        #    running_loss[depth_method_name][obj_label] += loss.item()
        #    trained_object_ids.add(obj_label)

        #    if i % 10 == 0:
        #        logging.info(
        #            f"{obj_label} - Batch {(i + 1)} loss: {loss.item()} pc loss: {pc_loss.item()} kp loss: {kp_loss.item()}"
        #        )
        #        logging.info(f"{obj_label} - Batch {(i + 1)} fra cert: {fra_cert.item()}")

        #    tensorboard_writer.add_scalar(
        #        tag=f"Loss/train/{depth_method_name}/{obj_label}",
        #        scalar_value=loss.item(),
        #        global_step=per_object_global_steps[depth_method_name][obj_label],
        #    )

        #    fra_certi_track[depth_method_name][obj_label].append(fra_cert)

        # Adjust learning weights
        optimizer.step()

        # free up gpu
        del inputs, outputs, certified_outputs

    # average training loss calculation
    avg_tlss = {
        depth_method_name: {
            k: (total_loss / (i + 1) if k in trained_object_ids else float("Inf"))
            for k, total_loss in running_loss[depth_method_name].items()
        }
    }

    return avg_tlss, fra_certi_track


def self_supervised_train_one_epoch_frame_objs(
    training_loader, mesh_db_batched, seg_model, model, cert_model, optimizer, device, cfg
):
    running_loss = 0.0
    fra_certi_track = []

    for i, data in enumerate(training_loader):
        optimizer.zero_grad()

        # data is a PoseData
        # prepare data and parameters
        batch_size, _, h, w = data.images.shape
        images = cast2cuda(data.images).float() / 255.0  # (B, 3, H, W)
        K = cast2cuda(data.K).float()  # (B, 3, 3)

        # perform segmentation
        if cfg["detector"]["det_type"] == "gt":
            # c3po inputs: ground truth masked out object point clouds
            object_batched_pcs = dict()
            for k, v in data.model_to_batched_pcs.items():
                if len(v) != 0:
                    object_batched_pcs[k] = cast2cuda(v).float()
            # cosypose detections: list of dictionaries, with the following keys:
            # - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
            #           ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            # - labels (Int64Tensor[N]): the predicted labels for each image
            # - scores (Tensor[N]): the scores or each prediction
            # - masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to
            #           obtain the final segmentation masks, the soft masks can be thresholded, generally
            #           with a value of 0.5 (mask >= 0.5)
            gt_seg_outputs = []
            for bid in range(batch_size):
                c_bboxes = [
                    cast2cuda(torch.as_tensor(data.objects[obj_idx]["bbox"])).float()
                    for obj_idx in data.frame_to_objects_index[bid]
                ]
                c_labels = [
                    torch.as_tensor(
                        cfg["detector"]["label_to_category_id"][data.objects[obj_idx]["name"]], dtype=torch.int64
                    )
                    for obj_idx in data.frame_to_objects_index[bid]
                ]
                c_scores = [torch.as_tensor(1.0, dtype=torch.float) for _ in data.frame_to_objects_index[bid]]
                c_masks = torch.stack(
                    [
                        torch.as_tensor(data.objects[obj_idx]["obj_mask"]).to(torch.uint8)
                        for obj_idx in data.frame_to_objects_index[bid]
                    ]
                ).unsqueeze(1)
                # c_pcs shape: (B, 3, N)
                c_pcs = torch.stack(
                    [
                        torch.as_tensor(data.objects[obj_idx]["point_cloud"]).to(torch.uint8)
                        for obj_idx in data.frame_to_objects_index[bid]
                    ]
                )
                gt_seg_outputs.append(
                    dict(boxes=c_bboxes, labels=c_labels, scores=c_scores, masks=c_masks, point_clouds=c_pcs)
                )

            cosypose_input_detections = build_cosypose_detections(
                img_height=h,
                img_width=w,
                seg_outputs=gt_seg_outputs,
                cfg=cfg,
                output_masks=True,
                output_pcs=True,
                detection_th=None,
                one_instance_per_class=False,
            )
        else:
            seg_model(images=images)
            # Need to: generate C3PO inputs by masking out points using segmentation masks
            # Pass the detections to cosypose
            raise NotImplementedError

        # forward pass on the MultiModel
        # make sure the order is consistent with the configuration yaml file's spec
        # c3po model inputs:
        # - input point cloud
        # cosypose model inputs:
        # - images, K, detections, TCO, n_iterations
        # NOTE: Check the input/model object point clouds' scale
        # make sure to normalize input for C3PO
        inputs = dict(
            c3po_multi=dict(object_batched_pcs=object_batched_pcs),
            cosypose_coarse_refine=dict(
                images=images,
                K=K,
                detections=cosypose_input_detections,
                TCO=None,  # if TCO=None, then cosypose will use a default canonical initial pose
                n_coarse_iterations=cfg["cosypose_coarse_refine"]["coarse"]["n_iterations"],
                n_refiner_iterations=cfg["cosypose_coarse_refine"]["refiner"]["n_iterations"],
            ),
        )

        # outputs format:
        # c3po: a dictionary with keys = object names, values = tuples containing:
        #       predicted_pc, corrected_kpts, R, t, correction, predicted_model_kpts
        # cosypose: a size-2 tuple,
        #        0: final output, a PandasTensorCollection
        #           with fields: -infos: containing pose information
        #        1: a dictionary, containing outputs at each iteration
        #           keys are: coarse/iteration=i,
        #                     refiner/iteration=i
        outputs = model(**inputs)

        manage_visualization(
            data=data, model=model, model_inputs=inputs, model_outputs=outputs, mesh_db=mesh_db_batched, cfg=cfg
        )

        # Certify each model's output
        # and return only the
        breakpoint()
        # for each object, grab the frame id
        certified_outputs = cert_model.certify(
            inputs,
            outputs,
            K=torch.stack([data.K[obj["frame_id_in_batch"], ...] for obj in data.objects]).to(device=device),
            resolution=data.images[0, ...].shape[-2:],
        )
        breakpoint()

        # Certification
        # certi = certify(
        #    input_point_cloud=input_point_cloud,
        #    predicted_point_cloud=predicted_point_cloud,
        #    corrected_keypoints=corrected_keypoints,
        #    predicted_model_keypoints=predicted_model_keypoints,
        #    epsilon=cfg["training"]["cert_epsilon"],
        # )
        # certi = certi.squeeze(-1)  # (B,)

        # Compute the loss and its gradients
        loss, pc_loss, kp_loss, fra_cert = self_supervised_loss(
            input_point_cloud=input_point_cloud,
            predicted_point_cloud=predicted_point_cloud,
            keypoint_correction=correction,
            certi=certi,
            theta=cfg["training"]["loss_theta"],
        )
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1 == 0:
            logging.info(f"Batch {(i + 1)} loss: {loss.item()} pc loss: {pc_loss.item()} kp loss: {kp_loss.item()}")
            logging.info(f"Batch {(i + 1)} fra cert: {fra_cert.item()}")

        fra_certi_track.append(fra_cert)

        del input_point_cloud, predicted_point_cloud, correction

    ave_tloss = running_loss / (i + 1)

    return ave_tloss, fra_certi_track


def validate_rgb_only(validation_loader, mesh_db_batched, model, cert_model, tensorboard_writer, device, cfg):
    """Validation function for the model

    Args:
        mesh_db_batched:
        cert_model:
        validation_loader:
        model:
        device:
        cfg:

    Returns:

    """
    depth_method_name = "c3po_multi"
    with torch.no_grad():
        running_vloss = dict(c3po_multi={k: 0.0 for k in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()})
        validated_object_ids = set()

        for i, vdata in enumerate(tqdm(validation_loader)):
            # get GT point clouds of objects
            object_batched_pcs = dict()
            for k, v in vdata.model_to_batched_pcs.items():
                if len(v) != 0:
                    object_batched_pcs[k] = cast2cuda(v).float()

            # construct input
            inputs = dict(
                c3po_multi=dict(object_batched_pcs=object_batched_pcs),
            )

            # outputs format:
            # c3po: a dictionary with keys = object names, values = tuples containing:
            #       predicted_pc, corrected_kpts, R, t, correction, predicted_model_kpts
            outputs = model(**inputs)

            # certification
            certified_outputs = cert_model.certify(
                inputs,
                outputs,
                K=torch.stack([vdata.K[obj["frame_id_in_batch"], ...] for obj in vdata.objects]).to(device=device),
                resolution=vdata.images[0, ...].shape[-2:],
            )

            # validation loss
            for obj_label in certified_outputs[depth_method_name].keys():
                predicted_point_cloud, _, _, _, _, _ = outputs[depth_method_name][obj_label]
                vloss = validation_loss(
                    inputs[depth_method_name]["object_batched_pcs"][obj_label],
                    predicted_point_cloud,
                    certi=certified_outputs[depth_method_name][obj_label],
                )

                running_vloss[depth_method_name][obj_label] += vloss
                validated_object_ids.add(obj_label)

            del inputs, outputs, certified_outputs

        avg_vloss = {
            depth_method_name: {
                k: (total_loss / (i + 1) if k in validated_object_ids else float("Inf"))
                for k, total_loss in running_vloss[depth_method_name].items()
            }
        }

    return avg_vloss


def train_without_supervision(
    self_supervised_train_loader=None,
    validation_loader=None,
    mesh_db_batched=None,
    seg_model=None,
    model: MultiModel = None,
    cert_model=None,
    optimizer=None,
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
    best_vloss = {
        k: {kk: 1_000_000.0 for kk in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()}
        for k in cfg["models_to_use"]
    }

    # these tracking meters are all per method / per object
    make_multi_tracking_meters = lambda: {
        model_name: {obj_label: TrackingMeter() for obj_label in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()}
        for model_name in cfg["models_to_use"]
    }
    train_loss = make_multi_tracking_meters()
    val_loss = make_multi_tracking_meters()
    certi_all_train_batches = make_multi_tracking_meters()
    epoch_number = 0

    # select the correct training functions
    if cfg["training"]["dataloader_type"] == "frame_objs" and len(cfg["models_to_use"]) > 1:
        raise NotImplementedError
    elif cfg["training"]["dataloader_type"] == "frame_objs" and cfg["models_to_use"][0] == "cosypose_coarse_refine":
        train_one_epoch = self_supervised_train_one_epoch_frame_objs_rgb_only
    else:
        raise NotImplementedError

    # select the correct validation function
    if cfg["training"]["dataloader_type"] == "frame_objs" and len(cfg["models_to_use"]) > 1:
        raise NotImplementedError
    elif cfg["training"]["dataloader_type"] == "frame_objs" and cfg["models_to_use"][0] == "cosypose_coarse_refine":
        validate = validate_rgb_only
    else:
        raise NotImplementedError

    # tracker for whether max cert has reached
    # set to false if the max certification threshold has been reached
    per_object_train_flags = dict(c3po_multi={k: True for k in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()})

    for epoch in range(num_epochs):
        logging.info(f"EPOCH : {epoch_number + 1} TIME: {datetime.now()}")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        logging.info("Training on real data with self-supervision: ")
        avg_loss_self_supervised, _fra_cert = train_one_epoch(
            training_loader=self_supervised_train_loader,
            mesh_db_batched=mesh_db_batched,
            seg_model=seg_model,
            model=model,
            cert_model=cert_model,
            optimizer=optimizer,
            per_object_train_flags=per_object_train_flags,
            tensorboard_writer=tensorboard_writer,
            device=device,
            cfg=cfg,
        )

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        logging.info("Run validation.")
        avg_vloss = validate(
            validation_loader,
            mesh_db_batched=mesh_db_batched,
            model=model,
            cert_model=cert_model,
            tensorboard_writer=tensorboard_writer,
            device=device,
            cfg=cfg,
        )

        logging.info(f"\nLOSS self-supervised train {avg_loss_self_supervised}")
        logging.info(f"\nLOSS valid (-%cert) {avg_vloss}")
        update_multi_tracking_meters(meters_to_be_updated=train_loss, update=avg_loss_self_supervised)
        update_multi_tracking_meters(meters_to_be_updated=val_loss, update=avg_vloss, fun=lambda x: -x)
        update_multi_tracking_meters(meters_to_be_updated=certi_all_train_batches, update=_fra_cert)

        #  update the vloss per object per method
        best_vloss = check_and_update_best_loss(best_vloss=best_vloss, avg_vloss=avg_vloss)

        # update tensorboard validation loss
        log_multimodel_vloss(avg_vloss=avg_vloss, epoch_number=epoch, tb_writer=tensorboard_writer)

        # Saving the model every epoch
        torch.save(model.state_dict(), os.path.join(model_save_folder, f"_epoch_{epoch_number}{best_model_save_fname}"))

        epoch_number += 1

        with open(os.path.join(model_save_folder, train_loss_save_fname), "wb") as outp:
            pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_save_folder, val_loss_save_fname), "wb") as outp:
            pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_save_folder, cert_save_fname), "wb") as outp:
            pickle.dump(certi_all_train_batches, outp, pickle.HIGHEST_PROTOCOL)

        per_object_train_flags = check_cert_against_threshold(
            avg_vloss=avg_vloss, old_flags=per_object_train_flags, cfg=cfg
        )

        # break if every model (per object) has finished training
        if all_training_stopped(per_object_train_flags):
            logging.info("All models have reached max certification thresholds.")
            break

    return train_loss, val_loss, certi_all_train_batches


def train_detector(cfg, detector_type="point_transformer", **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running self_supervised_training: {datetime.now()}")
    logging.info(f"device is {device}")

    # create dataset and dataloader
    # note: validation set dataloader has a subset random sampler
    ds_train, ds_val, ds_iter_train, ds_iter_val = None, None, None, None
    if cfg["training"]["dataloader_type"] == "rand_objs":
        ds_train, ds_val, ds_iter_train, ds_iter_val = load_rand_objs_datasets(cfg)
    elif cfg["training"]["dataloader_type"] == "frame_objs":
        ds_train, ds_val, ds_iter_train, ds_iter_val = load_frame_objs_datasets(cfg)

    object_ds, mesh_db_batched = load_objects(cfg)

    # load batch renderer
    batch_renderer = load_batch_renderer(cfg)

    # load multi model
    model = load_multi_model(batch_renderer=batch_renderer, meshdb_batched=mesh_db_batched, device=device, cfg=cfg)

    # load segmentation model
    seg_model = load_segmentation_model(device, cfg)

    # load certifier
    all_cad_models = load_all_cad_models(device=device, cfg=cfg)
    cert_model = load_certifier(all_cad_models, batch_renderer, cfg)

    # model save locations
    model_save_dir = cfg["save_folder"]
    best_model_save_fname = f"_best_self_supervised_kp_{detector_type}.pth"
    train_loss_save_fname = "_sstrain_loss.pkl"
    val_loss_save_fname = "_ssval_loss.pkl"
    cert_save_fname = "_certi_all_batches.pkl"

    # optimization parameters
    lr_sgd = cfg["optimizer"]["lr_sgd"]
    momentum_sgd = cfg["optimizer"]["momentum_sgd"]

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd)

    # tensorboard loss writer
    tb_writer = SummaryWriter(os.path.join(cfg["training"]["tb_log_dir"], cfg["timestamp"]))

    # training
    train_loss, val_loss, fra_cert_ = train_without_supervision(
        self_supervised_train_loader=ds_iter_train,
        validation_loader=ds_iter_val,
        mesh_db_batched=mesh_db_batched,
        seg_model=seg_model,
        model=model,
        cert_model=cert_model,
        optimizer=optimizer,
        model_save_folder=model_save_dir,
        best_model_save_fname=best_model_save_fname,
        train_loss_save_fname=train_loss_save_fname,
        val_loss_save_fname=val_loss_save_fname,
        cert_save_fname=cert_save_fname,
        tensorboard_writer=tb_writer,
        device=device,
        cfg=cfg,
    )

    return None
