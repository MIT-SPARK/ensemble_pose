import copy
import jinja2
import logging
import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import yaml
from pathlib import Path

import datasets.bop_constants as bop_constants
from expt_self_supervised.cfg import CosyposeCfg
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.utils import tensor_collection as tc
from datasets import bop_constants
from datasets.bop import make_scene_dataset, make_object_dataset
from datasets.pose import PoseDataset, FramePoseDataset, ObjectPCPoseDataset, ObjectPoseDataset
from datasets.bop_detections import BOPDetections
from datasets.samplers import PartialSampler
from datasets.utils import MultiEpochDataLoader
from utils import visualization_utils as vutils
from utils.general import TrackingMeter
from utils.math_utils import depth_to_point_cloud_batched


def make_multi_tracking_meters(cfg):
    """Make a multi model tracking meter dictionary"""
    result = {}
    for model_name in cfg["models_to_use"]:
        if model_name == "c3po_multi":
            result[model_name] = {
                obj_label: TrackingMeter() for obj_label in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()
            }
        else:
            raise NotImplementedError
    return result


def update_multi_tracking_meters(meters_to_be_updated, update, fun=lambda x: x):
    """Update a multi model tracking meter"""
    for model_name, v in update.items():
        if isinstance(v, dict):
            for obj_name, val in v.items():
                meters_to_be_updated[model_name][obj_name].add_item(fun(val))
        else:
            raise NotImplementedError
    return meters_to_be_updated


def check_and_update_best_loss(best_vloss, avg_vloss):
    """Check multi loss and update the best loss if lower"""
    new_best_vloss = copy.deepcopy(best_vloss)

    for model_name, best_model_objs_vlosses in best_vloss.items():
        if isinstance(best_model_objs_vlosses, dict):
            for obj_name, best_obj_vloss in best_model_objs_vlosses.items():
                if avg_vloss[model_name][obj_name] < best_obj_vloss:
                    new_best_vloss[model_name][obj_name] = avg_vloss[model_name][obj_name]
        else:
            raise NotImplementedError

    return new_best_vloss


def check_cert_against_threshold(avg_vloss, old_flags, cfg):
    """Check per object certification fractions against thresholds

    Args:
        avg_vloss:
        old_flags: A dictionary with the same internal structure as avg vloss. Each True entry means the particular model
        is still undergoing self-supervised training. False indicates it has already stopped.
        cfg:
    """
    new_flags = copy.deepcopy(old_flags)

    for k, v in avg_vloss.items():
        if isinstance(v, dict):
            for obj_name, vloss in v.items():
                frac_cert = -vloss
                if not old_flags[k][obj_name]:
                    # training has already stopped for this instance
                    continue
                else:
                    if frac_cert > cfg["training"]["train_stop_cert_threshold"]:
                        logging.info(f"ENDING TRAINING ({k}-{obj_name}). REACHED MAX. CERTIFICATION (AT VALIDATION).")
                        new_flags[k][obj_name] = False

    return new_flags


def all_training_stopped(tflags):
    """Return true if all training flags are false

    Args:
        tflags: dictionary; element True if the specific object model is still training
    """
    all_stopped = True
    for k, v in tflags.items():
        if isinstance(v, dict):
            for obj_name, flag in v.items():
                if flag:
                    all_stopped = False
                    break
    return all_stopped


def log_multimodel_vloss(avg_vloss, epoch_number, tb_writer):
    """Logging validation loss from MultiModel for one epoch"""
    methods = avg_vloss.keys()
    for method in methods:
        if isinstance(avg_vloss[method], dict):
            obj_labels = avg_vloss[method].keys()
            for obj_label in obj_labels:
                vloss = avg_vloss[method][obj_label]
                tb_writer.add_scalar(tag=f"Loss/val/{method}/{obj_label}", scalar_value=vloss, global_step=epoch_number)
        else:
            vloss = avg_vloss[method]
            tb_writer.add_scalar(tag=f"Loss/val/{method}", scalar_value=vloss, global_step=epoch_number)


def load_yaml_cfg(config_params_file, object_id):
    stream = open(config_params_file, "r")
    template = jinja2.Template(stream.read().rstrip())
    if object_id is not None:
        processed_yaml = template.render(
            project_root=Path(__file__).parent.parent.parent.resolve(),
            exp_root=Path(__file__).parent.resolve(),
            object_id=object_id,
        )
    else:
        processed_yaml = template.render(
            project_root=Path(__file__).parent.parent.parent.resolve(),
            exp_root=Path(__file__).parent.resolve(),
        )
    cfg = yaml.full_load(processed_yaml)

    if "certifier" in cfg.keys():
        if "objects_thresholds_path" in cfg["certifier"]:
            logging.info(f"Loading objects' specific thresholds from {cfg['certifier']['objects_thresholds_path']}.")
            with open(cfg["certifier"]["objects_thresholds_path"], 'r') as stream:
                thresholds = yaml.full_load(stream)
            for k, v in thresholds["certifier"].items():
                cfg["certifier"][k] = v
    return cfg


def manage_visualization(
    data, detector_outputs=None, model=None, model_inputs=None, model_outputs=None, mesh_db=None, cfg=None
):
    """Handle visualization logic"""
    B = data.depths.shape[0]

    if cfg["visualization"]["gt_rgb_segmentation"]:
        for b in range(B):
            bboxes_in_frame = torch.as_tensor([data.objects[i]["bbox"] for i in data.frame_to_objects_index[b]])
            masks_in_frame = torch.as_tensor([data.objects[i]["obj_mask"] for i in data.frame_to_objects_index[b]])
            logging.info("Drawing GT RGB masks and bounding boxes.")
            images = data.images
            vutils.visualize_bop_rgb_obj_masks(
                rgb=images[b, ...], bboxes=bboxes_in_frame, masks=masks_in_frame, show=True
            )

    if cfg["visualization"]["gt_pc_segmentation"]:
        for b in range(B):
            objs_in_frame = [data.objects[i] for i in data.frame_to_objects_index[b]]

            logging.info("Drawing GT masked point clouds.")
            scene_pcs = depth_to_point_cloud_batched(
                data.depths,
                data.K,
                x_index=2,
                y_index=1,
                mask=None,
                pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
                device="cpu",
            )

            logging.info(f"Visualizing frame ID = {b} out of {B} frames in batch.")
            logging.info(f"Total # of objects in frame: {len(objs_in_frame)}")
            vutils.visualize_bop_obj_point_clouds_in_frame(
                scene_pc=scene_pcs[b, ...], objs_in_frame=objs_in_frame, mesh_db=mesh_db
            )

    if cfg["visualization"]["detector_outputs"]:
        raise NotImplementedError

    if cfg["visualization"]["model_inputs"]:
        logging.info("Drawing detector results")
        # plot inputs to c3po
        c3po_inputs = model_inputs["c3po_multi"]["object_batched_pcs"]
        for obj_name, obj_batched_pcs in c3po_inputs.items():
            logging.info(f"Plotting input to C3PO-{obj_name}.")
            vutils.visualize_pcs(pcs=[obj_batched_pcs[i, ...] for i in range(obj_batched_pcs.shape[0])])

        # plot inputs to cosypose
        try:
            cosypose_inputs = model_inputs["cosypose_coarse_refine"]
            cosypose_input_detections = cosypose_inputs["detections"]
            # cosypose input detections have two fields: info, and bboxes.
            # each entry corresponds to one bbox detection in a particular frame
            vutils.visualize_cosypose_input_detections(data.images, cosypose_input_detections, show=True)
        except KeyError:
            logging.info("Skipping CosyPose model in model inputs visualization.")

    if cfg["visualization"]["c3po_outputs"]:
        logging.info("Drawing C3PO outputs")
        # draw the object CAD mesh transformed by the C3PO estimated transformation into camera frame
        for obj_name, obj_output in model_outputs["c3po_multi"].items():
            logging.info(f"Visualizing C3PO output for {obj_name}.")
            input_point_cloud = model_inputs["c3po_multi"]["object_batched_pcs"][obj_name]
            model_keypoints = model.pipeline_modules["c3po_multi"].obj_reg_modules[obj_name].model_keypoints
            vutils.visualize_c3po_outputs(input_point_cloud, obj_output, model_keypoints)

    if cfg["visualization"]["cosypose_outputs"]:
        logging.info("Drawing CosyPose outputs")
        renderer = model.pipeline_modules["cosypose_coarse_refine"].vis_renderer
        # cosypose_outputs is a tuple of size 2
        # 0: result at the final iteration
        # 1: all iterations' results
        cosypose_outputs = model_outputs["cosypose_coarse_refine"]
        vutils.visualize_cosypose_output(
            rgbs=data.images, preds=cosypose_outputs[0], K=data.K, renderer=renderer, show=True
        )

    return


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
        load_rgb_for_points=cfg_training["load_rgb_for_points"],
    )
    ds_train = FramePoseDataset(scene_ds_train, **ds_kwargs)
    ds_val = FramePoseDataset(scene_ds_val, **ds_kwargs)

    train_sampler = PartialSampler(
        ds_train, epoch_size=cfg_training["epoch_size"], randomize=cfg_training["shuffle_train"]
    )
    ds_iter_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        collate_fn=ds_train.collate_fn,
        sampler=train_sampler,
        drop_last=False,
        pin_memory=True,
    )

    # sub-sample validation set according to params provided
    # equally spaced sampling
    val_sampler = PartialSampler(ds_val, epoch_size=cfg_training["val_ds_size"], randomize=cfg_training["shuffle_val"])
    ds_iter_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        collate_fn=ds_val.collate_fn,
        sampler=val_sampler,
        drop_last=False,
        pin_memory=True,
    )

    return ds_train, ds_val, ds_iter_train, ds_iter_val


def load_single_obj_pc_img_dataset(object_id, obj_diameter, cfg, collate_fn=None):
    """Load data loaders that return single object only"""
    cfg_training = cfg["training"]
    logging.info(f"Loading train={cfg['train_ds_name']}, val={cfg['val_ds_name']}.")

    scene_ds_train = make_scene_dataset(cfg["train_ds_name"], bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)
    scene_ds_val = make_scene_dataset(cfg["val_ds_name"], bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)

    if cfg["detector"]["det_type"] == "bop_default":
        logging.info("Loading BOP default segmentations (MaskRCNN).")
        bop_detections = BOPDetections(detections_path=cfg["detector"]["default_det_path"])
    elif cfg["detector"]["det_type"] == "gt":
        logging.info("Loading BOP GT segmentations.")
        bop_detections = None
    else:
        raise ValueError(f"Unknown detection type: {cfg['detector']['det_type']}.")

    if cfg["dataset"] == "ycbv":
        all_model_names = list(bop_constants.YCBV.keys())
    elif cfg["dataset"] == "tless":
        all_model_names = list(bop_constants.TLESS.keys())
    else:
        raise NotImplementedError
    ds_kwargs = dict(
        object_diameter=obj_diameter,
        dataset_name=cfg["dataset"],
        min_area=cfg_training["min_area"],
        pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
        load_rgb_for_points=cfg_training["load_rgb_for_points"],
        zero_center_pc=cfg_training["zero_center_pc"],
        use_robust_centroid=cfg_training["use_robust_centroid"],
        resample_invalid_pts=cfg_training["resample_invalid_pts"],
        normalize_pc=cfg_training["normalize_pc"],
        load_data_from_cache=cfg_training["load_single_obj_data_from_cache"],
        cache_save_dir=cfg_training["single_obj_data_cache_dir"],
        preload_to_ram=cfg_training["preload_to_ram"],
        bop_detections=bop_detections,
    )

    # load the single obj dataset
    if cfg["train_ds_name"] != cfg["val_ds_name"]:
        ds_train = ObjectPoseDataset(scene_ds_train, object_id, **ds_kwargs)
        ds_val = ObjectPoseDataset(scene_ds_val, object_id, **ds_kwargs)
        train_size = cfg_training["epoch_size"]
        if cfg_training["epoch_size"] == -1:
            train_size = int(len(ds_train))
        val_size = int(cfg_training["val_ds_frac_size"] * train_size)
    else:
        # random split
        temp_set = ObjectPoseDataset(scene_ds_train, object_id, **ds_kwargs)
        if cfg_training["allow_val_train_overlap"]:
            train_size = cfg_training["epoch_size"]
            if cfg_training["epoch_size"] < 0:
                train_size = int(len(temp_set))
            val_size = int(cfg_training["val_ds_frac_size"] * train_size)
            ds_train = temp_set
            ds_val = temp_set
        else:
            if cfg_training["epoch_size"] == -1:
                train_size = int(len(temp_set) * (1 - cfg_training["val_ds_frac_size"]))
                val_size = int(len(temp_set) - train_size)
                ds_train, ds_val = torch.utils.data.random_split(
                    temp_set,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(0),
                )
            else:
                ds_train, ds_val = torch.utils.data.random_split(
                    temp_set,
                    [cfg_training["epoch_size"], cfg_training["val_ds_frac_size"] * cfg_training["epoch_size"]],
                    generator=torch.Generator().manual_seed(0),
                )
            train_size, val_size = len(ds_train), len(ds_val)

    logging.info(f"Train set size={train_size}, val set size={val_size}.")
    train_sampler = PartialSampler(ds_train, epoch_size=train_size, randomize=cfg_training["shuffle_train"])
    ds_iter_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        collate_fn=collate_fn,
        sampler=train_sampler,
        drop_last=False,
    )

    val_sampler = PartialSampler(ds_val, epoch_size=val_size, randomize=cfg_training["shuffle_val"])
    ds_iter_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        collate_fn=collate_fn,
        sampler=val_sampler,
        drop_last=False,
    )

    return ds_train, ds_val, ds_iter_train, ds_iter_val


def load_single_obj_pc_dataset(object_id, obj_diameter, cfg):
    """Load data loaders that return single object only"""
    cfg_training = cfg["training"]
    logging.info(f"Loading train={cfg['train_ds_name']}, val={cfg['val_ds_name']}.")

    scene_ds_train = make_scene_dataset(cfg["train_ds_name"], bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)
    scene_ds_val = make_scene_dataset(cfg["val_ds_name"], bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)

    if cfg["dataset"] == "ycbv":
        all_model_names = list(bop_constants.YCBV.keys())
    elif cfg["dataset"] == "tless":
        all_model_names = list(bop_constants.TLESS.keys())
    else:
        raise NotImplementedError
    ds_kwargs = dict(
        object_diameter=obj_diameter,
        dataset_name=cfg["dataset"],
        min_area=cfg_training["min_area"],
        pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
        load_rgb_for_points=cfg_training["load_rgb_for_points"],
        zero_center_pc=cfg_training["zero_center_pc"],
        use_robust_centroid=cfg_training["use_robust_centroid"],
        resample_invalid_pts=cfg_training["resample_invalid_pts"],
        normalize_pc=cfg_training["normalize_pc"],
        load_data_from_cache=cfg_training["load_single_obj_data_from_cache"],
        cache_save_dir=cfg_training["single_obj_data_cache_dir"],
    )

    # load the single obj dataset
    if cfg["train_ds_name"] != cfg["val_ds_name"]:
        ds_train = ObjectPCPoseDataset(scene_ds_train, object_id, **ds_kwargs)
        ds_val = ObjectPCPoseDataset(scene_ds_val, object_id, **ds_kwargs)
        train_size = cfg_training["epoch_size"]
        if cfg_training["epoch_size"] == -1:
            train_size = int(len(ds_train))
        val_size = int(cfg_training["val_ds_frac_size"] * train_size)
    else:
        # random split
        temp_set = ObjectPCPoseDataset(scene_ds_train, object_id, **ds_kwargs)
        if cfg_training["epoch_size"] == -1:
            train_size = int(len(temp_set) * (1 - cfg_training["val_ds_frac_size"]))
            val_size = int(len(temp_set) - train_size)
            ds_train, ds_val = torch.utils.data.random_split(
                temp_set,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(0),
            )
        else:
            ds_train, ds_val = torch.utils.data.random_split(
                temp_set,
                [cfg_training["epoch_size"], cfg_training["val_ds_frac_size"] * cfg_training["epoch_size"]],
                generator=torch.Generator().manual_seed(0),
            )
        train_size, val_size = len(ds_train), len(ds_val)

    logging.info(f"Train set size={train_size}, val set size={val_size}.")
    train_sampler = PartialSampler(ds_train, epoch_size=train_size, randomize=cfg_training["shuffle_train"])
    ds_iter_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        sampler=train_sampler,
        drop_last=False,
    )

    val_sampler = PartialSampler(ds_val, epoch_size=val_size, randomize=cfg_training["shuffle_val"])
    ds_iter_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg_training["batch_size"],
        num_workers=cfg_training["n_dataloader_workers"],
        sampler=val_sampler,
        drop_last=False,
    )

    return ds_train, ds_val, ds_iter_train, ds_iter_val


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


def build_cosypose_detections(
    img_height,
    img_width,
    seg_outputs,
    cfg,
    output_masks=False,
    output_pcs=False,
    detection_th=None,
    one_instance_per_class=False,
    device="cpu",
):
    """Build the appropriate detections data structure for Cosypose Coarse + Refine to run

    Args:
        outputs: outputs from a torchvision MaskRCNN model
        cfg:
    """
    # these lists will store the flattened version of objects
    # in other words, if we have 3 objs in frame 1, 2 objs in frame 2,
    # these lists will be of lengths 5 (3+2)
    infos = []
    bboxes = []
    masks = []
    pcs = []
    # seg_outputs is in terms of frame in the batch
    for n, outputs_n in enumerate(seg_outputs):
        outputs_n["labels"] = [
            cfg["detector"]["category_id_to_label"][category_id.item()] for category_id in outputs_n["labels"]
        ]

        for obj_id in range(len(outputs_n["boxes"])):
            bbox = outputs_n["boxes"][obj_id]
            info = dict(
                batch_im_id=n,
                label=outputs_n["labels"][obj_id],
                score=outputs_n["scores"][obj_id].item(),
            )
            mask = outputs_n["masks"][obj_id, 0] > cfg["detector"]["mask_th"]
            bboxes.append(torch.as_tensor(bbox))
            masks.append(torch.as_tensor(mask))
            if output_pcs:
                pc = outputs_n["point_clouds"][obj_id, ...]
                pcs.append(torch.as_tensor(pc))
            infos.append(info)

    if len(bboxes) > 0:
        bboxes = torch.stack(bboxes).float().to(device)
    else:
        infos = dict(score=[], label=[], batch_im_id=[])
        bboxes = torch.empty(0, 4).float().to(device)

    outputs = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        bboxes=bboxes,
    )

    if output_masks:
        # only move mask data to CUDA if we output it
        if len(bboxes) > 0:
            masks = torch.stack(masks).to(device)
        else:
            masks = torch.empty(0, img_height, img_width, dtype=torch.bool).to(device)
        outputs.register_tensor("masks", masks)

    if output_pcs:
        pcs = torch.stack(pcs).float().to(device)
        outputs.register_tensor("point_clouds", pcs)

    if detection_th is not None:
        keep = np.where(outputs.infos["score"] > detection_th)[0]
        outputs = outputs[keep]

    if one_instance_per_class:
        infos = outputs.infos
        infos["det_idx"] = np.arange(len(infos))
        keep_ids = infos.sort_values("score", ascending=False).drop_duplicates("label")["det_idx"].values
        outputs = outputs[keep_ids]
        outputs.infos = outputs.infos.drop("det_idx", axis=1)
    return outputs


def make_sgd_optimizer_scheduler(model, cfg):
    lr_sgd = cfg["optimizer"]["lr_sgd"]
    momentum_sgd = cfg["optimizer"]["momentum_sgd"]
    weight_decay_sgd = cfg["optimizer"]["weight_decay_sgd"]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd, weight_decay=weight_decay_sgd)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["step_lr_scheduler"]["lr_epoch_decay"],
        gamma=cfg["step_lr_scheduler"]["gamma"],
    )
    return optimizer, lr_scheduler


def multi_step(x):
    """Helper function that calls step() on all items in a dictionary"""
    for _, v in x.items():
        v.step()


def multi_zero_grad(optimizers):
    """Helper function that calls zero_grad() on all items in a dictionary"""
    for _, opt in optimizers.items():
        opt.zero_grad()
