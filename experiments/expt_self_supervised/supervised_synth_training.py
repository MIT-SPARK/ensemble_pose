"""
This code implements supervised and self-supervised training, and validation, for keypoint detector with registration.
It uses registration during supervised training. It uses registration plus corrector during self-supervised training.

"""

import numpy as np
import os
import pandas as pd
import pickle
import torch.utils.data
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import cosypose.utils.tensor_collection as tc
# experiment specific imports
from proposed_model import (
    MultiModel,
    load_multi_model,
    load_certifier,
    load_all_cad_models,
    load_batch_renderer,
)
from training_utils import *
from utils.loss_functions import bounded_avg_kpt_distance_loss
from utils.math_utils import depth_to_point_cloud_batched
from utils.torch_utils import cast2cuda


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


def build_cosypose_detections(
    img_height,
    img_width,
    seg_outputs,
    cfg,
    output_masks=False,
    output_pcs=False,
    detection_th=None,
    one_instance_per_class=False,
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
            pc = outputs_n["point_clouds"][obj_id, ...]
            bboxes.append(torch.as_tensor(bbox))
            masks.append(torch.as_tensor(mask))
            pcs.append(torch.as_tensor(pc))
            infos.append(info)

    if len(bboxes) > 0:
        bboxes = torch.stack(bboxes).cuda().float()
    else:
        infos = dict(score=[], label=[], batch_im_id=[])
        bboxes = torch.empty(0, 4).cuda().float()

    outputs = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        bboxes=bboxes,
    )

    if output_masks:
        # only move mask data to CUDA if we output it
        if len(bboxes) > 0:
            masks = torch.stack(masks).cuda()
        else:
            masks = torch.empty(0, img_height, img_width, dtype=torch.bool).cuda()
        outputs.register_tensor("masks", masks)

    if output_pcs:
        pcs = torch.stack(pcs).cuda().float()
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


def synth_supervised_train_one_epoch_frame_objs_depth_only(
    training_loader,
    mesh_db_batched,
    cad_models_db,
    seg_model,
    model,
    cert_model,
    optimizer,
    per_object_train_flags,
    tensorboard_writer,
    per_object_global_steps,
    device,
    cfg,
):
    depth_method_name = "c3po_multi"

    # storing losses and fraction certifiable for each method & objects
    running_loss = dict(c3po_multi={k: 0.0 for k in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()})
    fra_certi_track = dict(c3po_multi={k: [] for k in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()})
    trained_object_ids = set()
    kpt_reg_coef = cfg["training"]["kpt_distance_reg_coef"]

    for i, data in enumerate(tqdm(training_loader)):
        optimizer.zero_grad()

        # data is a PoseData
        # prepare data and parameters
        batch_size, _, h, w = data.images.shape

        if cfg["detector"]["det_type"] == "gt":
            # c3po inputs: ground truth masked out object point clouds
            object_batched_pcs = dict()
            for k, v in data.model_to_batched_pcs.items():
                if len(v) != 0:
                    object_batched_pcs[k] = cast2cuda(v).float()
        else:
            raise NotImplementedError

        # get ground truth object R & t
        objects_batched_gt_Rs = dict()
        for k, v in data.model_to_batched_gt_R.items():
            if len(v) != 0:
                objects_batched_gt_Rs[k] = cast2cuda(v).float()

        objects_batched_gt_ts = dict()
        for k, v in data.model_to_batched_gt_t.items():
            if len(v) != 0:
                objects_batched_gt_ts[k] = cast2cuda(v).float()

        # forward pass on the MultiModel
        # make sure the order is consistent with the configuration yaml file's spec
        # c3po model inputs:
        # - input point cloud
        # NOTE: Check the input/model object point clouds' scale
        # make sure to normalize input for C3PO
        inputs = dict(
            c3po_multi=dict(object_batched_pcs=object_batched_pcs),
        )

        # outputs format:
        # c3po: a dictionary with keys = object names, values = tuples containing:
        #       predicted_pc, corrected_kpts, R, t, correction, predicted_model_kpts
        outputs = model(**inputs)

        manage_visualization(
            data=data, model=model, model_inputs=inputs, model_outputs=outputs, mesh_db=mesh_db_batched, cfg=cfg
        )

        ## certification
        # certified_outputs = cert_model.certify(
        #    inputs,
        #    outputs,
        #    K=torch.stack([data.K[obj["frame_id_in_batch"], ...] for obj in data.objects]).to(device=device),
        #    resolution=data.images[0, ...].shape[-2:],
        # )

        # supervised loss
        for obj_label in outputs[depth_method_name].keys():
            per_object_global_steps[depth_method_name][obj_label] += 1

            # calculate the KP loss
            _, kp_pred, R, t, _, _ = outputs[depth_method_name][obj_label]
            model_keypoints = cad_models_db[obj_label]["original_model_keypoints"]
            R_gt = objects_batched_gt_Rs[obj_label]
            t_gt = torch.reshape(objects_batched_gt_ts[obj_label], (R_gt.shape[0], 3, 1))
            kp_gt = R_gt @ model_keypoints + t_gt

            # visualize
            if cfg["visualization"]["gt_keypoints"]:
                vutils.visualize_gt_and_pred_keypoints(
                    input_point_cloud=inputs[depth_method_name]["object_batched_pcs"][obj_label],
                    kp_gt=kp_gt,
                    kp_pred=kp_pred,
                )

            kp_loss = ((kp_gt - kp_pred) ** 2).sum(dim=1).mean(dim=1).mean()
            kp_dist_reg = kpt_reg_coef * bounded_avg_kpt_distance_loss(kp_pred)
            loss = kp_loss + kp_dist_reg
            loss.backward()

            running_loss[depth_method_name][obj_label] += loss.item()
            trained_object_ids.add(obj_label)

            tensorboard_writer.add_scalar(
                tag=f"Loss/train/{depth_method_name}/{obj_label}",
                scalar_value=loss.item(),
                global_step=per_object_global_steps[depth_method_name][obj_label],
            )

        # Adjust learning weights
        optimizer.step()

        # free up gpu
        del (
            inputs,
            outputs,
        )  # certified_outputs

    # average training loss calculation
    avg_tlss = {
        depth_method_name: {
            k: (total_loss / (i + 1) if k in trained_object_ids else float("Inf"))
            for k, total_loss in running_loss[depth_method_name].items()
        }
    }

    return avg_tlss, fra_certi_track


def validate_depth_only(
    validation_loader, mesh_db_batched, cad_models_db, model, cert_model, tensorboard_writer, device, cfg
):
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

            # get ground truth object R & t
            objects_batched_gt_Rs = dict()
            for k, v in vdata.model_to_batched_gt_R.items():
                if len(v) != 0:
                    objects_batched_gt_Rs[k] = cast2cuda(v).float()

            objects_batched_gt_ts = dict()
            for k, v in vdata.model_to_batched_gt_t.items():
                if len(v) != 0:
                    objects_batched_gt_ts[k] = cast2cuda(v).float()

            # construct input
            inputs = dict(
                c3po_multi=dict(object_batched_pcs=object_batched_pcs),
            )

            # outputs format:
            # c3po: a dictionary with keys = object names, values = tuples containing:
            #       predicted_pc, corrected_kpts, R, t, correction, predicted_model_kpts
            outputs = model(**inputs)

            ## certification
            # certified_outputs = cert_model.certify(
            #    inputs,
            #    outputs,
            #    K=torch.stack([vdata.K[obj["frame_id_in_batch"], ...] for obj in vdata.objects]).to(device=device),
            #    resolution=vdata.images[0, ...].shape[-2:],
            # )

            # validation loss
            for obj_label in outputs[depth_method_name].keys():
                _, kp_pred, R, t, _, _ = outputs["c3po_multi"][obj_label]
                model_keypoints = cad_models_db[obj_label]["original_model_keypoints"]
                R_gt = objects_batched_gt_Rs[obj_label]
                t_gt = torch.reshape(objects_batched_gt_ts[obj_label], (R_gt.shape[0], 3, 1))
                kp_gt = R_gt @ model_keypoints + t_gt

                if cfg["visualization"]["gt_keypoints"]:
                    vutils.visualize_gt_and_pred_keypoints(
                        input_point_cloud=inputs[depth_method_name]["object_batched_pcs"][obj_label],
                        kp_gt=kp_gt,
                        kp_pred=kp_pred,
                    )

                vloss = ((kp_gt - kp_pred) ** 2).sum(dim=1).mean(dim=1).mean()
                running_vloss[depth_method_name][obj_label] += vloss.item()
                validated_object_ids.add(obj_label)

                # saving predicted keypoints for visualization in tensorboard
                if i % 500 == 0:
                    kp_vis = torch.cat(
                        (
                            torch.transpose(kp_gt, -1, -2),
                            torch.transpose(kp_pred, -1, -2),
                        ),
                        dim=1,
                    )

                    kp_colors = torch.cat(
                        (
                            torch.tensor([0, 255, 0]).repeat(kp_pred.shape[0], kp_pred.shape[-1], 1),
                            torch.tensor([255, 0, 0]).repeat(kp_pred.shape[0], kp_pred.shape[-1], 1),
                        ),
                        dim=1,
                    )
                    point_size_config = {"material": {"cls": "PointsMaterial", "size": 0.2}}

                    tensorboard_writer.add_mesh(
                        tag=f"Kp_pred_gt/val/{depth_method_name}/{obj_label}",
                        vertices=kp_vis,
                        colors=kp_colors,
                        global_step=i,
                        config_dict={"material": point_size_config},
                    )

            del inputs, outputs

        avg_vloss = {
            depth_method_name: {
                k: (total_loss / (i + 1) if k in validated_object_ids else float("Inf"))
                for k, total_loss in running_vloss[depth_method_name].items()
            }
        }

    return avg_vloss


def train_with_synth_supervision(
    synth_supervised_train_loader=None,
    validation_loader=None,
    mesh_db_batched=None,
    cad_models_db=None,
    seg_model=None,
    model: MultiModel = None,
    cert_model=None,
    optimizer=None,
    scheduler=None,
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

    # select the correct training functions
    if cfg["training"]["dataloader_type"] == "frame_objs" and cfg["models_to_use"][0] == "c3po_multi":
        train_one_epoch = synth_supervised_train_one_epoch_frame_objs_depth_only
    else:
        raise NotImplementedError

    # select the correct validation function
    if cfg["training"]["dataloader_type"] == "frame_objs" and len(cfg["models_to_use"]) > 1:
        raise NotImplementedError
    elif cfg["training"]["dataloader_type"] == "frame_objs" and cfg["models_to_use"][0] == "c3po_multi":
        validate = validate_depth_only
    else:
        raise NotImplementedError

    # tracker for whether max cert has reached
    # set to false if the max certification threshold has been reached
    per_object_train_flags = dict(c3po_multi={k: True for k in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()})
    per_object_global_steps = dict(c3po_multi={k: 0 for k in bop_constants.BOP_MODEL_INDICES[cfg["dataset"]].keys()})

    for epoch in range(start_epoch, num_epochs):
        logging.info(f"EPOCH : {epoch + 1} TIME: {datetime.now()} LR: {scheduler.get_last_lr()}")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        logging.info("Training with supervision on synt. data: ")
        avg_loss_synth_supervised, _fra_cert = train_one_epoch(
            training_loader=synth_supervised_train_loader,
            mesh_db_batched=mesh_db_batched,
            cad_models_db=cad_models_db,
            seg_model=seg_model,
            model=model,
            cert_model=cert_model,
            optimizer=optimizer,
            per_object_train_flags=per_object_train_flags,
            tensorboard_writer=tensorboard_writer,
            per_object_global_steps=per_object_global_steps,
            device=device,
            cfg=cfg,
        )

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        logging.info("Run validation.")
        avg_vloss = validate(
            validation_loader,
            mesh_db_batched=mesh_db_batched,
            cad_models_db=cad_models_db,
            model=model,
            cert_model=cert_model,
            tensorboard_writer=tensorboard_writer,
            device=device,
            cfg=cfg,
        )

        logging.info(f"\nLOSS synth-supervised train {avg_loss_synth_supervised}")
        logging.info(f"\nLOSS valid {avg_vloss}")
        update_multi_tracking_meters(meters_to_be_updated=train_loss, update=avg_loss_synth_supervised)
        update_multi_tracking_meters(meters_to_be_updated=val_loss, update=avg_vloss, fun=lambda x: -x)
        update_multi_tracking_meters(meters_to_be_updated=certi_all_train_batches, update=_fra_cert)

        # update tensorboard validation loss
        log_multimodel_vloss(avg_vloss=avg_vloss, epoch_number=epoch, tb_writer=tensorboard_writer)

        #  update the vloss per object per method
        best_vloss = check_and_update_best_loss(best_vloss=best_vloss, avg_vloss=avg_vloss)

        # Saving the model every epoch
        torch.save(
            {
                "state_dict": model.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            },
            os.path.join(model_save_folder, f"_epoch_{epoch+1}{best_model_save_fname}"),
        )

        scheduler.step()

        with open(os.path.join(model_save_folder, train_loss_save_fname), "wb") as outp:
            pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_save_folder, val_loss_save_fname), "wb") as outp:
            pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_save_folder, cert_save_fname), "wb") as outp:
            pickle.dump(certi_all_train_batches, outp, pickle.HIGHEST_PROTOCOL)

    return train_loss, val_loss, certi_all_train_batches


def train_detector(cfg, detector_type="point_transformer", **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running synth_supervised_training: {datetime.now()}")
    logging.info(f"device is {device}")

    # create dataset and dataloader
    # note: validation set dataloader has a subset random sampler
    ds_train, ds_val, ds_iter_train, ds_iter_val = None, None, None, None
    if cfg["training"]["dataloader_type"] == "rand_objs":
        ds_train, ds_val, ds_iter_train, ds_iter_val = load_rand_objs_datasets(cfg)
    elif cfg["training"]["dataloader_type"] == "frame_objs":
        ds_train, ds_val, ds_iter_train, ds_iter_val = load_frame_objs_datasets(cfg)
    elif cfg["training"]["dataloader_type"] == "single_obj":
        raise NotImplementedError

    object_ds, mesh_db_batched = load_objects(cfg)

    # load batch renderer
    batch_renderer = load_batch_renderer(cfg)

    # load scene renderer
    # (not used for supervised training)
    scene_renderer = []

    # load multi model
    model = None
    assert len(cfg["models_to_use"]) == 1
    if cfg["models_to_use"][0] == 'c3po_multi':
        model = load_multi_model(batch_renderer=batch_renderer, meshdb_batched=mesh_db_batched, device=device, cfg=cfg)
    elif cfg["models_to_use"][0] == "c3po":
        raise NotImplementedError

    # load segmentation model
    seg_model = None

    # load certifier
    all_cad_models = load_all_cad_models(device=device, cfg=cfg)
    cert_model = load_certifier(all_cad_models, scene_renderer, cfg)

    # model save locations
    model_save_dir = cfg["save_folder"]
    best_model_save_fname = f"_synth_supervised_kp_{detector_type}.pth.tar"
    train_loss_save_fname = "_sstrain_loss.pkl"
    val_loss_save_fname = "_ssval_loss.pkl"
    cert_save_fname = "_certi_all_batches.pkl"

    # optimizer
    lr_sgd = cfg["optimizer"]["lr_sgd"]
    momentum_sgd = cfg["optimizer"]["momentum_sgd"]
    weight_decay_sgd = cfg["optimizer"]["weight_decay_sgd"]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd, weight_decay=weight_decay_sgd)

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["step_lr_scheduler"]["lr_epoch_decay"],
        gamma=cfg["step_lr_scheduler"]["gamma"],
    )

    if kwargs["resume_run"]:
        ckpt_path = kwargs["multimodel_checkpoint_path"]
        logging.info(f"Loading checkpoint from {ckpt_path}")
        save = torch.load(ckpt_path)

        model.load_state_dict(save["state_dict"])
        lr_scheduler.load_state_dict(save["scheduler_state_dict"])
        optimizer.load_state_dict(save["optimizer_state_dict"])

        start_epoch = save["epoch"] + 1
    else:
        start_epoch = 0

    # tensorboard loss writer
    tb_writer = SummaryWriter(os.path.join(cfg["training"]["tb_log_dir"], cfg["dataset"], cfg["timestamp"]))

    # training
    train_loss, val_loss, fra_cert_ = train_with_synth_supervision(
        synth_supervised_train_loader=ds_iter_train,
        validation_loader=ds_iter_val,
        mesh_db_batched=mesh_db_batched,
        cad_models_db=all_cad_models,
        seg_model=seg_model,
        model=model,
        cert_model=cert_model,
        scheduler=lr_scheduler,
        start_epoch=start_epoch,
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
