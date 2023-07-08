"""
This code implements supervised training of keypoint detector in simulation.

It can use registration during supervised training.

"""

import logging
import os
import pickle
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from datasets import ycbv, tless
from proposed_model import PointsRegressionModel as ProposedModel
from proposed_model import load_c3po_model, load_c3po_cad_models, load_all_cad_models
from utils.general import TrackingMeter
from utils.loss_functions import bounded_avg_kpt_distance_loss
import utils.visualization_utils as vutils
import utils.math_utils as mutils
from utils.torch_utils import get_grad_norm
from utils.visualization_utils import display_results

from training_utils import *


def train_one_epoch(
    training_loader,
    object_id,
    mesh_db_batched,
    cad_models_db,
    model,
    optimizer,
    current_epoch_num,
    tensorboard_writer,
    visualize=False,
    device=None,
    cfg=None,
):
    running_loss = 0.0
    kpt_reg_coef = cfg["training"]["kpt_distance_reg_coef"]
    kpt_reg_eps_bound = cfg["training"]["kpt_distance_reg_eps_bound"] * model.min_intra_kpt_dist / model.object_diameter
    max_grad_norm = cfg["training"]["max_grad_norm"]
    epoch_size = len(training_loader)

    for i, data in enumerate(training_loader):
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Every data instance is an input + label pair
        pc, kp, R, t = data

        pc = pc.to(device)
        kp = kp.to(device)
        R = R.to(device)
        t = t.to(device)

        # print("Test:: pc_shape: ", pc.shape)
        # print(pc[0, ...])
        out = model(object_id, pc)

        if cfg["training"]["normalize_pc"]:
            kp_gt = kp
        else:
            kp_gt = kp / model.object_diameter
        kp_pred = out[1] / model.object_diameter

        # note: loss calculated in the normalized frame
        # kp_dist_reg = - kpt_reg_coef * avg_kpt_distance_regularizer(kp_pred)
        kp_loss = ((kp_gt - kp_pred) ** 2).sum(dim=1).mean(dim=1).mean()
        kp_dist_reg = kpt_reg_coef * bounded_avg_kpt_distance_loss(kp_pred, eps_bound=kpt_reg_eps_bound)
        loss = kp_loss + kp_dist_reg
        loss.backward()

        if visualize:
            logging.info("Visualizing predictions and keypoint annotations.")
            # visualize: input point cloud, ground truth keypoint annotations, predicted keypoints
            vutils.visualize_gt_and_pred_keypoints(pc[:, :3, :], kp_gt, kp_pred=kp_pred)

        tensorboard_writer.add_scalar(
            tag=f"GradNorm/train/c3po/{object_id}",
            scalar_value=get_grad_norm(model.parameters()),
            global_step=current_epoch_num * epoch_size + i,
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()  # Note: the output of supervised_loss is already averaged over batch_size
        if i % 10 == 0:
            print("Batch ", (i + 1), " loss: ", loss.item(), " kp dist. regular. term: ", kp_dist_reg.item())

        tensorboard_writer.add_scalar(
            tag=f"Loss/train/c3po/{object_id}",
            scalar_value=loss.item(),
            global_step=current_epoch_num * epoch_size + i,
        )

    del pc, kp, R, t, kp_pred, kp_gt, loss
    torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)

    return ave_tloss


# Validation code
def validate(validation_loader, object_id, model, device, tensorboard_writer, visualize=False, cfg=None):
    # We don't need gradients on to do reporting.
    with torch.no_grad():

        running_vloss = 0.0
        running_max_intra_kp_dist = 0.0
        running_min_intra_kp_dist = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, kp, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            kp = kp.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)

            # Make predictions for this batch
            out = model(object_id, input_point_cloud)
            if cfg["training"]["normalize_pc"]:
                kp_gt = kp
            else:
                kp_gt = kp / model.object_diameter
            kp_pred = out[1] / model.object_diameter

            vloss = ((kp_gt - kp_pred) ** 2).sum(dim=1).mean(dim=1).mean()

            running_vloss += vloss.item()

            max_intra_kpt_dist, min_intra_kpt_dist = mutils.get_max_min_intra_pts_dists(kp_pred)
            running_max_intra_kp_dist += max_intra_kpt_dist
            running_min_intra_kp_dist += min_intra_kpt_dist

            # visualize
            if visualize:
                logging.info("Visualizing validation predictions and keypoint annotations.")
                vutils.visualize_gt_and_pred_keypoints(input_point_cloud, kp, kp_pred=kp_pred)

        del (
            vloss,
            input_point_cloud,
            kp,
            kp_gt,
            kp_pred,
            R_target,
            t_target,
        )
        torch.cuda.empty_cache()

        avg_vloss = running_vloss / (i + 1)
        max_avg_intra_kp_dist = running_max_intra_kp_dist / (i + 1)
        min_avg_intra_kp_dist = running_min_intra_kp_dist / (i + 1)
        logging.info(f"Validation avg. intra kp dist max: {max_avg_intra_kp_dist}, min: {min_avg_intra_kp_dist}")

    return avg_vloss


def train_with_synth_supervision(
    synth_supervised_train_loader=None,
    validation_loader=None,
    mesh_db_batched=None,
    cad_models_db=None,
    model=None,
    object_id=None,
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

    train_loss = TrackingMeter()
    val_loss = TrackingMeter()

    for epoch in range(start_epoch, num_epochs):
        logging.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logging.info(f"Current LR: {scheduler.get_last_lr()}")

        # training
        model.train(True)
        logging.info("Training with supervision on synt. data (single object): ")
        avg_loss_synth_supervised = train_one_epoch(
            training_loader=synth_supervised_train_loader,
            model=model,
            object_id=object_id,
            optimizer=optimizer,
            current_epoch_num=epoch,
            mesh_db_batched=mesh_db_batched,
            cad_models_db=cad_models_db,
            tensorboard_writer=tensorboard_writer,
            visualize=cfg["visualization"]["c3po_outputs"],
            device=device,
            cfg=cfg,
        )
        train_loss.add_item(avg_loss_synth_supervised)

        # validation
        model.train(False)
        logging.info("Run validation.")
        avg_vloss = validate(
            validation_loader=validation_loader,
            object_id=object_id,
            model=model,
            tensorboard_writer=tensorboard_writer,
            visualize=cfg["visualization"]["c3po_outputs"],
            device=device,
            cfg=cfg,
        )
        val_loss.add_item(avg_vloss)

        logging.info(f"\nLOSS synth-supervised train {avg_loss_synth_supervised}")
        logging.info(f"\nLOSS valid {avg_vloss}")

        # update tensorboard for validation
        tensorboard_writer.add_scalar(tag=f"Loss/val/c3po/{object_id}", scalar_value=avg_vloss, global_step=epoch)

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

    return train_loss, val_loss


def train_detector(cfg, detector_type="pointnet", model_id="obj_000001", use_corrector=False, **kwargs):
    """Main training function"""
    dataset = cfg["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running supervised_training: {datetime.now()}")
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
    if cfg["training"]["dataloader_type"] == "single_obj":
        ds_train, ds_val, ds_iter_train, ds_iter_val = load_single_obj_pc_dataset(
            object_id=model_id, obj_diameter=obj_diameter, cfg=cfg
        )

    # load model
    model = load_c3po_model(
        model_id=model_id,
        cad_models=original_cad_model,
        model_keypoints=original_model_keypoints,
        object_diameter=obj_diameter,
        device=device,
        cfg=cfg,
    )

    # model save locations
    model_save_dir = cfg["save_folder"]
    best_model_save_fname = f"_synth_supervised_single_obj_kp_{detector_type}.pth.tar"
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

    # log hparams
    tb_writer.add_hparams(
        hparam_dict={
            "robust_centroid": cfg["training"]["use_robust_centroid"],
            "point_transformer_sampling": cfg["c3po"]["point_transformer"]["sampling_type"],
        },
        metric_dict={},
        # hack to have hparams in the same tf event file as scalar
        # https://github.com/pytorch/pytorch/issues/32651
        run_name=os.path.join(cfg["training"]["tb_log_dir"], cfg["dataset"], cfg["timestamp"])
    )

    # training
    train_loss, val_loss = train_with_synth_supervision(
        synth_supervised_train_loader=ds_iter_train,
        validation_loader=ds_iter_val,
        mesh_db_batched=mesh_db_batched,
        model=model,
        object_id=model_id,
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
