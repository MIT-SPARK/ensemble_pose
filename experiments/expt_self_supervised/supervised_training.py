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
from proposed_model import PointsRegressionModel
from proposed_model import load_c3po_model, load_c3po_cad_models, load_all_cad_models
from utils.general import TrackingMeter
from utils.loss_functions import bounded_avg_kpt_distance_loss
from utils.visualization_utils import display_results


# Training code
def supervised_train_one_epoch(training_loader, model, object_id, optimizer, device, cfg):
    running_loss = 0.0
    kpt_reg_coef = cfg["training"]["kpt_distance_reg_coef"]

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        pc, kp, R, t = data

        if "gen_outliers" in cfg["training"].keys():
            if cfg["training"]["gen_outliers"]:
                num_outliers_to_gen = int(cfg["training"]["outlier_ratio"] * pc.shape[-1])
                outlier_indices = torch.randperm(pc.shape[-1])[:num_outliers_to_gen]
                pc[:3, outlier_indices] = (
                    torch.rand((pc.shape[0], pc.shape[1], num_outliers_to_gen)) * cfg["training"]["outlier_scale"]
                )

        pc = pc.to(device)
        kp = kp.to(device)
        R = R.to(device)
        t = t.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # print("Test:: pc_shape: ", pc.shape)
        # print(pc[0, ...])
        out = model(object_id, pc)
        kp_pred = out[1]

        # kp_dist_reg = - kpt_reg_coef * avg_kpt_distance_regularizer(kp_pred)
        kp_loss = ((kp - kp_pred) ** 2).sum(dim=1).mean(dim=1).mean()
        kp_dist_reg = kpt_reg_coef * bounded_avg_kpt_distance_loss(kp_pred, eps_bound=model.min_intra_kpt_dist * 0.1)
        loss = kp_loss + kp_dist_reg
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()  # Note: the output of supervised_loss is already averaged over batch_size
        if i % 10 == 0:
            print("Batch ", (i + 1), " loss: ", loss.item(), " kp reg loss: ", kp_dist_reg.item())

        del pc, kp, R, t, kp_pred
        torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)

    return ave_tloss


# Validation code
def validate(validation_loader, object_id, model, device, visualize=False):
    # We don't need gradients on to do reporting.
    with torch.no_grad():

        running_vloss = 0.0
        running_max_intra_kp_dist = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, keypoints_target, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)

            # Make predictions for this batch
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _ = model(
                object_id, input_point_cloud
            )

            vloss = ((keypoints_target - predicted_keypoints) ** 2).sum(dim=1).mean(dim=1).mean()
            intra_kp_dists = torch.cdist(
                torch.transpose(predicted_keypoints, -1, -2), torch.transpose(predicted_keypoints, -1, -2), p=2
            )

            running_vloss += vloss
            running_max_intra_kp_dist += torch.max(torch.triu(intra_kp_dists, diagonal=1)).item()

            ## visualize
            if visualize:
                pc = input_point_cloud.clone().detach().to("cpu")
                pc_p = predicted_point_cloud.clone().detach().to("cpu")
                # pc_t = pc_t.clone().detach().to("cpu")
                kp = keypoints_target.clone().detach().to("cpu")
                kp_p = predicted_keypoints.clone().detach().to("cpu")
                print("DISPLAY: INPUT AND PREDICTED PC")
                display_results(
                    input_point_cloud=pc, detected_keypoints=kp_p, target_point_cloud=pc_p, target_keypoints=kp
                )

            del (
                input_point_cloud,
                keypoints_target,
                R_target,
                t_target,
                predicted_point_cloud,
                predicted_keypoints,
                R_predicted,
                t_predicted,
            )

        avg_vloss = running_vloss / (i + 1)
        avg_intra_kp_dist = running_max_intra_kp_dist / (i + 1)
        logging.info(f"Validation avg. intra kp dist: {avg_intra_kp_dist}")

    return avg_vloss


def train_with_supervision(
    supervised_training_loader,
    validation_loader,
    object_id,
    model,
    optimizer,
    scheduler,
    best_model_save_file,
    device,
    cfg,
):
    num_epochs = cfg["training"]["num_epochs"]
    best_vloss = 1_000_000.0

    train_loss = TrackingMeter()
    val_loss = TrackingMeter()
    epoch_number = 0

    for epoch in range(num_epochs):
        logging.info("EPOCH {}:".format(epoch_number + 1))
        logging.info(f"Current LR: {scheduler.get_last_lr()}")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        logging.info("Training on simulated data with supervision:")
        avg_loss_supervised = supervised_train_one_epoch(
            training_loader=supervised_training_loader,
            model=model,
            object_id=object_id,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
        )
        # Validation.
        model.train(False)
        logging.info("Validation on real data: ")
        avg_vloss = validate(validation_loader, object_id, model, device=device)

        logging.info("LOSS supervised-train {}, valid {}".format(avg_loss_supervised, avg_vloss))
        train_loss.add_item(avg_loss_supervised)
        val_loss.add_item(avg_vloss)

        # Saving the model with the best vloss
        if avg_vloss < best_vloss:
            logging.info("Saving model because best validation losses has decreased.")
            best_vloss = avg_vloss
            torch.save(model.state_dict(), best_model_save_file)

        epoch_number += 1
        scheduler.step()

        torch.cuda.empty_cache()

    return train_loss, val_loss


def train_detector(cfg, detector_type="pointnet", model_id="obj_000001", use_corrector=False, **kwargs):
    """Main training function"""
    dataset = cfg["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running supervised_training: {datetime.now()}")
    logging.info(f"device is {device}")
    torch.cuda.empty_cache()

    # models save directory structure:
    # save_folder
    # |- dataset
    #    |- model_id
    #       |- timestamped path
    #          |- model files
    save_folder = cfg["save_folder"]
    best_model_save_file = os.path.join(save_folder, f"_{dataset}_best_supervised_kp_{detector_type}.pth")
    train_loss_save_file = os.path.join(save_folder, f"_{dataset}_train_loss_{detector_type}.pkl")
    val_loss_save_file = os.path.join(save_folder, f"_{dataset}_val_loss_{detector_type}.pkl")

    # optimization parameters
    lr_sgd = cfg["optimizer"]["lr_sgd"]
    momentum_sgd = cfg["optimizer"]["momentum_sgd"]

    # simulated training data
    train_dataset_len = cfg["training"]["train_dataset_len"]
    train_batch_size = cfg["training"]["train_batch_size"]
    train_num_of_points = cfg["training"]["train_num_of_points"]

    if dataset == "ycbv":
        supervised_train_dataset = ycbv.SE3PointCloud(
            model_id=model_id,
            num_of_points=train_num_of_points,
            dataset_len=train_dataset_len,
            cad_points_sampling_method=cfg["training"]["cad_model_sampling_method"],
            load_pointwise_color=cfg["training"]["load_rgb_for_points"],
            normalize=cfg["training"]["normalize_pc"],
        )
        supervised_train_loader = torch.utils.data.DataLoader(
            supervised_train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=5
        )

        # simulated validation dataset:
        val_dataset_len = cfg["training"]["val_dataset_len"]
        val_batch_size = cfg["training"]["val_batch_size"]
        val_num_of_points = cfg["training"]["val_num_of_points"]

        val_dataset = ycbv.SE3PointCloud(
            model_id=model_id,
            num_of_points=val_num_of_points,
            dataset_len=val_dataset_len,
            cad_points_sampling_method=cfg["training"]["cad_model_sampling_method"],
            load_pointwise_color=cfg["training"]["load_rgb_for_points"],
            normalize=cfg["training"]["normalize_pc"],
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=5)
    elif dataset == "tless":
        supervised_train_dataset = tless.SE3PointCloud(
            model_id=model_id,
            num_of_points=train_num_of_points,
            dataset_len=train_dataset_len,
            cad_points_sampling_method=cfg["training"]["cad_model_sampling_method"],
            normalize=cfg["training"]["normalize_pc"],
        )
        supervised_train_loader = torch.utils.data.DataLoader(
            supervised_train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=5
        )

        # simulated validation dataset:
        val_dataset_len = cfg["training"]["val_dataset_len"]
        val_batch_size = cfg["training"]["val_batch_size"]
        val_num_of_points = cfg["training"]["val_num_of_points"]

        val_dataset = tless.SE3PointCloud(
            model_id=model_id,
            num_of_points=val_num_of_points,
            dataset_len=val_dataset_len,
            cad_points_sampling_method=cfg["training"]["cad_model_sampling_method"],
            normalize=cfg["training"]["normalize_pc"],
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=5)
    else:
        raise NotImplementedError

    # Generate a shape category, CAD model objects, etc.
    cad_models = supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)

    # model
    _, _, original_cad_model, original_model_keypoints, obj_diameter = load_c3po_cad_models(
        model_id, device, output_unit="m", cfg=cfg
    )

    c3po_cfg = cfg["c3po"]
    if cfg["training"]["normalize_pc"]:
        obj_diameter = 1.0
    model = PointsRegressionModel(
        model_id=model_id,
        model_keypoints=model_keypoints,
        cad_models=cad_models,
        object_diameter=obj_diameter,
        keypoint_detector=detector_type,
        correction_flag=use_corrector,
        use_icp=False,
        zero_center_input=c3po_cfg["zero_center_input"],
        use_robust_centroid=c3po_cfg["use_robust_centroid"],
        c3po_config=cfg["c3po"],
        input_pc_normalized=cfg["training"]["normalize_pc"],
    ).to(device)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logging.info(f"Number of trainable parameters: {num_parameters}")

    # optimizer
    weight_decay_sgd = cfg["optimizer"]["weight_decay_sgd"]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd, weight_decay=weight_decay_sgd)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler"]["step_size"], gamma=cfg["scheduler"]["gamma"]
    )

    # training
    train_loss, val_loss = train_with_supervision(
        supervised_training_loader=supervised_train_loader,
        validation_loader=val_loader,
        object_id=model_id,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        best_model_save_file=best_model_save_file,
        device=device,
        cfg=cfg,
    )

    with open(train_loss_save_file, "wb") as outp:
        pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

    with open(val_loss_save_file, "wb") as outp:
        pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

    del (
        supervised_train_dataset,
        supervised_train_loader,
        val_dataset,
        val_loader,
        cad_models,
        model_keypoints,
        optimizer,
        model,
    )

    return None


def visual_test(test_loader, model, device=None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()

    for i, vdata in enumerate(test_loader):
        input_point_cloud, keypoints_target, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        keypoints_target = keypoints_target.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Make predictions for this batch
        model.eval()
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _ = model(input_point_cloud)
        model.train()
        pc = input_point_cloud.clone().detach().to("cpu")
        pc_p = predicted_point_cloud.clone().detach().to("cpu")
        kp = keypoints_target.clone().detach().to("cpu")
        kp_p = predicted_keypoints.clone().detach().to("cpu")
        # display_results(input_point_cloud=pc_p, detected_keypoints=kp_p, target_point_cloud=pc,
        #                 target_keypoints=kp)
        display_results(input_point_cloud=pc, detected_keypoints=kp_p, target_point_cloud=pc, target_keypoints=kp)

        del pc, pc_p, kp, kp_p
        del (
            input_point_cloud,
            keypoints_target,
            R_target,
            t_target,
            predicted_point_cloud,
            predicted_keypoints,
            R_predicted,
            t_predicted,
        )

        if i >= 10:
            break
