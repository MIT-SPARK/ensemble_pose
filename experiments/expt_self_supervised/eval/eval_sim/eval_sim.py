import argparse
import logging
import os
import pickle
import torch
from tqdm import tqdm

from expt_self_supervised.proposed_model import PointsRegressionModel, load_c3po_cad_models
from expt_self_supervised.training_utils import load_yaml_cfg
from datasets import ycbv
from utils.math_utils import set_all_random_seeds
import utils.visualization_utils as vutils


def create_dataloaders(cfg, mc_runs=100):
    """Create dataset and dataloaders"""
    eval_dataset, eval_loader = None, None
    if dataset == "ycbv":
        # Evaluation
        # validation dataset:
        # eval_dataset_len = hyper_param["val_dataset_len"]
        # eval_batch_size = hyper_param["val_batch_size"]
        eval_num_of_points = cfg["training"]["val_num_of_points"]
        eval_dataset_len = mc_runs
        eval_batch_size = 1

        if "normalize_pc" not in cfg["training"].keys():
            cfg["training"]["normalize_pc"] = True

        eval_dataset = ycbv.SE3PointCloud(
            model_id=model_id,
            num_of_points=eval_num_of_points,
            dataset_len=eval_dataset_len,
            normalize=cfg["training"]["normalize_pc"],
        )
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    else:
        raise NotImplementedError
    return eval_dataset, eval_loader


def create_model(
    object_id,
    cad_models,
    model_keypoints,
    weights_path,
    cfg,
    device="cuda",
    zero_center_input=False,
    use_robust_centroid=False,
    input_pc_normalized=False,
):
    """Create model under test"""
    detector_type = cfg["c3po"]["detector_type"]

    if use_robust_centroid:
        logging.info("Setting C3PO to use the robust centroid algorithm to find the center of the point cloud.")
        zero_center_input = True

    if "num_of_points_to_sample" not in cfg["c3po"]["point_transformer"].keys():
        cfg["c3po"]["point_transformer"]["num_of_points_to_sample"] = 1000
    if "sampling_type" not in cfg["c3po"]["point_transformer"].keys():
        cfg["c3po"]["point_transformer"]["sampling_type"] = "fps"
    if "sampling_ratio" not in cfg["c3po"]["point_transformer"].keys():
        cfg["c3po"]["point_transformer"]["sampling_ratio"] = 0.5
    if "input_feature_dim" not in cfg["c3po"]["point_transformer"].keys():
        cfg["c3po"]["point_transformer"]["input_feature_dim"] = 0

    _, _, original_cad_model, original_model_keypoints, obj_diameter = load_c3po_cad_models(
        object_id, device, output_unit="m", cfg=cfg
    )
    if input_pc_normalized:
        obj_diameter = 1.0
    model = PointsRegressionModel(
        model_id=object_id,
        model_keypoints=model_keypoints,
        cad_models=cad_models,
        # note: the simulated full pc dataset returns models that have diameters already normalized to 1
        object_diameter=obj_diameter,
        keypoint_detector=detector_type,
        need_predicted_keypoints=True,
        correction_flag=False,
        use_icp=False,
        zero_center_input=zero_center_input,
        use_robust_centroid=use_robust_centroid,
        input_pc_normalized=input_pc_normalized,
        c3po_config=cfg["c3po"],
    ).to(device)

    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)

    return model


def evaluate_model_with_outliers(
    outlier_ratios,
    eval_loader,
    object_id,
    model,
    cfg,
    device=None,
    outlier_scale=1.0,
    method_name="point_transformer",
    use_gt_center=True,
    visualize=True,
):
    """Test across outlier levels
    Currently evaluates 1) keypoints distances
    """
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outlier_err_data = []
    with torch.no_grad():
        model.eval()

        for outlier_ratio in tqdm(outlier_ratios):
            logging.info(f"Testing at outlier ratio = {outlier_ratio}.")

            for i, vdata in tqdm(enumerate(eval_loader), leave=False):
                input_point_cloud, keypoints_target, R_target, t_target = vdata
                input_point_cloud = input_point_cloud.to(device)
                keypoints_target = keypoints_target.to(device)
                R_target = R_target.to(device)
                t_target = t_target.to(device)

                # add outliers
                num_outliers_to_gen = int(outlier_ratio * input_point_cloud.shape[-1])
                outlier_indices = torch.randperm(input_point_cloud.shape[-1])[:num_outliers_to_gen]
                outlier_pc = input_point_cloud.clone()

                outliers = (
                    torch.rand(
                        (input_point_cloud.shape[0], input_point_cloud.shape[1], num_outliers_to_gen), device=device
                    )
                    - 0.5
                )
                outliers /= torch.linalg.norm(outliers, dim=1, keepdim=True)
                outliers *= 1.0 + outlier_scale
                outlier_pc[..., outlier_indices] = outliers

                if use_gt_center:
                    # center the input point cloud with the ground truth translation
                    outlier_pc_gt_centered = outlier_pc - t_target
                    # run detector
                    (
                        predicted_point_cloud,
                        predicted_keypoints,
                        R_predicted,
                        t_predicted,
                        _,
                        predicted_model_keypoints,
                    ) = model(object_id, outlier_pc_gt_centered)
                    # de-center the predicted keypoints
                    predicted_keypoints = predicted_keypoints + t_target
                else:
                    # run detector
                    (
                        predicted_point_cloud,
                        predicted_keypoints,
                        R_predicted,
                        t_predicted,
                        _,
                        predicted_model_keypoints,
                    ) = model(object_id, outlier_pc)

                # calculate keypoint distances
                kp_err = ((keypoints_target - predicted_keypoints) ** 2).sum(dim=1).mean(dim=1).mean()
                # kp_pred2gt_dist = (
                #    (torch.sqrt((keypoints_target - predicted_keypoints) ** 2)).sum(dim=1).mean(dim=1).mean()
                # )
                kp_pred2gt_dist = ((keypoints_target - predicted_keypoints) ** 2).sum(dim=1).sqrt().mean(dim=1).mean()
                if visualize:
                    logging.info("Visualizing predictions and keypoint annotations.")
                    # visualize: input point cloud, ground truth keypoint annotations, predicted keypoints
                    vutils.visualize_gt_and_pred_keypoints(
                        outlier_pc[:, :3, :], keypoints_target, kp_pred=predicted_keypoints
                    )
                outlier_err_data.append(
                    {
                        "method": method_name,
                        "noise_sigma": 0,
                        "kp_err": kp_err,
                        "kp_pred2gt_dist": kp_pred2gt_dist,
                        "kp_pred": predicted_keypoints,
                        "kp_gt": keypoints_target,
                        "outlier_ratio": outlier_ratio,
                        "object_id": object_id,
                    }
                )

    return outlier_err_data


def evaluate_model_with_noise(
    noise_sigmas, eval_loader, object_id, model, cfg, device=None, method_name="point_transformer", use_gt_center=True
):
    """Run evaluation on the model. Generate plots and statistics.
    Currently evaluates 1) keypoints distances
    """
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kp_err_data_noisy = []
    with torch.no_grad():
        model.eval()
        # test different noise levels
        for n_sigma in tqdm(noise_sigmas):
            logging.info(f"Testing at noise sigma={n_sigma}.")

            for i, vdata in tqdm(enumerate(eval_loader), leave=False):
                input_point_cloud, keypoints_target, R_target, t_target = vdata
                input_point_cloud = input_point_cloud.to(device)
                keypoints_target = keypoints_target.to(device)
                R_target = R_target.to(device)
                t_target = t_target.to(device)

                noisy_input_pc = input_point_cloud + torch.normal(
                    mean=0, std=n_sigma, size=input_point_cloud.shape, device=device
                )

                if use_gt_center:
                    # center the input point cloud with gt translation
                    centered_noisy_input_pc = noisy_input_pc - t_target
                    (
                        predicted_point_cloud,
                        predicted_keypoints,
                        R_predicted,
                        t_predicted,
                        _,
                        predicted_model_keypoints,
                    ) = model(object_id, centered_noisy_input_pc)
                    # de-center with the gt translation
                    predicted_keypoints = predicted_keypoints + t_target
                else:
                    (
                        predicted_point_cloud,
                        predicted_keypoints,
                        R_predicted,
                        t_predicted,
                        _,
                        predicted_model_keypoints,
                    ) = model(object_id, noisy_input_pc)

                # calculate keypoint distances
                kp_err = ((keypoints_target - predicted_keypoints) ** 2).sum(dim=1).mean(dim=1).mean()
                # kp_pred2gt_dist = (
                #    (torch.sqrt((keypoints_target - predicted_keypoints) ** 2)).sum(dim=1).mean(dim=1).mean()
                # )
                kp_pred2gt_dist = ((keypoints_target - predicted_keypoints) ** 2).sum(dim=1).sqrt().mean(dim=1).mean()
                kp_err_data_noisy.append(
                    {
                        "method": method_name,
                        "noise_sigma": n_sigma,
                        "kp_err": kp_err,
                        "kp_pred2gt_dist": kp_pred2gt_dist,
                        "kp_pred": predicted_keypoints,
                        "kp_gt": keypoints_target,
                        "outlier_ratio": 0,
                        "object_id": object_id,
                    }
                )

    return kp_err_data_noisy


if __name__ == "__main__":
    """ Run evaluation on model trained on sim dataset
    
    Example:
    python eval_sim.py ycbv obj_000001 pt_transformer_bn \
    ./exp_results/supervised_reg/ycbv/obj_000001/_ycbv_best_supervised_kp_point_transformer.pth \
    ./exp_results/supervised_reg/ycbv/obj_000001/config.yml 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="specify the dataset.", type=str)
    parser.add_argument("object_model_id", help="id of the object", type=str)
    parser.add_argument("model_type_to_analyze", help="name of model (will be used to tag the data)", type=str)
    parser.add_argument("weights_path", help="path to the model weights", type=str)
    parser.add_argument("config", help="path to the model config.yml", type=str)
    parser.add_argument("--noise_to_test", default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.5], nargs="*")
    parser.add_argument("--noise_data_save_path", default="./eval/eval_sim/data")
    parser.add_argument("--outlier_ratios_to_test", default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], nargs="*")
    parser.add_argument("--outlier_data_save_path", default="./eval/eval_sim/data")
    parser.add_argument("--runs", help="number of MC runs per exp", default=100)
    parser.add_argument(
        "--zero_center_input",
        help="set true to zero center inputs in C3PO model; this center will be affected by the outliers.",
        default=True,
    )
    parser.add_argument(
        "--use_robust_centroid",
        help="set True to use the robust centroid algorithm to counteract the effects of outliers",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    # load config params
    config_params_file = args.config
    model_id = args.object_model_id
    logging.info(f"Evaluating {model_id}")
    dataset = args.dataset
    model_weights_path = args.weights_path

    cfg = load_yaml_cfg(config_params_file, object_id=model_id)
    if "bop_ds_dir" not in cfg.keys():
        cfg["bop_ds_dir"] = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../data/bop/bop_datasets/")
        )

    # set random seed for reproducible results
    set_all_random_seeds(0)

    # handle https://github.com/pytorch/pytorch/issues/77527
    torch.backends.cuda.preferred_linalg_library("cusolver")

    # decide on dataloader
    eval_dataset, eval_loader = create_dataloaders(cfg, mc_runs=args.runs)

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)
    if not args.use_robust_centroid:
        logging.info("C3PO does not use robust centroid.")
    else:
        logging.info("C3PO uses robust centroid.")
    model = create_model(
        object_id=model_id,
        cad_models=cad_models,
        model_keypoints=model_keypoints,
        weights_path=model_weights_path,
        zero_center_input=args.zero_center_input,
        use_robust_centroid=args.use_robust_centroid,
        # note: the dataloader returns normalized keypoints/model, and we handle it by setting the object diameter to be 1
        input_pc_normalized=cfg["training"]["normalize_pc"],
        cfg=cfg,
    )

    use_gt_center = True
    if args.zero_center_input:
        # zero center in C3PO model
        use_gt_center = False
    if args.use_robust_centroid:
        # use robust centroid GNC
        use_gt_center = False

    # evaluate at different noise levels & save
    kp_noise_err = evaluate_model_with_noise(
        method_name=args.model_type_to_analyze,
        noise_sigmas=args.noise_to_test,
        eval_loader=eval_loader,
        object_id=model_id,
        use_gt_center=use_gt_center,
        model=model,
        cfg=cfg,
        device=device,
    )
    noise_data_fname = f"kp_noise_err_data_{dataset}_{model_id}_{args.model_type_to_analyze}.pkl"
    with open(os.path.join(args.noise_data_save_path, dataset, noise_data_fname), "wb") as outp:
        pickle.dump(kp_noise_err, outp, pickle.HIGHEST_PROTOCOL)

    # evaluate at different outlier ratios & save
    outlier_err = evaluate_model_with_outliers(
        method_name=args.model_type_to_analyze,
        outlier_ratios=args.outlier_ratios_to_test,
        eval_loader=eval_loader,
        object_id=model_id,
        model=model,
        use_gt_center=use_gt_center,
        cfg=cfg,
        device=device,
        visualize=False,
    )
    outlier_data_fname = f"kp_outlier_err_data_{dataset}_{model_id}_{args.model_type_to_analyze}.pkl"
    outlier_data_path = os.path.join(args.outlier_data_save_path, dataset, outlier_data_fname)
    logging.info(f"Data saved to {outlier_data_path}")
    with open(outlier_data_path, "wb") as outp:
        pickle.dump(outlier_err, outp, pickle.HIGHEST_PROTOCOL)
