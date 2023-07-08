import argparse
import torch
from pathlib import Path

from datasets import ycbv
from datasets.bop import make_scene_dataset, make_object_dataset
from datasets.pose import PoseDataset, FramePoseDataset, ObjectPCPoseDataset
from proposed_model import PointsRegressionModel as ProposedModel
from proposed_model import load_c3po_cad_models, load_c3po_model
from datasets import bop_constants
from training_utils import load_yaml_cfg
from utils.evaluation_metrics import add_s_error
import utils.visualization_utils as vutils
from utils.visualization_utils import display_results


def visual_test(test_loader, model_id, model, cad_models, cfg, device=None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cad_models = cad_models.to(device)

    model.eval()
    for i, vdata in enumerate(test_loader):
        input_point_cloud, keypoints_target, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        keypoints_target = keypoints_target.to(device)
        R_target = R_target.float().to(device)
        t_target = t_target.float().to(device)

        # Make predictions for this batch
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _, predicted_model_keypoints = model(
            model_id, input_point_cloud
        )

        if cfg["training"]["normalize_pc"]:
            kp_gt = keypoints_target
        else:
            kp_gt = keypoints_target / model.object_diameter
        kp_pred = predicted_keypoints / model.object_diameter

        # intra kp predicted pairwise distances
        inds = torch.triu_indices(kp_pred.shape[1], kp_pred.shape[2], offset=1)
        euclidian_dists = torch.cdist(torch.transpose(kp_pred, -1, -2), torch.transpose(kp_pred, -1, -2), p=2)
        print(f"Max intra predicted kp dist: {torch.max(euclidian_dists[:, inds[0, :], inds[1, :]])}")
        print(f"Min intra predicted kp dist: {torch.min(euclidian_dists[:, inds[0, :], inds[1, :]])}")

        gt_euclidian_dists = torch.cdist(torch.transpose(kp_gt, -1, -2), torch.transpose(kp_gt, -1, -2), p=2)
        print(f"Max intra GT kp dist: {torch.max(gt_euclidian_dists[:, inds[0, :], inds[1, :]])}")
        print(f"Min intra GT kp dist: {torch.min(gt_euclidian_dists[:, inds[0, :], inds[1, :]])}")

        print("Visualizing predictions and keypoint annotations.")
        vutils.visualize_gt_and_pred_keypoints(
            input_point_cloud[:, :3, :].to("cpu"), kp_gt.to("cpu"), kp_pred=kp_pred.to("cpu")
        )


if __name__ == "__main__":
    """
    usage:
    python visualize_model.py ycbv obj_000001 kp_detector ./exp_results/supervised/ycbv/obj_000001/_ycbv_best_supervised_kp_point_transformer.pth

    single obj vis (synth):
    python visualize_model.py ycbv obj_000001 ./exp_results/synth_sup_single_obj_depth_nopretrain/ycbv/obj_000001/_epoch_100_synth_supervised_single_obj_kp_point_transformer.pth.tar --config=./configs/synth_supervised_single_obj/ycbv/depth_nopretrain.yml
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="specify the dataset (with split).", type=str)
    parser.add_argument("object_model_id", help="specify the object name.", type=str)
    parser.add_argument("weights_path", help="path to the model weights")
    parser.add_argument(
        "--config",
        help="path of the config file",
        default=f"./configs/supervised_ycbv.yml",
        type=str,
    )

    args = parser.parse_known_args()
    args = args[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config params
    config_params_file = args.config
    cfg = load_yaml_cfg(config_params_file)

    model_id = args.object_model_id
    dataset = args.dataset
    model_weights_path = args.weights_path
    detector_type = cfg["c3po"]["detector_type"]

    # decide on dataloader
    eval_dataset, eval_loader = None, None
    if "ycbv" in dataset:
        all_model_names = list(bop_constants.YCBV.keys())
    elif "tless" in dataset:
        all_model_names = list(bop_constants.TLESS.keys())
    else:
        raise NotImplementedError

    # create model
    _, _, original_cad_model, original_model_keypoints, obj_diameter = load_c3po_cad_models(
        model_id, device, output_unit="m", cfg=cfg
    )

    eval_batch_size = 1
    if dataset == "ycbv.fullpc":
        # Evaluation
        # validation dataset:
        # eval_dataset_len = hyper_param["val_dataset_len"]
        # eval_batch_size = hyper_param["val_batch_size"]
        eval_num_of_points = cfg["training"]["val_num_of_points"]
        eval_dataset_len = 20

        eval_dataset = ycbv.SE3PointCloud(
            model_id=model_id,
            num_of_points=eval_num_of_points,
            dataset_len=eval_dataset_len,
        )
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    else:
        eval_scene_ds = make_scene_dataset(cfg["train_ds_name"], bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)
        ds_kwargs = dict(
            object_diameter=obj_diameter,
            pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
            min_area=None,
            load_rgb_for_points=cfg["training"]["load_rgb_for_points"],
            dataset_name=cfg["dataset"],
            zero_center_pc=cfg["training"]["zero_center_pc"],
            use_robust_centroid=cfg["training"]["use_robust_centroid"],
            resample_invalid_pts=cfg["training"]["resample_invalid_pts"],
            normalize_pc=cfg["training"]["normalize_pc"],
        )
        eval_dataset = ObjectPCPoseDataset(eval_scene_ds, model_id, **ds_kwargs)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    # load model
    model = load_c3po_model(
        model_id=model_id,
        cad_models=original_cad_model,
        model_keypoints=original_model_keypoints,
        object_diameter=obj_diameter,
        device=device,
        cfg=cfg,
    )

    # load weights
    state_dict = torch.load(args.weights_path)
    if "state_dict" in state_dict.keys():
        model.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict)

    with torch.no_grad():
        model.train(False)
        visual_test(test_loader=eval_loader, model_id=model_id, model=model, cad_models=original_cad_model, cfg=cfg)
