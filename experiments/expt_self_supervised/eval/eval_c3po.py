import torch
import os
import pickle
from tqdm import tqdm
from datetime import datetime
import argparse
import pathlib
from pathlib import Path

from datasets import ycbv, tless
from expt_self_supervised.proposed_model import PointsRegressionModel as ProposedModel
from expt_self_supervised.proposed_model import load_c3po_model, load_c3po_cad_models, load_all_cad_models
from utils.general import TrackingMeter
from utils.loss_functions import bounded_avg_kpt_distance_loss
import utils.visualization_utils as vutils
import utils.math_utils as mutils
from utils.torch_utils import get_grad_norm
from utils.visualization_utils import display_results
from utils.file_utils import safely_make_folders

from expt_self_supervised.training_utils import *
from utils.evaluation_metrics import add_s_error, chamfer_dist


def test_one_iteration(model, object_id, vdata, cad_models, object_diameter, visualize, adds_threshold, device, cfg):
    """Run one iteration"""
    # input_point_cloud, keypoints_target, R_target, t_target = vdata

    pc, kp, R_gt, t_gt = vdata

    pc = pc.to(device)
    kp = kp.to(device)
    R_gt = R_gt.float().to(device)
    t_gt = t_gt.float().to(device)

    pc_gt = R_gt @ cad_models + t_gt[:, :, None]

    # make predictions
    out = model(object_id, pc)

    if cfg["training"]["normalize_pc"]:
        kp_gt = kp
    else:
        kp_gt = kp / model.object_diameter
    pc_pred = out[0] / model.object_diameter
    kp_pred = out[1] / model.object_diameter
    R_pred, t_pred = out[2], out[3]

    # gt data
    pc_gt = pc_gt.clone().detach()

    # predicted data
    kp_pred = kp_pred.clone().detach()
    pc_pred = pc_pred.clone().detach()

    if visualize:
        print("Visualizing input point cloud vs. predicted CAD.")
        vutils.visualize_gt_and_pred_keypoints(
            pc[:, :3, :].to("cpu"), kp.to("cpu"), kp_pred=kp_pred.to("cpu"), pc_pred=pc_pred.to("cpu"), radius=0.005
        )
        print("Visualizing GT transformed CAD vs. predicted CAD.")
        vutils.visualize_gt_and_pred_keypoints(
            pc_gt[:, :3, :].to("cpu"), kp.to("cpu"), kp_pred=kp_pred.to("cpu"), pc_pred=pc_pred.to("cpu"), radius=0.005
        )

    ## certification
    # certi = certify(
    #    input_point_cloud=input_point_cloud,
    #    predicted_point_cloud=predicted_point_cloud,
    #    corrected_keypoints=predicted_keypoints,
    #    predicted_model_keypoints=predicted_model_keypoints,
    #    epsilon=hyper_param["epsilon"],
    # )
    # print("Certifiable: ", certi)

    # add-s
    c_dist = chamfer_dist(pc_pred, pc_gt, max_loss=False)
    add_s = add_s_error(
        predicted_point_cloud=pc_pred,
        ground_truth_point_cloud=pc_gt,
        threshold=adds_threshold,
    )
    kp_loss = ((kp_gt - kp_pred) ** 2).sum(dim=1).mean(dim=1)

    # intra kp predicted pairwise distances
    inds = torch.triu_indices(kp_pred.shape[1], kp_pred.shape[2], offset=1)
    euclidian_dists = torch.cdist(torch.transpose(kp_pred, -1, -2), torch.transpose(kp_pred, -1, -2), p=2)
    print(f"Max intra predicted kp dist: {torch.max(euclidian_dists[:, inds[0, :], inds[1, :]])}")
    print(f"Min intra predicted kp dist: {torch.min(euclidian_dists[:, inds[0, :], inds[1, :]])}")

    gt_euclidian_dists = torch.cdist(torch.transpose(kp, -1, -2), torch.transpose(kp, -1, -2), p=2)
    print(f"Max intra GT kp dist: {torch.max(gt_euclidian_dists[:, inds[0, :], inds[1, :]])}")
    print(f"Min intra GT kp dist: {torch.min(gt_euclidian_dists[:, inds[0, :], inds[1, :]])}")

    return {
        "object_id": object_id,
        "object_diameter": object_diameter,
        "normalized_chamfer_dist": c_dist.flatten().tolist(),
        "normalized_add_s_err": add_s[0].flatten().tolist(),
        "normalized_auc": add_s[1].flatten().tolist(),
        "R_gt": R_gt.cpu().numpy(),
        "t_gt": t_gt.cpu().numpy(),
        "R_pred": R_pred.cpu().numpy(),
        "t_pred": t_pred.cpu().numpy(),
        "kp_loss": kp_loss.cpu().numpy(),
        "kp_pred": kp_pred.cpu().numpy(),
        "kp_gt": kp_gt.cpu().numpy()
    }


def run_eval_one_epoch(
    test_loader,
    model_id,
    object_diameter,
    model,
    cad_models,
    visualize,
    cfg,
    device=None,
    adds_threshold=0.05,
):
    """One evaluation loop through the dataset"""
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = {
        "normalized_chamfer_dist": [],
        "normalized_add_s_err": [],
        "normalized_auc": [],
        "R_gt": [],
        "t_gt": [],
        "R_pred": [],
        "t_pred": [],
        "kp_loss": [],
        "kp_pred": [],
        "kp_gt": [],
    }
    with torch.no_grad():
        for i, vdata in tqdm(enumerate(test_loader), total=len(test_loader)):
            pyld = test_one_iteration(
                model, model_id, vdata, cad_models, object_diameter, visualize, adds_threshold, device, cfg
            )
            for k in data.keys():
                data[k].extend(pyld[k])

    data["object_id"] = model_id
    data["object_diameter"] = object_diameter

    return data


def main(args, data_fname, dump_folder):

    config_params_file = args.config
    cfg = load_yaml_cfg(config_params_file)

    model_id = args.object_model_id
    dataset = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # object datasets
    _, _, original_cad_model, original_model_keypoints, obj_diameter = load_c3po_cad_models(
        model_id, device, output_unit="m", cfg=cfg
    )

    eval_batch_size = 100
    eval_scene_ds = make_scene_dataset(args.dataset, bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)
    ds_kwargs = dict(
        object_diameter=obj_diameter,
        dataset_name=cfg["dataset"],
        min_area=cfg["training"]["min_area"],
        pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
        load_rgb_for_points=cfg["training"]["load_rgb_for_points"],
        zero_center_pc=cfg["training"]["zero_center_pc"],
        use_robust_centroid=cfg["training"]["use_robust_centroid"],
        resample_invalid_pts=cfg["training"]["resample_invalid_pts"],
        normalize_pc=cfg["training"]["normalize_pc"],
        load_data_from_cache=True,
        cache_save_dir=args.cache_dir,
    )
    eval_dataset = ObjectPCPoseDataset(eval_scene_ds, model_id, **ds_kwargs)
    if args.indices is not None:
        inds = list(np.load(args.indices))
        subset_dataset = torch.utils.data.Subset(eval_dataset, inds)
        eval_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=eval_batch_size, shuffle=False)
    else:
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

    exp_results = run_eval_one_epoch(
        test_loader=eval_loader,
        model_id=model_id,
        object_diameter=obj_diameter,
        model=model,
        cad_models=original_cad_model.float() / obj_diameter,
        visualize=False,
        cfg=cfg,
        device=device,
        adds_threshold=0.05,
    )

    # save the data
    safely_make_folders([dump_folder])
    with open(os.path.join(dump_folder, data_fname), "wb") as outp:
        pickle.dump(exp_results, outp, pickle.HIGHEST_PROTOCOL)

    return exp_results


if __name__ == "__main__":
    """Run this script to evaluate a C3PO model on a dataset

    Example:
    python eval/eval_c3po.py ycbv.test obj_000001 ./exp_results/ablations/fps/ycbv/obj_000001/_epoch_1_synth_supervised_single_obj_kp_point_transformer.pth.tar --config=./exp_results/ablations/fps/ycbv/obj_000001/config.yml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="specify the dataset (with split).", type=str)
    parser.add_argument("object_model_id", help="specify the object name.", type=str)
    parser.add_argument("weights_path", help="path to the model weights")
    parser.add_argument("exp_name", help="name of the datafile to store")
    parser.add_argument("--indices", help="npy file containing indices to use for the test set.", default=None)
    parser.add_argument("--cache_dir", help="cache save directory", default=None)
    parser.add_argument(
        "--config",
        help="path of the config file",
        default=f"./configs/supervised_ycbv.yml",
        type=str,
    )

    args = parser.parse_args()

    data_fname = f"{args.exp_name}_{args.dataset}_{args.object_model_id}.pkl"
    dump_folder = os.path.join(pathlib.Path(__file__).parent.resolve(), "data", args.dataset, args.object_model_id)

    exp_results = main(args, data_fname, dump_folder)
