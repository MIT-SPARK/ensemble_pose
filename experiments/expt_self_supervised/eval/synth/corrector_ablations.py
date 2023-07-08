import copy
import os
import pandas
import numpy as np
import pickle
import pathlib
import argparse
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from pathlib import Path

from datasets import ycbv
from datasets.bop import make_scene_dataset, make_object_dataset
from datasets.pose import PoseDataset, FramePoseDataset, ObjectPCPoseDataset
from datasets import bop_constants
from utils.evaluation_metrics import add_s_error, chamfer_dist
import utils.visualization_utils as vutils
from utils.file_utils import safely_make_folders
from utils.visualization_utils import display_results

from expt_self_supervised.proposed_model import PointsRegressionModel as ProposedModel
from expt_self_supervised.proposed_model import load_c3po_cad_models, load_c3po_model
from expt_self_supervised.training_utils import load_yaml_cfg


def test_one_iteration(model, object_id, vdata, cad_models, object_diameter, visualize, adds_threshold, device, cfg):
    """Run one iteration"""
    input_point_cloud, keypoints_target, R_target, t_target = vdata
    input_point_cloud = input_point_cloud.to(device)
    keypoints_target = keypoints_target.to(device)
    R_target = R_target.float().to(device)
    t_target = t_target.float().to(device)
    pc_t = R_target @ cad_models + t_target[:, :, None]

    # Make predictions for this batch
    (
        predicted_point_cloud,
        predicted_keypoints,
        R_predicted,
        t_predicted,
        _,
        predicted_model_keypoints,
    ) = model(object_id, input_point_cloud)

    # gt data
    pc = input_point_cloud.clone().detach()
    kp = keypoints_target.clone().detach()
    pc_t = pc_t.clone().detach()

    # predicted data
    kp_p = predicted_keypoints.clone().detach()
    pc_p = predicted_point_cloud.clone().detach()

    if cfg["training"]["normalize_pc"]:
        # input_point_cloud/keypoints are normalized -> back to original frame (potentially centered)
        # R_target and t_target are good
        pc *= object_diameter
        kp *= object_diameter

    if visualize:
        print("Visualizing input point cloud vs. predicted CAD.")
        vutils.visualize_gt_and_pred_keypoints(
            pc[:, :3, :].to("cpu"), kp.to("cpu"), kp_pred=kp_p.to("cpu"), pc_pred=pc_p.to("cpu"), radius=0.005
        )
        print("Visualizing GT transformed CAD vs. predicted CAD.")
        vutils.visualize_gt_and_pred_keypoints(
            pc_t[:, :3, :].to("cpu"), kp.to("cpu"), kp_pred=kp_p.to("cpu"), pc_pred=pc_p.to("cpu"), radius=0.005
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
    c_dist = chamfer_dist(pc_p, pc_t, max_loss=False)
    add_s = add_s_error(
        predicted_point_cloud=predicted_point_cloud,
        ground_truth_point_cloud=pc_t,
        threshold=adds_threshold,
    )

    # intra kp predicted pairwise distances
    inds = torch.triu_indices(kp_p.shape[1], kp_p.shape[2], offset=1)
    euclidian_dists = torch.cdist(torch.transpose(kp_p, -1, -2), torch.transpose(kp_p, -1, -2), p=2)
    print(f"Max intra predicted kp dist: {torch.max(euclidian_dists[:, inds[0, :], inds[1, :]])}")
    print(f"Min intra predicted kp dist: {torch.min(euclidian_dists[:, inds[0, :], inds[1, :]])}")

    gt_euclidian_dists = torch.cdist(torch.transpose(kp, -1, -2), torch.transpose(kp, -1, -2), p=2)
    print(f"Max intra GT kp dist: {torch.max(gt_euclidian_dists[:, inds[0, :], inds[1, :]])}")
    print(f"Min intra GT kp dist: {torch.min(gt_euclidian_dists[:, inds[0, :], inds[1, :]])}")

    del pc, pc_p, kp, kp_p, pc_t
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

    return {
        "chamfer_dist": c_dist.flatten().tolist(),
        "add_s_err": add_s[0].flatten().tolist(),
    }


def test_runner(
    test_loader,
    model_id,
    object_diameter,
    models,
    cad_models,
    cfg,
    device=None,
    visualize=False,
    adds_threshold=0.05,
):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _, v in models.items():
        v.eval()

    results = []
    cad_models = cad_models.to(device)
    for i, vdata in tqdm(enumerate(test_loader), total=len(test_loader), desc="w/o corrector test"):

        plds = {
            k: test_one_iteration(
                model=x,
                object_id=model_id,
                vdata=vdata,
                cad_models=cad_models,
                object_diameter=object_diameter,
                visualize=visualize,
                adds_threshold=adds_threshold,
                device=device,
                cfg=cfg,
            )
            for k, x in models.items()
        }

        for k, v in plds.items():
            results.append(
                {
                    "add_s_err": v["add_s_err"],
                    "chamfer_dist": v["chamfer_dist"],
                    "type": k,
                }
            )

    return results


def main_exp(args, data_fname, dump_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config params
    config_params_file = args.config
    cfg = load_yaml_cfg(config_params_file, object_id=args.object_model_id)

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

    eval_batch_size = 50
    eval_scene_ds = make_scene_dataset(args.dataset, bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)
    ds_kwargs = dict(
        object_diameter=obj_diameter,
        pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
        min_area=cfg["training"]["min_area"],
        load_rgb_for_points=cfg["training"]["load_rgb_for_points"],
        dataset_name=cfg["dataset"],
        zero_center_pc=cfg["training"]["zero_center_pc"],
        use_robust_centroid=cfg["training"]["use_robust_centroid"],
        resample_invalid_pts=cfg["training"]["resample_invalid_pts"],
        normalize_pc=cfg["training"]["normalize_pc"],
        load_data_from_cache=True,
    )
    eval_dataset = ObjectPCPoseDataset(eval_scene_ds, model_id, **ds_kwargs)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=5)

    # load model
    model_wo_corrector = load_c3po_model(
        model_id=model_id,
        cad_models=original_cad_model,
        model_keypoints=original_model_keypoints,
        object_diameter=obj_diameter,
        device=device,
        cfg=cfg,
    )
    cfg_co = copy.deepcopy(cfg)
    cfg_co["c3po"]["use_corrector"] = True
    cfg_co["c3po"]["corrector"]["chamfer_loss_clamp_thres"] = 0.3
    model_w_corrector = load_c3po_model(
        model_id=model_id,
        cad_models=original_cad_model,
        model_keypoints=original_model_keypoints,
        object_diameter=obj_diameter,
        device=device,
        cfg=cfg_co,
    )

    # non-robust corrector
    cfg_non_robust_co = copy.deepcopy(cfg_co)
    cfg_non_robust_co["c3po"]["corrector"]["clamp_chamfer_loss"] = False
    model_w_nonrobust_corrector = load_c3po_model(
        model_id=model_id,
        cad_models=original_cad_model,
        model_keypoints=original_model_keypoints,
        object_diameter=obj_diameter,
        device=device,
        cfg=cfg_non_robust_co,
    )

    # load weights
    state_dict = torch.load(args.weights_path)
    if "state_dict" in state_dict.keys():
        model_wo_corrector.load_state_dict(state_dict["state_dict"])
        model_w_corrector.load_state_dict(state_dict["state_dict"])
        model_w_nonrobust_corrector.load_state_dict(state_dict["state_dict"])
    else:
        model_wo_corrector.load_state_dict(state_dict)
        model_w_corrector.load_state_dict(state_dict)
        model_w_nonrobust_corrector.load_state_dict(state_dict)

    models = {
        "w_corrector": model_w_corrector,
        "w_nonrobust_corrector": model_w_nonrobust_corrector,
        "wo_corrector": model_wo_corrector,
    }

    with torch.no_grad():
        # model test without corrector
        model_w_corrector.train(False)
        model_wo_corrector.train(False)
        model_w_nonrobust_corrector.train(False)
        exp_results = test_runner(
            test_loader=eval_loader,
            model_id=model_id,
            object_diameter=obj_diameter,
            models=models,
            cad_models=original_cad_model,
            adds_threshold=0.05 * obj_diameter,
            device=device,
            cfg=cfg,
            visualize=False,
        )

    # save the data
    safely_make_folders([dump_folder])
    with open(os.path.join(dump_folder, data_fname), "wb") as outp:
        pickle.dump(exp_results, outp, pickle.HIGHEST_PROTOCOL)

    return


def plot_data(data_fname, dump_folder):
    """plot function"""
    with open(os.path.join(dump_folder, data_fname), "rb") as f:
        data = pickle.load(f)

    # flatten the data
    ks = data[0].keys()
    flattened_data = {k: [] for k in ks}
    for row in tqdm(data):
        batch_size = len(row["chamfer_dist"])
        for k in ks:
            if k == "type":
                flattened_data[k].extend([row[k]] * batch_size)
            elif k == "chamfer_dist":
                flattened_data[k].extend(np.sqrt(np.array(row[k])).tolist())
            else:
                flattened_data[k].extend(row[k])

    # plot 1: chamfer distance box plot w & w/o corrector
    df = pandas.DataFrame.from_dict(flattened_data, orient="index").transpose()
    bp = sns.boxplot(data=df, x="type", y="chamfer_dist")
    bp.set(yscale="log")
    plt.show()

    plt.figure()
    kp = sns.kdeplot(data=df, x="chamfer_dist", hue="type")
    kp.set(xlim=(0, 0.25))
    plt.grid()
    plt.show()

    return


if __name__ == "__main__":
    """
    Run this script to calculate data for
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="specify the dataset (with split).", type=str)
    parser.add_argument("object_model_id", help="specify the object name.", type=str)
    parser.add_argument("weights_path", help="path to the model weights")
    parser.add_argument("--plot_only", help="only plot the data", action="store_true")
    parser.add_argument(
        "--config",
        help="path of the config file",
        default=f"./configs/supervised_ycbv.yml",
        type=str,
    )

    args = parser.parse_args()

    data_fname = f"synth_corrector_or_not_data_{args.dataset}_{args.object_model_id}.pkl"
    dump_folder = os.path.join(pathlib.Path(__file__).parent.resolve(), "data_rss", args.dataset)

    if not args.plot_only:
        main_exp(args, data_fname, dump_folder)

    # plotting
    # load the data
    #plot_data(data_fname, dump_folder)
