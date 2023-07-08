import pandas as pd
import pickle
import argparse
import torch

import os

import pathlib
from os import listdir
from os.path import isdir, join

from expt_self_supervised.proposed_model import (
    load_c3po_model,
    load_c3po_cad_models,
    load_all_cad_models,
    load_batch_renderer,
    load_certifier,
    load_cosypose_coarse_refine_model,
)

from utils.evaluation_metrics import VOCap
from expt_self_supervised.eval.model_comps.data_utils import load_one_method, cleanup_df, load_self6dpp_results
from datasets import bop_constants

ycbv_cad_cfg = {
    "dataset": "ycbv",
    "bop_ds_dir": os.path.abspath(
        os.path.join(pathlib.Path(__file__).parent.resolve(), "../../../../data/bop/bop_datasets/")
    ),
    "c3po": {"point_transformer": {"num_of_points_to_sample": 1000}},
}
tless_cad_cfg = {
    "dataset": "tless",
    "bop_ds_dir": os.path.abspath(
        os.path.join(pathlib.Path(__file__).parent.resolve(), "../../../../data/bop/bop_datasets/")
    ),
    "c3po": {"point_transformer": {"num_of_points_to_sample": 1000}},
}


def make_adds_table(results, dataset, adds_thres=0.05, adds_auc_thres=0.1, objects_to_use=None):
    """Each column is obj_add_s, obj_add_s_auc
    Threshold for ADD-S: 5%
    Threshold for ADD-S (AUC): 10%

    Each method is a row.
    """
    if dataset == "ycbv":
        cad_cfg = ycbv_cad_cfg
    elif dataset == "tless":
        cad_cfg = tless_cad_cfg
    else:
        raise NotImplementedError

    output_rec = []
    cert_output_rec = []

    for m, df in results.items():
        if m == "self6dpp":
            # a dictionary, not a dataframe
            obj_df = results[m]
        else:
            obj_df = df.groupby("object_id")

        # for each object,
        # extract chamfer_mean
        # calculate ADD-S threshold
        # calculate ADD-S auc
        c_row = {"method_name": m}
        if m != "self6dpp":
            c_row_cert = {"method_name": f"{m}_cert"}
        else:
            c_row_cert = None

        avg_adds = 0
        avg_adds_auc = 0
        avg_adds_cert = 0
        avg_adds_auc_cert = 0
        for obj_id in objects_to_use:
            try:
                _, _, _, _, obj_diameter = load_c3po_cad_models(obj_id, "cpu", output_unit="m", cfg=cad_cfg)
                obj_adds_thres = obj_diameter * adds_thres
                obj_adds_auc_thres = obj_diameter * adds_auc_thres

                if m == "self6dpp":
                    # handle self6d
                    obj_chamfer_dists = torch.as_tensor(obj_df[obj_id])
                    adds_mask = obj_chamfer_dists < obj_adds_thres
                    adds = torch.sum(adds_mask) / adds_mask.shape[0]
                    adds_auc = VOCap(obj_chamfer_dists, threshold=obj_adds_auc_thres)
                    c_row[f"{obj_id}_adds"] = adds.item()
                    c_row[f"{obj_id}_adds_auc"] = adds_auc.item()
                    avg_adds += adds.item()
                    avg_adds_auc += adds_auc.item()
                else:
                    c_obj_df = obj_df.get_group(obj_id)
                    if m == "c3po_cosypose_2_refine":
                        c_obj_df = c_obj_df[c_obj_df["name"] == "c3po_cosypose_2_refine_joint"]

                    # non cert
                    obj_chamfer_dists = torch.as_tensor(c_obj_df["chamfer_mean"].to_numpy())
                    adds_mask = obj_chamfer_dists < obj_adds_thres
                    adds = torch.sum(adds_mask) / adds_mask.shape[0]
                    adds_auc = VOCap(obj_chamfer_dists, threshold=obj_adds_auc_thres)
                    c_row[f"{obj_id}_adds"] = adds.item()
                    c_row[f"{obj_id}_adds_auc"] = adds_auc.item()
                    avg_adds += adds.item()
                    avg_adds_auc += adds_auc.item()

                    # cert
                    cert_c_obj_df = c_obj_df[c_obj_df["cert"]]
                    cert_obj_chamfer_dists = torch.as_tensor(cert_c_obj_df["chamfer_mean"].to_numpy())
                    cert_adds_mask = cert_obj_chamfer_dists < obj_adds_thres
                    cert_adds = torch.sum(cert_adds_mask) / cert_adds_mask.shape[0]
                    cert_adds_auc = VOCap(cert_obj_chamfer_dists, threshold=obj_adds_auc_thres)
                    c_row_cert[f"{obj_id}_adds"] = cert_adds.item()
                    c_row_cert[f"{obj_id}_adds_auc"] = cert_adds_auc.item()
                    avg_adds_cert += cert_adds.item()
                    avg_adds_auc_cert += cert_adds_auc.item()
            except:
                print(f"Skipping {obj_id}")
                continue

        avg_adds /= len(all_obj_ids)
        avg_adds_auc /= len(all_obj_ids)
        avg_adds_cert /= len(all_obj_ids)
        avg_adds_auc_cert /= len(all_obj_ids)

        c_row["avg_adds"] = avg_adds
        c_row["avg_adds_auc"] = avg_adds_auc

        output_rec.append(c_row)
        if c_row_cert is not None:
            c_row_cert["avg_adds"] = avg_adds_cert
            c_row_cert["avg_adds_auc"] = avg_adds_auc_cert
            cert_output_rec.append(c_row_cert)

    # make the table
    output_rec.extend(cert_output_rec)
    output_table = pd.DataFrame.from_records(output_rec)
    return output_table


if __name__ == "__main__":
    """
    Sample usage:
    python eval/model_comps/make_table.py ycbv ./exp_results/eval/ --self6d_data_folder=/mnt/datasets/bop/self6d_outputs/self6dpp
    """
    print("Make table for all objects' chamfer distances among all methods in test sets")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="specify the dataset.", type=str)
    parser.add_argument("results_folder", help="folder containing all eval data", type=str)
    parser.add_argument("--self6d_data_folder", help="folder containing self6d data", type=str, default=None)
    parser.add_argument(
        "--output_folder",
        help="folder to dump plots",
        type=str,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "tables"),
    )

    args = parser.parse_args()

    if "ycbv" in args.dataset:
        all_obj_ids = sorted(list(bop_constants.BOP_MODEL_INDICES["ycbv"].keys()))
    elif "tless" in args.dataset:
        all_obj_ids = sorted(list(bop_constants.BOP_MODEL_INDICES["tless"].keys()))
    else:
        raise NotImplementedError

    # list of methods to plot
    methods_to_plot = [
        "c3po_real",
        "c3po_synth",
        "c3po_synth_w_icp",
        "c3po_synth_w_corrector",
        "cosypose_real_2_refine",
        "cosypose_synth_2_refine",
        "c3po_cosypose_2_refine",
    ]

    data_folder = args.results_folder
    all_methods_folders = sorted([f for f in listdir(data_folder) if isdir(join(data_folder, f))])

    results = {}
    original_dfs = {}
    for method in all_methods_folders:
        if method in methods_to_plot:
            df = load_one_method(data_folder=os.path.join(data_folder, method, args.dataset), args=args)
            original_dfs[method] = df
            results[method] = cleanup_df(method, df)

    # load self6dpp data
    if args.self6d_data_folder is not None:
        print("loading Self6D data folder.")
        self6d_data = load_self6dpp_results(args.self6d_data_folder, args=args)
        results["self6dpp"] = self6d_data
        methods_to_plot.append("self6dpp")

    output_table = make_adds_table(results, dataset=args.dataset, objects_to_use=all_obj_ids)
    output_table.to_csv(os.path.join(args.output_folder, f"adds_table_{args.dataset}.csv"))
    output_table.style.to_latex(os.path.join(args.output_folder, f"adds_table_{args.dataset}.tex"))
