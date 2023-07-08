import numpy as np
import pandas as pd
import pickle

import os
import re
import json

import pathlib
from os import listdir
from os.path import isdir, join

from tqdm import tqdm

from datasets import bop_constants


def load_self6dpp_results(data_folder, args):
    """ Run this function to load self6dpp results

    Args:
        data_folder: the directory containing per object results
        args:
    """
    if "ycbv" in args.dataset:
        all_obj_ids = sorted(list(bop_constants.BOP_MODEL_INDICES["ycbv"].keys()))
    elif "tless" in args.dataset:
        all_obj_ids = sorted(list(bop_constants.BOP_MODEL_INDICES["tless"].keys()))
    else:
        raise NotImplementedError

    regex = re.compile("(errors_.*json$)")

    # load available object error directories
    per_obj_data = {}
    all_data_folders = sorted([f for f in listdir(data_folder) if isdir(join(data_folder, f))])
    for obj_dir in all_data_folders:
        c_dir_obj_id = f"obj_{int(obj_dir.split('_')[0]):06d}"
        if c_dir_obj_id not in all_obj_ids:
            print(f"Unknown object ID: {c_dir_obj_id}")

        # walk the object directory
        errors_candidates = []
        for root, dirs, files in os.walk(os.path.join(data_folder, obj_dir)):
            for file in files:
                if regex.match(file):
                    errors_candidates.append(os.path.join(root, file))

        # only take the ones with AUC
        adi_jsons = [x for x in errors_candidates if "error:AUCadi" in x]
        adi_data = []
        for jf in adi_jsons:
            with open(jf, "r") as f:
                data = json.load(f)
                if len(data) == 0:
                    continue
                # each entry in data is a dictionary containing a "errors" field, which is a dictionary
                # that is length one, and the value is the average chamfer distance in cm
                for entry in data:
                    # convert to m
                    adi_data.append(entry["errors"][list(entry["errors"].keys())[0]][0] / 100)

        per_obj_data[c_dir_obj_id] = adi_data
    return per_obj_data


def load_one_method(data_folder, args):
    df = None

    if "ycbv" in args.dataset:
        all_obj_ids = sorted(list(bop_constants.BOP_MODEL_INDICES["ycbv"].keys()))
    elif "tless" in args.dataset:
        all_obj_ids = sorted(list(bop_constants.BOP_MODEL_INDICES["tless"].keys()))
    else:
        raise NotImplementedError

    all_data_folders = sorted([f for f in listdir(data_folder) if isdir(join(data_folder, f))])

    all_obj_data = []
    for obj_dir in all_data_folders:
        if obj_dir not in all_obj_ids:
            print(f"Unknown obj dir: {obj_dir}")
            continue

        pkl_data_paths = list(pathlib.Path(os.path.join(data_folder, obj_dir)).glob("*.pkl"))
        if len(pkl_data_paths) != 1:
            print("More than one pkl data present.")
        data_path = pkl_data_paths[0]

        with open(str(data_path), "rb") as f:
            data = pickle.load(f)

        data = [dict(item, object_id=obj_dir) for item in data]
        all_obj_data.extend(data)

    df = pd.DataFrame.from_records(all_obj_data)
    return df


def cleanup_df(method, df, joint_selection_strat="c3po"):
    records = df.to_records()
    new_records = []

    def make_new_row(m, old_row, new_name, cert=None):
        if cert is None:
            cert = np.logical_and(old_row[f"cert_{m}_pc"], old_row[f"cert_{m}_mask"])
        return {
            "name": new_name,
            "chamfer_mean": old_row[f"{m}_chamfer_mean"],
            "chamfer_max": old_row[f"{m}_chamfer_max"],
            "chamfer_min": old_row[f"{m}_chamfer_min"],
            "chamfer_median": old_row[f"{m}_chamfer_median"],
            "object_id": old_row["object_id"],
            "cert": cert,
        }

    if "c3po_cosypose" in method:
        if "cert_c3po" not in df.keys():
            df["cert_c3po"] = df['cert_c3po_pc'] & df['cert_c3po_mask']
        if "cert_cosypose" not in df.keys():
            df["cert_cosypose"] = df['cert_cosypose_pc'] & df['cert_cosypose_mask']
        if "cert_joint" not in df.keys():
            df["cert_joint"] = df["cert_c3po"] & df["cert_cosypose"]
        records = df.to_records()

        for row in tqdm(records):
            # each row produces:
            # c3po only prediction
            # cosypose only prediction
            # c3po+cosypose joint prediction
            c3po_row = make_new_row("c3po", row, new_name=method + "_c3po_self_sup")
            cosypose_row = make_new_row("cosypose", row, new_name=method + "_cosypose_self_sup")

            new_records.append(c3po_row)
            new_records.append(cosypose_row)
            if row["cert_joint"]:
                if joint_selection_strat == "c3po":
                    joint_row = make_new_row("c3po", row, new_name=method + "_joint", cert=True)
                else:
                    joint_row = make_new_row("cosypose", row, new_name=method + "_joint", cert=True)
            elif row["use_c3po"] or row["use_cosypose"]:
                # note that in the eval code, if both cert, selects C3PO
                joint_row = make_new_row("joint", row, new_name=method + "_joint", cert=True)
            else:
                # if both fail, select C3PO
                if joint_selection_strat == "c3po":
                    joint_row = make_new_row("c3po", row, new_name=method + "_joint", cert=False)
                else:
                    joint_row = make_new_row("cosypose", row, new_name=method + "_joint", cert=False)
            new_records.append(joint_row)

        new_df = pd.DataFrame.from_records(new_records)
    elif method == "c3po_real" or method == "c3po_synth" or method == "c3po_synth_w_corrector" or method == "c3po_synth_w_icp":
        for row in tqdm(records):
            c3po_row = make_new_row("c3po", row, new_name=method)
            new_records.append(c3po_row)
        new_df = pd.DataFrame.from_records(new_records)
    elif (
        method == "cosypose_real_2_refine"
        or method == "cosypose_synth_2_refine"
        or method == "cosypose_synth_2_refine_w_corrector"
    ):
        for row in tqdm(records):
            cosypose_row = make_new_row("cosypose", row, new_name=method)
            new_records.append(cosypose_row)
        new_df = pd.DataFrame.from_records(new_records)
    else:
        raise NotImplementedError

    return new_df
