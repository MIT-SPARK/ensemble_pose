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

from os import listdir
from os.path import isfile, join


def plot_data(dump_folder, all_obj_ids):
    """plot function"""

    all_pkl_files = sorted([f for f in listdir(dump_folder) if isfile(join(dump_folder, f)) and ".pkl" in f])

    flattened_data = {"object_id": []}
    for pkl_f in all_pkl_files:

        with open(os.path.join(dump_folder, pkl_f), "rb") as f:
            data = pickle.load(f)

        obj_id = "_".join(pkl_f.split("_")[-2:]).split(".")[0]
        if obj_id not in all_obj_ids:
            print(f"Unknown object id: {obj_id}")
            continue

        # flatten the data
        ks = data[0].keys()
        for k in ks:
            if k not in flattened_data.keys():
                flattened_data[k] = []

        for row in tqdm(data):
            batch_size = len(row["chamfer_dist"])
            for k in ks:
                if k == "type":
                    flattened_data[k].extend([row[k]] * batch_size)
                elif k == "chamfer_dist":
                    flattened_data[k].extend(np.sqrt(np.array(row[k])).tolist())
                else:
                    flattened_data[k].extend(row[k])
            flattened_data["object_id"].extend([obj_id] * batch_size)

    # plot 1: chamfer distance box plot w & w/o corrector
    df = pandas.DataFrame.from_dict(flattened_data, orient="columns")
    bp = sns.boxplot(data=df, x="type", y="chamfer_dist")
    bp.set(yscale="log")
    plt.show()

    plt.figure()
    kp = sns.kdeplot(data=df, x="chamfer_dist", hue="type")
    kp.set(xlim=(0, 0.25))
    plt.grid()
    plt.show()

    plt.figure()
    kp = sns.ecdfplot(data=df, x="chamfer_dist", hue="type")
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

    args = parser.parse_args()

    dump_folder = os.path.join(pathlib.Path(__file__).parent.resolve(), "data_rss", args.dataset)

    if "ycbv" in args.dataset:
        all_obj_ids = sorted(list(bop_constants.BOP_MODEL_INDICES["ycbv"].keys()))
    elif "tless" in args.dataset:
        all_obj_ids = sorted(list(bop_constants.BOP_MODEL_INDICES["tless"].keys()))
    else:
        raise NotImplementedError

    # plotting
    # load the data
    plot_data(dump_folder, all_obj_ids=all_obj_ids)
