import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import seaborn as sns
import torch.utils.data
from scipy.ndimage.filters import gaussian_filter1d

from datasets.bop_constants import BOP_MODEL_INDICES
from utils.visualization_utils import plt_save_figures
from expt_self_supervised.eval.frac_cert_progression.load_tb_data import *

from tensorboard.backend.event_processing import event_accumulator

plt.rcParams.update({"font.size": 16})


name2legend = {
    "csy_2d_filtered": "CosyPose ($oc_{2D} = 1$)",
    "csy_3d_filtered": "CosyPose ($oc_{3D} = 1$)",
    "csy_filtered": "CosyPose ($oc = 1$)",
    "c3po_2d_filtered": "RKN ($oc_{2D} = 1$)",
    "c3po_3d_filtered": "RKN ($oc_{3D} = 1$)",
    "c3po_filtered": "RKN ($oc = 1$)",
    "total_filtered": "Ensemble ($oc = 1$)",
    "total": "Ensemble ($oc = 1$)",
}

name2lss = {
    "csy_2d_filtered": ":",
    "csy_3d_filtered": "--",
    "csy_filtered": "-",
    "c3po_2d_filtered": ":",
    "c3po_3d_filtered": "--",
    "c3po_filtered": "-",
    "total_filtered": "-",
    "total": "-",
}


def plot_bar_plot(data, base_filename="frac_cert_increase_bars", base_folder="./", save_fig=False):
    X = ["CosyPose\n(Refine)", "CosyPose\n(Coarse)", "RKN"]
    width = 0.6

    plt.minorticks_on()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), layout="constrained", sharey=False)
    ax1.set_axisbelow(True)
    sns.barplot(
        x=X,
        y=np.array([data["csy_2d_filtered"], data["csy_coarse_2d_filtered"], data["c3po_2d_filtered"]]) * 100,
        ax=ax1,
        width=width,
    )
    ax1.minorticks_on()
    ax1.set(title="$oc_{2D} = 1$ ")
    ax1.set(ylabel="Improvement (%)")
    ax1.spines[["right", "top"]].set_visible(False)
    ax1.grid(True, which="both", axis="y")

    ax2.set_axisbelow(True)
    sns.barplot(
        x=X,
        y=np.array([data["csy_3d_filtered"], data["csy_coarse_3d_filtered"], data["c3po_3d_filtered"]]) * 100,
        ax=ax2,
        width=width,
    )
    ax2.set(title="$oc_{3D} = 1$ ")
    ax2.minorticks_on()
    ax2.spines[["right", "top"]].set_visible(False)
    ax2.grid(True, which="both", axis="y")

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{base_filename}", base_folder, formats=["pdf", "png"])
    plt.close()

    return


if __name__ == "__main__":
    """
    Usage example:
    python eval/frac_cert_progression/plot_frac_cert_bar_plot.py
    """
    dataset = "ycbv"
    all_obj_ids = BOP_MODEL_INDICES[dataset].keys()
    c_dir = pathlib.Path(__file__).parent.resolve()

    # load logs
    # 2 refine
    logs_dir_2_refine = os.path.join(c_dir, "2_refine_all_logs", "ycbv")
    obj_tb_db_2_refine = get_data_dirs(logs_dir_2_refine, all_obj_ids)
    all_objs_dfs_2_refine = build_dfs(dataset, obj_tb_db=obj_tb_db_2_refine)

    # 0 refine
    logs_dir_0_refine = os.path.join(c_dir, "0_refine_all_logs", "ycbv")
    obj_tb_db_0_refine = get_data_dirs(logs_dir_0_refine, all_obj_ids)
    all_objs_dfs_0_refine = build_dfs(dataset, obj_tb_db=obj_tb_db_0_refine)

    # calculate improvements
    improvements = {
        "csy_2d_2_refine": None,
        "csy_3d_2_refine": None,
        "c3po_2d": None,
        "c3po_3d": None,
    }
    running_abs_improvements = None
    running_rel_improvements = None
    for obj_id in all_obj_ids:
        c_df = all_objs_dfs_2_refine[obj_id]

        csy_coarse = all_objs_dfs_0_refine[obj_id]
        c_df["csy_coarse_2d_filtered"] = csy_coarse["csy_2d_filtered"]
        c_df["csy_coarse_3d_filtered"] = csy_coarse["csy_3d_filtered"]
        c_df["csy_coarse_filtered"] = csy_coarse["csy_filtered"]

        abs_improvements = c_df.iloc[-1].subtract(c_df.iloc[0])
        rel_improvements = c_df.iloc[-1].subtract(c_df.iloc[0]).div(c_df.iloc[0])
        rel_improvements.replace(np.nan, 0)

        if running_abs_improvements is None:
            running_abs_improvements = abs_improvements
        else:
            running_abs_improvements += abs_improvements

        if running_rel_improvements is None:
            running_rel_improvements = rel_improvements
        else:
            running_rel_improvements += rel_improvements

    avg_abs_improvements = running_abs_improvements / len(all_obj_ids)
    avg_rel_improvements = running_rel_improvements / len(all_obj_ids)

    plot_bar_plot(
        avg_abs_improvements,
        base_filename="frac_cert_increase_bars",
        base_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), "plots"),
        save_fig=True,
    )
    # plot_bar_plot(avg_rel_improvements)
