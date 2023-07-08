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

from tensorboard.backend.event_processing import event_accumulator
from expt_self_supervised.eval.frac_cert_progression.plot_frac_cert import (
    parse_tensorboard,
    smooth_df,
    name2lss,
    name2legend,
)

plt.rcParams.update({"font.size": 16})

name2legend["total_filtered"] = "Ensemble"


def plot_frac_cert_progression(df, base_folder="./", save_fig=True):
    #hue_order = ["c3po_filtered", "csy_filtered", "total_filtered"]
    hue_order = ["total_filtered"]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i in range(len(hue_order)):
        ax.plot(df[hue_order[i]] * 100, label=name2legend[hue_order[i]], ls=name2lss[hue_order[i]], linewidth=3.5)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set(xlabel="Iterations")
    ax.set(ylabel="% Observably Correct")
    ax.set(xlim=(10, 3000))
    ax.set(yscale="linear")
    ax.legend()

    sns.move_legend(
       ax,
       "lower center",
       bbox_to_anchor=(0.5, 1),
       ncol=3,
       title=None,
       frameon=False,
       handletextpad=0.2,
       columnspacing=0.3,
       borderpad=0.2,
    )

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"frac_cert_quali", base_folder, formats=["pdf", "png", "svg"])
    plt.close()

    return


if __name__ == "__main__":
    print("Plotting fraction certifiable progression")
    c_dir = pathlib.Path(__file__).parent.resolve()
    # data_dir = os.path.join(c_dir, "data", "20230130_132935")
    # data_dir = os.path.join(c_dir, "data", "20230130_002818")
    #data_dir = os.path.join(c_dir, "data", "20230130_133428")

    #frac_cert_dirs = {
    #    "csy_2d": os.path.join(data_dir, "FractionCert_train_obj_000008_csy-2d"),
    #    "csy_3d": os.path.join(data_dir, "FractionCert_train_obj_000008_csy-3d"),
    #    "csy": os.path.join(data_dir, "FractionCert_train_obj_000008_cosypose"),
    #    "c3po_2d": os.path.join(data_dir, "FractionCert_train_obj_000008_c3po-2d"),
    #    "c3po_3d": os.path.join(data_dir, "FractionCert_train_obj_000008_c3po-3d"),
    #    "c3po": os.path.join(data_dir, "FractionCert_train_obj_000008_c3po"),
    #    "total": os.path.join(data_dir, "FractionCert_train_obj_000008_total"),
    #}
    #scalar = "FractionCert/train/obj_000008"
    data_dir = os.path.join(c_dir, "data", "20230129_223540")

    obj_id = "obj_000001"

    frac_cert_dirs = {
        "csy_2d": os.path.join(data_dir, f"FractionCert_train_{obj_id}_csy-2d"),
        "csy_3d": os.path.join(data_dir, f"FractionCert_train_{obj_id}_csy-3d"),
        "csy": os.path.join(data_dir, f"FractionCert_train_{obj_id}_cosypose"),
        "c3po_2d": os.path.join(data_dir, f"FractionCert_train_{obj_id}_c3po-2d"),
        "c3po_3d": os.path.join(data_dir, f"FractionCert_train_{obj_id}_c3po-3d"),
        "c3po": os.path.join(data_dir, f"FractionCert_train_{obj_id}_c3po"),
        "total": os.path.join(data_dir, f"FractionCert_train_{obj_id}_total"),
    }
    scalar = "FractionCert/train/obj_000001"

    record_dict = {}
    for key, d_path in frac_cert_dirs.items():
        events_file = os.listdir(d_path)
        event_file = [x for x in events_file if "events.out" in x][0]
        df = parse_tensorboard(os.path.join(d_path, event_file), [scalar])[scalar]
        if "step" not in record_dict.keys():
            record_dict["step"] = df["step"]
        record_dict[key] = df["value"]

    df = pd.DataFrame.from_dict(record_dict, orient="columns")

    # smoothing
    df = smooth_df(df, sigma=15)
    # dfm = filtered_df.melt("step", var_name="cert_type", value_name="cert_frac")

    # plot
    plot_frac_cert_progression(
        df, base_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), "plots"), save_fig=True
    )
