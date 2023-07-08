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


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar
    Credit: https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically
    """
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(s in ea.Tags()["scalars"] for s in scalars), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


def smooth_df(df, sigma=20):
    cols_to_filter = ["csy_2d", "csy_3d", "csy", "c3po_2d", "c3po_3d", "c3po", "total"]
    for col in cols_to_filter:
        filtered_col = gaussian_filter1d(df[col], sigma=sigma)
        df[col + "_filtered"] = filtered_col
    return df


def plot_frac_cert_progression(
    df, base_folder="./", basename="frac_cert", save_fig=True, yscale="linear", yfactor=1, ylabel="% Observably Correct"
):
    # line plot
    # c3po 2d, 3d, total
    # cosypose 2d, 3d, total
    hue_order = [
        "csy_filtered",
        "c3po_filtered",
        "csy_2d_filtered",
        "c3po_2d_filtered",
        "csy_3d_filtered",
        "c3po_3d_filtered",
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(hue_order)):
        ax.plot(df[hue_order[i]] * yfactor, label=name2legend[hue_order[i]], ls=name2lss[hue_order[i]])
    ax.spines[["right", "top"]].set_visible(False)
    ax.set(xlabel="Iterations")
    ax.set(ylabel=ylabel)
    ax.set(xlim=(0, 3000))
    ax.set(yscale=yscale)
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
        plt_save_figures(basename, base_folder, formats=["pdf", "png"])
    plt.close()

    return


if __name__ == "__main__":
    print("Plotting fraction certifiable progression")
    c_dir = pathlib.Path(__file__).parent.resolve()
    # data_dir = os.path.join(c_dir, "data", "20230130_132935")
    # data_dir = os.path.join(c_dir, "data", "20230130_002818")
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
    df = smooth_df(df)
    # dfm = filtered_df.melt("step", var_name="cert_type", value_name="cert_frac")

    ## plot
    plot_frac_cert_progression(
        df,
        base_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), "plots"),
        save_fig=True,
        yfactor=100,
        yscale="linear",
    )

    rel_df = df.subtract(df.iloc[0], axis="columns")
    rel_df = rel_df.div(df.iloc[0], axis="columns")
    plot_frac_cert_progression(
        rel_df,
        basename="rel_frac_cert",
        base_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), "plots"),
        save_fig=True,
        yfactor=100,
        yscale="linear",
        ylabel="% Rel. Increase in % Observably Correct"
    )
