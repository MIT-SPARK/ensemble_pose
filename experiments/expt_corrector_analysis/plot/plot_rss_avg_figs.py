import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import pandas as pd
from matplotlib import colors as mcolors
from pathlib import Path
from tqdm import tqdm

from plot_err_plots import prepare_fraction_certifiable_data
from plot_shared import load_data, evaluate_certifier, get_certified_instances
from utils.file_utils import safely_make_folders
from utils.general import generate_filename
from utils.visualization_utils import plt_save_figures
from datasets.bop_constants import *

# from expt_corrector_analysis.plot.plot_rss_figs import plot_adds, plot_frac_cert

BASE_DIR = Path(__file__).parent.parent.parent.parent
plt.rcParams.update({"font.size": 12})
label_font_size=15

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
NO_CORRECTOR_COLOR = COLORS[0]
ROBUST_CORRECTOR_COLOR = COLORS[1]
ROBUST_CORRECTOR_CERT_COLOR = COLORS[1]
ICP_COLOR = COLORS[3]

def plot_all_objs_add_s(adds_data, save_fig=True, adds_base_filename="adds_comp", base_folder="./"):
    df = pd.DataFrame.from_records(adds_data)

    # naive
    naive_chamfer = df.groupby("type").get_group("naive").groupby("noise_var")["chamfer_metric"]

    # naive + cert
    naive_cert_chamfer = (
        df.groupby("type")
        .get_group("naive")
        .groupby("certified")
        .get_group(True)
        .groupby("noise_var")["chamfer_metric"]
    )

    # corrector
    corrector_chamfer = df.groupby("type").get_group("corrector").groupby("noise_var")["chamfer_metric"]

    # corrector + cert
    corrector_cert_chamfer = (
        df.groupby("type")
        .get_group("corrector")
        .groupby("certified")
        .get_group(True)
        .groupby("noise_var")["chamfer_metric"]
    )

    # icp
    icp_chamfer = df.groupby("type").get_group("icp").groupby("noise_var")["chamfer_metric"]

    # icp + cert
    icp_cert_chamfer = (
        df.groupby("type")
        .get_group("icp")
        .groupby("certified")
        .get_group(True)
        .groupby("noise_var")["chamfer_metric"]
    )

    # fig_adds, ax_adds = plt.subplots(figsize=(7, 5))
    fig_adds, ax_adds = plt.subplots()
    ax_adds.errorbar(
        x=naive_chamfer.mean().index,
        y=naive_chamfer.mean(),
        yerr=naive_chamfer.std(),
        fmt="-o",
        color=NO_CORRECTOR_COLOR,
        ecolor=NO_CORRECTOR_COLOR,
        elinewidth=1,
        capsize=3,
        label="Naive",
    )
    #ax_adds.errorbar(
    #   x=naive_cert_chamfer.mean().index,
    #   y=naive_cert_chamfer.mean(),
    #   yerr=naive_cert_chamfer.std(),
    #   fmt="--o",
    #   color=NO_CORRECTOR_COLOR,
    #   ecolor=NO_CORRECTOR_COLOR,
    #   elinewidth=2,
    #   capsize=3,
    #   label="Naive ($oc$ = 1)",
    #)
    ax_adds.errorbar(
        x=icp_chamfer.mean().index,
        y=icp_chamfer.mean(),
        yerr=icp_chamfer.std(),
        fmt="-^",
        color=ICP_COLOR,
        ecolor=ICP_COLOR,
        elinewidth=1,
        capsize=3,
        label="Naive + ICP",
    )
    ax_adds.errorbar(
        x=icp_cert_chamfer.mean().index,
        y=icp_cert_chamfer.mean(),
        yerr=icp_cert_chamfer.std(),
        fmt="--^",
        color=ICP_COLOR,
        ecolor=ICP_COLOR,
        elinewidth=2,
        capsize=3,
        label="Naive + ICP ($oc$ = 1)",
    )
    ax_adds.errorbar(
        x=corrector_chamfer.mean().index,
        y=corrector_chamfer.mean(),
        yerr=corrector_chamfer.std(),
        fmt="-x",
        color=ROBUST_CORRECTOR_COLOR,
        ecolor=ROBUST_CORRECTOR_COLOR,
        elinewidth=1,
        capsize=3,
        label="Robust Corrector",
    )
    ax_adds.errorbar(
        x=corrector_cert_chamfer.mean().index,
        y=corrector_cert_chamfer.mean(),
        yerr=corrector_cert_chamfer.std(),
        fmt="--x",
        color=ROBUST_CORRECTOR_CERT_COLOR,
        ecolor=ROBUST_CORRECTOR_CERT_COLOR,
        elinewidth=2,
        capsize=3,
        label="Robust Corrector ($oc$ = 1)",
    )
    ax_adds.legend(loc="upper left", facecolor="white", framealpha=1, frameon=True)
    ax_adds.set_xlabel("Noise Parameter $\sigma$", fontsize=label_font_size)
    ax_adds.set_ylabel("Normalized ADD-S Score", fontsize=label_font_size)

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{adds_base_filename}_all_objects", base_folder, dpi=200)
    plt.close()

    return


def plot_frac_cert_shaded_line(frac_cert_data, base_folder="./", base_filename="frac_cert", save_fig=False):
    """Shaded line plot"""
    df = pd.DataFrame.from_records(frac_cert_data)

    def get_x_y_err_band(y_name):
        y_mean = df.groupby("frac_cert_x").mean()[y_name]
        frac_cert_x = y_mean.index

        y_std = df.groupby("frac_cert_x").std()[y_name]
        lower = y_mean - y_std
        upper = y_mean + y_std
        return frac_cert_x, y_mean, lower, upper

    frac_cert_x, naive_y_mean, naive_y_lower, naive_y_upper = get_x_y_err_band("frac_cert_naive_y")
    _, corrector_y_mean, corrector_y_lower, corrector_y_upper = get_x_y_err_band("frac_cert_corrector_y")
    _, icp_y_mean, icp_y_lower, icp_y_upper = get_x_y_err_band("frac_cert_icp_y")

    # fig, ax = plt.subplots(figsize=(7, 5))
    fig, ax = plt.subplots()
    ax.plot(frac_cert_x, naive_y_mean * 100, "o-", label="Naive", color=NO_CORRECTOR_COLOR)
    ax.plot(frac_cert_x, naive_y_upper * 100, alpha=0.1, color=NO_CORRECTOR_COLOR)
    naive_y_lower[naive_y_lower < 0] = 0
    ax.plot(frac_cert_x, naive_y_lower * 100, alpha=0.1, color=NO_CORRECTOR_COLOR)
    ax.fill_between(frac_cert_x, naive_y_lower * 100, naive_y_upper * 100, alpha=0.1, color=NO_CORRECTOR_COLOR)

    ax.plot(frac_cert_x, icp_y_mean * 100, "x-", label="Naive + ICP", color=ICP_COLOR)
    ax.plot(frac_cert_x, icp_y_lower * 100, alpha=0.1, color=ICP_COLOR)
    icp_y_upper[icp_y_upper > 1] = 1
    ax.plot(frac_cert_x, icp_y_upper * 100, alpha=0.1, color=ICP_COLOR)
    ax.fill_between(frac_cert_x, icp_y_lower * 100, icp_y_upper * 100, alpha=0.1, color=ICP_COLOR)

    ax.plot(frac_cert_x, corrector_y_mean * 100, "x-", label="Robust Corrector", color=ROBUST_CORRECTOR_COLOR)
    ax.plot(frac_cert_x, corrector_y_lower * 100, alpha=0.1, color=ROBUST_CORRECTOR_COLOR)
    corrector_y_upper[corrector_y_upper > 1] = 1
    ax.plot(frac_cert_x, corrector_y_upper * 100, alpha=0.1, color=ROBUST_CORRECTOR_COLOR)
    ax.fill_between(frac_cert_x, corrector_y_lower * 100, corrector_y_upper * 100, alpha=0.1, color=ROBUST_CORRECTOR_COLOR)

    ax.set_xlabel("Noise Parameter $\sigma$", fontsize=label_font_size)
    ax.set_ylabel("Observably Correct (%)", fontsize=label_font_size)
    ax.legend(loc="upper right", facecolor="white", frameon=True)
    ax.set_ylim([0, 100])

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{base_filename}_all_objs", base_folder, dpi=200)
    plt.close()

    return


def main_plot(obj_ids, datafiles, base_folder):
    # data
    frac_cert_data = []
    add_s_data = []
    for obj_id, datafile in zip(obj_ids, datafiles):
        # load data
        filename = BASE_DIR / datafile
        data_payload = load_data(filename)

        # run certifier and compare errors
        certifier_cfg = dict(
            epsilon=0.04,
            epsilon_bound_method="quantile",
            epsilon_quantile=0.96,
            epsilon_type="relative",
            # outlier clamping
            clamp_method="fixed",
            clamp_threshold=0.3,
        )
        data_payload["chamfer_clamp_thres_factor"] = certifier_cfg["clamp_threshold"]
        (
            certi_naive,
            certi_corrector,
            certi_icp,
            certi_naive_failure_modes,
            certi_corrector_failure_modes,
            certi_icp_failure_modes,
            certifier_eps_used,
            certifier_clamp_threshold_used,
        ) = get_certified_instances(payload=data_payload, certifier_cfg=certifier_cfg)
        certi_masks = {"certi_naive": certi_naive.cpu().numpy(),
                       "certi_corrector": certi_corrector.cpu().numpy(),
                       "certi_icp": certi_icp.cpu().numpy(),
                       }

        results_data = evaluate_certifier(data_payload=data_payload, certi_masks=certi_masks)
        kp_noise_var_range = data_payload["kp_noise_var_range"]

        # save add-s data
        # need to have:
        # naive, naive + certification, corrector, corrector + certification
        for i in range(kp_noise_var_range.shape[0]):
            noise_var = kp_noise_var_range[i]
            # icp
            for j in range(results_data["chamfer_pose_icp_to_gt_pose"].shape[1]):
                add_s_data.append(
                    {
                        "noise_var": noise_var,
                        "type": "icp",
                        "chamfer_metric": results_data["chamfer_pose_icp_to_gt_pose"][i, j].to("cpu").item(),
                        "certified": certi_masks["certi_icp"][i, j],
                    }
                )
            # naive
            for j in range(results_data["chamfer_pose_naive_to_gt_pose"].shape[1]):
                add_s_data.append(
                    {
                        "noise_var": noise_var,
                        "type": "naive",
                        "chamfer_metric": results_data["chamfer_pose_naive_to_gt_pose"][i, j].to("cpu").item(),
                        "certified": certi_masks["certi_naive"][i, j],
                    }
                )
            # corrector
            for j in range(results_data["chamfer_pose_corrected_to_gt_pose"].shape[1]):
                add_s_data.append(
                    {
                        "noise_var": noise_var,
                        "type": "corrector",
                        "chamfer_metric": results_data["chamfer_pose_corrected_to_gt_pose"][i, j].to("cpu").item(),
                        "certified": certi_masks["certi_corrector"][i, j],
                    }
                )

        # save frac cert data
        certi_naive = results_data["certi_naive"]
        certi_corrector = results_data["certi_corrector"]
        frac_cert_x, frac_cert_naive_y, frac_cert_corrector_y, frac_cert_icp_y = prepare_fraction_certifiable_data(
            kp_noise_var_range, certi_naive, certi_corrector, certi_icp,
        )
        for x, naive_y, corrector_y, icp_y in zip(frac_cert_x, frac_cert_naive_y, frac_cert_corrector_y,
                                                  frac_cert_icp_y):
            frac_cert_data.append(
                {
                    "frac_cert_x": x,
                    "frac_cert_naive_y": naive_y,
                    "frac_cert_corrector_y": corrector_y,
                    "frac_cert_icp_y": icp_y,
                }
            )

    plot_all_objs_add_s(
        add_s_data,
        save_fig=True,
        adds_base_filename="adds_comp",
        base_folder=base_folder,
    )

    # avg frac cert plot for all objects
    plot_frac_cert_shaded_line(
        frac_cert_data,
        save_fig=True,
        base_filename="frac_cert",
        base_folder=base_folder,
    )

    return


def plot_ycbv():
    ycbv_object_ids = YCBV.keys()
    save_folder = f"./figures/ycbv"
    datafiles_folder = "/opt/project/CASPER-3D/local_data/corrector_analysis/ycbv.train.real/"
    datafiles_ycbv = []
    for sub_dir in ycbv_object_ids:
        cf_path = os.path.join(datafiles_folder, sub_dir)
        cpkl_files = sorted(
            [f for f in os.listdir(cf_path) if os.path.isfile(os.path.join(cf_path, f)) and ".pickle" in f])
        datafiles_ycbv.append(os.path.join(cf_path, cpkl_files[-1]))

    exp_folder_path = Path(__file__).parent.parent.resolve()
    dump_path = os.path.join(exp_folder_path, save_folder)
    print(f"Dump path: {dump_path}")
    base_folder = os.path.join(dump_path, "rss_plots")
    safely_make_folders([base_folder])
    main_plot(ycbv_object_ids, datafiles_ycbv, base_folder)
    return


def plot_tless():
    save_folder = f"./figures/tless"
    datafiles_tless = [
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000001/20221017_231302_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000002/20221017_231727_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000003/20221017_232153_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000004/20221017_232620_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000005/20221017_233046_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000006/20221017_233510_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000007/20221017_233937_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000008/20221017_234405_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000009/20221017_234831_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000010/20221017_235256_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000011/20221017_235720_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000012/20221018_000145_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000013/20221018_000613_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000014/20221018_001039_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000015/20221018_001504_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000016/20221018_001929_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000017/20221018_002355_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000018/20221018_002821_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000019/20221018_003245_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000020/20221018_003711_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000021/20221018_004135_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000022/20221018_004600_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000023/20221018_005025_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000024/20221018_005449_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000025/20221018_005914_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000026/20221018_010339_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000027/20221018_010804_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000028/20221018_011229_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000029/20221018_011654_torch-gd-accel_experiment.pickle",
        "/opt/project/CASPER-3D/local_data/corrector_analysis/tless.primesense.train/obj_000030/20221018_012119_torch-gd-accel_experiment.pickle",
    ]

    tless_object_ids = TLESS.keys()
    exp_folder_path = Path(__file__).parent.parent.resolve()
    dump_path = os.path.join(exp_folder_path, save_folder)
    print(f"Dump path: {dump_path}")
    base_folder = os.path.join(dump_path, "rss_plots")
    safely_make_folders([base_folder])
    main_plot(tless_object_ids, datafiles_tless, base_folder)

    return


if __name__ == "__main__":
    print("Plot ADD-S error, fraction certifiable and histogram of chamfer distances.")
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    plot_ycbv()
    # plot_tless()
