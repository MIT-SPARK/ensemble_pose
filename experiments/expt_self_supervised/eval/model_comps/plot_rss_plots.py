import copy
import torch

import pandas as pd
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from expt_self_supervised.eval.model_comps.data_utils import load_one_method, cleanup_df, load_self6dpp_results
from utils.visualization_utils import plt_save_figures

import os
from os import listdir
from os.path import isdir, join
import pathlib


plt.rcParams.update({"font.size": 15})

RKN_NAME = "RKN"
CDF_Y_NAME = "Cumulative Distribution of ADD-S Scores"
ENSEMBLE_RKN_LB = f"Ensemble\n-{RKN_NAME} \n(SSL)"
ENSEMBLE_COSYPOSE_LB = "Ensemble\n-CosyPose \n(SSL)"
ENSEMBLE_LB = "Ensemble \n(SSL)"

ENSEMBLE_RKN = f"Ensemble-{RKN_NAME} (SSL)"
ENSEMBLE_COSYPOSE = "Ensemble-CosyPose (SSL)"
ENSEMBLE = "Ensemble (SSL)"
ENSEMBLE_RKN_CERT = f"Ensemble-{RKN_NAME} (SSL, $oc = 1$)"
ENSEMBLE_COSYPOSE_CERT = "Ensemble-CosyPose (SSL, $oc = 1$)"
ENSEMBLE_CERT = "Ensemble (SSL, $oc = 1$)"

CDF_LW = 3

name2legend = {
    "c3po_cosypose_2_refine_c3po_self_sup": ENSEMBLE_RKN,
    "c3po_cosypose_2_refine_cosypose_self_sup": ENSEMBLE_COSYPOSE,
    "c3po_cosypose_2_refine_joint": ENSEMBLE,
    "c3po_cosypose_2_refine_joint_cert": ENSEMBLE_CERT,
    "c3po_cosypose_2_refine_c3po_self_sup_cert": ENSEMBLE_RKN_CERT,
    "c3po_cosypose_2_refine_cosypose_self_sup_cert": ENSEMBLE_COSYPOSE_CERT,
    "c3po_real": f"{RKN_NAME} (Real)",
    "c3po_synth": f"{RKN_NAME} (Synth.)",
    "cosypose_real_2_refine": f"CosyPose (Real)",
    "cosypose_synth_2_refine": f"CosyPose (Synth.)",
    "c3po_synth_w_corrector": f"{RKN_NAME} (Synth., Robust Corrector)",
    "cosypose_synth_2_refine_w_corrector": f"CosyPose (Synth., Robust Corrector)",
    "self6dpp": f"Self6D++ (SSL)",
}

name2color = {
    "c3po_cosypose_2_refine_c3po_self_sup": sns.color_palette("tab10")[4],
    "c3po_cosypose_2_refine_cosypose_self_sup": sns.color_palette("tab10")[1],
    "c3po_cosypose_2_refine_joint": sns.color_palette("tab10")[2],
    "c3po_cosypose_2_refine_joint_cert": sns.color_palette("tab10")[0],
    "c3po_cosypose_2_refine_c3po_self_sup_cert": sns.color_palette("tab10")[5],
    "c3po_cosypose_2_refine_cosypose_self_sup_cert": sns.color_palette("tab10")[6],
    "c3po_real": sns.color_palette("tab10")[7],
    "c3po_synth": sns.color_palette("tab10")[8],
    "cosypose_real_2_refine": sns.color_palette("tab10")[9],
    "cosypose_synth_2_refine": sns.color_palette("tab10")[3],
    "c3po_synth_w_corrector": sns.color_palette("Set2")[0],
    "cosypose_synth_2_refine_w_corrector": sns.color_palette("Set2")[1],
    "self6dpp": sns.color_palette("Set2")[3],
}

name2lines = {
    "c3po_cosypose_2_refine_c3po_self_sup": ":",
    "c3po_cosypose_2_refine_cosypose_self_sup": ":",
    "c3po_cosypose_2_refine_joint": "-",
    "c3po_cosypose_2_refine_joint_cert": "-",
    "c3po_cosypose_2_refine_c3po_self_sup_cert": ":",
    "c3po_cosypose_2_refine_cosypose_self_sup_cert": ":",
    "c3po_real": "--",
    "c3po_synth": "-.",
    "cosypose_real_2_refine": "--",
    "cosypose_synth_2_refine": "-.",
    "c3po_synth_w_corrector": "-",
    "cosypose_synth_2_refine_w_corrector": "-",
    "self6dpp": (5, (10, 1)),
}


def plot_frac_cert_bar_plots_single_figure(df, base_filename="frac_cert_bars", base_folder="./", save_fig=False):
    """Fraction certifiable bar plots"""
    joint_df = df.groupby("name").get_group("c3po_cosypose_2_refine_joint")
    joint_total = joint_df.shape[0]
    joint_frac_cert = joint_df[joint_df["cert"] == True].shape[0] / joint_total
    joint_cert_chamfer_mean = joint_df[joint_df["cert"] == True]["chamfer_mean"].mean()

    c3po_df = df.groupby("name").get_group("c3po_cosypose_2_refine_c3po_self_sup")
    c3po_total = c3po_df.shape[0]
    c3po_frac_cert = c3po_df[c3po_df["cert"] == True].shape[0] / c3po_total
    c3po_cert_chamfer_mean = c3po_df[c3po_df["cert"] == True]["chamfer_mean"].mean()

    cosypose_df = df.groupby("name").get_group("c3po_cosypose_2_refine_cosypose_self_sup")
    cosypose_total = cosypose_df.shape[0]
    cosypose_frac_cert = cosypose_df[cosypose_df["cert"] == True].shape[0] / cosypose_total
    cosypose_cert_chamfer_mean = cosypose_df[cosypose_df["cert"] == True]["chamfer_mean"].mean()

    width = 0.6

    # make bar plot
    plt.rcParams.update({"font.size": 20})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")
    ax1.set_axisbelow(True)
    sns.barplot(
        x=[ENSEMBLE_COSYPOSE_LB, ENSEMBLE_RKN_LB, ENSEMBLE_LB],
        y=np.array([cosypose_cert_chamfer_mean, c3po_cert_chamfer_mean, joint_cert_chamfer_mean]),
        ax=ax1,
        width=width,
    )
    ax1.minorticks_on()
    ax1.set(ylabel="ADD-S Scores")
    ax1.grid(True, which="both", axis="y")
    # ax1.spines[["right", "top"]].set_visible(False)

    ax2.set_axisbelow(True)
    sns.barplot(
        x=[ENSEMBLE_COSYPOSE_LB, ENSEMBLE_RKN_LB, ENSEMBLE_LB],
        y=np.array([cosypose_frac_cert, c3po_frac_cert, joint_frac_cert]) * 100,
        ax=ax2,
        width=width,
    )
    ax2.minorticks_on()
    # ax2.spines[["right", "top"]].set_visible(False)
    sns.despine()
    ax2.set(ylabel="Observably\nCorrect (%)")
    ax2.grid(True, which="both", axis="y")
    plt.rcParams.update({"font.size": 15})

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{base_filename}", base_folder, formats=["pdf", "png"])
    plt.close()


def plot_frac_cert_bar_plots(df, base_filename="frac_cert_bars", base_folder="./", save_fig=False):
    """Fraction certifiable bar plots"""
    joint_df = df.groupby("name").get_group("c3po_cosypose_2_refine_joint")
    joint_total = joint_df.shape[0]
    joint_frac_cert = joint_df[joint_df["cert"] == True].shape[0] / joint_total
    joint_cert_chamfer_mean = joint_df[joint_df["cert"] == True]["chamfer_mean"].mean()

    c3po_df = df.groupby("name").get_group("c3po_cosypose_2_refine_c3po_self_sup")
    c3po_total = c3po_df.shape[0]
    c3po_frac_cert = c3po_df[c3po_df["cert"] == True].shape[0] / c3po_total
    c3po_cert_chamfer_mean = c3po_df[c3po_df["cert"] == True]["chamfer_mean"].mean()

    cosypose_df = df.groupby("name").get_group("c3po_cosypose_2_refine_cosypose_self_sup")
    cosypose_total = cosypose_df.shape[0]
    cosypose_frac_cert = cosypose_df[cosypose_df["cert"] == True].shape[0] / cosypose_total
    cosypose_cert_chamfer_mean = cosypose_df[cosypose_df["cert"] == True]["chamfer_mean"].mean()

    # make bar plot
    plt.figure(figsize=(9, 4))
    ax1 = sns.barplot(
        x=[ENSEMBLE_COSYPOSE_LB, ENSEMBLE_RKN_LB, ENSEMBLE_LB],
        y=np.array([cosypose_cert_chamfer_mean, c3po_cert_chamfer_mean, joint_cert_chamfer_mean]),
    )
    ax1.set_axisbelow(True)
    ax1.minorticks_on()
    ax1.grid(True, which="both", axis="y")
    ax1.set(ylabel="ADD-S Scores")
    ax1.spines[["right", "top"]].set_visible(False)
    plt.tight_layout()
    for item in [ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels():
        item.set_fontsize(25)
    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{base_filename}_adds", base_folder, formats=["pdf", "png"])
    plt.close()

    plt.figure(figsize=(9, 4))
    ax2 = sns.barplot(
        x=[ENSEMBLE_COSYPOSE_LB, ENSEMBLE_RKN_LB, ENSEMBLE_LB],
        y=np.array([cosypose_frac_cert, c3po_frac_cert, joint_frac_cert]) * 100,
    )
    for item in [ax2.title, ax2.xaxis.label, ax2.yaxis.label] + ax2.get_xticklabels() + ax2.get_yticklabels():
        item.set_fontsize(25)
    ax2.set_axisbelow(True)
    ax2.minorticks_on()
    ax2.grid(True, which="both", axis="y")
    ax2.spines[["right", "top"]].set_visible(False)
    ax2.set(ylabel="Observably\nCorrect (%)")
    plt.tight_layout()

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{base_filename}", base_folder, formats=["pdf", "png"])
    plt.close()


def plot_self_sup_cdf(results, base_filename="self_sup_cdf", base_folder="./", only_joint_cert=True, save_fig=False):
    """
    In this CDF, we show 1) joint cert only, 2) joint all, 3) C3PO self sup all 4) Cosypose self sup all
    """
    df = pd.concat(
        [results["c3po_cosypose_2_refine"], results["c3po_synth"], results["cosypose_synth_2_refine"]], axis=0
    )
    records = df.to_dict("records")
    new_records = []
    for row in records:
        new_records.append(row)
        if only_joint_cert:
            if "joint" in row["name"] and row["cert"]:
                new_row = copy.deepcopy(row)
                new_row["name"] = row["name"] + "_cert"
                new_records.append(new_row)
        else:
            if row["cert"]:
                new_row = copy.deepcopy(row)
                new_row["name"] = row["name"] + "_cert"
                new_records.append(new_row)

    df = pd.DataFrame.from_records(new_records)

    plt.figure()

    hue_order = [
        "c3po_cosypose_2_refine_joint_cert",
        "c3po_cosypose_2_refine_joint",
        "c3po_cosypose_2_refine_c3po_self_sup",
        # "c3po_synth",
        "c3po_cosypose_2_refine_cosypose_self_sup",
        # "cosypose_synth_2_refine",
    ]
    if not only_joint_cert:
        hue_order = [
            "c3po_cosypose_2_refine_joint_cert",
            "c3po_cosypose_2_refine_joint",
            "c3po_cosypose_2_refine_c3po_self_sup_cert",
            "c3po_cosypose_2_refine_c3po_self_sup",
            # "c3po_synth",
            "c3po_cosypose_2_refine_cosypose_self_sup_cert",
            "c3po_cosypose_2_refine_cosypose_self_sup",
            # "cosypose_synth_2_refine",
        ]

    ax = sns.ecdfplot(data=df, x="chamfer_mean", hue="name", hue_order=hue_order, palette=name2color, linewidth=CDF_LW)
    ax.spines[["right", "top"]].set_visible(False)
    lss = []
    for i in range(len(ax.legend_.texts)):
        ctxt = ax.legend_.texts[i]._text
        ax.legend_.texts[i].set_text(name2legend[ctxt])
        lss.append(name2lines[ctxt])
    lss = lss[::-1]

    handles = ax.legend_.legendHandles[::-1]
    for line, ls, handle in zip(ax.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)

    ax.set_xscale("log")
    ax.set(xlim=(0.001, 0.8))
    ax.set(xlabel="ADD-S Score")
    ax.set_ylabel(CDF_Y_NAME, fontsize=13)
    sns.move_legend(
        ax,
        "lower right",
        title=None,
        frameon=True,
    )

    plt.grid()
    plt.tight_layout()

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{base_filename}", base_folder, formats=["pdf", "png"])
    plt.close()


def plot_corrector_sim2real_cdf(results, base_filename="baselines_comp_cdf", base_folder="./", save_fig=False):
    """
    Plot c3po_synth w and wo corrector; cosypose synth w and wo corrector; cosypose real
    """
    df = pd.concat(
        [
            results["c3po_real"],
            results["cosypose_real_2_refine"],
            results["c3po_synth"],
            results["cosypose_synth_2_refine"],
            results["c3po_synth_w_corrector"],
            results["cosypose_synth_2_refine_w_corrector"],
        ],
        axis=0,
    )

    records = df.to_dict("records")
    new_records = []
    for row in records:
        new_records.append(row)
        if "joint" in row["name"] and row["cert"]:
            new_row = copy.deepcopy(row)
            new_row["name"] = row["name"] + "_cert"
            new_records.append(new_row)

    df = pd.DataFrame.from_records(new_records)

    hue_order_joint = [
        "c3po_real",
        "c3po_synth_w_corrector",
        "c3po_synth",
        "cosypose_real_2_refine",
        "cosypose_synth_2_refine_w_corrector",
        "cosypose_synth_2_refine",
    ]
    hue_order_c3po = [
        "c3po_real",
        "c3po_synth_w_corrector",
        "c3po_synth",
    ]
    hue_order_cosypose = [
        "cosypose_real_2_refine",
        "cosypose_synth_2_refine_w_corrector",
        "cosypose_synth_2_refine",
    ]

    def make_one_cdf(horder, fname, fs=12):
        plt.figure()
        ax = sns.ecdfplot(data=df, x="chamfer_mean", hue="name", hue_order=horder, palette=name2color, linewidth=CDF_LW)
        ax.spines[["right", "top"]].set_visible(False)
        lss = []
        for i in range(len(ax.legend_.texts)):
            ctxt = ax.legend_.texts[i]._text
            ax.legend_.texts[i].set_text(name2legend[ctxt])
            lss.append(name2lines[ctxt])
        lss = lss[::-1]

        handles = ax.legend_.legendHandles[::-1]
        for line, ls, handle in zip(ax.lines, lss, handles):
            line.set_linestyle(ls)
            handle.set_ls(ls)

        ax.set_xscale("log")
        ax.set(xlim=(0.001, 5))
        ax.set(xlabel="ADD-S Score")
        ax.set(ylabel=CDF_Y_NAME)

        sns.move_legend(
            ax,
            "lower right",
            title=None,
            frameon=True,
            fontsize=fs,
        )

        plt.grid()
        plt.tight_layout()

        if not save_fig:
            plt.show()
        else:
            plt_save_figures(f"{fname}", base_folder, formats=["pdf", "png"])
        plt.close()

    make_one_cdf(hue_order_joint, f"{base_filename}_both")
    make_one_cdf(hue_order_cosypose, f"{base_filename}_cosypose", fs=14)
    make_one_cdf(hue_order_c3po, f"{base_filename}_c3po", fs=14)

    # report statistics
    method_df = df.groupby("name")
    def adds_helper(mname, thres=1e-2):
        obj_chamfer_dists = torch.as_tensor(method_df.get_group(mname)["chamfer_mean"].to_numpy())
        adds_mask = obj_chamfer_dists < thres
        adds = torch.sum(adds_mask) / adds_mask.shape[0]
        return adds

    c3po_real_adds = adds_helper("c3po_real")
    c3po_synth_w_corrector_adds = adds_helper("c3po_synth_w_corrector")
    c3po_synth_adds = adds_helper("c3po_synth")
    cosypose_real_adds = adds_helper("cosypose_real_2_refine")
    cosypose_synth_w_corrector_adds = adds_helper("cosypose_synth_2_refine_w_corrector")
    cosypose_synth_adds = adds_helper("cosypose_synth_2_refine")
    print(f"c3po_real_adds: {c3po_real_adds}")
    print(f"c3po_synth_w_corrector_adds: {c3po_synth_w_corrector_adds}")
    print(f"c3po_synth_adds: {c3po_synth_adds}")
    print(f"cosypose_real_adds: {cosypose_real_adds}")
    print(f"cosypose_synth_w_corrector_adds: {cosypose_synth_w_corrector_adds}")
    print(f"cosypose_synth_adds: {cosypose_synth_adds}")


def plot_baselines_cdf(results, base_filename="baselines_comp_cdf", base_folder="./", save_fig=False):
    """
    In this CDF, we show 1) CosyPose real, 2) CosyPose synth, 3) C3PO Real 4) C3PO synth 5) joint self sup all, 6) joint self sup cert
    """
    df = pd.concat(
        [
            results["c3po_cosypose_2_refine"],
            results["c3po_real"],
            results["cosypose_real_2_refine"],
            results["c3po_synth"],
            results["cosypose_synth_2_refine"],
        ],
        axis=0,
    )
    records = df.to_dict("records")
    new_records = []
    for row in records:
        new_records.append(row)
        if "joint" in row["name"] and row["cert"]:
            new_row = copy.deepcopy(row)
            new_row["name"] = row["name"] + "_cert"
            new_records.append(new_row)

    # handle self6d
    if "self6dpp" in results.keys():
        for obj_id, data in results["self6dpp"].items():
            for x in data:
                new_records.append({"chamfer_mean": x, "name": "self6dpp"})

    df = pd.DataFrame.from_records(new_records)

    if "self6dpp" in results.keys():
        hue_order = [
            "c3po_cosypose_2_refine_joint_cert",
            "c3po_cosypose_2_refine_joint",
            "self6dpp",
            # "c3po_cosypose_2_refine_c3po_self_sup",
            "c3po_real",
            "c3po_synth",
            # "c3po_cosypose_2_refine_cosypose_self_sup",
            "cosypose_real_2_refine",
            "cosypose_synth_2_refine",
        ]
    else:
        hue_order = [
            "c3po_cosypose_2_refine_joint_cert",
            "c3po_cosypose_2_refine_joint",
            # "c3po_cosypose_2_refine_c3po_self_sup",
            "c3po_real",
            "c3po_synth",
            # "c3po_cosypose_2_refine_cosypose_self_sup",
            "cosypose_real_2_refine",
            "cosypose_synth_2_refine",
        ]

    plt.figure()
    ax = sns.ecdfplot(data=df, x="chamfer_mean", hue="name", hue_order=hue_order, palette=name2color, linewidth=CDF_LW)
    ax.spines[["right", "top"]].set_visible(False)
    lss = []
    for i in range(len(ax.legend_.texts)):
        ctxt = ax.legend_.texts[i]._text
        ax.legend_.texts[i].set_text(name2legend[ctxt])
        lss.append(name2lines[ctxt])
    lss = lss[::-1]

    handles = ax.legend_.legendHandles[::-1]
    for line, ls, handle in zip(ax.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)

    ax.set_xscale("log")
    ax.set(xlim=(0.001, 0.8))
    ax.set(xlabel="ADD-S Score")
    ax.set_ylabel(CDF_Y_NAME, fontsize=13)
    sns.move_legend(
        ax,
        "lower right",
        title=None,
        frameon=True,
        fontsize=14,
    )

    plt.grid()
    plt.tight_layout()

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{base_filename}", base_folder, formats=["pdf", "png"])
    plt.close()


if __name__ == "__main__":
    """
    Sample usage:

    """
    print("Plot CDF for all objects' chamfer distances among all methods in test sets")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="specify the dataset.", type=str)
    parser.add_argument("results_folder", help="folder containing all eval data", type=str)
    parser.add_argument("--self6d_data_folder", help="folder containing self6d data", type=str, default=None)
    parser.add_argument(
        "--output_folder",
        help="folder to dump plots",
        type=str,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "plots"),
    )

    args = parser.parse_args()

    if args.dataset == "ycbv":
        joint_selection_strat = "c3po"
    elif args.dataset == "tless":
        joint_selection_strat = "c3po"
    else:
        raise NotImplementedError

    # list of methods to plot
    methods_to_plot = [
        "c3po_real",
        "c3po_synth",
        "cosypose_real_2_refine",
        "cosypose_synth_2_refine",
        "c3po_cosypose_2_refine",
        "c3po_synth_w_corrector",
        #"cosypose_synth_2_refine_w_corrector",
    ]
    if args.dataset == "ycbv":
        methods_to_plot.append("cosypose_synth_2_refine_w_corrector")


    data_folder = args.results_folder
    all_methods_folders = sorted([f for f in listdir(data_folder) if isdir(join(data_folder, f))])

    results = {}
    original_dfs = {}
    for method in all_methods_folders:
        if method in methods_to_plot:
            df = load_one_method(data_folder=os.path.join(data_folder, method, args.dataset), args=args)
            original_dfs[method] = df
            results[method] = cleanup_df(method, df, joint_selection_strat=joint_selection_strat)

    if args.self6d_data_folder is not None:
        print("loading Self6D data folder.")
        self6d_data = load_self6dpp_results(args.self6d_data_folder, args=args)
        results["self6dpp"] = self6d_data
        methods_to_plot.append("self6dpp")

    # corrector closing sim2real gap
    if args.dataset == "ycbv":
        plot_corrector_sim2real_cdf(
            results=results,
            base_filename="corrector_sim2real",
            base_folder=os.path.join(args.output_folder, args.dataset),
            save_fig=True,
        )

    # bar plots
    plot_frac_cert_bar_plots(
        results["c3po_cosypose_2_refine"],
        base_filename="frac_cert_bars",
        base_folder=os.path.join(args.output_folder, args.dataset),
        save_fig=True,
    )

    plot_frac_cert_bar_plots_single_figure(
        results["c3po_cosypose_2_refine"],
        base_filename="frac_cert_bars_w_adds",
        base_folder=os.path.join(args.output_folder, args.dataset),
        save_fig=True,
    )

    # baseline comps
    plot_baselines_cdf(
        results=results,
        base_filename="baselines_comp_cdf",
        base_folder=os.path.join(args.output_folder, args.dataset),
        save_fig=True,
    )

    # plot 2: cdf of self-sup methods
    plot_self_sup_cdf(
        results=results,
        base_filename="self_sup_cdf",
        base_folder=os.path.join(args.output_folder, args.dataset),
        only_joint_cert=True,
        save_fig=True,
    )
    plot_self_sup_cdf(
        results=results,
        base_filename="self_sup_cdf_all",
        base_folder=os.path.join(args.output_folder, args.dataset),
        only_joint_cert=False,
        save_fig=True,
    )
