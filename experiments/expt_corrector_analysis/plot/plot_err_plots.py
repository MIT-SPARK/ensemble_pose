import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from pathlib import Path
from tqdm import tqdm

from plot_histograms import load_data
from plot_shared import evaluate_certifier, get_certified_instances
from utils.file_utils import safely_make_folders
from utils.visualization_utils import plt_save_figures

BASE_DIR = Path(__file__).parent.parent.parent.parent


def prepare_cert_failure_distribution_data(certi_naive_failure_modes, certi_corrector_failure_modes):
    """Helper function to prepare failure distribution data"""

    def helper_fun(failure_modes):
        pt_failures_mask = np.asarray(torch.as_tensor(failure_modes["pt"]).cpu())
        kp_failures_mask = np.asarray(torch.as_tensor(failure_modes["kp"]).cpu())

        # AND the mask to get the cases where both fails
        both_failures_mask = np.logical_and(pt_failures_mask, kp_failures_mask)
        pt_only_failure_mask = np.logical_and(pt_failures_mask, np.logical_not(kp_failures_mask))
        kp_only_failure_mask = np.logical_and(np.logical_not(pt_failures_mask), kp_failures_mask)

        both_failures_count = np.array(
            [np.count_nonzero(both_failures_mask[ii, :]) for ii in range(both_failures_mask.shape[0])]
        )
        pt_only_failures_count = np.array(
            [np.count_nonzero(pt_only_failure_mask[ii, :]) for ii in range(pt_only_failure_mask.shape[0])]
        )
        kp_only_failures_count = np.array(
            [np.count_nonzero(kp_only_failure_mask[ii, :]) for ii in range(kp_only_failure_mask.shape[0])]
        )

        total_failures_count = pt_only_failures_count + kp_only_failures_count + both_failures_count
        pt_fractions = pt_only_failures_count / total_failures_count
        kp_fractions = kp_only_failures_count / total_failures_count
        both_failure_fractions = both_failures_count / total_failures_count
        return pt_fractions, kp_fractions, both_failure_fractions

    naive_pt_fractions, naive_kp_fractions, naive_both_fractions = helper_fun(certi_naive_failure_modes)
    corrector_pt_fractions, corrector_kp_fractions, corrector_both_fractions = helper_fun(certi_corrector_failure_modes)

    return dict(
        naive_pt_fractions=naive_pt_fractions,
        naive_kp_fractions=naive_kp_fractions,
        naive_both_fractions=naive_both_fractions,
        corrector_pt_fractions=corrector_pt_fractions,
        corrector_kp_fractions=corrector_kp_fractions,
        corrector_both_fractions=corrector_both_fractions,
    )


def prepare_fraction_certifiable_data(kp_noise_var_range, certi_naive, certi_corrector, certi_icp):
    """Convert certification fraction data to plottable x & y"""
    x = []
    naive_y = []
    corrector_y = []
    icp_y = []
    num_kp_noise_vars = len(kp_noise_var_range)
    for var_i in range(num_kp_noise_vars):
        kp_noise_var = kp_noise_var_range[var_i]

        # naive
        cert_flags_naive = certi_naive[var_i]
        frac_certified_naive = np.count_nonzero(cert_flags_naive) / float(len(cert_flags_naive))

        # corrector
        cert_flags_corrector = certi_corrector[var_i]
        frac_certified_corrector = np.count_nonzero(cert_flags_corrector) / float(len(cert_flags_corrector))

        # icp
        cert_flags_icp = certi_icp[var_i].cpu()
        frac_certified_icp = np.count_nonzero(cert_flags_icp) / float(len(cert_flags_icp))

        x.append(kp_noise_var)
        naive_y.append(frac_certified_naive)
        corrector_y.append(frac_certified_corrector)
        icp_y.append(frac_certified_icp)

    return x, naive_y, corrector_y, icp_y


def plot_cert_failure_distribution(ax, kp_noise_vars, pt_fractions, kp_fractions, both_failure_fractions):
    """Helper plotting function to visualize certification failure distributions"""
    width = 0.35
    ax.bar(kp_noise_vars, pt_fractions, width, label="Chamfer Distance Only Failures")
    ax.bar(kp_noise_vars, kp_fractions, width, bottom=pt_fractions, label="Kpt. Distance Only Failures")
    ax.bar(kp_noise_vars, both_failure_fractions, width, bottom=pt_fractions + kp_fractions, label="Both Failures")

    ax.set_ylabel("Fraction of Total Failures")
    ax.set_xlabel("Kpt. Noise Var.")
    ax.legend(facecolor="white", framealpha=1, frameon=True)
    return


def make_error_hist(ax, data_payload, err_data, **kwargs):
    num_kp_noise_vars = err_data.shape[0]
    for var_i in tqdm(range(num_kp_noise_vars)):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]
        ax.hist(err_data[var_i, :], bins=20, log=False, label=f"Noise Var={kp_noise_var:.2f}", **kwargs)


def plot_error_comparisons(kp_noise_var_range, save_fig=True, base_filename="err_comp", base_folder="./", **kwargs):
    """Plot comparisons of rotation / translation errors between certified and all instances"""
    certi_naive = kwargs["certi_naive"]
    certi_corrector = kwargs["certi_corrector"]
    certi_naive_failure_modes = kwargs["certi_naive_failure_modes"]
    certi_corrector_failure_modes = kwargs["certi_corrector_failure_modes"]

    # load errors from kwargs
    Rerr_naive = kwargs["Rerr_naive"]
    Rerr_corrector = kwargs["Rerr_corrector"]
    Rerr_naive_mean = kwargs["Rerr_naive_mean"]
    Rerr_naive_var = kwargs["Rerr_naive_var"]
    Rerr_naive_certi_mean = kwargs["Rerr_naive_certi_mean"]
    Rerr_naive_certi_var = kwargs["Rerr_naive_certi_var"]
    Rerr_corrector_mean = kwargs["Rerr_corrector_mean"]
    Rerr_corrector_var = kwargs["Rerr_corrector_var"]
    Rerr_corrector_certi_mean = kwargs["Rerr_corrector_certi_mean"]
    Rerr_corrector_certi_var = kwargs["Rerr_corrector_certi_var"]

    terr_naive = kwargs["terr_naive"]
    terr_corrector = kwargs["terr_corrector"]
    terr_naive_mean = kwargs["terr_naive_mean"]
    terr_naive_var = kwargs["terr_naive_var"]
    terr_naive_certi_mean = kwargs["terr_naive_certi_mean"]
    terr_naive_certi_var = kwargs["terr_naive_certi_var"]
    terr_corrector_mean = kwargs["terr_corrector_mean"]
    terr_corrector_var = kwargs["terr_corrector_var"]
    terr_corrector_certi_mean = kwargs["terr_corrector_certi_mean"]
    terr_corrector_certi_var = kwargs["terr_corrector_certi_var"]

    Rerr_naive_var = torch.sqrt(Rerr_naive_var).T
    Rerr_corrector_var = torch.sqrt(Rerr_corrector_var).T
    Rerr_naive_certi_var = torch.sqrt(Rerr_naive_certi_var).T
    Rerr_corrector_certi_var = torch.sqrt(Rerr_corrector_certi_var).T

    terr_naive_var = torch.sqrt(terr_naive_var).T
    terr_corrector_var = torch.sqrt(terr_corrector_var).T
    terr_naive_certi_var = torch.sqrt(terr_naive_certi_var).T
    terr_corrector_certi_var = torch.sqrt(terr_corrector_certi_var).T

    # subplots:
    # 1. rotation error
    # 2. translation error
    # 3. failure modes
    fig, axs = plt.subplots(2, 3, figsize=(12, 12))
    title = (
        f"clamp_thres={kwargs['certifier_cfg']['clamp_threshold']},"
        + f"eps={kwargs['certifier_cfg']['epsilon']},"
        + f"method={kwargs['certifier_cfg']['epsilon_bound_method']},"
    )
    if kwargs["certifier_cfg"]["epsilon_bound_method"] == "quantile":
        title += f"eps_quantile={kwargs['certifier_cfg']['epsilon_quantile']}"
    st = fig.suptitle(title, fontsize="x-large")
    axs[0, 0].errorbar(
        x=kp_noise_var_range,
        y=Rerr_naive_mean,
        yerr=Rerr_naive_var,
        fmt="-x",
        color="black",
        ecolor="gray",
        elinewidth=1,
        capsize=3,
        label="Naive",
    )
    axs[0, 0].errorbar(
        x=kp_noise_var_range,
        y=Rerr_naive_certi_mean,
        yerr=Rerr_naive_certi_var,
        fmt="--o",
        color="grey",
        ecolor="lightgray",
        elinewidth=3,
        capsize=0,
        label="Naive + Certification",
    )
    axs[0, 0].errorbar(
        x=kp_noise_var_range,
        y=Rerr_corrector_mean,
        yerr=Rerr_corrector_var,
        fmt="-x",
        color="red",
        ecolor="salmon",
        elinewidth=1,
        capsize=3,
        label="Corrector",
    )
    axs[0, 0].errorbar(
        x=kp_noise_var_range,
        y=Rerr_corrector_certi_mean,
        yerr=Rerr_corrector_certi_var,
        fmt="--o",
        color="orangered",
        ecolor="salmon",
        elinewidth=3,
        capsize=0,
        label="Corrector + Certification",
    )
    axs[0, 0].legend(loc="upper left")
    axs[0, 0].set_xlabel("Noise Variance Parameter $\sigma$")
    axs[0, 0].set_ylabel("Rotation Error")

    # translation errors
    axs[1, 0].errorbar(
        x=kp_noise_var_range,
        y=terr_naive_mean,
        yerr=terr_naive_var,
        fmt="-x",
        color="black",
        ecolor="gray",
        elinewidth=1,
        capsize=3,
        label="naive",
    )
    axs[1, 0].errorbar(
        x=kp_noise_var_range,
        y=terr_naive_certi_mean,
        yerr=terr_naive_certi_var,
        fmt="--o",
        color="grey",
        ecolor="lightgray",
        elinewidth=3,
        capsize=0,
        label="naive + certification",
    )
    axs[1, 0].errorbar(
        x=kp_noise_var_range,
        y=terr_corrector_mean,
        yerr=terr_corrector_var,
        fmt="-x",
        color="red",
        ecolor="salmon",
        elinewidth=1,
        capsize=3,
        label="corrector",
    )
    axs[1, 0].errorbar(
        x=kp_noise_var_range,
        y=terr_corrector_certi_mean,
        yerr=terr_corrector_certi_var,
        fmt="--o",
        color="salmon",
        ecolor="orangered",
        elinewidth=3,
        capsize=0,
        label="corrector + certification",
    )
    axs[1, 0].legend(loc="upper left")
    axs[1, 0].set_xlabel("Noise Variance Parameter $\sigma$")
    axs[1, 0].set_ylabel("Translation Error")

    # fraction of certifiable
    frac_cert_x, frac_cert_naive_y, frac_cert_corrector_y = prepare_fraction_certifiable_data(
        data_payload, certi_naive, certi_corrector
    )
    axs[0, 1].plot(frac_cert_x, frac_cert_naive_y, "o-", label="Naive")
    axs[0, 1].plot(frac_cert_x, frac_cert_corrector_y, "x-", label="Corrector")
    axs[0, 1].set_title("Fraction Certifiable")
    axs[0, 1].set_xlabel("Kpt. Noise Var")
    axs[0, 1].set_ylabel("Fraction of Certifiable")
    axs[0, 1].legend(loc="upper right")

    # fraction of failure modes
    cert_failures_data = prepare_cert_failure_distribution_data(
        certi_naive_failure_modes, certi_corrector_failure_modes
    )
    # corrector failure modes
    plot_cert_failure_distribution(
        axs[1, 1],
        kp_noise_var_range,
        cert_failures_data["corrector_pt_fractions"],
        cert_failures_data["corrector_kp_fractions"],
        cert_failures_data["corrector_both_fractions"],
    )

    # error distribution
    make_error_hist(axs[0, 2], data_payload, Rerr_corrector)
    axs[0, 2].set_title("Rot. Err")
    axs[0, 2].legend(loc="upper right")

    make_error_hist(axs[1, 2], data_payload, terr_corrector)
    axs[1, 2].set_title("Trs. Err")
    axs[1, 2].legend(loc="upper right")

    # title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
    # plt.title(title)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # rand_string = generate_filename()
    # filename = fig_save_prefix[:-7] + "_rotation_error_plot_" + timestamp + "_" + rand_string + ".jpg"
    # fig.savefig(filename)
    # plt.close(fig)

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(base_filename, base_folder)
    plt.close()

    return


def plot_adds_comparisons(kp_noise_var_range, save_fig=True, base_filename="adds_comp", base_folder="./", **kwargs):
    """Plot ADD-S comparison plots"""
    certi_naive = kwargs["certi_naive"]
    certi_corrector = kwargs["certi_corrector"]
    certi_naive_failure_modes = kwargs["certi_naive_failure_modes"]
    certi_corrector_failure_modes = kwargs["certi_corrector_failure_modes"]

    # load errors from kwargs
    chamfer_pose_corrected_to_gt_pose = kwargs["chamfer_pose_corrected_to_gt_pose"]
    chamfer_metric_corrected_var = kwargs["chamfer_metric_corrected_var"]
    chamfer_metric_corrected_mean = kwargs["chamfer_metric_corrected_mean"]
    chamfer_metric_corrected_certi_var = kwargs["chamfer_metric_corrected_certi_var"]
    chamfer_metric_corrected_certi_mean = kwargs["chamfer_metric_corrected_certi_mean"]

    chamfer_pose_naive_to_gt_pose = kwargs["chamfer_pose_naive_to_gt_pose"]
    chamfer_metric_naive_var = kwargs["chamfer_metric_naive_var"]
    chamfer_metric_naive_mean = kwargs["chamfer_metric_naive_mean"]
    chamfer_metric_naive_certi_var = kwargs["chamfer_metric_naive_certi_var"]
    chamfer_metric_naive_certi_mean = kwargs["chamfer_metric_naive_certi_mean"]

    # to standard deviations
    chamfer_metric_naive_certi_var = torch.sqrt(chamfer_metric_naive_certi_var).T
    chamfer_metric_naive_var = torch.sqrt(chamfer_metric_naive_var).T
    chamfer_metric_corrected_certi_var = torch.sqrt(chamfer_metric_corrected_certi_var).T
    chamfer_metric_corrected_var = torch.sqrt(chamfer_metric_corrected_var).T

    # subplots:
    # 1. add-s errors
    # 2. certifiable fraction vs. kp var
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax_adds = axs[0]
    ax_frac_cert = axs[1]
    title = (
        f"{kwargs['object_label']} dia.={kwargs['object_diameter']:.2f}, "
        + f"clamp_thres={kwargs['certifier_cfg']['clamp_threshold']}, "
        + f"eps={kwargs['certifier_cfg']['epsilon']}, "
        + f"method={kwargs['certifier_cfg']['epsilon_bound_method']}, "
    )
    if kwargs["certifier_cfg"]["epsilon_bound_method"] == "quantile":
        title += f"eps_quantile={kwargs['certifier_cfg']['epsilon_quantile']}"
    st = fig.suptitle(title, fontsize="x-large")
    ax_adds.errorbar(
        x=kp_noise_var_range,
        y=chamfer_metric_naive_mean,
        yerr=chamfer_metric_naive_var,
        fmt="-x",
        color="black",
        ecolor="gray",
        elinewidth=1,
        capsize=3,
        label="naive",
    )
    ax_adds.errorbar(
        x=kp_noise_var_range,
        y=chamfer_metric_naive_certi_mean,
        yerr=chamfer_metric_naive_certi_var,
        fmt="--o",
        color="grey",
        ecolor="lightgray",
        elinewidth=3,
        capsize=0,
        label="naive + certification",
    )
    ax_adds.errorbar(
        x=kp_noise_var_range,
        y=chamfer_metric_corrected_mean,
        yerr=chamfer_metric_corrected_var,
        fmt="-x",
        color="red",
        ecolor="salmon",
        elinewidth=1,
        capsize=3,
        label="corrector",
    )
    ax_adds.errorbar(
        x=kp_noise_var_range,
        y=chamfer_metric_corrected_certi_mean,
        yerr=chamfer_metric_corrected_certi_var,
        fmt="--o",
        color="orangered",
        ecolor="salmon",
        elinewidth=3,
        capsize=0,
        label="corrector + certification",
    )
    ax_adds.legend(loc="upper left")
    ax_adds.set_xlabel("Noise variance parameter $\sigma$")
    ax_adds.set_ylabel("Normalized ADD-S")

    # fraction of certifiable
    frac_cert_x, frac_cert_naive_y, frac_cert_corrector_y = prepare_fraction_certifiable_data(
        kp_noise_var_range, certi_naive, certi_corrector
    )
    ax_frac_cert.plot(frac_cert_x, frac_cert_naive_y, "o-", label="Naive")
    ax_frac_cert.plot(frac_cert_x, frac_cert_corrector_y, "x-", label="Corrector")
    ax_frac_cert.set_title("Fraction Certifiable")
    ax_frac_cert.set_xlabel("Kpt. Noise Var")
    ax_frac_cert.set_ylabel("Fraction of Certifiable")
    ax_frac_cert.legend(loc="upper right")

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(base_filename, base_folder)
    plt.close()

    return


if __name__ == "__main__":
    print("Generate error plots for corrector analysis.")
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datafile",
        help="path of the data file wrt to repo root",
        default="local_data/corrector_analysis/ycbv.train.real/obj_000006/20221009_000424_torch-gd-accel_experiment.pickle",
        type=str,
    )

    parser.add_argument(
        "--save_folder",
        help="path to the folder in which we save figures wrt to experiment folder root",
        default="figures",
        type=str,
    )

    parser.add_argument("--object_label", help="object label in the dataset", default="obj_000006", type=str)

    args = parser.parse_args()
    print("CLI args: ")
    print(args)

    exp_folder_path = Path(__file__).parent.parent.resolve()
    dump_path = os.path.join(exp_folder_path, args.save_folder)
    print(f"Dump path: {dump_path}")
    safely_make_folders([dump_path])

    # load data
    filename = BASE_DIR / args.datafile
    data_payload = load_data(filename)

    # run certifier and compare errors
    certifier_cfg = dict(
        epsilon=0.04,
        epsilon_bound_method="quantile",
        epsilon_quantile=0.9,
        epsilon_type="relative",
        # outlier clamping
        clamp_method="fixed",
        clamp_threshold=0.1,
    )
    (
        certi_naive,
        certi_corrector,
        certi_naive_failure_modes,
        certi_corrector_failure_modes,
        _,
        _,
    ) = get_certified_instances(payload=data_payload, certifier_cfg=certifier_cfg)
    certi_masks = {"certi_naive": certi_naive.cpu().numpy(), "certi_corrector": certi_corrector.cpu().numpy()}

    results_data = evaluate_certifier(data_payload=data_payload, certi_masks=certi_masks)

    plot_error_comparisons(
        data_payload["kp_noise_var_range"],
        certi_naive_failure_modes=certi_naive_failure_modes,
        certi_corrector_failure_modes=certi_corrector_failure_modes,
        certifier_cfg=certifier_cfg,
        save_fig=True,
        base_filename=f"err_comp_{args.object_label}",
        base_folder=os.path.join(dump_path, "err_plots"),
        **results_data,
    )

    plot_adds_comparisons(
        data_payload["kp_noise_var_range"],
        certi_naive_failure_modes=certi_naive_failure_modes,
        certi_corrector_failure_modes=certi_corrector_failure_modes,
        certifier_cfg=certifier_cfg,
        save_fig=True,
        base_filename=f"adds_comp_{args.object_label}",
        base_folder=os.path.join(dump_path, "adds_plots"),
        **results_data,
    )
