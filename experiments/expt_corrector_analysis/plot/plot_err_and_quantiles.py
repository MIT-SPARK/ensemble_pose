import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from matplotlib import colors as mcolors
from pathlib import Path
from tqdm import tqdm

from plot_err_plots import prepare_fraction_certifiable_data
from plot_shared import load_data, evaluate_certifier, get_certified_instances
from utils.file_utils import safely_make_folders
from utils.general import generate_filename
from utils.visualization_utils import plt_save_figures

plt.style.use("seaborn-whitegrid")
COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
BASE_DIR = Path(__file__).parent.parent.parent.parent


def plot_instance_histogram(
    sqdists,
    quantile=0.9,
    abs_sq_epsilon=10,
    sq_clamp_threshold=10,
    save_folder="./",
    save_fig=True,
    file_basename="instance_histogram",
    title="",
):
    """Plot relevant histograms for a single instance

    Args:
        sqdists:
        quantile: draw a vertical line at this quantile among the data
        abs_epsilon: absolute epsilon value used in the certifier
        save_folder:
        save_fig:
        file_basename:
        title:
    """
    fig_quant, ax_quant = plt.subplots()
    # eps line
    ax_quant.axvline(x=abs_sq_epsilon, color="red", label="$\epsilon^2$")

    # quantile line
    clamped_sqdists = sqdists[sqdists < sq_clamp_threshold]
    data_quantile_val = np.quantile(clamped_sqdists, quantile, method="inverted_cdf")
    ax_quant.axvline(x=data_quantile_val, color="lime", label=f"{quantile} Quantile")

    # histogram
    ax_quant.hist(clamped_sqdists, bins=100, log=True)
    ax_quant.set_xlabel("Squared Chamfer Distances")
    ax_quant.set_ylabel("Counts")
    ax_quant.grid(True)
    ax_quant.set_title(title)
    ax_quant.legend()

    if not save_fig:
        plt.show()
        plt.close()
    else:
        plt_save_figures(file_basename, save_folder)
        plt.close()


def plot_sample_quantile_histograms(
    data_payload,
    certi_masks,
    num_random_samples=1,
    certifier_sq_epsilon=10,
    certifier_sq_clamp_threshold=10,
    certifier_quantile=0.9,
    save_fig=True,
    base_filename="sample_sqdist_histogram",
    base_folder="./",
):
    def gen_rand_indices_or_all(data_array, n):
        """Sample random indices; if not enough points, return all indices"""
        if len(data_array) > n:
            return np.random.choice(data_array, n)
        else:
            return np.asarray(data_array)

    # sample a certified
    num_kp_noise_vars = data_payload["sqdist_kp_naiveest"].shape[0]
    for var_i in tqdm(range(num_kp_noise_vars)):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]

        # masks for certified / non-certified results
        certi_mask_naive = certi_masks["certi_naive"][var_i]
        certi_mask_corrector = certi_masks["certi_corrector"][var_i]
        not_certi_mask_naive = np.logical_not(certi_mask_naive)
        not_certi_mask_corrector = np.logical_not(certi_mask_corrector)

        pt_sqdist_naive = data_payload["sqdist_input_naiveest"][var_i]
        pt_sqdist_correctorest = data_payload["sqdist_input_correctorest"][var_i]
        kp_sqdist_naive = data_payload["sqdist_kp_naiveest"][var_i]
        kp_sqdist_correctorest = data_payload["sqdist_kp_correctorest"][var_i]

        certi_naive_rand_indices = gen_rand_indices_or_all(np.argwhere(certi_mask_naive).flatten(), num_random_samples)
        certi_corrector_rand_indices = gen_rand_indices_or_all(
            np.argwhere(certi_mask_corrector).flatten(), num_random_samples
        )

        # sample random non-certified samples
        not_certi_naive_rand_indices = gen_rand_indices_or_all(
            np.argwhere(not_certi_mask_naive).flatten(), num_random_samples
        )
        not_certi_corrector_rand_indices = gen_rand_indices_or_all(
            np.argwhere(not_certi_mask_corrector).flatten(), num_random_samples
        )
        assert len(set(certi_naive_rand_indices).intersection(set(not_certi_naive_rand_indices))) == 0
        assert len(set(certi_corrector_rand_indices).intersection(set(not_certi_corrector_rand_indices))) == 0

        # plot certified corrector instance
        rand_string = generate_filename()
        if len(certi_corrector_rand_indices) != 0:
            print(f"Noise Var={kp_noise_var:.2f} Sampled idx={certi_corrector_rand_indices[0]}; Cert")
            cert_inst_filename = f"{base_filename}_cert_{kp_noise_var:.2f}_{rand_string}"
            plot_instance_histogram(
                pt_sqdist_correctorest[certi_corrector_rand_indices[0]],
                quantile=certifier_quantile,
                abs_sq_epsilon=certifier_sq_epsilon,
                sq_clamp_threshold=certifier_sq_clamp_threshold,
                save_fig=save_fig,
                file_basename=cert_inst_filename,
                save_folder=base_folder,
            )

        # plot non certified corrector instance
        if len(not_certi_corrector_rand_indices) != 0:
            print(f"Noise Var={kp_noise_var:.2f} Sampled idx={certi_corrector_rand_indices[0]}; Not Cert")
            nocert_inst_filename = f"{base_filename}_notcert_{kp_noise_var:.2f}_{rand_string}"
            plot_instance_histogram(
                pt_sqdist_correctorest[not_certi_corrector_rand_indices[0]],
                quantile=certifier_quantile,
                abs_sq_epsilon=certifier_sq_epsilon,
                sq_clamp_threshold=certifier_sq_clamp_threshold,
                save_fig=save_fig,
                file_basename=nocert_inst_filename,
                save_folder=base_folder,
            )

    return


def plot_frac_cert(kp_noise_var_range, save_fig=True, base_filename="frac_cert", base_folder="./", **kwargs):
    """Plot fraction certifiable"""
    certi_naive = kwargs["certi_naive"]
    certi_corrector = kwargs["certi_corrector"]

    frac_cert_x, frac_cert_naive_y, frac_cert_corrector_y = prepare_fraction_certifiable_data(
        kp_noise_var_range, certi_naive, certi_corrector
    )

    fig_frac_cert, ax_frac_cert = plt.subplots()
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


def plot_adds(kp_noise_var_range, save_fig=True, adds_base_filename="adds_comp", base_folder="./", **kwargs):
    """Plot 3 plots:
    1. ADD-S error
    2. Fraction certifiable
    3. A sample histogram w/ cutoff threshold
    """
    """Plot ADD-S comparison plots"""
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

    # 1. ADD-S plot
    fig_adds, ax_adds = plt.subplots()
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

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(adds_base_filename, base_folder)
    plt.close()

    return


if __name__ == "__main__":
    print("Plot ADD-S error, fraction certifiable and histogram of chamfer distances.")
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
        default="figures/ycbv",
        type=str,
    )

    parser.add_argument("--object_label", help="object label in the dataset", default="obj_000006", type=str)

    args = parser.parse_args()
    print("CLI args: ")
    print(args)

    exp_folder_path = Path(__file__).parent.parent.resolve()
    dump_path = os.path.join(exp_folder_path, args.save_folder)
    print(f"Dump path: {dump_path}")
    base_folder = os.path.join(dump_path, "err_quantile_plots", args.object_label)
    safely_make_folders([base_folder])

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
    data_payload["chamfer_clamp_thres_factor"] = certifier_cfg["clamp_threshold"]
    (
        certi_naive,
        certi_corrector,
        certi_naive_failure_modes,
        certi_corrector_failure_modes,
        certifier_eps_used,
        certifier_clamp_threshold_used,
    ) = get_certified_instances(payload=data_payload, certifier_cfg=certifier_cfg)
    certi_masks = {"certi_naive": certi_naive.cpu().numpy(), "certi_corrector": certi_corrector.cpu().numpy()}

    results_data = evaluate_certifier(data_payload=data_payload, certi_masks=certi_masks)

    # plot ADDS plot
    plot_adds(
        data_payload["kp_noise_var_range"],
        save_fig=True,
        adds_base_filename="adds_comp",
        base_folder=base_folder,
        certi_naive_failure_modes=certi_naive_failure_modes,
        certi_corrector_failure_modes=certi_corrector_failure_modes,
        **results_data,
    )

    # plot fraction certifiable plot
    plot_frac_cert(
        data_payload["kp_noise_var_range"], save_fig=True, base_filename="frac_cert", base_folder=base_folder, **results_data
    )

    # plot sample histogram
    plot_sample_quantile_histograms(
        data_payload,
        certi_masks,
        certifier_sq_epsilon=certifier_eps_used**2,
        certifier_sq_clamp_threshold=certifier_clamp_threshold_used**2,
        certifier_quantile=certifier_cfg["epsilon_quantile"],
        save_fig=True,
        base_filename="sample_quantile_histogram",
        base_folder=base_folder
    )
