import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
from datetime import datetime
from matplotlib import colors as mcolors
from pathlib import Path
from tqdm import tqdm

from casper3d.certifiability import certifiability
from plot_shared import load_data
from utils.file_utils import safely_make_folders
from utils.general import generate_filename
from utils.visualization_utils import plt_save_figures

plt.style.use("seaborn-whitegrid")
COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
BASE_DIR = Path(__file__).parent.parent.parent


# def masked_var_mean(data, mask):
#     """
#     inputs:
#     data    : torch.tensor of shape (B, n)
#     mask    : torch.tensor of shape (B, n). dtype=torch.bool
#
#     outputs:
#     mean    : torch.tensor of shape (B, 1)
#     var     : torch.tensor of shape (B, 1)
#     """
#
#     mean = (torch.sum(data*mask.float(), dim=1)/torch.sum(mask.float(), dim=1)).unsqueeze(-1)
#
#     data_centered = data - mean
#     var = (torch.sum((data_centered**2)*mask.float(), dim=1)/(torch.sum(mask.float(), dim=1)-1)).unsqueeze(-1)
#
#     return var.squeeze(-1), mean.squeeze(-1)


def masked_varul_mean(data, mask):
    """
    inputs:
    data    : torch.tensor of shape (B, n)
    mask    : torch.tensor of shape (B, n). dtype=torch.bool

    outputs:
    var     : torch.tensor of shape (B, 2)  :
        var[:, 0] = lower variance
        var[:, 1] = upper variance

    mean    : torch.tensor of shape (B,)

    """
    device_ = data.device
    batch_size = data.shape[0]

    var = torch.zeros(batch_size, 2).to(device_)
    mean = torch.zeros(batch_size).to(device_)

    for batch, (d, m) in enumerate(zip(data, mask)):
        dm = torch.masked_select(d, m)

        dm_mean = dm.mean()
        dm_centered = dm - dm_mean
        dm_centered_up = dm_centered * (dm_centered >= 0)
        dm_centered_lo = dm_centered * (dm_centered < 0)
        len = dm_centered.shape[0]

        dm_var_up = torch.sum(dm_centered_up**2) / (len + 0.001)
        dm_var_lo = torch.sum(dm_centered_lo**2) / (len + 0.001)

        mean[batch] = dm_mean
        var[batch, 0] = dm_var_lo
        var[batch, 1] = dm_var_up

    return var, mean


def varul_mean(data):
    """
    inputs:
    data    : torch.tensor of shape (B, n)

    outputs:
    var     : torch.tensor of shape (B, 2)  :
        var[:, 0] = lower variance
        var[:, 1] = upper variance

    mean    : torch.tensor of shape (B,)

    """

    mean = data.mean(dim=1).unsqueeze(-1)

    data_centered = data - mean
    data_pos = data_centered * (data_centered >= 0)
    data_neg = data_centered * (data_centered < 0)
    len = data_centered.shape[1]

    var_up = torch.sum(data_pos**2, dim=1) / (len + 0.001)
    var_low = torch.sum(data_neg**2, dim=1) / (len + 0.001)

    var_up = var_up.unsqueeze(-1)
    var_low = var_low.unsqueeze(-1)
    var = torch.cat([var_low, var_up], dim=1)

    return var, mean.squeeze(-1)


def certification(data, parameters, epsilon, num_iterations=100, full_batch=False):
    object_diameter = parameters["diameter"]
    chamfer_clamp_thres_factor = parameters["chamfer_clamp_thres_factor"]
    xlen = parameters["kp_noise_var_range"].shape[0]

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    certify = certifiability(
        epsilon=epsilon * object_diameter, clamp_thres=chamfer_clamp_thres_factor * object_diameter
    )

    ###
    certi_naive = torch.zeros(size=(xlen, num_iterations), dtype=torch.bool).to(device=device_)
    certi_corrector = torch.zeros(size=(xlen, num_iterations), dtype=torch.bool).to(device=device_)

    ###
    sqdist_input_naiveest = data["sqdist_input_naiveest"]
    sqdist_input_correctorest = data["sqdist_input_correctorest"]
    sqdist_kp_naiveest = data["sqdist_kp_naiveest"]
    sqdist_kp_correctorest = data["sqdist_kp_correctorest"]
    pc_padding_masks = data["pc_padding_masks"]

    for kp_noise_var_i in range(len(sqdist_input_naiveest)):
        # print("kp_noise_var_i", kp_noise_var_i)
        c_naive = torch.zeros((num_iterations, 1), dtype=torch.bool).to(device=device_)
        c_corrector = torch.zeros((num_iterations, 1), dtype=torch.bool).to(device=device_)
        # if experiments were full batch, just set output of certify.forward_with_distances
        # to c_naive and c_corrector
        for batch_i in range(len(sqdist_input_naiveest[kp_noise_var_i])):
            # print("batch_i", batch_i)

            # len 100 or batch size 100
            sqdist_input_naive = sqdist_input_naiveest[kp_noise_var_i][batch_i]
            sqdist_input_corrector = sqdist_input_correctorest[kp_noise_var_i][batch_i]
            sqdist_kp_naive = sqdist_kp_naiveest[kp_noise_var_i][batch_i]
            sqdist_kp_corrector = sqdist_kp_correctorest[kp_noise_var_i][batch_i]
            pc_padding = pc_padding_masks[kp_noise_var_i][batch_i]
            certi_naive_batch = certify.forward_with_distances(sqdist_input_naive[0], sqdist_kp_naive, pc_padding)
            certi_corrector_batch = certify.forward_with_distances(
                sqdist_input_corrector[0], sqdist_kp_corrector, pc_padding
            )
            if full_batch:  # full batch
                c_naive = certi_naive_batch
                c_corrector = certi_corrector_batch
            else:
                print("certi_naive_batch.shape", certi_naive_batch.shape)
                c_naive[batch_i] = certi_naive_batch
                c_corrector[batch_i] = certi_corrector_batch
        certi_naive[kp_noise_var_i, ...] = c_naive.squeeze(-1)
        certi_corrector[kp_noise_var_i, ...] = c_corrector.squeeze(-1)

    return certi_naive, certi_corrector


def evaluate_for_hyperparameter_tuning(file_names, epsilon):
    use_adds_metric = True

    for name in file_names:

        fp = open(name, "rb")
        parameters, data = pickle.load(fp)
        fp.close()

        if use_adds_metric:
            for noise_idx in range(len(data["chamfer_pose_naive_to_gt_pose_list"])):
                data["chamfer_pose_naive_to_gt_pose_list"][noise_idx] = np.asarray(
                    data["chamfer_pose_naive_to_gt_pose_list"][noise_idx][0].squeeze().to("cpu")
                )
                data["chamfer_pose_corrected_to_gt_pose_list"][noise_idx] = np.asarray(
                    data["chamfer_pose_corrected_to_gt_pose_list"][noise_idx][0].squeeze().to("cpu")
                )

        print("-" * 80)

        kp_noise_var_range = parameters["kp_noise_var_range"].to("cpu")
        kp_noise_fra = parameters["kp_noise_fra"]
        ds_name = parameters["ds_name"]
        object_label = parameters["object_label"]
        object_diameter = parameters["diameter"]
        algo = parameters["algo"]
        # chamfer_clamp_thres_factor = parameters['chamfer_clamp_thres_factor']
        chamfer_clamp_thres_factor = 0.1
        # breakpoint()

        Rerr_naive = data["rotation_err_naive"].to("cpu")
        Rerr_corrector = data["rotation_err_corrector"].to("cpu")
        terr_naive = data["translation_err_naive"].to("cpu") / object_diameter
        terr_corrector = data["translation_err_corrector"].to("cpu") / object_diameter
        # breakpoint()

        # CALCULATE DYNAMICALLY
        # epsilon = .02
        fig_save_folder = "/".join(str(name).split("/")[:-1] + [algo] + ["eps" + str(epsilon)[2:]])
        if not os.path.exists(fig_save_folder):
            os.makedirs(fig_save_folder)
        fig_save_prefix = fig_save_folder + "/" + str(name).split("/")[-1]
        print(fig_save_prefix)
        certi_naive, certi_corrector = certification(data, parameters, epsilon=epsilon, full_batch=True)
        certi_naive = certi_naive.to("cpu")
        certi_corrector = certi_corrector.to("cpu")

        # sqdist_input_naiveest = data['sqdist_input_naiveest']
        # sqdist_input_correctorest = data['sqdist_input_correctorest']
        # sqdist_kp_naiveest = data['sqdist_kp_naiveest']
        # sqdist_kp_correctorest = data['sqdist_kp_correctorest']

        if use_adds_metric:
            chamfer_pose_naive_to_gt_pose_list = torch.from_numpy(
                np.asarray(data["chamfer_pose_naive_to_gt_pose_list"])
            )
            chamfer_pose_corrected_to_gt_pose_list = torch.from_numpy(
                np.asarray(data["chamfer_pose_corrected_to_gt_pose_list"])
            )

            # normalizing it with respect to the diameter
            chamfer_pose_naive_to_gt_pose_list = chamfer_pose_naive_to_gt_pose_list / object_diameter
            chamfer_pose_corrected_to_gt_pose_list = chamfer_pose_corrected_to_gt_pose_list / object_diameter

        # Plotting rotation distribution
        # fig = plt.figure()
        # plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_naive, label='naive', color='lightgray')
        # plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_naive*certi_naive, label='naive + certification', color='royalblue')
        # plt.show()
        # plt.close(fig)
        #
        # fig = plt.figure()
        # plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_corrector, label='corrector', color='lightgray')
        # plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_corrector * certi_corrector, label='corrector + certification', color='orangered')
        # plt.show()
        # plt.close(fig)

        if use_adds_metric:
            chamfer_metric_naive_var, chamfer_metric_naive_mean = varul_mean(chamfer_pose_naive_to_gt_pose_list)
            chamfer_metric_corrected_var, chamfer_metric_corrected_mean = varul_mean(
                chamfer_pose_corrected_to_gt_pose_list
            )
            chamfer_metric_naive_certi_var, chamfer_metric_naive_certi_mean = masked_varul_mean(
                chamfer_pose_naive_to_gt_pose_list, mask=certi_naive
            )
            chamfer_metric_corrected_certi_var, chamfer_metric_corrected_certi_mean = masked_varul_mean(
                chamfer_pose_corrected_to_gt_pose_list, mask=certi_corrector
            )

        Rerr_naive_var, Rerr_naive_mean = varul_mean(Rerr_naive)
        Rerr_corrector_var, Rerr_corrector_mean = varul_mean(Rerr_corrector)
        terr_naive_var, terr_naive_mean = varul_mean(terr_naive)
        terr_corrector_var, terr_corrector_mean = varul_mean(terr_corrector)

        Rerr_naive_certi_var, Rerr_naive_certi_mean = masked_varul_mean(Rerr_naive, mask=certi_naive)
        Rerr_corrector_certi_var, Rerr_corrector_certi_mean = masked_varul_mean(Rerr_corrector, mask=certi_corrector)
        terr_naive_certi_var, terr_naive_certi_mean = masked_varul_mean(terr_naive, mask=certi_naive)
        terr_corrector_certi_var, terr_corrector_certi_mean = masked_varul_mean(terr_corrector, mask=certi_corrector)

        fraction_not_certified_naive_var, fraction_not_certified_naive_mean = varul_mean(1 - certi_naive.float())
        fraction_not_certified_corrector_var, fraction_not_certified_corrector_mean = varul_mean(
            1 - certi_corrector.float()
        )

        Rerr_naive_var = torch.sqrt(Rerr_naive_var).T
        Rerr_corrector_var = torch.sqrt(Rerr_corrector_var).T
        terr_naive_var = torch.sqrt(terr_naive_var).T
        terr_corrector_var = torch.sqrt(terr_corrector_var).T
        Rerr_naive_certi_var = torch.sqrt(Rerr_naive_certi_var).T
        Rerr_corrector_certi_var = torch.sqrt(Rerr_corrector_certi_var).T
        terr_naive_certi_var = torch.sqrt(terr_naive_certi_var).T
        terr_corrector_certi_var = torch.sqrt(terr_corrector_certi_var).T
        fraction_not_certified_corrector_var = torch.sqrt(fraction_not_certified_corrector_var).T
        fraction_not_certified_naive_var = torch.sqrt(fraction_not_certified_naive_var).T

        # breakpoint()
        print("Rerr_corrector_mean: ", Rerr_corrector_mean)
        print("Rerr_corrector_certi_mean: ", Rerr_corrector_certi_mean)
        print("fraction not certified corrector mean: ", fraction_not_certified_corrector_mean)

        return Rerr_corrector_mean, Rerr_corrector_certi_mean, fraction_not_certified_corrector_mean


def make_plots(file_names, epsilon):
    use_adds_metric = True

    for name in file_names:

        fp = open(name, "rb")
        parameters, data = pickle.load(fp)
        fp.close()

        if use_adds_metric:
            for noise_idx in range(len(data["chamfer_pose_naive_to_gt_pose_list"])):
                data["chamfer_pose_naive_to_gt_pose_list"][noise_idx] = np.asarray(
                    data["chamfer_pose_naive_to_gt_pose_list"][noise_idx][0].squeeze().to("cpu")
                )
                data["chamfer_pose_corrected_to_gt_pose_list"][noise_idx] = np.asarray(
                    data["chamfer_pose_corrected_to_gt_pose_list"][noise_idx][0].squeeze().to("cpu")
                )

        print("-" * 80)

        kp_noise_var_range = parameters["kp_noise_var_range"].to("cpu")
        kp_noise_fra = parameters["kp_noise_fra"]
        ds_name = parameters["ds_name"]
        object_label = parameters["object_label"]
        object_diameter = parameters["diameter"]
        algo = parameters["algo"]
        # chamfer_clamp_thres_factor = parameters['chamfer_clamp_thres_factor']
        chamfer_clamp_thres_factor = 0.1
        # breakpoint()

        Rerr_naive = data["rotation_err_naive"].to("cpu")
        Rerr_corrector = data["rotation_err_corrector"].to("cpu")
        terr_naive = data["translation_err_naive"].to("cpu") / object_diameter
        terr_corrector = data["translation_err_corrector"].to("cpu") / object_diameter
        # breakpoint()

        # CALCULATE DYNAMICALLY
        # epsilon = .02
        fig_save_folder = "/".join(str(name).split("/")[:-1] + [algo] + ["eps" + str(epsilon)[2:]])
        if not os.path.exists(fig_save_folder):
            os.makedirs(fig_save_folder)
        fig_save_prefix = fig_save_folder + "/" + str(name).split("/")[-1]
        print(fig_save_prefix)
        certi_naive, certi_corrector = certification(data, parameters, epsilon=epsilon, full_batch=True)
        certi_naive = certi_naive.to("cpu")
        certi_corrector = certi_corrector.to("cpu")

        # sqdist_input_naiveest = data['sqdist_input_naiveest']
        # sqdist_input_correctorest = data['sqdist_input_correctorest']
        # sqdist_kp_naiveest = data['sqdist_kp_naiveest']
        # sqdist_kp_correctorest = data['sqdist_kp_correctorest']

        if use_adds_metric:
            chamfer_pose_naive_to_gt_pose_list = torch.from_numpy(
                np.asarray(data["chamfer_pose_naive_to_gt_pose_list"])
            )
            chamfer_pose_corrected_to_gt_pose_list = torch.from_numpy(
                np.asarray(data["chamfer_pose_corrected_to_gt_pose_list"])
            )

            # normalizing it with respect to the diameter
            chamfer_pose_naive_to_gt_pose_list = chamfer_pose_naive_to_gt_pose_list / object_diameter
            chamfer_pose_corrected_to_gt_pose_list = chamfer_pose_corrected_to_gt_pose_list / object_diameter

        # Plotting rotation distribution
        # fig = plt.figure()
        # plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_naive, label='naive', color='lightgray')
        # plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_naive*certi_naive, label='naive + certification', color='royalblue')
        # plt.show()
        # plt.close(fig)
        #
        # fig = plt.figure()
        # plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_corrector, label='corrector', color='lightgray')
        # plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_corrector * certi_corrector, label='corrector + certification', color='orangered')
        # plt.show()
        # plt.close(fig)

        if use_adds_metric:
            chamfer_metric_naive_var, chamfer_metric_naive_mean = varul_mean(chamfer_pose_naive_to_gt_pose_list)
            chamfer_metric_corrected_var, chamfer_metric_corrected_mean = varul_mean(
                chamfer_pose_corrected_to_gt_pose_list
            )
            chamfer_metric_naive_certi_var, chamfer_metric_naive_certi_mean = masked_varul_mean(
                chamfer_pose_naive_to_gt_pose_list, mask=certi_naive
            )
            chamfer_metric_corrected_certi_var, chamfer_metric_corrected_certi_mean = masked_varul_mean(
                chamfer_pose_corrected_to_gt_pose_list, mask=certi_corrector
            )

        Rerr_naive_var, Rerr_naive_mean = varul_mean(Rerr_naive)
        Rerr_corrector_var, Rerr_corrector_mean = varul_mean(Rerr_corrector)
        terr_naive_var, terr_naive_mean = varul_mean(terr_naive)
        terr_corrector_var, terr_corrector_mean = varul_mean(terr_corrector)

        Rerr_naive_certi_var, Rerr_naive_certi_mean = masked_varul_mean(Rerr_naive, mask=certi_naive)
        Rerr_corrector_certi_var, Rerr_corrector_certi_mean = masked_varul_mean(Rerr_corrector, mask=certi_corrector)
        terr_naive_certi_var, terr_naive_certi_mean = masked_varul_mean(terr_naive, mask=certi_naive)
        terr_corrector_certi_var, terr_corrector_certi_mean = masked_varul_mean(terr_corrector, mask=certi_corrector)

        fraction_not_certified_naive_var, fraction_not_certified_naive_mean = varul_mean(1 - certi_naive.float())
        fraction_not_certified_corrector_var, fraction_not_certified_corrector_mean = varul_mean(
            1 - certi_corrector.float()
        )

        Rerr_naive_var = torch.sqrt(Rerr_naive_var).T
        Rerr_corrector_var = torch.sqrt(Rerr_corrector_var).T
        terr_naive_var = torch.sqrt(terr_naive_var).T
        terr_corrector_var = torch.sqrt(terr_corrector_var).T
        Rerr_naive_certi_var = torch.sqrt(Rerr_naive_certi_var).T
        Rerr_corrector_certi_var = torch.sqrt(Rerr_corrector_certi_var).T
        terr_naive_certi_var = torch.sqrt(terr_naive_certi_var).T
        terr_corrector_certi_var = torch.sqrt(terr_corrector_certi_var).T
        fraction_not_certified_corrector_var = torch.sqrt(fraction_not_certified_corrector_var).T
        fraction_not_certified_naive_var = torch.sqrt(fraction_not_certified_naive_var).T

        if use_adds_metric:
            chamfer_metric_naive_var = torch.sqrt(chamfer_metric_naive_var).T
            chamfer_metric_corrected_var = torch.sqrt(chamfer_metric_corrected_var).T
            chamfer_metric_naive_certi_var = torch.sqrt(chamfer_metric_naive_certi_var).T
            chamfer_metric_corrected_certi_var = torch.sqrt(chamfer_metric_corrected_certi_var).T

        # plotting chamfer metric
        if use_adds_metric:
            fig = plt.figure()
            plt.errorbar(
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
            plt.errorbar(
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
            plt.errorbar(
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
            plt.errorbar(
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
            # new colors
            # plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_naive_mean, yerr=chamfer_metric_naive_var,
            #              fmt='-', color='red', ecolor='red', elinewidth=1, capsize=3, label='naive')
            # plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_naive_certi_mean, yerr=chamfer_metric_naive_certi_var,
            #              fmt='--o', color='red', ecolor=(1.0,0,0,0.3), elinewidth=3, capsize=3, label='naive + certification')
            # plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_corrected_mean, yerr=chamfer_metric_corrected_var,
            #              fmt='-', color='green', ecolor='green', elinewidth=1, capsize=3, label='corrector')
            # plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_corrected_certi_mean, yerr=chamfer_metric_corrected_certi_var,
            #              fmt='--o', color='green', ecolor=(0,.5,0,0.3), elinewidth=3, capsize=3, label='corrector + certification')
            plt.legend(loc="upper left")
            # title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
            # plt.title(title)
            plt.xlabel("Noise variance parameter $\sigma$")
            plt.ylabel("Normalized ADD-S")
            # plt.show()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rand_string = generate_filename()
            filename = fig_save_prefix[:-7] + "_chamfer_metric_plot_" + timestamp + "_" + rand_string + ".jpg"
            fig.savefig(filename)
            plt.close(fig)

        # plotting rotation errors
        fig = plt.figure()
        plt.errorbar(
            x=kp_noise_var_range,
            y=Rerr_naive_mean,
            yerr=Rerr_naive_var,
            fmt="-x",
            color="black",
            ecolor="gray",
            elinewidth=1,
            capsize=3,
            label="naive",
        )
        plt.errorbar(
            x=kp_noise_var_range,
            y=Rerr_naive_certi_mean,
            yerr=Rerr_naive_certi_var,
            fmt="--o",
            color="grey",
            ecolor="lightgray",
            elinewidth=3,
            capsize=0,
            label="naive + certification",
        )
        plt.errorbar(
            x=kp_noise_var_range,
            y=Rerr_corrector_mean,
            yerr=Rerr_corrector_var,
            fmt="-x",
            color="red",
            ecolor="salmon",
            elinewidth=1,
            capsize=3,
            label="corrector",
        )
        plt.errorbar(
            x=kp_noise_var_range,
            y=Rerr_corrector_certi_mean,
            yerr=Rerr_corrector_certi_var,
            fmt="--o",
            color="orangered",
            ecolor="salmon",
            elinewidth=3,
            capsize=0,
            label="corrector + certification",
        )

        # new colors
        # plt.errorbar(x=kp_noise_var_range, y=Rerr_naive_mean, yerr=Rerr_naive_var, fmt='-', color='red', ecolor='red', elinewidth=1, capsize=3, label='naive')
        # plt.errorbar(x=kp_noise_var_range, y=Rerr_naive_certi_mean, yerr=Rerr_naive_certi_var, fmt='--o', color='red', ecolor=(1.0,0,0,0.3), elinewidth=3, capsize=3, label='naive + certification')
        # plt.errorbar(x=kp_noise_var_range, y=Rerr_corrector_mean, yerr=Rerr_corrector_var, fmt='-', color='green', ecolor='green', elinewidth=1, capsize=3, label='corrector')
        # plt.errorbar(x=kp_noise_var_range, y=Rerr_corrector_certi_mean, yerr=Rerr_corrector_certi_var, fmt='--o', color='green', ecolor=(0,.5,0,0.3), elinewidth=3, capsize=3, label='corrector + certification')
        plt.legend(loc="upper left")
        # title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        # plt.title(title)
        plt.xlabel("Noise variance parameter $\sigma$")
        plt.ylabel("Rotation error")
        # plt.show()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_string = generate_filename()
        filename = fig_save_prefix[:-7] + "_rotation_error_plot_" + timestamp + "_" + rand_string + ".jpg"
        fig.savefig(filename)
        plt.close(fig)

        # Plotting translation errors
        fig = plt.figure()
        plt.errorbar(
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
        plt.errorbar(
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
        plt.errorbar(
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
        plt.errorbar(
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

        # new colors
        # plt.errorbar(x=kp_noise_var_range, y=terr_naive_mean, yerr=terr_naive_var, fmt='-', color='red', ecolor='red', elinewidth=1, capsize=3, label='naive')
        # plt.errorbar(x=kp_noise_var_range, y=terr_naive_certi_mean, yerr=terr_naive_certi_var, fmt='--o', color='red', ecolor=(1.0,0,0,0.3), elinewidth=3, capsize=3, label='naive + certification')
        # plt.errorbar(x=kp_noise_var_range, y=terr_corrector_mean, yerr=terr_corrector_var, fmt='-', color='green', ecolor='green', elinewidth=1, capsize=3, label='corrector')
        # plt.errorbar(x=kp_noise_var_range, y=terr_corrector_certi_mean, yerr=terr_corrector_certi_var, fmt='--o', color='green', ecolor=(0,.5,0,0.3), elinewidth=3, capsize=3, label='corrector + certification')
        plt.legend(loc="upper left")
        # title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        # plt.title(title)
        plt.xlabel("Noise variance parameter $\sigma$")
        plt.ylabel("Translation error")
        # plt.show()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_string = generate_filename()
        filename = fig_save_prefix[:-7] + "_translation_error_plot_" + timestamp + "_" + rand_string + ".jpg"
        fig.savefig(filename)
        plt.close(fig)

        # Plotting fraction not certified
        fig = plt.figure()
        plt.bar(
            x=kp_noise_var_range - 0.01,
            width=0.02,
            height=fraction_not_certified_naive_mean,
            color="grey",
            align="center",
            label="naive",
        )
        # plt.errorbar(x=kp_noise_var_range-0.01, y=fraction_not_certified_naive_mean,
        #              yerr=fraction_not_certified_naive_var,
        #              fmt='o', color='black', ecolor='darkgray', elinewidth=1, capsize=3)
        plt.bar(
            x=kp_noise_var_range + 0.01,
            width=0.02,
            height=fraction_not_certified_corrector_mean,
            color="salmon",
            align="center",
            label="corrector",
        )
        # plt.errorbar(x=kp_noise_var_range+0.01, y=fraction_not_certified_corrector_mean,
        #              yerr=fraction_not_certified_corrector_var,
        #              fmt='o', color='red', ecolor='orangered', elinewidth=1, capsize=3)
        plt.legend(loc="upper left")
        plt.xlabel("Noise variance parameter $\sigma$")
        plt.ylabel("Fraction not certifiable")
        # title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        # plt.title(title)
        # plt.show()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_string = generate_filename()
        filename = fig_save_prefix[:-7] + "_fraction_not_certifiable_plot_" + timestamp + "_" + rand_string + ".jpg"
        fig.savefig(filename)
        plt.close(fig)


def remove_nans(dist_mat):
    """Remove NaN entries in the distance matrix"""
    return dist_mat[:, ~np.any(np.isnan(dist_mat), axis=(0, 2)), :]


def get_certified_instances(payload, epsilon):
    """Return certified instances only.
    Combine with other histogram functions to plot only certified data
    """
    object_diameter = payload["object_diameter"]
    chamfer_clamp_thres_factor = payload["chamfer_clamp_thres_factor"]
    xlen = payload["kp_noise_var_range"].shape[0]
    num_samples = payload["sqdist_input_naiveest"].shape[1]

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    certify = certifiability(
        epsilon=epsilon * object_diameter, clamp_thres=chamfer_clamp_thres_factor * object_diameter
    )

    certi_naive = torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_)
    certi_corrector = torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_)
    certi_naive_failure_modes = {
        "pt": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
        "kp": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
    }
    certi_corrector_failure_modes = {
        "pt": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
        "kp": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
    }

    sqdist_input_naiveest = torch.as_tensor(payload["sqdist_input_naiveest"]).to(device_)
    sqdist_input_correctorest = torch.as_tensor(payload["sqdist_input_correctorest"]).to(device_)
    sqdist_kp_naiveest = torch.as_tensor(payload["sqdist_kp_naiveest"]).to(device_)
    sqdist_kp_correctorest = torch.as_tensor(payload["sqdist_kp_correctorest"]).to(device_)
    pc_padding_masks = torch.as_tensor(payload["pc_padding_masks"]).to(device_)

    for kp_noise_var_i in range(xlen):
        # c_naive = torch.zeros((num_samples, 1), dtype=torch.bool).to(device=device_)
        # c_corrector = torch.zeros((num_samples, 1), dtype=torch.bool).to(device=device_)

        sqdist_input_naive = sqdist_input_naiveest[kp_noise_var_i]
        sqdist_input_corrector = sqdist_input_correctorest[kp_noise_var_i]
        sqdist_kp_naive = sqdist_kp_naiveest[kp_noise_var_i]
        sqdist_kp_corrector = sqdist_kp_correctorest[kp_noise_var_i]
        pc_padding = pc_padding_masks[kp_noise_var_i]

        c_naive = certify.forward_with_distances(
            sqdist_input_naive.unsqueeze(-1), sqdist_kp_naive, zero_mask=pc_padding
        )

        c_corrector = certify.forward_with_distances(
            sqdist_input_corrector.unsqueeze(-1), sqdist_kp_corrector, zero_mask=pc_padding
        )
        certi_naive[kp_noise_var_i, ...] = c_naive.squeeze(-1).to(device=device_, dtype=torch.bool)
        certi_corrector[kp_noise_var_i, ...] = c_corrector.squeeze(-1).to(device=device_, dtype=torch.bool)

        # update failure modes
        c_naive_pc_score, c_naive_kp_score = certify.confidence_scores_with_distances(
            sqdist_input_naive.unsqueeze(-1), sqdist_kp_naive, zero_mask=pc_padding
        )
        c_corrector_pc_score, c_corrector_kp_score = certify.confidence_scores_with_distances(
            sqdist_input_corrector.unsqueeze(-1), sqdist_kp_corrector, zero_mask=pc_padding
        )

        # flags that are true if the specific condition failed
        c_naive_pc_failed = c_naive_pc_score > certify.epsilon
        c_naive_kp_failed = c_naive_kp_score > certify.epsilon
        c_corrector_pc_failed = c_corrector_pc_score > certify.epsilon
        c_corrector_kp_failed = c_corrector_kp_score > certify.epsilon

        # save the flags
        certi_naive_failure_modes["pt"][kp_noise_var_i] = c_naive_pc_failed.flatten()
        certi_naive_failure_modes["kp"][kp_noise_var_i] = c_naive_kp_failed.flatten()
        certi_corrector_failure_modes["pt"][kp_noise_var_i] = c_corrector_pc_failed.flatten()
        certi_corrector_failure_modes["kp"][kp_noise_var_i] = c_corrector_kp_failed.flatten()

    return certi_naive, certi_corrector, certi_naive_failure_modes, certi_corrector_failure_modes


def make_one_hist(sqdist_df, column_name, kp_noise_var, **kwargs):
    """Helper function to make one histogram plot at one noise var"""
    mask = sqdist_df["noise_var"] == kp_noise_var
    x = sqdist_df.loc[mask][column_name]
    data = plt.hist(x, bins=100, log=True, **kwargs)
    return data


def chamfer_distances_histogram_at_single_noise_var(idx, kp_noise_var, data_payload):
    """Plot histogram of chamfer distances at one specific noise var"""
    fig = plt.figure()

    # access chamfer distances
    df = pd.DataFrame(data={"sq_dists": None, "kp_sq_dists": None})

    return


def pointwise_distances_histogram_at_noise_vars(
    data_payload, sample_mask=None, save_fig=True, save_folder="./", prefix_str="", object_label=None
):
    """Plot histogram of chamfer distances at all noise var"""
    # create pandas dataframe for points closest distances
    # note: we combine all points across all trials
    # columns: noise var, sqdist_input_correctorest, sqdist_input_naiveest
    naive_sqdist_data = dict(noise_var=[], sqdist_input_naiveest=[])
    corrector_sqdist_data = dict(noise_var=[], sqdist_input_correctorest=[])
    assert data_payload["sqdist_input_naiveest"].shape == data_payload["sqdist_input_correctorest"].shape
    num_kp_noise_vars = data_payload["sqdist_input_naiveest"].shape[0]
    for var_i in range(num_kp_noise_vars):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]

        # remove pc padding masks
        # the naive and corrector result should have the same lengths
        # b/c the padding mask applies to the point cloud inputs
        valid_point_mask = np.logical_not(data_payload["pc_padding_masks"][var_i])

        if sample_mask is None:
            flatten_sqdist_naive = data_payload["sqdist_input_naiveest"][var_i][valid_point_mask].flatten()
            flatten_sqdist_correctorest = data_payload["sqdist_input_correctorest"][var_i][valid_point_mask].flatten()
        else:
            # do not add uncertifiable instances to the data
            flatten_sqdist_naive, flatten_sqdist_correctorest = [], []
            num_samples = data_payload["sqdist_input_naiveest"][var_i].shape[0]
            for sample_idx in range(num_samples):
                sample_pc_padding_mask = valid_point_mask[sample_idx]
                cert_flag_naive = sample_mask["certi_naive"][var_i, sample_idx].item()
                cert_flag_corrector = sample_mask["certi_corrector"][var_i, sample_idx].item()
                if cert_flag_naive:
                    flatten_sqdist_naive.extend(
                        list(data_payload["sqdist_input_naiveest"][var_i][sample_idx][sample_pc_padding_mask])
                    )
                if cert_flag_corrector:
                    flatten_sqdist_correctorest.extend(
                        list(data_payload["sqdist_input_correctorest"][var_i][sample_idx][sample_pc_padding_mask])
                    )
            flatten_sqdist_naive = np.asarray(flatten_sqdist_naive)
            flatten_sqdist_correctorest = np.asarray(flatten_sqdist_correctorest)

        for d_naive in flatten_sqdist_naive:
            naive_sqdist_data["noise_var"].append(kp_noise_var)
            naive_sqdist_data["sqdist_input_naiveest"].append(d_naive)

        for d_corrector in flatten_sqdist_correctorest:
            corrector_sqdist_data["noise_var"].append(kp_noise_var)
            corrector_sqdist_data["sqdist_input_correctorest"].append(d_corrector)

    # Two dataframes
    naive_sqdist_df = pd.DataFrame(data=naive_sqdist_data)
    corrector_sqdist_df = pd.DataFrame(data=corrector_sqdist_data)

    # make plots for corrector
    plt.figure()
    for var_i in tqdm(range(num_kp_noise_vars)):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]
        # corrector
        make_one_hist(
            corrector_sqdist_df,
            "sqdist_input_correctorest",
            kp_noise_var,
            label=f"Kpt. Noise Var.={kp_noise_var :.2f}",
            alpha=0.6,
        )
    plt.xlabel("Distance Squared")
    plt.ylabel("Counts")
    plt.title(f"Closest Point-to-Point Distance Distribution\n({object_label}, w/ Corrector)")
    plt.legend(loc="upper right")
    if not save_fig:
        plt.show()
        plt.close()
    else:
        plt.savefig(os.path.join(save_folder, f"{prefix_str}pt_sqdists_corrector_all_kp_noise_vars.pdf"))
        plt.savefig(os.path.join(save_folder, f"{prefix_str}pt_sqdists_corrector_all_kp_noise_vars.png"))
        plt.close()

    # make plots for naive
    plt.figure()
    for var_i in tqdm(range(num_kp_noise_vars)):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]
        # corrector
        make_one_hist(
            naive_sqdist_df,
            "sqdist_input_naiveest",
            kp_noise_var,
            label=f"Kpt. Noise Var.={kp_noise_var :.2f}",
            alpha=0.6,
        )
    plt.xlabel("Distance Squared")
    plt.ylabel("Counts")
    plt.title(f"Closest Point-to-Point Distance Distribution\n({object_label}, Naive)")
    plt.legend(loc="upper right")
    if not save_fig:
        plt.show()
        plt.close()
    else:
        plt.savefig(os.path.join(save_folder, f"{prefix_str}pt_sqdists_naive_all_kp_noise_vars.pdf"))
        plt.savefig(os.path.join(save_folder, f"{prefix_str}pt_sqdists_naive_all_kp_noise_vars.png"))
        plt.close()

    # compare naive w/ corrector at different kp vars
    for var_i in tqdm(range(num_kp_noise_vars)):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]
        make_one_hist(corrector_sqdist_df, "sqdist_input_correctorest", kp_noise_var, label=f"Corrector", alpha=0.6)
        make_one_hist(naive_sqdist_df, "sqdist_input_naiveest", kp_noise_var, label=f"Naive", alpha=0.6)
        plt.xlabel("Distance Squared")
        plt.ylabel("Counts")
        plt.title(
            f"Closest Point-to-Point Distance Distribution\nat Kpt. Noise. Var = {kp_noise_var:.2f} ({object_label})"
        )
        plt.legend(loc="upper right")
        if not save_fig:
            plt.show()
            plt.close()
        else:
            plt.savefig(
                os.path.join(
                    save_folder, f"{prefix_str}pt_sqdists_corrector_vs_naive_kp_noise_var_{kp_noise_var:.2E}.pdf"
                )
            )
            plt.savefig(
                os.path.join(
                    save_folder, f"{prefix_str}pt_sqdists_corrector_vs_naive_kp_noise_var_{kp_noise_var:.2E}.png"
                )
            )
            plt.close()

    return naive_sqdist_df, corrector_sqdist_df


def kp_distances_histogram_at_noise_vars(
    data_payload, sample_mask=None, save_fig=True, save_folder="./", prefix_str="", object_label=None
):
    """Plot histogram of chamfer distances at all noise var"""

    # create pandas dataframe for kp distance squared
    # note: we combine all points across all trials
    # columns: noise var, sqdist_input_correctorest, sqdist_input_naiveest
    naive_kp_sqdist_data = dict(noise_var=[], sqdist_kp_naiveest=[])
    corrector_kp_sqdist_data = dict(noise_var=[], sqdist_kp_correctorest=[])

    assert data_payload["sqdist_kp_naiveest"].shape == data_payload["sqdist_kp_correctorest"].shape
    num_kp_noise_vars = data_payload["sqdist_kp_naiveest"].shape[0]
    for var_i in range(num_kp_noise_vars):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]
        if sample_mask is None:
            flatten_sqdist_naive = data_payload["sqdist_kp_naiveest"][var_i].flatten()
            flatten_sqdist_correctorest = data_payload["sqdist_kp_correctorest"][var_i].flatten()
        else:
            # do not add uncertifiable instances to the data
            flatten_sqdist_naive, flatten_sqdist_correctorest = [], []
            num_samples = data_payload["sqdist_kp_naiveest"][var_i].shape[0]
            for sample_idx in range(num_samples):
                cert_flag_naive = sample_mask["certi_naive"][var_i, sample_idx].item()
                cert_flag_corrector = sample_mask["certi_corrector"][var_i, sample_idx].item()
                if cert_flag_naive:
                    flatten_sqdist_naive.extend(list(data_payload["sqdist_kp_naiveest"][var_i][sample_idx]))
                if cert_flag_corrector:
                    flatten_sqdist_correctorest.extend(list(data_payload["sqdist_kp_correctorest"][var_i][sample_idx]))
            flatten_sqdist_naive = np.asarray(flatten_sqdist_naive)
            flatten_sqdist_correctorest = np.asarray(flatten_sqdist_correctorest)

        for d_naive in flatten_sqdist_naive:
            naive_kp_sqdist_data["noise_var"].append(kp_noise_var)
            naive_kp_sqdist_data["sqdist_kp_naiveest"].append(d_naive)

        for d_corrector in flatten_sqdist_correctorest:
            corrector_kp_sqdist_data["noise_var"].append(kp_noise_var)
            corrector_kp_sqdist_data["sqdist_kp_correctorest"].append(d_corrector)

    # Two dataframes
    naive_sqdist_df = pd.DataFrame(data=naive_kp_sqdist_data)
    corrector_sqdist_df = pd.DataFrame(data=corrector_kp_sqdist_data)

    # make plots for corrector
    plt.figure()
    for var_i in tqdm(range(num_kp_noise_vars)):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]
        # corrector
        make_one_hist(
            corrector_sqdist_df,
            "sqdist_kp_correctorest",
            kp_noise_var,
            label=f"Kpt. Noise Var.={kp_noise_var :.2f}",
            alpha=0.6,
        )
    plt.xlabel("Distance Squared")
    plt.ylabel("Counts")
    plt.title(f"Closest Keypoint-to-Keypoint Distance Distribution\n({object_label}, w/ Corrector)")
    plt.legend(loc="upper right")
    if not save_fig:
        plt.show()
        plt.close()
    else:
        plt.savefig(os.path.join(save_folder, f"{prefix_str}kp_sqdists_corrector_all_kp_noise_vars.pdf"))
        plt.savefig(os.path.join(save_folder, f"{prefix_str}kp_sqdists_corrector_all_kp_noise_vars.png"))
        plt.close()

    # make plots for naive
    plt.figure()
    for var_i in tqdm(range(num_kp_noise_vars)):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]
        # corrector
        make_one_hist(
            naive_sqdist_df, "sqdist_kp_naiveest", kp_noise_var, label=f"Kpt. Noise Var.={kp_noise_var :.2f}", alpha=0.6
        )
    plt.xlabel("Distance Squared")
    plt.ylabel("Counts")
    plt.title(f"Closest Keypoint-to-Keypoint Distance Distribution\n({object_label}, Naive)")
    plt.legend(loc="upper right")
    if not save_fig:
        plt.show()
        plt.close()
    else:
        plt.savefig(os.path.join(save_folder, f"{prefix_str}kp_sqdists_naive_all_kp_noise_vars.pdf"))
        plt.savefig(os.path.join(save_folder, f"{prefix_str}kp_sqdists_naive_all_kp_noise_vars.png"))
        plt.close()

    # compare naive w/ corrector at different kp vars
    for var_i in tqdm(range(num_kp_noise_vars)):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]
        make_one_hist(corrector_sqdist_df, "sqdist_kp_correctorest", kp_noise_var, label=f"Corrector", alpha=0.6)
        make_one_hist(naive_sqdist_df, "sqdist_kp_naiveest", kp_noise_var, label=f"Naive", alpha=0.6)
        plt.xlabel("Distance Squared")
        plt.ylabel("Counts")
        plt.title(
            f"Closest Keypoint-to-Keypoint Distance Distribution\nat Kpt. Noise. Var = {kp_noise_var:.2f} ({object_label})"
        )
        plt.legend(loc="upper right")
        if not save_fig:
            plt.show()
            plt.close()
        else:
            plt.savefig(
                os.path.join(
                    save_folder, f"{prefix_str}kp_sqdists_corrector_vs_naive_kp_noise_var_{kp_noise_var:.2E}.pdf"
                )
            )
            plt.savefig(
                os.path.join(
                    save_folder, f"{prefix_str}kp_sqdists_corrector_vs_naive_kp_noise_var_{kp_noise_var:.2E}.png"
                )
            )
            plt.close()

    return naive_sqdist_df, corrector_sqdist_df


def plot_percent_certifiable_at_noise_vars(
    data_payload,
    certi_data,
    save_folder="./",
    save_fig=True,
    fig_basename="fraction_certifiable_vs_kp_noise_vars",
    object_label=None,
):
    """Generate a line plot of percent certifiable at different noise vars"""
    x = []
    naive_y = []
    corrector_y = []
    num_kp_noise_vars = data_payload["sqdist_kp_naiveest"].shape[0]
    for var_i in range(num_kp_noise_vars):
        kp_noise_var = data_payload["kp_noise_var_range"][var_i]

        # naive
        cert_flags_naive = certi_data["certi_naive"][var_i]
        frac_certified_naive = np.count_nonzero(cert_flags_naive) / float(len(cert_flags_naive))

        # corrector
        cert_flags_corrector = certi_data["certi_corrector"][var_i]
        frac_certified_corrector = np.count_nonzero(cert_flags_corrector) / float(len(cert_flags_corrector))

        x.append(kp_noise_var)
        naive_y.append(frac_certified_naive)
        corrector_y.append(frac_certified_corrector)

    plt.figure()
    plt.plot(x, naive_y, "o-", label="Naive")
    plt.plot(x, corrector_y, "x-", label="Corrector")
    plt.title(f"{object_label} Fraction Certifiable")
    plt.xlabel("Kpt. Noise Var")
    plt.ylabel("Fraction of Certifiable")
    plt.legend()
    if not save_fig:
        plt.show()
    else:
        plt.savefig(os.path.join(save_folder, f"{fig_basename}.pdf"))
        plt.savefig(os.path.join(save_folder, f"{fig_basename}.png"))
    plt.close()

    return


def plot_certification_failure_distribution(
    data_payload,
    certi_naive_failure_modes,
    certi_corrector_failure_modes,
    save_folder="./",
    save_fig=True,
    fig_basename="failure_modes",
    object_label=None,
):
    """Plot the distribution of failure modes for certification"""
    num_kp_noise_vars = data_payload["sqdist_kp_naiveest"].shape[0]
    kp_noise_vars = data_payload["kp_noise_var_range"]

    def plot_failure_modes_for_one_method(failure_modes, title, save_fig, base_folder, file_basename):
        fig, ax = plt.subplots()

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

        width = 0.35
        ax.bar(kp_noise_vars, pt_fractions, width, label="Chamfer Distance Only Failures")
        ax.bar(kp_noise_vars, kp_fractions, width, bottom=pt_fractions, label="Kpt. Distance Only Failures")
        ax.bar(kp_noise_vars, both_failure_fractions, width, bottom=pt_fractions + kp_fractions, label="Both Failures")

        ax.set_ylabel("Fraction of Total Failures")
        ax.set_xlabel("Kpt. Noise Var.")
        ax.set_title(title)
        ax.legend(facecolor="white", framealpha=1, frameon=True)

        if not save_fig:
            plt.show()
        else:
            plt_save_figures(file_basename, base_folder)
        plt.close()

    # corrector
    f_basename = f"corrector_{fig_basename}"
    title = f"Certification Failure Modes\n({object_label}, w/ Corrector)"
    plot_failure_modes_for_one_method(certi_corrector_failure_modes, title, save_fig, save_folder, f_basename)

    # naive
    f_basename = f"naive_{fig_basename}"
    title = f"Certification Failure Modes\n({object_label}, Naive)"
    plot_failure_modes_for_one_method(certi_naive_failure_modes, title, save_fig, save_folder, f_basename)

    return


def plot_instance_histogram(sqdists, save_folder="./", save_fig=True, file_basename="instance_histogram", title=""):
    """Plot relevant histograms for a single instance"""

    fig, axs = plt.subplots(2, 1)
    axs[0].hist(sqdists, bins=100, log=True)
    axs[0].set_ylabel("Counts")
    axs[0].grid(True)
    axs[0].set_title(title)

    axs[1].hist(sqdists[sqdists < np.median(sqdists) + 0.5], bins=100, log=True)
    axs[1].set_ylabel("Counts")
    axs[1].set_xlabel("Squared Distances (Trimmed)")
    axs[1].grid(True)
    if not save_fig:
        plt.show()
        plt.close()
    else:
        plt_save_figures(file_basename, save_folder, formats=["png"])
        plt.close()


def plot_instances_histograms(
    indices,
    kp_sqdists,
    pt_sqdists,
    save_folder="./",
    save_fig=True,
    file_basename="instance_histogram",
    title_suffix="",
):
    """Plot keypoint and p2p histograms given indices and data"""
    for idx in indices:
        fname_base = generate_filename()
        c_kp_sqdists = kp_sqdists[idx, :]
        c_pt_sqdists = pt_sqdists[idx, :]

        t1 = f"Keypoint-to-Keypoint Squared Distances {title_suffix}"
        # keypoint plot
        plot_instance_histogram(
            c_kp_sqdists,
            save_folder=save_folder,
            save_fig=save_fig,
            file_basename=f"kp_{file_basename}_{fname_base}",
            title=t1,
        )

        # p2p plot
        t2 = f"Point-to-Point Squared Distances {title_suffix}"
        plot_instance_histogram(
            c_pt_sqdists,
            save_folder=save_folder,
            save_fig=save_fig,
            file_basename=f"pt_{file_basename}_{fname_base}",
            title=t2,
        )

    return


def random_instances_histograms(
    data_payload, num_random_samples=5, certi_masks=None, save_fig=True, save_folder="./", object_label=None
):
    """Generate kp distance / p2p distances of some random instances"""

    def gen_rand_indices_or_all(data_array, n):
        """Sample random indices; if not enough points, return all indices"""
        if len(data_array) > n:
            return np.random.choice(data_array, n)
        else:
            return np.asarray(data_array)

    assert data_payload["sqdist_kp_naiveest"].shape == data_payload["sqdist_kp_correctorest"].shape
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
        n_samples = pt_sqdist_naive.shape[0]

        assert n_samples == kp_sqdist_naive.shape[0]

        # generate sample indices
        # sample from all
        rand_indices = np.random.choice(n_samples, min(num_random_samples, n_samples))

        # sample random certified samples
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

        # generate plots
        # given indices, generate plots
        # for random samples: plot corrector kp dists, sq dists and naive kp dists, sq dists
        # plot_instances_histograms(
        #    rand_indices, kp_sqdists=kp_sqdist_naive, pt_sqdists=pt_sqdist_naive, save_fig=save_fig
        # )
        # plot_instances_histograms(
        #    rand_indices, kp_sqdists=kp_sqdist_correctorest, pt_sqdists=pt_sqdist_correctorest, save_fig=save_fig
        # )

        # for certified corrector results: plot corrector kp dists and sq dists
        plot_instances_histograms(
            certi_corrector_rand_indices,
            kp_sqdists=kp_sqdist_correctorest,
            pt_sqdists=pt_sqdist_correctorest,
            save_fig=save_fig,
            title_suffix=f"({object_label}, Certified, w/ Corrector)",
            save_folder=save_folder,
            file_basename="cert_corrector_inst_histogram",
        )

        # for certified naive results: plot naive kp dists and sq dists
        plot_instances_histograms(
            certi_naive_rand_indices,
            kp_sqdists=kp_sqdist_naive,
            pt_sqdists=pt_sqdist_naive,
            save_fig=save_fig,
            title_suffix=f"({object_label}, Certified, Naive)",
            save_folder=save_folder,
            file_basename="cert_naive_inst_histogram",
        )

        # for non certified corrector results: plot corrector kp dists and sq dists
        plot_instances_histograms(
            not_certi_corrector_rand_indices,
            kp_sqdists=kp_sqdist_correctorest,
            pt_sqdists=pt_sqdist_correctorest,
            save_fig=save_fig,
            title_suffix=f"({object_label}, Not Certified, Corrector)",
            save_folder=save_folder,
            file_basename="not_cert_corrector_inst_histogram",
        )

        # for non certified naive results: plot naive kp dists and sq dists
        plot_instances_histograms(
            not_certi_naive_rand_indices,
            kp_sqdists=kp_sqdist_naive,
            pt_sqdists=pt_sqdist_naive,
            save_fig=save_fig,
            title_suffix=f"({object_label}, Not Certified, Naive)",
            save_folder=save_folder,
            file_basename="not_cert_naive_inst_histogram",
        )

    return


if __name__ == "__main__":
    print("Generate histogram plots for corrector analysis.")
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datafile",
        help="path of the data file wrt to repo root",
        default="local_data/corrector_analysis/ycbv.train.real/obj_000005/20221008_235859_torch-gd-accel_experiment.pickle",
        type=str,
    )

    parser.add_argument(
        "--save_folder",
        help="path to the folder in which we save figures",
        default="./figures",
        type=str,
    )

    parser.add_argument("--object_label", help="object label in the dataset", default="obj_000005", type=str)

    args = parser.parse_args()
    print("Plotting histograms for: ")
    print(args)

    dump_path = os.path.join(args.save_folder, args.object_label)
    safely_make_folders([dump_path])

    # load data files
    filename = BASE_DIR / args.datafile
    data_payload = load_data(filename)

    ## plot histogram of chamfer distances at different noise var
    #print("Plotting Point-to-Point Chamfer Distances Histograms")
    #pointwise_distances_histogram_at_noise_vars(
    #    data_payload, save_fig=True, save_folder=dump_path, object_label=args.object_label
    #)

    ## plot histogram of keypoint-to-keypoint distances at different noise var
    #print("Plotting Keypoint-to-Keypoint Distances Histograms")
    #kp_distances_histogram_at_noise_vars(
    #    data_payload, save_fig=True, save_folder=dump_path, object_label=args.object_label
    #)

    # investigate certification through histograms
    test_epsilon = 0.06
    certi_naive, certi_corrector, certi_naive_failure_modes, certi_corrector_failure_modes = get_certified_instances(
        payload=data_payload, epsilon=test_epsilon
    )
    not_certi_naive, not_certi_corrector = torch.logical_not(certi_naive), torch.logical_not(certi_corrector)
    certi_masks = {"certi_naive": certi_naive.cpu().numpy(), "certi_corrector": certi_corrector.cpu().numpy()}
    not_certi_masks = {"certi_naive": certi_naive.cpu().numpy(), "certi_corrector": certi_corrector.cpu().numpy()}

    pointwise_distances_histogram_at_noise_vars(
        data_payload,
        sample_mask=certi_masks,
        save_fig=True,
        save_folder=dump_path,
        prefix_str="cert_",
        object_label=args.object_label,
    )
    kp_distances_histogram_at_noise_vars(
        data_payload,
        sample_mask=certi_masks,
        save_fig=True,
        save_folder=dump_path,
        prefix_str="cert_",
        object_label=args.object_label,
    )

    # percentage certifiable through different epsilon thresholds: naiv & corrector
    plot_percent_certifiable_at_noise_vars(
        data_payload,
        certi_masks,
        save_folder=dump_path,
        save_fig=True,
        fig_basename=f"fraction_certifiable_vs_kp_noise_vars_eps{int(test_epsilon*1000):03d}",
        object_label=args.object_label,
    )

    # failure mode distribution (kp vs pt)
    plot_certification_failure_distribution(
        data_payload,
        certi_naive_failure_modes,
        certi_corrector_failure_modes,
        save_folder=dump_path,
        save_fig=True,
        fig_basename=f"cert_failure_modes_fractions_eps{int(test_epsilon*1000):03d}",
        object_label=args.object_label,
    )

    # random instances histogram
    print("Sample random instances for plots.")
    inst_path = os.path.join(dump_path, "instances")
    safely_make_folders([inst_path])
    random_instances_histograms(
        data_payload,
        num_random_samples=5,
        certi_masks=certi_masks,
        save_fig=True,
        save_folder=inst_path,
        object_label=args.object_label,
    )
