import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import seaborn as sns
import torch.utils.data

from datasets.bop_constants import BOP_MODEL_INDICES
from utils.visualization_utils import plt_save_figures


def load_all_objs_data(dataset_name, data_base_name, base_path="./eval/data/", method_name="pt_transformer_bn"):
    """Load YCBV results"""
    object_ids = BOP_MODEL_INDICES[dataset_name].keys()

    all_data = {}
    for obj_id in object_ids:
        fname = f"{data_base_name}_{dataset_name}_{obj_id}_{method_name}.pkl"
        try:
            logging.info(f"Loading {fname}")
            fp = open(os.path.join(base_path, fname), "rb")
            data = pickle.load(fp)

            def clean_entry(x):
                new_x = {}
                for k, v in x.items():
                    if torch.is_tensor(v):
                        new_x[k] = np.array(v.to("cpu"))
                        if new_x[k].size == 1:
                            new_x[k] = new_x[k].item()
                    else:
                        new_x[k] = v
                return new_x

            cleaned_data = [clean_entry(x) for x in data]
            all_data[obj_id] = cleaned_data
        except:
            logging.warning(f"Missing {fname}. Skipping it.")

    return all_data


def combine_data(methods_data):
    """Combine all objects' exp data from multiple methods together"""
    final_data = {}
    obj_ids = methods_data[0].keys()
    for obj_id in obj_ids:
        value = []
        for m_idx in range(len(methods_data)):
            try:
                value.extend(methods_data[m_idx][obj_id])
            except:
                continue
        final_data[obj_id] = value

    return final_data


def boxplot_data(
    noise_data,
    x="noise_sigma",
    y="kp_pred2gt_dist",
    save_fig=False,
    base_filename="kp_err_v_noise",
    base_folder="./eval/plots",
):
    noise_df = pd.DataFrame.from_records(noise_data)
    ax = sns.boxplot(noise_df, x=x, y=y, hue="method", orient="v")
    ax.set_yscale("log")
    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{base_filename}", base_folder, formats=["pdf", "png"])
    plt.close()


def boxplot_all_objects(
    dataset_name, all_objs_data, x="noise_sigma", y="kp_pred2gt_dist", base_filename="kpp_err_v_noise", save_fig=False
):
    object_ids = BOP_MODEL_INDICES[dataset_name].keys()
    for obj_id in object_ids:
        c_data = all_objs_data[obj_id]
        boxplot_data(
            c_data,
            x=x,
            y=y,
            save_fig=save_fig,
            base_filename=f"{base_filename}_{obj_id}",
            base_folder=f"./eval/plots/{dataset_name}",
        )
    return


if __name__ == "__main__":
    print("Plotting eval sim data")

    eval_dir = pathlib.Path(__file__).parent.resolve()

    # plot ycbv noise data (bn & ln)
    ycbv_noise_results_bn = load_all_objs_data(
        "ycbv", "kp_noise_err_data", base_path=os.path.join(eval_dir, "data", "ycbv"), method_name="pt_transformer_bn"
    )
    ycbv_outliers_results_bn = load_all_objs_data(
        "ycbv", "kp_outlier_err_data", base_path=os.path.join(eval_dir, "data", "ycbv"), method_name="pt_transformer_bn"
    )
    ycbv_noise_results_ln = load_all_objs_data(
        "ycbv", "kp_noise_err_data", base_path=os.path.join(eval_dir, "data", "ycbv"), method_name="pt_transformer_ln"
    )
    ycbv_outliers_results_ln = load_all_objs_data(
        "ycbv", "kp_outlier_err_data", base_path=os.path.join(eval_dir, "data", "ycbv"), method_name="pt_transformer_ln"
    )
    ycbv_noise_results_dense = load_all_objs_data(
        "ycbv",
        "kp_noise_err_data",
        base_path=os.path.join(eval_dir, "data", "ycbv"),
        method_name="pt_transformer_dense",
    )
    ycbv_outliers_results_dense = load_all_objs_data(
        "ycbv",
        "kp_outlier_err_data",
        base_path=os.path.join(eval_dir, "data", "ycbv"),
        method_name="pt_transformer_dense",
    )
    ycbv_noise_results_topk = load_all_objs_data(
        "ycbv", "kp_noise_err_data", base_path=os.path.join(eval_dir, "data", "ycbv"), method_name="pt_transformer_topk"
    )
    ycbv_outliers_results_topk = load_all_objs_data(
        "ycbv",
        "kp_outlier_err_data",
        base_path=os.path.join(eval_dir, "data", "ycbv"),
        method_name="pt_transformer_topk",
    )
    ycbv_noise_results_topk_rbst_cntr = load_all_objs_data(
        "ycbv",
        "kp_noise_err_data",
        base_path=os.path.join(eval_dir, "data", "ycbv"),
        method_name="pt_transformer_topk_rbst_cntr",
    )
    ycbv_outliers_results_topk_rbst_cntr = load_all_objs_data(
        "ycbv",
        "kp_outlier_err_data",
        base_path=os.path.join(eval_dir, "data", "ycbv"),
        method_name="pt_transformer_topk_rbst_cntr",
    )
    ycbv_noise_results_rand = load_all_objs_data(
        "ycbv",
        "kp_noise_err_data",
        base_path=os.path.join(eval_dir, "data", "ycbv"),
        method_name="pt_transformer_rand",
    )
    ycbv_outliers_results_rand = load_all_objs_data(
        "ycbv",
        "kp_outlier_err_data",
        base_path=os.path.join(eval_dir, "data", "ycbv"),
        method_name="pt_transformer_rand",
    )

    ycbv_noise_results = combine_data(
        [
            ycbv_noise_results_ln,
            ycbv_noise_results_bn,
            ycbv_noise_results_dense,
            ycbv_noise_results_topk,
            ycbv_noise_results_topk_rbst_cntr,
            ycbv_noise_results_rand,
        ]
    )
    ycbv_outliers_results = combine_data(
        [
            ycbv_outliers_results_ln,
            ycbv_outliers_results_bn,
            ycbv_outliers_results_dense,
            ycbv_outliers_results_topk,
            ycbv_outliers_results_topk_rbst_cntr,
            ycbv_noise_results_rand,
        ]
    )

    # outliers and noise plots (for all methods)
    boxplot_all_objects(
        "ycbv",
        ycbv_noise_results,
        x="noise_sigma",
        y="kp_pred2gt_dist",
        base_filename="kpp_err_v_noise_pt_transformer",
        save_fig=False,
    )
    boxplot_all_objects(
        "ycbv",
        ycbv_outliers_results,
        x="outlier_ratio",
        y="kp_pred2gt_dist",
        base_filename="kpp_err_v_outliers_pt_transformer",
        save_fig=False,
    )

    # plot ycbv outlier data
    # bn only
    # boxplot_all_objects(
    #    "ycbv",
    #    ycbv_noise_results_bn,
    #    x="noise_sigma",
    #    y="kp_pred2gt_dist",
    #    base_filename="kpp_err_v_noise_pt_transformer_bn",
    #    save_fig=True,
    # )
    # boxplot_all_objects(
    #    "ycbv",
    #    ycbv_outliers_results_bn,
    #    x="outlier_ratio",
    #    y="kp_pred2gt_dist",
    #    base_filename="kpp_err_v_outliers_pt_transformer_bn",
    #    save_fig=True,
    # )
