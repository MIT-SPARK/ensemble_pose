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

plt.rcParams.update({"font.size": 15})

def load_all_objs_data(dataset_name, data_base_name, base_path="./eval/data/"):
    """Load YCBV results"""
    object_ids = BOP_MODEL_INDICES[dataset_name].keys()

    all_data = {}
    for obj_id in object_ids:
        fname = f"{data_base_name}_{dataset_name}_{obj_id}.pkl"
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
    base_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), "./plots_rss"),
    name2legend=None,
    x_label=None,
    y_label=None,
    hue_order=None,
):
    noise_df = pd.DataFrame.from_records(noise_data)
    noise_df["outlier_ratio"] *= 100
    noise_df["outlier_ratio"] = noise_df["outlier_ratio"].astype(int)
    fix, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid(True, which="both", axis="y")
    ax.minorticks_on()
    ax = sns.boxplot(noise_df, x=x, y=y, hue="method", orient="v", ax=ax, palette="bright", fliersize=1, hue_order=hue_order)

    for i, box in enumerate([p for p in ax.patches if not p.get_label()]):
        color = box.get_facecolor()
        box.set_edgecolor(color)
        box.set_facecolor((0, 0, 0, 0))
        # iterate over whiskers, fliers and median lines
        for j in range(6 * i, 6 * (i + 1)):
            ax.lines[j].set_color(color)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    [ax.axvline(x + .5, color='grey') for x in ax.get_xticks()]

    ax.set_yscale("log")
    if name2legend is not None:
        for i in range(len(ax.legend_.texts)):
            ctxt = ax.legend_.texts[i]
            ax.legend_.texts[i].set_text(name2legend[ctxt._text])
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=None,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.3,
        borderpad=0.2,
    )
    if x_label is not None:
        ax.set(xlabel=x_label)
    if y_label is not None:
        ax.set(ylabel=y_label)

    if not save_fig:
        plt.show()
    else:
        plt_save_figures(f"{base_filename}", base_folder, formats=["pdf", "png"])
    plt.close()


def boxplot_all_objects(
    dataset_name,
    all_objs_data,
    x="noise_sigma",
    y="kp_pred2gt_dist",
    base_filename="kpp_err_v_noise",
    output_folder=f"./plots_rss/",
    save_fig=False,
    name2legend=None,
    hue_order=None,
):
    object_ids = BOP_MODEL_INDICES[dataset_name].keys()
    for obj_id in object_ids:
        try:
            c_data = all_objs_data[obj_id]
            boxplot_data(
                c_data,
                x=x,
                y=y,
                save_fig=save_fig,
                base_filename=f"{base_filename}_{obj_id}",
                base_folder=output_folder,
                name2legend=name2legend,
                hue_order=hue_order,
            )
        except KeyError:
            print(f"Missing {obj_id} -- skipping")
    return


def avg_box_plot(
    dataset_name,
    all_objs_data,
    x="noise_sigma",
    y="kp_pred2gt_dist",
    base_filename="kpp_err_v_noise",
    output_folder=f"./plots_rss/",
    save_fig=False,
    name2legend=None,
    x_label=None,
    y_label=None,
    hue_order=None,
):
    object_ids = BOP_MODEL_INDICES[dataset_name].keys()
    merged_data = []
    for obj_id in object_ids:
        c_data = all_objs_data[obj_id]
        if isinstance(merged_data, list):
            merged_data.extend(c_data)

    # box plot
    boxplot_data(
        merged_data,
        x=x,
        y=y,
        save_fig=save_fig,
        base_filename=f"{base_filename}_all_objs",
        base_folder=output_folder,
        name2legend=name2legend,
        x_label=x_label,
        y_label=y_label,
        hue_order=hue_order,
    )

    return


if __name__ == "__main__":
    print("Plotting eval sim data")

    eval_dir = pathlib.Path(__file__).parent.resolve()
    base_path = os.path.join(eval_dir, "data_corrector_ablations", "ycbv")
    logging.info(f"Importing data from {base_path}")

    method_names = [
        "no_corrector",
        "robust_corrector",
        "nonrobust_corrector",
    ]
    name2legend = {
        "no_corrector": "No Corrector",
        "robust_corrector": "Robust Corrector",
        "nonrobust_corrector": "Non-robust Corrector",
    }
    hue_order = ["robust_corrector", "nonrobust_corrector", "no_corrector"]

    ycbv_correctors_results = load_all_objs_data("ycbv", "corrector_err_data", base_path=base_path)

    boxplot_all_objects(
        "ycbv",
        ycbv_correctors_results,
        x="outlier_ratio",
        # y="kp_pred2gt_dist",
        y="chamfer_dist",
        base_filename="chamfer_dists_v_outliers_correctors",
        save_fig=True,
        output_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), "plots_rss_correctors/ycbv"),
        name2legend=name2legend,
        hue_order=hue_order,
    )

    # over all objects
    avg_box_plot(
        "ycbv",
        ycbv_correctors_results,
        x="outlier_ratio",
        # y="kp_pred2gt_dist",
        y="chamfer_dist",
        base_filename="chamfer_dists_v_outliers_correctors",
        save_fig=True,
        output_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), "plots_rss_correctors/ycbv"),
        name2legend=name2legend,
        x_label="Outlier Rate (%)",
        # y_label="Kpt. Pred. Error",
        y_label="Normalized ADD-S Score",
        hue_order=hue_order
    )