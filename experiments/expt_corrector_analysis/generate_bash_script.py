import datetime
import os
from pathlib import Path

from datasets.bop_constants import *

BASE_DIR = Path(__file__).parent.parent.parent


def gen_ycbv_cert_plots_script():
    data_folder = BASE_DIR / "local_data/corrector_analysis/ycbv.train.real"
    save_folder = "./figures/ycbv"
    ycbv_objects = list(BOP_MODEL_INDICES["ycbv"].keys())
    bash_str = ""
    for obj_label in ycbv_objects:
        c_folder = data_folder / obj_label

        # find the latest data file
        pickle_files = [c_folder / f for f in os.listdir(c_folder) if f.endswith(".pickle")]
        latest_file = max(pickle_files, key=os.path.getctime)
        cmd = f"python plot_err_plots.py --save_folder={save_folder} --datafile={latest_file} --object_label={obj_label}\n"
        bash_str += cmd
    print("Bash script for ycbv cert plot generated.")
    return bash_str


def gen_ycbv_err_quantile_plots_script():
    data_folder = BASE_DIR / "local_data/corrector_analysis/ycbv.train.real"
    save_folder = "./figures/ycbv"
    ycbv_objects = list(BOP_MODEL_INDICES["ycbv"].keys())
    bash_str = ""
    for obj_label in ycbv_objects:
        c_folder = data_folder / obj_label

        # find the latest data file
        pickle_files = [c_folder / f for f in os.listdir(c_folder) if f.endswith(".pickle")]
        latest_file = max(pickle_files, key=os.path.getctime)
        cmd = f"python plot/plot_err_and_quantiles.py --save_folder={save_folder} --datafile={latest_file} --object_label={obj_label}\n"
        bash_str += cmd
    print("Bash script for ycbv err and quantile plot generated.")
    return bash_str


def gen_tless_cert_plots_script():
    data_folder = BASE_DIR / "local_data/corrector_analysis/tless.primesense.train"
    save_folder = "./figures/tless"
    tless_objects = list(BOP_MODEL_INDICES["tless"].keys())
    bash_str = ""
    for obj_label in tless_objects:
        c_folder = data_folder / obj_label

        # find the latest data file
        pickle_files = [c_folder / f for f in os.listdir(c_folder) if f.endswith(".pickle")]
        latest_file = max(pickle_files, key=os.path.getctime)
        cmd = f"python plot_err_plots.py --save_folder={save_folder} --datafile={latest_file} --object_label={obj_label}\n"
        bash_str += cmd
    print("Bash script for ycbv cert plot generated.")
    return bash_str


def gen_tless_err_quantile_plots_script():
    data_folder = BASE_DIR / "local_data/corrector_analysis/tless.primesense.train"
    save_folder = "./figures/tless"
    ycbv_objects = list(BOP_MODEL_INDICES["tless"].keys())
    bash_str = ""
    for obj_label in ycbv_objects:
        c_folder = data_folder / obj_label

        # find the latest data file
        pickle_files = [c_folder / f for f in os.listdir(c_folder) if f.endswith(".pickle")]
        latest_file = max(pickle_files, key=os.path.getctime)
        cmd = f"python plot/plot_err_and_quantiles.py --save_folder={save_folder} --datafile={latest_file} --object_label={obj_label}\n"
        bash_str += cmd
    print("Bash script for tless err and quantile plot generated.")
    return bash_str


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    ## generate bash script for generating ycbv cert error plots
    # bash_str = gen_ycbv_cert_plots_script()
    # with open(f"scripts/plot_ycbv_cert_{timestamp}.sh", "w") as text_file:
    #    text_file.write(bash_str)

    #bash_str = gen_tless_cert_plots_script()
    #with open(f"scripts/plot_tless_cert_{timestamp}.sh", "w") as text_file:
    #    text_file.write(bash_str)

    bash_str = gen_ycbv_err_quantile_plots_script()
    with open(f"scripts/plot_ycbv_err_and_quantiles_{timestamp}.sh", "w") as text_file:
       text_file.write(bash_str)

    bash_str = gen_tless_err_quantile_plots_script()
    with open(f"scripts/plot_tless_err_and_quantiles_{timestamp}.sh", "w") as text_file:
        text_file.write(bash_str)
