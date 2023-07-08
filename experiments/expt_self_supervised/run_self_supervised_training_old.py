import argparse
import datetime
import jinja2
import logging
import os
import shutil
import sys
import tempfile
import torch
import yaml
from pathlib import Path

from self_supervised_training_depth_only import train_detector as train_detector_self_supervised_depth_only
from self_supervised_training_rgb_only import train_detector as train_detector_self_supervised_rgb_only
from utils.file_utils import set_up_logger, safely_make_folders
from utils.math_utils import set_all_random_seeds


def check_valid_train_mode(tmode):
    """Ensure the train mode is valid"""
    assert tmode in ["self_supervised", "supervised"]


def check_valid_cfg(args, cfg):
    """make sure cfg makes sense"""
    dataset = args.dataset
    assert cfg["dataset"] == dataset
    train_mode = cfg["train_mode"]
    check_valid_train_mode(train_mode)

    # check keys existence
    assert "c3po" in cfg.keys()

    # check use corrector flag: on only when self-supervised
    if train_mode == "self_supervised":
        assert cfg["c3po"]["use_corrector"]
    elif train_mode == "supervised":
        assert not cfg["c3po"]["use_corrector"]

    # check input resize
    if "tless" in args.config:
        assert cfg["training"]["input_resize"] == [540, 720]
    elif "ycbv" in args.config:
        assert cfg["training"]["input_resize"] == [480, 640]
    elif "bop-" in args.config:
        assert cfg["training"]["input_resize"] is None
    else:
        raise ValueError

    # check label to category id
    assert "detector" in cfg.keys()
    assert "label_to_category_id" in cfg["detector"].keys()
    # build inverse index
    cfg["detector"]["category_id_to_label"] = {v: k for k, v in cfg["detector"]["label_to_category_id"].items()}

    return


def load_yaml_cfg(config_params_file):
    stream = open(config_params_file, "r")
    template = jinja2.Template(stream.read().rstrip())
    processed_yaml = template.render(
        project_root=Path(__file__).parent.parent.parent.resolve(), exp_root=Path(__file__).parent.resolve()
    )
    cfg = yaml.full_load(processed_yaml)
    return cfg


def train_pipelines(dataset, cfg):
    train_mode = cfg["train_mode"]

    logging.info(f"Training on {dataset}; train_mode: {train_mode}")
    kwargs = {
        "cfg": cfg,
        "dataset": dataset,
    }
    if train_mode == "self_supervised" and len(cfg["models_to_use"]) == 1 and cfg["models_to_use"][0] == "c3po_multi":
        train_detector_self_supervised_depth_only(detector_type="c3po_multi", **kwargs)
    elif (
        train_mode == "self_supervised"
        and len(cfg["models_to_use"]) == 1
        and cfg["models_to_use"][0] == "cosypose_coarse_refine"
    ):
        train_detector_self_supervised_rgb_only(detector_type="cosypose_coarse_refine", **kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    """
    Self-supervised experiment.

    Intend to support YCB-Video and TLESS datasets.

    Train modes:
    - supervised: supervised training on synthetic dataset from YCBV and TLESS datasets
    - self-supervised:

    usage:
    >> python run_self_supervised_training.py ycbv --config=./configs/self_supervised_ycbv.yml
    """
    exp_launch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="specify the dataset.", type=str)
    parser.add_argument(
        "--config",
        help="path of the config file",
        default=f"./configs/self_supervised_ycbv_depth_only.yml",
        type=str,
    )
    parser.add_argument(
        "--logfile_name",
        help="name of the log file",
        default=f"expt_self_supervised_{exp_launch_time}.log",
        type=str,
    )

    # settings on creating timestamped results
    parser.add_argument(
        "--timestamp_save_dir", help="create a timestamp dir in the save directory to save results", action="store_true"
    )
    parser.add_argument(
        "--no_timestamp_save_dir",
        help="do not create a timestamp dir in the save directory to save results",
        dest="timestamp_save_dir",
        action="store_false",
    )
    parser.set_defaults(timestamp_save_dir=False)

    parser.add_argument("--rng_seed", help="RNG seed", default=0, type=int)
    parser.add_argument("--debug", help="enable to show debug level messages", action="store_true")
    parser.add_argument("--reproducible", help="enable to use fixed seed & deterministic algs", action="store_true")

    args = parser.parse_args()

    torch.set_printoptions(precision=10)
    if args.reproducible:
        set_all_random_seeds(args.rng_seed)
        torch.use_deterministic_algorithms(True)

    # handle https://github.com/pytorch/pytorch/issues/77527
    torch.backends.cuda.preferred_linalg_library("cusolver")

    # load config params
    config_params_file = args.config
    cfg = load_yaml_cfg(config_params_file)

    check_valid_cfg(args, cfg)

    # update save path by timestamp
    dataset = args.dataset
    if args.timestamp_save_dir:
        timestamped_save_path = os.path.join(cfg["save_folder"], dataset, exp_launch_time)
    else:
        timestamped_save_path = os.path.join(cfg["save_folder"], dataset)
    safely_make_folders([timestamped_save_path])
    cfg["timestamp"] = exp_launch_time
    cfg["save_folder"] = timestamped_save_path

    # copy config param yaml file to the save folder
    shutil.copy2(config_params_file, os.path.join(timestamped_save_path, exp_launch_time + "_config.yml"))

    # logging configurations
    # note: logging files are saved both in temp and the save folder
    set_up_logger(
        [sys.stdout],
        [
            os.path.join(tempfile.gettempdir(), args.logfile_name),
            os.path.join(cfg["save_folder"], args.logfile_name),
        ],
        level=(logging.DEBUG if args.debug else logging.INFO),
    )
    logging.info(f"Save folder: {cfg['save_folder']}")

    train_pipelines(dataset=dataset, cfg=cfg)
