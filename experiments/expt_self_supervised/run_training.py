import argparse
import datetime
import logging
import os
import shutil
import sys
import tempfile
import torch

from datasets import bop_constants
from training_utils import load_yaml_cfg
from utils.file_utils import set_up_logger, safely_make_folders
from utils.math_utils import set_all_random_seeds

# different training loops
from supervised_training import train_detector as train_detector_supervised
from supervised_synth_training import train_detector as train_detector_supervised_synth
from supervised_synth_single_obj_training import train_detector as train_detector_supervised_synth_single_obj
from self_supervised_single_obj_training import train_detector as train_detector_self_supervised_single_obj
from eval_single_obj import eval_detector as eval_detector_single_obj


def check_valid_train_mode(tmode):
    """Ensure the train mode is valid"""
    assert tmode in [
        "supervised",
        "synth_supervised",
        "synth_supervised_single_obj",
        "self_supervised_single_obj",
        "single_obj_eval",
    ]


def check_valid_cfg(args, cfg):
    """make sure cfg makes sense"""
    dataset = args.dataset
    assert cfg["dataset"] == dataset
    train_mode = cfg["train_mode"]
    check_valid_train_mode(train_mode)

    # check keys existence
    assert "c3po" in cfg.keys()

    # check use corrector flag: on only when self-supervised
    if train_mode != "single_obj_eval":
        if "self" not in train_mode:
            assert not cfg["c3po"]["use_corrector"]

    # update labels2id mappings
    if "detector" in cfg.keys():
        cfg["detector"]["label_to_category_id"] = bop_constants.BOP_LABEL_TO_CATEGORY_ID[dataset]
        cfg["detector"]["category_id_to_label"] = {v: k for k, v in cfg["detector"]["label_to_category_id"].items()}

    return


def prepare_data_save_folders(model_name, exp_launch_time, config_params_file, cfg):
    dataset = cfg["dataset"]

    if model_name == "all":
        model_names = [x for x in list(bop_constants.BOP_MODEL_INDICES["dataset"].keys())]
    else:
        model_names = [model_name]

    cfg["save_folder"] = dict()
    for obj_name in model_names:
        save_path = os.path.join(cfg["save_folder"], dataset, obj_name)
        safely_make_folders([save_path])
        cfg["save_folder"][obj_name] = save_path

        # copy config param yaml file to the save folder
        config_copy_path = os.path.join(save_path, "config.yml")
        if os.path.exists(config_copy_path):
            os.remove(config_copy_path)
        shutil.copy2(config_params_file, os.path.join(save_path, "config.yml"))

    cfg["timestamp"] = exp_launch_time
    return


def train_pipelines(dataset, cfg, **kwargs):
    train_mode = cfg["train_mode"]
    detector_type = cfg["c3po"]["detector_type"]

    if train_mode == "supervised":
        model_name = kwargs["model_name"]
        logging.info(f"Training on {dataset}; model name: {model_name}; train_mode: {train_mode}")
        kwargs = {
            "cfg": cfg,
            "dataset": dataset,
            "detector_type": detector_type,
            "model_id": model_name,
        }
        train_detector_supervised(**kwargs)
    elif train_mode == "synth_supervised":
        logging.info(f"Training on {dataset}; train_mode: {train_mode}")
        kwargs = {
            "cfg": cfg,
            "dataset": dataset,
            "detector_type": detector_type,
            "resume_run": kwargs["resume_run"],
            "multimodel_checkpoint_path": kwargs["multimodel_checkpoint_path"],
        }
        train_detector_supervised_synth(**kwargs)
    elif train_mode == "synth_supervised_single_obj":
        model_name = kwargs["model_name"]
        logging.info(f"Training on {dataset}; model name: {model_name}; train_mode: {train_mode}")
        kwargs = {
            "cfg": cfg,
            "dataset": dataset,
            "detector_type": detector_type,
            "resume_run": kwargs["resume_run"],
            "model_id": model_name,
        }
        train_detector_supervised_synth_single_obj(**kwargs)
    elif train_mode == "self_supervised_single_obj":
        model_name = kwargs["model_name"]
        logging.info(f"Training on {dataset}; model name: {model_name}; train_mode: {train_mode}")
        kwargs = {
            "cfg": cfg,
            "dataset": dataset,
            "detector_type": detector_type,
            "resume_run": kwargs["resume_run"],
            "checkpoint_path": kwargs["checkpoint_path"],
            "model_id": model_name,
        }
        train_detector_self_supervised_single_obj(**kwargs)
    elif train_mode == "single_obj_eval":
        model_name = kwargs["model_name"]
        logging.info(f"Training on {dataset}; model name: {model_name}; train_mode: {train_mode}")
        kwargs = {
            "cfg": cfg,
            "dataset": dataset,
            "detector_type": detector_type,
            "resume_run": kwargs["resume_run"],
            "checkpoint_path": kwargs["checkpoint_path"],
            "model_id": model_name,
        }
        eval_detector_single_obj(**kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    """
    Supervised training on synthetic dataset from YCBV and TLESS datasets

    Intend to support YCB-Video and TLESS datasets.

    usage:
    >> python run_supervised_training.py ycbv --config=./configs/supervised_ycbv.yml
    """
    exp_launch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="specify the dataset.", type=str)
    parser.add_argument(
        "--model_id",
        help="specify the object id (not used for synth supervised training)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--config",
        help="path of the config file",
        default=f"./configs/supervised.yml",
        type=str,
    )
    parser.add_argument(
        "--logfile_name",
        help="name of the log file",
        default=f"expt_supervised_{exp_launch_time}.log",
        type=str,
    )
    parser.add_argument(
        "--logfile_dir",
        help="folder to dump the log file",
        default=f"./logs",
        type=str,
    )

    # settings on creating timestamped results
    parser.add_argument("--zip_models", help="zip the save directory for backup", action="store_true")
    parser.add_argument(
        "--no_zip_models",
        help="do not zip the save directory for backup",
        dest="zip_models",
        action="store_false",
    )
    parser.set_defaults(zip_models=True)

    parser.add_argument("--resume_run", help="path to load the previous checkpoint", action="store_true")
    parser.set_defaults(resume_run=False)
    parser.add_argument("--checkpoint_path", help="path to checkpoint")

    parser.add_argument("--rng_seed", help="RNG seed", default=42, type=int)
    parser.add_argument("--debug", help="enable to show debug level messages", action="store_true")
    parser.add_argument("--reproducible", help="enable to use fixed seed & deterministic algs", action="store_true")

    args = parser.parse_args()

    torch.set_printoptions(precision=10)
    if args.reproducible:
        set_all_random_seeds(args.rng_seed)
        # torch.use_deterministic_algorithms(True)

    # handle https://github.com/pytorch/pytorch/issues/77527
    torch.backends.cuda.preferred_linalg_library("cusolver")

    # load config params
    config_params_file = args.config
    cfg = load_yaml_cfg(config_params_file, object_id=args.model_id)

    check_valid_cfg(args, cfg)

    # update save path by timestamp
    logging.info(f"Train mode: {cfg['train_mode']}")
    if cfg["train_mode"] == "supervised":
        assert args.model_id is not None
        model_id = args.model_id
        dataset = args.dataset
        save_path = os.path.join(cfg["save_folder"], dataset, model_id)
        safely_make_folders([save_path])
        pipeline_kwargs = dict(dataset=dataset, model_name=model_id, cfg=cfg)
    elif cfg["train_mode"] == "synth_supervised":
        # for synth supervision, create a single timestamped folder for saving the model
        dataset = args.dataset
        save_path = os.path.join(cfg["save_folder"], dataset, exp_launch_time)
        pipeline_kwargs = dict(
            dataset=dataset,
            cfg=cfg,
            resume_run=args.resume_run,
            multimodel_checkpoint_path=args.checkpoint_path,
        )
        safely_make_folders([save_path])
    elif cfg["train_mode"] == "synth_supervised_single_obj":
        # train on data from a single object only
        assert args.model_id is not None
        dataset = args.dataset
        save_path = os.path.join(cfg["save_folder"], dataset, args.model_id)
        pipeline_kwargs = dict(
            dataset=dataset,
            cfg=cfg,
            resume_run=args.resume_run,
            model_name=args.model_id,
        )
        safely_make_folders([save_path])
    elif cfg["train_mode"] == "self_supervised_single_obj":
        assert args.model_id is not None
        dataset = args.dataset
        save_path = os.path.join(cfg["save_folder"], dataset, args.model_id)
        pipeline_kwargs = dict(
            dataset=dataset,
            cfg=cfg,
            resume_run=args.resume_run,
            model_name=args.model_id,
            checkpoint_path=args.checkpoint_path,
        )
        safely_make_folders([save_path])
    elif cfg["train_mode"] == "single_obj_eval":
        assert args.model_id is not None
        args.zip_models = False
        dataset = args.dataset
        save_path = os.path.join(cfg["save_folder"], dataset, args.model_id)
        pipeline_kwargs = dict(
            dataset=dataset,
            cfg=cfg,
            resume_run=args.resume_run,
            model_name=args.model_id,
            checkpoint_path=args.checkpoint_path,
        )
        safely_make_folders([save_path])
    else:
        raise NotImplementedError

    cfg["timestamp"] = exp_launch_time
    cfg["save_folder"] = save_path

    # copy config param yaml file to the save folder
    shutil.copy2(config_params_file, os.path.join(save_path, "config.yml"))
    shutil.copy2(cfg['certifier']['objects_thresholds_path'], os.path.join(save_path, "certifier_objects_params.yml"))

    # logging configurations
    # note: logging files are saved both in temp and the save folder
    set_up_logger(
        [sys.stdout],
        [
            os.path.join(tempfile.gettempdir(), args.logfile_name),
            os.path.join(args.logfile_dir, args.logfile_name),
        ],
        level=(logging.DEBUG if args.debug else logging.INFO),
    )
    logging.info(f"Save folder: {cfg['save_folder']}")

    train_pipelines(**pipeline_kwargs)

    if args.zip_models:
        logging.info(f"Zipping folder at {save_path} for backup")
        if cfg["train_mode"] == "supervised":
            shutil.make_archive(
                f"supervised_models_{dataset}_{pipeline_kwargs['model_name']}_{exp_launch_time}", "zip", save_path
            )
        elif cfg["train_mode"] == "synth_supervised":
            shutil.make_archive(f"synth_supervised_models_{dataset}_{exp_launch_time}", "zip", save_path)
        elif cfg["train_mode"] == "synth_supervised_single_obj":
            shutil.make_archive(f"synth_supervised_{dataset}_{args.model_id}_{exp_launch_time}", "zip", save_path)
        elif cfg["train_mode"] == "self_supervised_single_obj":
            shutil.make_archive(f"self_supervised_{dataset}_{args.model_id}_{exp_launch_time}", "zip", save_path)
        else:
            raise NotImplementedError
