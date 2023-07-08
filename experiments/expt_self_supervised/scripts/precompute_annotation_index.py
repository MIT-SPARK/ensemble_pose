import os
import yaml
import argparse
from tqdm import tqdm

from expt_self_supervised.training_utils import *
from expt_self_supervised.proposed_model import load_c3po_cad_models
from datasets.bop_constants import BOP_SPLIT_DIRS
from datasets.pose import ObjectPoseDatasetBase


if __name__ == "__main__":
    """
    Example command: 
    python precompute_annotation_index.py ycbv.test --config=../configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path of the config file",
        default=f"./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml",
        type=str,
    )
    args = parser.parse_args()

    ycbv_ds_names = ["ycbv.train.real", "ycbv.train.synt", "ycbv.test", "ycbv.pbr"]
    tless_ds_names = ["tless.primesense.train", "tless.primesense.test", "tless.pbr"]

    # load config params
    config_params_file = args.config
    cfg = load_yaml_cfg(config_params_file)

    # load scene dataset
    for ds_name in tqdm(ycbv_ds_names):
        scene_ds_train = make_scene_dataset(ds_name, bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)
    for ds_name in tqdm(tless_ds_names):
        scene_ds_train = make_scene_dataset(ds_name, bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)
