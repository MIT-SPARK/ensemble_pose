import os
import yaml
import argparse

from expt_self_supervised.training_utils import *
from expt_self_supervised.proposed_model import load_c3po_cad_models
from datasets.bop_constants import BOP_SPLIT_DIRS
from datasets.pose import ObjectPoseDatasetBase


if __name__ == "__main__":
    """
    Example command: 
    python precompute_pc_img_datasets.py ycbv.test obj_000001 /home/jnshi/ --config=../configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", help="specify the dataset with split.", type=str)
    parser.add_argument("object_id", help="specify the object ID", type=str)
    parser.add_argument("output_dir", help="specify the output directory (without split/dataset)", type=str)
    parser.add_argument(
        "--config",
        help="path of the config file",
        default=f"./configs/synth_supervised_single_obj/ycbv/depth_rgb_nopretrain.yml",
        type=str,
    )
    args = parser.parse_args()

    # load config params
    config_params_file = args.config
    cfg = load_yaml_cfg(config_params_file)

    dataset = args.ds_name.split(".")[0]
    cfg["dataset"] = dataset

    object_ds, mesh_db_batched = load_objects(cfg)
    _, _, original_cad_model, original_model_keypoints, obj_diameter = load_c3po_cad_models(
        args.object_id, "cpu", output_unit="m", cfg=cfg
    )

    # load scene dataset
    logging.info(f"Loading dataset/split={args.ds_name}.")
    scene_ds_train = make_scene_dataset(args.ds_name, bop_ds_dir=Path(cfg["bop_ds_dir"]), load_depth=True)

    # load pc dataset
    cfg_training = cfg["training"]
    ds_kwargs = dict(
        object_diameter=obj_diameter,
        dataset_name=cfg["dataset"],
        min_area=cfg_training["min_area"],
        pc_size=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
        load_rgb_for_points=cfg_training["load_rgb_for_points"],
        zero_center_pc=cfg_training["zero_center_pc"],
        use_robust_centroid=cfg_training["use_robust_centroid"],
        resample_invalid_pts=cfg_training["resample_invalid_pts"],
        normalize_pc=cfg_training["normalize_pc"],
        load_data_from_cache=True,
        cache_save_dir=args.output_dir
    )

    # this should create the
    ds_train = ObjectPoseDatasetBase(scene_ds_train, args.object_id, **ds_kwargs)

    # dump cfg
    with open(os.path.join(args.output_dir, "config.yml"), 'w') as fp:
        yaml.dump(cfg, fp, default_flow_style=False)

