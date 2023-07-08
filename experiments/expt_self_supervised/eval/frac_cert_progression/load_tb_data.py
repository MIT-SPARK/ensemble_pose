import copy

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import seaborn as sns
import torch.utils.data
from scipy.ndimage.filters import gaussian_filter1d

from datasets.bop_constants import BOP_MODEL_INDICES
from utils.visualization_utils import plt_save_figures

from tensorboard.backend.event_processing import event_accumulator

from datasets.bop_constants import BOP_MODEL_INDICES


def smooth_df(df, sigma=20):
    cols_to_filter = ["csy_2d", "csy_3d", "csy", "c3po_2d", "c3po_3d", "c3po", "total"]
    for col in cols_to_filter:
        filtered_col = gaussian_filter1d(df[col], sigma=sigma)
        df[col + "_filtered"] = filtered_col
    return df

def generate_scalars(dataset):
    return {x: f"FractionCert/train/{x}" for x in BOP_MODEL_INDICES[dataset].keys()}


def get_data_dirs(logs_dir, all_obj_ids):
    obj_tb_db = {k: None for k in all_obj_ids}
    subfolders = [f.path for f in os.scandir(logs_dir) if f.is_dir()]
    for sfp in subfolders:
        # get directories containing obj ids
        sfp_dirs = [f.name for f in os.scandir(sfp) if f.is_dir()]
        obj_id = sfp_dirs[0][sfp_dirs[0].find("obj_") : sfp_dirs[0].find("obj_") + 10]
        frac_cert_dirs = {
            "csy_2d": os.path.join(sfp, f"FractionCert_train_{obj_id}_csy-2d"),
            "csy_3d": os.path.join(sfp, f"FractionCert_train_{obj_id}_csy-3d"),
            "csy": os.path.join(sfp, f"FractionCert_train_{obj_id}_cosypose"),
            "c3po_2d": os.path.join(sfp, f"FractionCert_train_{obj_id}_c3po-2d"),
            "c3po_3d": os.path.join(sfp, f"FractionCert_train_{obj_id}_c3po-3d"),
            "c3po": os.path.join(sfp, f"FractionCert_train_{obj_id}_c3po"),
            "total": os.path.join(sfp, f"FractionCert_train_{obj_id}_total"),
        }
        frac_cert_files = copy.deepcopy(frac_cert_dirs)
        for k, v in frac_cert_dirs.items():
            assert os.path.isdir(v)
            events_file = os.listdir(v)
            event_file = [x for x in events_file if "events.out" in x][0]
            frac_cert_files[k] = os.path.join(v, event_file)
            assert os.path.isfile(frac_cert_files[k])
        obj_tb_db[obj_id] = frac_cert_files
    return obj_tb_db


def build_dfs(dataset, obj_tb_db):
    """Build wide table DataFrames for all objects"""
    # each record contains:
    all_objs_dfs = {}
    scalars = generate_scalars(dataset)
    for obj_id, frac_cert_files in obj_tb_db.items():
        record_dict = {}
        for k, event_file in frac_cert_files.items():
            df = parse_tensorboard(event_file, scalars=[scalars[obj_id]])[scalars[obj_id]]
            if "step" not in record_dict.keys():
                record_dict["step"] = df["step"]
            record_dict[k] = df["value"]
        df = pd.DataFrame.from_dict(record_dict, orient="columns")
        df = smooth_df(df)
        all_objs_dfs[obj_id] = df
    return all_objs_dfs


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar
    Credit: https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically
    """
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(s in ea.Tags()["scalars"] for s in scalars), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
