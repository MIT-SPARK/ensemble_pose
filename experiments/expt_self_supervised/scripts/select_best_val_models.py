import numpy as np
import pickle
import os
import argparse
import shutil
from pathlib import Path
import matplotlib.pyplot as plt


def get_best_model_cand(f_dir):
    """Return the val loss and the abs path to the best model weights in the f_dir"""
    #loss_fname = "_ssval_loss.pkl"
    loss_fname = "_sstrain_loss.pkl"
    try:
        with open(os.path.join(f_dir, loss_fname), "rb") as val_f:
            tracker = pickle.load(val_f)
            vals = np.array(tracker.val)
            min_model_i = np.argmin(vals)
    except FileNotFoundError:
        return None

    # load all weights & sort
    object_weights_cands = sorted(
        [f.name for f in os.scandir(f_dir) if f.is_file() and ".pth.tar" in f.name], key=lambda x: int(x.split("_")[2])
    )

    return vals[min_model_i], os.path.join(f_dir, object_weights_cands[min_model_i]), os.path.join(f_dir, "config.yml")


if __name__ == "__main__":
    print("Run this script to select the best model based on val losses.")
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_folder_path", help="path to folder containing all objects' model weights")
    parser.add_argument("output_folder", help="path to output folder")

    args = parser.parse_args()

    # get all folders in root folder
    object_weights_cands_dirs = [f.path for f in os.scandir(args.weights_folder_path) if f.is_dir()]
    object_weights_cands = [f.name for f in os.scandir(args.weights_folder_path) if f.is_dir()]
    objects_ids = [x[x.find("obj") : x.find("obj") + 10] for x in object_weights_cands]

    # get all available objects
    avail_objs = set(objects_ids)
    best_models_cands = {k: [] for k in avail_objs}

    for obj_id, model_dir in zip(objects_ids, object_weights_cands_dirs):
        cand = get_best_model_cand(model_dir)
        if cand is not None:
            best_models_cands[obj_id].append(cand)

    # select the best
    best_models = {}
    for obj_id, cands in best_models_cands.items():
        if len(cands) != 0:
            sorted_cands = sorted(cands, key=lambda x: x[0])
            best_cand = sorted_cands[0]
            best_models[obj_id] = best_cand

    # copy to dump folder
    for obj_id, cand in best_models.items():
        Path(os.path.join(args.output_folder, obj_id)).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            cand[1],
            os.path.join(
                args.output_folder, obj_id, "_best_model.pth.tar"
            ),
        )
        shutil.copyfile(cand[2], os.path.join(args.output_folder, obj_id, "config.yml"))

    # plot
    y_data = []
    for obj_id in sorted(best_models.keys()):
        y_data.append(best_models[obj_id][0])
    plt.bar(np.array(range(len(y_data)))+1, y_data)
    plt.show()
    
