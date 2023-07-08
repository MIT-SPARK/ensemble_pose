import numpy as np
import pickle
import os
import pathlib

ab_options = ["fps", "proposed", "rand_sampling", "nonrobust_centroid"]

def analyze_single_obj(object_id, dataset):
    datafolder = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "data", dataset, object_model_id)

    # avg chamfer dist
    # avg kpt dist
    stats = []
    for ab_opt in ab_options:
        data_fname = os.path.join(datafolder, f"ablations_{ab_opt}_{dataset}_{object_id}.pkl")

        with open(data_fname, "rb") as f:
            data = pickle.load(f)

        avg_chamfer_dist = np.mean(data["normalized_chamfer_dist"])
        avg_kpt_loss = np.mean(data["kp_loss"])
        stats.append({"type": ab_opt, "avg_chamfer_dist": avg_chamfer_dist, "avg_kpt_loss": avg_kpt_loss})

    print("STATS: ")
    print(stats)

    return stats


if __name__ == "__main__":
    print("Script for calculating pooling methods' stats for ablation studies.")

    dataset = "ycbv.pbr"
    object_model_id = "obj_000018"
    analyze_single_obj(object_model_id, dataset)

