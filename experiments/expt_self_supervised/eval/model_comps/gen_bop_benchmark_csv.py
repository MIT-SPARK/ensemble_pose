import argparse
import os
import pandas as pd
import pathlib
import csv

from expt_self_supervised.eval.model_comps.data_utils import load_one_method, cleanup_df, load_self6dpp_results


def flatten(l):
    return [item for sublist in l for item in sublist]


def create_table(df, joint_selection_strat="c3po"):
    """Create the detection table according to BOP format"""
    """
    columns: 
    scene_id,im_id,obj_id,score,R (9 elements),t (3 elements),time

    scene_id, im_id, and obj_id is the ID of scene, image and object respectively.
    score is a confidence of the estimate (the range of confidence values is not restricted).
    R is a 3x3 rotation matrix whose elements are saved row-wise and separated by a white space (i.e. r11 r12 r13 r21 r22 r23 r31 r32 r33, where rij is an element from the i-th row and the j-th column of the matrix).
    t is a 3x1 translation vector (in mm) whose elements are separated by a white space (i.e. t1 t2 t3).
    time is the time the method took to estimate poses for all objects in image im_id from scene scene_id. All estimates with the same scene_id and im_id must have the same value of time. Report the wall time from the point right after the raw data (the image, 3D object models etc.) is loaded to the point when the final pose estimates are available (a single real number in seconds, -1 if not available).
    """

    R, t, time = [], [], []
    scene_ids, im_id, cat_id, scores = [], [], [], []
    if "cert_c3po" not in df.keys():
        df["cert_c3po"] = df["cert_c3po_pc"] & df["cert_c3po_mask"]
    if "cert_cosypose" not in df.keys():
        df["cert_cosypose"] = df["cert_cosypose_pc"] & df["cert_cosypose_mask"]
    if "cert_joint" not in df.keys():
        df["cert_joint"] = df["cert_c3po"] & df["cert_cosypose"]
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if row["cert_joint"]:
            method_to_use = joint_selection_strat
        elif row["use_c3po"]:
            method_to_use = "c3po"
        elif row["use_cosypose"]:
            method_to_use = "cosypose"
        else:
            method_to_use = joint_selection_strat

        if method_to_use == "c3po":
            c_R = [f"{x:.6f}" for x in flatten(df["c3po_R_est"].iloc[i])]
            c_t = [f"{1000 * x:.6f}" for x in df["c3po_t_est"].iloc[i]]
            R.append(" ".join(c_R))
            t.append(" ".join(c_t))
        else:
            c_R = [f"{x:.6f}" for x in flatten(df["cosypose_R_est"].iloc[i])]
            c_t = [f"{1000 * x:.6f}" for x in df["cosypose_t_est"].iloc[i]]
            R.append(" ".join(c_R))
            t.append(" ".join(c_t))

        # scores always 1, time = -1
        scene_ids.append(row["scene_id"])
        im_id.append(row["view_id"])
        cat_id.append(row["category_id"])
        scores.append(row['det_scores'])
        time.append(-1)

    output_data = {
        "scene_id": scene_ids,
        "im_id": im_id,
        "obj_id": cat_id,
        "score": scores,
        "R": R,
        "t": t,
        "time": time,
    }
    df = pd.DataFrame.from_dict(output_data)
    df = df.sort_values(by=['scene_id', 'im_id'])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method_name", help="name of the method", type=str)
    parser.add_argument("dataset", help="specify the dataset.", type=str)
    parser.add_argument("results_folder", help="folder containing the eval data", type=str)
    parser.add_argument(
        "--output_folder",
        help="folder to dump the csv files",
        type=str,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "bop_csvs"),
    )

    args = parser.parse_args()

    df = load_one_method(data_folder=os.path.join(args.results_folder, args.dataset), args=args)

    # dump the dataframe into bop format
    output_df = create_table(df, joint_selection_strat="c3po")

    # dump to csv
    output_df.to_csv(
        os.path.join(args.output_folder, f"{args.method_name}_{args.dataset}-test.csv"), index=False, quoting=csv.QUOTE_NONE
    )
