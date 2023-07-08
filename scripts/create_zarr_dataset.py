import os
import argparse
import zarr
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import imageio.v2 as imageio
import time
from multiprocessing import Pool

from PIL import Image

BOP_SPLIT_DIRS = {
    "ycbv": ["train_real", "train_synt", "train_pbr", "test"],
    "tless": ["train_primesense", "test_primesense", "train_pbr"],
}


def get_img_size(img_path):
    rgb = np.array(Image.open(img_path))
    return rgb.shape


def load_depth(path):
    """Loads a depth image from a file.

    :param path: Path to the depth image file to load.
    :return: ndarray with the loaded depth image.
    """
    d = imageio.imread(path)
    return d.astype(np.float32)


def process_one_folder(folder_path, zarr_root, is_depth=False):
    """List all image files in the folder, create a zarr dataset and an index dictionary linking basename to the file"""
    folder_name = os.path.basename(folder_path).split(".")[0]

    # find all paths of images
    valid_path_suffix = {"png", "jpg", "tif"}
    candidate_files = sorted(
        [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.split(".")[-1] in valid_path_suffix
        ]
    )

    # make sure all suffixes are the same
    suffix = {x.split(".")[-1] for x in candidate_files}
    assert len(suffix) == 1

    # determine sizes
    img_shape = get_img_size(os.path.join(folder_path, candidate_files[0]))
    num_imgs = len(candidate_files)

    # create the zarr dataset
    shape = (num_imgs, *img_shape)
    chunks = (int(num_imgs / 10), *img_shape)
    if not is_depth:
        z_dataset = zarr_root.zeros(folder_name, shape=shape, chunks=chunks, dtype="u1")
    else:
        z_dataset = zarr_root.zeros(folder_name, shape=shape, chunks=chunks, dtype="f4")

    # write!
    index = {}
    np_data_array = []
    for i, img_f in enumerate(candidate_files):
        img_path = os.path.join(folder_path, img_f)
        if not is_depth:
            img_data = np.array(Image.open(img_path))
        else:
            img_data = np.array(load_depth(img_path))

        np_data_array.append(img_data)
        index[img_f.split(".")[0]] = i

    z_dataset[:] = np.asarray(np_data_array)
    z_dataset.attrs["index"] = index


def process_scene(scene_dir, output_scene_dir):
    """Process data in one scene folder"""
    # in each scene folder, we have:
    #
    # folders:
    # depth, mask, mask_visib, rgb|gray
    #
    # each folder:
    # - store an index dictionary file
    # name -> index
    # - store all images in zarr file
    print(f"Running on {scene_dir}.")
    Path(output_scene_dir).mkdir(parents=True, exist_ok=True)

    # create zarr group
    root = zarr.open(os.path.join(output_scene_dir, "cam_data"), mode="w")

    sub_folders = ["depth", "mask", "mask_visib", "rgb", "gray"]
    for sf in tqdm(sub_folders):
        fp = os.path.join(scene_dir, sf)
        if os.path.exists(fp):
            if not os.path.isdir(fp):
                print(f"{fp} is not a directory.")
                raise ValueError

            # create empty zarr array
            process_one_folder(fp, root, is_depth=sf == "depth")

    # copy all json files
    json_files = ["scene_camera.json", "scene_gt.json", "scene_gt_info.json"]
    for x in json_files:
        shutil.copyfile(os.path.join(scene_dir, x), os.path.join(output_scene_dir, x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="specify the root folder", type=str)
    parser.add_argument("dataset", help="specify the dataset.", type=str)
    parser.add_argument("split", help="specify the split", type=str)
    parser.add_argument("output_root_dir", help="specify the output root directory (without split/dataset)", type=str)

    args = parser.parse_args()
    d_splits = BOP_SPLIT_DIRS[args.dataset]
    assert args.split in d_splits

    # for each split directory in dataset
    split_path = os.path.join(args.root_dir, args.dataset, args.split)
    output_split_path = os.path.join(args.output_root_dir, args.dataset, args.split)

    # read and process each scene
    scene_ids = sorted(os.listdir(split_path))
    pool = Pool(10)
    start_time = time.perf_counter()
    processes = [
        pool.apply_async(process_scene, args=(os.path.join(split_path, s), os.path.join(output_split_path, s),))
        for s in scene_ids
    ]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
