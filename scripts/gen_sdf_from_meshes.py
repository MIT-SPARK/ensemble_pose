import argparse
import json
import logging
import os
import subprocess

import numpy as np
import trimesh
import yaml
from tqdm import tqdm
from utils.file_utils import safely_make_folders
from utils.sdf_gen.mesh2sdf import get_surface_point_cloud, scale_to_unit_cube
from utils.sdf_gen.mesh2sdf_floodfill import *

# for use in ssh/docker (without display)
os.environ["PYOPENGL_PLATFORM"] = "egl"


def get_sdf_output_name(model_name):
    return model_name + "_sdf.vox"


def get_grad_output_name(model_name):
    return model_name + "_grad.pkl"


def main_mesh2sdf(m_path, config, output_sdf_path=None, output_grad_path=None):
    """Using mesh2sdf to convert mesh to SDF"""
    mesh = trimesh.load(m_path)
    params = config["mesh2sdf"]

    mesh, s, t = scale_to_unit_cube(mesh)

    surface_point_cloud = get_surface_point_cloud(
        mesh,
        surface_point_method=params["surface_point_method"],
        bounding_radius=3**0.5,
        scan_count=params["scan_count"],
        scan_resolution=params["scan_resolution"],
        sample_point_count=params["sample_point_count"],
        calculate_normals=params["calculate_normals"],
    )

    vox = surface_point_cloud.get_voxels(
        params["voxel_resolution"],
        use_depth_buffer=params["use_depth_buffer"],
        sample_count=params["normal_sample_count"],
        pad=params["pad"],
        check_result=params["check_result"],
        return_gradients=params["return_gradients"],
    )
    raise NotImplementedError("This function hasn't been implemented.")


def main_mesh2sdf_floodfill(m_path, config, output_sdf_path=None, output_grad_path=None):
    """Call the mesh2sdf floodfill binary"""
    params = config["mesh2sdf_floodfill"]

    # check binary existence
    binary = os.path.abspath(params["exe_path"])
    if not os.path.exists(binary):
        raise ValueError(f"Binary does not exist at {binary}")

    # call binary on the file
    args = [
        binary,
        "--in",
        m_path,
        "--out",
        output_sdf_path,
        "--res",
        f"{params['resolution']}",
        "--trunc",
        f"{params['trunc']}",
        "--pad",
        f"{params['pad']}",
    ]
    if params["normalize"]:
        args.append("--normalize")
    logging.info(f"Binary arguments: \n{args}")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    logging.info(f"Binary output: \n{output}")

    # read the binary output file & calculate gradients
    if not os.path.exists(output_sdf_path):
        raise ValueError("Failed voxelization.")
    vox_data = load_vox_sdf(output_sdf_path)
    logging.info(f"Generating SDF gradients at {output_grad_path}")
    vox_grad = np.gradient(vox_data.sdf, vox_data.res)

    vox_sdf_grad = VoxSDFGrad(vox_data.dims, vox_data.res, vox_data.grid2world, vox_grad)

    # save the gradients to file
    write_vox_sdf_grad(output_grad_path, vox_sdf_grad)
    return


def load_models_in_folder(folder_path):
    """Return abs paths of all the models in the input directory

    Args:
        folder_path: folder containing the mesh files
    """
    ok_extensions = ["ply", "obj"]
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.split(".")[-1] in ok_extensions and os.path.isfile(os.path.join(folder_path, f))
    ]
    return files


def load_models(folder_path, type='flat'):
    """Return abs paths of all the models in the input directory

    Args:
        folder_path:
        type: flat / hierarchical
    """
    if type == 'flat':
        # acceptable extensions
        return load_models_in_folder(folder_path)
    elif type == 'hierarchical':
        # the files are organized as the following:
        # - class_id
        #   - model_id.ply
        class_dirs = next(os.walk(folder_path))[1]
        files = []
        for class_dir in class_dirs:
            files.extend(load_models_in_folder(os.path.join(folder_path, class_dir)))
        return files
    else:
        raise ValueError(f"Unknown file organization type: {type}")


if __name__ == "__main__":
    """
    Example usage:
    
    For all meshes inside the same folder (flat):
     PYTHONPATH="${PYTHONPATH}:/opt/project/CASPER-3D/src" python /opt/project/CASPER-3D/scripts/gen_sdf_from_meshes.py \
     --method mesh2sdf_floodfill \
     --config /opt/project/CASPER-3D/scripts/configs/sdf_gen.yml \
     --input_dir /mnt/datasets/sdf_lib/test_meshes/ \
     --input_dir_organization flat \
     --output_sdf_dir /mnt/datasets/sdf_lib/test_sdfs/ \
     --output_grad_dir /mnt/datasets/sdf_lib/test_sdf_grads/ -d 
     
     For hierarchical structure (hierarchical), similar to shapenet:
     PYTHONPATH="${PYTHONPATH}:/opt/project/CASPER-3D/src" python /opt/project/CASPER-3D/scripts/gen_sdf_from_meshes.py \
     --method mesh2sdf_floodfill \
     --config /opt/project/CASPER-3D/scripts/configs/sdf_gen.yml \
     --input_dir /mnt/datasets/KeypointNet/ShapeNetCore.v2.ply/ \
     --input_dir_organization hierarchical \
     --output_sdf_dir /mnt/datasets/KeypointNetCore_SDF/sdfs/ \
     --output_grad_dir /mnt/datasets/KeypointNetCore_SDF/sdf_grads/ \ 
     --output_dir_organization hierarchical \
     --instances_to_select /opt/project/CASPER-3D/scripts/configs/selected_instances.json -d 
    """
    parser = argparse.ArgumentParser(description="CLI tool for converting meshes to SDFs")
    parser.add_argument(
        "--method",
        type=str,
        default="mesh2sdf_floodfill",
        help="method to use to convert meshes to sdfs (default: mesh2sdf, render based",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to the config file (YAML)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="path to the input data directories containing mesh files",
    )
    parser.add_argument(
        "--instances_to_select",
        type=str,
        default="",
        help="path to the json file containing class id & instance id of meshes to use",
    )
    parser.add_argument(
        "--input_dir_organization",
        default='flat',
        const='flat',
        nargs='?',
        choices=['hierarchical', 'flat'],
        help="type of the file organization (flat: all mesh files are inside root dir;" +
             "hierarchical: class_id/model_id.ply)",
    )
    parser.add_argument(
        "--output_sdf_dir",
        type=str,
        default="./outputs_sdf",
        help="folder to dump the output sdf files",
    )
    parser.add_argument(
        "--output_grad_dir",
        type=str,
        default="./outputs_grad",
        help="folder to dump the output gradient files",
    )
    parser.add_argument(
        "--output_dir_organization",
        default='flat',
        const='flat',
        nargs='?',
        choices=['hierarchical', 'flat'],
        help="type of the file organization  to output (flat: all mesh files are inside root dir;" +
             "hierarchical: class_id/model_id.ply)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    # load config file
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    logging.debug(json.dumps(config))

    # load models
    model_paths = load_models(args.input_dir, type=args.input_dir_organization)
    if len(model_paths) == 0:
        raise ValueError(f"Empty input dir: {args.input_dir}")

    # prepare output dir
    safely_make_folders([args.output_sdf_dir, args.output_grad_dir])

    # load config file
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    logging.info(f"Total meshes: {len(model_paths)}")
    for m_path in tqdm(model_paths):
        logging.info(f"\nModel path: {m_path}")
        method_func = None
        # determine method
        if args.method == "mesh2sdf":
            logging.info("Using mesh2sdf method.")
            method_func = main_mesh2sdf
        elif args.method == "mesh2sdf_floodfill":
            logging.info("Using mesh2sdf_floodfill method.")
            method_func = main_mesh2sdf_floodfill
        if method_func is None:
            raise ValueError(f"Invalid method: {args.method}")

        # load current model file
        model_name = os.path.basename(m_path).split(".")[0]
        class_id = os.path.basename(os.path.dirname(m_path))
        if args.output_dir_organization == "flat":
            output_sdf_path = os.path.join(args.output_sdf_dir, get_sdf_output_name(model_name))
            output_grad_path = os.path.join(args.output_grad_dir, get_grad_output_name(model_name))
        elif args.output_dir_organization == "hierarchical":
            safely_make_folders([os.path.join(args.output_sdf_dir, class_id),
                                 os.path.join(args.output_grad_dir, class_id)])
            output_sdf_path = os.path.join(args.output_sdf_dir, class_id, get_sdf_output_name(model_name))
            output_grad_path = os.path.join(args.output_grad_dir, class_id, get_grad_output_name(model_name))
        else:
            raise ValueError(f"Unsupported output dir organization type: {args.output_dir_organization}")

        if args.input_dir_organization == "hierarchical":
            if len(args.instances_to_select) != 0:
                with open(args.instances_to_select, 'r') as f:
                    instances_dict = json.load(f)
                    if model_name not in instances_dict[class_id]:
                        logging.info(f"Skipping {class_id} : {model_name}")
                        continue

        logging.info(f"Output SDF path: {output_sdf_path}")
        logging.info(f"Output SDF grad path: {output_grad_path}")
        method_func(m_path, config=config, output_sdf_path=output_sdf_path, output_grad_path=output_grad_path)

        # check output success
        if not os.path.exists(output_sdf_path):
            raise ValueError("Failed voxelization.")
