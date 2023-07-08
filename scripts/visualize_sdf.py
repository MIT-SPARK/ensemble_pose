import copy
import os
import argparse
import logging

import pyvista
import pyvista as pv
import trimesh
import numpy as np

import utils.sdf_gen.mesh2sdf_floodfill.vox as vox
from utils.file_utils import assert_files_exist


def load_vox_sdf_file(fpath):
    """Loading vox file containg SDF data"""
    vox_grid = vox.load_vox_sdf(fpath)

    pv_grid = pv.UniformGrid()
    pv_grid.dimensions = vox_grid.dims
    pv_grid.spacing = (vox_grid.res, vox_grid.res, vox_grid.res)
    pv_grid.origin = vox_grid.grid2world[0:3, 3]
    pv_grid["sdf"] = vox_grid.sdf.flatten(order="F")
    return pv_grid


def load_sdf_grad_file(fpath):
    vox_sdf_grad = vox.load_vox_sdf_grad(fpath)
    pv_grid = pv.UniformGrid()
    dims = vox_sdf_grad.dims
    pv_grid.dimensions = dims
    res = vox_sdf_grad.res
    pv_grid.spacing = (res, res, res)
    pv_grid.origin = vox_sdf_grad.grid2world[0:3, 3]

    sdf_grad_vectors = np.zeros((dims[0] * dims[1] * dims[2], 3))
    for i in range(len(vox_sdf_grad.sdf_grad)):
        sdf_grad_vectors[:, i] = vox_sdf_grad.sdf_grad[i].flatten(order="F")

    pv_grid["sdf_grads"] = sdf_grad_vectors
    pv_grid["sdf_grad_norms"] = np.linalg.norm(sdf_grad_vectors, axis=1)
    return pv_grid


def sdf_grad_field_2_vectors(sdf_grad_grid: pyvista.UniformGrid):
    """Convert a grad field to points + vectors"""
    points = copy.deepcopy(sdf_grad_grid.points)
    vectors = sdf_grad_grid["sdf_grads"]
    return points, vectors


def load_mesh_file(fpath):
    tmesh = trimesh.load(fpath)
    mesh = pv.wrap(tmesh)
    return mesh


def plot_sdf(mesh, sdf_grid, grad_grid):
    plotter = pv.Plotter(shape=(1, 2))

    # SDF volume with mesh
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh)
    slices = sdf_grid.slice_along_axis(n=7, axis="y")
    plotter.add_mesh(slices, cmap="BrBG")
    plotter.show_axes()

    # SDF gradients with mesh
    plotter.subplot(0, 1)
    points, vectors = sdf_grad_field_2_vectors(grad_grid)
    indices = np.random.choice(points.shape[0], size=5000)
    pdata = pyvista.vector_poly_data(points[indices, :], vectors[indices, :])
    arrows = pdata.glyph(orient='vectors', scale='mag', factor=0.1)
    plotter.add_mesh(mesh)
    plotter.add_mesh(arrows, color='black')
    plotter.show_axes()

    plotter.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a given SDF file dump.")
    parser.add_argument(
        "--config",
        type=str,
        help="path to the config file (YAML)",
    )
    parser.add_argument(
        "--input_sdf_file",
        type=str,
        help="path to the input file containing SDF data",
    )
    parser.add_argument(
        "--input_sdf_grad_file",
        type=str,
        help="path to the input file containing SDF gradient data",
    )
    parser.add_argument(
        "--input_mesh_file",
        type=str,
        help="path to the input file containing mesh data",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    assert_files_exist([args.input_sdf_file, args.input_sdf_grad_file, args.input_mesh_file])

    # load files
    mesh = load_mesh_file(args.input_mesh_file)
    model_filename = os.path.basename(args.input_sdf_file)
    if model_filename.split(".")[-1] == "vox":
        sdf_grid = load_vox_sdf_file(args.input_sdf_file)
    else:
        raise ValueError(f"Unsupported SDF file: {model_filename}")
    sdf_grad_grid = load_sdf_grad_file(args.input_sdf_grad_file)

    plot_sdf(mesh, sdf_grid, sdf_grad_grid)
