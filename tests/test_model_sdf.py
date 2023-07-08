import logging
import pathlib

from casper3d.model_sdf import *


def test_model_sdf_construction():
    """Test ModelSDF class: constructing models"""
    logging.basicConfig(level=logging.DEBUG)

    # paths
    test_data_folder_path = pathlib.Path(__file__).parent.resolve().joinpath("./test_data/model_sdf")
    sdf_path = test_data_folder_path.joinpath("1a6f615e8b1b5ae4dbbc9440457e303e_sdf.vox")
    sdf_grad_path = test_data_folder_path.joinpath("1a6f615e8b1b5ae4dbbc9440457e303e_grad.pkl")

    # load sdf data
    sdf_grid = vox.load_vox_sdf(sdf_path)
    sdf_grad_grid = vox.load_vox_sdf_grad(sdf_grad_path)

    # create model SDF
    sdf_model = ModelSDF(type="voxel", sdf_grid=sdf_grid, sdf_grad_grid=sdf_grad_grid, device="cpu")

    return


def test_voxel_sdf_solve():
    """Testing VoxelSDF class solve function. Currently it only ensures it runs without errors."""
    logging.basicConfig(level=logging.DEBUG)

    # paths
    test_data_folder_path = pathlib.Path(__file__).parent.resolve().joinpath("./test_data/model_sdf")
    sdf_path = test_data_folder_path.joinpath("1a6f615e8b1b5ae4dbbc9440457e303e_sdf.vox")
    sdf_grad_path = test_data_folder_path.joinpath("1a6f615e8b1b5ae4dbbc9440457e303e_grad.pkl")

    # load sdf data
    sdf_grid = vox.load_vox_sdf(sdf_path)
    sdf_grad_grid = vox.load_vox_sdf_grad(sdf_grad_path)

    # create voxel SDF
    voxel_sdf_model = VoxelSDF(sdf_grid=sdf_grid, sdf_grad_grid=sdf_grad_grid, device="cpu")

    # single point
    input = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)
    input = torch.reshape(input, (1, 3, 1))
    result = voxel_sdf_model.solve(input)

    return


def test_voxel_sdf_gradient():
    """Testing VoxelSDF class gradient function. Currently it only ensures it runs without errors."""
    """Testing VoxelSDF class solve function. Currently it only ensures it runs without errors."""
    logging.basicConfig(level=logging.DEBUG)

    # paths
    test_data_folder_path = pathlib.Path(__file__).parent.resolve().joinpath("./test_data/model_sdf")
    sdf_path = test_data_folder_path.joinpath("1a6f615e8b1b5ae4dbbc9440457e303e_sdf.vox")
    sdf_grad_path = test_data_folder_path.joinpath("1a6f615e8b1b5ae4dbbc9440457e303e_grad.pkl")

    # load sdf data
    sdf_grid = vox.load_vox_sdf(sdf_path)
    sdf_grad_grid = vox.load_vox_sdf_grad(sdf_grad_path)

    # create voxel SDF
    voxel_sdf_model = VoxelSDF(sdf_grid=sdf_grid, sdf_grad_grid=sdf_grad_grid, device="cpu")

    # single point, single batch
    input = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)
    input = torch.reshape(input, (1, 3, 1))
    sdf_result = voxel_sdf_model.solve(input)
    sdf_grad_result = voxel_sdf_model.gradient(input)

    # multiple points, single batch
    # two points: (0,0,0) and (0.1, 0.1, 0.1)
    input = torch.tensor([[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]], dtype=torch.float)
    input = torch.reshape(input, (1, 3, 2))
    sdf_result = voxel_sdf_model.solve(input)
    sdf_grad_result = voxel_sdf_model.gradient(input)

    # multiple points, multiple batch
    # 4 batches, two points:
    # b1) (0,0,0) and (0.1, 0.1, 0.1)
    # b2) (0.2, 0.2, 0.2) and (0.3,0.3,0.3)
    # b3) (0.21, 0.21, 0.21) and (0.31,0.31,0.31)
    # b4) (0.22, 0.22, 0.22) and (0.32,0.32,0.32)
    input = torch.tensor(
        [[[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]],
         [[0.2, 0.3], [0.2, 0.3], [0.2, 0.3]],
         [[0.21, 0.31], [0.21, 0.31], [0.21, 0.31]],
         [[0.22, 0.32], [0.22, 0.32], [0.22, 0.32]],
         ], dtype=torch.float
    )
    sdf_result = voxel_sdf_model.solve(input)
    sdf_grad_result = voxel_sdf_model.gradient(input)

    return
