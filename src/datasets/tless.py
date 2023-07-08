import copy
import numpy as np
import open3d as o3d
import os
import torch
import trimesh
from PIL import Image
from pytorch3d import transforms

from utils.visualization_utils import visualize_torch_model_n_keypoints

ANNOTATIONS_FOLDER: str = "../../data/bop/bop_datasets/tless/annotations/"
MESH_FOLDER_NAME: str = "../../data/bop/bop_datasets/tless/models/"


def get_model_and_keypoints(model_id, mesh_root_folder_path, annotations_folder):
    """
    Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints from the TLESS dataset.

    inputs:
    class_id    : string
    model_id    : string

    output:
    mesh        : o3d.geometry.TriangleMesh
    pcd         : o3d.geometry.PointCloud
    keypoints   : o3d.utils.Vector3dVector(nx3)
    """
    object_mesh_file = os.path.join(mesh_root_folder_path, str(model_id) + ".ply")

    # trimesh mesh (with texture)
    # TLESS does not have texture images
    mesh = trimesh.load(object_mesh_file, process=False)

    # o3d mesh (without texture)
    mesh_o3d = o3d.io.read_triangle_mesh(filename=object_mesh_file)
    mesh_o3d.compute_vertex_normals()

    # load keypoints
    annotation_file = os.path.join(annotations_folder, model_id + ".npy")
    keypoints_xyz = np.load(annotation_file)

    return mesh_o3d, mesh, keypoints_xyz


class SE3PointCloud(torch.utils.data.Dataset):
    """
    Given class_id, model_id, and number of points generates various point clouds and SE3 transformations
    of the YCB-Video objects.

    Assume keypoint annotations have been done in the .npy format.

    Returns a batch of
        input_point_cloud, keypoints, rotation, translation
    """

    def __init__(
        self,
        model_id,
        num_of_points=1000,
        dataset_len=10000,
        normalize=True,
        cad_points_sampling_method="uniform",
        mesh_root_folder_path=MESH_FOLDER_NAME,
        annotations_folder_path=ANNOTATIONS_FOLDER,
    ):
        """
        class_id        : str   : class id of a TLESS object
        model_id        : str   : model id of a TLESS object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """

        self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len
        self.normalize = normalize

        # get model
        self.model_mesh, self.model_textured_mesh, self.keypoints_xyz = get_model_and_keypoints(
            model_id, mesh_root_folder_path=mesh_root_folder_path, annotations_folder=annotations_folder_path
        )

        # save the original mesh and keypoints
        self.original_model_mesh = copy.deepcopy(self.model_mesh)
        self.original_model_textured_mesh = copy.deepcopy(self.model_textured_mesh)
        self.original_keypoints_xyz = torch.from_numpy(copy.deepcopy(self.keypoints_xyz)).unsqueeze(0).to(torch.float)
        self.cad_points_sampling_method = cad_points_sampling_method

        # center the CAD model
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)
        self.model_textured_mesh.vertices -= center

        # center the keypoint annotations
        self.keypoints_xyz = self.keypoints_xyz - center[:, np.newaxis]
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound())
        )

        if normalize:
            self.keypoints_xyz = self.keypoints_xyz / self.diameter
            self.model_mesh.scale(1.0 / self.diameter, self.model_mesh.get_center())
            self.model_textured_mesh.vertices /= self.diameter

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        R = transforms.random_rotation()
        t = torch.rand(3, 1)

        if self.cad_points_sampling_method == "uniform":
            model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        elif self.cad_points_sampling_method == "poisson-disk":
            model_pcd = self.model_mesh.sample_points_poisson_disk(number_of_points=self.num_of_points)
        else:
            raise NotImplementedError
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return R @ model_pcd_torch + t, R @ self.keypoints_xyz.squeeze(0) + t, R, t

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_original_cad_models(self):
        """
        Returns a sampled point cloud of the TLESS model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """
        model_pcd = self.original_model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_original_cad_mesh(self):
        """Return the original CAD mesh (Open3D TriangleMesh)"""
        return self.original_model_mesh

    def _get_original_model_keypoints(self):
        """Return the original model keypoints"""
        return self.original_keypoints_xyz

    def _get_model_keypoints(self):
        """
        Returns keypoints of the TLESS model annotated in the TLESS dataset.

        output:
        model_keypoints : torch.tensor of shape (1, 3, N)

        where
        N = number of keypoints
        """

        return self.keypoints_xyz

    def _get_diameter(self):
        """
        Returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0
