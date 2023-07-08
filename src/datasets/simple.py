import json
import numpy as np
import open3d as o3d
import os
import torch
from pytorch3d import transforms

from utils.sdf_gen.mesh2sdf_floodfill import vox

# the same as shapenet
CLASS_ID: dict = {
    "airplane": "02691156",
    "bathtub": "02808440",
    "bed": "02818832",
    "bottle": "02876657",
    "cap": "02954340",
    "car": "02958343",
    "chair": "03001627",
    "guitar": "03467517",
    "helmet": "03513137",
    "knife": "03624134",
    "laptop": "03642806",
    "motorcycle": "03790512",
    "mug": "03797390",
    "skateboard": "04225987",
    "table": "04379243",
    "vessel": "04530566",
}

# the same as shapenet
CLASS_NAME: dict = {
    "02691156": "airplane",
    "02808440": "bathtub",
    "02818832": "bed",
    "02876657": "bottle",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03467517": "guitar",
    "03513137": "helmet",
    "03624134": "knife",
    "03642806": "laptop",
    "03790512": "motorcycle",
    "03797390": "mug",
    "04225987": "skateboard",
    "04379243": "table",
    "04530566": "vessel",
}


def get_model_and_keypoints(
    class_id, model_id, pcd_root_folder_path, mesh_root_folder_path, annotations_folder, sdf_folder, sdf_grad_folder
):
    """
    Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints from the KeypointNet dataset.

    inputs:
    class_id    : string
    model_id    : string

    output:
    mesh        : o3d.geometry.TriangleMesh
    pcd         : o3d.geometry.PointCloud
    keypoints   : o3d.utils.Vector3dVector(nx3)
    """
    object_pcd_file = os.path.join(pcd_root_folder_path, str(class_id), str(model_id) + ".pcd")
    object_mesh_file = os.path.join(mesh_root_folder_path, str(class_id), str(model_id) + ".ply")
    object_sdf_file = os.path.join(sdf_folder, str(class_id), str(model_id) + "_sdf.vox")
    object_sdf_grads_file = os.path.join(sdf_grad_folder, str(class_id), str(model_id) + "_grad.pkl")

    # load pcds, mesh, sdf and sdf grads
    pcd = o3d.io.read_point_cloud(filename=object_pcd_file)
    mesh = o3d.io.read_triangle_mesh(filename=object_mesh_file)
    mesh.compute_vertex_normals()
    sdf_grid = vox.load_vox_sdf(object_sdf_file)
    sdf_grad_grid = vox.load_vox_sdf_grad(object_sdf_grads_file)

    annotation_file = os.path.join(annotations_folder, CLASS_NAME[str(class_id)] + ".json")
    file_temp = open(str(annotation_file))
    anotation_data = json.load(file_temp)

    for idx, entry in enumerate(anotation_data):
        if entry["model_id"] == str(model_id):
            keypoints = entry["keypoints"]
            break

    keypoints_xyz = []
    for aPoint in keypoints:
        keypoints_xyz.append(aPoint["xyz"])

    keypoints_xyz = np.array(keypoints_xyz)

    return mesh, pcd, keypoints_xyz, sdf_grid, sdf_grad_grid


class SimpleSingleClassSDF(torch.utils.data.Dataset):
    """A simple dataset class for testing."""

    def __init__(
        self,
        class_id,
        model_id,
        num_of_points=1000,
        dataset_len=10000,
        sdf_dir="../../data/test_sdf_dataset/sdfs",
        sdf_grad_dir="../../data/test_sdf_dataset/sdf_grads",
        mesh_dir="../../data/KeypointNet/ShapeNetCore.v2.ply/",
        pcd_dir="../../data/KeypointNet/KeypointNet/pcds/",
        annotation_dir="../../data/KeypointNet/KeypointNet/annotations/",
        seed=0,
    ):
        self.class_id = class_id
        self.model_id = model_id
        self.len = dataset_len
        self.num_of_points = num_of_points
        self.seed = seed

        # get model
        self.model_mesh, _, self.keypoints_xyz, self.sdf_grid, self.sdf_grad_grid = get_model_and_keypoints(
            class_id, model_id, pcd_dir, mesh_dir, annotation_dir, sdf_dir, sdf_grad_dir
        )
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound())
        )

        return

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

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points, seed=self.seed)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return R @ model_pcd_torch + t, R @ self.keypoints_xyz.squeeze(0) + t, R, t

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points, seed=self.seed)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_sdf(self):
        """Returns SDF grid"""
        return self.sdf_grid

    def _get_model_sdf_grad(self):
        """Return SDF gradient grid"""
        return self.sdf_grad_grid

    def _get_model_keypoints(self):
        """
        Returns keypoints of the ShapeNetCore model annotated in the KeypointNet dataset.

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
