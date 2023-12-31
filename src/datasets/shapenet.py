import copy
import json
import math
import numpy as np
import open3d as o3d
import os
import pickle
import torch
from enum import Enum
from pytorch3d import transforms

import utils.general as gu
from casper3d.model_gen import ModelFromShape
from utils.file_utils import assert_files_exist
from utils.sdf_gen.mesh2sdf_floodfill import vox
from utils.visualization_utils import display_two_pcs, visualize_model_n_keypoints, \
    visualize_torch_model_n_keypoints

ANNOTATIONS_FOLDER: str = '../../data/KeypointNet/KeypointNet/annotations/'
PCD_FOLDER_NAME: str = '../../data/KeypointNet/KeypointNet/pcds/'
MESH_FOLDER_NAME: str = '../../data/KeypointNet/ShapeNetCore.v2.ply/'
OBJECT_CATEGORIES: list = ['airplane', 'bathtub', 'bed', 'bottle',
                           'cap', 'car', 'chair', 'guitar',
                           'helmet', 'knife', 'laptop', 'motorcycle',
                           'mug', 'skateboard', 'table', 'vessel']
CLASS_ID: dict = {'airplane': "02691156",
                  'bathtub': "02808440",
                  'bed': "02818832",
                  'bottle': "02876657",
                  'cap': "02954340",
                  'car': "02958343",
                  'chair': "03001627",
                  'guitar': "03467517",
                  'helmet': "03513137",
                  'knife': "03624134",
                  'laptop': "03642806",
                  'motorcycle': "03790512",
                  'mug': "03797390",
                  'skateboard': "04225987",
                  'table': "04379243",
                  'vessel': "04530566"}

CLASS_NAME: dict = {"02691156": 'airplane',
                    "02808440": 'bathtub',
                    "02818832": 'bed',
                    "02876657": 'bottle',
                    "02954340": 'cap',
                    "02958343": 'car',
                    "03001627": 'chair',
                    "03467517": 'guitar',
                    "03513137": 'helmet',
                    "03624134": 'knife',
                    "03642806": 'laptop',
                    "03790512": 'motorcycle',
                    "03797390": 'mug',
                    "04225987": 'skateboard',
                    "04379243": 'table',
                    "04530566": 'vessel'}

class ScaleAxis(Enum):
    X = 0
    Y = 1
    Z = 2

MODEL_TO_KPT_GROUPS = {
    "mug": [set([9])],
    "cap": [set([1])]
    }


# From C-3PO
def get_model_and_keypoints(class_id, model_id,
                            pcd_root_folder_path=PCD_FOLDER_NAME,
                            mesh_root_folder_path=MESH_FOLDER_NAME,
                            annotations_folder=ANNOTATIONS_FOLDER):
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

    pcd = o3d.io.read_point_cloud(filename=object_pcd_file)
    mesh = o3d.io.read_triangle_mesh(filename=object_mesh_file)
    mesh.compute_vertex_normals()

    annotation_file = os.path.join(annotations_folder, CLASS_NAME[str(class_id)] + ".json")
    file_temp = open(str(annotation_file))
    anotation_data = json.load(file_temp)

    for idx, entry in enumerate(anotation_data):
        if entry['model_id'] == str(model_id):
            keypoints = entry['keypoints']
            break

    keypoints_xyz = []
    for aPoint in keypoints:
        keypoints_xyz.append(aPoint['xyz'])

    keypoints_xyz = np.array(keypoints_xyz)

    return mesh, pcd, keypoints_xyz


def get_sdf(class_id, model_id, sdf_folder, sdf_grad_folder):
    """
    Given class_id and model_id this function outputs the SDF voxel data.

    Args:
        class_id: string
        model_id: string
        sdf_folder:
        sdf_grad_folder:
    """
    assert_files_exist([sdf_folder, sdf_grad_folder])
    object_sdf_file = os.path.join(sdf_folder, str(class_id), str(model_id) + "_sdf.vox")
    object_sdf_grads_file = os.path.join(sdf_grad_folder, str(class_id), str(model_id) + "_grad.pkl")

    sdf_grid = vox.load_vox_sdf(object_sdf_file)
    sdf_grad_grid = vox.load_vox_sdf_grad(object_sdf_grads_file)

    return sdf_grid, sdf_grad_grid


# From C-3PO
def visualize_model(class_id, model_id):
    """ Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints
    from the KeypointNet dataset and displays them using open3d.visualization.draw_geometries"""

    mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id=class_id, model_id=model_id)

    keypoint_markers = visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)
    _ = visualize_model_n_keypoints([pcd], keypoints_xyz=keypoints_xyz)

    return mesh, pcd, keypoints_xyz, keypoint_markers


# From C-3PO
def generate_depth_data(class_id, model_id, radius_multiple = [1.2, 3.0],
                        num_of_points=100000, num_of_depth_images_per_radius=200,
                        dir_location='../../data/learning-objects/keypointnet_datasets/'):
    """ Generates a dataset of depth point clouds of a CAD model from randomly generated views. """

    radius_multiple = np.asarray(radius_multiple)

    location = dir_location + str(class_id) + '/' + str(model_id) + '/'
    # get model
    model_mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id, model_id)
    model_pcd = model_mesh.sample_points_uniformly(number_of_points=num_of_points)

    center = model_pcd.get_center()
    model_pcd.translate(-center)
    model_mesh.translate(-center)
    keypoints_xyz = keypoints_xyz - center

    model_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # determining size of the 3D object
    diameter = np.linalg.norm(np.asarray(model_pcd.get_max_bound()) - np.asarray(model_pcd.get_min_bound()))

    # determining camera locations and radius
    camera_distance_vector = diameter*radius_multiple
    camera_locations = gu.get_camera_locations(camera_distance_vector, number_of_locations=num_of_depth_images_per_radius)
    radius = gu.get_radius(object_diameter=diameter, cam_location=np.max(camera_distance_vector))

    # visualizing 3D object and all the camera locations
    _ = visualize_model_n_keypoints([model_pcd], keypoints_xyz=keypoints_xyz, camera_locations=camera_locations)
    _ = visualize_model_n_keypoints([model_mesh], keypoints_xyz=keypoints_xyz, camera_locations=camera_locations)

    # generating radius for view sampling
    gu.sample_depth_pcd(centered_pcd=model_pcd, camera_locations=camera_locations, radius=radius, folder_name=location)

    # save keypoints_xyz at location
    np.save(file=location+'keypoints_xyz.npy', arr=keypoints_xyz)


# From C-3PO
class DepthAnisoPC(torch.utils.data.Dataset):
    """
            Given class id, model id, and number of points, it generates various depth point clouds and
            SE3 transformations of the ShapeNetCore object. The object is scaled anisotropically by a quantity in the
            range determined by shape_scaling in the direction determined by scale_direction

            Note:
                Outputs depth point clouds of the same shape by appending with
                zero points. It also outputs a flag to tell the user which outputs have been artificially added.

                This dataset can be used with a dataloader for any batch_size.

            Returns
                input_point_cloud, keypoints, rotation, translation, shape
            """

    def __init__(self, class_id, model_id, n=1000, shape_scaling=torch.tensor([0.5, 2.0]),
                 scale_direction=ScaleAxis.X,
                 radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=1000, dataset_len=10000):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        shape_scaling   : torch.tensor of shape (2) : lower and upper limit of isotropic shape scaling
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset  

        """
        self.class_id = class_id
        self.model_id = model_id
        self.n = n
        self.shape_scaling = shape_scaling
        self.scale_direction = scale_direction.value
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = num_of_points_to_sample
        self.len = dataset_len

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(
            -1)  # set a camera location, with respect to the origin

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        depth_pcd_torch     : torch.tensor of shape (3, m)                  : the depth point cloud
        keypoints           : torch.tensor of shape (3, N)                  : transformed keypoints
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        c                   : torch.tensor of shape (2, 1)                  : shape parameter
        model_pcd_torch     : torch.tensor of shape (3, self.num_of_points) : transformed full point cloud
        """
        # Choose a random scaling factor to scale the depth point cloud, bounded within the self.radius_multiple
        alpha = torch.rand(1, 1)
        scaling_factor = alpha * (self.shape_scaling[1] - self.shape_scaling[0]) + self.shape_scaling[0]

        # Choose a random rotation
        R = transforms.random_rotation()

        model_mesh = copy.deepcopy(self.model_mesh)

        # Sample a point cloud from the self.model_mesh
        pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_to_sample)

        model_pcd_points = np.asarray(pcd.points).transpose()  # (3, m)
        model_pcd_points[self.scale_direction] = scaling_factor * model_pcd_points[self.scale_direction]
        pcd.points = o3d.utility.Vector3dVector(model_pcd_points.transpose())

        pcd = pcd.rotate(R=R.numpy(), center=np.zeros((3, 1)))

        # scaled_diameter = max(self.diameter, self.diameter*scaling_factor)

        # Take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_location_factor = beta * (self.radius_multiple[1] - self.radius_multiple[0]) + self.radius_multiple[0]
        camera_location_factor = camera_location_factor * self.diameter
        radius = gu.get_radius(cam_location=camera_location_factor * self.camera_location.numpy(),
                               object_diameter=self.diameter)
        depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)

        depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
        depth_pcd_torch = depth_pcd_torch.to(torch.float)

        keypoints_xyz = torch.clone(self.keypoints_xyz.squeeze())
        keypoints_xyz[self.scale_direction] = scaling_factor * keypoints_xyz[self.scale_direction]
        keypoints_xyz = R @ keypoints_xyz

        # Translate by a random t
        t = torch.rand(3, 1)
        depth_pcd_torch = depth_pcd_torch + t
        keypoints_xyz = keypoints_xyz + t

        # shape parameter
        shape = torch.zeros(2, 1)
        shape[0] = 1 - alpha
        shape[1] = alpha

        point_cloud, padding = self._convert_to_fixed_sized_pc(depth_pcd_torch, n=self.n)

        return point_cloud, keypoints_xyz.squeeze(0), R, t, shape

    def _convert_to_fixed_sized_pc(self, pc, n):
        """
        inputs:
        pc  : torch.tensor of shape (3, m)  : input point cloud of size m (m could be anything)
        n   : int                           : number of points the output point cloud should have

        outputs:
        point_cloud     : torch.tensor of shape (3, n)
        padding         : torch.tensor of shape (n)

        """

        m = pc.shape[-1]

        if m > n:
            idx = torch.randperm(m)
            point_cloud = pc[:, idx[:n]]
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        elif m < n:

            pc_pad = torch.zeros(3, n-m)
            point_cloud = torch.cat([pc, pc_pad], dim=1)
            padding1 = torch.zeros(size=(m,), dtype=torch.bool)
            padding2 = torch.ones(size=(n-m,), dtype=torch.bool)
            padding = torch.cat([padding1, padding2], dim=0)
            # Write code to pad pc with (n-m) zeros

        else:
            point_cloud = pc
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        return point_cloud, padding

    def _get_cad_models(self):
        """
        Returns two point clouds as shape models, one with the min shape and the other with the max shape
        along scale_direction

        output:
        cad_models  : torch.tensor of shape (2, 3, self.num_of_points_to_sample)
        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_to_sample)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        cad_models = torch.zeros_like(model_pcd_torch).unsqueeze(0)
        cad_models = cad_models.repeat(2, 1, 1)

        cad_models[0, ...] = torch.clone(model_pcd_torch)
        cad_models[1, ...] = torch.clone(model_pcd_torch)

        cad_models[0, self.scale_direction, ...] = self.shape_scaling[0] * cad_models[0, self.scale_direction, ...]
        cad_models[1, self.scale_direction, ...] = self.shape_scaling[1] * cad_models[1, self.scale_direction, ...]

        return cad_models

    def _get_model_keypoints(self):
        """
        Returns two sets of keypoints, one with the min shape and the other with the max shape
                along scale_direction.

        output:
        model_keypoints : torch.tensor of shape (2, 3, N)

        where
        N = number of keypoints
        """

        keypoints = self.keypoints_xyz  # (3, N)

        model_keypoints = torch.zeros_like(keypoints)
        model_keypoints = model_keypoints.repeat(2, 1, 1)

        model_keypoints[0, ...] = torch.clone(keypoints)
        model_keypoints[1, ...] = torch.clone(keypoints)
        model_keypoints[0, self.scale_direction, ...] = self.shape_scaling[0] * model_keypoints[
            0, self.scale_direction, ...]
        model_keypoints[1, self.scale_direction, ...] = self.shape_scaling[1] * model_keypoints[
            1, self.scale_direction, ...]

        return model_keypoints

    def _get_diameter(self):
        """
        returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """
        return self.diameter * (self.shape_scaling[0] + self.shape_scaling[1]) * 0.5

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


# From C-3PO
class DepthIsoPC(torch.utils.data.Dataset):
    """
    Given class id, model id, and number of points, it generates various depth point clouds and
    SE3 transformations of the ShapeNetCore object. The object is scaled isotropically by a quantity in the
    range determined by shape_scaling.

    Note:
        Outputs depth point clouds of the same shape by appending with
        zero points. It also outputs a flag to tell the user which outputs have been artificially added.

        This dataset can be used with a dataloader for any batch_size.

    Returns
        input_point_cloud, keypoints, rotation, translation, shape
    """

    def __init__(self, class_id, model_id, n=1000, shape_scaling=torch.tensor([0.5, 2.0]),
                 radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=10000, dataset_len=10000):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        n               : int   : number of points in the output point cloud
        shape_scaling   : torch.tensor of shape (2) : lower and upper limit of isotropic shape scaling
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points_to_sample   : int   : number of points sampled on the surface of the CAD model object
        dataset_len     : int   : size of the dataset  

        """
        self.class_id = class_id
        self.model_id = model_id
        self.n = n
        self.shape_scaling = shape_scaling
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = num_of_points_to_sample
        self.len = dataset_len

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)
        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))
        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        # Randomly rotate the self.model_mesh
        model_mesh = copy.deepcopy(self.model_mesh)
        R = transforms.random_rotation()
        model_mesh = model_mesh.rotate(R=R.numpy())

        # Sample a point cloud from the self.model_mesh
        pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_to_sample)

        # Take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_location_factor = beta * (self.radius_multiple[1] - self.radius_multiple[0]) + self.radius_multiple[0]
        camera_location_factor = camera_location_factor * self.diameter
        radius = gu.get_radius(cam_location=camera_location_factor * self.camera_location.numpy(),
                               object_diameter=self.diameter)
        depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)

        # Scale the depth point cloud, by a random quantity, bounded within the self.radius_multiple
        alpha = torch.rand(1, 1)
        scaling_factor = alpha * (self.shape_scaling[1] - self.shape_scaling[0]) + self.shape_scaling[0]

        depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
        depth_pcd_torch = depth_pcd_torch.to(torch.float)
        depth_pcd_torch = scaling_factor * depth_pcd_torch

        keypoints_xyz = R @ self.keypoints_xyz
        keypoints_xyz = scaling_factor * keypoints_xyz

        # Translate by a random t
        t = torch.rand(3, 1)

        depth_pcd_torch = depth_pcd_torch + t
        keypoints_xyz = keypoints_xyz + t

        # shape parameter
        shape = torch.zeros(2, 1)
        shape[0] = 1 - alpha
        shape[1] = alpha

        model_pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch = scaling_factor * model_pcd_torch + t

        point_cloud, padding = self._convert_to_fixed_sized_pc(depth_pcd_torch, n=self.n)

        return point_cloud, keypoints_xyz.squeeze(0), R, t, shape

    def _convert_to_fixed_sized_pc(self, pc, n):
        """
        inputs:
        pc  : torch.tensor of shape (3, m)  : input point cloud of size m (m could be anything)
        n   : int                           : number of points the output point cloud should have

        outputs:
        point_cloud     : torch.tensor of shape (3, n)
        padding         : torch.tensor of shape (n)

        """

        m = pc.shape[-1]

        if m > n:
            idx = torch.randperm(m)
            point_cloud = pc[:, idx[:n]]
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        elif m < n:

            pc_pad = torch.zeros(3, n-m)
            point_cloud = torch.cat([pc, pc_pad], dim=1)
            padding1 = torch.zeros(size=(m,), dtype=torch.bool)
            padding2 = torch.ones(size=(n-m,), dtype=torch.bool)
            padding = torch.cat([padding1, padding2], dim=0)
            # Write code to pad pc with (n-m) zeros

        else:
            point_cloud = pc
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        return point_cloud, padding

    def _get_cad_models(self):
        """
        Returns two point clouds as shape models, one with the min shape and the other with the max shape.

        output:
        cad_models  : torch.tensor of shape (2, 3, self.num_of_points_to_sample)
        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_to_sample)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        cad_models = torch.zeros_like(model_pcd_torch).unsqueeze(0)
        cad_models = cad_models.repeat(2, 1, 1)

        cad_models[0, ...] = self.shape_scaling[0]*model_pcd_torch
        cad_models[1, ...] = self.shape_scaling[1]*model_pcd_torch

        return cad_models

    def _get_model_keypoints(self):
        """
        Returns two sets of keypoints, one with the min shape and the other with the max shape.

        output:
        model_keypoints : torch.tensor of shape (2, 3, N)

        where
        N = number of keypoints
        """

        keypoints = self.keypoints_xyz  # (3, N)

        model_keypoints = torch.zeros_like(keypoints)
        model_keypoints = model_keypoints.repeat(2, 1, 1)

        model_keypoints[0, ...] = self.shape_scaling[0]*keypoints
        model_keypoints[1, ...] = self.shape_scaling[1]*keypoints

        return model_keypoints

    def _get_diameter(self):
        """
        returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """
        return self.diameter*(self.shape_scaling[0] + self.shape_scaling[1])*0.5

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


# From C-3PO
class DepthPC(torch.utils.data.Dataset):
    """
    Given class id, model id, and number of points, it generates various depth point clouds and SE3 transformations
    of the ShapeNetCore object.

    Note:
        Outputs depth point clouds of the same shape by appending with
        zero points. It also outputs a flag to tell the user which outputs have been artificially added.

        This dataset can be used with a dataloader for any batch_size.

    Returns
        input_point_cloud, keypoints, rotation, translation
    """

    def __init__(self, class_id, model_id, n=1000, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=10000, dataset_len=10000, rotate_about_z=False,
                 sdf_dir=None,
                 sdf_grad_dir=None,
                 mesh_dir=MESH_FOLDER_NAME,
                 pcd_dir=PCD_FOLDER_NAME,
                 annotation_dir=ANNOTATIONS_FOLDER,
                 ):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object 
        n               : int   : number of points in the output point cloud
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points_to_sample   : int   : number of points sampled on the surface of the CAD model object
        dataset_len     : int   : size of the dataset  

        """

        self.class_id = class_id
        self.model_id = model_id
        self.n = n
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = num_of_points_to_sample
        self.len = dataset_len
        self.rotate_about_z = rotate_about_z
        self.pi = torch.tensor([math.pi])

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id,
                                                                         pcd_dir, mesh_dir, annotation_dir)
        # get SDF and SDF gradients
        if sdf_dir is not None and sdf_grad_dir is not None:
            self.sdf_grid, self.sdf_grad_grid = get_sdf(class_id, model_id, sdf_dir, sdf_grad_dir)

        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        # Randomly rotate the self.model_mesh
        model_mesh = copy.deepcopy(self.model_mesh)
        if self.rotate_about_z:
            R = torch.eye(3)
            angle = 2 * self.pi * torch.rand(1)
            c = torch.cos(angle)
            s = torch.sin(angle)

            # # z
            # R[0, 0] = c
            # R[0, 1] = -s
            # R[1, 0] = s
            # R[1, 1] = c

            # # x
            # R[1, 1] = c
            # R[1, 2] = -s
            # R[2, 1] = s
            # R[2, 2] = c

            # y
            R[0, 0] = c
            R[0, 2] = s
            R[2, 0] = -s
            R[2, 2] = c

        else:
            R = transforms.random_rotation()
        model_mesh = model_mesh.rotate(R=R.numpy())

        # Sample a point cloud from the self.model_mesh
        pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_to_sample)

        # Take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_location_factor = beta*(self.radius_multiple[1]-self.radius_multiple[0]) + self.radius_multiple[0]
        camera_location_factor = camera_location_factor * self.diameter
        radius = gu.get_radius(cam_location=camera_location_factor*self.camera_location.numpy(),
                               object_diameter=self.diameter)
        depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)

        depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
        depth_pcd_torch = depth_pcd_torch.to(torch.float)

        keypoints_xyz = R @ self.keypoints_xyz

        # Translate by a random t
        t = torch.rand(3, 1)

        depth_pcd_torch = depth_pcd_torch + t
        keypoints_xyz = keypoints_xyz + t

        model_pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch = model_pcd_torch + t

        point_cloud, padding = self._convert_to_fixed_sized_pc(depth_pcd_torch, n=self.n)

        return point_cloud, keypoints_xyz.squeeze(0), R, t

    def _convert_to_fixed_sized_pc(self, pc, n):
        """
        Adds (0,0,0) points to the point cloud if the number of points is less than
        n such that the resulting point cloud has n points.

        inputs:
        pc  : torch.tensor of shape (3, m)  : input point cloud of size m (m could be anything)
        n   : int                           : number of points the output point cloud should have

        outputs:
        point_cloud     : torch.tensor of shape (3, n)
        padding         : torch.tensor of shape (n)

        """

        m = pc.shape[-1]

        if m > n:
            idx = torch.randperm(m)
            point_cloud = pc[:, idx[:n]]
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        elif m < n:

            pc_pad = torch.zeros(3, n - m)
            point_cloud = torch.cat([pc, pc_pad], dim=1)
            padding1 = torch.zeros(size=(m,), dtype=torch.bool)
            padding2 = torch.ones(size=(n - m,), dtype=torch.bool)
            padding = torch.cat([padding1, padding2], dim=0)
            # Write code to pad pc with (n-m) zeros

        else:
            point_cloud = pc
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        return point_cloud, padding

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """
        if self.n is None:
            self.n = self.num_of_points_to_sample
        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.n)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):
        """
        Returns keypoints of the ShapeNetCore model annotated in the KeypointNet dataset.

        output:
        model_keypoints : torch.tensor of shape (1, 3, N)

        where
        N = number of keypoints
        """
        return self.keypoints_xyz

    def _get_model_sdf(self):
        """Returns SDF grid"""
        return self.sdf_grid

    def _get_model_sdf_grad(self):
        """Return SDF gradient grid"""
        return self.sdf_grad_grid

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


# From C-3PO
class FixedDepthPC(torch.utils.data.Dataset):
    """
        Given class id, model id, and number of points, it generates various depth point clouds and SE3 transformations
        of the ShapeNetCore object.

        Note:
            Outputs depth point clouds of the same shape by appending with
            zero points. It also outputs a flag to tell the user which outputs have been artificially added.

            This dataset can be used with a dataloader for any batch_size.

        Returns
            input_point_cloud, keypoints, rotation, translation
        """

    def __init__(self, class_id, model_id, n=1000, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=10000, dataset_len=1, rotate_about_z=False,
                 base_dataset_folder='../../data/learning_objects/shapenet_depthpc_eval_data/',
                 mixed_data=False):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object 
        n               : int   : number of points in the output point cloud
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points_to_sample   : int   : number of points sampled on the surface of the CAD model object
        dataset_len     : int   : size of the dataset  

        """
        self.base_dataset_folder = base_dataset_folder
        self.class_id = class_id
        self.class_name = CLASS_NAME[self.class_id]
        self.model_id = model_id
        if mixed_data:
            self.dataset_folder = self.base_dataset_folder + 'mixed/'
        else:
            self.dataset_folder = self.base_dataset_folder + self.class_name + '/' + self.model_id + '/'

        self.n = n
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = 1000
        self.len = len(os.listdir(self.dataset_folder))
        self.rotate_about_z = rotate_about_z
        self.pi = torch.tensor([math.pi])

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin


    def __len__(self):

        return self.len

    def __getitem__(self, idx):

        filename = self.dataset_folder + 'item_' + str(idx) + '.pkl'
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)

        return data[0].squeeze(0), data[1].squeeze(0), data[2].squeeze(0), data[3].squeeze(0)

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """
        if self.n is None:
            self.n = self.num_of_points_to_sample
        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.n)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

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

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


# From C-3PO
class MixedFixedDepthPC(torch.utils.data.Dataset):
    """
        Given class id, model id, and number of points, it generates various depth point clouds and SE3 transformations
        of the ShapeNetCore object.

        Note:
            Outputs depth point clouds of the same shape by appending with
            zero points. It also outputs a flag to tell the user which outputs have been artificially added.

            This dataset can be used with a dataloader for any batch_size.

        Returns
            input_point_cloud, keypoints, rotation, translation
        """

    def __init__(self, class_id, model_id, base_dataset_folder, n=1000, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=10000, dataset_len=1, rotate_about_z=False, mixed_data=True):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object 
        n               : int   : number of points in the output point cloud
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points_to_sample   : int   : number of points sampled on the surface of the CAD model object
        dataset_len     : int   : size of the dataset  

        """
        self.base_dataset_folder = base_dataset_folder
        self.class_id = class_id
        self.class_name = CLASS_NAME[self.class_id]
        self.model_id = model_id

        if mixed_data:
            self.dataset_folder = self.base_dataset_folder + 'mixed/'
        else:
            self.dataset_folder = self.base_dataset_folder + self.class_name + '/' + self.model_id + '/'

        self.n = n
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = 1000
        self.len = len(os.listdir(self.dataset_folder))
        self.rotate_about_z = rotate_about_z
        self.pi = torch.tensor([math.pi])

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin


    def __len__(self):

        return self.len

    def __getitem__(self, idx):

        filename = self.dataset_folder + 'item_' + str(idx) + '.pkl'
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)

        return data[0].squeeze(0), data[2].squeeze(0), data[3].squeeze(0)


    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """
        if self.n is None:
            self.n = self.num_of_points
        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.n)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

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

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


# From C-3PO
class SE3PointCloud(torch.utils.data.Dataset):
    """
    Given class_id, model_id, and number of points generates various point clouds and SE3 transformations
    of the ShapeNetCore object.

    Returns a batch of
        input_point_cloud, keypoints, rotation, translation
    """
    def __init__(self, class_id, model_id, num_of_points=1000, dataset_len=10000,
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """

        self.class_id = class_id
        self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

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

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
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

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


# From C-3PO
class SE3nIsotropicShapePointCloud(torch.utils.data.Dataset):
    """
    Given class id, model_id, number of points, and shape_scaling, it generates various point clouds and SE3
    transformations of the ShapeNetCore object, scaled by a quantity between the range determined by shape_scaling.


    Returns a batch of
        input_point_cloud, keypoints, rotation, translation, shape
    """
    def __init__(self, class_id, model_id, num_of_points=1000, dataset_len=10000,
                 shape_scaling=torch.tensor([0.5, 2.0]),
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        shape_scaling   : torch.tensor of shape (2) : lower and upper limit of isotropic shape scaling
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """

        self.class_id = class_id
        self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len
        self.shape_scaling = shape_scaling

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)
        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        depth_pcd_torch     : torch.tensor of shape (3, m)                  : the depth point cloud
        keypoints           : torch.tensor of shape (3, N)                  : transformed keypoints
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        c                   : torch.tensor of shape (2, 1)                  : shape parameter

        """

        # random scaling
        alpha = torch.rand(1, 1)
        scaling_factor = alpha*(self.shape_scaling[1]-self.shape_scaling[0]) + self.shape_scaling[0]

        model_mesh = self.model_mesh
        model_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch = scaling_factor*model_pcd_torch

        keypoints_xyz = scaling_factor*self.keypoints_xyz


        # random rotation and translation
        R = transforms.random_rotation()
        t = torch.rand(3, 1)

        model_pcd_torch = R @ model_pcd_torch + t
        keypoints_xyz = R @ keypoints_xyz + t

        # shape parameter
        shape = torch.zeros(2, 1)
        shape[0] = 1 - alpha
        shape[1] = alpha

        return model_pcd_torch, keypoints_xyz.squeeze(0), R, t, shape

    def _get_cad_models(self):
        """
        Returns two point clouds as shape models, one with the min shape and the other with the max shape.

        output:
        cad_models  : torch.tensor of shape (2, 3, self.num_of_points)
        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        cad_models = torch.zeros_like(model_pcd_torch).unsqueeze(0)
        cad_models = cad_models.repeat(2, 1, 1)

        cad_models[0, ...] = self.shape_scaling[0]*model_pcd_torch
        cad_models[1, ...] = self.shape_scaling[1]*model_pcd_torch

        return cad_models

    def _get_model_keypoints(self):
        """
        Returns two sets of keypoints, one with the min shape and the other with the max shape.

        output:
        model_keypoints : torch.tensor of shape (2, 3, N)

        where
        N = number of keypoints
        """

        keypoints = self.keypoints_xyz  # (3, N)

        model_keypoints = torch.zeros_like(keypoints)
        model_keypoints = model_keypoints.repeat(2, 1, 1)

        model_keypoints[0, ...] = self.shape_scaling[0]*keypoints
        model_keypoints[1, ...] = self.shape_scaling[1]*keypoints

        return model_keypoints

    def _get_diameter(self):
        """
        returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter*(self.shape_scaling[0] + self.shape_scaling[1])*0.5

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


# From C-3PO
class SE3nAnisotropicScalingPointCloud(torch.utils.data.Dataset):
    """
    Given class id, model_id, number of points, and shape_scaling, it generates various point clouds and SE3
    transformations of the ShapeNetCore object, scaled by a quantity between the range determined by shape_scaling
    in the direction described by scale_direction before se3 transformation.


    Returns a batch of
        input_point_cloud, keypoints, rotation, translation, shape
    """
    def __init__(self, class_id, model_id, num_of_points=1000, dataset_len=10000,
                 shape_scaling=torch.tensor([0.5, 2.0]), scale_direction=ScaleAxis.X,
                 dir_location='../../data/learning-objects/keypointnet_datasets/',
                 shape_dataset=None):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        shape_scaling   : torch.tensor of shape (2) : lower and upper limit of isotropic shape scaling
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """

        self.class_id = class_id
        self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len
        self.shape_scaling = shape_scaling
        self.scale_direction = scale_direction.value
        self.shape_dataset = shape_dataset

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)
        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))


    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        depth_pcd_torch     : torch.tensor of shape (3, m)                  : the depth point cloud
        keypoints           : torch.tensor of shape (3, N)                  : transformed keypoints
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        c                   : torch.tensor of shape (2, 1)                  : shape parameter

        """

        # random scaling
        if self.shape_dataset is not None:
            alpha = torch.tensor(self.shape_dataset[idx])
        else:
            alpha = torch.rand(1, 1)

        scaling_factor = alpha*(self.shape_scaling[1]-self.shape_scaling[0]) + self.shape_scaling[0]

        model_mesh = self.model_mesh
        model_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch[self.scale_direction] = scaling_factor * model_pcd_torch[self.scale_direction]
        keypoints_xyz = torch.clone(self.keypoints_xyz.squeeze())
        keypoints_xyz[self.scale_direction] = scaling_factor * keypoints_xyz[self.scale_direction]

        # random rotation and translation
        R = transforms.random_rotation()
        t = torch.rand(3, 1)

        model_pcd_torch = R @ model_pcd_torch + t
        keypoints_xyz = R @ keypoints_xyz + t

        # shape parameter
        shape = torch.zeros(2, 1)
        shape[0] = 1 - alpha
        shape[1] = alpha

        return model_pcd_torch, keypoints_xyz, R, t, shape


    def _get_cad_models(self):
        """
        Returns two point clouds as shape models, one with the min shape and the other with the max shape.

        output:
        cad_models  : torch.tensor of shape (2, 3, self.num_of_points)
        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        cad_models = torch.zeros_like(model_pcd_torch).unsqueeze(0)
        cad_models = cad_models.repeat(2, 1, 1)

        cad_models[0, ...] = torch.clone(model_pcd_torch)
        cad_models[1, ...] = torch.clone(model_pcd_torch)

        cad_models[0, self.scale_direction, ...] = self.shape_scaling[0] * cad_models[0, self.scale_direction, ...]
        cad_models[1, self.scale_direction, ...] = self.shape_scaling[1] * cad_models[1, self.scale_direction, ...]

        return cad_models

    def _get_model_keypoints(self):
        """
        Returns two sets of keypoints, one with the min shape and the other with the max shape.

        output:
        model_keypoints : torch.tensor of shape (2, 3, N)

        where
        N = number of keypoints
        """

        keypoints = self.keypoints_xyz  # (3, N)

        model_keypoints = torch.zeros_like(keypoints)
        model_keypoints = model_keypoints.repeat(2, 1, 1)

        model_keypoints[0, ...] = torch.clone(keypoints)
        model_keypoints[1, ...] = torch.clone(keypoints)
        model_keypoints[0,self.scale_direction, ...] = self.shape_scaling[0] * model_keypoints[0,self.scale_direction, ...]
        model_keypoints[1,self.scale_direction, ...] = self.shape_scaling[1] * model_keypoints[1,self.scale_direction, ...]

        return model_keypoints

    def _get_diameter(self):
        """
        returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter*(self.shape_scaling[0] + self.shape_scaling[1])*0.5

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


#
class DepthPCwVoxelSDFModel(torch.utils.data.Dataset):
    """
    Given class id, model id, and number of points, it generates various depth point clouds and SE3 transformations
    of the ShapeNetCore object.

    Note:
        Outputs depth point clouds of the same shape by appending with
        zero points. It also outputs a flag to tell the user which outputs have been artificially added.

        This dataset can be used with a dataloader for any batch_size.

    Returns
        input_point_cloud, keypoints, rotation, translation
    """

    def __init__(self, class_id, model_id, n=1000, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=10000, dataset_len=10000, rotate_about_z=False):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object 
        n               : int   : number of points in the output point cloud
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points_to_sample   : int   : number of points sampled on the surface of the CAD model object
        dataset_len     : int   : size of the dataset  

        """

        self.class_id = class_id
        self.model_id = model_id
        self.n = n
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = num_of_points_to_sample
        self.len = dataset_len
        self.rotate_about_z = rotate_about_z
        self.pi = torch.tensor([math.pi])

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)    #ToDo: Need to change get_model_and_keypoints to provide SDF model for self.model_mesh
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        # Randomly rotate the self.model_mesh
        model_mesh = copy.deepcopy(self.model_mesh)
        if self.rotate_about_z:
            R = torch.eye(3)
            angle = 2 * self.pi * torch.rand(1)
            c = torch.cos(angle)
            s = torch.sin(angle)

            # # z
            # R[0, 0] = c
            # R[0, 1] = -s
            # R[1, 0] = s
            # R[1, 1] = c

            # # x
            # R[1, 1] = c
            # R[1, 2] = -s
            # R[2, 1] = s
            # R[2, 2] = c

            # y
            R[0, 0] = c
            R[0, 2] = s
            R[2, 0] = -s
            R[2, 2] = c

        else:
            R = transforms.random_rotation()
        model_mesh = model_mesh.rotate(R=R.numpy())                                                                     #ToDo: Update. As model_mesh is now SDF.

        # Sample a point cloud from the self.model_mesh
        pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_to_sample)                         #ToDo: Update. As model_mesh is now SDF.

        # Take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_location_factor = beta*(self.radius_multiple[1]-self.radius_multiple[0]) + self.radius_multiple[0]
        camera_location_factor = camera_location_factor * self.diameter
        radius = gu.get_radius(cam_location=camera_location_factor*self.camera_location.numpy(),
                               object_diameter=self.diameter)
        depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)

        depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
        depth_pcd_torch = depth_pcd_torch.to(torch.float)

        keypoints_xyz = R @ self.keypoints_xyz

        # Translate by a random t
        t = torch.rand(3, 1)

        depth_pcd_torch = depth_pcd_torch + t
        keypoints_xyz = keypoints_xyz + t

        model_pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch = model_pcd_torch + t

        point_cloud, padding = self._convert_to_fixed_sized_pc(depth_pcd_torch, n=self.n)

        return point_cloud, keypoints_xyz.squeeze(0), R, t

    def _convert_to_fixed_sized_pc(self, pc, n):
        """
        Adds (0,0,0) points to the point cloud if the number of points is less than
        n such that the resulting point cloud has n points.

        inputs:
        pc  : torch.tensor of shape (3, m)  : input point cloud of size m (m could be anything)
        n   : int                           : number of points the output point cloud should have

        outputs:
        point_cloud     : torch.tensor of shape (3, n)
        padding         : torch.tensor of shape (n)

        """

        m = pc.shape[-1]

        if m > n:
            idx = torch.randperm(m)
            point_cloud = pc[:, idx[:n]]
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        elif m < n:

            pc_pad = torch.zeros(3, n - m)
            point_cloud = torch.cat([pc, pc_pad], dim=1)
            padding1 = torch.zeros(size=(m,), dtype=torch.bool)
            padding2 = torch.ones(size=(n - m,), dtype=torch.bool)
            padding = torch.cat([padding1, padding2], dim=0)
            # Write code to pad pc with (n-m) zeros

        else:
            point_cloud = pc
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        return point_cloud, padding

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """
        if self.n is None:
            self.n = self.num_of_points_to_sample
        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.n)                                    #ToDo: Update. As model_mesh is now SDF.
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)                                                                             #ToDo: Update. As model_mesh is now SDF. The output should be the SDF function.

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

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)                       #ToDo: Update. As model_mesh is now SDF.

        return 0


#
class DepthPCwDeepSDFModel(torch.utils.data.Dataset):
    """
    Given class id, model id, and number of points, it generates various depth point clouds and SE3 transformations
    of the ShapeNetCore object.

    Note:
        Outputs depth point clouds of the same shape by appending with
        zero points. It also outputs a flag to tell the user which outputs have been artificially added.

        This dataset can be used with a dataloader for any batch_size.

    Returns
        input_point_cloud, keypoints, rotation, translation
    """

    def __init__(self, class_id, model_id, n=1000, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=10000, dataset_len=10000, rotate_about_z=False):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object 
        n               : int   : number of points in the output point cloud
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points_to_sample   : int   : number of points sampled on the surface of the CAD model object
        dataset_len     : int   : size of the dataset  

        """

        self.class_id = class_id
        self.model_id = model_id
        self.n = n
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = num_of_points_to_sample
        self.len = dataset_len
        self.rotate_about_z = rotate_about_z
        self.pi = torch.tensor([math.pi])

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)    #ToDo: Need to change get_model_and_keypoints to provide SDF model for self.model_mesh
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        # Randomly rotate the self.model_mesh
        model_mesh = copy.deepcopy(self.model_mesh)
        if self.rotate_about_z:
            R = torch.eye(3)
            angle = 2 * self.pi * torch.rand(1)
            c = torch.cos(angle)
            s = torch.sin(angle)

            # # z
            # R[0, 0] = c
            # R[0, 1] = -s
            # R[1, 0] = s
            # R[1, 1] = c

            # # x
            # R[1, 1] = c
            # R[1, 2] = -s
            # R[2, 1] = s
            # R[2, 2] = c

            # y
            R[0, 0] = c
            R[0, 2] = s
            R[2, 0] = -s
            R[2, 2] = c

        else:
            R = transforms.random_rotation()
        model_mesh = model_mesh.rotate(R=R.numpy())                                                                     #ToDo: Update. As model_mesh is now SDF.

        # Sample a point cloud from the self.model_mesh
        pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_to_sample)                         #ToDo: Update. As model_mesh is now SDF.

        # Take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_location_factor = beta*(self.radius_multiple[1]-self.radius_multiple[0]) + self.radius_multiple[0]
        camera_location_factor = camera_location_factor * self.diameter
        radius = gu.get_radius(cam_location=camera_location_factor*self.camera_location.numpy(),
                               object_diameter=self.diameter)
        depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)

        depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
        depth_pcd_torch = depth_pcd_torch.to(torch.float)

        keypoints_xyz = R @ self.keypoints_xyz

        # Translate by a random t
        t = torch.rand(3, 1)

        depth_pcd_torch = depth_pcd_torch + t
        keypoints_xyz = keypoints_xyz + t

        model_pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch = model_pcd_torch + t

        point_cloud, padding = self._convert_to_fixed_sized_pc(depth_pcd_torch, n=self.n)

        return point_cloud, keypoints_xyz.squeeze(0), R, t

    def _convert_to_fixed_sized_pc(self, pc, n):
        """
        Adds (0,0,0) points to the point cloud if the number of points is less than
        n such that the resulting point cloud has n points.

        inputs:
        pc  : torch.tensor of shape (3, m)  : input point cloud of size m (m could be anything)
        n   : int                           : number of points the output point cloud should have

        outputs:
        point_cloud     : torch.tensor of shape (3, n)
        padding         : torch.tensor of shape (n)

        """

        m = pc.shape[-1]

        if m > n:
            idx = torch.randperm(m)
            point_cloud = pc[:, idx[:n]]
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        elif m < n:

            pc_pad = torch.zeros(3, n - m)
            point_cloud = torch.cat([pc, pc_pad], dim=1)
            padding1 = torch.zeros(size=(m,), dtype=torch.bool)
            padding2 = torch.ones(size=(n - m,), dtype=torch.bool)
            padding = torch.cat([padding1, padding2], dim=0)
            # Write code to pad pc with (n-m) zeros

        else:
            point_cloud = pc
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        return point_cloud, padding

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """
        # TODO: add return sdf function
        if self.n is None:
            self.n = self.num_of_points_to_sample
        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.n)                                    #ToDo: Update. As model_mesh is now SDF.
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)                                                                             #ToDo: Update. As model_mesh is now SDF. The output should be the SDF function.

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

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)                       #ToDo: Update. As model_mesh is now SDF.

        return 0




if __name__ == "__main__":

    dir_location = '../../data/learning_objects/keypointnet_datasets/'
    class_id = "03001627"  # chair
    model_id = "1cc6f2ed3d684fa245f213b8994b4a04"  # a particular chair model
    batch_size = 5


    #
    print("Test: DepthIsoPC()")
    dataset = DepthIsoPC(class_id=class_id, model_id=model_id)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t, c = data
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        # visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 2:
            break
    print("Test: DepthAnsoPC()")
    dataset = DepthAnisoPC(class_id=class_id, model_id=model_id)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t, c = data
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        # visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 2:
            break


    #
    print("Test: DepthPC()")
    dataset = DepthPC(class_id=class_id, model_id=model_id)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t = data
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        # visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 2:
            break

    #
    print("Test: get_model_and_keypoints()")
    mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id=class_id, model_id=model_id)
    # print(keypoints_xyz)
    # print(type(keypoints_xyz))
    # print(type(keypoints_xyz[0]))

    #
    print("Test: visualize_model_n_keypoints()")
    visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)

    #
    print("Test: visualize_model()")
    visualize_model(class_id=class_id, model_id=model_id)

    #
    print("Test: SE3PointCloud(torch.utils.data.Dataset)")
    dataset = SE3PointCloud(class_id=class_id, model_id=model_id)

    model = dataset.model_mesh
    length = dataset.len
    class_id = dataset.class_id
    model_id = dataset.model_id
    num_of_points = dataset.num_of_points

    print("Shape of keypoints_xyz: ", keypoints_xyz.shape)

    diameter = dataset._get_diameter()
    model_keypoints = dataset._get_model_keypoints()
    cad_models = dataset._get_cad_models()


    print("diameter: ", diameter)
    print("shape of model keypoints: ", model_keypoints.shape)
    print("shape of cad models: ", cad_models.shape)

    #
    print("Test: visualize_torch_model_n_keypoints()")
    visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)
    dataset._visualize()


    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t = data
        print(pc.shape)
        print(kp.shape)
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 2:
            break

    dataset = FixedDepthPC(class_id=class_id, model_id=model_id)
    model = dataset.model_mesh
    length = dataset.len
    class_id = dataset.class_id
    model_id = dataset.model_id

    diameter = dataset._get_diameter()
    model_keypoints = dataset._get_model_keypoints()
    cad_models = dataset._get_cad_models()

    print("diameter: ", diameter)
    print("shape of model keypoints: ", model_keypoints.shape)
    print("shape of cad models: ", cad_models.shape)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t = data
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        # visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 2:
            break

    #
    print("Test: SE3nIsotropicShapePointCloud(torch.utils.data.Dataset)")
    dataset = SE3nIsotropicShapePointCloud(class_id=class_id, model_id=model_id,
                                           shape_scaling=torch.tensor([5.0, 20.0]))
    model = dataset.model_mesh
    length = dataset.len
    class_id = dataset.class_id
    model_id = dataset.model_id

    diameter = dataset._get_diameter()
    model_keypoints = dataset._get_model_keypoints()
    cad_models = dataset._get_cad_models()

    print("diameter: ", diameter)
    print("shape of model keypoints: ", model_keypoints.shape)
    print("shape of cad models: ", cad_models.shape)

    #
    print("Test: visualize_torch_model_n_keypoints()")
    visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)
    dataset._visualize()

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    modelgen = ModelFromShape(cad_models=cad_models, model_keypoints=model_keypoints)

    for i, data in enumerate(loader):
        pc, kp, R, t, c = data
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)

        # getting the model from R, t, c
        kp_gen, pc_gen = modelgen.forward(shape=c)
        kp_gen = R @ kp_gen + t
        pc_gen = R @ pc_gen + t
        display_two_pcs(pc1=pc, pc2=pc_gen)
        display_two_pcs(pc1=kp, pc2=kp_gen)

        if i >= 5:
            break

    #
    print("Test: SE3nAnisotropicScalingPointCloud(torch.utils.data.Dataset)")
    dataset = SE3nAnisotropicScalingPointCloud(class_id=class_id, model_id=model_id,
                                           shape_scaling=torch.tensor([1.0, 2.0]),
                                           scale_direction=ScaleAxis.X)

    model = dataset.model_mesh
    length = dataset.len
    class_id = dataset.class_id
    model_id = dataset.model_id

    diameter = dataset._get_diameter()
    model_keypoints = dataset._get_model_keypoints()
    cad_models = dataset._get_cad_models()

    print("diameter: ", diameter)
    print("shape of model keypoints: ", model_keypoints.shape)
    print("shape of cad models: ", cad_models.shape)

    #
    print("Test: visualize_torch_model_n_keypoints()")
    visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)
    dataset._visualize()


    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    modelgen = ModelFromShape(cad_models=cad_models, model_keypoints=model_keypoints)

    for i, data in enumerate(loader):
        pc, kp, R, t, c = data
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)


        # getting the model from R, t, c
        kp_gen, pc_gen = modelgen.forward(shape=c)
        kp_gen = R @ kp_gen + t
        pc_gen = R @ pc_gen + t
        display_two_pcs(pc1=pc, pc2=pc_gen)
        display_two_pcs(pc1=kp, pc2=kp_gen)

        if i >= 5:
            break

