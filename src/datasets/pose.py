import logging
import os
import numpy as np
import random
from tqdm import tqdm
import torch
import pickle
from dataclasses import dataclass

import datasets.bop
from cosypose.config import LOCAL_DATA_DIR
from cosypose.datasets.augmentations import (
    CropResizeToAspectAugmentation,
    VOCBackgroundAugmentation,
    PillowBlur,
    PillowSharpness,
    PillowContrast,
    PillowBrightness,
    PillowColor,
    to_torch_uint8,
    GrayScale,
)
from casper3d.robust_centroid import robust_centroid_gnc
from cosypose.lib3d import invert_T
from utils.math_utils import depth_to_point_cloud_torch, depth_to_point_cloud_with_rgb_torch
from .wrappers.visibility_wrapper import VisibilityWrapper


def get_obj_data_from_bop_detections(
    idx,
    object_id,
    obj_keypoints,
    obj_diameter,
    pc_size,
    load_rgb_for_points,
    zero_center_pc,
    use_robust_centroid,
    normalize_pc,
    resample_invalid_pts,
    scene_frame_index,
    scene_ds,
    bop_detections,
):
    """Helper function to compute data given BOP detections
    Note that we don't have ground truth transformation for the detections.
    """
    obj_info = scene_frame_index[idx]
    frame_idx_row, scene_id, view_id, filtered_mask = (
        obj_info["index_row"],
        obj_info["scene_id"],
        obj_info["view_id"],
        obj_info["mask"].astype(np.uint8),
    )

    rgb, _, state = scene_ds[frame_idx_row]
    rgb, filtered_mask = to_torch_uint8(rgb), to_torch_uint8(filtered_mask)

    depths = torch.as_tensor(state["camera"]["depth"])

    K = np.asarray(state["camera"]["K"])
    assert K.shape[0] == 3
    assert K.shape[1] == 3

    rgb = torch.as_tensor(rgb).to(torch.uint8)
    assert rgb.shape[-1] == 3

    if load_rgb_for_points:
        temp_pc = depth_to_point_cloud_with_rgb_torch(
            depth=depths,
            rgb=rgb,
            K=K,
            mask=filtered_mask,
            x_index=1,
            y_index=0,
            pc_size=pc_size,
        )

    else:
        temp_pc = depth_to_point_cloud_torch(
            depth=depths,
            K=K,
            mask=filtered_mask,
            x_index=1,
            y_index=0,
            pc_size=pc_size,
        )

    if temp_pc is None:
        return None

    temp_pc = np.asarray(temp_pc)

    pc = torch.as_tensor(temp_pc).clone()
    num_zero_pts = torch.sum(pc == 0, dim=0)
    invalid_pts_mask = num_zero_pts >= 3
    center = None
    if zero_center_pc:
        # center the point cloud before returning
        num_zero_pts = torch.sum(invalid_pts_mask, dim=0)
        num_nonzero_pts = pc_size - num_zero_pts

        if num_nonzero_pts <= 1:
            logging.warning(
                f"Point cloud size = {num_nonzero_pts} (<1) encountered, caused by invalid depth measurements."
            )
            return None

        center = torch.sum(pc[:3, :], dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        est_outlier_mask = torch.zeros(pc_size).to(bool)

        if use_robust_centroid:
            ro_cent_pyld = robust_centroid_gnc(
                input_point_cloud=pc[:3, torch.logical_not(invalid_pts_mask)].unsqueeze(0),
                cost_type="gnc-tls",
                clamp_thres=obj_diameter / 1.9,
                cost_abs_stop_th=1e-5,
                cost_diff_stop_th=1e-5,
            )

            temp_est_outlier_mask = (ro_cent_pyld["weights"] < 0.5).squeeze()
            if torch.sum(temp_est_outlier_mask) < pc_size - num_zero_pts:
                # temp_est_outlier_mask is of size=number of valid points
                # we need to generate a full outlier mask of size=self.pc
                # 1. get indices of valid points
                # 2. select the outlier indices and fill those indices with one
                pc_valid_inds = torch.argwhere(torch.logical_not(invalid_pts_mask)).squeeze()
                est_outlier_mask.index_fill_(0, pc_valid_inds[temp_est_outlier_mask], 1)
                center = ro_cent_pyld["robust_centroid"].squeeze(0)
            else:
                logging.warning(
                    f"Robust centroid estimates all points to be outliers (index: {idx}; pc size: {num_nonzero_pts}). "
                    f"Using non-robust centroid as backup."
                )

        # if a point is invalid (=0):
        # -  do not subtract center
        # if a point is an outlier
        # - set to zero
        # - do not subtract center
        # end effects: invalid points U outliers = 0
        # so we can:
        # 1. OR outlier mask & invalid mask = points to zero mask
        # 2. subtract center & normalize everything
        # 3. set points in points to zero mask zero
        # 4. optionally resample
        pc[:3, :] = pc[:3, :] - center.expand(3, pc_size)

        # update valid pts mask
        invalid_pts_mask = torch.logical_or(invalid_pts_mask, est_outlier_mask)

        if normalize_pc:
            # normalize the output point cloud by object diameter
            pc /= obj_diameter

    if resample_invalid_pts:
        valid_pts_mask = torch.logical_not(invalid_pts_mask)
        num_invalid_pts = torch.sum(invalid_pts_mask)

        if num_invalid_pts > 0:
            # select valid pts
            valid_pts = pc[:, valid_pts_mask]

            # sample the # of invalid pts from the valid pts
            sampled_valid_inds = torch.randint(low=0, high=valid_pts.shape[-1], size=(num_invalid_pts.item(),))

            # replace invalid pts
            pc[:, invalid_pts_mask] = torch.index_select(pc, dim=1, index=sampled_valid_inds)

    final_invalid_mask = (pc[:3, :] == torch.zeros(3, 1).to(device=pc.device)).sum(dim=0) == 3
    if torch.any(final_invalid_mask):
        logging.warning(f"Invalid point cloud encountered.")
        return None

    # note:
    # We don't save the RGB due to memory constraints.
    # In the get_item() functions of the relevant pytorch dataset classes,
    # frame_idx_row can be used to recover rgb
    pyld = dict(
        centered_normalized_pc=pc,
        centroid=center,
        K=torch.as_tensor(state["camera"]["K"]).float(),
        scene_id=scene_id,
        view_id=view_id,
        frame_idx_row=frame_idx_row,
        bbox=obj_info["bbox"],
        category_id=obj_info['category_id'],
        score=obj_info["score"]
    )
    return pyld


def get_obj_data_from_scene_frame_index(
    idx,
    object_id,
    obj_keypoints,
    obj_diameter,
    pc_size,
    load_rgb_for_points,
    zero_center_pc,
    use_robust_centroid,
    normalize_pc,
    resample_invalid_pts,
    scene_frame_index,
    scene_ds,
):
    """Helper function to load a specific point cloud and images from a object scene frame index"""
    # access the scene and view
    obj_info = scene_frame_index[idx]
    frame_idx_row, scene_id, view_id, obj_index_in_view = (
        obj_info["index_row"],
        obj_info["scene_id"],
        obj_info["view_id"],
        obj_info["obj_index_in_view"],
    )

    rgb, mask, state = scene_ds[frame_idx_row]
    rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)

    depths = torch.as_tensor(state["camera"]["depth"])

    K = np.asarray(state["camera"]["K"])
    assert K.shape[0] == 3
    assert K.shape[1] == 3

    # access the object
    obj = state["objects"][obj_index_in_view]
    assert obj["label"] == object_id
    rgb = torch.as_tensor(rgb).to(torch.uint8)
    assert rgb.shape[-1] == 3

    # preprocess objects data
    # 1. calculate TCO
    TWO = torch.as_tensor(obj["TWO"])
    TWC = torch.as_tensor(state["camera"]["TWC"])
    TCO = invert_T(TWC) @ TWO
    obj["TCO"] = torch.as_tensor(np.asarray(TCO))

    # 2. extract point clouds for the objects
    mask_uniqs = set(np.unique(mask))
    if obj["id_in_segm"] not in mask_uniqs:
        logging.warning(f"Skipping {idx} as it is not in the seg mask. info: {obj_info}.")
        return None

    filtered_mask = mask == obj["id_in_segm"]
    if load_rgb_for_points:
        temp_pc = depth_to_point_cloud_with_rgb_torch(
            depth=depths,
            rgb=rgb,
            K=K,
            mask=filtered_mask,
            x_index=1,
            y_index=0,
            pc_size=pc_size,
        )

    else:
        temp_pc = depth_to_point_cloud_torch(
            depth=depths,
            K=K,
            mask=filtered_mask,
            x_index=1,
            y_index=0,
            pc_size=pc_size,
        )

    if temp_pc is None:
        return None
    temp_pc = np.asarray(temp_pc)

    # keypoints
    transformed_kpts = obj["TCO"][:3, :3] @ obj_keypoints + obj["TCO"][:3, -1][:, None]

    pc = torch.as_tensor(temp_pc).clone()
    num_zero_pts = torch.sum(pc == 0, dim=0)
    invalid_pts_mask = num_zero_pts >= 3
    center = None
    cent_R_cad, cent_t_cad = obj["TCO"][:3, :3], obj["TCO"][:3, -1]
    if zero_center_pc:
        # center the point cloud before returning
        num_zero_pts = torch.sum(invalid_pts_mask, dim=0)
        num_nonzero_pts = pc_size - num_zero_pts

        if num_nonzero_pts <= 1:
            logging.warning(
                f"Point cloud size = {num_nonzero_pts} (<1) encountered, caused by invalid depth measurements."
            )
            return None

        center = torch.sum(pc[:3, :], dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        est_outlier_mask = torch.zeros(pc_size).to(bool)

        if use_robust_centroid:
            ro_cent_pyld = robust_centroid_gnc(
                input_point_cloud=pc[:3, torch.logical_not(invalid_pts_mask)].unsqueeze(0),
                cost_type="gnc-tls",
                clamp_thres=obj_diameter / 1.9,
                cost_abs_stop_th=1e-5,
                cost_diff_stop_th=1e-5,
            )

            temp_est_outlier_mask = (ro_cent_pyld["weights"] < 0.5).squeeze()
            if torch.sum(temp_est_outlier_mask) < pc_size - num_zero_pts:
                # temp_est_outlier_mask is of size=number of valid points
                # we need to generate a full outlier mask of size=self.pc
                # 1. get indices of valid points
                # 2. select the outlier indices and fill those indices with one
                pc_valid_inds = torch.argwhere(torch.logical_not(invalid_pts_mask)).squeeze()
                est_outlier_mask.index_fill_(0, pc_valid_inds[temp_est_outlier_mask], 1)
                center = ro_cent_pyld["robust_centroid"].squeeze(0)
            else:
                logging.warning(
                    f"Robust centroid estimates all points to be outliers (index: {idx}; pc size: {num_nonzero_pts}). "
                    f"Using non-robust centroid as backup."
                )

        # if a point is invalid (=0):
        # -  do not subtract center
        # if a point is an outlier
        # - set to zero
        # - do not subtract center
        # end effects: invalid points U outliers = 0
        # so we can:
        # 1. OR outlier mask & invalid mask = points to zero mask
        # 2. subtract center & normalize everything
        # 3. set points in points to zero mask zero
        # 4. optionally resample
        pc[:3, :] = pc[:3, :] - center.expand(3, pc_size)

        # update valid pts mask
        invalid_pts_mask = torch.logical_or(invalid_pts_mask, est_outlier_mask)

        # center the keypoints & update the gt translation
        transformed_kpts -= center.expand(3, transformed_kpts.shape[-1])
        cent_t_cad -= center.squeeze()

        if normalize_pc:
            # normalize the output point cloud by object diameter
            pc /= obj_diameter
            transformed_kpts /= obj_diameter

    if resample_invalid_pts:
        valid_pts_mask = torch.logical_not(invalid_pts_mask)
        num_invalid_pts = torch.sum(invalid_pts_mask)

        if num_invalid_pts > 0:
            # select valid pts
            valid_pts = pc[:, valid_pts_mask]

            # sample the # of invalid pts from the valid pts
            sampled_valid_inds = torch.randint(low=0, high=valid_pts.shape[-1], size=(num_invalid_pts.item(),))

            # replace invalid pts
            pc[:, invalid_pts_mask] = torch.index_select(pc, dim=1, index=sampled_valid_inds)

    final_invalid_mask = (pc[:3, :] == torch.zeros(3, 1).to(device=pc.device)).sum(dim=0) == 3
    if torch.any(final_invalid_mask):
        logging.warning(f"Invalid point cloud encountered.")
        return None

    # note:
    # We don't save the RGB & masks due to memory constraints.
    # In the get_item() functions of the relevant pytorch dataset classes,
    # frame_idx_row can be used to recover rgb & mask from scene dataset.
    pyld = dict(
        centered_normalized_pc=pc,
        centered_normalized_kpts=transformed_kpts,
        cent_R_cad=cent_R_cad,
        cent_t_cad=cent_t_cad,
        centroid=center,
        K=torch.as_tensor(state["camera"]["K"]).float(),
        obj=obj,
        frame_idx_row=frame_idx_row,
    )
    return pyld


@dataclass
class PoseData:
    images: None
    depths: None
    masks: None
    bboxes: None
    TCO: None
    K: None
    objects: None

    def pin_memory(self):
        self.images = self.images.pin_memory()
        self.depths = self.depths.pin_memory()
        self.masks = self.masks.pin_memory()
        self.bboxes = self.bboxes.pin_memory()
        self.TCO = self.TCO.pin_memory()
        self.K = self.K.pin_memory()
        return self


@dataclass
class MultiPoseData:
    """This is a test class for dataclasses.

    This is the body of the docstring description.

    Args:
        images: (B, 3, H, W)
        depths: (B, H, W)
        masks: (B, H, W)
        K: (B, 3, 3)
        objects: list of all objects' info dictionaries
        frame_to_objects_index: dictionary with frame id in batch as key, and indices of objects as items
        model_to_objects_index: dictionary with model name as key, and indices of objects as items
        model_to_batched_pcs: dictionary with model name as key, and batched point clouds (B', 3, N) where B' is
        the total number of objects with the specific model name inside this batch of frames
    """

    # fixed size tensors
    images: None
    depths: None
    masks: None
    K: None
    # objects related (variable size across items in one batch)
    objects: None
    frame_to_objects_index: None
    model_to_objects_index: None
    model_to_batched_pcs: None
    model_to_batched_gt_R: None
    model_to_batched_gt_t: None

    def pin_memory(self):
        self.images = self.images.pin_memory()
        self.depths = self.depths.pin_memory()
        self.masks = self.masks.pin_memory()
        self.K = self.K.pin_memory()
        for k, v in self.model_to_batched_pcs.items():
            for x in v:
                x.pin_memory()
        for k, v in self.model_to_batched_gt_R.items():
            for x in v:
                x.pin_memory()
        for k, v in self.model_to_batched_gt_t.items():
            for x in v:
                x.pin_memory()
        return self


class NoObjectError(Exception):
    pass


class ObjectPoseDatasetBase(torch.utils.data.Dataset):
    """
    Dataset class to represent objects in the BOP dataset. During initialization, this class builds an index over all
    the BOP images objects, and allows for returns of a fixed batch size objects training data per iteration.
    This returns both the point cloud and RGB images for multi model training.

    Author: Jingnan Shi
    """

    def __init__(
        self,
        scene_ds,
        object_id,
        object_diameter=None,
        dataset_name=None,
        min_area=None,
        pc_size=1000,
        load_rgb_for_points=True,
        zero_center_pc=False,
        use_robust_centroid=False,
        resample_invalid_pts=False,
        normalize_pc=False,
        load_data_from_cache=False,
        cache_save_dir=None,
        bop_detections=None,
    ):
        """
        Args:
            object_diameter:
            scene_ds:
            object_id:
            dataset_name:
            min_area:
            all_model_names:
            pc_size:
            load_rgb_for_points:
            zero_center_pc: set to True if you want the returned object to be centered
            use_robust_centroid: set to True to use robust centroid
            resample_invalid_pts:
            normalize_pc: set to True to normalize the point cloud by object diameter; only when point cloud is centered
        """
        self.dataset_name = dataset_name
        self.scene_ds = scene_ds
        self.object_id = object_id
        self.object_diameter = object_diameter
        self.min_area = min_area
        self.pc_size = pc_size
        self.load_rgb_for_points = load_rgb_for_points
        self.zero_center_pc = zero_center_pc
        self.use_robust_centroid = use_robust_centroid
        self.resample_invalid_pts = resample_invalid_pts
        self.normalize_pc = normalize_pc
        self.load_data_from_cache = load_data_from_cache
        self.preloaded_data = None
        if cache_save_dir is None:
            self.cache_save_dir = self.scene_ds.ds_dir
        else:
            self.cache_save_dir = cache_save_dir

        if bop_detections is None:
            self.save_file_name = os.path.join(
                self.cache_save_dir, f"{self.scene_ds.split}_{self.object_id}_pc_img_data.pkl"
            )
            self.bop_detections = None
        else:
            self.save_file_name = os.path.join(
                self.cache_save_dir, f"{self.scene_ds.split}_{self.object_id}_bop_default_pc_img_data.pkl"
            )
            self.bop_detections = bop_detections

        # check if obj index exist
        if bop_detections is None:
            assert self.scene_ds.obj_img_index is not None
            # self.scene_frame_index: a list of tuples of (scene_id, view_id) that contains
            # the specified object
            self.scene_frame_index = self._preprocess(self.scene_ds.obj_img_index[object_id])
        else:
            self.scene_frame_index = self._preprocess_bop_detections(
                object_id=object_id, bop_detections=bop_detections, frame_index=self.scene_ds.frame_index
            )

        self.length = len(self.scene_frame_index)

        # load keypoints and models
        assert dataset_name is not None
        self.keypoints = datasets.bop.get_keypoints(object_label=object_id, ds_dir=self.scene_ds.ds_dir)
        # units: mm to m
        self.keypoints /= 1000

        # preload detections
        if self.load_data_from_cache:
            self._preload_dataset()

    def _preprocess_bop_detections(self, object_id, bop_detections, frame_index):
        """Do some preliminary filtering based on metadata information"""
        processed_entries = []
        for i, (s_id, v_id) in enumerate(zip(frame_index["scene_id"], frame_index["view_id"])):
            # go through all detections in the bop detection that contains our interest object
            c_dets = [
                x for x in bop_detections.get_detections(s_id, v_id) if f"obj_{x['category_id']:06d}" == object_id
            ]
            if len(c_dets) == 0:
                continue

            for det in c_dets:
                w, h = det["bbox"][2], det["bbox"][3]
                payload = dict(
                    index_row=i,
                    scene_id=s_id,
                    view_id=v_id,
                    visib_area=w * h,
                    mask=det["segmentation"]["mask"],
                    bbox=det["bbox"],
                    category_id=det['category_id'],
                    score=det["score"]
                )
                processed_entries.append(payload)

        return processed_entries

    def _preprocess(self, scene_frame_index):
        """Do some preliminary filtering based on metadata information"""
        processed_entries = []
        for entry in scene_frame_index:
            add = True
            # only keep visible objects
            if entry["visib_fract"] <= 0.002:
                add = False

            # not enough valid pixels
            if entry["px_count_valid"] < 10:
                add = False

            # not enough visible pixels
            if entry["px_count_visib"] < 10:
                add = False

            # area check
            if add and self.min_area is not None:
                if entry["visib_area"] < self.min_area:
                    add = False

            if add:
                processed_entries.append(entry)
        return processed_entries

    def _preload_dataset(self):
        """Call this function to pre-calculate all point clouds, save them to the file (if not already), and load
        them to memory.
        """
        file_to_save = self.save_file_name
        if not os.path.isfile(file_to_save):
            logging.info(f"No existing dataset cache exists at {file_to_save}. Building one now.")

            # pre-calculate all data
            all_data = []
            for i in tqdm(range(self.__len__()), desc="Precomputing for single obj. dataset."):
                data = self._compute_data(i)
                if data is None:
                    logging.warning(f"Skipping {i}-th entry.")
                    continue
                all_data.append(data)

            pyld = {
                "dataset_name": self.dataset_name,
                "split": self.scene_ds.split,
                "object_id": self.object_id,
                "keypoints": self.keypoints,
                "data": all_data,
            }

            # load from file
            file = open(file_to_save, "wb")
            pickle.dump(pyld, file)
            file.close()

        # load from file
        file = open(file_to_save, "rb")
        pyld = pickle.load(file)
        file.close()

        assert self.dataset_name == pyld["dataset_name"]
        assert self.object_id == pyld["object_id"]
        assert self.scene_ds.split == pyld["split"]
        self.preloaded_data = pyld["data"]
        self.length = len(self.preloaded_data)

    def __len__(self):
        """Total number of frames in the dataset that contains the specified object"""
        return self.length

    def _compute_data(self, idx):
        """Return the specified instance of object detected."""
        if self.bop_detections is None:
            # gt detections
            data = get_obj_data_from_scene_frame_index(
                idx=idx,
                object_id=self.object_id,
                obj_keypoints=self.keypoints,
                obj_diameter=self.object_diameter,
                pc_size=self.pc_size,
                load_rgb_for_points=self.load_rgb_for_points,
                zero_center_pc=self.zero_center_pc,
                use_robust_centroid=self.use_robust_centroid,
                normalize_pc=self.normalize_pc,
                resample_invalid_pts=self.resample_invalid_pts,
                scene_frame_index=self.scene_frame_index,
                scene_ds=self.scene_ds,
            )
        else:
            # bop detections
            data = get_obj_data_from_bop_detections(
                idx=idx,
                object_id=self.object_id,
                obj_keypoints=self.keypoints,
                obj_diameter=self.object_diameter,
                pc_size=self.pc_size,
                load_rgb_for_points=self.load_rgb_for_points,
                zero_center_pc=self.zero_center_pc,
                use_robust_centroid=self.use_robust_centroid,
                normalize_pc=self.normalize_pc,
                resample_invalid_pts=self.resample_invalid_pts,
                scene_frame_index=self.scene_frame_index,
                scene_ds=self.scene_ds,
                bop_detections=self.bop_detections,
            )

        return data


class ObjectPoseDataset(ObjectPoseDatasetBase):
    """
    Dataset class to represent objects in the BOP dataset. During initialization, this class builds an index over all
    the BOP images objects, and allows for returns of a fixed batch size objects training data per iteration.
    This returns both the point cloud and RGB images for multi model training.

    Author: Jingnan Shi
    """

    def __init__(
        self,
        scene_ds,
        object_id,
        object_diameter=None,
        dataset_name=None,
        min_area=None,
        pc_size=1000,
        load_rgb_for_points=True,
        zero_center_pc=False,
        use_robust_centroid=False,
        resample_invalid_pts=False,
        normalize_pc=False,
        load_data_from_cache=False,
        cache_save_dir=None,
        preload_to_ram=False,
        bop_detections=None,
    ):
        """
        Args:
            object_diameter:
            scene_ds:
            object_id:
            dataset_name:
            min_area:
            all_model_names:
            pc_size:
            load_rgb_for_points:
            zero_center_pc: set to True if you want the returned object to be centered
            use_robust_centroid: set to True to use robust centroid
            resample_invalid_pts:
            normalize_pc: set to True to normalize the point cloud by object diameter; only when point cloud is centered
        """
        super().__init__(
            scene_ds=scene_ds,
            object_id=object_id,
            object_diameter=object_diameter,
            dataset_name=dataset_name,
            min_area=min_area,
            pc_size=pc_size,
            load_rgb_for_points=load_rgb_for_points,
            zero_center_pc=zero_center_pc,
            use_robust_centroid=use_robust_centroid,
            resample_invalid_pts=resample_invalid_pts,
            normalize_pc=normalize_pc,
            load_data_from_cache=load_data_from_cache,
            cache_save_dir=cache_save_dir,
            bop_detections=bop_detections,
        )
        self.preload_to_ram = preload_to_ram
        self.data_ram_cache = []
        if self.preload_to_ram:
            logging.info("Preloading all data for ObjectPoseDataset to RAM.")
            for idx in tqdm(range(self.__len__())):
                self.data_ram_cache.append(self.process_item(idx))

    def process_item(self, idx):
        """Return an entry from dataset, loading from cache and disk"""
        if self.load_data_from_cache:
            # note: python list access through [] is atomic
            data = self.preloaded_data[idx]
        else:
            data = self._compute_data(idx)

        if data is None:
            return None

        if self.bop_detections is None:
            rgb, mask, state = self.scene_ds[data["frame_idx_row"]]
            rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)
            rgb = rgb.permute(2, 0, 1).to(torch.uint8)
            filtered_mask = mask == data["obj"]["id_in_segm"]
            return rgb, filtered_mask, data
        else:
            rgb, _, state = self.scene_ds[data["frame_idx_row"]]
            rgb = to_torch_uint8(rgb)
            rgb = rgb.permute(2, 0, 1).to(torch.uint8)

            obj_info = self.scene_frame_index[idx]
            filtered_mask = obj_info["mask"].astype(np.uint8)
            filtered_mask = to_torch_uint8(filtered_mask)
            return rgb, filtered_mask, data

    def __getitem__(self, idx):
        if not self.preload_to_ram:
            return self.process_item(idx)
        else:
            return self.data_ram_cache[idx]


class ObjectPCPoseDataset(ObjectPoseDatasetBase):
    """
    Dataset class to represent objects in the BOP dataset. During initialization, this class builds an index over all
    the BOP images objects, and allows for returns of a fixed batch size objects training data per iteration.
    This dataset only returns the point cloud data.

    Author: Jingnan Shi
    """

    def __init__(
        self,
        scene_ds,
        object_id,
        object_diameter=None,
        dataset_name=None,
        min_area=None,
        pc_size=1000,
        load_rgb_for_points=True,
        zero_center_pc=False,
        use_robust_centroid=False,
        resample_invalid_pts=False,
        normalize_pc=False,
        load_data_from_cache=False,
        cache_save_dir=None,
        bop_detections=None,
    ):
        """
        Args:
            object_diameter:
            scene_ds:
            object_id:
            dataset_name:
            min_area:
            all_model_names:
            pc_size:
            load_rgb_for_points:
            zero_center_pc: set to True if you want the returned object to be centered
            use_robust_centroid: set to True to use robust centroid
            resample_invalid_pts:
            normalize_pc: set to True to normalize the point cloud by object diameter; only when point cloud is centered
        """
        super().__init__(
            scene_ds=scene_ds,
            object_id=object_id,
            object_diameter=object_diameter,
            dataset_name=dataset_name,
            min_area=min_area,
            pc_size=pc_size,
            load_rgb_for_points=load_rgb_for_points,
            zero_center_pc=zero_center_pc,
            use_robust_centroid=use_robust_centroid,
            resample_invalid_pts=resample_invalid_pts,
            normalize_pc=normalize_pc,
            load_data_from_cache=load_data_from_cache,
            cache_save_dir=cache_save_dir,
            bop_detections=bop_detections,
        )

    def __getitem__(self, idx):
        if self.load_data_from_cache:
            # note: python list access through [] is atomic
            data = self.preloaded_data[idx]
        else:
            data = self._compute_data(idx)

        if data is None:
            return None
        else:
            return (
                data["centered_normalized_pc"],
                data["centered_normalized_kpts"],
                data["cent_R_cad"],
                data["cent_t_cad"],
            )


class FramePoseDataset(torch.utils.data.Dataset):
    """
    Dataset class to represent objects in each frame of the BOP dataset.
    Acts as a wrapper around the BOPDataset class

    Author: Jingnan Shi
    """

    def __init__(
        self,
        scene_ds,
        min_area=None,
        min_valid_frac=0.1,
        all_model_names=None,
        pc_size=1000,
        load_rgb_for_points=True,
    ):
        """

        Args:
            scene_ds:
            min_area:
            min_valid_frac:
            all_model_names:
            pc_size: size of the point cloud to return in each training loop. If the segmented object contains fewer
                     points, the point cloud will be padded with zeros.
            load_rgb_for_points: set to True to also return the point-wise RGB data for the point clouds.
        """
        self.scene_ds = VisibilityWrapper(scene_ds)
        self.min_area = min_area
        self.min_valid_frac = min_valid_frac
        self.pc_size = pc_size
        self.all_model_names = all_model_names
        self.load_rgb_for_points = load_rgb_for_points

    def __len__(self):
        return len(self.scene_ds)

    def collate_fn(self, batch):
        """Custom collate_fn
        The batch data will be in the following format:
        - images:
        - depths:
        - masks:
        - K:
        - objs: list of all objects
        - frame_obj_index: dictionary (k=frame id in batch) of lists (indices to objs)
        - model_obj_index: ditionary (k=object name) of lists (indices to objs)
        - batched_pcs_per_model: dictionary
        """
        data = dict()
        for k in batch[0].keys():
            if k in ("images", "depths", "masks", "K"):
                # these fields are just tensors
                v = [x[k] for x in batch]
                v = torch.as_tensor(np.stack(v))
                data[k] = v
            elif k == "objects":
                # build all objs & frame2objs index
                all_objs = []
                frame2objs = dict()
                for i, x in enumerate(batch):
                    start_idx = len(all_objs)
                    end_idx = start_idx + len(x["objects"])
                    # insert frame ID into objects
                    for c_obj in x["objects"]:
                        c_obj["frame_id_in_batch"] = i
                    # save current frame's objects
                    all_objs.extend(x["objects"])
                    # generate the indices corresponding to the objs in the current frame
                    frame2objs[i] = list(range(start_idx, end_idx))

                # build model2objs index
                model2objs = {k: [] for k in self.all_model_names}
                for i, obj in enumerate(all_objs):
                    model2objs[obj["name"]].append(i)

                # build batched pcs per model
                model2batched_pcs = {k: [] for k in self.all_model_names}
                model2batched_gt_R = {k: [] for k in self.all_model_names}
                model2batched_gt_t = {k: [] for k in self.all_model_names}

                for model_name, obj_idxs in model2objs.items():
                    pcs = [all_objs[obj_idx]["point_cloud"] for obj_idx in obj_idxs]
                    Ts = [all_objs[obj_idx]["TCO"] for obj_idx in obj_idxs]
                    if len(pcs) >= 1:
                        model2batched_pcs[model_name] = torch.as_tensor(np.stack(pcs))
                        Ts = np.stack(Ts)
                        Rs = np.copy(Ts[:, :3, :3])
                        ts = np.copy(Ts[:, :3, -1])
                        model2batched_gt_R[model_name] = torch.as_tensor(Rs)
                        model2batched_gt_t[model_name] = torch.as_tensor(ts)

                data["objects"] = all_objs
                data["frame_to_objects_index"] = frame2objs
                data["model_to_objects_index"] = model2objs
                data["model_to_batched_pcs"] = model2batched_pcs
                data["model_to_batched_gt_R"] = model2batched_gt_R
                data["model_to_batched_gt_t"] = model2batched_gt_t

            elif k == "default_detected_objects":
                # TODO: Implement this
                # Follow the same logic as above
                raise NotImplementedError

        data = MultiPoseData(**data)
        return data

    def __getitem__(self, idx):
        rgb, mask, state = self.scene_ds[idx]

        rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)
        mask_uniqs = set(np.unique(mask))

        depths = torch.as_tensor(state["camera"]["depth"])

        K = np.asarray(state["camera"]["K"])
        assert K.shape[0] == 3
        assert K.shape[1] == 3

        objects_visible = []
        for obj in state["objects"]:
            add = False
            if obj["id_in_segm"] in mask_uniqs and np.all(np.array(obj["bbox"]) >= 0):
                add = True

            if add and self.min_area is not None:
                bbox = np.array(obj["bbox"])
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                if area >= self.min_area:
                    add = True
                else:
                    add = False

            # check point cloud validity:
            # if point cloud are all zero, then the masked out portion
            # of the depth cloud for the object does not have valid depth
            # measurements
            if add:
                filtered_mask = mask == obj["id_in_segm"]
                if self.load_rgb_for_points:
                    temp_pc = np.asarray(
                        depth_to_point_cloud_with_rgb_torch(
                            depth=depths,
                            rgb=rgb,
                            K=K,
                            mask=filtered_mask,
                            x_index=1,
                            y_index=0,
                            pc_size=self.pc_size,
                        )
                    )
                else:
                    temp_pc = np.asarray(
                        depth_to_point_cloud_torch(
                            depth=depths,
                            K=K,
                            mask=filtered_mask,
                            x_index=1,
                            y_index=0,
                            pc_size=self.pc_size,
                        )
                    )
                invalid_pts = np.sum(temp_pc == 0, axis=0)
                num_valid_pts = np.sum(invalid_pts != temp_pc.shape[0])
                if num_valid_pts / self.pc_size < self.min_valid_frac:
                    add = False
                else:
                    obj["point_cloud"] = temp_pc
                    obj["obj_mask"] = np.asarray(filtered_mask)

            if add:
                objects_visible.append(obj)
        if len(objects_visible) == 0:
            raise NoObjectError
        # assert len(objects_visible) > 0, idx

        rgb = torch.as_tensor(rgb).permute(2, 0, 1).to(torch.uint8)
        assert rgb.shape[0] == 3

        # preprocess objects data
        # 1. calculate TCO
        # 2. extract point clouds for the objects
        for i, obj in enumerate(objects_visible):
            # save TCO
            TWO = torch.as_tensor(obj["TWO"])
            TWC = torch.as_tensor(state["camera"]["TWC"])
            TCO = invert_T(TWC) @ TWO
            obj["TCO"] = np.asarray(TCO)

        # TODO: Get the default detected objects using IDX
        # Make sure to check it's actually getting the same frame from the same scene

        data = dict(
            images=np.asarray(rgb),
            depths=np.asarray(depths),
            masks=mask,
            K=np.asarray(state["camera"]["K"]),
            objects=objects_visible,
            # TODO: Populate this new field
            # default_detected_objects=None
        )
        return data


class PoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_ds,
        resize=(640, 480),
        min_area=None,
        rgb_augmentation=False,
        gray_augmentation=False,
        background_augmentation=False,
    ):

        self.scene_ds = VisibilityWrapper(scene_ds)

        self.resize_augmentation = CropResizeToAspectAugmentation(resize=resize)
        self.min_area = min_area

        self.background_augmentation = background_augmentation
        if background_augmentation:
            self.background_augmentations = VOCBackgroundAugmentation(
                voc_root=LOCAL_DATA_DIR / "VOCdevkit/VOC2012", p=0.3
            )

        self.rgb_augmentation = rgb_augmentation
        self.rgb_augmentations = [
            PillowBlur(p=0.4, factor_interval=(1, 3)),
            PillowSharpness(p=0.3, factor_interval=(0.0, 50.0)),
            PillowContrast(p=0.3, factor_interval=(0.2, 50.0)),
            PillowBrightness(p=0.5, factor_interval=(0.1, 6.0)),
            PillowColor(p=0.3, factor_interval=(0.0, 20.0)),
        ]
        if gray_augmentation:
            self.rgb_augmentations.append(GrayScale(p=0.5))

    def __len__(self):
        return len(self.scene_ds)

    def collate_fn(self, batch):
        data = dict()
        for k in batch[0].__annotations__:
            v = [getattr(x, k) for x in batch]
            if k in ("images", "depths", "masks", "bboxes", "TCO", "K"):
                v = torch.as_tensor(np.stack(v))
            data[k] = v
        data = PoseData(**data)
        return data

    def get_data(self, idx):
        rgb, mask, state = self.scene_ds[idx]

        rgb, mask, state = self.resize_augmentation(rgb, mask, state)

        if self.background_augmentation:
            rgb, mask, state = self.background_augmentations(rgb, mask, state)

        if self.rgb_augmentation and random.random() < 0.8:
            for augmentation in self.rgb_augmentations:
                rgb, mask, state = augmentation(rgb, mask, state)

        rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)
        mask_uniqs = set(np.unique(mask))
        objects_visible = []
        for obj in state["objects"]:
            add = False
            if obj["id_in_segm"] in mask_uniqs and np.all(np.array(obj["bbox"]) >= 0):
                add = True

            if add and self.min_area is not None:
                bbox = np.array(obj["bbox"])
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                if area >= self.min_area:
                    add = True
                else:
                    add = False
            if add:
                objects_visible.append(obj)
        if len(objects_visible) == 0:
            raise NoObjectError
        # assert len(objects_visible) > 0, idx

        rgb = torch.as_tensor(rgb).permute(2, 0, 1).to(torch.uint8)
        assert rgb.shape[0] == 3

        # current the loader samples random obj from an arbitrary frame as output
        obj = random.sample(objects_visible, k=1)[0]
        TWO = torch.as_tensor(obj["TWO"])
        TWC = torch.as_tensor(state["camera"]["TWC"])
        TCO = invert_T(TWC) @ TWO

        depths = torch.as_tensor(state["camera"]["depth"])

        data = PoseData(
            images=np.asarray(rgb),
            depths=np.asarray(depths),
            bboxes=np.asarray(obj["bbox"]),
            TCO=np.asarray(TCO),
            K=np.asarray(state["camera"]["K"]),
            objects=obj,
            masks=mask,
        )
        return data

    def __getitem__(self, index):
        try_index = index
        valid = False
        n_attempts = 0
        while not valid:
            if n_attempts > 10:
                raise ValueError("Cannot find valid image in the dataset")
            try:
                data = self.get_data(try_index)
                valid = True
            except NoObjectError:
                try_index = random.randint(0, len(self.scene_ds) - 1)
                n_attempts += 1
        return data
