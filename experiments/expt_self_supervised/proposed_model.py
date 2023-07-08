"""
This writes a proposed model for expt_registration

"""
import logging
import os
import torch
import torch.nn as nn
import yaml

from casper3d.certifiability import MultiModelCertifier
from casper3d.icp import ICP, ICP_o3d
from casper3d.keypoint_corrector import kp_corrector_reg
from casper3d.keypoint_detector import RegressionKeypoints
from casper3d.point_set_registration import PointSetRegistration
from casper3d.robust_centroid import robust_centroid_gnc
from expt_self_supervised.cfg import CosyposeCfg
from cosypose.integrated.pose_predictor_with_grad import CoarseRefinePosePredictor
from cosypose.models.mask_rcnn import DetectorMaskRCNN
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.training.pose_models_cfg import create_model_pose
from datasets import ycbv, tless, bop_constants
from utils.ddn.node import ParamDeclarativeFunction
from utils.torch_utils import num_trainable_params
import utils.visualization_utils as vutils
import utils.math_utils as mutils


def load_certifier(cad_models, renderer, cfg):
    """Create the certifier model"""
    certifier = MultiModelCertifier(
        epsilon_pc_db=cfg["certifier"]["epsilon_pc"],
        epsilon_kp_db=cfg["certifier"]["epsilon_kp"],
        epsilon_mask_db=cfg["certifier"]["epsilon_mask"],
        clamp_thres_db=cfg["certifier"]["clamp_thres"],
        cad_models_db=cad_models,
        renderer=renderer,
        cfg=cfg,
    )
    return certifier


def load_segmentation_model(device, cfg):
    num_models = bop_constants.BOP_CATEGORIES[cfg["dataset"]]
    # we need to add background to the classes
    n_classes = num_models + 1
    model = DetectorMaskRCNN(
        input_resize=cfg["training"]["input_resize"],
        n_classes=n_classes,
        backbone_str=cfg["maskrcnn"]["backbone_str"],
        anchor_sizes=cfg["maskrcnn"]["anchor_sizes"],
    ).to(device=device)
    path = cfg["maskrcnn"]["pretrained_weights_path"]
    logging.info(f"Loading MaskRCNN model from {path}")
    save = torch.load(path)
    state_dict = save["state_dict"]
    model.load_state_dict(state_dict)
    model.train(False)
    if cfg["maskrcnn"]["freeze_weights"]:
        logging.info("Freezing all params' weights in the MaskRCNN model.")
        for param in model.parameters():
            param.requires_grad = False
    return model


def load_batch_renderer(cfg_dict):
    """Load batch renderer"""
    batch_renderer = BulletBatchRenderer(
        object_set=cfg_dict["urdf_ds_name"], n_workers=cfg_dict["cosypose_coarse_refine"]["n_rendering_workers"]
    )
    return batch_renderer


def load_renderer(cfg):
    """Load single scene renderer"""
    renderer = BulletSceneRenderer(urdf_ds=cfg["urdf_ds_name"], preload_cache=True)
    return renderer


def load_cosypose_model(mesh_db_batched, cfg_dict):
    """Create cosypose model and load pretrained cosypose weights"""
    cfg = CosyposeCfg()
    cfg.load_from_dict(cfg_dict)

    # convert dictionary configuration to cfg dataclass for cosypose
    model = CosyPoseRegressionModel(mesh_db=mesh_db_batched, cfg=cfg)

    # load weights
    ckpt = torch.load(cfg.cosypose_pretrained_weights_path)
    ckpt_weights = ckpt["state_dict"]
    model.load_cosypose_state_dict(ckpt_weights)

    return model


def load_cosypose_coarse_refine_model(
    batch_renderer,
    mesh_db_batched,
    model_id=None,
    cad_models=None,
    model_keypoints=None,
    object_diameter=None,
    cfg_dict=None,
):
    """Create coarse + refine Cosypose Model"""

    def load_cfg_and_ckpt(config_path, ckpt_path):
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
        ckpt = torch.load(ckpt_path)
        ckpt = ckpt["state_dict"]
        return cfg, ckpt

    coarse_cfg, coarse_ckpt = load_cfg_and_ckpt(
        cfg_dict["cosypose_coarse_refine"]["coarse"]["pretrained_model_config_path"],
        cfg_dict["cosypose_coarse_refine"]["coarse"]["pretrained_weights_path"],
    )
    if cfg_dict["cosypose_coarse_refine"]["refiner"]["n_iterations"] > 0:
        refiner_cfg, refiner_ckpt = load_cfg_and_ckpt(
            cfg_dict["cosypose_coarse_refine"]["refiner"]["pretrained_model_config_path"],
            cfg_dict["cosypose_coarse_refine"]["refiner"]["pretrained_weights_path"],
        )
    else:
        refiner_cfg, refiner_ckpt = None, None

    # create our wrapper for the CoarseRefinePredictor
    model = CosyPoseCoarseRefineModel(
        mesh_db=mesh_db_batched,
        batch_renderer=batch_renderer,
        urdf_ds_name=cfg_dict["urdf_ds_name"],
        n_rendering_workers=cfg_dict["cosypose_coarse_refine"]["n_rendering_workers"],
        coarse_cfg=coarse_cfg,
        coarse_ckpt=coarse_ckpt,
        refiner_cfg=refiner_cfg,
        refiner_ckpt=refiner_ckpt,
        use_corrector=cfg_dict["cosypose_coarse_refine"]["use_corrector"],
        model_id=model_id,
        model_keypoints=model_keypoints,
        cad_models=cad_models,
        object_diameter=object_diameter,
        corrector_config=cfg_dict["cosypose_coarse_refine"]["corrector"],
        cfg_dict=cfg_dict,
    )
    return model


def load_c3po_cad_models(model_id, device, output_unit, cfg):
    """Loading a specific object's CAD model as sampled points.
    In BOP datasets, all models have units in mm. Our annotated keypoints also have units in mm.
    """
    if cfg["dataset"] == "ycbv":
        mesh_root_folder_path = os.path.join(cfg["bop_ds_dir"], "ycbv", "models")
        annotations_folder_path = os.path.join(cfg["bop_ds_dir"], "ycbv", "annotations")
        obj_ds = ycbv.SE3PointCloud(
            model_id=model_id,
            num_of_points=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
            mesh_root_folder_path=mesh_root_folder_path,
            annotations_folder_path=annotations_folder_path,
        )
    elif cfg["dataset"] == "tless":
        mesh_root_folder_path = os.path.join(cfg["bop_ds_dir"], "tless", "models")
        annotations_folder_path = os.path.join(cfg["bop_ds_dir"], "tless", "annotations")
        obj_ds = tless.SE3PointCloud(
            model_id=model_id,
            num_of_points=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
            mesh_root_folder_path=mesh_root_folder_path,
            annotations_folder_path=annotations_folder_path,
        )
    else:
        raise NotImplementedError

    # point cloud sampled from zero-centered, normalized mesh
    cad_models = obj_ds._get_cad_models().to(torch.float).to(device=device)
    # zero-centered, normalized keypoints
    model_keypoints = obj_ds._get_model_keypoints().to(torch.float).to(device=device)

    if output_unit == "mm":
        scale = 1
    elif output_unit == "m":
        scale = 1e-3
    else:
        raise ValueError

    # diameter of the object
    obj_diameter = obj_ds._get_diameter() * scale
    # get the original CAD mesh
    original_cad_model = obj_ds._get_original_cad_models().to(device=device) * scale
    # get the original keypoints
    original_model_keypoints = obj_ds._get_original_model_keypoints().to(device=device) * scale

    return cad_models, model_keypoints, original_cad_model, original_model_keypoints, obj_diameter


def load_c3po_cad_meshes(model_id, device, output_unit, cfg):
    """Loading a specific object's CAD model as sampled points.
    In BOP datasets, all models have units in mm. Our annotated keypoints also have units in mm.
    """
    if cfg["dataset"] == "ycbv":
        mesh_root_folder_path = os.path.join(cfg["bop_ds_dir"], "ycbv", "models")
        annotations_folder_path = os.path.join(cfg["bop_ds_dir"], "ycbv", "annotations")
        obj_ds = ycbv.SE3PointCloud(
            model_id=model_id,
            num_of_points=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
            mesh_root_folder_path=mesh_root_folder_path,
            annotations_folder_path=annotations_folder_path,
        )
    elif cfg["dataset"] == "tless":
        mesh_root_folder_path = os.path.join(cfg["bop_ds_dir"], "tless", "models")
        annotations_folder_path = os.path.join(cfg["bop_ds_dir"], "tless", "annotations")
        obj_ds = tless.SE3PointCloud(
            model_id=model_id,
            num_of_points=cfg["c3po"]["point_transformer"]["num_of_points_to_sample"],
            mesh_root_folder_path=mesh_root_folder_path,
            annotations_folder_path=annotations_folder_path,
        )
    else:
        raise NotImplementedError

    if output_unit == "mm":
        scale = 1
    elif output_unit == "m":
        scale = 1e-3
    else:
        raise ValueError

    # get the original CAD mesh
    original_cad_mesh = obj_ds._get_original_cad_mesh().scale(scale, center=[0,0,0])
    original_textured_mesh = obj_ds.original_model_textured_mesh
    original_textured_mesh.vertices *= scale

    return original_cad_mesh, original_textured_mesh


def load_all_cad_models(device, cfg):
    """Load all CAD models into a dictionary"""
    dataset = cfg["dataset"]
    num_models = bop_constants.BOP_CATEGORIES[dataset]
    model_ids = list(bop_constants.BOP_MODEL_INDICES[dataset].keys())
    assert len(model_ids) == num_models
    all_cad_models = {}
    for model_id in model_ids:
        # note: cosypose uses meters, so here we make sure we are consistent
        cad_models, model_keypoints, original_cad_model, original_model_keypoints, obj_diameter = load_c3po_cad_models(
            model_id, device, output_unit="m", cfg=cfg
        )
        all_cad_models[model_id] = dict(
            cad_model_pc=cad_models,
            cad_model_keypoints=model_keypoints,
            original_cad_model=original_cad_model,
            original_model_keypoints=original_model_keypoints,
            cad_model_diameter=obj_diameter,
        )
    return all_cad_models


def load_multi_obj_c3po_model(device, cfg):
    """Load multiple C3PO models for objects"""
    dataset = cfg["dataset"]
    num_models = bop_constants.BOP_CATEGORIES[dataset]
    model_ids = list(bop_constants.BOP_MODEL_INDICES[dataset].keys())
    assert len(model_ids) == num_models
    models = {}
    for model_id in model_ids:
        _, _, original_cad_model, original_model_keypoints, obj_diameter = load_c3po_cad_models(
            model_id, device, output_unit="m", cfg=cfg
        )
        model = load_c3po_model(
            model_id=model_id,
            cad_models=original_cad_model,
            model_keypoints=original_model_keypoints,
            obj_diameter=obj_diameter,
            device=device,
            cfg=cfg,
        )
        models[model_id] = model
    c3po_multi_model = MultiPointsRegressionModel(c3po_modules=models)
    return c3po_multi_model


def load_c3po_model(model_id, cad_models, model_keypoints, object_diameter, device, cfg):
    """Create C3PO model and load pretrained C3PO weights"""
    # create datasets for keypoints and cad models
    logging.info(f"Creating C3PO for {model_id}.")
    # create model
    c3po_cfg = cfg["c3po"]
    model = PointsRegressionModel(
        model_id=model_id,
        model_keypoints=model_keypoints,
        cad_models=cad_models,
        object_diameter=object_diameter,
        keypoint_detector=c3po_cfg["detector_type"],
        correction_flag=c3po_cfg["use_corrector"],
        corrector_config=c3po_cfg["corrector"],
        need_predicted_keypoints=True,
        use_icp=c3po_cfg["use_icp"],
        zero_center_input=c3po_cfg["zero_center_input"],
        use_robust_centroid=c3po_cfg["use_robust_centroid"],
        input_pc_normalized=cfg["training"]["normalize_pc"],
        icp_config=c3po_cfg["icp"],
        c3po_config=c3po_cfg,
    ).to(device)

    # load weights
    if c3po_cfg["load_pretrained_weights"]:
        if "pretrained_weights_path" in c3po_cfg.keys():
            weights_path = c3po_cfg["pretrained_weights_path"]
        elif "pretrained_weights_dir" in c3po_cfg.keys():
            weights_path = os.path.join(c3po_cfg["pretrained_weights_dir"], cfg["dataset"], model_id)
            weights_path = os.path.join(
                weights_path, f"_{cfg['dataset']}_best_supervised_kp_{c3po_cfg['detector_type']}.pth"
            )
        else:
            raise FileNotFoundError("Failed to find a weights path.")
        logging.info(f"Loading pretrained weights for C3PO at {weights_path}.")
        if not os.path.isfile(weights_path):
            logging.error(f"Weights does not exist at {weights_path}")
            raise FileNotFoundError(f"Weights does not exist at {weights_path}")
        state_dict = torch.load(weights_path)
        if "state_dict" in state_dict.keys():
            model.load_state_dict(state_dict["state_dict"])
        else:
            model.load_state_dict(state_dict)

    return model


def load_multi_model(batch_renderer, meshdb_batched, device, cfg):
    """Load multi model for training

    Args:
        model_id: ID of the model, only used by C3PO
        meshdb_batched:
        device:
        cfg:
    """
    models_to_use = list(set(cfg["models_to_use"]))
    submodules = {}
    logging.info(f"Loading {models_to_use}")
    if "c3po" in models_to_use:
        # note: only C3PO uses the model_id parameter
        raise NotImplementedError
        # submodules["c3po"] = load_c3po_model(model_id, device, cfg)
        # logging.info(f"Number of trainable parameters - c3po: {num_trainable_params(submodules['c3po'])}")

    if "c3po_multi" in models_to_use:
        submodules["c3po_multi"] = load_multi_obj_c3po_model(device, cfg)
        logging.info(f"Number of trainable parameters - c3po_multi: {num_trainable_params(submodules['c3po_multi'])}")

    if "cosypose" in models_to_use:
        submodules["cosypose"] = load_cosypose_model(meshdb_batched, cfg)
        logging.info(f"Number of trainable parameters - cosypose: {num_trainable_params(submodules['cosypose'])}")

    if "cosypose_coarse_refine" in models_to_use:
        submodules["cosypose_coarse_refine"] = load_cosypose_coarse_refine_model(
            batch_renderer=batch_renderer, mesh_db_batched=meshdb_batched, cfg_dict=cfg
        )

    model = MultiModel(pipeline_modules=submodules).to(device)
    return model


class PointsRegressionModel(nn.Module):
    """
    Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation

    Returns:
        predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
        predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
    """

    def __init__(
        self,
        model_id=None,
        model_keypoints=None,
        cad_models=None,
        object_diameter=1.0,
        keypoint_detector=None,
        local_max_pooling=False,
        correction_flag=False,
        corrector_config=None,
        need_predicted_keypoints=False,
        use_icp=False,
        zero_center_input=True,
        use_robust_centroid=False,
        input_pc_normalized=False,
        icp_config=None,
        c3po_config=None,
    ):
        """

        Args:
            model_id:
            model_keypoints: torch.tensor of shape (K, 3, N)
            cad_models: torch.tensor of shape (K, 3, n)
            object_diameter: Input will be divided by this after zero center. Typically, it equals to object's
                                  original diameter.
            keypoint_detector: torch.nn.Module : detects N keypoints for any sized point cloud input
                                                 should take input : torch.tensor of shape (B, 3, m)
                                                 should output     : torch.tensor of shape (B, 3, N)
            local_max_pooling:
            correction_flag:
            corrector_config:
            need_predicted_keypoints:
            input_pc_normalized: set to True if the input PC has been normalized. Necessary as we guarantee the
            returned predicted keypoints are in the original scale, and that we use the original CAD frames
            for registration/corrector
            use_icp:
            icp_config:
        """
        super().__init__()

        # Parameters
        self.model_id = model_id
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.max_intra_kpt_dist, self.min_intra_kpt_dist = mutils.get_max_min_intra_pts_dists(model_keypoints)
        self.object_diameter = object_diameter
        self.zero_center_input = zero_center_input
        self.use_robust_centroid = use_robust_centroid
        self.input_pc_normalized = False
        self.device_ = self.cad_models.device
        self.input_pc_normalized = input_pc_normalized

        if self.input_pc_normalized:
            logging.info(f"Normalization done in dataloader. Disable C3PO normalization.")
            self.normalization_factor = 1.0
        else:
            self.normalization_factor = object_diameter

        # depth
        self.input_type = "d"

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)
        self.local_max_pooling = local_max_pooling
        self.correction_flag = correction_flag
        self.use_icp = use_icp
        self.need_predicted_keypoints = need_predicted_keypoints
        self.c3po_config = c3po_config

        # Keypoint Detector
        if keypoint_detector == None:
            logging.info("Using pointnet")
            self.keypoint_detector = RegressionKeypoints(N=self.N, method="pointnet")

        elif keypoint_detector == "pointnet":
            logging.info("Using pointnet")
            self.keypoint_detector = RegressionKeypoints(N=self.N, method="pointnet")

        elif keypoint_detector == "point_transformer":
            logging.info(f"Using point_transformer with norm-type={c3po_config['point_transformer']['norm_type']}")
            self.keypoint_detector = RegressionKeypoints(
                N=self.N, method="point_transformer", **c3po_config["point_transformer"]
            )

        elif keypoint_detector == "point_transformer_dense":
            logging.info(
                f"Using point_transformer_dense with norm-type={c3po_config['point_transformer']['norm_type']}"
            )
            self.keypoint_detector = RegressionKeypoints(
                N=self.N, method="point_transformer_dense", **c3po_config["point_transformer_dense"]
            )

        else:
            raise NotImplementedError

        # Registration
        self.point_set_registration = PointSetRegistration(source_points=self.model_keypoints / self.object_diameter)

        # Corrector
        # self.corrector = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
        if correction_flag:
            if corrector_config is None:
                corrector_node = kp_corrector_reg(
                    cad_models=self.cad_models / self.object_diameter,
                    model_keypoints=self.model_keypoints / self.object_diameter,
                    model_id=model_id,
                )
            else:
                corrector_node = kp_corrector_reg(
                    cad_models=self.cad_models / self.object_diameter,
                    model_keypoints=self.model_keypoints / self.object_diameter,
                    model_id=model_id,
                    max_solve_iters=corrector_config["max_solve_iters"],
                    solve_tol=corrector_config["solve_tol"],
                    algo=corrector_config["solve_alg"],
                    chamfer_max_loss=corrector_config["chamfer_loss_use_max"],
                    chamfer_clamped=corrector_config["clamp_chamfer_loss"],
                    chamfer_clamp_thres=corrector_config["chamfer_loss_clamp_thres"],
                )

            self.corrector = ParamDeclarativeFunction(problem=corrector_node)

        if use_icp:
            if icp_config is None:
                raise ValueError("No ICP config provided.")
            else:
                # non-differentiable ICP w/ open3d
                self.icp = ICP_o3d(corr_threshold=icp_config["corr_threshold"],
                                   iters_max=icp_config["iters_max"])

                # Old differentiable version
                #self.icp = ICP(
                #    iters_max=icp_config["iters_max"],
                #    mse_threshold=icp_config["mse_threshold"],
                #    corr_threshold=icp_config["corr_threshold"],
                #    solver_type=icp_config["solver_type"],
                #    corr_type=icp_config["corr_type"],
                #    dist_type=icp_config["dist_type"],
                #    differentiable=True,
                #)

    def forward(self, model_id=None, input_point_cloud=None):
        """Regress and register keypoints to input point cloud.
        The process:
        1. Depending on whether the input point cloud is normalized (controlled at initialization), we will
           center and normalize the input point cloud.
        2. We perform regression on the normalized point cloud. If specified to use the corrector, we will also
           use the corrector.
        3. Transform the normalized keypoints to the original scale. Calculate the poses required to transform
           the original CAD keypoints to the input frame (if we performed center operation, we will de-center).

        Args:
            model_id: id of the object
            input_point_cloud: (B, 3, m) point cloud of the detected object. Potentially normalized.

        Returns:
            pc_pred, kp_pred, R, t, correction, (model_kp_pred)
        """
        if model_id != self.model_id:
            logging.warning(f"Model ID mismatch. Expected: {self.model_id}, Actual: {model_id}")
            return None

        if input_point_cloud.shape[0] == 0:
            return None

        batch_size, _, m = input_point_cloud.shape
        device_ = input_point_cloud.device

        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        invalid_pts_mask = num_zero_pts == input_point_cloud.shape[1]
        num_zero_pts = torch.sum(invalid_pts_mask, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        # normalization stage
        # zero center the input point clouds if requested
        pc_centered = input_point_cloud.clone()
        with torch.no_grad():
            # note that in the case with rgb feature per point:
            # pc_centered will just be the 3D points excluding the RGB values
            # as RGB won't be affected by centering
            center = torch.zeros((input_point_cloud.shape[0], 3, 1)).to(device_)
            if self.zero_center_input:
                if self.use_robust_centroid:
                    # the clamp threshold should be roughly object diameter / 2
                    # which should be equal to the normalization factor / 2
                    ro_cent_pyld = robust_centroid_gnc(
                        input_point_cloud=torch.masked_select(
                            input_point_cloud, torch.logical_not(invalid_pts_mask).unsqueeze(1)
                        ).reshape(batch_size, input_point_cloud.shape[1], -1)[:, :3, :],
                        cost_type=self.c3po_config["robust_centroid"]["algorithm"],
                        clamp_thres=self.normalization_factor / 1.9,
                        cost_abs_stop_th=self.c3po_config["robust_centroid"]["abs_cost_termination_threshold"],
                        cost_diff_stop_th=self.c3po_config["robust_centroid"]["rel_cost_termination_threshold"],
                    )
                    # outlier mask shape: B, 1, N
                    est_outlier_mask = ro_cent_pyld["weights"] < 0.5
                    center = ro_cent_pyld["robust_centroid"]
                else:
                    center = torch.sum(pc_centered[:, :3, :], dim=-1) / num_nonzero_pts
                    center = center.unsqueeze(-1)
                    est_outlier_mask = torch.zeros(batch_size, 1, m).to(device_)

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
                pc_centered[:, :3, :] = pc_centered[:, :3, :] - center.expand(batch_size, 3, m)

                # zero out
                pts_to_zero_mask = torch.logical_or(invalid_pts_mask.unsqueeze(1), est_outlier_mask)
                for b in range(pts_to_zero_mask.shape[0]):
                    logging.warning("Resampling in C3PO model. Potentially slow; use dataloader instead.")
                    num_invalid_pts = torch.sum(pts_to_zero_mask[b, ...])
                    if num_invalid_pts > 0:
                        valid_pts_mask = torch.logical_not(pts_to_zero_mask[b, ...]).squeeze()
                        valid_pts = pc_centered[b, :3, valid_pts_mask]

                        # sample the # of invalid pts from the valid pts
                        sampled_valid_inds = torch.randint(
                            low=0, high=valid_pts.shape[-1], size=(num_invalid_pts.item(),)
                        ).to(pc_centered.device)

                        # replace invalid pts
                        pc_centered[b, :3, pts_to_zero_mask[b, ...].squeeze()] = torch.index_select(
                            valid_pts, dim=1, index=sampled_valid_inds
                        )

        # normalize the input point cloud
        # at this point, pc centered should be normalized & centered
        pc_centered[:, :3, :] = pc_centered[:, :3, :] / self.normalization_factor

        # run forward pass on keypoint detector
        # these keypoints are on the normalized & zero centered points
        kp_pred = self.keypoint_detector(pc_centered)

        correction = None
        if not self.correction_flag:
            # registration performed on normalized keypoints
            R, t = self.point_set_registration.forward(kp_pred)
        else:
            # corrector + registration performed on normalized keypoint and point clouds
            correction = self.corrector.forward(kp_pred, pc_centered[:, :3, :])
            kp_pred = kp_pred + correction
            R, t = self.point_set_registration.forward(kp_pred)

        # ICP refine
        # note: not differentiable
        # R_delta: B, 3, 3
        # t_delta: B, 3, 1
        if self.use_icp:
            logging.info("Starting ICP (non differentiable).")
            R, t, icp_mse = self.icp.forward(
                pc_src=self.cad_models / self.object_diameter,
                pc_dest=pc_centered[:, :3, :],
                init_R=R,
                init_t=t,
            )

        # de-normalization stage
        t *= self.object_diameter
        kp_pred *= self.object_diameter
        # decenter (if we have centered the input)
        if self.zero_center_input:
            kp_pred += center
            t += center

        pc_pred = R @ self.cad_models + t

        if not self.need_predicted_keypoints:
            return pc_pred, kp_pred, R, t, correction
        else:
            model_kp_pred = R @ self.model_keypoints + t
            return pc_pred, kp_pred, R, t, correction, model_kp_pred


class MultiPointsRegressionModel(nn.Module):
    """A wrapper for loading multiple C3PO models to handle multiple objects"""

    def __init__(self, c3po_modules):
        """Initialization

        Args:
            configs: A list of dictionaries containing keyword arguments to PointsRegressionModel
        """
        super().__init__()

        # modules saved indexed by their model ids
        self.obj_reg_modules = torch.nn.ModuleDict(c3po_modules)

    def forward(self, object_batched_pcs=None):
        outputs = dict()
        for object_id, batched_pc in object_batched_pcs.items():
            outputs[object_id] = self.obj_reg_modules[object_id].forward(
                model_id=object_id, input_point_cloud=batched_pc
            )
        return outputs


class CosyPoseRegressionModel(nn.Module):
    """A wrapper for Cosypose using RGB inputs"""

    def __init__(self, mesh_db, cfg):
        super().__init__()

        # initialize a renderer
        renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=cfg.n_rendering_workers)

        # create pose model
        self.input_type = "rgb"
        self.mesh_db = mesh_db
        self.pose_model = create_model_pose(cfg=cfg, renderer=renderer, mesh_db=mesh_db).cuda()
        self.pose_model.debug = cfg.debug

    def load_cosypose_state_dict(self, ckpt):
        self.pose_model.load_state_dict(ckpt)

    def forward(self, images=None, K=None, labels=None, TCO=None, n_iterations=1):
        return self.pose_model.forward(images, K, labels, TCO, n_iterations=n_iterations)


class CosyPoseCoarseRefineModel(nn.Module):
    """A wrapper for Cosypose using RGB inputs"""

    def __init__(
        self,
        mesh_db,
        urdf_ds_name,
        n_rendering_workers,
        coarse_cfg,
        refiner_cfg,
        coarse_ckpt=None,
        refiner_ckpt=None,
        batch_renderer=None,
        use_corrector=False,
        model_id=None,
        model_keypoints=None,
        cad_models=None,
        object_diameter=None,
        corrector_config=None,
        cfg_dict=None,
    ):
        super().__init__()

        # initialize a renderer
        if batch_renderer is None:
            batch_renderer = BulletBatchRenderer(object_set=urdf_ds_name, n_workers=n_rendering_workers)
        self.batch_renderer = batch_renderer

        # create pose model
        self.use_corrector = use_corrector
        self.input_type = "rgb"
        self.mesh_db = mesh_db
        self.cfg = cfg_dict

        # create model
        self.coarse_pose_model = create_model_pose(cfg=coarse_cfg, renderer=self.batch_renderer, mesh_db=mesh_db).cuda()
        if refiner_cfg is not None:
            self.refine_pose_model = create_model_pose(
                cfg=refiner_cfg, renderer=self.batch_renderer, mesh_db=mesh_db
            ).cuda()
        else:
            self.refine_pose_model = None

        if coarse_ckpt is not None and refiner_ckpt is not None:
            self.load_cosypose_state_dict(coarse_ckpt, refiner_ckpt)

        # coarse refine predictor
        self.model = CoarseRefinePosePredictor(
            coarse_model=self.coarse_pose_model, refiner_model=self.refine_pose_model, cfg=cfg_dict
        )

        self.corrector, self.model_keypoints, self.cad_models, self.object_diameter = None, None, None, None
        if self.use_corrector:
            logging.info("Cosypose CoarseRefineModel uses corrector.")
            if None in (model_id, object_diameter, model_keypoints, cad_models):
                raise ValueError("Corrector initialization failed in Cosypose CoarseRefineModel.")

            # save the parameters
            self.model_keypoints = model_keypoints
            self.cad_models = cad_models
            self.object_diameter = object_diameter

            # for registration after correction
            self.point_set_registration = PointSetRegistration(
                source_points=self.model_keypoints / self.object_diameter
            )

            # corrector (operates in normalized frame)
            corrector_node = kp_corrector_reg(
                cad_models=self.cad_models / self.object_diameter,
                model_keypoints=self.model_keypoints / self.object_diameter,
                model_id=model_id,
                max_solve_iters=corrector_config["max_solve_iters"],
                solve_tol=corrector_config["solve_tol"],
                algo=corrector_config["solve_alg"],
                chamfer_max_loss=corrector_config["chamfer_loss_use_max"],
                chamfer_clamped=corrector_config["clamp_chamfer_loss"],
                chamfer_clamp_thres=corrector_config["chamfer_loss_clamp_thres"],
                log_loss_traj=False,
            )
            self.corrector = ParamDeclarativeFunction(problem=corrector_node)

    def load_cosypose_state_dict(self, coarse_ckpt, refine_ckpt):
        self.coarse_pose_model.load_state_dict(coarse_ckpt)
        self.refine_pose_model.load_state_dict(refine_ckpt)

    def forward(
        self,
        images=None,
        K=None,
        detections=None,
        TCO=None,
        n_coarse_iterations=1,
        n_refiner_iterations=1,
        pc_centered_normalized=None,
        pc_centroids=None,
    ):
        """Run Cosypose CoarseRefineModel, optionally runs a corrector.

        Args:
            images:
            K:
            detections:
            TCO:
            n_coarse_iterations:
            n_refiner_iterations:
            pc_centered_normalized: point clouds used for the corrector. Note that this should be in the original camera frame.

        Returns:

        """
        pyld = self.model.forward(
            images,
            K,
            detections=detections,
            n_coarse_iterations=n_coarse_iterations,
            n_refiner_iterations=n_refiner_iterations,
        )
        if not self.use_corrector:
            return pyld
        else:
            # note: corrector operates in the normalized frame (scaled down by object diameter)
            # we also center the predicted keypoints to ensure large translation errors won't overwhelm the clamped
            # chamfer loss.
            kp_pred = pyld[0].poses[:, :3, :3] @ self.model_keypoints + pyld[0].poses[:, :3, -1][..., None]
            kp_pred_centroids = torch.mean(kp_pred, dim=-1, keepdim=True)
            kp_pred_centered_normalized = (kp_pred - kp_pred_centroids) / self.object_diameter

            # test visualization
            # vutils.visualize_gt_and_pred_keypoints(pc_centered_normalized[:, :3, :], kp_pred_centered_normalized)

            correction = self.corrector.forward(kp_pred_centered_normalized, pc_centered_normalized[:, :3, :])
            kp_pred_centered_normalized = kp_pred_centered_normalized + correction

            # test visualization
            # vutils.visualize_gt_and_pred_keypoints(pc_centered_normalized[:, :3, :], kp_pred_centered_normalized)

            # estimate R, t from normalized CAD frame to kp_pred_centered_normalized
            # note that t will include the effect of kp_pred_centroids
            R, t_normalized = self.point_set_registration.forward(kp_pred_centered_normalized)

            # test visualization
            # vutils.visualize_gt_and_pred_keypoints(
            #    pc_centered_normalized[:, :3, :],
            #    R @ self.model_keypoints / self.object_diameter + t_normalized.squeeze(-1)[..., None],
            # )

            # we add the previously estimated point cloud centroid back so the translation is from the CAD frame
            # to the original camera frame
            t = t_normalized * self.object_diameter + pc_centroids

            # test visualization
            # vutils.visualize_gt_and_pred_keypoints(pc_centered_normalized[:, :3, :], kp_pred_centered_normalized)

            ## visualization w & w/o corrector
            # vutils.visualize_gt_and_pred_keypoints(
            #    pc_centered_normalized[:, :3, :],
            #    (pyld[0].poses[:, :3, :3] @ self.model_keypoints + pyld[0].poses[:, :3, -1][..., None] - pc_centroids)
            #    / self.object_diameter,
            #    kp_pred=kp_pred_centered_normalized
            # )

            pyld[0].poses[:, :3, :3] = R
            pyld[0].poses[:, :3, -1] = t.squeeze(-1)
            pyld[1]["correction"] = correction
            return pyld


class MultiModel(nn.Module):
    """Multi-model Class

    Self-supervised training with multiple learned models
    """

    def __init__(self, pipeline_modules):
        """Initialize the MultiModel using a dictionary.
        Internally, we use a dictionary to index the different models.
        Doing so allows us to easilly swap different models for testing.
        For example, if we want to use a DataLoader that returns only a specific
        model_id, we just need to load an individual C3PO model.

        Args:
            sub_modules: A list of torch.nn.Module
        """
        super().__init__()

        # nn modules and their respective input types
        self.pipeline_modules = torch.nn.ModuleDict(pipeline_modules)
        self.available_models = set(pipeline_modules.keys())

    def forward(self, **kwargs):
        """Forward call to pass inputs to the child models

        Args:
            kwargs: a dictionary of dictionaries; key = module name,
             value = corresponding keyword arguments

        Returns:

        """
        outputs = dict()
        for module_name, module_args in kwargs.items():
            output = self.pipeline_modules[module_name].forward(**module_args)
            outputs[module_name] = output
        return outputs

    def save_individual_module(self, weights_paths):
        """Save module weights individually"""
        for m, weights_path in zip(self.pipeline_modules, weights_paths):
            torch.save(m.state_dict(), weights_path)
