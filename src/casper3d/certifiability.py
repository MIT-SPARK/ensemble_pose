"""
This code defines a metric of (epsilon, delta)-certifiability. Given two registered, shape aligned point clouds
pc and pc_ it determines if the registration + shape alignment is certifiable or not.

"""

# import os
import copy
import logging
import torch
from pytorch3d import ops

import utils.visualization_utils as vutils
from utils.loss_functions import half_chamfer_loss_clamped
from utils.math_utils import sq_half_chamfer_dists


def clamp_by_fixed_threshold(sq_dists, clamp_thres_squared):
    """Clamping by a fixed threshold. All clamped distances will be zero, which
    won't affect certification (always passes epsilon check).
    """
    # true means the point is not clamped
    clamp_mask = torch.le(sq_dists, clamp_thres_squared)
    return sq_dists * clamp_mask, clamp_mask


def eps_bound_by_max(sq_dists, valid_mask, sq_eps):
    """Eps bound through the max"""
    return (sq_dists * valid_mask).max(dim=1)[0] < sq_eps


def eps_bound_by_avg(sq_dists, valid_mask, sq_eps):
    """Eps bound through averaging"""
    return sq_dists.sum(dim=1) / valid_mask.sum(dim=1) < sq_eps


def eps_bound_by_quantile(sq_dists, valid_mask, quantile, sq_eps):
    """Eps bound through quantiles.
    We bound by asserting at least quantile fraction of points are below eps.
    In the case of 0.5 quantile, it is equivalent to saying that the median should be below eps.
    In the case of 1 quantile, it is equivalent to saying that the max should be below eps.
    """
    eps_quantile = torch.sum(torch.le(sq_dists, sq_eps) * valid_mask, dim=1) / valid_mask.sum(dim=1)
    return eps_quantile >= quantile


def certify_pc_by_chamfer_distances(eps_bound_fun, pc_sq_dists, pc_valid_mask, sq_epsilon):
    """Certify point clouds

    Args:
        eps_bound_fun:
        pc_sq_dists: (B, N)
        pc_valid_mask:
        sq_epsilon:
    """
    pc_flag = eps_bound_fun(pc_sq_dists, pc_valid_mask, sq_epsilon)
    return pc_flag


def certify_kp_by_distances(kp_sq_dists, sq_epsilon):
    """Certify keypoints

    Args:
        kp_sq_dists: (B, num keypoints)
        sq_epsilon:
    """
    kp_flag = kp_sq_dists.max(dim=1)[0] < sq_epsilon
    return kp_flag


# ToDo: I don't think certifiability should depend on the clamp_thres.
# - For C3PO, it was ensuring that all the points are epsilon away from the CAD model
# - Option 1: For C3PO++, it should ensure that all the inlier points are epsilon away from the CAD model
# - Option 2: For C3PO++, it should ensure that the points that are epsilon away, comprise at least 90% of the points
#             in the segmentation mask
# - We are doing Option 1 in the following.
#
class certifiability:
    def __init__(self, epsilon, clamp_thres):
        logging.warning("Initializing deprecated certifier.")
        self.epsilon = epsilon
        self.clamp_thres = clamp_thres

    def forward(self, X, Z, kp=None, kp_=None):
        """
        inputs:
        X   : input :   torch.tensor of shape (B, 3, n)
        Z   : model :   torch.tensor of shape (B, 3, m)
        kp  : detected/correct_keypoints    : torch.tensor of shape (B, 3, N)
        kp_ : model keypoints               : torch.tensor of shape (B, 3, N)

        outputs:
        cert    : list of len B of boolean variables
        overlap : torch.tensor of shape (B, 1) = overlap of input X with the model Z
        """
        logging.warning("Using deprecated certifier.")
        confidence_score_ = self.confidence_score(X, Z)

        if kp == None or kp_ == None:
            confidence_score_kp_ = 100000 * torch.ones_like(confidence_score_)
            print("Certifiability is not taking keypoint errors into account.")
            out = confidence_score_ < self.epsilon

        else:
            confidence_score_kp_ = self.confidence_score_kp(kp, kp_)
            out = (confidence_score_ < self.epsilon) & (confidence_score_kp_ < self.epsilon)

        return out

    def forward_with_distances(self, sq_dist_XZ, sq_kp_dist, zero_mask=None, max_loss=True):
        """
        NOTE: on filters out points at 0,0 if nn_idxx_ZX is not none
        inputs:
        sq_dist_XZ  : torch.tensor of shape (B, n, 1)   : sq. distance from every point in X to the closest point in Z
        sq_dist_ZX  : torch.tensor of shape (B, m, 1)   : sq. distance from every point in Z to the closest point in X

        Note: sq_kp_dist is in the shape of (B, K)

        where:
            X   : input point cloud
            Z   : model point cloud
            n   : number of points in X
            m   : number of points in Z
            B   : batch size

        outputs:
        cert    : list of len B of boolean variables
        overlap : torch.tensor of shape (B, 1) = overlap of input X with the model Z
        """

        not_zero_mask = torch.logical_not(zero_mask)
        sq_dist_XZ = sq_dist_XZ.squeeze(-1) * not_zero_mask
        sq_dist_XZ = sq_dist_XZ * torch.le(sq_dist_XZ, self.clamp_thres**2)

        _mask = torch.logical_and(torch.le(sq_dist_XZ.squeeze(-1), self.clamp_thres**2), not_zero_mask)

        if max_loss:
            loss = sq_dist_XZ.max(dim=1)[0]
        else:
            loss = sq_dist_XZ.sum(dim=1) / _mask.sum(dim=1)
        confidence_score_ = torch.sqrt(loss.unsqueeze(-1))
        confidence_score_kp_ = torch.sqrt(sq_kp_dist.max(dim=1)[0].unsqueeze(-1))

        return (confidence_score_ < self.epsilon) & (confidence_score_kp_ < self.epsilon)

    def confidence_scores_with_distances(self, sq_dist_XZ, sq_kp_dist, zero_mask=None, max_loss=True):
        """Returns the confidence scores. Used for testing failure modes."""
        not_zero_mask = torch.logical_not(zero_mask)
        sq_dist_XZ = sq_dist_XZ.squeeze(-1) * not_zero_mask
        sq_dist_XZ = sq_dist_XZ * torch.le(sq_dist_XZ, self.clamp_thres**2)

        _mask = torch.logical_and(torch.le(sq_dist_XZ.squeeze(-1), self.clamp_thres**2), not_zero_mask)

        if max_loss:
            loss = sq_dist_XZ.max(dim=1)[0]
        else:
            loss = sq_dist_XZ.sum(dim=1) / _mask.sum(dim=1)
        confidence_score_ = torch.sqrt(loss.unsqueeze(-1))
        confidence_score_kp_ = torch.sqrt(sq_kp_dist.max(dim=1)[0].unsqueeze(-1))

        return confidence_score_, confidence_score_kp_

    def chamfer_loss(self, pc, pc_, pc_padding=None, max_loss=True):
        """
        inputs:
        pc  : torch.tensor of shape (B, 3, n)
        pc_ : torch.tensor of shape (B, 3, m)
        pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
        max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

        output:
        loss    : (B, 1)
            returns max_loss if max_loss is true
        """

        if pc_padding is None:
            batch_size, _, n = pc.shape
            device_ = pc.device

            # computes a padding by flagging zero vectors in the input point cloud.
            pc_padding = (pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3

        sq_dist, _, _ = ops.knn_points(
            torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False
        )
        # dist (B, n, 1): distance from point in X to the nearest point in Y

        sq_dist = sq_dist.squeeze(-1) * torch.logical_not(pc_padding)
        sq_dist = sq_dist * torch.le(sq_dist, self.clamp_thres**2)
        aa = torch.logical_and(torch.le(sq_dist, self.clamp_thres**2), torch.logical_not(pc_padding))

        if max_loss:
            loss = sq_dist.max(dim=1)[0]
        else:
            loss = sq_dist.sum(dim=1) / aa.sum(dim=1)

        return loss.unsqueeze(-1)

    def confidence_score(self, pc, pc_):
        """
        inputs:
        pc  : input point cloud : torch.tensor of shape (B, 3, n)
        pc_ : model point cloud : torch.tensor of shape (B, 3, m)

        output:
        confidence  : torch.tensor of shape (B, 1)
        """

        return torch.sqrt(self.chamfer_loss(pc, pc_, max_loss=True))

    def confidence_score_kp(self, kp, kp_):
        """
        inputs:
        kp  : input point cloud : torch.tensor of shape (B, 3, n)
        kp_ : model point cloud : torch.tensor of shape (B, 3, m)

        output:
        confidence  : torch.tensor of shape (B, 1)

        """

        return torch.sqrt(((kp - kp_) ** 2).sum(dim=1).max(dim=1)[0].unsqueeze(-1))


class Certifier:
    """A certifier for point clouds"""

    def __init__(self, **cfg):
        """

        Args:
            **cfg: required keys:
                    epsilon
                    object_diameter
                    epsilon_bound_method
                    clamp_method

                    depending on epsilon_bound_method:
                        epsilon_quantile if using quantile

                    depending on clamp_method:
                        clamp_threshold if using fixed
        """
        self.object_diameter = cfg["object_diameter"]
        if cfg["epsilon_type"] == "absolute":
            self.epsilon = cfg["epsilon"]
        elif cfg["epsilon_type"] == "relative":
            self.epsilon = cfg["epsilon"] * cfg["object_diameter"]
        else:
            raise ValueError(f"Unknown epsilon type for certifier: {cfg['epsilon_type']}")
        if "kp_epsilon" in cfg.keys():
            if cfg["epsilon_type"] == "absolute":
                self.kp_epsilon = cfg["kp_epsilon"]
            elif cfg["epsilon_type"] == "relative":
                self.kp_epsilon = cfg["kp_epsilon"] * cfg["object_diameter"]
        else:
            self.kp_epsilon = self.epsilon

        self.epsilon_bound_method = cfg["epsilon_bound_method"]
        self.clamp_method = cfg["clamp_method"]

        # available clamping methods:
        # 1. fixed threshold
        # 2. otsu
        # customized functions
        if self.clamp_method == "fixed":
            self.clamp_threshold = cfg["clamp_threshold"] * cfg["object_diameter"]
            self.clamp_fun = lambda x: clamp_by_fixed_threshold(x, self.clamp_threshold**2)
        else:
            raise NotImplementedError

        # available epsilon bound method:
        # 1. avg
        # 2. max
        # 3. quantile
        self.eps_bound_fun = None
        if self.epsilon_bound_method == "max":
            self.eps_bound_fun = lambda x, valid_mask: eps_bound_by_max(x, valid_mask, self.epsilon**2)
        elif self.epsilon_bound_method == "avg":
            self.eps_bound_fun = lambda x, valid_mask: eps_bound_by_avg(x, valid_mask, self.epsilon**2)
        elif self.epsilon_bound_method == "quantile":
            self.eps_bound_fun = lambda x, valid_mask: eps_bound_by_quantile(
                x, valid_mask, cfg["epsilon_quantile"], self.epsilon**2
            )
        else:
            raise NotImplementedError

    def certify_by_distances(self, pc_sq_dists, kp_sq_dists, pc_valid_mask):
        pc_flag = self.eps_bound_fun(pc_sq_dists, pc_valid_mask)
        #kp_flag = kp_sq_dists.max(dim=1)[0] < self.kp_epsilon**2
        kp_flag = self.eps_bound_fun(kp_sq_dists, torch.ones_like(kp_sq_dists, dtype=torch.bool))
        return pc_flag, kp_flag

    def forward(self, X, Z, kp, kp_):
        """
        inputs:
        X   : input :   torch.tensor of shape (B, 3, n)
        Z   : model :   torch.tensor of shape (B, 3, m)
        kp  : detected/correct_keypoints    : torch.tensor of shape (B, 3, N)
        kp_ : model keypoints               : torch.tensor of shape (B, 3, N)

        outputs:
        cert    : list of len B of boolean variables
        overlap : torch.tensor of shape (B, 1) = overlap of input X with the model Z
        """
        batch_size, _, n = X.shape
        device_ = X.device

        # find pc valid point masks
        valid_mask = (X != torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3

        # calculate PC chamfer distance
        sq_dists = sq_half_chamfer_dists(X, Z)
        clamped_sq_dists, not_clamped_mask = self.clamp_fun(sq_dists)

        valid_mask = torch.logical_and(valid_mask, not_clamped_mask)
        if torch.sum(valid_mask) < 0.5 * n:
            logging.warning("More than half of the points are not valid or being clamped in certifier.")

        # calculate KP distances
        kp_sq_dists = (kp - kp_) ** 2

        # certify PC & KP
        pc_flag, kp_flag = self.certify_by_distances(clamped_sq_dists, kp_sq_dists, valid_mask)
        out = pc_flag & kp_flag

        return out, pc_flag, kp_flag


class MultiModelCertifier:
    """A certifier for multiple parallel models"""

    def __init__(
        self,
        epsilon_pc_db=None,
        epsilon_kp_db=None,
        epsilon_mask_db=None,
        clamp_thres_db=None,
        cad_models_db=None,
        renderer=None,
        cfg=None,
    ):
        """
        Certifier class for MultiModel. epsilon_db stores the epsilon value that we use as the
        upper bound for accepetable certifiable instances. clamp_thres_db stores the "outlier bound"
        where if points are outside this bound we don't consider it as affecting the certification results
        (when calculating chamfer loss).
        Args:
            epsilon_db: dictionary with keys = object name/type, value = epsilon
            clamp_thres_db: dictionary with keys = object name/type, value = epsilon
        """
        self.cfg = cfg

        # certification method parameters (data to be used for certification)
        self.available_cert_methods = {"point_clouds", "keypoints", "rendered_masks"}
        self.c3po_cert_methods = cfg["certifier"]["cert_methods_c3po"]
        assert set(self.c3po_cert_methods).issubset(self.available_cert_methods)
        self.cosypose_cert_methods = cfg["certifier"]["cert_methods_cosypose"]
        assert set(self.cosypose_cert_methods).issubset(self.available_cert_methods)

        # save parameters
        self.eps_pc_db = copy.deepcopy(epsilon_pc_db)
        self.eps_kp_db = copy.deepcopy(epsilon_kp_db)
        self.eps_mask_db = copy.deepcopy(epsilon_mask_db)
        if clamp_thres_db is None:
            # if clamp threshold DB is not provided, then it's infinite
            self.clamp_thres_db = dict()
            for k, v in epsilon_pc_db:
                self.clamp_thres_db[k] = float("inf")
        else:
            self.clamp_thres_db = copy.deepcopy(clamp_thres_db)
        assert epsilon_pc_db.keys() == clamp_thres_db.keys()

        if cad_models_db is None:
            raise ValueError("Missing CAD models DB for certifier.")
        else:
            self.cad_models = cad_models_db

        # update the thresholds & clamp thresholds wrt object diameter
        # note: division by 1000 is to be consistent with the cosypose convention
        for k in epsilon_pc_db.keys():
            self.eps_pc_db[k] *= self.cad_models[k]["cad_model_diameter"]
            self.eps_kp_db[k] *= self.cad_models[k]["cad_model_diameter"]
            self.clamp_thres_db[k] *= self.cad_models[k]["cad_model_diameter"]

        # configuration of the clamping method for point cloud certification
        self.pc_clamp_method = cfg["certifier"]["pc_clamp_method"]
        if self.pc_clamp_method == "fixed":
            self.clamp_fun = lambda x, obj_sq_clamp_threshold: clamp_by_fixed_threshold(x, obj_sq_clamp_threshold)
        else:
            raise NotImplementedError("Unknown PC clamp method.")

        # configuration of the point cloud epsilon bound methods
        self.eps_bound_fn = None
        self.eps_bound_method = cfg["certifier"]["pc_epsilon_bound_method"]
        if self.eps_bound_method == "max":
            self.eps_bound_fn = lambda x, valid_mask, obj_sq_epsilon: eps_bound_by_max(x, valid_mask, obj_sq_epsilon)
        elif self.eps_bound_method == "avg":
            self.eps_bound_fn = lambda x, valid_mask, obj_sq_epsilon: eps_bound_by_avg(x, valid_mask, obj_sq_epsilon)
        elif self.eps_bound_method == "quantile":
            self.eps_bound_fn = lambda x, valid_mask, obj_sq_epsilon: eps_bound_by_quantile(
                x, valid_mask, cfg["certifier"]["pc_epsilon_quantile"], obj_sq_epsilon
            )
        else:
            raise NotImplementedError("Unknown certification point cloud bound method.")

        # configuration of 2D cert
        if renderer is None:
            raise ValueError("Missing renderer")
        else:
            self.renderer = renderer

    def certify_pc(self, input_pc, predicted_pc, eps, clamp_thres):
        """Certify through point clouds"""
        assert input_pc.shape[0] == predicted_pc.shape[0]
        sq_epsilon = eps**2
        sq_clamp_thres = clamp_thres**2

        # mask for non-zero points
        pc_valid_mask = (input_pc != torch.zeros(3, 1).to(device=input_pc.device)).sum(dim=1) == 3
        # a point is valid if it is nonzero point and not clamped
        if torch.sum(pc_valid_mask) < 0.5 * input_pc.shape[-1]:
            logging.warning("More than half of the points are not valid in certifier.")

        # clamp the chamfer distances
        sq_dists = sq_half_chamfer_dists(input_pc, predicted_pc)
        sq_dists.clamp_(max=sq_clamp_thres)

        # certify the point cloud distances
        pc_flag = self.eps_bound_fn(sq_dists, pc_valid_mask, sq_epsilon)
        return pc_flag

    def certify_mask(self, detected_masks, predicted_poses, obj_infos, K, resolution, eps):
        """ Certify through 2D projections and segmentation masks

        Args:
            detected_masks: masks output from a detector model
            predicted_poses: (B, 4, 4) matrices of predicted object poses
            obj_infos: a list of dictionaries, with (name=object label)
            K: (B, 3, 3) camera intrinsics
            resolution:
        """

        renders = self.renderer.render(
            obj_infos=obj_infos,
            TCO=predicted_poses,
            K=K,
            resolution=resolution,
        )
        pred_masks = torch.sum(renders, dim=1) != 0

        confidence_score_render = self._confidence_score_render(detected_masks, pred_masks)
        cert_mask = confidence_score_render > eps
        return cert_mask, confidence_score_render, pred_masks

    def certify(self, model_inputs, model_outputs, K, resolution):
        """Given outputs from multiple models, return indices of certified results for each model"""
        logging.warning("Deprecated!")
        # iterate through each model's outputs
        assert model_inputs.keys() == model_outputs.keys()

        cert_outputs = {model_name: dict() for model_name in model_inputs.keys()}
        for model_name, model_outputs in model_outputs.items():
            if model_name == "c3po_multi":
                cert_outputs[model_name] = self.certify_c3po_multi(
                    c3po_inputs=model_inputs[model_name],
                    c3po_outputs=model_outputs,
                )
            elif model_name == "cosypose_coarse_refine":
                cert_outputs[model_name] = self.certify_cosypose(
                    cosypose_inputs=model_inputs[model_name], cosypose_outputs=model_outputs, K=K, resolution=resolution
                )
                return
            else:
                logging.warning(f"Skipping unknown model type in certification: {model_name}")

        return cert_outputs

    def certify_cosypose(self, cosypose_inputs, cosypose_outputs, K, resolution):
        """Certify results from Cosypose"""
        # 1. certify point clouds
        # 2. certify projection
        logging.warning("Deprecated!")
        raise NotImplementedError

        cert_outputs = {}
        preds = cosypose_outputs[0]
        detections = cosypose_inputs["detections"]

        # point clouds are extracted from the depth map using segmentations (either GT or from seg model)
        # size = (M, 3, N) where M = total number of detections
        all_input_pcs = detections._tensors["point_clouds"]

        num_total_detections = len(detections.infos)
        assert num_total_detections == all_input_pcs.shape[0]
        assert num_total_detections == len(preds.poses)

        # batch iou scores
        if "rendered_masks" in self.cosypose_cert_methods:
            renders = self.renderer.render(
                obj_infos=[dict(name=preds.infos.loc[i, "label"]) for i in range(len(preds))],
                TCO=preds.poses,
                K=K,
                resolution=resolution,
            )
            pred_masks = torch.sum(renders, dim=1) != 0
            det_masks = detections._tensors["masks"]
            if self.cfg["visualization"]["certifier_rendered_mask"]:
                logging.info("Visualizing rendered mask fro certifier.")

                # visualizing detected & predicted masks
                vutils.visualize_det_and_pred_masks(
                    rgbs=cosypose_inputs["images"],
                    batch_im_id=preds.infos["batch_im_id"],
                    det_masks=det_masks,
                    pred_masks=pred_masks,
                    show=True,
                )

            breakpoint()
            confidence_score_render = self._confidence_score_render(det_masks, pred_masks)
            breakpoint()

        # for each output object detection:
        # 1. get the object name, pose and generate the predicted point cloud
        # 2. find the corresponding input point cloud using mask
        # Note that cosypose only needs bounding boxes. However we need mask to extract out the point cloud from depth
        # image using masks
        cert_outputs = {}
        flags = []
        for i in range(num_total_detections):
            object_name = detections.infos["label"][i]
            epsilon_pc = self.eps_pc_db[object_name]
            epsilon_mask = self.eps_mask_db[object_name]
            clamp_thres = self.clamp_thres_db[object_name]

            # extract:
            # input_pc, predicted_pc
            c_cad_model = self.cad_models[object_name]["cad_model_pc"].squeeze(0)
            input_pc = all_input_pcs[i, ...]
            T = preds.poses[i].detach()
            R = T[:3, :3]
            t = T[:3, -1]
            predicted_pc = R @ c_cad_model + t.view(3, 1)

            if "point_clouds" in self.cosypose_cert_methods:
                confidence_score_pc = self._confidence_score_pc(
                    input_pc.unsqueeze(0), predicted_pc.unsqueeze(0), clamp_thres
                )
                flags.append(confidence_score_pc < epsilon_pc)
            # if "rendered_masks" in self.cosypose_cert_methods:
            #    det_mask = detections[i]._tensors["masks"]
            #    pred_mask = self._render_mask(
            #        label=detections.infos["label"][i],
            #        TCO=preds.poses[i].detach().cpu().numpy(),
            #        K=K[i, ...].detach().cpu().numpy(),
            #        resolution=resolution,
            #    )
            #    confidence_score_mask = self._confidence_score_render(
            #        torch.as_tensor(det_mask).to(device="cuda"), torch.as_tensor(pred_mask).to(device="cuda")
            #    )
            #    flags.append(confidence_score_mask < epsilon_mask)

            cert_outputs[object_name] = all(flags)
            flags.clear()

        return cert_outputs

    def certify_c3po_multi(self, c3po_inputs, c3po_outputs):
        """Certify results from C3PO (multi-model)

        Args:
            c3po_inputs: dictionary input to C3PO multi-regression model; dictionary with keys = object name,
                         values = batched point clouds
            c3po_outputs: output from C3PO multi-regression model; dictionary with keys = object name
        """
        # the outputs will be a dictionary with keys=different object names
        # certify each using the stored epsilon DB and clamp thres DB
        logging.warning("Deprecated!")
        cert_outputs = {}
        for obj_name, obj_outputs in c3po_outputs.items():
            # to check we are using the correct version of the point regression model
            assert len(obj_outputs) == 6

            epsilon = self.eps_pc_db[obj_name]
            sq_epsilon = epsilon**2
            clamp_thres = self.clamp_thres_db[obj_name]
            sq_clamp_thres = clamp_thres**2

            # predicted_point_cloud: CAD models transformed by estimated R & t
            # corrected_keypoints: detected keypoints after correction
            # predicted_model_keypoints: model keypoints transformed by estimated R & t
            predicted_pc, corrected_kpts, _, _, _, predicted_model_kpts = obj_outputs
            input_pc = c3po_inputs["object_batched_pcs"][obj_name]
            # batch size needs to be the same
            assert input_pc.shape[0] == predicted_pc.shape[0]

            cert_temp = torch.ones(input_pc.shape[0]).to(dtype=torch.bool, device=input_pc.device)
            if "point_clouds" in self.c3po_cert_methods:
                # mask for non-zero points
                pc_valid_mask = (input_pc != torch.zeros(3, 1).to(device=input_pc.device)).sum(dim=1) == 3

                # clamp the chamfer distances
                sq_dists = sq_half_chamfer_dists(input_pc, predicted_pc)
                clamped_sq_dists, not_clamped_mask = self.clamp_fun(sq_dists, sq_clamp_thres)

                # a point is valid if it is nonzero point and not clamped
                pc_valid_mask = torch.logical_and(pc_valid_mask, not_clamped_mask)
                if torch.sum(pc_valid_mask) < 0.5 * input_pc.shape[-1]:
                    logging.warning("More than half of the points are not valid or being clamped in certifier.")

                # certify the point cloud distances
                pc_flag = self.eps_bound_fn(clamped_sq_dists, pc_valid_mask, sq_epsilon)
                cert_temp = torch.logical_and(pc_flag, cert_temp)
            if "keypoints" in self.c3po_cert_methods:
                kp_sq_dists = (corrected_kpts - predicted_model_kpts) ** 2
                kp_flag = kp_sq_dists.sum(dim=1).max(dim=1)[0] < sq_epsilon
                cert_temp = torch.logical_and(kp_flag, cert_temp)

            cert_outputs[obj_name] = cert_temp

        return cert_outputs

    # def _certify_pc(self, input_pc, predicted_pc, epsilon, clamp_thres):
    #    """Certification through 3D chamfer loss and keypoint loss"""
    #    confidence_score_ = self._confidence_score(input_pc, predicted_pc, thres=clamp_thres)
    #    return confidence_score_ < epsilon

    # def _certify_kpts(self, corrected_kpts, predicted_model_kpts, epsilon):
    #    """Certify keypoints"""
    #    confidence_score_kp_ = self._confidence_score_kp(predicted_model_kpts, corrected_kpts)
    #    return confidence_score_kp_ < epsilon

    def _confidence_score_pc(self, pc, pc_, thres):
        """
        Point cloud confidence score computation.

        Args:
            pc: input point cloud : torch.tensor of shape (B, 3, n)
            pc_: model point cloud : torch.tensor of shape (B, 3, m)

        Returns:
            confidence  : torch.tensor of shape (B, 1)
        """
        return torch.sqrt(half_chamfer_loss_clamped(pc, pc_, thres=thres, max_loss=True))

    def _confidence_score_kp(self, kp, kp_):
        """
        Keypoint confidence score calculation.

        Args:
            kp:  input point cloud : torch.tensor of shape (B, 3, n)
            kp_: model point cloud : torch.tensor of shape (B, 3, m)

        Returns:
            confidence  : torch.tensor of shape (B, 1)
        """
        return torch.sqrt(((kp - kp_) ** 2).sum(dim=1).max(dim=1)[0].unsqueeze(-1))

    def _confidence_score_render(self, det_mask, pred_mask):
        """Return IOU between two masks, one is the detected & partial mask,
        the other is the predicted & full mask (without occlusion).
        """
        intersect_mask = torch.logical_and(det_mask, pred_mask)
        intersects_area = torch.count_nonzero(intersect_mask, dim=(1,2))
        det_area = torch.count_nonzero(det_mask, dim=(1, 2))
        return intersects_area.float() / (det_area.float() + 1.0e-7)


if __name__ == "__main__":

    print("test")

    pc = torch.rand(10, 3, 5)
    # pc_ = pc
    pc_ = pc + 0.1 * torch.rand(size=pc.shape)

    epsilon = 0.1
    clamp_thres = 0.5
    certify = certifiability(epsilon=epsilon, clamp_thres=clamp_thres)
    cert = certify.forward(pc, pc_)

    print(cert)
