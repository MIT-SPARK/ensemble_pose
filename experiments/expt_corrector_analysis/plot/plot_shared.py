import numpy as np
import pickle
import torch

from casper3d.certifiability import Certifier
from utils.math_utils import varul_mean, masked_varul_mean


def get_trials_containing_nans(dist_mat):
    return np.unique(np.argwhere(np.isnan(dist_mat))[:, 1])


def load_data(fname):
    """Load a data file"""
    fp = open(fname, "rb")
    parameters, data = pickle.load(fp)
    fp.close()

    # if use_adds_metric:
    #    for noise_idx in range(len(data['chamfer_pose_naive_to_gt_pose_list'])):
    #        data['chamfer_pose_naive_to_gt_pose_list'][noise_idx] = np.asarray(
    #            data['chamfer_pose_naive_to_gt_pose_list'][noise_idx][0].squeeze().to('cpu'))
    #        data['chamfer_pose_corrected_to_gt_pose_list'][noise_idx] = np.asarray(
    #            data['chamfer_pose_corrected_to_gt_pose_list'][noise_idx][0].squeeze().to('cpu'))
    # some parameters
    num_samples = 100
    num_points = 1200

    algo = parameters["algo"]
    object_diameter = parameters["diameter"]
    kp_noise_var_range = parameters["kp_noise_var_range"].to("cpu")
    num_noise_vars = len(kp_noise_var_range)
    kp_noise_fra = parameters["kp_noise_fra"]
    ds_name = parameters["ds_name"]
    object_label = parameters["object_label"]
    # chamfer_clamp_thres_factor = parameters['chamfer_clamp_thres_factor']
    chamfer_clamp_thres_factor = 0.1

    Rerr_naive = np.asarray(data["rotation_err_naive"].to("cpu"))
    Rerr_corrector = np.asarray(data["rotation_err_corrector"].to("cpu"))
    Rerr_icp = np.asarray(data["rotation_err_icp"].to("cpu"))
    terr_naive = np.asarray(data["translation_err_naive"].to("cpu") / object_diameter)
    terr_corrector = np.asarray(data["translation_err_corrector"].to("cpu") / object_diameter)
    terr_icp = np.asarray(data["translation_err_icp"].to("cpu") / object_diameter)

    def preprocess_pointwise_data(field_name):
        data_mat = np.zeros((num_noise_vars, num_samples, num_points))
        data_mat[:] = np.nan
        for i in range(num_noise_vars):
            # distances from each point in the input point cloud to the transformed CAD model
            data_mat[i, :, :] = data[field_name][i][0][0].squeeze(-1).to("cpu").numpy()
            ## distances from each transformed CAD model's point to its closest input point
            # sqdist[i, 1, :, :] = data[field_name][i][0][1].squeeze(-1).to("cpu").numpy()
            # sqdist[i, 2, :, :] = data[field_name][i][0][2].squeeze(-1).to("cpu").numpy()
        return data_mat

    # raw chamfer distance of naive reg & corrector reg
    # size: (num noise vars) * 1 * (tuple of size 3, xy dist, yx dist, yx_index) * (torch tensors of 100x1200x1)
    sqdist_input_naiveest = preprocess_pointwise_data("sqdist_input_naiveest")
    sqdist_input_correctorest = preprocess_pointwise_data("sqdist_input_correctorest")
    sqdist_input_icp = preprocess_pointwise_data("sqdist_input_icp")

    nan_trials = []
    print(f"Naive p2p data trials containing NaNs: {get_trials_containing_nans(sqdist_input_naiveest)}")
    nan_trials.extend(get_trials_containing_nans(sqdist_input_naiveest))
    print(f"Corrector p2p data trials containing NaNs: {get_trials_containing_nans(sqdist_input_correctorest)}")
    nan_trials.extend(get_trials_containing_nans(sqdist_input_correctorest))
    print(f"ICP p2p data trials containing NaNs: {get_trials_containing_nans(sqdist_input_icp)}")
    nan_trials.extend(get_trials_containing_nans(sqdist_input_icp))
    # sqdist_input_naiveest = remove_nans(sqdist_input_naiveest)
    # sqdist_input_correctorest = remove_nans(sqdist_input_correctorest)
    # print(f"NaN trials removed.")

    # raw keypoint distances
    # size: (num noise vars) * 1 * (torch tensors of 100*num_keypoints)
    num_keypoints = data["sqdist_kp_naiveest"][0][0].shape[-1]

    def preprocess_kp_sqdist(field_name):
        kp_sqdist = np.zeros((num_noise_vars, num_samples, num_keypoints))
        kp_sqdist[:] = np.nan
        for i in range(num_noise_vars):
            kp_sqdist[i, :, :] = data[field_name][i][0].to("cpu").numpy()
        return kp_sqdist

    sqdist_kp_naiveest = preprocess_kp_sqdist("sqdist_kp_naiveest")
    sqdist_kp_correctorest = preprocess_kp_sqdist("sqdist_kp_correctorest")
    sqdist_kp_icp = preprocess_kp_sqdist("sqdist_kp_icp")

    print(f"Naive kp2kp data trials containing NaNs: {get_trials_containing_nans(sqdist_kp_naiveest)}")
    nan_trials.extend(get_trials_containing_nans(sqdist_kp_naiveest))
    print(f"Corrector kp2kp data trials containing NaNs: {get_trials_containing_nans(sqdist_kp_correctorest)}")
    nan_trials.extend(get_trials_containing_nans(sqdist_kp_correctorest))
    print(f"ICP kp2kp data trials containing NaNs: {get_trials_containing_nans(sqdist_kp_icp)}")
    nan_trials.extend(get_trials_containing_nans(sqdist_kp_icp))
    # sqdist_kp_naiveest = remove_nans(sqdist_kp_naiveest)
    # sqdist_kp_correctorest = remove_nans(sqdist_kp_correctorest)
    # print(f"NaN trials removed.")

    # process pc padding masks
    pc_padding_masks = preprocess_pointwise_data("pc_padding_masks").astype(bool)

    # preprocess ADD-S metric
    def preprocess_adds(field_name):
        A = np.zeros(
            (
                num_noise_vars,
                num_samples,
            )
        )
        A[:] = np.nan
        for i in range(num_noise_vars):
            A[i, :] = data[field_name][i][0].squeeze().to("cpu").numpy()
        return A

    # preprocess naive and corrector ADD-S metric
    chamfer_pose_corrected_to_gt_pose = preprocess_adds("chamfer_pose_corrected_to_gt_pose_list")
    chamfer_pose_naive_to_gt_pose = preprocess_adds("chamfer_pose_naive_to_gt_pose_list")
    chamfer_pose_icp_to_gt_pose = preprocess_adds("chamfer_pose_icp_to_gt_pose_list")

    # remove NaN trials
    # note: we remove the same trials from every noise var to ensure the columns in the table
    # have the same lengths
    sqdist_kp_naiveest = np.delete(sqdist_kp_naiveest, nan_trials, axis=1)
    sqdist_kp_correctorest = np.delete(sqdist_kp_correctorest, nan_trials, axis=1)
    sqdist_kp_icp = np.delete(sqdist_kp_icp, nan_trials, axis=1)

    sqdist_input_naiveest = np.delete(sqdist_input_naiveest, nan_trials, axis=1)
    sqdist_input_correctorest = np.delete(sqdist_input_correctorest, nan_trials, axis=1)
    sqdist_input_icp = np.delete(sqdist_input_icp, nan_trials, axis=1)

    pc_padding_masks = np.delete(pc_padding_masks, nan_trials, axis=1)

    Rerr_naive = np.delete(Rerr_naive, nan_trials, axis=1)
    Rerr_corrector = np.delete(Rerr_corrector, nan_trials, axis=1)
    Rerr_icp = np.delete(Rerr_icp, nan_trials, axis=1)

    terr_naive = np.delete(terr_naive, nan_trials, axis=1)
    terr_corrector = np.delete(terr_corrector, nan_trials, axis=1)
    terr_icp = np.delete(terr_icp, nan_trials, axis=1)

    chamfer_pose_corrected_to_gt_pose = np.delete(chamfer_pose_corrected_to_gt_pose, nan_trials, axis=1)
    chamfer_pose_naive_to_gt_pose = np.delete(chamfer_pose_naive_to_gt_pose, nan_trials, axis=1)
    chamfer_pose_icp_to_gt_pose = np.delete(chamfer_pose_icp_to_gt_pose, nan_trials, axis=1)

    payload = dict(
        algo=algo,
        object_diameter=object_diameter,
        kp_noise_var_range=np.asarray(kp_noise_var_range),
        kp_noise_fra=kp_noise_fra,
        ds_name=ds_name,
        object_label=object_label,
        # chamfer_clamp_thres_factor = parameters['chamfer_clamp_thres_factor']
        chamfer_clamp_thres_factor=0.1,
        Rerr_naive=Rerr_naive,
        Rerr_corrector=Rerr_corrector,
        Rerr_icp=Rerr_icp,
        terr_naive=terr_naive,
        terr_corrector=terr_corrector,
        terr_icp=terr_icp,
        # raw chamfer distances
        sqdist_input_naiveest=sqdist_input_naiveest,
        sqdist_input_correctorest=sqdist_input_correctorest,
        sqdist_input_icp=sqdist_input_icp,
        # raw keypoint distances
        sqdist_kp_naiveest=sqdist_kp_naiveest,
        sqdist_kp_correctorest=sqdist_kp_correctorest,
        sqdist_kp_icp=sqdist_kp_icp,
        # ADD-S results
        chamfer_pose_corrected_to_gt_pose=chamfer_pose_corrected_to_gt_pose,
        chamfer_pose_naive_to_gt_pose=chamfer_pose_naive_to_gt_pose,
        chamfer_pose_icp_to_gt_pose=chamfer_pose_icp_to_gt_pose,
        # other data
        pc_padding_masks=pc_padding_masks,
    )
    return payload


def evaluate_certifier(data_payload, certi_masks):
    """Compare errors"""
    object_diameter = data_payload["object_diameter"]
    num_kp_noise_vars = data_payload["sqdist_kp_naiveest"].shape[0]

    Rerr_naive = torch.as_tensor(data_payload["Rerr_naive"])
    Rerr_corrector = torch.as_tensor(data_payload["Rerr_corrector"])
    terr_naive = torch.as_tensor(data_payload["terr_naive"] / object_diameter)
    terr_corrector = torch.as_tensor(data_payload["terr_corrector"] / object_diameter)

    # get rotation / translation errors mean / variances of all instances
    Rerr_naive_var, Rerr_naive_mean = varul_mean(Rerr_naive)
    Rerr_corrector_var, Rerr_corrector_mean = varul_mean(Rerr_corrector)
    terr_naive_var, terr_naive_mean = varul_mean(terr_naive)
    terr_corrector_var, terr_corrector_mean = varul_mean(terr_corrector)

    # get rotation / translation errors mean / variances of certified instances
    certi_naive, certi_corrector, certi_icp = torch.as_tensor(certi_masks["certi_naive"]), torch.as_tensor(
        certi_masks["certi_corrector"]), torch.as_tensor(certi_masks["certi_icp"])

    Rerr_naive_certi_var, Rerr_naive_certi_mean = masked_varul_mean(Rerr_naive, mask=certi_naive)
    Rerr_corrector_certi_var, Rerr_corrector_certi_mean = masked_varul_mean(Rerr_corrector, mask=certi_corrector)
    terr_naive_certi_var, terr_naive_certi_mean = masked_varul_mean(terr_naive, mask=certi_naive)
    terr_corrector_certi_var, terr_corrector_certi_mean = masked_varul_mean(terr_corrector, mask=certi_corrector)

    # plotting rotation errors
    kp_noise_var_range = data_payload["kp_noise_var_range"]

    # adds errors
    chamfer_pose_corrected_to_gt_pose = torch.as_tensor(
        data_payload["chamfer_pose_corrected_to_gt_pose"] / object_diameter
    )
    chamfer_pose_naive_to_gt_pose = torch.as_tensor(data_payload["chamfer_pose_naive_to_gt_pose"] / object_diameter)
    chamfer_pose_icp_to_gt_pose = torch.as_tensor(data_payload["chamfer_pose_icp_to_gt_pose"] / object_diameter)

    chamfer_metric_naive_var, chamfer_metric_naive_mean = varul_mean(chamfer_pose_naive_to_gt_pose)
    chamfer_metric_naive_certi_var, chamfer_metric_naive_certi_mean = masked_varul_mean(
        chamfer_pose_naive_to_gt_pose, mask=certi_naive
    )

    chamfer_metric_corrected_var, chamfer_metric_corrected_mean = varul_mean(chamfer_pose_corrected_to_gt_pose)
    chamfer_metric_corrected_certi_var, chamfer_metric_corrected_certi_mean = masked_varul_mean(
        chamfer_pose_corrected_to_gt_pose, mask=certi_corrector
    )

    chamfer_metric_icp_var, chamfer_metric_icp_mean = varul_mean(chamfer_pose_icp_to_gt_pose)
    chamfer_metric_icp_certi_var, chamfer_metric_icp_certi_mean = masked_varul_mean(
        chamfer_pose_icp_to_gt_pose, mask=certi_icp
    )

    payload = dict(
        object_label=data_payload["object_label"],
        object_diameter=object_diameter,
        certi_naive=certi_naive,
        certi_corrector=certi_corrector,
        certi_icp=certi_icp,
        # errors
        Rerr_corrector=Rerr_corrector,
        Rerr_naive=Rerr_naive,
        terr_corrector=terr_corrector,
        terr_naive=terr_naive,
        # rot error summary statistics
        Rerr_naive_mean=Rerr_naive_mean,
        Rerr_naive_var=Rerr_naive_var,
        Rerr_naive_certi_mean=Rerr_naive_certi_mean,
        Rerr_naive_certi_var=Rerr_naive_certi_var,
        Rerr_corrector_mean=Rerr_corrector_mean,
        Rerr_corrector_var=Rerr_corrector_var,
        Rerr_corrector_certi_mean=Rerr_corrector_certi_mean,
        Rerr_corrector_certi_var=Rerr_corrector_certi_var,
        # translation errors statistics
        terr_naive_mean=terr_naive_mean,
        terr_naive_var=terr_naive_var,
        terr_naive_certi_mean=terr_naive_certi_mean,
        terr_naive_certi_var=terr_naive_certi_var,
        terr_corrector_mean=terr_corrector_mean,
        terr_corrector_var=terr_corrector_var,
        terr_corrector_certi_mean=terr_corrector_certi_mean,
        terr_corrector_certi_var=terr_corrector_certi_var,
        # ADD-S errors naive
        chamfer_pose_naive_to_gt_pose=chamfer_pose_naive_to_gt_pose,
        chamfer_metric_naive_var=chamfer_metric_naive_var,
        chamfer_metric_naive_mean=chamfer_metric_naive_mean,
        chamfer_metric_naive_certi_var=chamfer_metric_naive_certi_var,
        chamfer_metric_naive_certi_mean=chamfer_metric_naive_certi_mean,
        # ADD-S errors corrector
        chamfer_pose_corrected_to_gt_pose=chamfer_pose_corrected_to_gt_pose,
        chamfer_metric_corrected_var=chamfer_metric_corrected_var,
        chamfer_metric_corrected_mean=chamfer_metric_corrected_mean,
        chamfer_metric_corrected_certi_var=chamfer_metric_corrected_certi_var,
        chamfer_metric_corrected_certi_mean=chamfer_metric_corrected_certi_mean,
        # ADD-S errors icp
        chamfer_pose_icp_to_gt_pose=chamfer_pose_icp_to_gt_pose,
        chamfer_metric_icp_var=chamfer_metric_icp_var,
        chamfer_metric_icp_mean=chamfer_metric_icp_mean,
        chamfer_metric_icp_certi_var=chamfer_metric_icp_certi_var,
        chamfer_metric_icp_certi_mean=chamfer_metric_icp_certi_mean,
    )
    return payload


def get_certified_instances(payload, certifier_cfg):
    """Return certified instances only.
    Combine with other histogram functions to plot only certified data
    """
    object_diameter = payload["object_diameter"]
    chamfer_clamp_thres_factor = payload["chamfer_clamp_thres_factor"]
    xlen = payload["kp_noise_var_range"].shape[0]
    num_samples = payload["sqdist_input_naiveest"].shape[1]

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    certifier = Certifier(object_diameter=object_diameter, **certifier_cfg)

    certi_naive = torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_)
    certi_corrector = torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_)
    certi_icp = torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_)

    certi_naive_failure_modes = {
        "pt": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
        "kp": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
    }
    certi_corrector_failure_modes = {
        "pt": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
        "kp": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
    }
    certi_icp_failure_modes = {
        "pt": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
        "kp": torch.zeros(size=(xlen, num_samples), dtype=torch.bool).to(device=device_),
    }

    sqdist_input_naiveest = torch.as_tensor(payload["sqdist_input_naiveest"]).to(device_)
    sqdist_input_correctorest = torch.as_tensor(payload["sqdist_input_correctorest"]).to(device_)
    sqdist_input_icpest = torch.as_tensor(payload["sqdist_input_icp"]).to(device_)
    sqdist_kp_naiveest = torch.as_tensor(payload["sqdist_kp_naiveest"]).to(device_)
    sqdist_kp_correctorest = torch.as_tensor(payload["sqdist_kp_correctorest"]).to(device_)
    sqdist_kp_icpest = torch.as_tensor(payload["sqdist_kp_icp"]).to(device_)
    pc_padding_masks = torch.as_tensor(payload["pc_padding_masks"]).to(device_)

    for kp_noise_var_i in range(xlen):

        sqdist_input_naive = sqdist_input_naiveest[kp_noise_var_i]
        sqdist_input_corrector = sqdist_input_correctorest[kp_noise_var_i]
        sqdist_input_icp = sqdist_input_icpest[kp_noise_var_i]

        sqdist_kp_naive = sqdist_kp_naiveest[kp_noise_var_i]
        sqdist_kp_corrector = sqdist_kp_correctorest[kp_noise_var_i]
        sqdist_kp_icp = sqdist_kp_icpest[kp_noise_var_i]

        pc_padding = pc_padding_masks[kp_noise_var_i]

        valid_mask = torch.logical_not(pc_padding)

        # certify PC and KP for icp
        clamped_sq_dists_icp, not_clamped_mask_icp = certifier.clamp_fun(sqdist_input_icp)
        if torch.sum(not_clamped_mask_icp * valid_mask) < 0.5 * sqdist_input_icp.shape[1]:
            print("More than half of the points are not valid or being clamped in certifier for icp.")
        c_icp_cert_pc, c_icp_cert_kp = certifier.certify_by_distances(
            clamped_sq_dists_icp, sqdist_kp_icp, valid_mask
        )
        #c_icp = c_icp_cert_pc & c_icp_cert_kp
        c_icp = c_icp_cert_pc

        # certify PC and KP for naive
        clamped_sq_dists_naive, not_clamped_mask_naive = certifier.clamp_fun(sqdist_input_naive)
        if torch.sum(not_clamped_mask_naive * valid_mask) < 0.5 * sqdist_input_naive.shape[1]:
            print("More than half of the points are not valid or being clamped in certifier for naive.")
        c_naive_cert_pc, c_naive_cert_kp = certifier.certify_by_distances(
            clamped_sq_dists_naive, sqdist_kp_naive, valid_mask
        )
        #c_naive = c_naive_cert_pc & c_naive_cert_kp
        c_naive = c_naive_cert_pc

        # certify PC and KP for corrector
        clamped_sq_dists_corrector, not_clamped_mask_corrector = certifier.clamp_fun(sqdist_input_corrector)
        print(
            f"Avg num points clamped per trial: {torch.mean(torch.sum(torch.logical_not(not_clamped_mask_corrector), dim=1).to(dtype=float))}"
        )
        if torch.sum(not_clamped_mask_corrector * valid_mask) < 0.5 * sqdist_input_corrector.shape[1]:
            print("More than half of the points are not valid or being clamped in certifier for naive.")
        c_corrector_cert_pc, c_corrector_cert_kp = certifier.certify_by_distances(
            clamped_sq_dists_corrector, sqdist_kp_corrector, valid_mask
        )
        #c_corrector = c_corrector_cert_pc & c_corrector_cert_kp
        c_corrector = c_corrector_cert_pc

        certi_naive[kp_noise_var_i, ...] = c_naive.squeeze(-1).to(device=device_, dtype=torch.bool)
        certi_corrector[kp_noise_var_i, ...] = c_corrector.squeeze(-1).to(device=device_, dtype=torch.bool)
        certi_icp[kp_noise_var_i, ...] = c_icp.squeeze(-1).to(device=device_, dtype=torch.bool)

        # save the failure modes
        certi_naive_failure_modes["pt"][kp_noise_var_i] = torch.logical_not(c_naive_cert_pc).flatten()
        certi_naive_failure_modes["kp"][kp_noise_var_i] = torch.logical_not(c_naive_cert_kp).flatten()
        certi_corrector_failure_modes["pt"][kp_noise_var_i] = torch.logical_not(c_corrector_cert_pc).flatten()
        certi_corrector_failure_modes["kp"][kp_noise_var_i] = torch.logical_not(c_corrector_cert_kp).flatten()
        certi_icp_failure_modes["pt"][kp_noise_var_i] = torch.logical_not(c_icp_cert_pc).flatten()
        certi_icp_failure_modes["kp"][kp_noise_var_i] = torch.logical_not(c_icp_cert_kp).flatten()

    return (
        certi_naive,
        certi_corrector,
        certi_icp,
        certi_naive_failure_modes,
        certi_corrector_failure_modes,
        certi_icp_failure_modes,
        certifier.epsilon,
        certifier.clamp_threshold,
    )
