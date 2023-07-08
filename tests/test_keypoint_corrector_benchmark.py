import logging
import numpy as np
import os
import pathlib
import torch

from casper3d.keypoint_corrector import kp_corrector_reg_sdf, kp_corrector_reg, keypoint_perturbation, registration_eval
from casper3d.model_sdf import VoxelSDF
from casper3d.point_set_registration import PointSetRegistration
from datasets.simple import SimpleSingleClassSDF
from utils.ddn.node import ParamDeclarativeFunction


def test_kp_corrector_points_chamfer_gpu_timing():
    logging.basicConfig(level=logging.DEBUG)

    # parameters
    B = 1
    trial_count = 50

    # load test dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_id = "03001627"
    model_id = "1a6f615e8b1b5ae4dbbc9440457e303e"
    dataset_root = os.path.join(pathlib.Path(__file__).parent.resolve(), "test_data/simple_dataset")
    sdf_dir = os.path.join(dataset_root, "sdfs")
    sdf_grads_dir = os.path.join(dataset_root, "sdf_grads")
    mesh_dir = os.path.join(dataset_root, "meshes")
    pcds_dir = os.path.join(dataset_root, "pcds")
    annotation_dir = os.path.join(dataset_root, "annotations")
    dataset = SimpleSingleClassSDF(
        class_id,
        model_id,
        sdf_dir=sdf_dir,
        sdf_grad_dir=sdf_grads_dir,
        mesh_dir=mesh_dir,
        pcd_dir=pcds_dir,
        annotation_dir=annotation_dir,
        dataset_len=trial_count,
        seed=0
    )
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False)

    # load SDF models & keypoints
    if not torch.cuda.is_available():
        logging.warning("CUDA device not available. Skipping GPU timing test.")
        return
    logging.info(f"Testing class_id={class_id};\nmodel_id={model_id}")

    model_keypoints = dataset._get_model_keypoints()  # (1, 3, N)
    cad_models = dataset._get_cad_models()  # (1, 3, m)
    model_keypoints = model_keypoints.to(device=device)
    cad_models = cad_models.to(device=device)

    model_keypoints = dataset._get_model_keypoints()  # (1, 3, N)
    model_keypoints = model_keypoints.to(device=device)

    # prepare the keypoint corrector
    corrector_node = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)
    corrector = ParamDeclarativeFunction(problem=corrector_node)

    logging.info(f"model_keypoints shape: {model_keypoints.shape}")
    point_set_reg = PointSetRegistration(source_points=model_keypoints)

    # this
    timing_data = {"naive_reg_time": [], "corrector_time": [], "corrector_time_per_iter": [],"corrector_iterations": []}
    for i, data in enumerate(dataset_loader):
        # ground truth data from the dataset
        input_point_cloud, keypoints_true, rotation_true, translation_true = data

        input_point_cloud = input_point_cloud.to(device=device)
        keypoints_true = keypoints_true.to(device=device)
        rotation_true = rotation_true.to(device=device)
        translation_true = translation_true.to(device=device)

        # generating perturbed keypoints
        # keypoints_true = rotation_true @ model_keypoints + translation_true
        # detected_keypoints = keypoints_true
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, var=0.8, fra=1.0)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using point set registration on perturbed keypoints
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
        end.record()
        torch.cuda.synchronize()
        naive_dur = start.elapsed_time(end)
        timing_data["naive_reg_time"].append(naive_dur)
        logging.info(f"Naive registration time: {naive_dur} ms")

        # # estimate model: using the sdf keypoint corrector
        detected_keypoints.requires_grad = True
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        correction = corrector.forward(detected_keypoints, input_point_cloud)
        end.record()
        torch.cuda.synchronize()
        corrector_dur = start.elapsed_time(end)
        timing_data["corrector_iterations"].append(corrector.problem.iters)
        timing_data["corrector_time"].append(corrector_dur)
        timing_data["corrector_time_per_iter"].append(corrector_dur / corrector.problem.iters)
        logging.info(f"Corrector time: {corrector_dur} ms")

        loss = torch.norm(correction, p=2) ** 2
        loss = loss.sum()
        logging.info("Testing backward: ")
        loss.backward()
        logging.info(f"Shape of detected_keypoints.grad: {detected_keypoints.grad.shape}")
        logging.info(
            f"Sum of abs() of all elements in the detected_keypoints.grad: {detected_keypoints.grad.abs().sum()}"
        )
        #

        # correction = torch.zeros_like(correction)
        R, t = point_set_reg.forward(target_points=detected_keypoints + correction)
        # model_estimate = R @ cad_models + t
        # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate)

        # evaluate the two metrics
        logging.info(
            f"Evaluation error (wo correction): {registration_eval(R_naive, rotation_true, t_naive, translation_true).mean()}",
        )
        logging.info(
            f"Evaluation error (w correction): {registration_eval(R, rotation_true, t, translation_true).mean()}"
        )

    # calculate average times
    logging.info(f"Avg. naive reg time: {np.average(timing_data['naive_reg_time'])} ms")
    logging.info(f"Avg. chamfer corrector time: {np.average(timing_data['corrector_time'])} ms")
    logging.info(f"Avg. chamfer corrector iter: {np.average(timing_data['corrector_iterations'])}")
    logging.info(f"Avg. chamfer corrector per iter time: {np.average(timing_data['corrector_time_per_iter'])} ms")

    return


def test_kp_corrector_sdf_gpu_timing():
    """Test SDF-based keypoint corrector. Currently only ensure it runs without errors."""
    logging.basicConfig(level=logging.DEBUG)

    # parameters
    B = 1
    trial_count = 50

    # load test dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_id = "03001627"
    model_id = "1a6f615e8b1b5ae4dbbc9440457e303e"
    dataset_root = os.path.join(pathlib.Path(__file__).parent.resolve(), "test_data/simple_dataset")
    sdf_dir = os.path.join(dataset_root, "sdfs")
    sdf_grads_dir = os.path.join(dataset_root, "sdf_grads")
    mesh_dir = os.path.join(dataset_root, "meshes")
    pcds_dir = os.path.join(dataset_root, "pcds")
    annotation_dir = os.path.join(dataset_root, "annotations")
    dataset = SimpleSingleClassSDF(
        class_id,
        model_id,
        sdf_dir=sdf_dir,
        sdf_grad_dir=sdf_grads_dir,
        mesh_dir=mesh_dir,
        pcd_dir=pcds_dir,
        annotation_dir=annotation_dir,
        dataset_len=trial_count,
        seed=0
    )
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False)

    # load SDF models & keypoints
    logging.info(f"Testing class_id={class_id};\nmodel_id={model_id}")
    sdf_grid = dataset._get_model_sdf()
    sdf_grad_grid = dataset._get_model_sdf_grad()
    sdf_model_node = VoxelSDF(sdf_grid, sdf_grad_grid, device=device)
    sdf_model = ParamDeclarativeFunction(problem=sdf_model_node)

    model_keypoints = dataset._get_model_keypoints()  # (1, 3, N)
    model_keypoints = model_keypoints.to(device=device)

    # prepare the keypoint corrector
    sdf_corrector_node = kp_corrector_reg_sdf(sdf_model=sdf_model, model_keypoints=model_keypoints)
    sdf_corrector = ParamDeclarativeFunction(problem=sdf_corrector_node)

    logging.info(f"model_keypoints shape: {model_keypoints.shape}")
    point_set_reg = PointSetRegistration(source_points=model_keypoints)

    # this
    timing_data = {"naive_reg_time": [], "corrector_time": [], "corrector_time_per_iter": [],"corrector_iterations": []}
    for i, data in enumerate(dataset_loader):
        # ground truth data from the dataset
        input_point_cloud, keypoints_true, rotation_true, translation_true = data

        input_point_cloud = input_point_cloud.to(device=device)
        keypoints_true = keypoints_true.to(device=device)
        rotation_true = rotation_true.to(device=device)
        translation_true = translation_true.to(device=device)

        # generating perturbed keypoints
        # keypoints_true = rotation_true @ model_keypoints + translation_true
        # detected_keypoints = keypoints_true
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, var=0.8, fra=1.0)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using point set registration on perturbed keypoints
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
        end.record()
        torch.cuda.synchronize()
        naive_dur = start.elapsed_time(end)
        timing_data["naive_reg_time"].append(naive_dur)
        logging.info(f"Naive registration time: {naive_dur} ms")

        # # estimate model: using the sdf keypoint corrector
        detected_keypoints.requires_grad = True
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        sdf_correction = sdf_corrector.forward(detected_keypoints, input_point_cloud)
        end.record()
        torch.cuda.synchronize()
        corrector_dur = start.elapsed_time(end)
        timing_data["corrector_iterations"].append(sdf_corrector.problem.iters)
        timing_data["corrector_time"].append(corrector_dur)
        timing_data["corrector_time_per_iter"].append(corrector_dur / sdf_corrector.problem.iters)
        logging.info(f"SDF Corrector time: {corrector_dur} ms")

        loss = torch.norm(sdf_correction, p=2) ** 2
        loss = loss.sum()
        logging.info("Testing backward: ")
        loss.backward()
        logging.info(f"Shape of detected_keypoints.grad: {detected_keypoints.grad.shape}")
        logging.info(
            f"Sum of abs() of all elements in the detected_keypoints.grad: {detected_keypoints.grad.abs().sum()}"
        )
        #

        # correction = torch.zeros_like(correction)
        R, t = point_set_reg.forward(target_points=detected_keypoints + sdf_correction)
        # model_estimate = R @ cad_models + t
        # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate)

        # evaluate the two metrics
        logging.info(
            f"Evaluation error (wo correction): {registration_eval(R_naive, rotation_true, t_naive, translation_true).mean()}",
        )
        logging.info(
            f"Evaluation error (w correction): {registration_eval(R, rotation_true, t, translation_true).mean()}"
        )

    # calculate average times
    logging.info(f"Avg. naive reg time: {np.average(timing_data['naive_reg_time'])} ms")
    logging.info(f"Avg. sdf corrector time: {np.average(timing_data['corrector_time'])} ms")
    logging.info(f"Avg. sdf corrector iter: {np.average(timing_data['corrector_iterations'])}")
    logging.info(f"Avg. sdf corrector per iter time: {np.average(timing_data['corrector_time_per_iter'])} ms")

    return
