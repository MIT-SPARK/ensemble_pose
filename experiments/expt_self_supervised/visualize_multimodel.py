import argparse
from tqdm import tqdm

from proposed_model import (
    load_multi_model,
    load_certifier,
    load_all_cad_models,
    load_batch_renderer,
)
from supervised_synth_training import manage_visualization
from training_utils import *
from utils.evaluation_metrics import add_s_error
from utils.math_utils import set_all_random_seeds
from utils.torch_utils import cast2cuda
from utils.visualization_utils import display_results


def visual_test(test_loader, model_id, model, cfg, device=None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cad_models = test_loader.dataset._get_cad_models()
    cad_models = cad_models.to(device)

    for i, vdata in enumerate(test_loader):
        input_point_cloud, keypoints_target, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        keypoints_target = keypoints_target.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Make predictions for this batch
        model.eval()
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _, predicted_model_keypoints = model(
            model_id, input_point_cloud
        )

        ## certification
        # certi = certify(
        #    input_point_cloud=input_point_cloud,
        #    predicted_point_cloud=predicted_point_cloud,
        #    corrected_keypoints=predicted_keypoints,
        #    predicted_model_keypoints=predicted_model_keypoints,
        #    epsilon=hyper_param["epsilon"],
        # )
        # print("Certifiable: ", certi)

        # add-s
        pc_t = R_target @ cad_models + t_target
        add_s = add_s_error(
            predicted_point_cloud=predicted_point_cloud,
            ground_truth_point_cloud=pc_t,
            threshold=cfg["training"]["adds_threshold"],
        )
        print("ADD-S: ", add_s)

        pc = input_point_cloud.clone().detach().to("cpu")
        pc_p = predicted_point_cloud.clone().detach().to("cpu")
        pc_t = pc_t.clone().detach().to("cpu")
        kp = keypoints_target.clone().detach().to("cpu")
        kp_p = predicted_keypoints.clone().detach().to("cpu")
        print("DISPLAY: INPUT PC")
        display_results(input_point_cloud=pc, detected_keypoints=None, target_point_cloud=None, target_keypoints=kp)
        print("DISPLAY: INPUT AND PREDICTED PC")
        display_results(input_point_cloud=pc, detected_keypoints=kp_p, target_point_cloud=pc_p, target_keypoints=kp)
        print("DISPLAY: TRUE AND PREDICTED PC")
        display_results(input_point_cloud=pc_t, detected_keypoints=kp_p, target_point_cloud=pc_p, target_keypoints=kp)

        del pc, pc_p, kp, kp_p, pc_t
        del (
            input_point_cloud,
            keypoints_target,
            R_target,
            t_target,
            predicted_point_cloud,
            predicted_keypoints,
            R_predicted,
            t_predicted,
        )

        if i >= 9:
            break


def vis_multimodel(model_weights_path, cad_models_db, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataset and dataloader
    # note: validation set dataloader has a subset random sampler
    ds_train, ds_val, ds_iter_train, ds_iter_val = load_frame_objs_datasets(cfg)

    object_ds, mesh_db_batched = load_objects(cfg)

    # load batch renderer
    batch_renderer = load_batch_renderer(cfg)

    # load scene renderer
    # (not used for supervised training)
    scene_renderer = []

    # load multi model
    model = load_multi_model(batch_renderer=batch_renderer, meshdb_batched=mesh_db_batched, device=device, cfg=cfg)
    model.eval()

    # load checkpoint
    logging.info(f"Loading checkpoint from {model_weights_path}")
    save = torch.load(model_weights_path)
    model.load_state_dict(save["state_dict"])

    # load segmentation model
    seg_model = None

    # load certifier
    all_cad_models = load_all_cad_models(device=device, cfg=cfg)
    cert_model = load_certifier(all_cad_models, scene_renderer, cfg)

    for i, data in enumerate(tqdm(ds_iter_train)):
        # data is a PoseData
        # prepare data and parameters
        batch_size, _, h, w = data.images.shape

        if cfg["detector"]["det_type"] == "gt":
            # c3po inputs: ground truth masked out object point clouds
            object_batched_pcs = dict()
            for k, v in data.model_to_batched_pcs.items():
                if len(v) != 0:
                    object_batched_pcs[k] = cast2cuda(v).float()
        else:
            raise NotImplementedError

        # get ground truth object R & t
        objects_batched_gt_Rs = dict()
        for k, v in data.model_to_batched_gt_R.items():
            if len(v) != 0:
                objects_batched_gt_Rs[k] = cast2cuda(v).float()

        objects_batched_gt_ts = dict()
        for k, v in data.model_to_batched_gt_t.items():
            if len(v) != 0:
                objects_batched_gt_ts[k] = cast2cuda(v).float()

        # forward pass on the MultiModel
        # make sure the order is consistent with the configuration yaml file's spec
        # c3po model inputs:
        # - input point cloud
        # NOTE: Check the input/model object point clouds' scale
        # make sure to normalize input for C3PO
        inputs = dict(
            c3po_multi=dict(object_batched_pcs=object_batched_pcs),
        )

        # outputs format:
        # c3po: a dictionary with keys = object names, values = tuples containing:
        #       predicted_pc, corrected_kpts, R, t, correction, predicted_model_kpts
        outputs = model(**inputs)

        manage_visualization(
            data=data, model=model, model_inputs=inputs, model_outputs=outputs, mesh_db=mesh_db_batched, cfg=cfg
        )

        for obj_label in outputs["c3po_multi"].keys():

            # calculate the KP loss
            _, kp_pred, R, t, _, _ = outputs["c3po_multi"][obj_label]
            model_keypoints = cad_models_db[obj_label]["original_model_keypoints"]
            R_gt = objects_batched_gt_Rs[obj_label]
            t_gt = torch.reshape(objects_batched_gt_ts[obj_label], (R_gt.shape[0], 3, 1))
            kp_gt = R_gt @ model_keypoints + t_gt

            # visualize
            if cfg["visualization"]["gt_keypoints"]:
                # TODO: Input point clouds do not match the GT transformed keypoints. Check ref frames
                vutils.visualize_gt_and_pred_keypoints(
                    input_point_cloud=inputs["c3po_multi"]["object_batched_pcs"][obj_label],
                    kp_gt=kp_gt,
                    kp_pred=kp_pred,
                )
    return


if __name__ == "__main__":
    """
    usage:
    python visualize_model.py ycbv obj_000001 kp_detector ./exp_results/supervised/ycbv/obj_000001/_ycbv_best_supervised_kp_point_transformer.pth
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", help="path to the multimodel checkpoint")
    parser.add_argument(
        "--config",
        help="path of the config file",
        default=f"./configs/vis_multimodel.yml",
        type=str,
    )

    args = parser.parse_args()
    model_weights_path = args.weights_path

    set_all_random_seeds(0)

    # handle https://github.com/pytorch/pytorch/issues/77527
    torch.backends.cuda.preferred_linalg_library("cusolver")

    # load config params
    config_params_file = args.config
    cfg = load_yaml_cfg(config_params_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_cad_models = load_all_cad_models(device=device, cfg=cfg)
    vis_multimodel(model_weights_path, all_cad_models, cfg)

