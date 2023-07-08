"""
This implements various visualization functions that are used in our code.

"""
import copy
import logging
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import torch
import torchvision.transforms.functional as F
import trimesh.points
import trimesh.creation
from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from utils.general import pos_tensor_to_o3d


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Credit: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_point_cloud_and_mesh(point_cloud, mesh_model=None):
    """
    point_cloud      : torch.tensor of shape (3, m)

    """
    point_cloud = pos_tensor_to_o3d(pos=point_cloud)
    point_cloud = point_cloud.paint_uniform_color([0.0, 0.0, 1])
    point_cloud.estimate_normals()
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coor_frame.scale(100, coor_frame.get_center())
    # breakpoint()
    if mesh_model is None:
        o3d.visualization.draw_geometries([point_cloud, coor_frame])
    else:
        mesh_model.compute_vertex_normals()
        mesh_model.paint_uniform_color([0.8, 0.1, 0.1])
        o3d.visualization.draw_geometries([point_cloud, mesh_model, coor_frame])

    return None


def visualize_bop_obj_point_clouds_in_frame(scene_pc, objs_in_frame, mesh_db):
    """Visualize object point clouds in one frame"""
    N_objs = len(objs_in_frame)
    for i in range(N_objs):
        obj_label = objs_in_frame[i]["name"]
        T = objs_in_frame[i]["TCO"]
        obj_pc = objs_in_frame[i]["point_cloud"]

        # CAD model
        mesh_path = mesh_db.infos[obj_label]["mesh_path"]
        obj = o3d.io.read_point_cloud(mesh_path)
        cam_obj = obj.scale(1.0e-3, [0, 0, 0])
        cam_obj = cam_obj.transform(np.array(T))

        # visualize
        pcs_to_vis = [scene_pc, obj_pc, cam_obj]
        colors = [
            [0.0, 0.0, 0.8],  # scene
            [0.0, 0.8, 0.0],  # masked scene
            [0.8, 0.0, 0.0],  # objects
        ]
        visualize_pcs(pcs=pcs_to_vis, colors=colors)

    return


def visualize_batched_bop_point_clouds(scene_pcs, masked_scene_pcs, batched_obj_labels, mesh_db, Ts):
    """Visualize based point clouds

    Args:
        scene_pcs: (B, 3, N)
        masked_scene_pcs: in the same frame as scene_pcs
        obj_pcs: T @ obj_pc is in the same frame as scene_pcs
        Ts:
    """
    batch_size = scene_pcs.shape[0]
    assert batch_size == masked_scene_pcs.shape[0]
    assert batch_size == batched_obj_labels.shape[0]
    assert batch_size == Ts.shape[0]

    for b in range(batch_size):
        obj_label = batched_obj_labels[b]
        mesh_path = mesh_db.infos[obj_label]["mesh_path"]
        obj = o3d.io.read_point_cloud(mesh_path)
        # note: mesh db's mesh is in mm, and we assume Ts are in m
        # so we need to scale the models down here
        cam_obj = obj.scale(1.0e-3, [0, 0, 0])
        cam_obj = cam_obj.transform(np.array(Ts[b, ...].detach().to("cpu")))
        pcs_to_vis = [scene_pcs[b, ...], masked_scene_pcs[b, ...], cam_obj]
        colors = [
            [0.0, 0.0, 0.8],  # scene
            [0.0, 0.8, 0.0],  # masked scene
            [0.8, 0.0, 0.0],  # objects
        ]
        visualize_pcs(pcs=pcs_to_vis, colors=colors)


def visualize_pcs(pcs, colors=None):
    """Visualize point clouds with objects transformed"""
    geo_list = []
    if colors is None:
        colors = [None] * len(pcs)
    for pc, color in zip(pcs, colors):
        if torch.is_tensor(pc):
            pc = pc.detach().to("cpu")
            pc_o3d = pos_tensor_to_o3d(pos=pc)
        elif isinstance(pc, np.ndarray):
            pc = torch.as_tensor(pc)
            pc_o3d = pos_tensor_to_o3d(pos=pc)
        else:
            pc_o3d = pc
        if color is not None:
            pc_o3d.paint_uniform_color(color)
        geo_list.append(pc_o3d)
    o3d.visualization.draw_geometries(geo_list)


def imgs_show(imgs):
    """Draw images on screen (using matplotlib)"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = torch.as_tensor(img).detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def visualize_rgb_bboxes(rgb, bboxes, show=False):
    """Visualize bounding boxes on an RGB frame"""
    # plot the boxes and the labels
    image_with_boxes = draw_bounding_boxes(rgb, boxes=bboxes, width=4)
    if show:
        imgs_show(image_with_boxes)
    return image_with_boxes


def visualize_rgb_segmentation(rgb, masks, alpha=0.7, show=False):
    """Visualize masks on an RGB frame"""
    images_with_segmentation = draw_segmentation_masks(rgb, masks=masks, alpha=alpha)
    if show:
        imgs_show(images_with_segmentation)
    return images_with_segmentation


def create_o3d_spheres(points, color, r=0.01):
    """Turn points into Open3D sphere meshes for visualization"""
    if points.shape[0] == 3:
        tgt_pts = copy.deepcopy(points.numpy().transpose())
    elif points.shape[1] == 3:
        tgt_pts = copy.deepcopy(points.numpy())
    else:
        raise ValueError("Incorrect input dimensions.")
    spheres = []
    for xyz_idx in range(len(tgt_pts)):
        kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        kpt_mesh.translate(tgt_pts[xyz_idx])
        kpt_mesh.paint_uniform_color(color)
        spheres.append(kpt_mesh)
    return spheres


def visualize_gt_and_pred_keypoints(input_point_cloud, kp_gt, kp_pred=None, pc_pred=None, meshes=None, radius=0.01):
    """Visualize ground truth keypoints
    Ground truth keypoints are green, predicted keypoints are blue
    """
    for b in range(input_point_cloud.shape[0]):
        pc = input_point_cloud[b, ...].clone().detach().to("cpu")
        kp = kp_gt[b, ...].clone().detach().to("cpu")

        # gt keypoints
        o3d_gt_keypoints = create_o3d_spheres(kp, color=[0, 1, 0], r=radius)

        # input point cloud
        o3d_input_point_cloud = pos_tensor_to_o3d(pc, color=[0, 1, 0])

        pcs_to_vis = [o3d_input_point_cloud]
        pcs_to_vis.extend(o3d_gt_keypoints)

        if pc_pred is not None:
            c_pc_pred = pc_pred[b, ...].clone().detach().to("cpu")
            pcs_to_vis.append(pos_tensor_to_o3d(c_pc_pred, color=[1.0, 0, 1.0]))

        # predicted keypoints
        if kp_pred is not None:
            kp_p = kp_pred[b, ...].clone().detach().to("cpu")
            o3d_pred_keypoints = create_o3d_spheres(kp_p, color=[0, 0, 1], r=radius)
            pcs_to_vis.extend(o3d_pred_keypoints)

        if meshes is not None:
            pcs_to_vis.append(meshes[b])
            visualize_pcs(pcs_to_vis)

    return


def visualize_gt_and_pred_keypoints_w_trimesh(
    input_point_cloud=None,
    kp_gt=None,
    pc_gt=None,
    kp_pred=None,
    pc_pred=None,
    meshes=None,
    radius=0.01,
    save_render=False,
    save_render_path="./",
    render_name="render",
):
    """Visualize ground truth keypoints
    Ground truth keypoints are green, predicted keypoints are blue
    """
    for b in range(input_point_cloud.shape[0]):
        pc = input_point_cloud[b, ...].clone().detach().to("cpu").numpy()
        scene = trimesh.scene.Scene()
        trimesh_input_point_cloud = trimesh.points.PointCloud(
            pc.T, colors=np.repeat([[0, 0, 255]], pc.shape[1], axis=0)
        )
        scene.add_geometry(trimesh_input_point_cloud)

        if kp_gt is not None:
            kp = kp_gt[b, ...].clone().detach().to("cpu").numpy()
            gt_kpts = points2trimesh_spheres(kp, r=radius, color=[255, 0, 0])
            for s in gt_kpts:
                scene.add_geometry(s)

        if pc_gt is not None:
            c_pc_gt = pc_gt[b, ...].clone().detach().to("cpu").numpy()
            trimesh_pc_gt = trimesh.points.PointCloud(
                c_pc_gt.T, colors=np.repeat([[0, 0, 255]], c_pc_gt.shape[1], axis=0)
            )
            scene.add_geometry(trimesh_pc_gt)

        if pc_pred is not None:
            c_pc_pred = pc_pred[b, ...].clone().detach().to("cpu").numpy()
            trimesh_pc_pred = trimesh.points.PointCloud(c_pc_pred.T)
            scene.add_geometry(trimesh_pc_pred)

        if kp_pred is not None:
            kp_p = kp_pred[b, ...].clone().detach().to("cpu").numpy()
            pred_kpts = points2trimesh_spheres(kp_p, r=radius, color=[0, 0, 255])
            for s in pred_kpts:
                scene.add_geometry(s)

        if meshes is not None:
            scene.add_geometry(meshes[b])

        # scene.show(viewer="gl", line_settings={"point_size": 15})
        corners = scene.bounds_corners
        t_r = scene.camera.look_at(corners["geometry_0"], distance=2)
        scene.camera_transform = t_r

        if not save_render:
            scene.show()
        else:
            png = scene.save_image(viewer="gl", line_settings={"point_size": 15})
            with open(os.path.join(save_render_path, render_name + ".png"), "wb") as f:
                f.write(png)
                f.close()

    return


def points2trimesh_spheres(x, r=0.05, color=None):
    """Convert 3 by N matrices to trimesh spheres"""
    N = x.shape[1]
    spheres = []
    for n in range(N):
        s = trimesh.creation.icosphere(radius=r, color=color)
        s.vertices += x[:, n].T
        spheres.append(s)
    return spheres


def visualize_c3po_outputs(input_point_cloud, outputs_data, model_keypoints):
    """Visualize outputs from C3PO

    Predicted point cloud: Blue
    Predicted model keypoints: Red
    Detected keypoints: Green
    """
    if len(outputs_data) == 5:
        predicted_point_cloud, corrected_keypoints, R, t, correction = outputs_data
        predicted_model_keypoints = R @ model_keypoints + t
    elif len(outputs_data) == 6:
        # explanations of the different variables
        # predicted_point_cloud: CAD models transformed by estimated R & t
        # corrected_keypoints: detected keypoints after correction
        # R: estimated rotation
        # t: estimated translation
        # correction: corrected amounts after corrector
        # predicted_model_keypoints: model keypoints transformed by estimated R & t
        predicted_point_cloud, corrected_keypoints, R, t, correction, predicted_model_keypoints = outputs_data
    else:
        raise ValueError("Unknown C3PO data.")

    # predicted point cloud: CAD model points transformed using estimted R & t
    # predicted keypoints: keypoints detected on the input point cloud (with corrector's correction)
    # R, t: estimated pose
    # correction: amount of correction
    # predicted model keypoints: model keypoints after transformation using R, t

    for b in range(predicted_point_cloud.shape[0]):
        pc = input_point_cloud[b, ...].clone().detach().to("cpu")
        kp = corrected_keypoints[b, ...].clone().detach().to("cpu")
        kp_p = predicted_model_keypoints[b, ...].clone().detach().to("cpu")
        pc_p = predicted_point_cloud[b, ...].clone().detach().to("cpu")

        # making O3D meshes
        # predicted point cloud
        o3d_predicted_point_cloud = pos_tensor_to_o3d(pc_p)
        o3d_predicted_point_cloud.paint_uniform_color(color=[0, 0, 1])

        # predicted model keypoints
        o3d_predicted_model_keypoints = create_o3d_spheres(kp_p, color=[1, 0, 0])

        # detected keypoints (after correction)
        o3d_detected_keypoints = create_o3d_spheres(kp, color=[0, 1, 0])

        # input point cloud
        o3d_input_point_cloud = pos_tensor_to_o3d(pc)

        pcs_to_vis = [o3d_predicted_point_cloud, o3d_input_point_cloud]
        pcs_to_vis.extend(o3d_detected_keypoints)
        pcs_to_vis.extend(o3d_predicted_model_keypoints)

        visualize_pcs(pcs_to_vis)


def visualize_cosypose_input_detections(rgbs, detection_inputs, show=False):
    """Visualize the detection inputs to CosyPose coarse+refine model"""
    imgs_w_bbox_drawn = []
    for det_id in range(detection_inputs.infos["batch_im_id"].shape[0]):
        c_im_id = detection_inputs.infos["batch_im_id"][det_id]
        c_bbox = detection_inputs.bboxes[det_id, :].cpu()
        imgs_w_bbox_drawn.append(visualize_rgb_bboxes(rgbs[c_im_id, ...], c_bbox.view(1, 4), show=show))
    return imgs_w_bbox_drawn


def overlay_image(rgb_input, rgb_rendered):
    """
    Overlay a rendered RGB mask on another RGB image
    Assume the color channels of the rgb_rendered
    """
    rgb_input = np.asarray(rgb_input)
    rgb_rendered = np.asarray(rgb_rendered)
    assert rgb_input.dtype == np.uint8 and rgb_rendered.dtype == np.uint8
    mask = ~(rgb_rendered.sum(axis=-1) == 0)

    overlay = np.zeros_like(rgb_input)
    overlay[~mask] = rgb_input[~mask] * 0.6 + 255 * 0.4
    overlay[mask] = rgb_rendered[mask] * 0.8 + 255 * 0.2
    # overlay[mask] = rgb_rendered[mask] * 0.3 + rgb_input[mask] * 0.7

    return overlay


def overlay_mask(rgb_input, masks):
    """
    Overlay boolean mask(s) on another RGB image.

    Args:
        rgb_input: (3, H, W)
        mask: (H, W)
    """
    rgb_input = np.asarray(rgb_input)
    overlay = np.copy(rgb_input)
    for i, mask in enumerate(masks):
        mask_input = np.asarray(mask, dtype=np.uint8)
        assert rgb_input.dtype == np.uint8 and mask_input.dtype == np.uint8

        shift = 0.3 * (i + 1)
        overlay[..., ~mask] = overlay[..., ~mask] * 0.5
        overlay[..., mask] = overlay[..., mask] * (1 - shift) + 255 * shift
        # overlay[..., mask] = 255 * shift

    return overlay


def render_cosypose_prediction_wrt_camera(renderer, pred, camera=None, resolution=(640, 480)):
    pred = pred.cpu()
    camera.update(TWC=np.eye(4))

    list_objects = []
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        obj = dict(
            name=row.label,
            color=(1, 1, 1, 1),
            TWO=pred.poses[n].detach().numpy(),
        )
        list_objects.append(obj)
    rgb_rendered = renderer.render_scene(list_objects, [camera])
    return rgb_rendered


def visualize_cosypose_output(rgbs, preds, K, renderer, show=False):
    """Render Cosypose results"""
    imgs_w_det_drawn = []
    unique_im_ids = np.unique(preds.infos["batch_im_id"])
    assert len(unique_im_ids) == K.shape[0]
    assert rgbs.shape[0] == K.shape[0]
    for c_im_id in range(K.shape[0]):
        c_K = K[c_im_id, ...].detach().numpy()
        camera = dict(K=c_K, resolution=rgbs[c_im_id, ...].shape[-2:], TWC=np.eye(4))
        # select predictions on current frame
        mask = preds.infos["batch_im_id"] == c_im_id
        keep_ids = np.where(mask)[0]
        c_preds = preds[keep_ids]
        rgb_rendered = render_cosypose_prediction_wrt_camera(renderer=renderer, pred=c_preds, camera=camera)[0]["rgb"]
        if show:
            imgs_show(rgb_rendered.transpose((2, 0, 1)))

        # overlay picture
        overlaid_image = overlay_image(rgbs[c_im_id, ...].detach().cpu().numpy().transpose((1, 2, 0)), rgb_rendered)
        if show:
            imgs_show(overlaid_image.transpose((2, 0, 1)))

        imgs_w_det_drawn.append(overlaid_image)

    return imgs_w_det_drawn


def visualize_det_and_pred_masks(rgbs, batch_im_id, det_masks, pred_masks, show=False, cert_scores=None):
    """Visualize the detected and rendered/predicted masks"""
    # 2x2 grid
    # 1. det mask only
    # 2. pred mask only
    # 3. det mask & pred mask
    imgs_w_det_drawn = []
    unique_im_ids = np.unique(batch_im_id)
    assert len(unique_im_ids) == rgbs.shape[0]
    for c_im_id in range(rgbs.shape[0]):
        keep_ids_mask = batch_im_id == c_im_id
        keep_ids = np.where(keep_ids_mask)[0]
        for kk in keep_ids:
            det_overlay = overlay_mask(
                (rgbs[c_im_id, ...].detach().cpu().numpy() * 255).astype("uint8"),
                [det_masks[kk, ...].detach().cpu().numpy()],
            )
            pred_overlay = overlay_mask(
                (rgbs[c_im_id, ...].detach().cpu().numpy() * 255).astype("uint8"),
                [pred_masks[kk, ...].detach().cpu().numpy()],
            )
            both_overlay = overlay_mask(
                (rgbs[c_im_id, ...].detach().cpu().numpy() * 255).astype("uint8"),
                [det_masks[kk, ...].detach().cpu().numpy(), pred_masks[kk, ...].detach().cpu().numpy()],
            )

            if cert_scores is not None:
                logging.info(f"Mask scores (for certification): {cert_scores[kk]}")

            if show:
                imgs_show([det_overlay, pred_overlay, both_overlay])

    return


def visualize_bop_rgb_obj_masks(rgb, bboxes, masks=None, alpha=0.7, show=False):
    """Visualize object RGB masks on one image

    Args:
        rgb:
        bboxes:
        masks:
        alpha:
        show:
    """
    if rgb.dtype is torch.float:
        image = (rgb.clone().detach() * 255.0).to(dtype=torch.uint8)
    else:
        image = rgb.clone().to(dtype=torch.uint8)

    # plot bboxes on images
    image = visualize_rgb_bboxes(image, bboxes, show=False)

    # plot segmentation mask on images
    if masks is not None:
        image = visualize_rgb_segmentation(image, masks=masks, alpha=alpha, show=False)
    if show:
        imgs_show(image)
    return image


def visualize_batched_bop_masks(rgbs, bboxes, masks=None, alpha=0.7, show=False):
    """Helper function to visualize batched data returned from a PoseData dataloader
    for the BOP dataset.

    Args:
        rgbs:
        bboxes: (B, 4)
        masks: batched tensors with uint, each id indicate a new object type
        alpha:
        show:
    """
    assert rgbs.shape[0] == masks.shape[0]
    batch_size = rgbs.shape[0]
    for b in range(batch_size):
        logging.info(f"Visualizing image-{b}")
        # plot bboxes on images
        if rgbs.dtype is torch.float:
            c_image = (rgbs[b, ...].clone().detach() * 255.0).to(dtype=torch.uint8)
        else:
            c_image = rgbs[b, ...].clone().to(dtype=torch.uint8)
        c_image = visualize_rgb_bboxes(c_image, bboxes[b, :].view(1, 4), show=False)
        # plot segmentation mask on images
        if masks is not None:
            c_image = visualize_rgb_segmentation(c_image, masks=masks[b, ...], alpha=alpha, show=False)
        if show:
            imgs_show(c_image)


def visualize_model_n_keypoints(model_list, keypoints_xyz, camera_locations=o3d.geometry.PointCloud()):
    """
    Displays one or more models and keypoints.
    :param model_list: list of o3d Geometry objects to display
    :param keypoints_xyz: list of 3d coordinates of keypoints to visualize
    :param camera_locations: optional camera location to display
    :return: list of o3d.geometry.TriangleMesh mesh objects as keypoint markers
    """
    d = 0
    for model in model_list:
        max_bound = model.get_max_bound()
        min_bound = model.get_min_bound()
        d = max(np.linalg.norm(max_bound - min_bound, ord=2), d)

    keypoint_radius = 0.03 * d

    keypoint_markers = []
    for xyz in keypoints_xyz:
        new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
        new_mesh.translate(xyz)
        new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
        keypoint_markers.append(new_mesh)

    camera_locations.paint_uniform_color([0.1, 0.5, 0.1])
    o3d.visualization.draw_geometries(keypoint_markers + model_list + [camera_locations])

    return keypoint_markers


def visualize_torch_model_n_keypoints(cad_models, model_keypoints):
    """
    cad_models      : torch.tensor of shape (B, 3, m)
    model_keypoints : torch.tensor of shape (B, 3, N)

    """
    batch_size = model_keypoints.shape[0]

    for b in range(batch_size):

        point_cloud = cad_models[b, ...]
        keypoints = model_keypoints[b, ...].cpu()

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        point_cloud = point_cloud.paint_uniform_color([0.0, 0.0, 1])
        point_cloud.estimate_normals()
        keypoints = keypoints.transpose(0, 1).numpy()

        visualize_model_n_keypoints([point_cloud], keypoints_xyz=keypoints)

    return 0


def display_two_pcs(pc1, pc2):
    """
    pc1 : torch.tensor of shape (3, n)
    pc2 : torch.tensor of shape (3, m)
    """
    pc1 = pc1.detach()[0, ...].to("cpu")
    pc2 = pc2.detach()[0, ...].to("cpu")
    object1 = pos_tensor_to_o3d(pos=pc1)
    object2 = pos_tensor_to_o3d(pos=pc2)

    object1.paint_uniform_color([0.8, 0.0, 0.0])
    object2.paint_uniform_color([0.0, 0.0, 0.8])

    o3d.visualization.draw_geometries([object1, object2])

    return None


def scatter_bar_plot(plt, x, y, label, color="orangered"):
    """
    x   : torch.tensor of shape (n)
    y   : torch.tensor of shape (n, k)

    """
    n, k = y.shape
    width = 0.2 * torch.abs(x[1] - x[0])

    x_points = x.unsqueeze(-1).repeat(1, k)
    x_points += width * (torch.rand(size=x_points.shape) - 1)
    y_points = y

    plt.scatter(x_points, y_points, s=20.0, c=color, alpha=0.5, label=label)

    return plt


def update_pos_tensor_to_keypoint_markers(vis, keypoints, keypoint_markers):

    keypoints = keypoints[0, ...].to("cpu")
    keypoints = keypoints.numpy().transpose()

    for i in range(len(keypoint_markers)):
        keypoint_markers[i].translate(keypoints[i], relative=False)
        vis.update_geometry(keypoint_markers[i])
        vis.poll_events()
        vis.update_renderer()
    print("FINISHED UPDATING KEYPOINT MARKERS IN CORRECTOR")
    return keypoint_markers


def display_results(
    input_point_cloud, detected_keypoints, target_point_cloud, target_keypoints=None, render_text=False
):
    """
    inputs:
    input_point_cloud   :   torch.tensor of shape (B, 3, m)
    detected_keypoints  :   torch.tensor of shape (B, 3, N)
    target_point_cloud  :   torch.tensor of shape (B, 3, n)
    target_keypoints    :   torch.tensor of shape (B, 3, N)

    where
    B = batch size
    N = number of keypoints
    m = number of points in the input point cloud
    n = number of points in the target point cloud
    """

    if render_text:
        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        window.add_child(scene)
        # displaying only the first item in the batch
    if input_point_cloud is not None:
        input_point_cloud = input_point_cloud[0, ...].to("cpu")
    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints[0, ...].to("cpu")
    if target_point_cloud is not None:
        target_point_cloud = target_point_cloud[0, ...].to("cpu")

    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints.numpy().transpose()
        keypoint_markers = []
        for xyz in detected_keypoints:
            kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            kpt_mesh.translate(xyz)
            kpt_mesh.paint_uniform_color([0, 0.8, 0.0])
            keypoint_markers.append(kpt_mesh)
        detected_keypoints = keypoint_markers

    if target_keypoints is not None:
        target_keypoints = target_keypoints[0, ...].to("cpu")
        target_keypoints = target_keypoints.numpy().transpose()
        keypoint_markers = []
        for xyz_idx in range(len(target_keypoints)):
            kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            kpt_mesh.translate(target_keypoints[xyz_idx])
            kpt_mesh.paint_uniform_color([1, 0.0, 0.0])
            xyz_label = target_keypoints[xyz_idx] + np.array([0.0, 0.0, 0.0])
            if render_text:
                scene_label = scene.add_3d_label(xyz_label, str(xyz_idx))
                # scene_label.scale = 2.0
            keypoint_markers.append(kpt_mesh)
        target_keypoints = keypoint_markers

    if target_point_cloud is not None:
        target_point_cloud = pos_tensor_to_o3d(target_point_cloud)
        target_point_cloud.paint_uniform_color([0.0, 0.0, 0.7])

    if input_point_cloud is not None:
        input_point_cloud = pos_tensor_to_o3d(input_point_cloud)
        input_point_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    elements_to_viz = []
    if target_point_cloud is not None:
        elements_to_viz = elements_to_viz + [target_point_cloud]
        if render_text:
            bounds = target_point_cloud.get_axis_aligned_bounding_box()
            scene.setup_camera(60, bounds, bounds.get_center())

    if input_point_cloud is not None:
        elements_to_viz = elements_to_viz + [input_point_cloud]
    if detected_keypoints is not None:
        elements_to_viz = elements_to_viz + detected_keypoints
    if target_keypoints is not None:
        elements_to_viz = elements_to_viz + target_keypoints

    if render_text:
        for idx, element_to_viz in enumerate(elements_to_viz):
            scene.scene.add_geometry(str(idx), element_to_viz, rendering.MaterialRecord())
        gui.Application.instance.run()  # Run until user closes window
    else:
        # draw_geometries_with_rotation(elements_to_viz)
        o3d.visualization.draw_geometries(elements_to_viz)

    return None


def temp_expt_1_viz(cad_models, model_keypoints, gt_keypoints=None, colors=None):
    batch_size = model_keypoints.shape[0]
    if gt_keypoints is None:
        gt_keypoints = model_keypoints
    # print("model_keypoints.shape", model_keypoints.shape)
    # print("gt_keypoints.shape", gt_keypoints.shape)
    # print("cad_models.shape", cad_models.shape)

    for b in range(batch_size):
        point_cloud = cad_models[b, ...]
        keypoints = model_keypoints[b, ...].cpu()
        gt_keypoints = gt_keypoints[b, ...].cpu()

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        if colors is not None:
            point_cloud.colors = colors
        else:
            point_cloud = point_cloud.paint_uniform_color([1.0, 1.0, 1])
        point_cloud.estimate_normals()
        keypoints = keypoints.transpose(0, 1).numpy()
        gt_keypoints = gt_keypoints.transpose(0, 1).numpy()

        # visualize_model_n_keypoints([point_cloud], keypoints_xyz=keypoints)

        d = 0
        max_bound = point_cloud.get_max_bound()
        min_bound = point_cloud.get_min_bound()
        d = max(np.linalg.norm(max_bound - min_bound, ord=2), d)

        keypoint_radius = 0.01 * d

        keypoint_markers = []
        for xyz in keypoints:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0, 0.8, 0.0])
            keypoint_markers.append(new_mesh)
        for xyz in gt_keypoints:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.8, 0, 0.0])
            keypoint_markers.append(new_mesh)

        custom_draw_geometry_with_key_callback(keypoint_markers + [point_cloud])
        # o3d.visualization.draw_geometries(keypoint_markers + [point_cloud])

        return keypoint_markers

    return 0


def viz_rgb_pcd(
    target_object,
    viewpoint_camera,
    referenceCamera,
    viewpoint_angle,
    viz=False,
    dataset_path="../../data/ycb/models/ycb/",
):
    pcd = o3d.io.read_point_cloud(
        dataset_path
        + target_object
        + "/clouds/rgb/pc_"
        + viewpoint_camera
        + "_"
        + referenceCamera
        + "_"
        + viewpoint_angle
        + "_masked_rgb.ply"
    )
    xyzrgb = np.load(
        dataset_path
        + target_object
        + "/clouds/rgb/pc_"
        + viewpoint_camera
        + "_"
        + referenceCamera
        + "_"
        + viewpoint_angle
        + "_masked_rgb.npy"
    )
    print(xyzrgb.shape)
    rgb = xyzrgb[0, :, 3:]
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(float) / 255.0)
    print(np.asarray(pcd.points).shape)
    if viz:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def draw_geometries_with_rotation(elements, toggle=True):
    def rotate_view(vis, toggle=toggle):
        ctr = vis.get_view_control()
        if toggle:
            ctr.rotate(0.05, 0)
        else:
            ctr.rotate(-0.05, 0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback(elements, rotate_view)


def custom_draw_geometry_with_key_callback(elements):
    def rotate_view_cw(vis):
        ctr = vis.get_view_control()
        ctr.rotate(5, 0)
        return False

    def rotate_view_ccw(vis):
        ctr = vis.get_view_control()
        ctr.rotate(-5, 0)
        return False

    key_to_callback = {}
    key_to_callback[ord("A")] = rotate_view_cw
    key_to_callback[ord("D")] = rotate_view_ccw

    o3d.visualization.draw_geometries_with_key_callbacks(elements, key_to_callback)


if __name__ == "__main__":

    print("Testing generate_random_keypoints:")
    B = 10
    K = 5
    N = 7
    model_keypoints = torch.rand(K, 3, N)
    y, rot, trans, shape = generate_random_keypoints(batch_size=B, model_keypoints=model_keypoints)
    print(y.shape)
    print(rot.shape)
    print(trans.shape)
    print(shape.shape)


def plt_save_figures(basename, save_folder="./", formats=None, dpi='figure'):
    """Helper function to save figures"""
    if formats is None:
        formats = ["pdf", "png"]

    for format in formats:
        fname = f"{basename}.{format}"
        plt.savefig(os.path.join(save_folder, fname), bbox_inches="tight", dpi=dpi)
