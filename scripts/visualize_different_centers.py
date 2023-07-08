import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh


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


def o3d_mesh_center(fname):
    """o3d mesh center (average of mesh vertices)"""
    mesh_o3d = o3d.io.read_triangle_mesh(filename=fname)
    center = mesh_o3d.get_center()
    return np.array(center).flatten()


def o3d_uniformly_sample_then_avg(fname):
    """Average of uniformly sampled points"""
    mesh_o3d = o3d.io.read_triangle_mesh(filename=fname)
    pcd = mesh_o3d.sample_points_uniformly(number_of_points=50000)
    mean = np.average(np.array(pcd.points), axis=0)
    return mean.flatten()


def o3d_oriented_bbox(fname):
    mesh_o3d = o3d.io.read_triangle_mesh(filename=fname)
    obbox = mesh_o3d.get_oriented_bounding_box()
    return np.array(obbox.center).flatten()


def trimesh_centroid(fname):
    """trimesh's centroid function (weighted average of triangle centroid by surface area)"""
    mesh = trimesh.load(fname, process=False)
    return np.array(mesh.centroid).flatten()


def trimesh_voxelize_centroid(fname):
    """voxelize using trimesh then compute centroid"""
    mesh = trimesh.load(fname, process=False)
    # voxelized = mesh.voxelized(pitch=0.02, method="binvox", exact=True, binvox_path="../external/binvox_bin/binvox")
    voxelized = mesh.voxelized(pitch=5, method="subdivide")
    center = np.average(voxelized.points, axis=0)
    return np.array(center)


def trimesh_uniformly_sample_then_avg(fname):
    """Average of uniformly sampled points"""
    mesh = trimesh.load(fname, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, 50000)
    mean = np.average(np.array(pts), axis=0)
    return mean.flatten()


def ycbv_test_model(model_id):
    mesh_root_folder_path: str = "../data/bop/bop_datasets/ycbv/models/"
    object_mesh_file = os.path.join(mesh_root_folder_path, str(model_id) + ".ply")
    mesh = trimesh.load(object_mesh_file, process=False)

    o3d_center = o3d_mesh_center(object_mesh_file)
    o3d_obbox = o3d_oriented_bbox(object_mesh_file)
    o3d_uniform_sample = o3d_uniformly_sample_then_avg(object_mesh_file)
    trimesh_center = trimesh_centroid(object_mesh_file)
    trimesh_uniform_sample = trimesh_uniformly_sample_then_avg(object_mesh_file)
    trimesh_vox_center = trimesh_voxelize_centroid(object_mesh_file)

    centers = {
        "o3d_center": o3d_center,
        "o3d_obbox": o3d_obbox,
        "o3d_uniform_sample": o3d_uniform_sample,
        "trimesh_centroid": trimesh_center,
        "trimesh_uniform_sample": trimesh_uniform_sample,
        "trimesh_voxelize_centroid": trimesh_vox_center,
    }

    # timing data
    # o3d_center_t = timeit.timeit(lambda: o3d_mesh_center(object_mesh_file), number=50)
    # o3d_uniform_sample_t = timeit.timeit(lambda: o3d_uniformly_sample_then_avg(object_mesh_file), number=50)
    # o3d_obbox_t = timeit.timeit(lambda: o3d_oriented_bbox(object_mesh_file), number=50)
    # trimesh_centroid_t = timeit.timeit(lambda: trimesh_centroid(object_mesh_file), number=50)
    # trimesh_uniform_sample_t = timeit.timeit(lambda: trimesh_uniformly_sample_then_avg(object_mesh_file), number=50)
    # trimesh_vox_centroid_t = timeit.timeit(lambda: trimesh_voxelize_centroid(object_mesh_file), number=50)

    times = None
    # times = {
    #    "o3d_center": o3d_center_t,
    #    "o3d_obbox": o3d_obbox_t,
    #    "o3d_uniform_sample": o3d_uniform_sample_t,
    #    "trimesh_centroid": trimesh_centroid_t,
    #    "trimesh_uniform_sample": trimesh_uniform_sample_t,
    #    "trimesh_voxelize_centroid": trimesh_vox_centroid_t,
    # }

    return mesh, centers, times


if __name__ == "__main__":
    print("Visualize centers of meshes calculated with different methods")

    model_id = "obj_000004"
    mesh, centers, times = ycbv_test_model(model_id)

    # visualize
    names = list(centers.keys())
    simplified_mesh = mesh.simplify_quadratic_decimation(1000)
    all_centers = np.array([centers[name] for name in names])

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_trisurf(
        simplified_mesh.vertices[:, 0],
        simplified_mesh.vertices[:, 1],
        simplified_mesh.vertices[:, 2],
        triangles=simplified_mesh.faces,
        linewidth=0.2,
        alpha=0.2,
    )
    for i in range(len(names)):
        ax.scatter(all_centers[i, 0], all_centers[i, 1], all_centers[i, 2], label=names[i])

    set_axes_equal(ax)
    ax.legend()
    plt.show()
