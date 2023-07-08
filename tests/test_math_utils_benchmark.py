import logging
import torch.cuda
import torch.profiler
import torch.functional
from torch import _VF
from pytorch3d import ops

from utils.math_utils import *


def test_kNN_torch_benchmark():
    """ Benchmarking different kNN implementations """
    logging.basicConfig(level=logging.DEBUG)
    shape = (50, 1000, 3)
    src = torch.rand(shape).cuda()
    dst = torch.rand(shape).cuda()

    N = 1000
    # kNN_torch
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    start1.record()
    for n in range(N):
        _ = kNN_torch(query=src, dataset=dst, k=16)
    end1.record()
    torch.cuda.synchronize()
    logging.info(f"knn torch time: {start1.elapsed_time(end1) / N}")

    # kNN_torch_fast
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for n in range(N):
        _ = kNN_torch_fast(query=src, dataset=dst, k=16)
    end2.record()
    torch.cuda.synchronize()
    logging.info(f"knn torch fast time: {start2.elapsed_time(end2) / N}")

    # pytorch 3d ops
    start3 = torch.cuda.Event(enable_timing=True)
    end3 = torch.cuda.Event(enable_timing=True)
    start3.record()
    for n in range(N):
        _, _, _ = ops.knn_points(src, dst, K=16, return_sorted=True)
    end3.record()
    torch.cuda.synchronize()
    logging.info(f"pytorch3d knn points time: {start3.elapsed_time(end3) / N}")

    return


def test_square_distance_square_matrices_benchmark():
    logging.basicConfig(level=logging.DEBUG)
    # generate random square matrices
    shape = (50, 1000, 3)
    src = torch.rand(shape).cuda()
    dst = torch.rand(shape).cuda()

    # use square distance function
    N = 1000
    start_sqdist = torch.cuda.Event(enable_timing=True)
    end_sqdist = torch.cuda.Event(enable_timing=True)

    start_sqdist.record()
    for n in range(N):
        dist1 = square_distance(src, dst)
    end_sqdist.record()
    torch.cuda.synchronize()
    logging.info(f"sqdist time: {start_sqdist.elapsed_time(end_sqdist)}")

    # use torch cdist
    start_cdist = torch.cuda.Event(enable_timing=True)
    end_cdist = torch.cuda.Event(enable_timing=True)
    start_cdist.record()
    for n in range(N):
        dist2 = torch.square(torch.functional.cdist(src, dst, compute_mode="use_mm_for_euclid_dist"))
    end_cdist.record()
    torch.cuda.synchronize()
    logging.info(f"cdist time: {start_cdist.elapsed_time(end_cdist)}")

    # use vf cdist
    start_vfcdist = torch.cuda.Event(enable_timing=True)
    end_vfcdist = torch.cuda.Event(enable_timing=True)
    start_vfcdist.record()
    for n in range(N):
        dist3 = torch.square(_VF.cdist(src, dst, 2.0, 1))
    end_vfcdist.record()
    torch.cuda.synchronize()
    logging.info(f"vf cdist time: {start_vfcdist.elapsed_time(end_vfcdist)}")


def test_square_distance_nonsquare_matrices_benchmark():
    logging.basicConfig(level=logging.DEBUG)
    # generate random square matrices
    shape1 = (50, 1000, 3)
    shape2 = (50, 700, 3)
    src = torch.rand(shape1).cuda()
    dst = torch.rand(shape2).cuda()

    # use square distance function
    N = 1000
    start_sqdist = torch.cuda.Event(enable_timing=True)
    end_sqdist = torch.cuda.Event(enable_timing=True)

    start_sqdist.record()
    for n in range(N):
        dist1 = square_distance(src, dst)
    end_sqdist.record()
    torch.cuda.synchronize()
    logging.info(f"sqdist time: {start_sqdist.elapsed_time(end_sqdist)}")

    # use torch cdist
    start_cdist = torch.cuda.Event(enable_timing=True)
    end_cdist = torch.cuda.Event(enable_timing=True)
    start_cdist.record()
    for n in range(N):
        dist2 = torch.functional.cdist(src, dst, compute_mode="use_mm_for_euclid_dist")
    end_cdist.record()
    torch.cuda.synchronize()
    logging.info(f"cdist time: {start_cdist.elapsed_time(end_cdist)}")

    # use vf cdist
    start_vfcdist = torch.cuda.Event(enable_timing=True)
    end_vfcdist = torch.cuda.Event(enable_timing=True)
    start_vfcdist.record()
    for n in range(N):
        dist3 = _VF.cdist(src, dst, 2.0, 1)
    end_vfcdist.record()
    torch.cuda.synchronize()
    logging.info(f"vf cdist time: {start_vfcdist.elapsed_time(end_vfcdist)}")


def test_trilinear_interp_exact_torch_benchmark():
    """Timing the trilinear interpolation function on pytorch"""
    logging.basicConfig(level=logging.DEBUG)

    # generate test data
    def f(x, y, z):
        return 3 * x + 2 * y + z

    device = "cuda" if torch.cuda.is_available() else "cpu"
    origin = torch.tensor([0, 0, 0], device=device)
    dims = torch.tensor((11, 11, 11), device=device)
    x = torch.linspace(origin[0], 10, dims[0])
    y = torch.linspace(origin[1], 20, dims[1])
    z = torch.linspace(origin[2], 30, dims[2])
    res = torch.tensor((x[1] - x[0], y[1] - y[0], z[1] - z[0]), device=device)
    xg, yg, zg = torch.meshgrid(x, y, z, indexing="ij")
    data_grid = f(xg, yg, zg)
    data_grid = torch.tensor(data_grid, device=device)

    # p1 & p2: on the original vertices
    # p3 & p4: arbitrary point
    p1_1 = torch.tensor([0, 0, 0], device=device)
    p2_1 = torch.tensor([10, 20, 30], device=device)
    p3_1 = torch.tensor([1.5, 2.5, 3.5], device=device)
    p4_1 = torch.tensor([5.5, 15.5, 25.5], device=device)

    # outside bounds
    p1_2 = torch.tensor([10, 0, 35], device=device)
    # outside bounds
    p2_2 = torch.tensor([0, -25, 0], device=device)
    p3_2 = torch.tensor([2.5, 3.5, 4.5], device=device)
    p4_2 = torch.tensor([6.5, 16.5, 26.5], device=device)

    # get values
    f1_1, f2_1, f3_1, f4_1 = f(*p1_1), f(*p2_1), f(*p3_1), f(*p4_1)
    f1_2, f2_2, f3_2, f4_2 = f(*p1_2), f(*p2_2), f(*p3_2), f(*p4_2)

    p1_1 = p1_1.reshape((1, 3, 1))
    p2_1 = p2_1.reshape((1, 3, 1))
    p3_1 = p3_1.reshape((1, 3, 1))
    p4_1 = p4_1.reshape((1, 3, 1))

    p1_2 = p1_2.reshape((1, 3, 1))
    p2_2 = p2_2.reshape((1, 3, 1))
    p3_2 = p3_2.reshape((1, 3, 1))
    p4_2 = p4_2.reshape((1, 3, 1))

    # merge into two batches
    batch_1 = torch.cat((p1_1, p2_1, p3_1, p4_1), dim=2)
    batch_2 = torch.cat((p1_2, p2_2, p3_2, p4_2), dim=2)
    points = torch.cat((batch_1, batch_2), dim=0)

    # ensure interpolation result equals to original value at vertices
    f_interp = grid_trilinear_interp_exact_torch(points, origin, res, dims, data_grid, outside_value=0.0)

    # check values
    assert f1_1 == f_interp[0, 0, 0]
    assert f2_1 == f_interp[0, 0, 1]
    assert f3_1 == f_interp[0, 0, 2]
    assert f4_1 == f_interp[0, 0, 3]

    # first two in second batch are out of bounds
    assert 0 == f_interp[1, 0, 0]
    assert 0 == f_interp[1, 0, 1]
    assert f3_2 == f_interp[1, 0, 2]
    assert f4_2 == f_interp[1, 0, 3]


def test_trilinear_interp_torch_benchmark():
    """Timing the trilinear interpolation function on pytorch"""
    logging.basicConfig(level=logging.DEBUG)

    num_pointsets = 200
    trials = 50

    # generate test data
    def f(x, y, z):
        return 3 * x + 2 * y + z

    device = "cuda" if torch.cuda.is_available() else "cpu"
    origin = torch.tensor([0, 0, 0], device=device)
    dims = torch.tensor((11, 11, 11), device=device)
    x = torch.linspace(origin[0], 10, dims[0])
    y = torch.linspace(origin[1], 20, dims[1])
    z = torch.linspace(origin[2], 30, dims[2])
    res = torch.tensor((x[1] - x[0], y[1] - y[0], z[1] - z[0]), device=device)
    xg, yg, zg = torch.meshgrid(x, y, z, indexing="ij")
    data_grid = f(xg, yg, zg)
    data_grid = torch.tensor(data_grid, device=device)

    # p1 & p2: on the original vertices
    # p3 & p4: arbitrary point
    p1_1 = torch.tensor([0, 0, 0], device=device)
    p2_1 = torch.tensor([9.9, 19.9, 29.9], device=device)
    p3_1 = torch.tensor([1.5, 2.5, 3.5], device=device)
    p4_1 = torch.tensor([5.5, 15.5, 25.5], device=device)

    # outside bounds
    p1_2 = torch.tensor([10, 0, 35], device=device)
    # outside bounds
    p2_2 = torch.tensor([0, -25, 0], device=device)
    p3_2 = torch.tensor([2.5, 3.5, 4.5], device=device)
    p4_2 = torch.tensor([6.5, 16.5, 26.5], device=device)

    p1_1 = p1_1.reshape((1, 3, 1))
    p2_1 = p2_1.reshape((1, 3, 1))
    p3_1 = p3_1.reshape((1, 3, 1))
    p4_1 = p4_1.reshape((1, 3, 1))

    p1_2 = p1_2.reshape((1, 3, 1))
    p2_2 = p2_2.reshape((1, 3, 1))
    p3_2 = p3_2.reshape((1, 3, 1))
    p4_2 = p4_2.reshape((1, 3, 1))

    # merge into two batches
    batch_1 = torch.cat((p1_1, p2_1, p3_1, p4_1), dim=2)
    batch_2 = torch.cat((p1_2, p2_2, p3_2, p4_2), dim=2)
    batches = torch.cat((batch_1, batch_2), dim=0)

    # concatenate points to have lots of points
    points = batches.clone()
    for i in range(num_pointsets):
        points = torch.cat((points, batches), dim=2)

    # ensure interpolation result equals to original value at vertices
    with torch.profiler.profile(with_stack=True) as prof:
        for i in range(trials):
            f_interp = grid_trilinear_interp_torch(points, origin, res, dims, data_grid, outside_value=0.0)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("/tmp/trace.json")
