import torch.cuda
import torch.functional
from pytorch3d import ops

from utils.math_utils import *


def test_square_distance():
    """Test square distance function"""
    logging.basicConfig(level=logging.DEBUG)
    # generate random square matrices
    shape = (50, 1000, 3)
    N = 100

    for n in range(N):
        src = torch.rand(shape).cuda()
        dst = torch.rand(shape).cuda()
        dist1 = square_distance(src, dst)
        dist2 = torch.square(torch.functional.cdist(src, dst, compute_mode="use_mm_for_euclid_dist"))
        assert torch.allclose(dist1, dist2, atol=1e-06)


def test_kNN_torch():
    logging.basicConfig(level=logging.DEBUG)
    # generate random square matrices
    shape = (50, 1000, 3)
    shape2 = (50, 700, 3)
    N = 100

    for n in range(N):
        src = torch.rand(shape).cuda()
        dst = torch.rand(shape).cuda()

        neighbors1 = kNN_torch(query=src, dataset=dst, k=16)
        neighbors2 = kNN_torch_fast(query=src, dataset=dst, k=16)
        _, neighbors3, _ = ops.knn_points(src, dst, K=16, return_sorted=True)
        assert neighbors3.shape == neighbors1.shape
        assert neighbors2.shape == neighbors1.shape

        # due to numerical inaccuracies, the sequence of the neighbors might differ
        # we just want to ensure most of them are consistent
        unequal_count = torch.argwhere(neighbors1 != neighbors2).shape[0]
        assert unequal_count / (1000 * 1000) < 0.05

        unequal_count = torch.argwhere(neighbors1 != neighbors3).shape[0]
        assert unequal_count / (1000 * 1000) < 0.05

    # generate random rectangular matrices (query > dataset)
    for n in range(N):
        src = torch.rand(shape).cuda()
        dst = torch.rand(shape2).cuda()

        neighbors1 = kNN_torch(query=src, dataset=dst, k=16)
        neighbors2 = kNN_torch_fast(query=src, dataset=dst, k=16)
        _, neighbors3, _ = ops.knn_points(src, dst, K=16, return_sorted=True)
        assert neighbors3.shape == neighbors1.shape
        assert neighbors2.shape == neighbors1.shape

        # due to numerical inaccuracies, the sequence of the neighbors might differ
        # we just want to ensure most of them are consistent
        unequal_count = torch.argwhere(neighbors1 != neighbors2).shape[0]
        assert unequal_count / (1000 * 1000) < 0.05

        unequal_count = torch.argwhere(neighbors1 != neighbors3).shape[0]
        assert unequal_count / (1000 * 1000) < 0.05

    # generate random rectangular matrices (query < dataset)
    for n in range(N):
        src = torch.rand(shape2).cuda()
        dst = torch.rand(shape).cuda()

        neighbors1 = kNN_torch(query=src, dataset=dst, k=16)
        neighbors2 = kNN_torch_fast(query=src, dataset=dst, k=16)
        _, neighbors3, _ = ops.knn_points(src, dst, K=16, return_sorted=True)
        assert neighbors3.shape == neighbors1.shape
        assert neighbors2.shape == neighbors1.shape

        # due to numerical inaccuracies, the sequence of the neighbors might differ
        # we just want to ensure most of them are consistent
        unequal_count = torch.argwhere(neighbors1 != neighbors2).shape[0]
        assert unequal_count / (1000 * 1000) < 0.05

        unequal_count = torch.argwhere(neighbors1 != neighbors3).shape[0]
        assert unequal_count / (1000 * 1000) < 0.05

    return


def test_trilinear_interp_vertices():
    """Test trilinear interpolation on grid vertices"""
    logging.basicConfig(level=logging.DEBUG)

    # generate test data
    def f(x, y, z):
        return 3 * x + 2 * y + z

    origin = np.array([0, 0, 0])
    dims = np.array((11, 11, 11))
    x = np.linspace(origin[0], 10, dims[0])
    y = np.linspace(origin[1], 10, dims[1])
    z = np.linspace(origin[2], 10, dims[2])
    res = np.array((x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
    data_grid = f(xg, yg, zg)

    for i in range(x.size):
        for j in range(y.size):
            for k in range(z.size):
                p = np.array([x[i], y[i], z[i]])
                f_v = f(*p)
                f_interp = grid_trilinear_interp(p, origin, res, dims, data_grid)
                assert f_v == f_interp

    return


def test_trilinear_interp_equal_res():
    """Test trilinear interpolation in a grid with equal resolution on x, y, and z"""
    logging.basicConfig(level=logging.DEBUG)

    # generate test data
    def f(x, y, z):
        return 3 * x + 2 * y + z

    origin = np.array([0, 0, 0])
    dims = np.array((11, 11, 11))
    x = np.linspace(origin[0], 10, dims[0])
    y = np.linspace(origin[1], 10, dims[1])
    z = np.linspace(origin[2], 10, dims[2])
    res = np.array((x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
    data_grid = f(xg, yg, zg)

    # ensure interpolation result equals to original value at vertices
    p1 = np.array([0, 0, 0])
    f1 = f(*p1)
    f1_interp = grid_trilinear_interp(p1, origin, res, dims, data_grid)
    assert f1 == f1_interp

    p2 = np.array([10, 10, 10])
    f2 = f(*p2)
    f2_interp = grid_trilinear_interp(p2, origin, res, dims, data_grid)
    assert f2 == f2_interp

    # interpolate arbitrary point
    # because f is linear, we should get exact results
    p3 = np.array([1.5, 2.5, 3.5])
    f3 = f(*p3)
    f3_interp = grid_trilinear_interp(p3, origin, res, dims, data_grid)
    assert f3 == f3_interp

    # interpolate arbitrary point
    # because f is linear, we should get exact results
    p3 = np.array([1.5, 2.5, 3.5])
    f3 = f(*p3)
    f3_interp = grid_trilinear_interp(p3, origin, res, dims, data_grid)
    assert f3 == f3_interp

    # interpolate a point outside the bound
    p4 = np.array([-1.5, 2.5, 3.5])
    f4 = f(*p4)
    try:
        f4_interp = grid_trilinear_interp(p4, origin, res, dims, data_grid)
    except AssertionError:
        assert True


def test_trilinear_interp_unequal_res():
    """Test trilinear interpolation on grid with unequal resolution on x, y and z"""
    logging.basicConfig(level=logging.DEBUG)

    # generate test data
    def f(x, y, z):
        return 3 * x + 2 * y + z

    origin = np.array([0, 0, 0])
    dims = np.array((11, 11, 11))
    x = np.linspace(origin[0], 10, dims[0])
    y = np.linspace(origin[1], 20, dims[1])
    z = np.linspace(origin[2], 30, dims[2])
    res = np.array((x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
    data_grid = f(xg, yg, zg)

    # ensure interpolation result equals to original value at vertices
    p1 = np.array([0, 0, 0])
    f1 = f(*p1)
    f1_interp = grid_trilinear_interp(p1, origin, res, dims, data_grid)
    assert f1 == f1_interp

    p2 = np.array([10, 20, 30])
    f2 = f(*p2)
    f2_interp = grid_trilinear_interp(p2, origin, res, dims, data_grid)
    assert f2 == f2_interp

    # interpolate arbitrary point
    # because f is linear, we should get exact results
    p3 = np.array([1.5, 2.5, 3.5])
    f3 = f(*p3)
    f3_interp = grid_trilinear_interp(p3, origin, res, dims, data_grid)
    assert f3 == f3_interp

    return


def test_grid_trilinear_interp_torch_vertices():
    """Test trilinear interpolation (torch) on grid vertices"""
    logging.basicConfig(level=logging.DEBUG)

    # generate test data
    def f(x, y, z):
        return 3 * x + 2 * y + z

    device = "cuda" if torch.cuda.is_available() else "cpu"
    origin = torch.tensor([0, 0, 0], device=device)
    dims = torch.tensor((11, 11, 11), device=device)
    x = torch.linspace(origin[0], 10, dims[0])
    y = torch.linspace(origin[1], 10, dims[1])
    z = torch.linspace(origin[2], 10, dims[2])
    res = torch.tensor((x[1] - x[0], y[1] - y[0], z[1] - z[0]), device=device)
    xg, yg, zg = torch.meshgrid(x, y, z, indexing="ij")
    data_grid = f(xg, yg, zg)
    data_grid = torch.tensor(data_grid, device=device)

    for i in range(x.size(dim=0)):
        for j in range(y.size(dim=0)):
            for k in range(z.size(dim=0)):
                p = torch.tensor([x[i], y[i], z[i]], device=device)
                f_v = f(p[0], p[1], p[2])
                f_interp = grid_trilinear_interp_exact_torch(p.reshape((1, 3, 1)), origin, res, dims, data_grid)
                assert f_v == f_interp

    return


def test_grid_trilinear_interp_torch_unequal_res():
    """Test trilinear interpolation (torch) on grid with unequal resolution on x, y and z"""
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

    # ensure interpolation result equals to original value at vertices
    p1 = torch.tensor([0, 0, 0], device=device)
    f1 = f(*p1)
    f1_interp = grid_trilinear_interp_exact_torch(p1.reshape((1, 3, 1)), origin, res, dims, data_grid)
    assert f1 == f1_interp

    p2 = torch.tensor([10, 20, 30], device=device)
    f2 = f(*p2)
    f2_interp = grid_trilinear_interp_exact_torch(p2.reshape((1, 3, 1)), origin, res, dims, data_grid)
    assert f2 == f2_interp

    # interpolate arbitrary point
    # because f is linear, we should get exact results
    p3 = torch.tensor([1.5, 2.5, 3.5], device=device)
    f3 = f(*p3)
    f3_interp = grid_trilinear_interp_exact_torch(p3.reshape((1, 3, 1)), origin, res, dims, data_grid)
    assert f3 == f3_interp

    return


def test_trilinear_interp_torch():
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
    p2_1 = torch.tensor([9.9, 19.9, 29.9], device=device)
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
    f_interp = grid_trilinear_interp_torch(points, origin, res, dims, data_grid, outside_value=0.0)

    # check values
    assert torch.isclose(f1_1.float(), f_interp[0, 0, 0].float())
    assert torch.isclose(f2_1.float(), f_interp[0, 0, 1].float())
    assert torch.isclose(f3_1.float(), f_interp[0, 0, 2].float())
    assert torch.isclose(f4_1.float(), f_interp[0, 0, 3].float())

    ## first two in second batch are out of bounds
    assert torch.isclose(torch.zeros(1, device=device), f_interp[1, 0, 0].float())
    assert torch.isclose(torch.zeros(1, device=device), f_interp[1, 0, 1].float())
    assert torch.isclose(f3_2.float(), f_interp[1, 0, 2].float())
    assert torch.isclose(f4_2.float(), f_interp[1, 0, 3].float())


def test_depth_to_point_cloud_torch():
    """Unit test for depth_to_point_cloud_torch"""
    # create depth map
    h, w = 50, 50

    # all points = 0
    depth = torch.zeros((h, w))
    N = h * w
    K = torch.eye(3)
    pc = depth_to_point_cloud_torch(depth, K, pc_size=N)
    assert torch.allclose(pc, torch.zeros_like(pc), atol=1.0e-6)
    assert pc.shape[0] == 3 and pc.shape[1] == N

    # 1 point: (0,0,10), project to u,v = (0,0)
    N = 1
    d = 10
    pc_input = torch.tensor([0, 0, d], dtype=torch.float).reshape((3, 1))
    K = torch.eye(3)
    pc_proj = K @ pc_input
    pc_img = (pc_proj / pc_proj[2, :]).int()
    depth = torch.zeros((h, w))
    depth[pc_img[0].long(), pc_img[1].long()] = d
    pc = depth_to_point_cloud_torch(depth, K, pc_size=N)
    assert pc.shape[0] == 3 and pc.shape[1] == N
    assert torch.allclose(pc.float(), torch.tensor([[0], [0], [10]], dtype=torch.float))

    # 1 point with mask at 1 point
    N = 1
    d = 10
    pc_input = torch.tensor([0, 0, d], dtype=torch.float).reshape((3, 1))
    K = torch.eye(3)
    pc_proj = K @ pc_input
    pc_img = (pc_proj / pc_proj[2, :]).int()
    depth = torch.zeros((h, w))
    depth[pc_img[0].long(), pc_img[1].long()] = d
    mask = torch.zeros((h, w), dtype=torch.bool)
    mask[pc_img[0].long(), pc_img[1].long()] = 1
    pc = depth_to_point_cloud_torch(depth, K, mask=mask, pc_size=N)
    assert pc.shape[0] == 3 and pc.shape[1] == N
    assert torch.allclose(pc.float(), torch.tensor([[0], [0], [10]], dtype=torch.float))


def test_depth_to_point_cloud_batched():
    """Unit test for depth_to_point_cloud_batched"""
    # create depth map
    B = 5
    h, w = 50, 50

    # all points = 0
    depth = torch.zeros((B, h, w))
    N = h * w
    K = torch.eye(3).repeat(B, 1, 1)
    pc = depth_to_point_cloud_batched(depth, K, pc_size=N)
    assert pc.shape[0] == B and pc.shape[1] == 3 and pc.shape[2] == N
    assert torch.allclose(pc, torch.zeros_like(pc), atol=1.0e-6)

    # 1 point: (0,0,10), project to u,v = (0,0)
    N = 1
    d = 10
    pc_input = torch.tensor([0, 0, d], dtype=torch.float).reshape((3, 1)).repeat(B, 1, 1)
    K = torch.eye(3).repeat(B, 1, 1)
    pc_proj = K @ pc_input
    pc_img = torch.zeros_like(pc_proj)
    for b in range(B):
        pc_img[b, :, :] = pc_proj[b, :, :] / pc_proj[b, 2, :]
    depth = torch.zeros((B, h, w))
    for b in range(B):
        depth[b, pc_img[b, 0].long(), pc_img[b, 1].long()] = d
    pc = depth_to_point_cloud_batched(depth, K, pc_size=N)
    assert pc.shape[0] == B and pc.shape[1] == 3 and pc.shape[2] == N
    assert torch.allclose(pc.float(), torch.tensor([[0], [0], [10]], dtype=torch.float).repeat(B, 1, 1))

    # 1 point with mask at 1 point
    N = 1
    d = 10
    pc_input = torch.tensor([0, 0, d], dtype=torch.float).reshape((3, 1)).repeat(B, 1, 1)
    K = torch.eye(3).repeat(B, 1, 1)
    pc_proj = K @ pc_input
    pc_img = torch.zeros_like(pc_proj)
    for b in range(B):
        pc_img[b, :, :] = pc_proj[b, :, :] / pc_proj[b, 2, :]

    depth = torch.zeros((B, h, w))
    for b in range(B):
        depth[b, pc_img[b, 0].long(), pc_img[b, 1].long()] = d

    mask = torch.zeros((B, h, w), dtype=torch.bool)
    for b in range(B):
        mask[b, pc_img[b, 0].long(), pc_img[b, 1].long()] = 1

    pc = depth_to_point_cloud_batched(depth, K, pc_size=N, mask=mask)
    assert pc.shape[0] == B and pc.shape[1] == 3 and pc.shape[2] == N
    assert torch.allclose(pc.float(), torch.tensor([[0], [0], [10]], dtype=torch.float).repeat(B, 1, 1))
