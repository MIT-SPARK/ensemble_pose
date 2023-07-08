import torch.cuda

from casper3d.robust_centroid import *


def test_weighted_mean_1():
    # same points for all points in the pc
    B = 3
    N = 100

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1)
        test_pc = exp_mean.repeat(1, 1, N)

        weights = torch.ones(B, 1, N)
        actual_mean = weighted_mean(weights, test_pc)
        assert torch.allclose(actual_mean, exp_mean)


def test_weighted_mean_2():
    # same points for all points in the pc, different weights
    B = 3
    N = 100

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1)
        test_pc = exp_mean.repeat(1, 1, N)

        # select two points to set to outliers
        test_pc[0, 0, 0] = 100
        test_pc[0, 1, 0] = 200
        test_pc[0, 0, 1] = 100
        test_pc[0, 1, 1] = 200

        test_pc[1, 0, 0] = 100
        test_pc[1, 1, 0] = 200

        # disable outlier weights
        weights = torch.ones(B, 1, N)

        weights[0, 0, 0] = 0
        weights[0, 0, 1] = 0
        weights[1, 0, 0] = 0

        actual_mean = weighted_mean(weights, test_pc)
        assert torch.allclose(actual_mean, exp_mean)


def test_weighted_objective_1():
    # objective of the gt point should be close to zero
    B = 3
    N = 100

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1)
        test_pc = exp_mean.repeat(1, 1, N)

        weights = torch.ones(B, 1, N)
        c, r, sq_r = weighted_objective(exp_mean, weights, test_pc)

        assert torch.allclose(c, torch.tensor(0.0))
        assert torch.allclose(r, torch.tensor(0.0))
        assert torch.allclose(sq_r, torch.tensor(0.0))


def test_weighted_objective_2():
    # test weighted cost: if we set weights of outliers to zero the cost should still be zero
    B = 3
    N = 5

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1)
        test_pc = exp_mean.repeat(1, 1, N)

        # select two points to set to outliers
        test_pc[0, 0, 0] = 100
        test_pc[0, 1, 0] = 200
        test_pc[0, 0, 1] = 100
        test_pc[0, 1, 1] = 200

        test_pc[1, 0, 0] = 100
        test_pc[1, 1, 0] = 200

        # disable outlier weights
        weights = torch.ones(B, 1, N)

        weights[0, 0, 0] = 0
        weights[0, 0, 1] = 0
        weights[1, 0, 0] = 0

        c, r, sq_r = weighted_objective(exp_mean, weights, test_pc)
        assert torch.allclose(c, torch.tensor(0.0))

        # make sure the residuals are corrector
        assert torch.allclose(r[0, :, 2:], torch.tensor(0.0))
        assert torch.allclose(r[1, :, 1:], torch.tensor(0.0))
        assert torch.allclose(r[2, ...], torch.tensor(0.0))

        # make sure the residuals are squared
        for b in range(B):
            assert torch.allclose(torch.square(r[b, ...]), sq_r[b, ...])


def test_robust_centroid_gnc_1():
    # with no outliers, the robust centroid should be equivalent to mean
    B = 3
    N = 5

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1)
        test_pc = exp_mean.repeat(1, 1, N)
        result = robust_centroid_gnc(test_pc, cost_type="gnc-tls", clamp_thres=0.1, max_iterations=50)
        assert torch.allclose(result["robust_centroid"], exp_mean)

        result = robust_centroid_gnc(test_pc, cost_type="gnc-gm", clamp_thres=0.1, max_iterations=50)
        assert torch.allclose(result["robust_centroid"], exp_mean)


def test_robust_centroid_gnc_2():
    # 1 outlier in a batch with only 1 point cloud
    B = 1
    N = 10

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1)
        test_pc = exp_mean.repeat(1, 1, N)

        # select two points to set to outliers
        # pc0
        test_pc[0, 0, 0] = 10
        test_pc[0, 1, 0] = 20

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-tls",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert act_exp_dist < 1e-4

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-gm",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert act_exp_dist < 1e-4


def test_robust_centroid_gnc_3():
    # a batch with 2 point clouds: one with zero outliers, one with 1 outlier
    B = 2
    N = 10

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1)
        test_pc = exp_mean.repeat(1, 1, N)

        # select two points to set to outliers
        # pc0 has one outlier
        # pc1 has no outliers
        # expectation is we should be able to recover the correct mean for each pc individually
        test_pc[0, 0, 0] = 10
        test_pc[0, 1, 0] = 20
        test_pc[0, 2, 0] = 30

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-tls",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert torch.all(act_exp_dist < 1e-4)

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-gm",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert torch.all(act_exp_dist < 1e-4)


def test_robust_centroid_gnc_4():
    # a batch with 5 point clouds: four with zero outliers, one with 2 outliers
    B = 5
    N = 100

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1)
        test_pc = exp_mean.repeat(1, 1, N)

        test_pc[2, :, 0] *= 10
        test_pc[2, :, -1] *= 10

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-tls",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert torch.all(act_exp_dist < 1e-4)

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-gm",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert torch.all(act_exp_dist < 1e-4)


def test_robust_centroid_gnc_5():
    # large point clouds
    B = 5
    N = 1000

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1)
        test_pc = exp_mean.repeat(1, 1, N)

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-tls",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert torch.all(act_exp_dist < 1e-4)

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-gm",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert torch.all(act_exp_dist < 1e-4)


def test_robust_centroid_gnc_6():
    # large point clouds on cuda
    B = 5
    N = 1000

    for i in range(100):
        exp_mean = torch.rand(B, 3, 1, device="cuda")
        test_pc = exp_mean.repeat(1, 1, N)

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-tls",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert torch.all(act_exp_dist < 1e-4)

        result = robust_centroid_gnc(
            test_pc,
            cost_type="gnc-gm",
            clamp_thres=0.1,
            max_iterations=50,
            cost_abs_stop_th=1e-5,
            cost_diff_stop_th=1e-6,
        )
        act_mean = result["robust_centroid"]
        act_exp_dist = torch.linalg.vector_norm(act_mean - exp_mean, dim=(1, 2))
        assert torch.all(act_exp_dist < 1e-4)
