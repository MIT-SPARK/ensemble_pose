from utils.loss_functions import *


def test_chamfer_loss_simple():
    """Testing the chamfer loss function"""
    # same point, expect distance to be zero
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p1 = torch.tensor([0, 0, 0], device=device, dtype=torch.float)
    loss = chamfer_loss(p1.reshape((1, 3, 1)), p1.reshape((1, 3, 1)), pc_padding=torch.tensor([[False]], device=device))
    loss = torch.squeeze(loss)
    assert torch.allclose(loss, torch.zeros(1, device=device))

    # two points, expect distance to be one
    p1 = torch.tensor([0, 0, 0], device=device, dtype=torch.float)
    p2 = torch.tensor([1, 0, 0], device=device, dtype=torch.float)
    loss = chamfer_loss(p1.reshape((1, 3, 1)), p2.reshape((1, 3, 1)), pc_padding=torch.tensor([[False]], device=device))
    loss = torch.squeeze(loss)
    assert torch.allclose(loss, torch.ones(1, device=device))

    # two points in each point cloud, expect distance to be one
    pc = torch.tensor([[0, 0, 0], [0, 0, 0]], device=device, dtype=torch.float)
    pc_ = torch.tensor([[1, 0, 0], [1, 0, 0]], device=device, dtype=torch.float)
    loss = chamfer_loss(
        pc.reshape((1, 3, 2)), pc_.reshape((1, 3, 2)), pc_padding=torch.tensor([[False, False]], device=device)
    )
    loss = torch.squeeze(loss)
    assert torch.allclose(loss, torch.ones(1, device=device))

    return


def test_chamfer_loss_multibatch():
    """Testing the chamfer loss function with multiple batches"""
    # two points in each point cloud, two batches expect distance to be one
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc1 = torch.tensor([[0.1, 0, 0], [0.1, 0, 0]], device=device, dtype=torch.float)
    pc2 = torch.tensor([[0.1, 0, 0], [0.1, 0, 0]], device=device, dtype=torch.float)
    pc = torch.cat((pc1.reshape((1, 3, 2)), pc2.reshape((1, 3, 2))), dim=0)

    pc1_ = torch.tensor([[1, 0, 0], [1, 0, 0]], device=device, dtype=torch.float)
    pc2_ = torch.tensor([[2, 0, 0], [2, 0, 0]], device=device, dtype=torch.float)
    pc_ = torch.cat((pc1_.reshape((1, 3, 2)), pc2_.reshape((1, 3, 2))), dim=0)

    # L2 loss version
    loss = chamfer_loss(pc, pc_)

    # max loss version
    loss_max_loss = chamfer_loss(pc, pc_, max_loss=True)
    assert loss.shape == loss_max_loss.shape


def test_clamped_chamfer_loss_simple():
    """Testing the clamped chamfer loss function"""
    # same point, expect distance to be zero
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p1 = torch.tensor([0, 0, 0], device=device, dtype=torch.float)
    loss = half_chamfer_loss_clamped(
        p1.reshape((1, 3, 1)), p1.reshape((1, 3, 1)), 0.1, pc_padding=torch.tensor([[False]], device=device)
    )
    loss = torch.squeeze(loss)
    assert torch.allclose(loss, torch.zeros(1, device=device))

    # two points, expect distance to be one
    p1 = torch.tensor([0, 0, 0], device=device, dtype=torch.float)
    p2 = torch.tensor([1, 0, 0], device=device, dtype=torch.float)
    loss = half_chamfer_loss_clamped(
        p1.reshape((1, 3, 1)), p2.reshape((1, 3, 1)), 10, pc_padding=torch.tensor([[False]], device=device)
    )
    loss = torch.squeeze(loss)
    assert torch.allclose(loss, torch.ones(1, device=device))

    # two points in each point cloud, expect distance to be one
    pc = torch.tensor([[0, 0, 0], [0, 0, 0]], device=device, dtype=torch.float)
    pc_ = torch.tensor([[1, 0, 0], [1, 0, 0]], device=device, dtype=torch.float)

    loss = half_chamfer_loss_clamped(
        pc.reshape((1, 3, 2)), pc_.reshape((1, 3, 2)), 10, pc_padding=torch.tensor([[False]], device=device)
    )
    loss = torch.squeeze(loss)
    assert torch.allclose(loss, torch.ones(1, device=device))

    # thresholding, expect distance to be one
    pc = torch.tensor([[0, 0, 0], [0, 0, 0]], device=device, dtype=torch.float)
    pc_ = torch.tensor([[1, 0, 0], [45, 0, 0]], device=device, dtype=torch.float)

    loss = half_chamfer_loss_clamped(
        pc.reshape((1, 3, 2)), pc_.reshape((1, 3, 2)), 10, pc_padding=torch.tensor([[False]], device=device)
    )
    loss = torch.squeeze(loss)
    assert torch.allclose(loss, torch.ones(1, device=device))
