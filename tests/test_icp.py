import torch

from casper3d import icp


def test_icp_simple():
    """ Some simple transformation estimation task for icp """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 5
    N = 1000

    # Identity transformation
    src = torch.rand((B, 3, N), device=device)
    target = torch.clone(src)

    icp_solver = icp.ICP(100, 1e-6, 10, 'svd', 'nn', 'point', True)
    R, t, mse = icp_solver.forward(src, target, pc_dest_is_valid_mask=torch.ones((5, 1000), dtype=torch.bool))
    for b in range(B):
        assert torch.allclose(R[b, :, :], torch.eye(3, device=device), atol=1e-3)
        assert torch.allclose(t[b, :, :], torch.zeros((3,), device=device), atol=1e-3)

    # Translation only
    # TODO: Use roma's geodesic distances
    t_gt = torch.reshape(torch.tensor([10, -5, 4], dtype=torch.float, device=device), (3, 1))
    target = src + t_gt
    R, t, mse = icp_solver.forward(src, target, pc_dest_is_valid_mask=torch.ones((5, 1000), dtype=torch.bool))
    for b in range(B):
        assert torch.allclose(R[b, :, :], torch.eye(3, device=device), atol=1e-2)
        assert torch.allclose(t[b, :, :].squeeze(), t_gt.squeeze(), atol=1e-2)
