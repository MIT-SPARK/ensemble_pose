import torch
import torch.nn as nn


class GumbelSubsetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k, sampling_ratio):
        """ A learned downsampling layer based on Gumbel Softmax.

        Takes in features & xyz, and run
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.sampling_ratio = sampling_ratio
        # TODO: Add linear layer

    def forward(self, x, p):
        """
        inputs
            x: (B, N, in_channels) shaped torch Tensor (A set of feature vectors)
            p: (B, N, 3) shaped torch Tensor (3D coordinates)
        outputs
            y: (B, M, out_channels) shaped torch Tensor
            p_out: (B, M, 3) shaped torch Tensor
        M = N * sampling ratio
        """
        raise NotImplementedError


class GumbelSubsetOperator(torch.nn.Module):
    def __init__(self, k, tau=1.0, eps=1e-10, hard=False):
        """
        Credit: https://github.com/ermongroup/subsets/
        Args:
            k:
            tau:
            eps:
            hard:
        """
        super(GumbelSubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau
        self.eps = eps

    def forward(self, scores):
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([self.eps]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # will do straight through estimation if training
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res
