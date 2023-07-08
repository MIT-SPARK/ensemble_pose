# credit: https://github.com/itailang/SampleNet/
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from utils.math_utils import kNN_torch


class SoftProjection(nn.Module):
    def __init__(
        self,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-4,
    ):
        """Computes a soft nearest neighbor point cloud.
        Arguments:
            group_size: An integer, number of neighbors in nearest neighborhood.
            initial_temperature: A positive real number, initialization constant for temperature parameter.
            is_temperature_trainable: bool.
        Inputs:
            point_cloud: A `Tensor` of shape (batch_size, 3, num_orig_points), database point cloud.
            query_cloud: A `Tensor` of shape (batch_size, 3, num_query_points), query items to project or propogate to.
            point_features [optional]: A `Tensor` of shape (batch_size, num_features, num_orig_points), features to propagate.
            action [optional]: 'project', 'propagate' or 'project_and_propagate'.
        Outputs:
            Depending on 'action':
            propagated_features: A `Tensor` of shape (batch_size, num_features, num_query_points)
            projected_points: A `Tensor` of shape (batch_size, 3, num_query_points)
        """

        super().__init__()
        self._group_size = group_size

        # create temperature variable
        self._temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                requires_grad=is_temperature_trainable,
                dtype=torch.float32,
            )
        )

        self._min_sigma = torch.tensor(min_sigma, dtype=torch.float32)

    def forward(self, point_cloud, query_cloud, point_features=None, action="project"):
        point_cloud = point_cloud.contiguous()
        query_cloud = query_cloud.contiguous()

        if action == "project":
            return self.project(point_cloud, query_cloud)
        elif action == "propagate":
            return self.propagate(point_cloud, point_features, query_cloud)
        elif action == "project_and_propagate":
            return self.project_and_propagate(point_cloud, point_features, query_cloud)
        else:
            raise ValueError("action should be one of the following: 'project', 'propagate', 'project_and_propagate'")

    def _group_points(self, point_cloud, query_cloud, point_features=None):
        group_size = self._group_size

        # find nearest group_size neighbours in point_cloud
        # idx shape: Batch, QueryPoints, k
        idx = kNN_torch(query_cloud, point_cloud, group_size)

        # self._dist = dist.unsqueeze(1).permute(0, 1, 3, 2) ** 2

        idx = idx.permute(0, 2, 1).type(torch.int32)  # index should be Batch x QueryPoints x K
        grouped_points = group_point(point_cloud, idx)  # B x 3 x QueryPoints x K
        grouped_features = (
            None if point_features is None else group_point(point_features, idx)
        )  # B x F x QueryPoints x K
        return grouped_points, grouped_features

    def _get_distances(self, grouped_points, query_cloud):
        deltas = grouped_points - query_cloud.unsqueeze(-1).expand_as(grouped_points)
        #dist = torch.sum(deltas**2, dim=_axis_to_dim(3), keepdim=True) / self.sigma()
        dist = torch.sum(deltas**2, dim=1, keepdim=True) / self.sigma()
        return dist

    def sigma(self):
        device = self._temperature.device
        return torch.max(self._temperature**2, self._min_sigma.to(device))

    def project_and_propagate(self, point_cloud, point_features, query_cloud):
        # group into (batch_size, num_query_points, group_size, 3),
        # (batch_size, num_query_points, group_size, num_features)
        grouped_points, grouped_features = self._group_points(point_cloud, query_cloud, point_features)
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        #weights = torch.softmax(-dist, dim=_axis_to_dim(2))
        weights = torch.softmax(-dist, dim=3)

        # get weighted average of grouped_points
        projected_points = torch.sum(
            #grouped_points * weights, dim=_axis_to_dim(2)
            grouped_points * weights, dim=3
        )  # (batch_size, num_query_points, num_features)
        propagated_features = torch.sum(
            #grouped_features * weights, dim=_axis_to_dim(2)
            grouped_features * weights, dim=3
        )  # (batch_size, num_query_points, num_features)

        return (projected_points, propagated_features)

    def propagate(self, point_cloud, point_features, query_cloud):
        grouped_points, grouped_features = self._group_points(point_cloud, query_cloud, point_features)
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        #weights = torch.softmax(-dist, dim=_axis_to_dim(2))
        weights = torch.softmax(-dist, dim=3)

        # get weighted average of grouped_points
        propagated_features = torch.sum(
            #grouped_features * weights, dim=_axis_to_dim(2)
            grouped_features * weights, dim=3
        )  # (batch_size, num_query_points, num_features)

        return propagated_features

    def project(self, point_cloud, query_cloud, hard=False):
        grouped_points, _ = self._group_points(point_cloud, query_cloud)
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        #weights = torch.softmax(-dist, dim=_axis_to_dim(2))
        weights = torch.softmax(-dist, dim=3)
        if hard:
            raise NotImplementedError

        # get weighted average of grouped_points
        weights = weights.repeat(1, 3, 1, 1)
        projected_points = torch.sum(
            #grouped_points * weights, dim=_axis_to_dim(2)
            grouped_points * weights, dim=3
        )  # (batch_size, num_query_points, num_features)
        return projected_points


class SampleNet(nn.Module):
    def __init__(
        self,
        num_out_points,
        bottleneck_size,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-2,
        input_shape="bcn",
        output_shape="bcn",
        complete_fps=True,
        skip_projection=False,
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.name = "samplenet"

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, bottleneck_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(bottleneck_size)

        self.fc1 = nn.Linear(bottleneck_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3 * num_out_points)

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)

        # projection and matching
        self.project = SoftProjection(group_size, initial_temperature, is_temperature_trainable, min_sigma)
        self.skip_projection = skip_projection
        self.complete_fps = complete_fps

        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError("allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' ")
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError("allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' ")
        if input_shape != output_shape:
            warnings.warn("SampleNet: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))  # Batch x 128 x NumInPoints

        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 128

        y = F.relu(self.bn_fc1(self.fc1(y)))
        y = F.relu(self.bn_fc2(self.fc2(y)))
        y = F.relu(self.bn_fc3(self.fc3(y)))
        y = self.fc4(y)

        y = y.view(-1, 3, self.num_out_points)

        # Simplified points
        simp = y
        match = None
        proj = None

        # Projected points
        if self.training:
            if not self.skip_projection:
                proj = self.project(point_cloud=x, query_cloud=y)
            else:
                proj = simp

        # Matched points
        else:  # Inference
            # Retrieve nearest neighbor indices
            _, idx = KNN(1, transpose_mode=False)(x.contiguous(), y.contiguous())

            """Notice that we detach the tensors and do computations in numpy,
            and then convert back to Tensors.
            This should have no effect as the network is in eval() mode
            and should require no gradients.
            """

            # Convert to numpy arrays in B x N x 3 format. we assume 'bcn' format.
            x = x.permute(0, 2, 1).cpu().detach().numpy()
            y = y.permute(0, 2, 1).cpu().detach().numpy()

            idx = idx.cpu().detach().numpy()
            idx = np.squeeze(idx, axis=1)

            z = sputils.nn_matching(x, idx, self.num_out_points, complete_fps=self.complete_fps)

            # Matched points are in B x N x 3 format.
            match = torch.tensor(z, dtype=torch.float32).cuda()

        # Change to output shapes
        if self.output_shape == "bnc":
            simp = simp.permute(0, 2, 1)
            if proj is not None:
                proj = proj.permute(0, 2, 1)
        elif self.output_shape == "bcn" and match is not None:
            match = match.permute(0, 2, 1)
            match = match.contiguous()

        # Assert contiguous tensors
        simp = simp.contiguous()
        if proj is not None:
            proj = proj.contiguous()
        if match is not None:
            match = match.contiguous()

        out = proj if self.training else match

        return simp, out

    def sample(self, x):
        simp, proj, match, feat = self.__call__(x)
        return proj

    # Losses:
    # At inference time, there are no sampling losses.
    # When evaluating the model, we'd only want to asses the task loss.

    #def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
    #    if self.skip_projection or not self.training:
    #        return torch.tensor(0).to(ref_pc)
    #    # ref_pc and samp_pc are B x N x 3 matrices
    #    cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
    #    max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
    #    max_cost = torch.mean(max_cost)
    #    cost_p1_p2 = torch.mean(cost_p1_p2)
    #    cost_p2_p1 = torch.mean(cost_p2_p1)
    #    loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
    #    return loss

    def get_projection_loss(self):
        sigma = self.project.sigma()
        if self.skip_projection or not self.training:
            return torch.tensor(0).to(sigma)
        return sigma


if __name__ == "__main__":
    point_cloud = np.random.randn(1, 3, 1024)
    point_cloud_pl = torch.tensor(point_cloud, dtype=torch.float32).cuda()
    net = SampleNet(5, 128, group_size=10, initial_temperature=0.1, complete_fps=True)

    net.cuda()
    net.eval()

    for param in net.named_modules():
        print(param)

    simp, proj, match = net.forward(point_cloud_pl)
    simp = simp.detach().cpu().numpy()
    proj = proj.detach().cpu().numpy()
    match = match.detach().cpu().numpy()

    print("*** SIMPLIFIED POINTS ***")
    print(simp)
    print("*** PROJECTED POINTS ***")
    print(proj)
    print("*** MATCHED POINTS ***")
    print(match)

    mse_points = np.sum((proj - match) ** 2, axis=1)
    print("projected points vs. matched points error per point:")
    print(mse_points)
