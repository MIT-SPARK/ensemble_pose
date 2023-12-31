"""
This implements model generation modules.

Note:
    Given shape parameters, model generation modules output a model corresponding to that shape.
"""

import torch
import torch.nn as nn


# CASPER-3D
class ModelSDFwShape():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models      : list of K ModelSDF class instances
        model_keypoints : torch.tensor of shape (K, 3, N)

        where 
        K = number of cad models
        n = number of points in each cad model

        Assumption:
        I am assuming that each CAD model is src.casper3d.model_sdf.ModelSDF class instance
        I am assuming that the intermediate shape can be obtained from 
            the shape parameter c and the cad models by weighted averaging 
        """
        self.cad_models = cad_models
        # self.model_keypoints = model_keypoints
        # self.cad_models = cad_models.unsqueeze(0)  # (1, K, 3, n)
        self.model_keypoints = model_keypoints.unsqueeze(0)  # (1, K, 3, N)

    def forward(self, shape, inputs):
        """

        Args:
            shape: torch.tensor of shape (B, K, 1)
            inputs: torch.tensor of shape (B, 3, m)

        where
            B = batch size
            K = number of cad models

        Returns:
            model_keypoints:    torch.tensor of shape (B, 3, N)
            sdf:                torch.tensor of shape (B, 1, m)
        """
        #ToDo: To be verified.
        B, K, _ = shape.shape
        _, _, m = inputs.shape

        sdf = torch.zeros(B, K, 1, m).to(device=inputs.device)
        i = 0
        for cad_model in self.cad_models:
            sdf[:, i, ...] = cad_model.forward(inputs)

        shape = shape.unsqueeze(-1) # (B, K, 1, 1)
        # averaging model keypoints
        return torch.einsum('bkmn,ukij->bij', shape, self.model_keypoints), torch.einsum('bkmn,bkij->bij', shape, sdf)


# From C-3PO
class ModelFromShape():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (K, 3, n)
        model_keypoints : torch.tensor of shape (K, 3, N)

        where 
        K = number of cad models
        n = number of points in each cad model

        Assumption:
        I am assuming that each CAD model is a point cloud of n points: (3, n)
        I am assuming that the intermediate shape can be obtained from 
            the shape parameter c and the cad models 
            by just weighted average 
        """
        self.cad_models = cad_models.unsqueeze(0)  # (1, K, 3, n)
        self.model_keypoints = model_keypoints.unsqueeze(0)  # (1, K, 3, N)

    def forward(self, shape):
        """
        shape: torch.tensor of shape (B, K, 1)

        where
        B = batch size
        K = number of cad models

        intermediate:
        self.cad_models: torch.tensor of shape (1, K, 3, n)

        output:
        predicted_keypoints: torch.tensor of shape (B, 3, N)
        predicted_model: torch.tensor of shape (B, 3, n)
        """
        shape = shape.unsqueeze(-1) # (B, K, 1, 1)
        # averaging model keypoints
        return torch.einsum('bkmn,ukij->bij', shape, self.model_keypoints), torch.einsum('bkmn,ukij->bij', shape, self.cad_models)


# From C-3PO
class ModelFromShapeModule(nn.Module):
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (K, 3, n)
        model_keypoints : torch.tensor of shape (K, 3, N)

        where 
        K = number of cad models
        n = number of points in each cad model

        Assumption:
        I am assuming that each CAD model is a point cloud of n points: (3, n)
        I am assuming that the intermediate shape can be obtained from 
            the shape parameter c and the cad models 
            by just weighted average 
        """

        self.model_from_shape = ModelFromShape(cad_models=cad_models, model_keypoints=model_keypoints)


    def forward(self, shape):
        """
        shape: torch.tensor of shape (B, K, 1)

        where
        B = batch size
        K = number of cad models

        intermediate:
        self.cad_models: torch.tensor of shape (1, K, 3, n)

        output:
        keypoints: torch.tensor of shape (B, 3, N)
        model: torch.tensor of shape (B, 3, n)
        """

        return self.model_from_shape.forward(shape=shape)





if __name__ == "__main__":

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    B = 10
    K = 5
    N = 8
    n = 100
    cad_models = torch.rand(K, 3, n).to(device=device)
    model_keypoints = cad_models[:, :, 0:N]

    shape = torch.rand(B, K, 1).to(device=device)
    shape = shape/shape.sum(1).unsqueeze(1)

    shape_to_model_fn = ModelFromShapeModule(cad_models=cad_models, model_keypoints=model_keypoints).to(device=device)
    keypoints, model = shape_to_model_fn(shape=shape)

    print("cad models have shape: ", cad_models[0, :, :].shape)
    print("output model has shape: ", model.shape)
    print("output keypoints has shape: ", keypoints.shape)