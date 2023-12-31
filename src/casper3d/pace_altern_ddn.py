"""
This code is an attempt to implement ddn as a torch.autograd.Function.

Note:
    It uses Alternating Method in computing the rotation and shape parameter.

Reference:
[1] Jingnan Shi, Heng Yang, and Luca Carlone "Optimal Pose and Shape Estimation for Category-level 3D Object
    Perception" RSS 2021.

"""
# import cvxpy as cp
# import numpy as np
# import os
# import pymanopt as pym
import sys
import time
import torch
# import torch.nn as nn
import torch.nn.functional as F
# from cvxpylayers.torch import CvxpyLayer

sys.path.append("../")

from src.casper3d.point_set_registration import wahba
# from src.utils.ddn.node import EqConstDeclarativeNode, ParamDeclarativeFunction, AbstractDeclarativeNode
from src.utils.general import generate_random_keypoints
from src.utils.evaluation_metrics import shape_error, translation_error, rotation_euler_error


class PACErotationNshape():
    def __init__(self, bar_B, lambda_constant=torch.tensor([1.0]), batch_size=32):
        super().__init__()
        """
        bar_B   : torch.tensor of shape (1, 3N, K)  : see bar_B in (10) in [1].
        
        """
        self.b = batch_size
        self.bar_B = bar_B                          # (1, 3N, K)
        self.lambda_constant = lambda_constant
        self.K = self.bar_B.shape[-1]
        self.N = int(self.bar_B.shape[-2]/3)
        self.device_ = bar_B.device

        self.A = self._getA()
        self.d = self._getd()
        self.G, self.g = self._getGg()              # G: (1, K, K), g: (1, K, 1)

        # Alternating Method parameters
        self.method_altern_tol = 1e-12
        self.altern_iter_max = 100

    def objective(self, bar_y, y):
        """
        inputs:
        bar_y   : torch.tensor of shape (B, 3*N)    : normalized keypoints. see (9) in [1].
        y       : torch.tensor of shape (B, 9+K)    : y[b, 0:9] : vectorized rotation matrix
                                                    : y[b, 9:] : shape vector

        outputs:
        objective value : torch.tensor of shape (B, 1)  : see (11) in [1]

        """
        # self._test(bar_y, y)

        batch_size = bar_y.shape[0]

        R, c = self._ytoRc(y)   # R: (B, 3, 3), c: (B, K, 1)

        temp1 = self.bar_B @ c
        temp2 = torch.transpose(R, -1, -2) @ torch.transpose(bar_y.reshape(batch_size, -1, 3), -1, -2)
        temp2 = torch.transpose(temp2, -1, -2).reshape(batch_size, -1).unsqueeze(-1)

        # print("Shape of temp1 ", temp1.shape)
        # print("Shape of temp2 ", temp2.shape)

        obj1 = F.mse_loss(temp1, temp2, reduction='none')             # (B, 3*N, 1)
        obj1 = obj1.mean(1)                                                             # (B, 1)
        obj2 = F.mse_loss(c, torch.zeros_like(c), reduction='none')   # (B, K, 1)
        obj2 = obj2.mean(1)                                                             # (B, 1)

        return obj1 + self.lambda_constant*obj2

    def forward(self, bar_y):
        """
        inputs:
        bar_y   : torch.tensor of shape (B, 3*N)    : normalized keypoints. see (9) in [1].

        outputs:
        y       : torch.tensor of shape (B, 9+K)    : y[b, 0:9] : vectorized rotation matrix
                                                    : y[b, 9:] : shape vector

        NOTE: This only mimics solve method. We are keeping this till we implement the .gradient() correctly.

        """
        y, _ = self.solve(bar_y)

        return y

    def solve(self, bar_y):
        """
        inputs:
        bar_y   : torch.tensor of shape (B, 3*N)    : normalized keypoints. see (9) in [1].

        outputs:
        y       : torch.tensor of shape (B, 9+K)    : y[b, 0:9] : vectorized rotation matrix
                                                    : y[b, 9:] : shape vector

        """
        R, c = self._method_altern(bar_y)
        y = self._Rctoy(R, c)

        return y, None

    def gradient(self, bar_y, y=None, v=None, ctx=None):

        # g = self._gradient_method1(bar_y=bar_y, y=y, v=v, ctx=ctx)
        g = self._gradient_method2(bar_y=bar_y, y=y, v=v, ctx=ctx)
        return g

    def _gradient_method2(self, bar_y, y=None, v=None, ctx=None):
        # ToDo: to be tested.
        """
        inputs:
        bar_y   : torch.tensor of shape (B, 3*N)    : normalized keypoints. see (9) in [1].
        y       : torch.tensor of shape (B, 9+K)    : y[b, 0:9] : vectorized rotation matrix
                                                    : y[b, 9:] : shape vector
        v       : torch.tensor of shape (B, 9+K)    : grad_output with respect to loss, used during backprop()

        outputs:
        grad_input   : torch.tensor of shape (B, 3*N)

        """

        if v==None:
            v = torch.ones_like(y)

        def f(x):
            y, _ = self.solve(x)
            return y

        Jf = torch.autograd.functional.jacobian(f, (bar_y,))

        # print("Jf lenght: ", len(Jf))
        # print("Jf shape: ", Jf[0].shape)

        grad_input = torch.einsum('bobi,bo->bi', Jf[0], v)

        return (grad_input,)

    def _gradient_method1(self, bar_y, y=None, v=None, ctx=None):
        #ToDo: Implement the explicit gradient method here.
        """
        inputs:
        bar_y   : torch.tensor of shape (B, 3*N)    : normalized keypoints. see (9) in [1].
        y       : torch.tensor of shape (B, 9+K)    : y[b, 0:9] : vectorized rotation matrix
                                                    : y[b, 9:] : shape vector
        v       : torch.tensor of shape (B, 9+K)    : grad_output with respect to loss, used during backprop()

        outputs:
        grad_input   : torch.tensor of shape (B, 3*N)

        """
        batch_size = bar_y.shape[0]
        if y==None:
            y, _ = self.solve(bar_y=bar_y)

        if v==None:
            v = torch.ones_like(y)

        R, c = self._ytoRc(y)       # R: (B, 3, 3), c: (B, K, 1)

        def f(x, y, z):
            """
            x   : torch.tensor of shape (3*N)
            y   : torch.tensor of shape (3, 3)
            z   : torch.tensor of shape (K, 1)

            output:
            loss    : torch.tensor of shape (1,)
            """
            loss = self._objective(bar_y=x.unsqueeze(0), R=y.unsqueeze(0), c=z.unsqueeze(0))

            return loss.squeeze(0)

        grad_input = torch.zeros_like(bar_y)
        for b in range(batch_size):

            batch_bar_y = bar_y[b, ...]
            batch_R = R[b, ...]
            batch_c = c[b, ...]

            Hf = torch.autograd.functional.hessian(f, (batch_bar_y, batch_R, batch_c))

            # print("Shape of Hf[0][0]: ", Hf[0][0].shape)
            # print("Shape of Hf[0][1]: ", Hf[0][1].shape)
            # print("Shape of Hf[0][2]: ", Hf[0][2].shape)
            # print("Shape of Hf[1][0]: ", Hf[1][0].shape)
            # print("Shape of Hf[1][1]: ", Hf[1][1].shape)
            # print("Shape of Hf[1][2]: ", Hf[1][2].shape)
            # print("Shape of Hf[2][0]: ", Hf[2][0].shape)
            # print("Shape of Hf[2][1]: ", Hf[2][1].shape)
            # print("Shape of Hf[2][2]: ", Hf[2][2].shape)

            fyR = Hf[0][1].reshape(3 * self.N, 9)
            fyc = Hf[0][2].reshape(3 * self.N, self.K)
            fRR = Hf[1][1].reshape(9, 9)
            fRc = Hf[1][2].reshape(9, self.K)
            fcR = Hf[2][1].reshape(self.K, 9)
            fcc = Hf[2][2].reshape(self.K, self.K)

            Amat = torch.zeros(9 + 1 + self.K + 15, 9 + self.K).to(device=self.device_)
            Bmat = torch.zeros(9 + 1 + self.K + 15, 3 * self.N).to(device=self.device_)

            # Amat[0:9, 0:9] = fRR
            Amat[0:9, 9:] = fRc
            Amat[9:10, 9:] = torch.ones(self.K).to(device=self.device_)
            Amat[10:10 + self.K, :9] = fcR
            Amat[10:10 + self.K, 9:] = fcc

            r = batch_R.T.reshape(9, 1)
            A = self._getA()
            for i in range(15):
                Amat[10 + self.K + i, 0:9] = r.T @ A[i, 1:, 1:]

            Bmat[:9, :] = -fyR.T
            Bmat[10:10 + self.K, :] = -fyc.T

            print("Amat shape: ", Amat.shape)
            print("Bmat shape: ", Bmat.shape)
            print("Amat rank: ", torch.matrix_rank(Amat))
            # Solve Amat X = Bmat
            X = torch.lstsq(Bmat, Amat).solution[:Amat.size(1)]
            # print("X shape: ", X.shape)
            # DR = X[:9, :]
            # Dc = X[9:, :]
            # print("DR shape: ", DR.shape)
            # print("Dc shape: ", Dc.shape)
            # print("DR")
            # print(DR)
            # print("Dc")
            # print(Dc)
            grad_input[b, :] = X.T @ v[b, :]

        return (grad_input,)

    def _method_altern(self, bar_y):
        """
        inputs:
        bar_y   : torch.tensor of shape (B, 3*N)    : normalized keypoints. see (9) in [1].


        outputs:
        R   : torch.tensor of shape (B, 3, 3)
        c   : torch.tensor of shape (B, self.K, 1)

        """

        batch_size = bar_y.shape[0]

        # initialize R          # (B, 3, 3)
        R = torch.eye(3, device=self.device_).unsqueeze(0)
        R = R.repeat(batch_size, 1, 1)
        R_final = torch.eye(3, device=self.device_).unsqueeze(0)
        # R_final = R_final.repeat(batch_size, 1, 1)

        # initialize c          # (B, K, 1)
        c = torch.ones((batch_size, self.K, 1), device=self.device_)/self.K
        # c_final = torch.zeros((batch_size, self.K, 1), device=self.device_)

        #initialize convergence flag    # (B, 1)
        test_convergence = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device_)

        iter = 0
        obj_ = self._objective(bar_y, R, c)

        # while iter < self.altern_iter_max:
        while not test_convergence.prod().to(dtype=torch.bool):

            iter += 1
            obj = obj_

            flag = test_convergence.unsqueeze(-1)

            c = torch.logical_not(flag.repeat(1, self.K, 1))*self._altern_compute_c(bar_y, R) + \
                flag.repeat(1, self.K, 1)*c
            # c = self._altern_compute_c(bar_y, R)

            R = torch.logical_not(flag.repeat(1, 3, 3))*self._altern_compute_R(bar_y, c) + flag.repeat(1, 3, 3)*R
            # R = self._altern_compute_R(bar_y, c)

            obj_ = self._objective(bar_y, R, c)
            test_convergence = self._altern_test_convergence(obj, obj_)

            if iter > self.altern_iter_max:
                # ("Alternating Method reached max. iterations for ", test_convergence.sum(), '/', batch_size)
                break

        return R, c

    def _objective(self, bar_y, R, c):

        y = self._Rctoy(R, c)
        obj = self.objective(bar_y, y)

        return obj

    def _altern_compute_R(self, bar_y, c):

        batch_size = bar_y.shape[0]

        source_points = torch.transpose((self.bar_B @ c).reshape(batch_size, self.N, 3), -1, -2)
        target_points = torch.transpose(bar_y.reshape(batch_size, -1, 3), -1, -2)

        R = wahba(source_points=source_points, target_points=target_points) # (B, 3, 3)
        # print("Shape of R: ", R.shape)
        return R

    def _altern_compute_c(self, bar_y, R):

        batch_size = bar_y.shape[0]

        # temp = torch.kron(torch.eye(self.N, device=self.device_), torch.transpose(R, -1, -2))

        temp = torch.transpose(R, -1, -2) @ torch.transpose(bar_y.reshape(batch_size, -1, 3), -1, -2)
        # temp = torch.transpose(R, -1, -2) @ bar_y.reshape(batch_size, 3, -1)
        temp = torch.transpose(temp, -1, -2).reshape(batch_size, -1).unsqueeze(-1)

        c = 2*self.G @ torch.transpose(self.bar_B, -1, -2) @ temp + self.g  # (B, K, 1)
        # print("Shape of c: ", c.shape)
        return c

    def _altern_test_convergence(self, obj, obj_):
        """
        inputs:
        obj     : torch.tensor of shape (B, 1)  : cost at current iteration
        obj_    : torch.tensor of shape (B, 1)  : cost at previous iteration

        outputs:
        test_mask   : torch.tensor of shape (B, 1)  : dtype=torch.bool

        """

        return torch.abs(obj-obj_) < self.method_altern_tol

    # def equality_constraints(self, bar_y, y):
    #    #Note: We are no longer using the ddn setup to compute gradients. It gives errors anyway.
    #     """
    #     inputs:
    #     bar_y   : torch.tensor of shape (B, 3*N)    : normalized keypoints. see (9) in [1].
    #     y       : torch.tensor of shape (B, 9+K)    : y[b, 0:9] : vectorized rotation matrix
    #                                                 : y[b, 9:] : shape vector
    #
    #     outputs:
    #     eq_constraints : torch.tensor of shape (B, 1)  : see (11) in [1]
    #
    #     """
    #     batch_size = bar_y.shape[0]
    #
    #     # Rotation constraint
    #     r = y[:, 0:9]
    #     r = torch.cat([torch.ones((batch_size, 1), device=self.device_), r], dim=1)
    #
    #     batch_size = r.shape[0]
    #     eq_constraints = torch.zeros((batch_size, 17), device=self.device_)
    #
    #     for constraint_idx in range(16):
    #         temp_const = r.unsqueeze(1) @ self.A[constraint_idx, :, :].unsqueeze(0) @ r.unsqueeze(-1) - self.d[
    #             constraint_idx]
    #         temp_const = temp_const.squeeze(-1).squeeze(-1)
    #         eq_constraints[:, constraint_idx] = temp_const
    #
    #     # Shape constraint
    #     c = y[:, 9:]    # (B, K)
    #     eq_constraints[:, 16] = c.sum(1) - 1
    #
    #     print("Shape of eq_constraints: ", eq_constraints.shape)
    #
    #     return eq_constraints

    def _vecR(self, R):
        """
        input:
        R: torch.tensor of shape (B, 3, 3)

        where
        B = batch size

        output:
        r: torch.tensor of shape (B, 9)
        """
        batch_size = R.shape[0]
        r = torch.reshape(torch.transpose(R, 1, 2), (batch_size, 9))
        # r = torch.reshape(torch.transpose(R, 1, 2), (batch_size, 9))

        return r

    def _ytoRc(self, y):

        batch_size = y.shape[0]

        R = y[:, 0:9].reshape(batch_size, 3, 3)  # (B, 3, 3)
        R = torch.transpose(R, -1, -2)
        c = y[:, 9:].reshape(batch_size, self.K, 1)  # (B, K, 1)

        return R, c

    def _Rctoy(self, R, c):

        batch_size = R.shape[0]

        y = torch.zeros((batch_size, 9 + self.K), device=self.device_)
        y[:, 0:9] = self._vecR(R)
        y[:, 9:] = c.squeeze(-1)

        return y

    def _getGg(self):
        """
        intermediate:
           self.bar_B: torch.tensor of shpae (1, 3N, K), where B = batch size

        output:
           self.G: torch.tensor of shape (1, K, K)
           self.g: torch.tensor of shape (1, K, 1)

        """
        bar_B = self.bar_B.squeeze(0)

        bar_H = 2 * (torch.matmul(bar_B.T, bar_B) + self.lambda_constant * torch.eye(self.K, device=self.device_))
        bar_Hinv = torch.inverse(bar_H)
        Htemp = torch.matmul(bar_Hinv, torch.ones(bar_Hinv.shape[-1], 1, device=self.device_))

        G = bar_Hinv - (torch.matmul(Htemp, Htemp.T)) / (
            torch.matmul(torch.ones(1, Htemp.shape[0], device=self.device_), Htemp))  # (K, K)
        g = Htemp / (torch.matmul(torch.ones(1, Htemp.shape[0], device=self.device_), Htemp))  # (K, 1)

        return G.unsqueeze(0), g.unsqueeze(0)

    def _getA(self):
        """
        output:
        A: torch.tensor of shape (16, 10, 10)
        """

        A = torch.zeros(16, 10, 10, device=self.device_)

        A[0, 0, 0] = 1

        A[1, 0, 0] = 1
        A[1, 1, 1] = -1
        A[1, 2, 2] = -1
        A[1, 3, 3] = -1

        A[2, 0, 0] = 1
        A[2, 4, 4] = -1
        A[2, 5, 5] = -1
        A[2, 6, 6] = -1

        A[3, 0, 0] = 1
        A[3, 7, 7] = -1
        A[3, 8, 8] = -1
        A[3, 9, 9] = -1

        A[4, 1, 4] = 1
        A[4, 2, 5] = 1
        A[4, 3, 6] = 1
        A[4, 4, 1] = 1
        A[4, 5, 2] = 1
        A[4, 6, 3] = 1

        A[5, 1, 7] = 1  #
        A[5, 2, 8] = 1
        A[5, 3, 9] = 1
        A[5, 7, 1] = 1
        A[5, 8, 2] = 1
        A[5, 9, 3] = 1

        A[6, 4, 7] = 1
        A[6, 5, 8] = 1
        A[6, 6, 9] = 1
        A[6, 7, 4] = 1
        A[6, 8, 5] = 1
        A[6, 9, 6] = 1

        A[7, 2, 6] = 1
        A[7, 3, 5] = -1
        A[7, 0, 7] = -1
        A[7, 6, 2] = 1
        A[7, 5, 3] = -1
        A[7, 7, 0] = -1

        A[8, 3, 4] = 1
        A[8, 1, 6] = -1
        A[8, 0, 8] = -1
        A[8, 4, 3] = 1
        A[8, 6, 1] = -1
        A[8, 8, 0] = -1

        A[9, 1, 5] = 1
        A[9, 0, 9] = -1
        A[9, 2, 4] = -1
        A[9, 5, 1] = 1
        A[9, 9, 0] = -1
        A[9, 4, 2] = -1

        A[10, 5, 9] = 1
        A[10, 0, 1] = -1
        A[10, 6, 8] = -1
        A[10, 9, 5] = 1
        A[10, 1, 0] = -1
        A[10, 8, 6] = -1

        A[11, 6, 7] = 1
        A[11, 4, 9] = -1
        A[11, 0, 2] = -1
        A[11, 7, 6] = 1
        A[11, 9, 4] = -1
        A[11, 2, 0] = -1

        A[12, 4, 8] = 1
        A[12, 0, 3] = -1
        A[12, 5, 7] = -1
        A[12, 8, 4] = 1
        A[12, 3, 0] = -1
        A[12, 7, 5] = -1

        A[13, 3, 8] = 1
        A[13, 2, 9] = -1
        A[13, 0, 4] = -1
        A[13, 8, 3] = 1
        A[13, 9, 2] = -1
        A[13, 4, 0] = -1

        A[14, 1, 9] = 1
        A[14, 0, 5] = -1
        A[14, 3, 7] = -1
        A[14, 9, 1] = 1
        A[14, 5, 0] = -1
        A[14, 7, 3] = -1

        A[15, 2, 7] = 1
        A[15, 1, 8] = -1
        A[15, 0, 6] = -1
        A[15, 7, 2] = 1
        A[15, 8, 1] = -1
        A[15, 6, 0] = -1

        return A

    def _getd(self):

        d = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device_)
        return d


class PACEbp():
    """
    PACE implementation as a differentiable function. The class parameterizes the PACE function. This uses Alternating
    Method in computing rotation and shape parameter.

    Note:
        This implment rotation computation is implemented as a declarative node (ddn.node.EqConstDeclarativeNode)
       This module can run on
    """
    def __init__(self, model_keypoints, weights=None, batch_size=32):
        super().__init__()
        """
        weights: torch.tensor of shape (N, 1)
        model_keypoints: torch.tensor of shape (K, 3, N) 
        lambda_constant: torch.tensor of shape (1, 1)
        """
        self.device_ = model_keypoints.device
        self.model_keypoints = model_keypoints.to(device=self.device_)      # (K, 3, N)

        self.N = self.model_keypoints.shape[-1]                             # (1, 1)
        self.K = self.model_keypoints.shape[0]                              # (1, 1)

        if weights == None:
            weights = torch.ones((self.N, 1), device=self.device_)
        self.w = weights.unsqueeze(0).to(device=self.device_)               # (1, N, 1)

        self.b = batch_size
        self.lambda_constant = torch.tensor([1.0]).to(device=self.device_)

        self.b_w = self._get_b_w()                  # (1, K, 3)
        self.bar_B = self._get_bar_B()              # (1, 3N, K)

        # Rotation ddn layer
        # option 1: as a function. This will backprop through the Alternating Method algorithm steps.
        self.rotation_n_shape_fn = PACErotationNshape(bar_B=self.bar_B, lambda_constant=self.lambda_constant)
        # option 2: ParamDeclarativeFunction. This will backprop using a .gradient() module, that can compute
        # input-output derivatives. NOTE: Currently, this method is not implemented. So, do not use it.
        # self.rotation_n_shape = PACErotationNshape(bar_B=self.bar_B, lambda_constant=self.lambda_constant)
        # self.rotation_n_shape_fn = ParamDeclarativeFunction(self.rotation_n_shape)

    def forward(self, y):
        """
        input:
        y: torch.tensor of shape (B, 3, N), where B = batch size

        output:
        R: torch.tensor of shape (B, 3, 3), where B = batch size
        t: torch.tensor of shape (B, 3, 1), where B = batch size
        c: torch.tensor of shape (B, K, 1), where B = batch size
        """
        batch_size = y.shape[0]
        y_w = self._get_y_w(y=y)
        bar_y = self._get_bar_y(y=y, y_w=y_w)

        bar_y_eq9 = torch.transpose(bar_y, -1, -2).reshape(batch_size, -1)
        out = self.rotation_n_shape_fn.forward(bar_y_eq9)
        R, c = self.rotation_n_shape_fn._ytoRc(out)
        # R = torch.transpose(R, -1, -2)
        t = self._translation(y_w=y_w, R=R, c=c)

        return R, t, c

    def _get_y_w(self, y):
        """
        input:
        y: torch.tensor of shape (B, 3, N), where B = batch size

        intermediate:
        self.w: torch.tensor of shape (1, N, 1)

        output:
        y_w: torch.tensor of shape (B, 3, 1), where B = batch size
        """

        return torch.matmul(y, self.w)/self.w.sum()

    def _translation(self, y_w, R, c):
        """
        input:
        y_w: torch.tensor of shape (B, 3, 1), where B = batch size
        R: torch.tensor of shape (B, 3, 3), where B = batch size
        c: torch.tensor of shape (B, K, 1), where B = batch size

        intermediate:
        self.b_w: torch.tensor of shape (1, K, 3)
        self.g: torch.tensor of shape (K, 1)

        output:
        t: torch.tensor of shape (B, 3, 1), where B = batch size
        """

        return y_w - torch.matmul(R, torch.matmul(torch.transpose(self.b_w, -1, -2), c))

    def _get_b_w(self):
        """
        intermediate:
        self.model_keypoints: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)

        output:
        b_w: torch.tensor of shape (1, K, 3)
        """

        b_w = torch.matmul(self.model_keypoints, self.w)/self.w.sum() # (K, 3, 1)

        return b_w.squeeze(-1).unsqueeze(0) # (1, K, 3)

    def _get_bar_y(self, y, y_w):
        """
        input:
        y: torch.tensor of shape (B, 3, N), where B = batch size
        y_w: torch.tensor of shape (B, 3, 1), where B = batch size

        intermediate:
        self.w: torch.tensor of shape (1, N, 1)

        output:
        bar_y: torch.tensor of shape (B, 3, N), where B = batch size
        """

        return torch.sqrt(torch.transpose(self.w, -1, -2)) * (y-y_w)

    def _get_bar_B(self):
        """
        intermediate:
        self.model_keypoints: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)
        b_w: torch.tensor of shape (1, K, 3)

        output:
        bar_B: torch.tensor of shape (1, 3N, K), where B = batch size
        """

        bar_b = torch.sqrt(torch.transpose(self.w, -1, -2))\
                *(self.model_keypoints - self.b_w.squeeze(0).unsqueeze(-1))  # (K, 3, N)
        bar_B = torch.transpose(bar_b, -1, -2).reshape(bar_b.shape[0], bar_b.shape[-1]*bar_b.shape[-2], 1)  # (K, 3N, 1)
        bar_B = bar_B.squeeze(-1)   # (K, 3N)
        bar_B = torch.transpose(bar_B, -1, -2)  # (3N, K)

        return bar_B.unsqueeze(0)   #(1, 3N, K)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    B = 40
    N = 10
    K = 44
    n = 400
    weights = torch.rand(N, 1).to(device=device)
    model_keypoints = torch.rand(K, 3, N).to(device=device)
    lambda_constant = torch.tensor([1.0]).to(device=device)
    cad_models = torch.rand(K, 3, n).to(device=device)

    pace_model = PACEbp(weights=weights, model_keypoints=model_keypoints, batch_size=B)

    keypoints, rotations, translations, shape = generate_random_keypoints(batch_size=B,
                                                                          model_keypoints=model_keypoints.to('cpu'))
    keypoints = keypoints.to(device=device)
    rotations = rotations.to(device=device)
    translations = translations.to(device=device)
    shape = shape.to(device=device)

    # batch_size = B
    # y = keypoints
    #
    # # Inside the forward pass:
    # y_w = pace_model._get_y_w(y=y)
    # bar_y = pace_model._get_bar_y(y=y, y_w=y_w)
    # bar_y_eq9 = torch.transpose(bar_y, -1, -2).reshape(batch_size, -1)
    # out = pace_model.rotation_n_shape_fn.forward(bar_y_eq9)
    # R, c = pace_model.rotation_n_shape._ytoRc(out)
    #
    # grad_in = pace_model.rotation_n_shape.gradient(bar_y_eq9, out)
    # print("Input gradients: ")
    # print(grad_in.shape)
    # print(grad_in)


    keypoints.requires_grad = True
    start = time.process_time()
    rot_est, trans_est, shape_est = pace_model.forward(y=keypoints)
    print("Time for PACEbp: ", 1000 * (time.process_time() - start) / B, ' ms')

    er_shape = shape_error(shape, shape_est)
    er_trans = translation_error(translations, trans_est)
    er_rot = rotation_euler_error(rotations, rot_est)

    print("rotation error: ", er_rot.mean())
    print("translation error: ", er_trans.mean())
    print("shape error: ", er_shape.mean())

    loss = er_shape.mean() + er_trans.mean() + er_rot.mean()
    loss.backward()

    print("Shape of gradients at keypoints ", keypoints.grad.shape)
    print(keypoints.grad)
    print('-' * 20)



