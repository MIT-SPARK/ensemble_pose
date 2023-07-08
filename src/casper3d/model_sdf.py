"""
This implements model SDF functions and their backpropagation.

We implement three various of SDF: deepSDF, voxelSDF, and octreeSDF.

"""

from utils.ddn.node import ParamDeclarativeFunction
from utils.math_utils import *
from utils.sdf_gen.mesh2sdf_floodfill import vox


class DeepSDF:
    def __init__(self, class_id, model_id):
        self.class_id = class_id
        self.model_id = model_id

    def solve(self, input):
        """

        Args:
            input: torch.tensor of shape (B, 3, m)

        Returns:
            output: torch.tensor of shape (B, 1, m)

        """

        # ToDo: implement output = SDF(input)
        output = torch.rand(10, 10)

        return output, None

    def gradient(self, input, y=None, v=None, ctx=None):

        if v == None:
            v = torch.ones_like(input)

        # ToDo: implement gradient
        gradient = torch.rand(10, 10)
        # ToDo: need to multiply gradient by v

        return (gradient, None)


class VoxelSDF:
    def __init__(self, sdf_grid: vox.VoxSDF, sdf_grad_grid: vox.VoxSDFGrad, device="cuda"):
        """Initialize a voxel SDF model"""
        # basic consistency checks
        assert np.array_equal(sdf_grid.dims, sdf_grad_grid.dims)
        assert np.array_equal(sdf_grid.res, sdf_grad_grid.res)
        assert np.array_equal(sdf_grid.grid2world, sdf_grad_grid.grid2world)

        # save parameters
        self.device = device
        self.sdf_grid = sdf_grid
        self.sdf_grad_grid = sdf_grad_grid

        # all data are converted to torch tensors on the correct device
        if type(sdf_grid.res) is float:
            self.res = torch.tensor((sdf_grid.res, sdf_grid.res, sdf_grid.res), device=self.device)
        else:
            self.res = torch.tensor(sdf_grid.res, device=self.device)
        self.dims = torch.tensor(sdf_grid.dims, device=self.device)
        self.origin = torch.tensor(sdf_grid.grid2world[0:3, 3], device=self.device)
        self.sdf_grid_data = torch.tensor(sdf_grid.sdf, device=self.device)
        self.sdf_grad_grid_data = [torch.tensor(x, device=self.device) for x in sdf_grad_grid.sdf_grad]

        # compute max sdf value
        self.max_sdf = torch.max(self.sdf_grid_data)

    def solve(self, input):
        """Return SDF value at input
        Args:
            input: torch.tensor of shape (B, 3, m)

        Returns:
            output: torch.tensor of shape (B, 1, m)
        """
        output = grid_trilinear_interp_torch(
            input, self.origin, self.res, self.dims, self.sdf_grid_data, outside_value=self.max_sdf
        )

        return output, None

    def gradient(self, input, y=None, v=None, ctx=None):
        """Return the VJP = J' * v

        Args:
            input: (B, 3, m)
            y:
            v: (B, 1, m)
            ctx:

        Returns:
            gradient: (B, 1, m)

        """
        # v is of size (B, 1, m)
        B = input.size(dim=0)
        m = input.size(dim=2)
        if v == None:
            v = torch.ones(B, 1, m)

        # separate interpolation for x/y/z partial
        # each partial is of size (B, 1, m)
        partial_x = grid_trilinear_interp_torch(
            input, self.origin, self.res, self.dims, self.sdf_grad_grid_data[0], outside_value=0
        )
        partial_y = grid_trilinear_interp_torch(
            input, self.origin, self.res, self.dims, self.sdf_grad_grid_data[1], outside_value=0
        )
        partial_z = grid_trilinear_interp_torch(
            input, self.origin, self.res, self.dims, self.sdf_grad_grid_data[2], outside_value=0
        )

        # J is of size (B, 3, m)
        J = torch.cat((partial_x, partial_y, partial_z), dim=1)
        # grad is of size (B, 3, m)
        # for each point: grad * v
        grad = J * v

        return (grad, None)


class OctreeSDF:
    def __init__(self, class_id, model_id):
        self.class_id = class_id
        self.model_id = model_id

    def solve(self, input):
        """

        Args:
            input: torch.tensor of shape (B, 3, m)

        Returns:
            output: torch.tensor of shape (B, 1, m)

        """

        # ToDo: implement output = SDF(input)
        output = torch.rand(10, 10)

        return output, None

    def gradient(self, input, y=None, v=None, ctx=None):

        if v == None:
            v = torch.ones_like(input)

        # ToDo: implement gradient
        gradient = torch.rand(10, 10)
        # ToDo: need to multip[ly gradient by v

        return (gradient, None)


class ModelSDF:
    def __init__(self, type="deep", **kwargs):
        """

        Args:
            class_id: class_id for ShapeNet
            model_id: ID of the CAD model.
            type: {'deep', 'voxel', 'octree'}

        Returns:
            Class that implements SDF computation (and gradient).

        """
        self.type = type
        if self.type == "deep":
            sdf_node = DeepSDF(**kwargs)  # ToDo: Write DeepSDF
            self.model_sdf = ParamDeclarativeFunction(problem=sdf_node)

        elif self.type == "voxel":
            sdf_node = VoxelSDF(kwargs["sdf_grid"], kwargs["sdf_grad_grid"], kwargs["device"])  # ToDo: Write VoxelSDF
            self.model_sdf = ParamDeclarativeFunction(problem=sdf_node)

        elif self.type == "octree":
            sdf_node = OctreeSDF(**kwargs)  # ToDo: Write OctreeSDF
            self.model_sdf = ParamDeclarativeFunction(problem=sdf_node)

        else:
            raise NotImplementedError

    def forward(self, input):
        """

        Args:
            input: torch.tensor of shape (B, 3, m)

        Returns:
            torch.tensor of shape (B, 1, m)

        """
        return self.model_sdf.forward(input)
