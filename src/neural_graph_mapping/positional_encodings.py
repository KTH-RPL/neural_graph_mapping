"""Module defining various positional encodings with a common interface."""
from abc import abstractmethod
from typing import Literal

import numpy as np
import permutohedral_encoding
import torch


class PositionalEncoding(torch.nn.Module):
    """Positional encoding mapping ND points to a higher dimensional encoding."""

    @abstractmethod
    def get_out_dim(self) -> int:
        """Return number of output channels."""
        raise NotImplementedError()


class PermutohedralEncoding(permutohedral_encoding.PermutoEncoding, PositionalEncoding):
    """Wrapper adding get_out_dim to permutohedral_encoding.PermutoEncoding."""

    def __init__(
        self,
        pos_dim: int,
        log2_hashmap_size: int,
        nr_levels: int,
        nr_feat_per_level: int,
        coarsest_scale: float,
        finest_scale: float,
        appply_random_shift_per_level: bool = True,
        concat_points: bool = False,
        concat_points_scaling: float = 1.0,
        init_scale: float = 1e-5,
    ) -> None:
        """Initialize PermutohedralEncoding.

        Scales are generated using geometric spacing between coarsest and finest scale.

        Args:
            pos_dim: Input dimensionality.
            log2_hashmap_size: log2 of hashmap size.
            nr_levels: Number of scale levels.
            nr_feat_per_level: Number of features per level. Must be 2 right now.
            coarsest_scale: Coarsest scale.
            finest_scale: Finest scale.
            apply_random_shift_per_level: Whether to apply random shift at each level.
            concat_points: Whether to append raw points.
            concat_points_scaling: Scaling factor applied before appending raw points.
        """
        scale_per_level = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        capacity = 2**log2_hashmap_size
        super().__init__(
            pos_dim,
            capacity,
            nr_levels,
            nr_feat_per_level,
            scale_per_level,
            appply_random_shift_per_level,
            concat_points,
            concat_points_scaling,
            init_scale=init_scale
        )

    def get_out_dim(self) -> int:
        """Return number of output channels."""
        return self.output_dims()


class TriplaneEncoding(PositionalEncoding):
    """Learned triplane encoding.

    Based on nerfstudio's implementation with additional option for concatenation.

    The encoding at [i,j,k] is an n dimensional vector corresponding to the
    element-wise product of the three n dimensional vectors at plane_coeff[i,j],
    plane_coeff[i,k], and plane_coeff[j,k].

    This allows for marginally more expressivity than the TensorVMEncoding, and each
    component is self standing and symmetrical, unlike with VM decomposition where we
    needed one component with a vector along all the x, y, z directions for symmetry.

    This can be thought of as 3 planes of features perpendicular to the x, y, and z
    axes, respectively and intersecting at the origin, and the encoding being the
    element-wise product of the element at the projection of [i, j, k] on these planes.

    The use for this is in representing a tensor decomp of a 4D embedding tensor:
    (x, y, z, feature_size)

    This will return a tensor of shape (bs:..., num_components)
    """

    def __init__(
        self,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        mode: Literal["sum", "product", "concat"] = "sum",
    ) -> None:
        """Initialize TriplaneEncoding.

        Args:
            resolution: Resolution of grid.
            num_components:
                The number of scalar triplanes to use.
                Equal to output size for mode==sum and mode==product.
                Equal to output size / 3 for mode==concat.
            init_scale: The scale of the initial values of the planes
            mode:
                Whether to sum, multiply or concatenate the features of each triplane.
        """
        super().__init__()

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.mode = mode

        self.plane_coef = torch.nn.Parameter(
            self.init_scale
            * torch.randn((3, self.num_components, self.resolution, self.resolution))
        )

    def get_out_dim(self) -> int:
        """Return number of output channels."""
        if self.mode in ["sum", "product"]:
            return self.num_components
        elif self.mode == "concat":
            return 3 * self.num_components
        else:
            raise ValueError(f"{self.mode=} is not supported.")

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Sample features from this encoder.

        Expects in_tensor to be in range [-1, 1].
        """
        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack(
            [in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]],
            dim=0,
        )  # [3, batch_size, 2]

        plane_coord = plane_coord.view(3, -1, 1, 2)  # [3, batch_size, 1, 2]
        plane_features = torch.nn.functional.grid_sample(
            self.plane_coef, plane_coord, align_corners=True, padding_mode="border"
        )  # [3, num_components, flattened_batch_size, 1]

        if self.mode == "product":
            plane_features = (
                plane_features.prod(0).squeeze(-1).T
            )  # [flattened_batch_size, num_components]
        elif self.mode == "sum":
            plane_features = plane_features.sum(0).squeeze(-1).T
        elif self.mode == "concat":
            plane_features = plane_features.squeeze(-1).reshape(3 * self.num_components, -1).T
        else:
            raise ValueError(f"{self.mode=} is not supported.")

        return plane_features.reshape(*original_shape[:-1], -1)


class PositionalEncodingFourier(PositionalEncoding):
    """Child class for positional encoding nn.Module."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        mu: float,
        sigma: float,
        raw_coords: bool,
    ) -> None:
        """Initialize positional encoding layer.

        Args:
            in_features: number of input connections to the layer
            out_features: number of nodes in the layer
            mu: mean of normal distribution for weight initialization
            sigma: standard deviation of normal distribution for weight initialization
            raw_coords:
                if True, input coordinates are concatenated with.
                Fourier features. otherwise, positional encoding consists of
                (out_features) Fourier features
        """
        super().__init__()
        self._linear = torch.nn.Linear(
            dim_in, dim_out - dim_in if raw_coords else dim_out, False
        )

        self._dim_out = dim_out
        self._raw_coords = raw_coords

        torch.nn.init.normal_(self._linear.weight, mu, sigma)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Applies positional encoding to points.

        Args:
            points:
                Points for which to compute positional encoding. Shape (..., dim_in).

        Returns:
            Positional encodings for each point. Shape (..., dim_out).
        """
        fourier_features = torch.sin(self._linear(points))
        return (
            torch.cat((points, fourier_features), dim=-1)
            if self._raw_coords
            else fourier_features
        )

    def get_out_dim(self) -> int:
        """Return number of output channels."""
        return self._dim_out


class PositionalEncodingNeRF(PositionalEncoding):
    """Positional encoding for continuous points.

    As described in:
        NeRF Representing Scenes as Neural Radiance Fields for View Synthesis,
        Mildenhall et al., 2020
    """

    def __init__(self, dim_in: int, num_octaves: int = 8, start_octave: int = 0) -> None:
        """Initialize positional encoding.

        Positional encoding can be described by f(2^i * pi * input),
        i = start_octave, ..., start_octave + num_octaves - 1.

        f will be sin and cos, and the respective encodings are concatenated.

        Args:
            dim_in: Number of input dimensions. Used to calculate output dimension.
            num_octaves: Number of octaves. Equals output dimension.
            start_octave: First octave.
        """
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave
        self.dim_in = dim_in

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Applies positional encoding to points.

        Args:
            points:
                Points for which to compute positional encoding. Shape (..., dim_in).

        Returns:
            Positional encodings for each point. Shape (..., 2*dim_in*num_octaves).
        """
        leading_dims = points.shape[:-1]

        octaves = torch.arange(
            self.start_octave,
            self.start_octave + self.num_octaves,
            device=points.device,
            dtype=torch.float,
        )
        multipliers = 2**octaves * torch.pi
        points = points.unsqueeze(-1)

        scaled_points = points * multipliers

        sines = torch.sin(scaled_points).reshape(*leading_dims, -1)
        cosines = torch.cos(scaled_points).reshape(*leading_dims, -1)

        result = torch.cat((sines, cosines), -1)
        return result

    def get_out_dim(self) -> int:
        """Return number of output channels."""
        return self.dim_in * self.num_octaves * 2
