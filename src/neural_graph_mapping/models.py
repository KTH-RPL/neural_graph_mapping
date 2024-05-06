"""This module defines modular layers used to build the scene representations."""
from typing import Literal, Optional

import einops
import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.transforms import quaternion_apply, quaternion_invert

from neural_graph_mapping.utils import str_to_object


def complex_invert(comp: torch.Tensor) -> torch.Tensor:
    """Invert the rotation given by a complex number.

    This is the equivalent to the complex conjugate.

    Args:
        comp: Complex numbers. Shape (..., 2), real part first.

    Returns:
        The inverse, a tensor of complex nummbers of shape (..., 2).
    """
    scaling = torch.tensor([1, -1], device=comp.device)
    return comp * scaling


def complex_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two complex numbers.

    Usual torch rules for broadcasting apply.

    Args:
        a: Complex numbers. Shape (..., 2), real part first.
        b: Complex numbers. Shape (..., 2), real part first.

    Returns:
        The product of a and b. Real part first. Shape (..., 2).
    """
    ar, ai = torch.unbind(a, -1)
    br, bi = torch.unbind(b, -1)

    outr = ar * br - ai * bi
    outi = ar * bi + br * ai

    return torch.stack((outr, outi), -1)


def complex_apply(comp: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Apply the rotation given by a complex number to a 2D point.

    Usual torch rules for broadcasting apply. Implementation based on
    pytorch3d.transforms.quaternion_apply.

    Args:
        comp: Tensor of complex numbers, real part first. Shape (..., 2).
        point: Tensor 2D points. Shape (...,2).

    Returns:
        Tensor of rotated points. Shape (...,2).
    """
    if point.size(-1) != 2:
        raise ValueError(f"Points are not in 2D, {point.shape}.")
    return complex_raw_multiply(comp, point)


class NeuralField(torch.nn.Module):
    """Simple neural field composed of positional encoding and MLP."""

    def __init__(
        self,
        encoding_type: str,
        encoding_kwargs: dict,
        num_layers: int,
        dim_out: int,
        dim_mlp_out: Optional[int] = None,
        skip_mode: Literal["no", "add", "concat", "rezero"] = "no",
        initial_geometry_bias: float = 0.0,
        neus_initial_sd: Optional[float] = None,
    ) -> None:
        """Initialize NeuralField.

        Args:
            encoding_type: Type of encoding.
            encoding_kwargs: Keyword arguments for encoding.
            num_layers: Number of layers following the positional encoding.
            dim_out: Output dimensionality of last linear layer.
            dim_mlp_out: Output dimensionality of linear layers.
            skip_mode: One of no, add, concat, rezero.
            initial_geometry_bias: Will be added to initial bias of last output.
        """
        super().__init__()

        self._encoding_type = str_to_object(encoding_type)
        self._encoding_kwargs = encoding_kwargs
        self._encoding = self._encoding_type(**self._encoding_kwargs)
        self._dim_encoding = self._encoding.get_out_dim()
        self._dim_out = dim_out
        self._dim_mlp_out = dim_mlp_out
        if self._dim_mlp_out is None:
            self._dim_mlp_out = self._dim_encoding
        self._skip_mode = skip_mode
        self._initial_geometry_bias = initial_geometry_bias
        self._num_layers = num_layers

        if self._skip_mode in ["no", "add", "rezero"]:
            self._dim_mlp_in = self._dim_mlp_out
        elif self._skip_mode == "concat":
            self._dim_mlp_in = self._dim_mlp_out + self._dim_encoding
        else:
            raise ValueError(f"Skip mode {self._skip_mode} is not available.")

        if self._skip_mode == "rezero":
            self._rezero = torch.nn.Parameter(torch.empty(self._num_layers, device="cuda"))

        self._dims_in = [self._dim_encoding] + [
            self._dim_mlp_in for _ in range(self._num_layers)
        ]
        self._dims_out = [self._dim_mlp_out for _ in range(self._num_layers)]
        self._dims_out.append(self._dim_out)  # final layer

        if neus_initial_sd is not None:
            self._neus_sd = torch.nn.Parameter(torch.tensor(neus_initial_sd))

        self._linears = torch.nn.ModuleList()
        for dim_in, dim_out in zip(self._dims_in, self._dims_out):
            self._linears.append(torch.nn.Linear(dim_in, dim_out))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._skip_mode == "rezero":
            self._rezero.zero_()

        with torch.no_grad():
            self._linears[-1].bias[-1] += self._initial_geometry_bias
            self._linears[-1].weight *= 1

    def numel(self) -> int:
        """Return number of parameters in this field."""
        field_parameters = list(self.parameters())
        return sum(fp.numel() for fp in field_parameters)

    def forward(self, query_points: torch.Tensor) -> torch.Tensor:
        """Compute output."""
        outs_encoding = outs = self._encoding(query_points)


        for i, linear in enumerate(self._linears):
            prev_outs = outs

            outs = linear(outs)

            if i == self._num_layers:
                break


            outs = torch.relu(outs)

            # skip connection
            if self._skip_mode == "concat":
                outs = torch.cat((outs, outs_encoding), dim=-1)
            elif self._skip_mode == "add":
                outs = torch.cat(
                    (
                        outs[..., : self._dim_encoding] + outs_encoding,
                        outs[..., self._dim_encoding :],
                    ),
                    dim=-1,
                )
            elif self._skip_mode == "rezero":
                if i == 0:
                    outs = torch.cat(
                        (
                            self._rezero[i] * outs[..., : self._dim_encoding] + prev_outs,
                            self._rezero[i] * outs[..., self._dim_encoding :],
                        ),
                        dim=-1,
                    )
                else:
                    outs = self._rezero[i] * outs + prev_outs

        return outs


class NeuralFieldSet(torch.nn.Module):
    """Set of posed neural fields."""

    def __init__(
        self,
        dim_points: int,
        field_type: str,
        field_kwargs: dict,
        num_knn: int,
        distance_factor: float,
        outside_value: float,
        field_radius: Optional[float] = None,
        scale_mode: Literal["no", "unit_ball", "unit_cube"] = "no",
    ) -> None:
        """Initialize NeuralFieldSet.

        Args:
            dim_points: Dimensionality of query points.
            num_layers:
                Number of linear layers following the encoding; excluding output layer.
            num_knn: Number of k-nearest MLPs to be evaluated.
            distance_factor: Factor to scale distances before softmax is applied.
            field_radius:
                Fields will return empty space outside this radius. If None, the
                k-nearest fields will be evaluated regardless of distance.
            scale_mode:
                One of "no", "unit_ball", "unit_cube". "unit_ball" and "unit_cube" are
                only supported in combination with a field_radius.
        """
        super().__init__()

        self._scale_mode = scale_mode
        self._field_radius = field_radius
        if scale_mode != "no" and field_radius is None:
            raise ValueError(f"{scale_mode=} requires field_radius to be specified.")

        self._dim_points = dim_points

        self._num_knn = num_knn
        self._distance_factor = distance_factor

        # used to call and to initialize
        self._prototype_field = str_to_object(field_type)(**field_kwargs)
        self._vmap_wrapper = lambda params, buffers, queries: torch.func.functional_call(
            self._prototype_field, (params, buffers), queries
        )

        self.all_fields_params = None
        self.vmap_fields_params = None
        self._outside_value = outside_value

        if self._dim_points == 2:
            self._orientation_apply = complex_apply
            self._orientation_invert = complex_invert
        elif self._dim_points == 3:
            self._orientation_apply = quaternion_apply
            self._orientation_invert = quaternion_invert
        else:
            raise NotImplementedError("Only 2D and 3D spaces are supported.")

    def add_fields(self, num_fields: int) -> None:
        """Add additional fields.

        Args:
            num_fields: Number of fields to add.

        Returns:
            List of individual field parameters.
        """
        new_field_params = {
            k: einops.repeat(v, "... -> n ...", n=num_fields).clone()
            for k, v in self._prototype_field.state_dict().items()
        }
        if self.all_fields_params is None:
            self.all_fields_params = new_field_params
        else:
            self.all_fields_params = {
                k: torch.cat((v, new_field_params[k]))
                for k, v in self.all_fields_params.items()
            }

    def set_vmap_fields(self, field_ids: Optional[torch.Tensor]) -> None:
        """Set subset of fields as active fields.

        These active fields correspond to the batch dimension of the forward pass.
        """
        if field_ids is None:
            self.vmap_fields_params = self.all_fields_params
        else:
            self.vmap_fields_params = {
                k: v[field_ids] for k, v in self.all_fields_params.items()
            }

    def _scale_local_points(self, local_points: torch.Tensor) -> torch.Tensor:
        if self._scale_mode == "unit_cube":
            return local_points / (2 * self._field_radius) + 0.5
        elif self._scale_mode == "unit_ball":
            return local_points / self._field_radius
        elif self._scale_mode == "no":
            return local_points
        raise NotImplementedError(f"{self._scale_mode=} is not available.")

    def forward(
        self,
        query_points: torch.Tensor,
        field_positions: Optional[torch.Tensor] = None,
        field_orientations: Optional[torch.Tensor] = None,
        field_ids: Optional[torch.Tensor] = None,
        use_vmap: bool = True,
        field_radius: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute color and density / distance for query points.

        Args:
            query_points:
                Query points.
                Shape (num_vmap_fields, points_per_field, dim_points) if active fields
                are set, else (num_points, dim_points).
            field_positions:
                Field positions in the world frame.
                Shape (num_vmap_fields, dim_points) if use_vmap=True, else
                (num_fields, dim_points).
                If None, query_points are assumed to be in local coordinates already.
            field_orientations:
                Field orientations. These are the orientation in the world frame.
                Real-first complex number (for 2D points) or real-first quaternion (for
                3D points).
                Shape (num_vmap_fields, 2 or 4) if active fields are set, else
                (num_fields, 2 or 4).
                If None, query_points are assumed to be in local coordinates already.
            field_ids:
                Which fields to evaluate. Ignored if use_vmap=True. All fields are
                evaluated if None. Shape (num_fields,).
            use_vmap:
                Whether to use vmap for evaluation.
            field_radius:
                If provided, overwrites field radius of class.

        Returns:
            outputs: Color for each query point. Shape (..., dim_out).
        """
        if field_radius is None:
            field_radius = self._field_radius

        if use_vmap:
            if field_positions is not None:
                local_query_points = query_points - field_positions.unsqueeze(-2)
                local_query_points = self._orientation_apply(
                    self._orientation_invert(field_orientations).unsqueeze(-2),
                    local_query_points,
                )
            else:
                local_query_points = query_points

            local_query_points = self._scale_local_points(local_query_points)


            result = torch.vmap(self._vmap_wrapper)(
                self.vmap_fields_params, {}, local_query_points
            )
            return result

        if field_ids is None:
            field_ids = torch.arange(0, len(field_positions), device=field_positions.device)

        # no vmap -> knn
        leading_dims = query_points.shape[:-1]
        query_points = query_points.view(-1, self._dim_points)
        num_query_points = len(query_points)
        device = query_points.device
        if len(field_positions) < self._num_knn:
            num_knn = len(field_positions)
        else:
            num_knn = self._num_knn

        # find scene embeddings for each query point
        knn_dists, knn_indices, _ = knn_points(
            query_points.unsqueeze(0),
            field_positions.unsqueeze(0),
            K=num_knn,
            return_sorted=True,
        )
        knn_dists = torch.sqrt(knn_dists[0])

        radius_mask = knn_dists[:, 0] < field_radius
        knn_dists = knn_dists[radius_mask]
        knn_indices = knn_indices[0][radius_mask]
        knn_positions = field_positions[knn_indices]
        knn_orientations = field_orientations[knn_indices]
        query_points = query_points[radius_mask]

        # transform points to sr reference frames
        local_query_points = query_points.unsqueeze(-2) - knn_positions
        local_query_points = self._orientation_apply(
            self._orientation_invert(knn_orientations), local_query_points
        )
        local_query_points = self._scale_local_points(local_query_points)

        # knn_dists[knn_dists > field_radius] = 1000.0
        dist_weights = torch.nn.functional.softmax(-self._distance_factor * knn_dists, dim=-1)

        field_indices = knn_indices.unique()

        knn_outs = torch.empty(*local_query_points.shape[:-1], 4, device=device)

        field_masks = knn_indices == field_indices[:, None, None]

        for field_index, field_mask in zip(field_indices.tolist(), field_masks):
            params = {k: v[field_ids[field_index]] for k, v in self.all_fields_params.items()}
            knn_outs[field_mask] = self._vmap_wrapper(
                params, {}, local_query_points[field_mask]
            )

        # composite based on weights
        in_radius_outs = einops.einsum(dist_weights, knn_outs, "... k, ... k dc -> ... dc")

        outs = torch.full((num_query_points, 4), device=device, fill_value=self._outside_value)
        outs[radius_mask] = in_radius_outs

        # change back to original batch dimensions
        return outs.reshape(*leading_dims, -1)

    def numel(self) -> int:
        """Return number of parameters in the set."""
        num_fields = len(self.all_fields_params)
        params_per_field = self._prototype_field.numel()
        return params_per_field * num_fields
