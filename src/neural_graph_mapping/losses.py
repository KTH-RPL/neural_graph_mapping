"""Module that defines various loss functions."""

from typing import Literal, Optional, Tuple

import torch

from neural_graph_mapping import utils


def photometric_loss(
    mode: Literal["l1", "l2", "gaussian_nll"],
    measured_colors: torch.Tensor,
    rendered_colors: torch.Tensor,
    rendered_color_vars: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the mean photometric error between a measured and a rendered image.

    Args:
        measured_colors: RGB value of pixels on the measured images. Shape (..., 3).
        rendered_colors: RGB value of pixels on the rendered images. Shape (..., 3).
        rendered_color_vars: Variance of the rendered colors. Shape (..., 3).

    Returns:
        error: Mean photometric error for the set of corresponding points. Scalar.
    """
    if mode == "l1":
        return torch.mean(torch.abs(measured_colors - rendered_colors))
    if mode == "l2":
        return torch.mean((measured_colors - rendered_colors) ** 2)
    elif mode == "gaussian_nll":
        nlls = 0.5 * (
            rendered_colors - measured_colors
        ) ** 2 / rendered_color_vars + torch.log(torch.sqrt(rendered_color_vars))
        loss = nlls.mean()
        if nlls.mean() > 2:
            return torch.mean(torch.abs(measured_colors - rendered_colors))
        else:
            return loss


@utils.benchmark
def depth_loss(
    mode: Literal["huber", "gaussian_nll", "laplacian_nll"],
    measured_depths: torch.Tensor,
    rendered_depths: torch.Tensor,
    rendered_depth_vars: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute depth loss.

    Args:
        mode: The depth loss mode.
        measured_depths: Depth value of points on the measured images. Shape (...).
        rendered_depths: Depth value of points on the rendered images. Shape (...).
        rendered_depths_vars:
            Depth variance of points on the rendered images. Shape (...).

    Returns:
        error: Mean absolute depth error for the set of corresponding points. Scalar.
    """
    if mode == "huber":
        # mean L1-norm between measured and rendered depths
        # return torch.mean(torch.abs(measured_depths - rendered_depths))
        return torch.nn.functional.huber_loss(rendered_depths, measured_depths, delta=0.05)
    elif mode == "gaussian_nll":
        rendered_depth_vars = rendered_depth_vars + 1e-15
        nlls = 0.5 * (
            rendered_depths - measured_depths
        ) ** 2 / rendered_depth_vars + torch.log(torch.sqrt(rendered_depth_vars))
        return nlls.mean()
    elif mode == "laplacian_nll":
        # mean NLL of the measured depths given the rendered depths
        nlls = torch.abs(measured_depths - rendered_depths) / torch.sqrt(
            0.5 * rendered_depth_vars + 1e-6
        ) + 0.5 * torch.log(2 * rendered_depth_vars + 1e-6)
        return nlls.mean()


def eikonal_term(signed_distances: torch.Tensor, points: Tuple[torch.Tensor]) -> torch.Tensor:
    """Compute eikonal term.

    Args:
        signed_distances:
            Signed distances predicted by network. Shape (num_rays, num_samples).
        points:
            Input points used to calculated signed distances. Sequence of tensors.
            Each tensor i has shape (num_rays, num_samples_i, 3).

    Returns:
        Squared deviation of signed distance gradient (with respect to points) magnitude
        from 1.
    """
    grad_outputs = torch.ones_like(signed_distances)
    grads = torch.hstack(
        torch.autograd.grad(signed_distances, points, grad_outputs, create_graph=True)
    )
    return ((torch.linalg.norm(grads, dim=-1) - 1) ** 2).mean()
