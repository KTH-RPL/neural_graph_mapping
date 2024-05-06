"""Module defining geometric primitives."""
from __future__ import annotations

import numbers
from typing import Union

import torch


class AABBs:
    """Collection of axis aligned bounding boxes."""

    minimas: torch.Tensor
    maximas: torch.Tensor

    def __init__(self, minimas: torch.Tensor, maximas: torch.Tensor) -> None:
        """Intitialize AABBs.

        Args:
            minimas: Minimum points of bounding box. Shape (..., 3).
            maximas: Maximum point of bounding box. Shape (..., 3).
        """
        self.minimas = minimas
        self.maximas = maximas

    def intersects_aabbs(self, aabbs: AABBs) -> torch.Tensor:
        """Check which spheres intersects axis .

        Args:
            spheres: Spheres to check.

        Returns:
            Boolean mask, indicating which spheres intersect the bounding box.
            Shape (*leading_dims_of_arg, *leading_dims_of_self).
        """
        leading_dims_self = self.minimas.shape[:-1]
        leading_dims_arg = aabbs.minimas.shape[:-1]
        aabbs_minmas = aabbs.minimas.view(*leading_dims_arg, *leading_dims_self, 3)
        aabbs_maximas = aabbs.maximas.view(*leading_dims_arg, *leading_dims_self, 3)
        mask_1 = (aabbs_minmas <= self.maximas).all(dim=-1)
        mask_2 = (aabbs_maximas >= self.minimas).all(dim=-1)
        return mask_1 * mask_2


class LineSegments:
    """Collection of line segments."""

    p1s: torch.Tensor
    p2s: torch.Tensor

    def __init__(self, p1s: torch.Tensor, p2s: torch.Tensor) -> None:
        """Initialize line segments.

        Args:
            p1s:
                First points of the line segments.
                Shape (..., 3) or broadcastable to p2s.
            p2s:
                First points of the line segments.
                Shape (..., 3) or broadcastable to p1s.
        """
        broadcasted_shape = torch.broadcast_shapes(p1s.shape, p2s.shape)

        self.p1s = p1s.expand(broadcasted_shape)
        self.p2s = p2s.expand(broadcasted_shape)

    def intersects_spheres(self, spheres: Spheres) -> torch.Tensor:
        """Check which spheres intersects which line segment.

        Returns:
            Boolean tensor containting which sphere interesects which line segment.
            Shape (*leading_dims_of_spheres, *leading_dims_of_line_segments).
        """
        leading_dims_line_segs = self.p1s.shape[:-1]
        leading_dims_spheres = spheres.centers.shape[:-1]
        closest_points = self.closest_points(spheres.centers)
        sphere_centers = spheres.centers.view(
            *leading_dims_spheres, *((1,) * len(leading_dims_line_segs)), 3
        )
        distances_sq = ((sphere_centers - closest_points) ** 2).sum(-1)
        radii = spheres.radii.view(
            *leading_dims_spheres, *((1,) * len(leading_dims_line_segs))
        )
        return distances_sq <= radii**2

    def closest_points(self, points: torch.Tensor) -> torch.Tensor:
        """Compute closest point on each line segment for points.

        Args:
            points:
                The points to check. Shape (..., 3).

        Returns:
            Closest point for all points and all line segments.
            Shape (*leading_dims_of_points, *leading_dims_of_line_segments, 3).
        """
        leading_dims_line_segs = self.p1s.shape[:-1]
        leading_dims_points = points.shape[:-1]
        dirs = self.p2s - self.p1s  # (*leading_dims_of_line_segs, 3)
        sq_dirs = (dirs * dirs).sum(-1, keepdim=True)  # (*leading_dims_of_line_segs, 1)
        points = points.view(*leading_dims_points, *((1,) * len(leading_dims_line_segs)), 3)
        # handle dirs == 0 case (i.e., 0 length line segments)
        sq_dirs[sq_dirs == 0] = 1.0
        t = ((points - self.p1s) * dirs).sum(-1, keepdim=True) / sq_dirs
        return self.p1s + dirs * torch.clamp(t, 0.0, 1.0)


class Spheres:
    """Collection of spheres."""

    centers: torch.Tensor
    radii: torch.Tensor

    def __init__(
        self, centers: torch.Tensor, radii: Union[numbers.Number, torch.Tensor]
    ) -> None:
        """Initialize spheres.

        Args:
            centers: Centers of spheres. Shape (..., 3).
            radii: Radii of spheres. Either scalar or shape (...,).
        """
        leading_dims = centers.shape[:-1]
        self.centers = centers

        if isinstance(radii, numbers.Number):
            radii = torch.tensor(radii, dtype=centers.dtype, device=centers.device)
        if radii.dim() == 0:
            radii = radii.expand(leading_dims)

        self.radii = radii

    def aabbs(self) -> AABBs:
        """Return AABBs for spheres."""
        radii = self.radii.unsqueeze(-1)
        return AABBs(self.centers - radii, self.centers + radii)
