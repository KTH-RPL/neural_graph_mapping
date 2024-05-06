"""This module provides a pinhole camera class."""
import numbers
from typing import Literal, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch

from neural_graph_mapping import utils

# FIXME change from OpenCV / OpenGL to RDF / RUB


class Camera:
    """Pinhole camera parameters.

    This class allows conversion between different pixel conventions, i.e., pixel
    center at (0.5, 0.5) (as common in computer graphics), and (0, 0) as common in
    computer vision.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        s: float = 0.0,
        pixel_center: float = 0.0,
    ) -> None:
        """Initialize camera parameters.

        Note that the principal point is only fully defined in combination with
        pixel_center.

        The pixel_center defines the relation between continuous image plane
        coordinates and discrete pixel coordinates.

        A discrete image coordinate (x, y) will correspond to the continuous
        image coordinate (x + pixel_center, y + pixel_center). Normally pixel_center
        will be either 0 or 0.5. During calibration it depends on the convention
        the point features used to compute the calibration matrix.

        Note that if pixel_center == 0, the corresponding continuous coordinate
        interval for a pixel are [x-0.5, x+0.5). I.e., proper rounding has to be done
        to convert from continuous coordinate to the corresponding discrete coordinate.

        For pixel_center == 0.5, the corresponding continuous coordinate interval for a
        pixel are [x, x+1). I.e., floor is sufficient to convert from continuous
        coordinate to the corresponding discrete coordinate.

        Args:
            width: Number of pixels in horizontal direction.
            height: Number of pixels in vertical direction.
            fx: Horizontal focal length.
            fy: Vertical focal length.
            cx: Principal point x-coordinate.
            cy: Principal point y-coordinate.
            s: Skew.
            pixel_center: The center offset for the provided principal point.
        """
        # focal length
        self.fx = fx
        self.fy = fy

        # principal point (stored as pixel center 0.5)
        self.cx = cx - pixel_center + 0.5
        self.cy = cy - pixel_center + 0.5

        # skew
        self.s = s

        if self.s != 0:
            raise NotImplementedError("Skew != 0 not supported.")

        # image dimensions
        self.width = width
        self.height = height

    def get_o3d_pinhole_camera_parameters(self) -> o3d.camera.PinholeCameraParameters():
        """Convert camera to Open3D pinhole camera parameters.

        Open3D camera is at (0,0,0) looking along positive z axis (i.e., positive z
        values are in front of camera). Open3D expects camera with pixel_center = 0
        and does not support skew.

        Returns:
            The pinhole camera parameters.
        """
        fx, fy, cx, cy, _ = self.get_pinhole_camera_parameters(0)
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic.set_intrinsics(self.width, self.height, fx, fy, cx, cy)
        params.extrinsic = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return params

    def get_pinhole_camera_parameters(self, pixel_center: float) -> Tuple:
        """Convert camera to general camera parameters.

        Args:
            pixel_center:
                At which ratio of a square the pixel center should be for the resulting
                parameters. Typically 0 or 0.5. See class documentation for more info.

        Returns:
            - fx, fy: The horizontal and vertical focal length
            - cx, cy:
                The position of the principal point in continuous image plane
                coordinates considering the provided pixel center and the pixel center
                specified during the construction.
            - s: The skew.
        """
        cx_corrected = self.cx - 0.5 + pixel_center
        cy_corrected = self.cy - 0.5 + pixel_center
        return self.fx, self.fy, cx_corrected, cy_corrected, self.s

    @utils.benchmark
    def project_points(
        self,
        points: torch.Tensor,
        convention: Literal["opencv", "opengl"],
        pixel_center: float = 0.5,
        return_in_front_mask: bool = False,
    ) -> Union[torch.FloatTensor, tuple[torch.FloatTensor, torch.BoolTensor]]:
        """Project 3D points in camera frame to image plane.

        Args:
            points: 3D points. Shape (..., 3).
            convention: Which camera convention to follow.
            pixel_center: Which pixel center convention to use for projection.
            return_in_front_mask: Whether to return mask of pixels in front of camera.

        Returns:
            points_2d:
                Continuous image points, following the provided pixel center convention.
                Top left pixel will always be (0, 0). First coordinate is x (columns of image),
                second coordinate is y (row of image).
                Shape (..., 2).
            in_front_mask:
                Only returned if return_in_front_mask is True. Boolean tensor indicating which
                points were in front of the camera. Shape (...).
        """
        image_from_camera = self.get_projection_matrix(convention, pixel_center).to(
            points.device
        )
        # have to make points (...,3, 1) so that batched matmul works, squeeze to remove
        image_points_homogeneous = torch.einsum("oi,...i->...o", image_from_camera, points)
        # image_points_homogeneous = (image_from_camera @ points[..., None]).squeeze(-1)
        z = image_points_homogeneous[..., 2]
        points2d = image_points_homogeneous[..., :2] / z.unsqueeze(-1)
        if return_in_front_mask:
            return points2d, z > 0.0
        return points2d

    def get_projection_matrix(
        self, convention: Literal["opencv", "opengl"] = "opencv", pixel_center: float = 0.5
    ) -> torch.Tensor:
        """Return 3x3 projection matrix.

        Args:
            convention:
                The camera frame convention to use. One of:
                    "opengl": x right, y up, z back
                    "opencv": x right, y down, z forward
                Typically OpenCV is used where all values in matrix are positive.

        Returns:
            Return projection matrix that transforms 3D point in camera frame
            to Shape (3,3).
        """
        fx, fy, cx, cy, _ = self.get_pinhole_camera_parameters(pixel_center)
        if convention == "opencv":
            return torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                dtype=torch.float,
            )
        elif convention == "opengl":
            return torch.tensor(
                [[fx, 0, -cx], [0, -fy, -cy], [0, 0, -1]],
                dtype=torch.float,
            )
        else:
            raise ValueError(f"Unsupported camera convention {convention}.")

    def ijs_to_directions(self, ijs: torch.Tensor, convention: str = "opengl") -> torch.Tensor:
        """Convert row, column indices to 3D unit vectors."""
        fx, fy, cx, cy, _ = self.get_pinhole_camera_parameters(0.0)
        d_x = (ijs[..., 1] - cx) / fx
        d_y = (ijs[..., 0] - cy) / fy
        d_z = torch.ones_like(d_x)

        if convention == "opengl":
            d_y = -d_y
            d_z = -torch.ones_like(d_x)
        elif convention == "opencv":
            pass
        else:
            raise ValueError(f"Unsupported camera convention {convention}.")

        dirs = torch.stack([d_x, d_y, d_z], dim=-1)
        dirs = torch.nn.functional.normalize(dirs, dim=-1)
        return dirs

    def scaled_camera(self, scale_factor: float):
        return Camera(
            int(self.width * scale_factor),
            int(self.height * scale_factor),
            self.fx * scale_factor,
            self.fy * scale_factor,
            self.cx * scale_factor,
            self.cy * scale_factor,
        )

    def sample_ijs_uniform(
        self,
        ijs: torch.Tensor,
        num_samples: int,
        near_distances: Optional[Union[float, torch.Tensor]] = None,
        far_distances: Optional[Union[float, torch.Tensor]] = None,
        weights: Optional[torch.Tensor] = None,
        boundaries: Optional[torch.Tensor] = None,
        convention: str = "opengl",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample 3D points along camera rays from row and column indices.

        Args:
            ijs:
                Image coordinates to render. Each row contains row and column index.
                Shape (...,2).
            num_samples: Number of samples along the ray.
            near_distances:
                Either scalar value representing minimum depth of sampled points or
                tensor with shape (...) representing minimum depth per ray. If None,
                boundaries and weights will be used.
            far_distances:
                Either scalar value representing maximum distance of sampled points or
                tnesor with shape (...) representing maximum depth per ray. If None,
                boundaries and weights will be used.
            boundaries:
                Distance boundaries of bins for weighed stratified sampling. Should be
                sorted. If None, stratified sampling between near and far will be used.
                Shape (..., num_bins+1).
            weights:
                Weight of bins specified by boundaries. Should sum up to 1.
                If None, stratified sampling between near and far will be used.
                Shape (..., num_bins).
            convention:
                The camera frame convention to use. One of:
                    "opengl": x right, y up, z back
                    "opencv": x right, y down, z forward

        Returns:
            Points in camera frame along the rays. Shape (...,num_samples,3).
            Euclidean distance of each point from the origin. Shape (...,num_samples).
        """
        device = ijs.device
        leading_dims = ijs.shape[:-1]
        if (weights is None) != (boundaries is None):
            raise ValueError("Either both or none of weights and boundaries must be None.")

        dirs = self.ijs_to_directions(ijs, convention=convention)

        if isinstance(far_distances, numbers.Number):
            far_distances = torch.tensor(far_distances, device=device).expand(leading_dims)
        if isinstance(near_distances, numbers.Number):
            near_distances = torch.tensor(near_distances, device=device).expand(leading_dims)

        if boundaries is None:
            deltas = (far_distances - near_distances) / num_samples
            boundaries = torch.linspace(0.0, 1.0, steps=num_samples + 1, device=device)
            boundaries = boundaries[None] * (far_distances - near_distances)[..., None]
            distances = (
                deltas[..., None] * torch.rand(*leading_dims, num_samples, device=device)
                + boundaries[..., :-1]
            ) + near_distances[..., None]
        else:
            # weighted sampling from bins specified by their boundaries along each ray
            # boundaries are depth values
            cum_weights = torch.cumsum(weights, dim=-1) + 1e-3
            bins = torch.searchsorted(
                cum_weights, torch.rand(*leading_dims, num_samples, device=device)
            )
            deltas = boundaries[..., 1:] - boundaries[..., :-1]
            bin_starts = torch.gather(boundaries, -1, bins)
            bin_sizes = torch.gather(deltas, -1, bins)
            distances = bin_starts + bin_sizes * torch.rand(
                *leading_dims, num_samples, device=device
            )

        points = dirs.unsqueeze(-2) * distances.unsqueeze(-1)
        return points, distances

    def distance_to_depth(
        self,
        distances: torch.Tensor,
        ijs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert distance values to depth values along rays.

        Args:
            distances:
                Distance values to convert.
                Shape (H,W) if ijs is None, (...) otherwise.
            ijs:
                Optional pixel position (row, column) of the distance values.
                Shape (..., 2).

        Returns: Corresponding depth value for distance values. Same shape as distance.
        """
        if ijs is None:
            ijs = torch.cartesian_prod(
                torch.arange(self.height, device=distances.device),
                torch.arange(self.width, device=distances.device),
            )
        dirs = self.ijs_to_directions(ijs, convention="opencv")
        return distances * dirs[..., 2]

    def depth_to_distance(
        self, depths: torch.Tensor, ijs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convert depth values to distance values along rays.

        Args:
            depths:
                Depth values to convert.
                Shape (H,W) if ijs is None, (...) otherwise.
            ijs:
                Optional pixel position (row, column) of the depth values.
                Shape (..., 2).

        Returns: Corresponding distance value for depth values. Same shape as depths.
        """
        if ijs is None:
            ijs = torch.cartesian_prod(
                torch.arange(self.height, device=depths.device),
                torch.arange(self.width, device=depths.device),
            )
        dirs = self.ijs_to_directions(ijs, convention="opencv")
        return depths / dirs[..., 2]

    def depth_to_pointcloud(
        self,
        depth_image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        convention: str = "opengl",
        return_ijs: bool = False,
    ) -> torch.Tensor:
        """Convert depth image to pointcloud.

        Only pixels with depth != 0 will be backprojected.

        Args:
            depth_image:
                The depth image to convert to pointcloud. In meters. Shape (H,W).
            normalize: Whether to normalize the pointcloud with 0 centroid.
            mask:
                Only points with mask != 0 will be added to pointcloud.
                No masking will be performed if None.
            convention:
                The camera frame convention to use. One of:
                    "opengl": x right, y up, z back
                    "opencv": x right, y down, z forward

        Returns:
            points:
                The pointcloud in the camera frame, in specified convention, shape (N,3).
            ijs:
                Only returned if return_ijs=True. row, column pairs for each point. Shape (N,3).
        """
        fx, fy, cx, cy, _ = self.get_pinhole_camera_parameters(0.0)

        if mask is None:
            ijs = torch.nonzero(depth_image)
        else:
            ijs = torch.nonzero(depth_image * mask)
        depth_values = depth_image[ijs[:, 0], ijs[:, 1]]
        points = torch.stack((ijs[:, 1].float(), ijs[:, 0].float(), depth_values), dim=-1)

        if convention == "opengl":
            final_points = torch.empty_like(points)
            final_points[:, 0] = (points[:, 0] - cx) * points[:, 2] / fx
            final_points[:, 1] = -(points[:, 1] - cy) * points[:, 2] / fy
            final_points[:, 2] = -points[:, 2]
        elif convention == "opencv":
            final_points = torch.empty_like(points)
            final_points[:, 0] = (points[:, 0] - cx) * points[:, 2] / fx
            final_points[:, 1] = (points[:, 1] - cy) * points[:, 2] / fy
            final_points[:, 2] = points[:, 2]
        else:
            raise ValueError(f"Unsupported camera convention {convention}.")

        if return_ijs:
            return final_points, ijs
        return final_points
