"""Script to run neural graph mapping on a SLAM dataset."""

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import pathlib
import random
import time
from collections import defaultdict, namedtuple
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rerun as rr
import tabulate
import torch
import tqdm
import wandb
import yoco
from pytorch3d import transforms as p3dt
from pytorch3d.io.ply_io import _save_ply as save_ply
from pytorch3d.ops.ball_query import ball_query
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from torch.utils.data import DataLoader

from neural_graph_mapping import (
    camera,
    evaluation,
    geometry,
    graph,
    losses,
    slam_dataset,
    utils,
)

Target = namedtuple(
    "Target",
    [
        "ijs",
        "c2ws",
        "near_distances",
        "far_distances",
        "gt_distances",
        "field_ids",
        "rgbds",
        "rgb_mask",
        "depth_mask",
        "term_probs",
        "term_mask",
    ],
)
Prediction = namedtuple(
    "Prediction",
    [
        "rgbds",
        "color_vars",
        "depth_vars",
        "term_probs",
        "freespace_geometry",
        "tsdf_residuals",
    ],
)


def _wandb_log(*args, **kwargs):
    try:
        return wandb.log(*args, **kwargs)
    except BrokenPipeError:
        print("Logging to wandb failed. Investigate if this causes issues.")


wandb.log = _wandb_log


def _mean_metric_dict(metric_dicts: list[dict]) -> dict:
    """Compute arithmetic mean of metric dictionaries."""
    mean_metric_dict = defaultdict(lambda: 0)
    metric_counts = defaultdict(lambda: 0)
    for metric_dict in metric_dicts:
        for metric_name, metric_value in metric_dict.items():
            mean_metric_dict[metric_name] += metric_value
            metric_counts[metric_name] += 1
    for metric_name in mean_metric_dict:
        mean_metric_dict[metric_name] /= metric_counts[metric_name]
    return dict(mean_metric_dict)


class NeuralGraphMap:
    """Neural Graph Map class. Includes inference and training."""

    _camera: camera.Camera
    _dataset: Optional[slam_dataset.SLAMDataset]

    def __init__(self, config: dict) -> None:
        """Parse config."""
        self._read_config(config)

        self._dataset = None

        self._init_model()
        if self._model_path is not None:
            self.load_model(self._model_path)
        else:
            self._init_map_dict()
            self._init_graph()

        self.train()  # by default in train mode

    def _read_config(self, config: dict) -> None:
        self._config = config
        self._dataset_type = utils.str_to_object(config["dataset_type"])
        self._dataset_config = config["dataset_config"]
        self._model_type = utils.str_to_object(config["model_type"])
        self._model_kwargs = config["model_kwargs"]
        self._field_type = utils.str_to_object(config["model_kwargs"]["field_type"])
        self._field_kwargs = config["model_kwargs"]["field_kwargs"]
        self._device = config["device"]
        self._learning_rate = config["learning_rate"]
        self._adam_eps = config["adam_eps"]
        self._adam_weight_decay = config.get("adam_weight_decay", 0.0)
        self._freeze_model = config["freeze_model"]
        self._termination_weight = config["termination_weight"]
        self._photometric_weight = config["photometric_weight"]
        self._photometric_loss = config["photometric_loss"]
        self._depth_weight = config["depth_weight"]
        self._depth_loss = config["depth_loss"]
        self._freespace_weight = config["freespace_weight"]
        self._tsdf_weight = config.get("tsdf_weight", 0.0)
        self._geometry_mode = config["depth_loss"]
        self._field_radius = config["field_radius"]
        self._block_size = config["block_size"]
        self._pixel_block_size = config["pixel_block_size"]
        self._num_train_fields = config["num_train_fields"]
        self._num_rays_per_field = config["num_rays_per_field"]
        self._num_samples_depth_guided = config["num_samples_depth_guided"]
        self._range_depth_guided = config.get("range_depth_guided", None)

        self._preview_res_factor = config["preview_res_factor"]
        self._render_frames = config["render_frames"]
        self._render_frame_freq = config["render_frame_freq"]
        self._extract_mesh_frame_freq = config["extract_mesh_frame_freq"]
        self._extract_mesh_frames = config["extract_mesh_frames"]
        self._extract_mesh_fields = config["extract_mesh_fields"]
        self._log_iteration_freq = config["log_iteration_freq"]
        self._num_iterations_per_frame = config["num_iterations_per_frame"]
        self._rerun_vis = config["rerun_vis"]
        self._rerun_save = config["rerun_save"]
        self._rerun_connect_addr = config["rerun_connect_addr"]
        self._rerun_field_details = config.get("rerun_field_details", None)
        self._model_path = config.get("model", None)
        self._max_depth = config.get("max_depth", None)
        self._disable_relative_fields = config["disable_relative_fields"]
        self._keyframes_only = config.get("keyframes_only", False)
        self._store_intermediate_meshes = config.get("store_intermediate_meshes", False)

        self._disable_eval = config.get("disable_eval", False)
        self._render_vis = config.get("render_vis", False)
        self._metric_dicts_for_chunks = []
        self._geometry_mode = config["geometry_mode"]
        self._truncation_distance = config.get("truncation_distance", None)

        if self._range_depth_guided is None:
            self._range_depth_guided = self._truncation_distance

        self._color_factor = config.get("color_factor", 1.0)
        self._geometry_factor = config.get("geometry_factor", 1.0)
        self._single_field_id = config["single_field_id"]

        self._update_mode = config["update_mode"]

        # train specific (used by default)
        self._train_near_distance = config["near_distance"]
        self._train_far_distance = config["far_distance"]
        self._train_num_samples = config["num_samples_coarse"]

        # eval specific

        # NEWTON
        self._eval_near_distance = config.get("eval_near_distance", 0.0)
        self._eval_far_distance = config.get("eval_far_distance", 8.0)
        self._eval_num_samples = config.get("eval_num_samples", None)
        self._eval_ratio = config.get("eval_ratio", 0.0)
        self._eval_chunk_freq = config.get("eval_chunk_freq", None)
        self._eval_render_metrics = config.get("eval_metrics", [])

        # COSLAM
        self._eval_mesh = config.get("eval_mesh", False)
        self._eval_mesh_num_points = config.get("eval_mesh_num_points", 200000)
        self._eval_mesh_alignment = config.get("eval_mesh_alignment", True)
        self._eval_culling_method = config.get("eval_culling_method", None)

        if self._eval_num_samples is None:
            if self._num_samples_depth_guided > 0:
                self._sample_spacing = (
                    2 * self._range_depth_guided / self._num_samples_depth_guided
                )
            else:
                self._sample_spacing = 2 * self._field_radius / self._train_num_samples
            eval_distance = self._eval_far_distance - self._eval_near_distance
            self._eval_num_samples = int(eval_distance / self._sample_spacing)

        self._eval_crop = config.get("eval_crop", None)
        self._eval_store_details = config.get("eval_store_details", True)
        self._eval_details = []

        # initialize state
        self._last_update = None
        self._optim_state = None
        self._run_name = None

        # logging
        utils.benchmark.enabled = config["benchmark"]
        logging.basicConfig(level=config["loglevel"])

    def _init_model(self) -> None:
        """Initialize model and load weights if self._model_path is not None."""
        if self._single_field_id:
            self._model = self._field_type(**self._field_kwargs)
            self._model.to(self._device)
        else:
            self._model = self._model_type(**self._model_kwargs)
            self._model.to(self._device)

    def _init_map_dict(
        self,
    ) -> None:
        """Initialize self._global_map_dict."""
        self._global_map_dict = {
            "positions": torch.zeros(32, 3, device=self._device),
            "orientations": torch.zeros(32, 4, device=self._device),
            "kf_ids": torch.zeros(
                32,
                device=self._device,
                dtype=torch.long,
            ),
            "training_iterations": torch.zeros(32, device=self._device, dtype=torch.long),
            "num": 0,
        }
        self._kf2fields = defaultdict(set)

    def _init_graph(self) -> None:
        """Initailize self._graph."""
        self._graph = {}

    def _extend_map_dict(self, required_size: int) -> None:
        """Extend self._global_map_dict by doubling its memory allocation."""
        current_size = self._global_map_dict["positions"].shape[0]
        r = math.ceil(required_size / current_size)
        self._global_map_dict["positions"] = self._global_map_dict["positions"].repeat((r, 1))
        self._global_map_dict["orientations"] = self._global_map_dict["orientations"].repeat(
            (r, 1)
        )
        self._global_map_dict["kf_ids"] = self._global_map_dict["kf_ids"].repeat(r)
        self._global_map_dict["training_iterations"] = self._global_map_dict[
            "training_iterations"
        ].repeat((r,))

    @utils.benchmark
    @torch.no_grad()
    def _extend_global_map_dict(
        self,
        depth_image: torch.Tensor,
        frame_id: int,
        c2w: torch.Tensor,
        camera: camera.Camera,
        active_map_dict: Optional[dict] = None,
    ) -> dict:
        """Ensure fields cover depth image or add new fields."""
        logging.debug("NeuralGraphMap._extend_global_map_dict")

        xyz_cam = camera.depth_to_pointcloud(depth_image, convention="opengl")
        xyz_world = utils.transform_points(xyz_cam, c2w)

        bb_max = xyz_world.max(dim=0)[0]
        bb_min = xyz_world.max(dim=0)[0]

        self._bb_max = torch.where(bb_max > self._bb_max, bb_max, self._bb_max)
        self._bb_min = torch.where(bb_min < self._bb_min, bb_min, self._bb_min)

        # check uncovered points exactly
        if active_map_dict is not None:
            _, idx, _ = ball_query(
                xyz_world.unsqueeze(0),
                active_map_dict["positions"].unsqueeze(0),
                K=1,
                radius=self._field_radius,
                return_nn=False,
            )
            xyz_world = xyz_world[idx[0, :, 0] == -1]

        # use grid to cover these points
        cell_size = 2 * self._field_radius / math.sqrt(3)
        shift = torch.empty((3,), device=self._device).uniform_(0.0, cell_size)
        to_be_covered_ijk = ((xyz_world + shift) / cell_size).floor().unique(dim=0)

        if active_map_dict is not None:
            global_field_positions = active_map_dict["positions"]
            covered_ijk = ((global_field_positions + shift) / cell_size).floor().unique(dim=0)
        else:
            covered_ijk = torch.empty(0, 3, device=self._device)

        combined_ijk = torch.cat((to_be_covered_ijk, covered_ijk))
        _, inv, counts = combined_ijk.unique(dim=0, return_inverse=True, return_counts=True)
        new_ijk = to_be_covered_ijk[counts[inv[: len(to_be_covered_ijk)]] == 1]

        num_prev = self._global_map_dict["num"]
        num_new = len(new_ijk)

        if num_new == 0:
            return

        if (
            self._global_map_dict["positions"].shape[0]
            <= self._global_map_dict["num"] + num_new
        ):
            self._extend_map_dict(self._global_map_dict["num"] + len(new_ijk))

        cell_center = (new_ijk - shift + 0.5) * cell_size

        start = self._global_map_dict["num"]
        end = self._global_map_dict["num"] + num_new

        self._global_map_dict["positions"][start:end] = cell_center
        self._global_map_dict["orientations"][start:end] = torch.zeros(
            num_new, 4, device=self._device
        )
        self._global_map_dict["orientations"][start:end, 0] = 1.0
        self._global_map_dict["kf_ids"][start:end] = frame_id
        self._global_map_dict["num"] += num_new
        self._global_map_dict["training_iterations"][start:end] = torch.zeros(
            num_new, device=self._device, dtype=torch.long
        )

        self._add_fields(num_new)

        self._kf2fields[frame_id] = {
            field_id for field_id in range(num_prev, self._global_map_dict["num"])
        }

    def _init_optimizer(self) -> torch.optim.Optimizer:
        if self._single_field_id:
            self._optimizer = torch.optim.Adam(
                self._model.parameters(),
                lr=self._learning_rate,
                eps=self._adam_eps,
                weight_decay=self._adam_weight_decay,
            )
        else:
            # add parameters as they are initialized
            self._optimizer = torch.optim.Adam(
                [torch.autograd.Variable(torch.tensor(0.0, device="cuda"))],
                lr=self._learning_rate,
                eps=self._adam_eps,
                weight_decay=self._adam_weight_decay,
            )

    @utils.benchmark
    def _add_fields(self, num_new: int) -> None:
        if self._single_field_id is not None:
            return

        self._model.add_fields(num_new)

        # NOTE below we manually maintain the optimizer's internal state
        num_total = self._global_map_dict["num"]
        all_params = self._model.all_fields_params
        param_names = all_params.keys()
        new_state = {}
        old_state = self._optim_state

        for p in param_names:
            new_state[p] = {}
            new_state[p]["step"] = torch.tensor(0.0)
            new_state[p]["exp_avg"] = torch.zeros_like(all_params[p])
            new_state[p]["exp_avg_sq"] = torch.zeros_like(all_params[p])

            if old_state is not None:
                new_state[p]["step"] = old_state[p]["step"]
                new_state[p]["exp_avg"][:-num_new] = old_state[p]["exp_avg"]
                new_state[p]["exp_avg_sq"][:-num_new] = old_state[p]["exp_avg_sq"]

        self._optim_state = new_state

    def _preview_camera(self) -> camera.Camera:
        """Returns camera with reduced resolution."""
        preview_camera = copy.deepcopy(self._camera)
        preview_camera.width = int(preview_camera.width * self._preview_res_factor)
        preview_camera.height = int(preview_camera.height * self._preview_res_factor)
        preview_camera.fx *= self._preview_res_factor
        preview_camera.fy *= self._preview_res_factor
        preview_camera.cx *= self._preview_res_factor
        preview_camera.cy *= self._preview_res_factor
        return preview_camera

    @torch.no_grad()
    def render_image(
        self,
        c2w: torch.Tensor,
        camera: camera.Camera,
        progressbar: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Render image at specified camera position.

        Args:
            c2w:
                Transformation matrix from camera to world coordinates.
                OpenGL convention. Shape (4,4).
            camera: Camera to render the image.
            progressbar: Whether to show progressbar.

        Returns:
            rgbds:
                Rendered RGBD values for each image coordinate. 0-1, meters.
                Shape (H, W, 4).
            d_vars: Variance of the depth values. Shape (H, W).
        """
        h, w = camera.height, camera.width
        ijs = torch.cartesian_prod(
            torch.arange(h, device=self._device),
            torch.arange(w, device=self._device),
        )
        rgbds, _, d_vars, _, _, _ = utils.batched_evaluation(
            lambda x: self._render_ijs(x, c2w, camera),
            ijs,
            block_size=self._pixel_block_size,
            progressbar=progressbar,
        )
        rgbds = rgbds.reshape(h, w, 4)
        d_vars = d_vars.reshape(h, w)
        return rgbds, d_vars

    @utils.benchmark
    def _render_ijs(
        self,
        ijs: torch.LongTensor,
        c2ws: torch.Tensor,
        camera: camera.Camera,
        field_ids: Optional[torch.LongTensor] = None,
        use_vmap: bool = False,
        near_distances: Optional[torch.Tensor] = None,
        far_distances: Optional[torch.Tensor] = None,
        gt_distances: Optional[torch.Tensor] = None,
        overwrite_samples_behind_camera: bool = True,
    ) -> Prediction:
        """Render image at specified image coordinates.

        This supports independent rendering of single fields by providing the field_ids.
        Otherwise all fields will be used.

        Args:
            ijs:
                Image coordinates to render. Each row contains row and column index.
                Shape (num_fields, num_rays, 2) or (num_rays, 2).
            c2ws:
                Transformation matrix from camera to world coordinates.
                OpenGL convention. Shape (4,4) or (..., 4, 4) matching ijs leading
                dimensions.
            camera: Camera to render the image.
            field_ids:
                Shape (num_fields,).
                If None, use all fields.
            use_vmap:
                Whether to use vmap to render and evaluate fields independently.
            near_distances:
                Optional per-ray near distances to use for sampling.
                Shape (...) matching ijs leading dimensions.
            far_distances:
                Optional per-ray far distances to use for sampling.
                Shape (...) matching ijs leading dimesions.
            gt_distances:
                Optional per-ray ground-truth distances to use for depth guided samples.
                0.0 will be interpreted as unavailable.
                Uniform sampling between near and far will be used in that case.
                Shape (...) matching ijs leading dimensions.
            overwrite_samples_behind_camera:
                If True, samples behind the camera (which can be generated when nears is
                negative) will be overwritten with empty space.

        Returns:
            rgbds:
                Rendered RGBD values for each image coordinate. 0-1, meters.
                Shape (N, 4).
            d_vars: Variance of the depth values. Shape (N,).
            term_probs:
                Probability of ray terminating before far plane.
        """
        if near_distances is None or (near_distances >= 0).all():
            overwrite_samples_behind_camera = False

        if use_vmap and field_ids is None:
            raise ValueError("field_ids=None only supported for use_vmap=False")
        elif use_vmap and field_ids is not None:
            self._set_vmap_fields(field_ids)

        if field_ids is not None:
            field_positions = self._global_map_dict["positions"][field_ids]
            field_orientations = self._global_map_dict["orientations"][field_ids]
        else:
            num = self._global_map_dict["num"]
            field_positions = self._global_map_dict["positions"][:num]
            field_orientations = self._global_map_dict["orientations"][:num]

        if c2ws.dim() == 2:
            c2ws = c2ws[None]

        points_cam, sample_distances = camera.sample_ijs_uniform(
            ijs,
            self._num_samples,
            self._near_distance if near_distances is None else near_distances,
            self._far_distance if far_distances is None else far_distances,
            convention="opengl",
        )

        if gt_distances is not None and self._num_samples_depth_guided > 0:
            mask = (
                (gt_distances == 0.0)
                + (near_distances > gt_distances)
                + (far_distances < gt_distances)
            )
            depth_guided_near_distances = gt_distances - self._range_depth_guided
            depth_guided_far_distances = gt_distances + self._range_depth_guided
            depth_guided_near_distances[mask] = near_distances[mask]
            depth_guided_far_distances[mask] = far_distances[mask]
            guided_points_cam, guided_sample_distances = camera.sample_ijs_uniform(
                ijs,
                self._num_samples_depth_guided,
                depth_guided_near_distances,
                depth_guided_far_distances,
                convention="opengl",
            )
            points_cam = torch.cat([points_cam, guided_points_cam], dim=-2)
            sample_distances = torch.cat([sample_distances, guided_sample_distances], dim=-1)
            sample_distances, sort_indices = torch.sort(sample_distances, dim=-1)
            points_cam = torch.gather(
                points_cam,
                dim=-2,
                index=sort_indices.unsqueeze(-1).expand(-1, -1, -1, 3),
            )

        points_world = utils.transform_points(points_cam, c2ws.unsqueeze(-3))

        if (
            self._rerun_field_details is not None
            and (field_ids == self._rerun_field_details).sum() != 0
        ):
            if self._rerun_vis:
                rr.log(
                    "slam/sample_points",
                    rr.Points3D(
                        points_world[field_ids == self._rerun_field_details]
                        .reshape(-1, 3)
                        .cpu(),
                    ),
                )
                term_probs = self._target.term_probs[field_ids == self._rerun_field_details][0]
                rr.log(
                    "slam/sample_points_term_probs",
                    rr.Points3D(
                        points_world[field_ids == self._rerun_field_details]
                        .reshape(-1, 3)
                        .numpy(force=True),
                        class_ids=term_probs[:, None]
                        .expand(-1, self._num_samples + self._num_samples_depth_guided)
                        .reshape(-1)
                        .long()
                        .numpy(force=True),
                    ),
                )

        if self._single_field_id is None:
            if use_vmap:
                sample_outs = self._model(
                    points_world.reshape(len(field_ids), -1, 3),
                    field_positions,
                    field_orientations,
                    field_ids,
                    use_vmap,
                ).view(len(field_ids), points_world.shape[1], points_world.shape[2], -1)
            else:
                leading_dims = points_world.shape[:-1]
                sample_outs = utils.batched_evaluation(
                    lambda x: self._model(
                        x, field_positions, field_orientations, field_ids, use_vmap
                    ),
                    points_world.reshape(-1, 3),
                    self._block_size,
                )
                sample_outs = sample_outs.view(*leading_dims, 4)
        else:
            field_points_world = points_world[field_ids == self._single_field_id]
            field_position = field_positions[field_ids == self._single_field_id]
            field_orientation = field_orientations[field_ids == self._single_field_id]
            sample_distances = sample_distances[field_ids == self._single_field_id][0]
            points_cam = points_cam[field_ids == self._single_field_id][0]
            field_points_local = field_points_world - field_position
            field_points_local = quaternion_apply(
                quaternion_invert(field_orientation), field_points_local
            )
            sample_outs = self._model(field_points_local.view(-1, 3)).view(
                field_points_local.shape[1], field_points_local.shape[2], 4
            )

        sample_colors = self._color_factor * sample_outs[..., :3]
        sample_densities = sample_outs[..., 3]
        sample_depths = -points_cam[..., 2]

        if overwrite_samples_behind_camera:
            if self._geometry_mode == "occupancy":
                sample_densities[points_cam[..., 2] > 0] = -100.0
            elif self._geometry_mode == "density":
                sample_densities[points_cam[..., 2] > 0] = -100.0
            elif self._geometry_mode == "neus":
                sample_densities[points_cam[..., 2] > 0] = 1.0
            elif self._geometry_mode == "nrgbd":
                sample_densities[points_cam[..., 2] > 0] = 1.0

        if self._freespace_weight != 0.0 and gt_distances is not None:
            mask = sample_distances < (gt_distances[..., None] - self._truncation_distance) * (
                gt_distances[..., None] != 0.0
            )
            freespace_distances = sample_densities[mask] * self._truncation_distance
        else:
            freespace_distances = None

        if self._tsdf_weight != 0.0 and gt_distances is not None:
            deltas = gt_distances[..., None] - sample_distances
            mask = (torch.abs(deltas) < self._truncation_distance) * (
                gt_distances[..., None] != 0.0
            )
            tsdf_residuals = sample_densities[mask] * self._truncation_distance - deltas[mask]
        else:
            tsdf_residuals = None

        if self._geometry_mode == "neus" and use_vmap:
            neus_isds = 1.0 / torch.abs(
                self._model.vmap_fields_params["_neus_sd"].view(-1, 1, 1)
            )
        elif self._geometry_mode == "neus" and use_vmap:
            breakpoint()
        else:
            neus_isds = None

        colors, depths, color_vars, depth_vars, term_probs, _ = self._quadrature(
            sample_colors=sample_colors,
            sample_geometries=sample_densities,
            sample_distances=sample_distances,
            sample_depths=sample_depths,
            neus_isds=neus_isds,
        )

        rgbds = torch.cat([colors, depths[..., None]], dim=-1)
        return Prediction(
            rgbds=rgbds,
            color_vars=color_vars,
            depth_vars=depth_vars,
            term_probs=term_probs,
            freespace_geometry=freespace_distances,
            tsdf_residuals=tsdf_residuals,
        )

    @utils.benchmark
    @torch.no_grad()
    def _set_vmap_fields(self, field_ids: torch.Tensor) -> None:
        if self._single_field_id is not None:
            return

        self._model.set_vmap_fields(field_ids)

        # only update the vmapped fields
        new_params = list(self._model.vmap_fields_params.values())
        param_names = self._model.vmap_fields_params.keys()
        for new_param in new_params:
            new_param.requires_grad_()

        if self._optimizer is None:
            return

        if len(self._optimizer.state) != 0:
            old_params = self._optimizer.param_groups[0]["params"]

            for param_name, old_param, new_param in zip(param_names, old_params, new_params):
                if old_param not in self._optimizer.state:
                    continue
                if isinstance(self._optimizer, torch.optim.Adam):
                    self._optimizer.state[new_param] = {
                        "step": self._optim_state[param_name]["step"],
                        "exp_avg": self._optim_state[param_name]["exp_avg"][field_ids],
                        "exp_avg_sq": self._optim_state[param_name]["exp_avg_sq"][field_ids],
                    }
                elif isinstance(self._optimizer, torch.optim.RMSprop):
                    self._optimizer.state[new_param] = {
                        "step": self._optim_state[param_name]["step"],
                        "square_avg": self._optim_state[param_name]["square_avg"][field_ids],
                        "momentum_buffer": self._optim_state[param_name]["momentum_buffer"][
                            field_ids
                        ],
                    }
                del self._optimizer.state[old_param]

        self._optimizer.param_groups[0]["params"] = new_params

    def _quadrature(
        self,
        sample_colors: torch.Tensor,
        sample_geometries: torch.Tensor,
        sample_distances: torch.Tensor,
        sample_depths: torch.Tensor,
        neus_isds: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Compute color, and depth of ray based on samples along it.

        Args:
            sample_colors:
                Sampled color along rays, shape (..., num_samples_per_ray, 3).
            sample_geometries:
                Sampled geometries along rays, shape (..., num_samples_per_ray).
                This can be SDF, density, or occupancy logit.
            sample_distances:
                Distance of samples along the ray, shape (..., num_samples_per_ray).
            sample_depths:
                Depths of samples along camera z-axis, shape (..., num_samples_per_ray).
            neus_istds:
                Inverse standard deviation for each sample. Broadcastable to sample_geometries.

        Returns:
            ray_colors: Expected color of rays, shape (...,3).
            ray_depths:
                Exptected termination depth (i.e., along camera z-axis) of rays,
                shape (...,).
            ray_depth_vars: Depth variance of rays, shape (...,).
            ray_term_probs:
                Probabilities of terminating between the provided samples per ray.
                Shape (...,).
            sample_weights: Weights of samples (..., num_samples_per_ray).
        """
        # background handling and deltas / midpoints
        leading_dims = sample_geometries.shape[:-1]

        if self._geometry_mode == "density":
            deltas = sample_distances[..., 1:] - sample_distances[..., :-1]
            occ_probs = 1 - torch.exp(-deltas * torch.relu(sample_geometries[..., :-1]))
            last_index = -1
        elif self._geometry_mode == "occupancy":
            occ_probs = torch.sigmoid(self._geometry_factor * sample_geometries)
            last_index = None
        elif self._geometry_mode == "neus":
            tno = torch.sigmoid(neus_isds * self._geometry_factor * sample_geometries)
            occ_probs = torch.clamp_min(
                (tno[..., :-1] - tno[..., 1:]) / (tno[..., :-1] + 1e-5), 0
            )
            last_index = -1
        elif self._geometry_mode == "nrgbd":
            temp = self._geometry_factor * sample_geometries
            occ_probs = 4 * torch.sigmoid(temp) * torch.sigmoid(-temp)
            last_index = None

        non_term_probs = torch.cat(
            [
                occ_probs.new_ones(*leading_dims, 1),
                torch.cumprod(1 - occ_probs[..., :-1], dim=-1),
            ],
            dim=-1,
        )
        sample_weights = occ_probs * non_term_probs  # termination probability

        # give remaining weight to background (non term prob)
        bg_weight = 1 - torch.sum(sample_weights, dim=-1)

        ray_colors = torch.sum(
            sample_colors[..., :last_index, :] * sample_weights[..., None], dim=-2
        )
        ray_depths = torch.sum(sample_depths[..., :last_index] * sample_weights, dim=-1)

        ray_color_vars = torch.sum(
            sample_weights[..., None]
            * (ray_colors.unsqueeze(-2) - sample_colors[..., :last_index, :]) ** 2,
            dim=-2,
        )

        ray_depth_vars = torch.sum(
            sample_weights * (ray_depths[..., None] - sample_depths[..., :last_index]) ** 2,
            dim=-1,
        )
        return (
            ray_colors,
            ray_depths,
            ray_color_vars,
            ray_depth_vars,
            1.0 - bg_weight,
            # sample_weights.max(dim=-1)[0],
            sample_weights,
        )

    def _get_ijs(self, cam: camera.Camera) -> torch.Tensor:
        return torch.cartesian_prod(
            torch.arange(cam.height, device=self._device),
            torch.arange(cam.width, device=self._device),
        )

    def _get_run_name(self) -> str:
        if self._run_name is None:
            dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            self._run_name = f"{self._model_type.__name__}_{self._dataset}_{dt_str}"
        return self._run_name

    def _closest_kf_id(self, frame_id: int) -> int:
        return max([kf for kf in self._graph.keys() if frame_id - kf >= 0])

    @utils.benchmark
    def _get_active_field_ids(self, frame_id: int, max_edges: int = 100) -> torch.Tensor:
        closest_kf_id = self._closest_kf_id(frame_id)
        neighbors = graph.get_neighbors(
            self._graph, {closest_kf_id}, max_edges=max_edges, include_queries=True
        )
        neighbors_with_field = [kf for kf in neighbors if self._kf2fields[kf]]
        indices = set.union(*(self._kf2fields[kf] for kf in neighbors_with_field))
        indices = torch.tensor(list(indices), device=self._device, dtype=torch.long)
        return indices

    @utils.benchmark
    def _get_active_map_dict(self, frame_id: int) -> dict:
        indices = self._get_active_field_ids(frame_id)
        return {
            "positions": self._global_map_dict["positions"][indices],
            "orientations": self._global_map_dict["orientations"][indices],
            "field_ids": indices,
            "kf_ids": self._global_map_dict["kf_ids"][indices],
            "num": len(indices),
            "training_iterations": self._global_map_dict["training_iterations"][indices],
        }

    @utils.benchmark
    def _get_kf2ws(self, at_frame_id: int) -> torch.Tensor:
        """Return keyframe poses."""
        return self._dataset.get_slam_c2ws(at_frame_id=at_frame_id).to(self._device)

    def _relative_map_dict_to_absolute(
        self, relative_map_dict: dict, kf2ws: torch.Tensor
    ) -> dict:
        kf_ids = relative_map_dict["kf_ids"]
        kf2ws = kf2ws.index_select(dim=0, index=kf_ids).to_dense()
        relative_positions = relative_map_dict["positions"]  # (num_fields, 3)
        relative_orientations = relative_map_dict["orientations"]  # (num_fields, 4)

        absolute_positions = utils.transform_points(relative_positions, kf2ws)
        absolute_orientations = utils.transform_quaternions(relative_orientations, kf2ws)

        absolute_map_dict = {
            "positions": absolute_positions,
            "orientations": absolute_orientations,
            "kf_ids": relative_map_dict["kf_ids"],
            "num": relative_map_dict["num"],
            "training_iterations": relative_map_dict["training_iterations"],
        }
        return absolute_map_dict

    def _absolute_map_dict_to_relative(
        self, absolute_map_dict: dict, kf2ws: torch.Tensor
    ) -> dict:
        kf_ids = absolute_map_dict["kf_ids"]
        kf2ws_subset = kf2ws.index_select(dim=0, index=kf_ids).to_dense()

        w2kfs = kf2ws_subset.inverse()  # (num_fields, 4, 4)

        absolute_positions = absolute_map_dict["positions"]  # (num_fields, 3)
        absolute_orientations = absolute_map_dict["orientations"]  # (num_fields, 4)

        relative_positions = utils.transform_points(absolute_positions, w2kfs)
        relative_orientations = utils.transform_quaternions(absolute_orientations, w2kfs)

        relative_map_dict = {
            "positions": relative_positions,
            "orientations": relative_orientations,
            "kf_ids": absolute_map_dict["kf_ids"],
            "num": absolute_map_dict["num"],
            "training_iterations": absolute_map_dict["training_iterations"],
        }
        return relative_map_dict

    @utils.benchmark
    def _update_graph(self, current_frame_id: int) -> None:
        """Update pose graph and handle removed keyframes if necessary."""
        new_graph = self._dataset.get_slam_essential_graph(current_frame_id)

        # nothing else to do if this is the first graph
        if self._last_update is None:
            self._graph = new_graph
            self._last_update = current_frame_id
            self._prev_kf2ws = self._get_kf2ws(self._last_update)
            return

        prev_kfs = self._kf_ids
        new_kfs = set(new_graph.keys())
        removed_kfs = prev_kfs - new_kfs

        new_kfs = prev_kfs - removed_kfs
        if self._dataset.is_keyframe(current_frame_id):
            new_kfs.add(current_frame_id)

        for removed_kf in removed_kfs:
            self._kf_ids.remove(removed_kf)

            if self._update_mode == "single_view":
                #  remove from self._kfs if not in graph
                del self._kfs[removed_kf]
            elif self._update_mode == "multi_view":
                self._free_rgbd_tensor_indices.append(removed_kf)
                self._nc_frame_id_tensor[self._nc_frame_id_tensor == removed_kf] = -1

            # rewire kf-field edge with self._kf2fields
            kf_after = min([i for i in new_kfs if i >= removed_kf], default=None)
            kf_before = max([i for i in new_kfs if removed_kf >= i])

            new_anchor_kf = kf_after if kf_after in prev_kfs else kf_before

            self._kf2fields[new_anchor_kf].update(self._kf2fields[removed_kf])
            del self._kf2fields[removed_kf]
            field_mask = self._global_map_dict["kf_ids"] == removed_kf
            self._global_map_dict["kf_ids"][field_mask] = new_anchor_kf

            # NOTE no transformations necessary for rewiring since map_dict stores
            #  absolute positions

        # update state
        self._update_field_poses()
        self._last_update = current_frame_id
        self._graph = new_graph

    @utils.benchmark
    def _update_field_poses(self) -> None:
        # update global scene representation poses
        if self._disable_relative_fields:
            return
        else:
            prev_absolute_map_dict = self._global_map_dict
            new_kf2ws = self._get_kf2ws(self._current_frame_id)

            relative_map_dict = self._absolute_map_dict_to_relative(
                prev_absolute_map_dict,
                self._prev_kf2ws,
            )
            self._global_map_dict = self._relative_map_dict_to_absolute(
                relative_map_dict, new_kf2ws
            )
            self._prev_kf2ws = new_kf2ws

    def _log_metrics(self) -> None:
        if self._disable_eval:
            return
        wandb.log(self._metrics)

    def _split_sequence(self) -> None:
        all_frame_ids = list(range(len(self._dataset)))
        last_frame_id = all_frame_ids[-1]
        self._eval_frame_ids = set()
        self._train_frame_ids = set()

        if self._eval_ratio == 0.0:
            self._train_frame_ids.update(all_frame_ids)
            self._chunks = []
            return self._train_frame_ids, self._eval_frame_ids, []

        eval_freq = math.floor(1 / self._eval_ratio)

        self._chunks = [
            {
                "eval_frame_ids": set(),
                "at_frame_id": None,
            }
        ]

        kf_counter = 0
        for frame_id in all_frame_ids:
            if self._dataset.is_keyframe(frame_id, at_frame_id=last_frame_id):
                kf_counter += 1

                if kf_counter % self._eval_chunk_freq == 0:
                    self._chunks.append(
                        {
                            "eval_frame_ids": set(),
                            "at_frame_id": None,
                        }
                    )

                self._chunks[-1]["at_frame_id"] = frame_id
                if kf_counter % eval_freq == 0:
                    self._chunks[-1]["eval_frame_ids"].add(frame_id)
                    self._eval_frame_ids.add(frame_id)
                else:
                    self._train_frame_ids.add(frame_id)
            else:
                self._train_frame_ids.add(frame_id)

    def fit(self) -> None:
        """Fit dataset and visualize output."""
        self._dataset = self._dataset_type(self._dataset_config)
        self._dataset.load_slam_results()
        self._dataset.set_mode("sequence")
        self._data_loader = DataLoader(self._dataset, num_workers=32, pin_memory=True)
        self._data_iterator = iter(self._data_loader)

        # used to align meshes with ground-truth mesh
        # mapping is done on raw SLAM results
        self._gt_from_est = None
        try:
            self._gt_from_est = self._dataset.gt_from_est_transform("umeyama").to(self._device)
        except ValueError as e:
            print(f"Trajectory alignment with Umeyama failed with:\n\t{str(e)}")
            print("Using no alignment instead.")

        self._split_sequence()

        self._camera = self._dataset.camera

        self._bb_min = torch.full((3,), fill_value=torch.inf, device=self._device)
        self._bb_max = torch.full((3,), fill_value=-torch.inf, device=self._device)

        self._kfs = {}  # dict mapping keyframe id to RGBD tensor
        self._kf_ids = set()
        self._latest_kf = None
        self._current_active_map_dict = None
        self._metrics = None
        self._total_optimization_time = 0.0
        if self._update_mode == "multi_view":
            self._init_mv_training_data()

        self._init_optimizer()

        run = wandb.init(
            project="ngs_fitscenepg",
            config=utils.prepare_dict_for_wandb(self._config),
            name=self._get_run_name(),
        )

        custom_run_dir = os.path.join("wandb", self._get_run_name(), "files")
        if run.disabled:
            os.makedirs(custom_run_dir, exist_ok=True)
            self._run_dir = custom_run_dir
        else:
            self._run_dir = wandb.run.dir  # this is the files dir
            os.makedirs(self._run_dir, exist_ok=True)
            os.symlink(pathlib.Path(wandb.run.dir).parent, pathlib.Path(custom_run_dir).parent)

        self._eval_data_dir = pathlib.Path(self._run_dir).parent / "eval_data"
        os.makedirs(self._eval_data_dir, exist_ok=True)

        if self._rerun_vis:
            utils.rr_init(
                "neural_graph_mapping",
                rrd_path=self._get_run_name() + ".rrd" if self._rerun_save else None,
                connect_addr=self._rerun_connect_addr,
            )

            if self._gt_from_est is not None:
                slam_transform = self._gt_from_est
            else:
                slam_transform = torch.eye(4, device=self._device)

            if self._dataset.up_axis is not None:
                rr.log("/", utils.rr_up_axis(self._dataset.up_axis), timeless=True)

            rr.log(
                "slam",
                rr.Transform3D(
                    translation=slam_transform[:3, 3].cpu(),
                    mat3x3=slam_transform[:3, :3].cpu(),
                ),
                timeless=True,
            )

            rr.log("slam/camera/image", rr.ViewCoordinates.RUB, timeless=True)
            rr.log(
                "slam/camera/image",
                rr.Pinhole(
                    image_from_camera=self._camera.get_projection_matrix(),
                    width=self._camera.width,
                    height=self._camera.height,
                ),
                timeless=True,
            )

        self._current_iteration = 1
        self._current_chunk_id = 0

        for current_frame_id in tqdm.tqdm(range(len(self._dataset)), dynamic_ncols=True):
            self._current_frame_id = current_frame_id
            self._current_frame_optimization()

        if self._rerun_vis or self._store_intermediate_meshes or self._eval_mesh:
            self._extract_mesh(
                self._est_mesh_path,
                transform=self._gt_from_est,
                resolution=0.02,
                field_ids=self.get_field_ids(min_iterations=50),
                log_to_rerun=self._rerun_vis,
            )
            torch.cuda.empty_cache()
            gc.collect()

            for single_field_id in self._extract_mesh_fields:
                self._extract_mesh(
                    self._est_mesh_path.with_stem(
                        f"{self._est_mesh_path.stem}_{single_field_id}"
                    ),
                    transform=self._gt_from_est,
                    resolution=0.02,
                    field_ids=torch.tensor([single_field_id]),
                )
                torch.cuda.empty_cache()
                gc.collect()

        self._evaluate_full()
        self._log_metrics()
        self.save_model()

    @utils.benchmark
    def _optimization_iteration(self) -> dict:
        """Execute one optimization iteration."""
        if self._update_mode == "single_view":
            # alternate between current and random previous keyframe
            # when current_c2w is missing (tracking lost by SLAM system), train on random
            # keyframe instead
            if self._current_frame_iteration % 2 != 0:
                if self._keyframes_only or self._current_c2w.isnan().any():
                    frame_id = self._latest_kf
                    active_field_ids = self._get_active_field_ids(self._latest_kf)
                    c2w = self._dataset.get_slam_c2ws(
                        self._latest_kf, self._current_frame_id
                    ).to(self._device)
                    rgbd_image = self._kfs[self._latest_kf]
                else:
                    frame_id = self._current_frame_id
                    active_field_ids = self._get_active_field_ids(frame_id)
                    c2w = self._current_c2w
                    rgbd_image = self._current_rgbd
            else:
                frame_id = random.choice(list(self._kfs.keys()))
                active_field_ids = self._get_active_field_ids(frame_id)
                c2w = self._dataset.get_slam_c2ws(frame_id, self._current_frame_id).to(
                    self._device
                )
                rgbd_image = self._kfs[frame_id]

        if self._update_mode == "single_view":
            target = self._sample_target_sv(rgbd_image, c2w, active_field_ids)
        elif self._update_mode == "multi_view":
            target = self._sample_target_mv(self._current_field_ids)

        if (
            self._single_field_id is not None
            and not (target.field_ids == self._single_field_id).any()
        ):
            return {}

        self._target = target

        prediction = self._render_ijs(
            target.ijs,
            target.c2ws,
            self._camera,
            near_distances=target.near_distances,
            far_distances=target.far_distances,
            gt_distances=target.gt_distances,
            field_ids=target.field_ids,
            use_vmap=True,
        )

        loss_dict = self._compute_losses(target, prediction)

        self._update_step(loss_dict, target.field_ids)

        self._current_iteration += 1

        return loss_dict

    @utils.benchmark
    def _update_step(self, loss_dict: dict, field_ids: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        loss_dict["combined"].backward()

        self._global_map_dict["training_iterations"][field_ids] += 1

        if self._single_field_id is not None:
            self._optimizer.step()
        else:
            self._optimizer.step()

            all_params = self._model.all_fields_params
            vmap_params = self._model.vmap_fields_params
            param_names = vmap_params.keys()
            state = self._optimizer.state
            f_ids = field_ids

            with torch.no_grad():
                # write updated tensor back to full parameters
                for p in param_names:
                    all_params[p][f_ids] = vmap_params[p]

                # update full optimizer state
                for p in param_names:
                    param = vmap_params[p]
                    if param not in self._optimizer.state:
                        continue

                    if isinstance(self._optimizer, torch.optim.Adam):
                        self._optim_state[p]["step"] = state[param]["step"]
                        self._optim_state[p]["exp_avg"][f_ids] = state[param]["exp_avg"]
                        self._optim_state[p]["exp_avg_sq"][f_ids] = state[param]["exp_avg_sq"]
                    elif isinstance(self._optimizer, torch.optim.RMSprop):
                        self._optim_state[p]["step"] = state[param]["step"]
                        self._optim_state[p]["square_avg"][f_ids] = state[param]["square_avg"]
                        self._optim_state[p]["momentum_buffer"][f_ids] = state[param][
                            "momentum_buffer"
                        ]

    @utils.benchmark
    def _current_frame_optimization(self) -> None:
        log_time = 0
        torch.cuda.synchronize()
        start = time.time()

        if self._current_frame_id in self._train_frame_ids:
            self._update_slam_state()

            for self._current_frame_iteration in range(self._num_iterations_per_frame):
                loss_dict = self._optimization_iteration()

                torch.cuda.synchronize()
                log_start = time.time()

                self._log(loss_dict)

                torch.cuda.synchronize()
                log_time += time.time() - log_start
        else:
            # eval frame -> advance data iterator without using the data
            next(self._data_iterator)["rgbd"][0].to(self._device)

        torch.cuda.synchronize()
        end = time.time()

        self._total_optimization_time += (end - start) - log_time
        self._fps_estimate = (self._current_frame_id + 1) / self._total_optimization_time
        self._spf_estimate = self._total_optimization_time / (self._current_frame_id + 1)

        if self._current_chunk_id < len(self._chunks):
            current_chunk = self._chunks[self._current_chunk_id]
            if current_chunk["at_frame_id"] == self._current_frame_id:
                self._evaluate_chunk(current_chunk)
                self._current_chunk_id += 1

    @utils.benchmark
    @torch.no_grad()
    def _sample_target_mv(
        self,
        current_field_ids: torch.LongTensor,
    ) -> Target:
        """Sample fields, rays, and supervision targets.

        Args:
            current_field_ids:
                Currently observed field ids.

        Returns:
            ijs:
                The corresponding pixel (row, column) for each ray.
                Shape (num_fields, num_rays_per_field,).
            nears: Minimum depth value for each ray.
            fars: Maximum depth value for each ray.
            field_ids: The field_ids to train on. Shape (num_fields,).
            rgbds:
                The RGBD targets for each ray.
                Shape (num_fields, num_rays_per_field, 4,).
            depth_mask:
                Mask indicating whether depth target should be used for supervision.
            term_probs:
                Boolean probabilities indicating whether rays should terminate or not
                within field. Shape (num_fields, num_rays_per_field).
            term_mask:
                Mask indicating whether term_probs should be used for supervision.
        """
        MARGIN = 0.0
        train_radius = self._field_radius + MARGIN
        num_field_samples = 20

        num_frames = len(self._c_c2w_tensor)

        num_observed_fields = min(self._num_train_fields // 2, len(current_field_ids))
        subset_observed = torch.multinomial(
            torch.ones(len(current_field_ids), device=self._device), num_observed_fields
        )
        target_observed_field_ids = current_field_ids[subset_observed]

        num_missing_fields = self._num_train_fields - len(target_observed_field_ids)
        num_random_fields = min(
            num_missing_fields, self._num_fields - len(target_observed_field_ids)
        )
        if num_random_fields > 0:
            random_distribution = torch.ones(self._num_fields, device=self._device)
            random_distribution[target_observed_field_ids] = 0.0
            subset_random = torch.multinomial(
                random_distribution,
                num_random_fields,
            )
            target_random_field_ids = self.get_field_ids()[subset_random]
            target_field_ids = torch.unique(
                torch.cat((target_random_field_ids, target_observed_field_ids))
            )
        else:
            target_field_ids = target_observed_field_ids

        num_target_fields = len(target_field_ids)

        field_pos_w = self._global_map_dict["positions"][target_field_ids]

        # find cams where fields can be supervised
        field_samples_offsets = torch.randn((num_field_samples, 3), device=self._device)
        field_samples_offsets /= torch.linalg.norm(field_samples_offsets, dim=-1, keepdim=True)
        field_samples_w = field_pos_w.unsqueeze(1) + field_samples_offsets * train_radius * 1.0
        field_samples_c = utils.transform_points(
            field_samples_w.unsqueeze(-2), self._c_c2w_tensor, inv=True
        )
        field_samples_depths = -field_samples_c[..., 2]
        field_samples_2d = self._camera.project_points(field_samples_c, "opengl")
        field_samples_frame_cids = (
            torch.arange(len(self._c_c2w_tensor), device=self._device)
            .expand(num_target_fields, num_field_samples, -1)
            .reshape(-1)
        )
        field_samples_ijs = field_samples_2d.int().view(-1, 2)

        valid_mask = (
            (field_samples_ijs[:, 0] >= 0)
            * (field_samples_ijs[:, 0] < self._camera.width)
            * (field_samples_ijs[:, 1] >= 0)
            * (field_samples_ijs[:, 1] < self._camera.height)
        )
        field_samples_kf_depths = torch.zeros_like(field_samples_depths)

        field_samples_kf_depths[
            valid_mask.view(num_target_fields, num_field_samples, num_frames)
        ] = self._nc_rgbd_tensor[
            self._frame_cid_to_ncid[field_samples_frame_cids[valid_mask]],
            field_samples_ijs[valid_mask, 1],
            field_samples_ijs[valid_mask, 0],
            3,
        ]

        # find field <-> keyframe correspondences
        in_front_mask = (field_samples_depths > 0).any(dim=-2)
        in_front_depth_mask = (field_samples_depths < field_samples_kf_depths).any(dim=-2)
        in_frustum_mask = valid_mask.view(
            num_target_fields, num_field_samples, num_frames
        ).any(dim=-2)
        field_kf_mask = in_front_mask * in_front_depth_mask * in_frustum_mask

        # only continue with fields that are visible in at least one keyframe
        field_mask = field_kf_mask.any(dim=-1)

        if (
            self._rerun_field_details is not None
            and (target_field_ids == self._rerun_field_details).any()
        ):
            rr.log_points(
                "slam/field_samples",
                field_samples_w[target_field_ids == self._rerun_field_details].squeeze().cpu(),
            )

        field_kf_mask = field_kf_mask[field_mask]
        target_field_ids = target_field_ids[field_mask]
        target_field_pos_w = field_pos_w[field_mask]
        field_samples_2d = field_samples_2d[field_mask]
        num_target_fields = len(target_field_ids)

        # sample keyframes for final targets
        target_frame_cids = torch.multinomial(
            field_kf_mask.float(), self._num_rays_per_field, replacement=True
        )

        # compute bounding box for kfs
        min_xys = field_samples_2d.min(dim=1)[0].clamp_min_(0.0)
        max_xys = field_samples_2d.max(dim=1)[0].clamp_max_(
            max=torch.tensor((self._camera.width, self._camera.height), device=self._device)
        )
        target_min_xys = torch.gather(
            min_xys, dim=1, index=target_frame_cids[..., None].expand(-1, -1, 2)
        )
        target_max_xys = torch.gather(
            max_xys, dim=1, index=target_frame_cids[..., None].expand(-1, -1, 2)
        )

        # compute target pixels
        target_xys = (target_max_xys - target_min_xys) * torch.rand(
            num_target_fields, self._num_rays_per_field, 2, device=self._device
        ) + target_min_xys
        target_jis = target_xys.int().clamp_max_(
            torch.tensor(
                (self._camera.width - 1, self._camera.height - 1), device=self._device
            )
        )
        target_ijs = torch.stack((target_jis[..., 1], target_jis[..., 0]), dim=-1)

        # collect remaining target quantities
        target_c2ws = self._c_c2w_tensor[target_frame_cids]

        # compute nears and fars for each sampled line segment
        target_field_pos_c = utils.transform_points(
            target_field_pos_w.unsqueeze(1), target_c2ws, inv=True
        )
        target_dirs = self._camera.ijs_to_directions(target_ijs)
        center_distance = (target_field_pos_c * target_dirs).sum(-1)
        target_near_distances = center_distance - train_radius
        target_near_distances[target_near_distances < 0.0] = 0.0
        target_far_distances = center_distance + train_radius
        target_far_distances[target_far_distances < 0.0] = 0.0

        # compute RGBD targets
        target_kf_ncids = self._frame_cid_to_ncid[target_frame_cids]
        target_rgbds = self._nc_rgbd_tensor[
            target_kf_ncids, target_ijs[..., 0], target_ijs[..., 1]
        ]
        target_gt_distances = self._camera.depth_to_distance(target_rgbds[..., 3], target_ijs)
        # breakpoint()
        valid_depth_mask = target_gt_distances != 0.0
        target_depth_mask = (
            (target_gt_distances > target_near_distances)
            * (target_gt_distances < target_far_distances)
            * valid_depth_mask
        )
        # target_rgb_mask = target_depth_mask
        target_rgb_mask = (target_rgbds[..., :2] != 0.0).any(dim=-1)

        # compute termination targets
        # everything in front of far plane should be terminating
        # also mask with known mask to accont for missing depth
        target_term_probs = (target_gt_distances < target_far_distances).float()
        # only supervise termination when depth available
        target_term_mask = (target_gt_distances > target_near_distances) * valid_depth_mask

        return Target(
            ijs=target_ijs,
            c2ws=target_c2ws,
            near_distances=target_near_distances,
            far_distances=target_far_distances,
            gt_distances=target_gt_distances,
            field_ids=target_field_ids,
            rgbds=target_rgbds,
            rgb_mask=target_rgb_mask,
            depth_mask=target_depth_mask,
            term_probs=target_term_probs,
            term_mask=target_term_mask,
        )

    @utils.benchmark
    @torch.no_grad()
    def _sample_target_sv(
        self,
        rgbd_image: torch.Tensor,
        c2w: torch.Tensor,
        active_field_ids: torch.Tensor,
    ) -> Target:
        """Sample fields, rays, and supervision targets.

        Args:
            rgbd_image:
                Observed ground-truth image used to compute targets.
                Shape (H, W, 4).
            c2w:
                Transformation matrix from camera to world coordinates.
                OpenGL convention. Shape (4,4).
            active_field_ids:
                Field indices that can be sampled. Shape (num_active_fields,).

        Returns:
            ijs:
                The corresponding pixel (row, column) for each ray.
                Shape (num_fields, num_rays_per_field,).
            nears: Minimum depth value for each ray.
            fars: Maximum depth value for each ray.
            field_ids: The field_ids to train on. Shape (num_fields,).
            rgbds:
                The RGBD targets for each ray.
                Shape (num_fields, num_rays_per_field, 4,).
            rgbds_mask:
                Mask indicating whether RGBD target should be used for supervision.
            term_probs:
                Boolean probabilities indicating whether rays should terminate or not
                within field. Shape (num_fields, num_rays_per_field).
            term_mask:
                Mask indicating whether term_probs should be used for supervision.
        """
        MARGIN = 0.0

        train_radius = self._field_radius + MARGIN
        field_pos_w = self._global_map_dict["positions"][active_field_ids]
        field_pos_c = utils.transform_points(field_pos_w, c2w, inv=True)
        points, ijs = self._camera.depth_to_pointcloud(rgbd_image[..., 3], return_ijs=True)

        subset = torch.multinomial(torch.ones(len(points), device=self._device), 50000)
        points = points[subset]
        ijs = ijs[subset]

        # remove fields whose AABB does not intersect point cloud AABB
        aabb = geometry.AABBs(
            points.min(dim=0)[0],
            points.max(dim=0)[0],
        )
        spheres = geometry.Spheres(field_pos_c, train_radius)
        aabb_mask = aabb.intersects_aabbs(spheres.aabbs())
        field_pos_c_in_aabb = field_pos_c[aabb_mask]
        spheres = geometry.Spheres(field_pos_c_in_aabb, train_radius)

        # find intersection of remaining fields and line segments
        origin = torch.zeros(3, device=self._device)
        line_segments = geometry.LineSegments(origin, points)
        intersects = line_segments.intersects_spheres(spheres)
        intersect_counts = line_segments.intersects_spheres(spheres).sum(dim=-1)

        # only keep fields intersecting with sufficient unique segments
        segment_mask = intersect_counts >= self._num_rays_per_field
        intersects = intersects[segment_mask]

        # sample subset of fields
        if len(intersects) <= self._num_train_fields:
            target_field_ids = active_field_ids[aabb_mask][segment_mask]
            target_field_pos_c = field_pos_c_in_aabb[segment_mask]
        else:
            subset = torch.multinomial(
                torch.ones(len(intersects), device=self._device),
                self._num_train_fields,
            )
            target_field_ids = active_field_ids[aabb_mask][segment_mask][subset]
            target_field_pos_c = field_pos_c_in_aabb[segment_mask][subset]
            intersects = intersects[subset]

        # sample intersects
        segments = torch.multinomial(intersects.float(), self._num_rays_per_field)

        # get ijs for each sampled segment
        target_ijs = ijs[segments]

        # compute nears and fars for each sampled line segment
        target_dirs = self._camera.ijs_to_directions(target_ijs)
        center_distance = (target_field_pos_c.unsqueeze(1) * target_dirs).sum(-1)
        target_near_distances = center_distance - train_radius
        target_far_distances = center_distance + train_radius

        # compute RGBD targets
        # by construction depth is further than near depth, so this is not checked again
        # here
        target_rgbds = rgbd_image[target_ijs[..., 0], target_ijs[..., 1]]
        target_gt_distances = self._camera.depth_to_distance(target_rgbds[..., 3], target_ijs)
        target_depth_mask = target_gt_distances < target_far_distances

        # compute termination targets
        # everything in front of far plane should be terminating
        # also mask with known mask to accont for missing depth
        target_term_probs = (target_depth_mask).float()
        target_term_mask = torch.ones_like(target_term_probs).bool()

        if target_ijs.shape[0] == 0:
            breakpoint()

        return Target(
            ijs=target_ijs,
            c2ws=c2w,
            near_distances=target_near_distances,
            far_distances=target_far_distances,
            gt_distances=target_gt_distances,
            field_ids=target_field_ids,
            rgbds=target_rgbds,
            rgb_mask=target_depth_mask,
            depth_mask=target_depth_mask,
            term_probs=target_term_probs,
            term_mask=target_term_mask,
        )

    @utils.benchmark
    @torch.no_grad()
    def _get_current_rgbd(self) -> torch.Tensor:
        current_rgbd = next(self._data_iterator)["rgbd"][0].to(self._device)
        # current_rgbd = self.dataset[self._current_frame_id]["rgbd"].to(self._device)
        if self._max_depth is not None:
            ind_1, ind_2 = torch.nonzero(
                current_rgbd[:, :, 3] > self._max_depth, as_tuple=True
            )
            current_rgbd[ind_1, ind_2, 3] = 0.0
        return current_rgbd

    @utils.benchmark
    @torch.no_grad()
    def _update_slam_state(self) -> None:
        self._current_rgbd = self._get_current_rgbd()
        self._current_c2w = self._dataset.get_slam_c2ws(self._current_frame_id).to(
            self._device
        )
        self._current_c2w_missing = self._current_c2w.isnan().any()

        # FIXME this whole update logic is messy right now

        self._update_graph(self._current_frame_id)

        # add current frame to keyframes if it is one
        if self._dataset.is_keyframe(self._current_frame_id):
            self._kf_ids.add(self._current_frame_id)

            if self._update_mode == "single_view":
                self._kfs[self._current_frame_id] = self._current_rgbd

            self._extend_global_map_dict(
                self._current_rgbd[:, :, 3],
                self._current_frame_id,
                self._current_c2w,
                self._camera,
                self._current_active_map_dict,
            )
            self._current_is_keyframe = True
        else:
            self._current_is_keyframe = False

        if self._update_mode == "single_view":
            self._latest_kf = max(self._kfs.keys())

        self._current_active_map_dict = self._get_active_map_dict(self._current_frame_id)

        if self._update_mode == "multi_view":
            if not self._current_c2w_missing:
                self._current_field_ids = self._get_observed_fields(
                    self._current_rgbd, self._current_c2w
                )
            self._update_mv_training_data()

        self._log_to_rerun()

    @utils.benchmark
    def _get_observed_fields(
        self, rgbd_image: torch.Tensor, c2w: torch.Tensor
    ) -> torch.LongTensor:
        field_pos_w = self._global_map_dict["positions"][: self._num_fields]
        field_pos_c = utils.transform_points(field_pos_w, c2w, inv=True)
        field_ids = torch.arange(self._num_fields, device=self._device)

        points, ijs = self._camera.depth_to_pointcloud(rgbd_image[..., 3], return_ijs=True)
        subset = torch.multinomial(torch.ones(len(points), device=self._device), 500)
        points = points[subset]

        # remove fields whose AABB does not intersect point cloud AABB
        aabb = geometry.AABBs(
            points.min(dim=0)[0],
            points.max(dim=0)[0],
        )
        spheres = geometry.Spheres(field_pos_c, self._field_radius)
        aabb_mask = aabb.intersects_aabbs(spheres.aabbs())
        field_pos_c_in_aabb = field_pos_c[aabb_mask]
        field_ids_in_aabb = field_ids[aabb_mask]
        spheres = geometry.Spheres(field_pos_c_in_aabb, self._field_radius)

        # find intersection of remaining fields and line segments
        origin = torch.zeros(3, device=self._device)
        line_segments = geometry.LineSegments(origin, points)
        observed = line_segments.intersects_spheres(spheres).any(dim=-1)
        observed_field_ids = field_ids_in_aabb[observed]
        return observed_field_ids

    @utils.benchmark
    def _init_mv_training_data(self) -> None:
        self._free_rgbd_tensor_indices = list(range(1000))
        if not self._keyframes_only:
            # 0 will be used for current frame
            self._free_rgbd_tensor_indices.pop(0)
        self._nc_rgbd_tensor = torch.empty(
            (1000, self._camera.height, self._camera.width, 4), device=self._device
        )  # preallocate space for 2000 keyframes
        self._nc_frame_id_tensor = torch.full(
            (1000,), fill_value=-1, device=self._device, dtype=torch.long
        )

    @utils.benchmark
    def _update_mv_training_data(self) -> None:
        if not self._keyframes_only:
            if self._current_c2w_missing:
                self._nc_frame_id_tensor[0] = -1
            else:
                self._nc_rgbd_tensor[0] = self._current_rgbd
                self._nc_frame_id_tensor[0] = self._current_frame_id

        if self._current_is_keyframe:
            if len(self._free_rgbd_tensor_indices) == 0:
                raise ValueError("Maximum number of keyframes reached.")
            tensor_idx = self._free_rgbd_tensor_indices.pop(0)
            self._nc_rgbd_tensor[tensor_idx] = self._current_rgbd
            self._nc_frame_id_tensor[tensor_idx] = self._current_frame_id

        self._td_mask = self._nc_frame_id_tensor != -1

        self._frame_cid_to_ncid = torch.arange(len(self._td_mask), device=self._device)[
            self._td_mask
        ]
        kf2ws = self._get_kf2ws(self._current_frame_id)
        kf2ws = kf2ws.index_select(
            dim=0, index=self._nc_frame_id_tensor[self._nc_frame_id_tensor != -1]
        ).to_dense()
        if not self._keyframes_only:
            self._c_c2w_tensor = torch.cat((self._current_c2w[None], kf2ws[1:]))
        else:
            self._c_c2w_tensor = kf2ws

    def _log(
        self,
        loss_dict: dict,
    ) -> None:
        """Log various entities to WandB and / or Rerun.

        This function logs losses, renders, and meshes.
        """
        if self._current_iteration % self._log_iteration_freq == 0:
            wandb.log(loss_dict, step=self._current_iteration)
            wandb.log(
                {"current_frame_id": self._current_frame_id},
                step=self._current_iteration,
            )

        # + 1 so nothing is rendered after the first frame
        if (
            self._current_frame_id + 1
        ) % self._render_frame_freq == 0 and self._current_frame_iteration == 0:
            self._log_renders()

        if (
            (self._rerun_vis or self._store_intermediate_meshes)
            and (self._current_frame_id + 1) % self._extract_mesh_frame_freq == 0
            or self._current_frame_id in self._extract_mesh_frames
        ) and self._current_frame_iteration == 0:
            aligned_prefix = "aligned_" if self._gt_from_est is not None else ""
            mesh_path = (
                self._eval_data_dir / f"{aligned_prefix}frame_{self._current_frame_id}.ply"
            )
            self._extract_mesh(
                mesh_path,
                resolution=0.03,
                transform=self._gt_from_est,
                field_ids=self.get_field_ids(50),
                log_to_rerun=self._rerun_vis,
            )
            torch.cuda.empty_cache()
            gc.collect()

            for single_field_id in self._extract_mesh_fields:
                self._extract_mesh(
                    mesh_path.with_stem(f"{mesh_path.stem}_{single_field_id}"),
                    transform=self._gt_from_est,
                    resolution=0.03,
                    field_ids=torch.tensor([single_field_id]),
                )
                torch.cuda.empty_cache()
                gc.collect()

        # if self._rerun_vis:
        #     for k, v in loss_dict.items():
        #         rr.log_scalar(f"losses/{k}", v.item())

    def _compute_losses(
        self,
        target: Target,
        prediction: Prediction,
    ) -> dict:
        """Compute losses based on target and prediction."""
        if self._single_field_id is not None:
            mask = target.field_ids == self._single_field_id
            target_term_mask = target.term_mask[mask][0]
            target_term_probs = target.term_probs[mask][0]
            target_depth_mask = target.depth_mask[mask][0]
            target_rgbds = target.rgbds[mask][0]
        else:
            target_term_mask = target.term_mask
            target_term_probs = target.term_probs
            target_depth_mask = target.depth_mask
            target_rgbds = target.rgbds

        depth_mask = target_depth_mask * (prediction.term_probs > 0.8)
        rgb_mask = depth_mask

        combined_loss = 0
        loss_dict = {}

        # rgb_mask = target.depth_mask * (prediction.term_probs > 0.8)
        # rgb_mask = (
        #     target.rgb_mask * (prediction.term_probs > 0.8) * target.term_mask
        #     + ~target.term_mask * target.rgb_mask
        # )

        # terminating = target_term_mask * (target_term_probs == 1.0)
        # non_terminating = target_term_mask * (target_term_probs == 0.0)

        # termination loss
        termination_loss = (
            (prediction.term_probs[target_term_mask] - target_term_probs[target_term_mask])
            ** 2
        ).mean()
        combined_loss = combined_loss + self._termination_weight * termination_loss
        # print(prediction.term_probs)
        loss_dict["termination"] = termination_loss

        # non_termination_loss = (
        #     (prediction.term_probs[non_terminating] - target_term_probs[non_terminating]) ** 2
        # ).mean()

        # termination_loss = torch.nn.functional.binary_cross_entropy(
        #     prediction.term_probs[target.term_mask], target.term_probs[target.term_mask]
        # )

        # photometric loss
        photometric_loss = losses.photometric_loss(
            self._photometric_loss,
            prediction.rgbds[rgb_mask][:, :3],
            target_rgbds[rgb_mask][:, :3],
            prediction.color_vars[rgb_mask],
        )
        combined_loss = combined_loss + self._photometric_weight * photometric_loss
        loss_dict[f"photometric_{self._photometric_loss}"] = photometric_loss

        # depth loss
        depth_loss = losses.depth_loss(
            self._depth_loss,
            target_rgbds[depth_mask][:, 3],
            prediction.rgbds[depth_mask][:, 3],
            prediction.depth_vars[depth_mask],
        )
        combined_loss = combined_loss + self._depth_weight * depth_loss
        loss_dict[f"depth_{self._depth_loss}"] = depth_loss

        # print(prediction.depth_vars)

        # freespace loss
        if prediction.freespace_geometry is not None:
            freespace_loss = (
                (prediction.freespace_geometry - self._truncation_distance) ** 2
            ).mean()
            combined_loss = combined_loss + self._freespace_weight * freespace_loss
            loss_dict["freespace"] = freespace_loss

        if prediction.tsdf_residuals is not None:
            tsdf_loss = (prediction.tsdf_residuals**2).mean()
            combined_loss = combined_loss + self._tsdf_weight * tsdf_loss
            loss_dict["tsdf"] = tsdf_loss

        # print(
        #     "raw",
        #     depth_loss.item(),
        #     termination_loss.item(),
        #     freespace_loss.item(),
        #     tsdf_loss.item(),
        #     photometric_loss.item(),
        # )
        # print(
        #     "weighted",
        #     self._depth_weight * depth_loss.item(),
        #     self._termination_weight * termination_loss.item(),
        #     self._freespace_weight * freespace_loss.item(),
        #     self._tsdf_weight * tsdf_loss.item(),
        #     self._photometric_weight * photometric_loss.item(),
        # )

        loss_dict["combined"] = combined_loss
        return loss_dict

    @torch.no_grad()
    def _log_renders(self) -> plt.Figure:
        if not self._render_vis or len(self._render_frames) == 0:
            return

        self.eval()

        fig, ax = plt.subplots(len(self._render_frames), 2)
        preview_camera = self._preview_camera()

        # render frame is from [0,1] indicating position in dataset
        for i, render_frame in enumerate(self._render_frames):
            frame_id = self._closest_kf_id(int(render_frame * (len(self._dataset) - 1)))
            at_frame_id = (
                self._current_frame_id if self._current_frame_id > frame_id else frame_id
            )
            c2w = self._dataset.get_slam_c2ws(frame_id, at_frame_id).to(self._device)

            rgbd, _ = self.render_image(c2w, preview_camera, True)

            ax[i, 0].imshow(rgbd.cpu().numpy()[:, :, :3])
            ax[i, 1].imshow(rgbd.cpu().numpy()[:, :, 3], vmin=0.0, vmax=7.0)

            self._log_rgbd_to_rerun(rgbd, c2w, f"camera_{i}", preview_camera)

        wandb.log({"preview": fig})

        plt.close()

        torch.cuda.empty_cache()
        gc.collect()

        self.train()

    @torch.no_grad()
    def _evaluate_chunk(self, chunk: dict) -> None:
        if self._disable_eval:
            return

        metric_dicts_for_chunk = []
        for eval_frame_id in tqdm.tqdm(chunk["eval_frame_ids"]):
            metric_dict = self._evaluate_frame(eval_frame_id, chunk["at_frame_id"])
            metric_dicts_for_chunk.append(metric_dict)
        metric_dict_for_chunk = _mean_metric_dict(metric_dicts_for_chunk)
        self._metric_dicts_for_chunks.append(metric_dict_for_chunk)

    @torch.no_grad()
    def _evaluate_full(self) -> None:
        if self._disable_eval:
            return

        final_render_metrics = {}
        final_mesh_metrics = {}

        online_metrics = _mean_metric_dict(self._metric_dicts_for_chunks)

        if len(self._eval_render_metrics) != 0:
            metric_dicts = []
            for eval_frame_id in tqdm.tqdm(self._eval_frame_ids):
                metric_dict = self._evaluate_frame(eval_frame_id, len(self._dataset) - 1)
                metric_dicts.append(metric_dict)

            final_render_metrics = _mean_metric_dict(metric_dicts)

        if self._eval_mesh:
            if self._dataset.has_gt_mesh:
                final_mesh_metrics = evaluation.evaluate_raw_mesh(
                    self._est_mesh_path,
                    self._dataset,
                    self._eval_culling_method,
                    self._eval_culling_method,
                    self._eval_mesh_alignment,
                    self._eval_mesh_num_points,
                    self._rerun_vis,
                )
            else:
                print("Ground-truth mesh not available. Skipping mesh eval.")

        self._metrics = {}
        for k, v in online_metrics.items():
            self._metrics[f"online_{k}"] = v
        for k, v in final_render_metrics.items():
            self._metrics[f"final_{k}"] = v
        for k, v in final_mesh_metrics.items():
            self._metrics[f"mesh_{k}"] = v

        self._metrics["num_params_per_field"] = self._model.numel()
        self._metrics["num_fields"] = self._global_map_dict["num"]
        self._metrics["num_params"] = self._model.numel() * self._global_map_dict["num"]
        self._metrics["fps_estimate"] = self._fps_estimate
        self._metrics["spf_estimate"] = self._spf_estimate

    def eval(self) -> None:
        self._far_distance = self._eval_far_distance
        self._near_distance = self._eval_near_distance
        self._num_samples = self._eval_num_samples

    def train(self) -> None:
        self._far_distance = self._train_far_distance
        self._near_distance = self._train_near_distance
        self._num_samples = self._train_num_samples

    @torch.no_grad()
    def _evaluate_frame(self, frame_id: int, at_frame_id: int) -> dict:
        self.eval()  # set evaluation specific parameters

        out_img_path = os.path.join(self._eval_data_dir, f"{at_frame_id}_{frame_id}.png")
        c2w = self._dataset.get_slam_c2ws(frame_id, at_frame_id).to(self._device)
        eval_camera = self._dataset.camera
        rgbd, _ = self.render_image(c2w, eval_camera)
        target_rgbd = self._dataset[frame_id]["rgbd"].to(self._device)
        metric_dict = {}
        self._eval_details.append([out_img_path])

        for metric in self._eval_render_metrics:
            if metric == "lpips":
                metric_dict["lpips"] = evaluation.lpips(
                    rgbd[:, :, :3], target_rgbd[:, :, :3], self._eval_crop
                )
            elif metric == "ssim":
                metric_dict["ssim"] = evaluation.ssim(
                    rgbd[:, :, :3], target_rgbd[:, :, :3], self._eval_crop
                )
            elif metric == "psnr":
                metric_dict["psnr"] = evaluation.psnr(
                    rgbd[:, :, :3], target_rgbd[:, :, :3], self._eval_crop
                )
            elif metric == "depthl1":
                metric_dict["depthl1"] = evaluation.depthl1(
                    rgbd[:, :, 3], target_rgbd[:, :, 3], self._eval_crop
                )
            self._eval_details[-1].append(metric_dict[metric])

        if self._eval_store_details:
            comparison_rgb = torch.clamp(
                torch.cat((target_rgbd[:, :, :3], rgbd[:, :, :3]), dim=1), 0, 1
            )
            utils.save_image(comparison_rgb, out_img_path)
            out_table_path = os.path.join(self._eval_data_dir, "details.txt")
            headers = ["filename", *self._eval_render_metrics]
            with open(out_table_path, "w") as f:
                f.write(tabulate.tabulate(self._eval_details, headers=headers))

        gc.collect()

        self.train()  # go back to train specific parameters
        return metric_dict

    def _log_rgbd_to_rerun(
        self,
        rgbd: torch.Tensor,
        c2w: torch.Tensor,
        camera_name: str,
        cam: camera.Camera,
    ) -> None:
        if not self._rerun_vis:
            return
        ocv2ogl = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        parent_from_child = c2w.to("cpu") @ ocv2ogl
        trans = parent_from_child[:3, 3]
        quat_wxyz = p3dt.matrix_to_quaternion(parent_from_child[:3, :3])
        quat_xyzw = torch.roll(quat_wxyz, -1, dims=0)

        rr.set_time_sequence("frame_id", self._current_frame_id)
        rr.log(f"slam/{camera_name}", parent_from_child=(trans, quat_xyzw))
        rr.log(
            f"slam/{camera_name}/image",
            rr.Pinhole(
                image_from_camera=cam.get_projection_matrix(pixel_center=0.5),
                width=cam.width,
                height=cam.height,
            ),
        )
        rr.log(f"slam/{camera_name}/image", rr.ViewCoordinates.RDF)
        rr.log(f"slam/{camera_name}/image/rgb", rr.Image(rgbd[:, :, :3].numpy(force=True)))
        rr.log(
            f"slam/{camera_name}/image/depth",
            rr.DepthImage(
                rgbd[:, :, 3].numpy(force=True),
                meter=1.0,
            ),
        )

    @utils.benchmark
    def _log_to_rerun(self) -> None:
        if not self._rerun_vis:
            return

        # FIXME visualize keyframes + current camera frame

        rr.set_time_sequence("frame_id", self._current_frame_id)

        # added_edges = set()
        # line_segments = []
        # for keyframe_id in self._graph:
        #     for neighbor_id in self._graph[keyframe_id]:
        #         edge = tuple(sorted((keyframe_id, neighbor_id)))
        #         if edge not in added_edges:
        #             added_edges.add(edge)
        #             start2w = self._dataset.get_slam_c2ws(
        #                 keyframe_id, self._current_frame_id
        #             )
        #             start2w = start2w.numpy(force=True)
        #             end2w = self._dataset.get_slam_c2ws(
        #                 neighbor_id, self._current_frame_id
        #             )
        #             end2w = end2w.numpy(force=True)
        #             line_segments.append([start2w[0, 3], start2w[1, 3], start2w[2, 3]])
        #             line_segments.append([end2w[0, 3], end2w[1, 3], end2w[2, 3]])
        # if line_segments:
        #     rr.log_line_segments("slam/edges", line_segments, stroke_width=0.002)

        field_positions = self._global_map_dict["positions"][
            : self._global_map_dict["num"]
        ].cpu()

        field_class_ids = None
        if self._update_mode == "multi_view":
            field_class_ids = torch.zeros(len(field_positions), device=self._device)
            field_class_ids[self._current_field_ids] = 1
            field_class_ids = field_class_ids.numpy(force=True)

        rr.log(
            "slam/fields",
            rr.Points3D(
                self._global_map_dict["positions"][: self._global_map_dict["num"]].cpu(),
                radii=self._field_radius * 0.05,
                class_ids=field_class_ids,
            ),
        )

        translation = self._current_c2w.cpu()[:3, 3]
        rotation = self._current_c2w.cpu()[:3, :3]
        if self._current_frame_id % 5 == 0:
            rr.log(
                "slam/camera",
                rr.Transform3D(translation=translation, mat3x3=rotation),
            )
            rr.log(
                "slam/camera/image/rgb",
                rr.Image(self._current_rgbd[:, :, :3].cpu()),
            )
            rr.log(
                "slam/camera/image/depth",
                rr.DepthImage(self._current_rgbd[:, :, 3].cpu(), meter=1.0),
            )

        # field_line_segments = []
        # field_positions = self._global_map_dict["positions"].numpy(force=True)
        # for keyframe_id in self._kf2fields.keys():
        #     for field_id in self._kf2fields[keyframe_id]:
        #         start2w = self._dataset.get_slam_c2ws(
        #             keyframe_id, self._current_frame_id
        #         )
        #         start2w = start2w.numpy(force=True)
        #         end = field_positions[field_id]
        #         field_line_segments.append(
        #             [start2w[0, 3], start2w[1, 3], start2w[2, 3]]
        #         )
        #         field_line_segments.append([end[0], end[1], end[2]])

        # if field_line_segments:
        #     rr.log(
        #         "slam/field_edges", rr.LineStrips3D(field_line_segments, stroke_width=0.01)
        #     )

    def save_model(self) -> None:
        """Save model from disk."""
        torch.save(
            {
                "map_dict": self._global_map_dict,
                "all_fields_params": self._model.all_fields_params,
                "state_dict": self._model.state_dict(),
            },
            os.path.join(self._run_dir, f"{self._get_run_name()}.pt"),
        )
        model_config = copy.deepcopy(self._config)
        model_config["model"] = os.path.join(".", f"{self._get_run_name()}.pt")
        if self._metrics is not None:
            model_config["results"] = self._metrics
        yoco.save_config_to_file(
            os.path.join(self._run_dir, f"{self._get_run_name()}.yaml"), model_config
        )
        yoco.save_config_to_file(os.path.join(self._run_dir, "latest_run.yaml"), model_config)

    def load_model(self, path: str) -> None:
        """Load model from disk."""
        print(f"Load scene representation from {path}")
        dict_ = torch.load(path, map_location=self._device)
        self._global_map_dict = dict_["map_dict"]
        self._model.all_fields_params = dict_["all_fields_params"]
        self._model.load_state_dict(dict_["state_dict"])
        print("Done loading")

    def get_field_ids(self, min_iterations: Optional[int] = None) -> torch.Tensor:
        if min_iterations is not None:
            train_iterations = self._global_map_dict["training_iterations"][: self._num_fields]
            return torch.where(train_iterations >= min_iterations)[0]
        else:
            return torch.arange(0, self._num_fields, device=self._device)

    @property
    def _num_fields(self) -> int:
        return self._global_map_dict["num"]

    @torch.no_grad()
    def _extract_mesh(
        self,
        mesh_file_path: pathlib.Path,
        resolution: Optional[float] = None,
        threshold: Optional[float] = None,
        transform: Optional[torch.Tensor] = None,
        field_ids: Optional[torch.Tensor] = None,
        log_to_rerun: bool = False,
    ) -> None:
        """Extract mesh from scene representation."""
        if self._single_field_id is None:
            # get bounding box of mapped area from fields + radius
            field_positions = self._global_map_dict["positions"][: self._num_fields]
            field_orientations = self._global_map_dict["orientations"][: self._num_fields]
        else:
            if self._num_fields <= self._single_field_id:
                return

            field_positions = self._global_map_dict["positions"][self._single_field_id][None]
            field_orientations = self._global_map_dict["orientations"][self._single_field_id][
                None
            ]

        if transform is None:
            transform = torch.eye(4, device=self._device)
        field_positions = utils.transform_points(field_positions, transform)
        field_orientations = utils.transform_quaternions(field_orientations, transform)

        if self._single_field_id is None:
            if field_ids is not None:
                field_ids = field_ids[field_ids < self._num_fields]
                if len(field_ids) == 0:
                    return
                field_positions = field_positions[field_ids]
                field_orientations = field_orientations[field_ids]

        min_x, min_y, min_z = field_positions.min(dim=0)[0] - 2 * self._field_radius
        max_x, max_y, max_z = field_positions.max(dim=0)[0] + 2 * self._field_radius

        if resolution is None:
            resolution = self._sample_spacing

        # sample grid points
        all_xs = torch.arange(min_x, max_x, step=resolution, device=self._device)
        all_ys = torch.arange(min_y, max_y, step=resolution, device=self._device)
        all_zs = torch.arange(min_z, max_z, step=resolution, device=self._device)

        BLOCK_SIZE = 200  # reduce if running out of memory

        # NOTE subtract 1 here to ensure blocks have always at least two coordinates in each
        #  direction
        x_starts = range(0, len(all_xs) - 1, BLOCK_SIZE)
        y_starts = range(0, len(all_ys) - 1, BLOCK_SIZE)
        z_starts = range(0, len(all_zs) - 1, BLOCK_SIZE)

        all_verts = []
        all_faces = []
        all_vert_colors = []

        # we extract mesh in blocks of size block_size x block_size x block_size
        verts_offset = 0
        for x_s, y_s, z_s in itertools.product(x_starts, y_starts, z_starts):
            block_xs = all_xs[x_s : x_s + BLOCK_SIZE + 1]
            block_ys = all_ys[y_s : y_s + BLOCK_SIZE + 1]
            block_zs = all_zs[z_s : z_s + BLOCK_SIZE + 1]
            xyzs = torch.cartesian_prod(block_xs, block_ys, block_zs)

            if self._single_field_id is None:
                outs = utils.batched_evaluation(
                    lambda x: self._model(
                        x, field_positions, field_orientations, field_ids, use_vmap=False
                    ),
                    xyzs.reshape(-1, 3),
                    self._block_size,
                )
            else:
                xyzs_local = xyzs - field_positions
                xyzs_local = quaternion_apply(
                    quaternion_invert(field_orientations), xyzs_local
                )
                outs = self._model(xyzs_local)

            volume = outs[:, 3].reshape(1, len(block_xs), len(block_ys), len(block_zs))

            if volume.isnan().any() or volume.isinf().any():
                breakpoint()

            if self._geometry_mode == "occupancy":
                volume = torch.sigmoid(self._geometry_factor * volume)
                isolevel = 0.5
                low_is_inside = False
            elif self._geometry_mode == "density":
                isolevel = 30.0  # has to be tuned
                low_is_inside = False
            elif self._geometry_mode == "neus":
                isolevel = 0.0
                low_is_inside = True
            elif self._geometry_mode == "nrgbd":
                isolevel = 0.0
                low_is_inside = True

            if low_is_inside:
                volume = -volume

            if threshold is not None:
                isolevel = threshold

            block_verts, block_faces = marching_cubes(volume, isolevel)
            block_verts = block_verts[0]
            block_faces = block_faces[0]

            if len(block_verts) == 0:
                continue

            # scale vertices back to original xs, ys, zs
            block_verts[:, 0] = (0.5 * block_verts[:, 0] + 0.5) * (
                block_zs[-1] - block_zs[0]
            ) + block_zs[0]
            block_verts[:, 1] = (0.5 * block_verts[:, 1] + 0.5) * (
                block_ys[-1] - block_ys[0]
            ) + block_ys[0]
            block_verts[:, 2] = (0.5 * block_verts[:, 2] + 0.5) * (
                block_xs[-1] - block_xs[0]
            ) + block_xs[0]

            block_faces += verts_offset
            verts_offset += len(block_verts)

            block_verts_xyz = torch.stack(
                (block_verts[:, 2], block_verts[:, 1], block_verts[:, 0]), dim=-1
            )

            if self._single_field_id is None:
                outs = utils.batched_evaluation(
                    lambda x: self._model(
                        x,
                        field_positions,
                        field_orientations,
                        field_ids,
                        use_vmap=False,
                        # avoid black colors on field boundaries
                        field_radius=self._field_radius + 0.1,
                    ),
                    block_verts_xyz,
                    self._block_size,
                )
            else:
                block_verts_xyz_local = block_verts_xyz - field_positions
                block_verts_xyz_local = quaternion_apply(
                    quaternion_invert(field_orientations), block_verts_xyz_local
                )
                outs = self._model(block_verts_xyz_local)

            block_vert_colors = torch.clamp(self._color_factor * outs[..., :3], 0, 1) * 255

            all_verts.append(block_verts_xyz)
            all_faces.append(block_faces)
            all_vert_colors.append(block_vert_colors)

        if len(all_verts) == 0:
            print("Warning: could not extract mesh. Not crossing isosurface.")
            return

        all_verts = torch.cat(all_verts)
        all_faces = torch.cat(all_faces)
        all_vert_colors = torch.cat(all_vert_colors)

        if log_to_rerun:
            # mesh = slam_dataset.Mesh(
            #     vertices=all_verts, indices=all_faces, vertex_colors=all_vert_colors / 255
            # )
            # simplify_resolution = resolution + 0.01
            # while len(mesh.vertices) > 5000000:
            #     print("Simplifying mesh for Rerun vis")
            #     mesh.simplify(simplify_resolution)
            #     simplify_resolution += 0.01

            rr.log(
                "mesh",
                rr.Mesh3D(
                    vertex_positions=all_verts.cpu(),
                    triangle_indices=all_faces.cpu(),
                    vertex_colors=(all_vert_colors / 255).cpu(),
                ),
            )

        fields_file = mesh_file_path.with_name(mesh_file_path.stem + "_fields.txt")
        np.savetxt(fields_file, field_positions.numpy(force=True))
        with open(mesh_file_path, "wb") as fp:
            save_ply(
                fp,
                verts=all_verts,
                faces=all_faces,
                verts_colors=all_vert_colors,
                ascii=False,
                colors_as_uint8=False,
                verts_normals=None,
            )

    @property
    def _est_mesh_path(self) -> pathlib.Path:
        aligned_prefix = "aligned_" if self._gt_from_est is not None else ""
        return self._eval_data_dir / f"{aligned_prefix}final.ply"

    @property
    def _gt_mesh_path(self) -> pathlib.Path:
        return self._dataset.gt_mesh_path


def main() -> None:
    """Entry point."""
    search_paths = [
        "",  # current working dir
        "~/.neural_graph_mapping",  # dir for this package
        os.path.normpath(os.path.join(os.path.dirname(__file__), "config")),
    ]
    parser = argparse.ArgumentParser(description="Run mapping.")
    parser.add_argument("--config", default="neural_graph_map.yaml", nargs="+")
    config = yoco.load_config_from_args(parser, search_paths=search_paths)
    neural_graph_map = NeuralGraphMap(config)
    neural_graph_map.fit()


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.manual_seed(0)
    random.seed(0)
    main()
