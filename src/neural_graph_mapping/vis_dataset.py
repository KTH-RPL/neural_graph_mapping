import argparse
import os
from typing import Literal, Optional, TypedDict

import matplotlib
import numpy as np
import open3d as o3d
import rerun as rr
import torch
import yoco
from torch.utils.data import DataLoader

from neural_graph_mapping import slam_dataset, utils


def _index_to_color(index: int, cmap: str = "tab10") -> tuple[float]:
    return matplotlib.colormaps[cmap](index % matplotlib.colormaps[cmap].N)


def _get_pose(
    frame_id: int,
    dataset: slam_dataset.SLAMDataset,
    pose_source: Literal["gt", "slam_online", "slam_final"],
) -> torch.Tensor:
    if pose_source == "gt":
        return dataset.gt_c2ws[frame_id]
    elif pose_source == "slam_online":
        return dataset.slam_online_c2ws[frame_id]
    elif pose_source == "slam_final":
        return dataset.slam_final_c2ws[frame_id]
    else:
        raise ValueError(f"Unknown configuration {pose_source=}")


def _log_trajectory(
    entity_path: str,
    dataset: slam_dataset.SLAMDataset,
    pose_source: Literal["gt", "slam_online", "slam_final"],
    color: Optional[np.ndarray] = None,
    transform: Optional[torch.Tensor] = None,
) -> None:
    """Log three trajectories."""
    if transform is None:
        transform = torch.eye(4)

    line_strip = []
    for i in range(len(dataset)):
        c2w = transform @ _get_pose(i, dataset, pose_source)

        if c2w.isnan().any():
            continue

        line_strip.append(c2w[:3, 3].numpy())

    rr.log(
        entity_path + "_full",
        rr.LineStrips3D(np.stack(line_strip), colors=color),
        timeless=True,
    )


class DatasetVisualizerConfig(TypedDict, total=False):
    """Configuration dictionary for dataset visualization.

    Attributes:
        poses: At which pose to visualize the moving camera.
        dataset_type: At which pose to visualize the moving camera.
        dataset_config: Config dict passed to dataset constructor.
        max_depth: Replace depth with 0 when larger than this value.
        alignment_method: Alignment method used for SLAM trajectories.
        compute_bounding_box: Whether to compute ground-truth bounding box of pointcloud.
        white_background:
            Whether to use white background for logging (this will log a large white
            icosphere).
    """

    dataset_type: str
    dataset_config: dict
    poses: Literal["gt", "slam_online", "slam_final"]
    max_depth: Optional[float]
    rerun_save: Optional[str]
    rerun_connect_addr: Optional[str]
    alignment_method: Optional[Literal["origin", "umeyama"]]
    compute_bounding_box: bool
    white_background: bool


_default_config: DatasetVisualizerConfig = {
    "dataset_type": None,
    "dataset_config": None,
    "poses": "gt",
    "max_depth": None,
    "rerun_save": None,
    "rerun_connect_addr": None,
    "alignment_method": "umeyama",
    "compute_bounding_box": False,
    "white_background": False,
}


def _available_poses(dataset: slam_dataset.SLAMDataset) -> list[str]:
    available_poses = []
    if dataset.gt_c2ws is not None:
        available_poses.append("gt")
    if dataset.slam_online_c2ws is not None:
        available_poses.append("slam_online")
    if dataset.slam_final_c2ws is not None:
        available_poses.append("slam_final")
    return available_poses


def run_dataset_visualization(config: DatasetVisualizerConfig) -> None:
    """Run visualization of dataset."""
    # parse config
    config = yoco.load_config(config, _default_config)
    dataset_type = utils.str_to_object(config["dataset_type"])
    dataset_config = config["dataset_config"]
    max_depth = config["max_depth"]
    poses = config["poses"]
    rerun_connect_addr = config["rerun_connect_addr"]
    rerun_save = config["rerun_save"]
    alignment_method = config["alignment_method"]
    compute_bounding_box = config["compute_bounding_box"]
    white_background = config["white_background"]

    # initialize dataset
    dataset_config["device"] = "cpu"
    dataset: slam_dataset.SLAMDataset = dataset_type(dataset_config)
    dataset.load_slam_results()
    dataset.set_mode("sequence")
    cam = dataset.camera

    available_poses = _available_poses(dataset)
    if poses not in available_poses:
        print(f"{poses=} not available, using {available_poses[0]} instead.")
        poses = available_poses[0]

    utils.rr_init("dataset visualizer", rrd_path=rerun_save, connect_addr=rerun_connect_addr)

    if white_background:
        # create large white sphere to serve as background
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=100.0)
        sphere.paint_uniform_color([1.0, 1.0, 1.0])
        rr.log(
            "world/background",
            rr.Mesh3D(
                vertex_positions=sphere.vertices,
                triangle_indices=sphere.triangles,
                vertex_colors=sphere.vertex_colors,
            ),
            timeless=True,
        )

    rr.log("world/camera/image", rr.ViewCoordinates.RUB, timeless=True)

    if dataset.up_axis is not None:
        rr.log("world", utils.rr_up_axis(dataset.up_axis), timeless=True)

    if dataset.has_gt_mesh:
        print("Loading ground-truth mesh...")
        gt_mesh = dataset.load_gt_mesh()
        gt_mesh.simplify(0.01)  # avoid issues with logging large meshes
        rr.log(
            "world/gt_mesh",
            rr.Mesh3D(
                vertex_positions=gt_mesh.vertices,
                triangle_indices=gt_mesh.face_indices,
                vertex_colors=gt_mesh.vertex_colors,
                vertex_normals=gt_mesh.vertex_normals
                if gt_mesh.vertex_colors is None
                else None,
            ),
            timeless=True,
        )
    else:
        print("No ground-truth mesh available.")

    gt_from_est = torch.eye(4)
    if alignment_method is not None:
        try:
            gt_from_est = dataset.gt_from_est_transform(alignment_method)
        except ValueError as e:
            print(f"Alignment with {alignment_method=} failed with:\n\t{str(e)}")
            print("Using no alignment instead.")

    if dataset.gt_c2ws is not None:
        _log_trajectory("world/gt", dataset, "gt", _index_to_color(0))
    else:
        print("No ground-truth trajectory available.")

    if dataset.slam_online_c2ws is not None:
        _log_trajectory(
            "world/slam_online", dataset, "slam_online", _index_to_color(1), gt_from_est
        )
    else:
        print("No online SLAM trajectory available.")

    if dataset.slam_final_c2ws is not None:
        _log_trajectory(
            "world/slam_final", dataset, "slam_final", _index_to_color(2), gt_from_est
        )
    else:
        print("No final SLAM trajectory available.")

    vis_step = 3

    data_loader = DataLoader(dataset, num_workers=16, pin_memory=True)

    if poses == "slam_online" or poses == "slam_final":
        pose_transform = gt_from_est
    else:
        pose_transform = torch.eye(4)

    aabb_min = None
    aabb_max = None

    print(dataset.scene_bounds)

    for i, item in enumerate(data_loader):
        rr.set_time_seconds("time", item["time"][0])
        rr.set_time_sequence("frame_id", i)
        if compute_bounding_box:
            depth_image = item["rgbd"][0, :, :, 3]
            points_c = cam.depth_to_pointcloud(depth_image)
            gt_c2w = _get_pose(i, dataset, "gt")
            if gt_c2w.isnan().any():
                continue
            points_w = utils.transform_points(points_c, gt_c2w)
            if aabb_min is None:
                aabb_min = points_w.min(dim=0)[0]
                aabb_max = points_w.max(dim=0)[0]
            else:
                aabb_min = torch.min(aabb_min, points_w.min(dim=0)[0])
                aabb_max = torch.max(aabb_max, points_w.max(dim=0)[0])

            rr.log_point("world/min", aabb_min.numpy(force=True))
            rr.log_point("world/max", aabb_max.numpy(force=True))

        if i % vis_step != 0:
            continue

        depth_image = item["rgbd"][0, :, :, 3]
        if max_depth is not None:
            depth_image[depth_image > max_depth] = 0.0
        color_image = item["rgbd"][0, :, :, :3]

        c2w = pose_transform @ _get_pose(i, dataset, poses)

        if c2w.isnan().any():
            print(f"Skipping frame {i} because transform is missing.")
            continue

        translation = c2w[:3, 3]
        rotation = c2w[:3, :3]
        rr.log(
            "world/camera",
            rr.Transform3D(translation=translation, mat3x3=rotation),
        )
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                image_from_camera=cam.get_projection_matrix(),
                width=cam.width,
                height=cam.height,
            ),
        )
        rr.log("world/camera/image/depth", rr.DepthImage(depth_image, meter=1.0))
        rr.log("world/camera/image/rgb", rr.Image(color_image))

    if compute_bounding_box:
        print(aabb_min, aabb_max)


@torch.no_grad()
def main() -> None:
    """Entry point."""
    search_paths = [
        "",  # current working dir
        "~/.neural_graph_mapping",  # package folder in home dir
        os.path.normpath(os.path.join(os.path.dirname(__file__), "config")),
    ]
    parser = argparse.ArgumentParser(description="Visualize dataset.")
    parser.add_argument("--config", default="scannet_dataset.yaml", nargs="+")
    config = yoco.load_config_from_args(parser, search_paths=search_paths)
    run_dataset_visualization(config)


if __name__ == "__main__":
    main()
