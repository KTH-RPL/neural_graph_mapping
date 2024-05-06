"""Module to cull mesh in various ways.

Adapted from https://github.com/JingwenWang95/neural_slam_eval/
"""
import glob
import os
import pathlib
import time
from copy import deepcopy
from typing import Optional, Literal

import numpy as np
import open3d as o3d
import pyrender
import rerun as rr
import torch
import trimesh
from tqdm import tqdm

from neural_graph_mapping import camera, slam_dataset


def _load_virt_cam_poses(path: pathlib.Path) -> list[np.ndarray]:
    poses = []
    pose_paths = sorted(
        glob.glob(os.path.join(path, "*.txt")), key=lambda x: int(os.path.basename(x)[:-4])
    )
    for pose_path in pose_paths:
        c2w = np.loadtxt(pose_path).reshape(4, 4)
        # virt_cams are stored under OpenCV convention, so need to convert to OpenGL
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        poses.append(c2w)

    # assert len(poses) > 0, "Make sure the path: {} really has virtual views!".format(path)
    print("Added {} virtual views from {}".format(len(poses), path))

    return poses


def _render_depth_maps(
    mesh,
    poses,
    cam: camera.Camera,
    near: float = 0.01,
    far: float = 10.0,
) -> list[np.ndarray]:
    """Render depth map of mesh.

    Args:
        mesh: Mesh to be rendered
        poses: list of camera poses (c2w under OpenGL convention)
        projection_matrix: camera intrinsics [3, 3]
        width: width of image plane
        height: height of image plane
        near: near clip
        far: far clip

    Returns:
        list of rendered depth images [H, W]
    """
    projection_matrix = cam.get_projection_matrix().numpy()

    depth_maps = np.full((len(poses), cam.height, cam.width), np.inf)

    vertices = mesh.vertices
    all_faces = mesh.faces
    for faces in np.array_split(all_faces, np.arange(20_000_000, len(all_faces), 20_000_000)):
        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices, faces), smooth=True)
        scene.add(mesh)

        pyr_camera = pyrender.IntrinsicsCamera(
            fx=projection_matrix[0, 0],
            fy=projection_matrix[1, 1],
            cx=projection_matrix[0, 2],
            cy=projection_matrix[1, 2],
            znear=near,
            zfar=far,
        )
        camera_node = pyrender.Node(camera=pyr_camera, matrix=np.eye(4))
        scene.add_node(camera_node)
        renderer = pyrender.OffscreenRenderer(cam.width, cam.height)
        render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

        for i, pose in enumerate(tqdm(poses)):
            scene.set_pose(camera_node, pose)
            depth = renderer.render(scene, render_flags)
            scene.set_pose(camera_node, pose)
            depth = renderer.render(scene, render_flags)
            mask = (depth < depth_maps[i]) * depth != 0.0
            depth_maps[i, mask] = depth[mask]

        del scene, mesh, renderer, camera_node

    for depth_map in depth_maps:
        depth_map[np.isinf(depth_map)] = 0.0

    return depth_maps


def _render_depth_maps_doublesided(
    mesh, poses, cam: camera.Camera, near: float = 0.01, far: float = 10.0
):
    depth_maps_1 = _render_depth_maps(mesh, poses, cam, near=near, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    depth_maps_2 = _render_depth_maps(mesh, poses, cam, near=near, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[
        :, [2, 1]
    ]  # it's a pass by reference, so I restore the original order

    depth_maps = []
    for i in tqdm(range(len(depth_maps_1))):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where(
            (depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map), depth_maps_2[i], depth_map
        )
        depth_maps.append(depth_map)

    return depth_maps


def _cull_by_bounds(points: np.ndarray, bounds: np.ndarray, eps: float = 0.02) -> np.ndarray:
    """Cull points by axis aligned bounding box (AABB).

    Args:
        points: Points to check. Shape (...,D).
        bounds:
            Bounds of AABB. Shape (2, D). First row contains minimum point, second row maximum
            point.
        eps:
            Bounds will be increased by this amount.

    Returns:
        Boolean mask of points inside AABB.
    """
    inside_mask = np.all(points >= (bounds[0] - eps), axis=1) & np.all(
        points <= (bounds[1] + eps), axis=1
    )
    return inside_mask


def _cull_from_one_pose(
    points: torch.Tensor,
    pose: torch.Tensor,
    cam: camera.Camera,
    remove_occlusion: bool = True,
    rendered_depth: Optional[np.ndarray] = None,
    eps: float = 0.03,
):
    """

    Args:
        points: mesh vertices [V, 3] np array
        pose: c2w under OpenGL convention (right-up-back) [3, 3] np array
        K: camera intrinsics [3, 3] np array
        rendered_depth: rendered depth image (optional)
        remove_occlusion:

    Returns:
    """

    width, height = cam.width, cam.height
    c2w = deepcopy(pose)
    # convert to OpenCV pose
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    w2c = torch.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]

    # pts under camera coordinate frame
    camera_space = rotation @ points.T + translation[:, None]  # [3, N]
    uvz = (cam.get_projection_matrix().to("cuda") @ camera_space).T  # [N, 3]
    pz = uvz[:, 2] + 1e-8
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # step 1: frustum
    in_frustum_mask = (0 <= px) & (px <= width - 1) & (0 <= py) & (py <= height - 1) & (pz > 0)
    u = torch.clip(px, 0, width - 1).int()
    v = torch.clip(py, 0, height - 1).int()

    # step 2: self occlusion
    obs_mask = in_frustum_mask
    if remove_occlusion:
        assert rendered_depth is not None, "remove_occlusion requires rendered depth image!!!"
        obs_mask = in_frustum_mask & (pz < (rendered_depth[v, u] + eps))

    return in_frustum_mask.int(), obs_mask.int()


def _apply_culling_strategy(
    points,
    poses,
    cam: camera.Camera,
    rendered_depth_list: Optional[np.ndarray] = None,
    remove_occlusion: bool = True,
    verbose: bool = False,
    virt_cam_starts: int = -1,
    eps: float = 0.03,
):
    in_frustum_mask = torch.zeros(points.shape[0], device="cuda")
    obs_mask = torch.zeros(points.shape[0], device="cuda")
    points = torch.from_numpy(points).to(device="cuda", dtype=torch.float)
    for i, pose in enumerate(tqdm(poses)):
        if verbose:
            print("Processing pose " + str(i + 1) + " out of " + str(len(poses)))
        rr.set_time_sequence("frame_id", i)
        rendered_depth = rendered_depth_list[i] if rendered_depth_list is not None else None
        in_frustum, obs = _cull_from_one_pose(
            points,
            torch.from_numpy(pose).to(device="cuda", dtype=torch.float),
            cam=cam,
            rendered_depth=torch.from_numpy(rendered_depth).to("cuda"),
            remove_occlusion=remove_occlusion,
            eps=eps,
        )
        obs_mask = obs_mask + obs
        # NOTE virtual camera views shouldn't contribute to in_frustum_mask, it only adds more
        #  entries to obs_mask
        if virt_cam_starts < 0 or i < virt_cam_starts:
            in_frustum_mask = in_frustum_mask + in_frustum

    return in_frustum_mask.numpy(force=True), obs_mask.numpy(force=True)


def _cull_mesh(
    mesh_path: pathlib.Path,
    save_path: pathlib.Path,
    dataset: slam_dataset.SLAMDataset,
    remove_occlusion: bool = True,
    virtual_cameras: bool = False,
    subdivide: bool = True,
    max_edge: float = 0.1,
    th_obs: float = 0,
    eps: float = 0.03,
    silent: bool = True,
    platform: str = "egl",
) -> None:
    """Cull a mesh.

    Args:
        mesh_path: Path of mesh to cull.
        save_path: Output path of culled mesh.
        dataset: Dataset used for culling.
        remove_occlusion: Whether self occlusions should be removed.
        virtual_cameras:
        bounds:
        th_obs:
        silent:
        platform:
        dataset
    """
    cam = dataset.camera.scaled_camera(0.5)

    # load original mesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    if subdivide:
        mesh = mesh.subdivide_to_size(max_edge)

    vertices = mesh.vertices  # [V, 3]
    triangles = mesh.faces  # [F, 3]
    colors = mesh.visual.vertex_colors if hasattr(mesh.visual, "vertex_colors") else None

    custom_scene_bounds = dataset.custom_scene_bounds
    auto_scene_bounds = dataset.scene_bounds
    scene_bounds = None
    if custom_scene_bounds is not None and auto_scene_bounds is not None:
        # take minimum of the two bounds
        minimum = torch.max(custom_scene_bounds[0], auto_scene_bounds[0])
        maximum = torch.min(custom_scene_bounds[1], auto_scene_bounds[1])
        scene_bounds = torch.stack((minimum, maximum))
    elif custom_scene_bounds is not None:
        scene_bounds = custom_scene_bounds
    elif auto_scene_bounds is not None:
        scene_bounds = auto_scene_bounds
    else:
        scene_bounds = None

    if scene_bounds is not None:
        inside_mask = _cull_by_bounds(vertices, scene_bounds.numpy())
        inside_mask = (
            inside_mask[triangles[:, 0]]
            | inside_mask[triangles[:, 1]]
            | inside_mask[triangles[:, 2]]
        )
        triangles = triangles[inside_mask, :]
    else:
        print("No scene bounds available. Skipping culling by bounds.")

    os.environ["PYOPENGL_PLATFORM"] = platform

    c2w_list = list(dataset.gt_c2ws.numpy(force=True)[::2])

    # add virtual cameras to camera poses list
    if virtual_cameras:
        virt_cam_starts = len(c2w_list)
        virt_cam_path = dataset.scene_dir_path / "virtual_cameras"
        c2w_list = c2w_list + _load_virt_cam_poses(virt_cam_path)
    else:
        virt_cam_starts = -1

    # update the mesh vertices and faces
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()

    if remove_occlusion:
        print("rendering depth maps...")
        rendered_depth_maps = _render_depth_maps_doublesided(
            mesh,
            c2w_list,
            cam,
            near=0.01,
            far=10.0,
        )
    else:
        rendered_depth_maps = None

    # start culling
    points = vertices[:, :3]  # [V, 3]
    in_frustum_mask, obs_mask = _apply_culling_strategy(
        points,
        c2w_list,
        cam,
        rendered_depth_list=rendered_depth_maps,
        verbose=(not silent),
        remove_occlusion=remove_occlusion,
        virt_cam_starts=virt_cam_starts,
        eps=eps,
    )
    inf1 = in_frustum_mask[triangles[:, 0]]  # [F, 3]
    inf2 = in_frustum_mask[triangles[:, 1]]
    inf3 = in_frustum_mask[triangles[:, 2]]
    in_frustum_mask = (inf1 > th_obs) | (inf2 > th_obs) | (inf3 > th_obs)
    if remove_occlusion:
        obs1 = obs_mask[triangles[:, 0]]
        obs2 = obs_mask[triangles[:, 1]]
        obs3 = obs_mask[triangles[:, 2]]
        obs_mask = (obs1 > th_obs) | (obs2 > th_obs) | (obs3 > th_obs)
        valid_mask = in_frustum_mask & obs_mask  # [F,]
    else:
        valid_mask = in_frustum_mask
    triangles_observed = triangles[valid_mask, :]

    # save culled mesh
    mesh = trimesh.Trimesh(vertices, triangles_observed, vertex_colors=colors, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.export(save_path)

CullingMethod = Literal["virt_cams", "occlusion", "frustum"]

def cull_mesh(
    in_mesh_path: pathlib.Path,
    out_mesh_path: pathlib.Path,
    culling_method: CullingMethod,
    dataset: slam_dataset.SLAMDataset,
) -> None:
    """Cull mesh and save culled mesh to path."""
    if culling_method == "virt_cams":
        remove_occlusion = True
        virtual_cameras = True
    elif culling_method == "occlusion":
        remove_occlusion = True
        virtual_cameras = False
    elif culling_method == "frustum":
        remove_occlusion = False
        virtual_cameras = False
    else:
        raise ValueError(f"Unknown culling method {culling_method}")

    _cull_mesh(
        mesh_path=in_mesh_path,
        save_path=out_mesh_path,
        dataset=dataset,
        remove_occlusion=remove_occlusion,
        virtual_cameras=virtual_cameras,
        th_obs=0,
        eps=0.03,
        silent=True,
        platform="egl",
    )
