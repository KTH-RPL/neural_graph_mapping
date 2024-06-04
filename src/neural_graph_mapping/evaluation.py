"""This module contains metrics and functions to evaluate neural SLAM methods."""
import pathlib

import numpy as np
import open3d as o3d
import rerun as rr
import torch
import torchmetrics
import torchmetrics.functional as tmf
import trimesh
from scipy import spatial

from neural_graph_mapping import mesh_culling, slam_dataset

_lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(normalize=True).to(
    "cuda"
)


def ssim(prediction: torch.Tensor, target: torch.Tensor, crop: int = 0) -> float:
    prediction = prediction.clone()
    target = target.clone()
    if crop > 0:
        target = target[crop:-crop, crop:-crop]
        prediction = prediction[crop:-crop, crop:-crop]
    prediction = prediction.permute(2, 0, 1).unsqueeze(0)
    prediction.clamp_(0.0, 1.0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    target.clamp_(0.0, 1.0)
    return tmf.structural_similarity_index_measure(prediction, target, data_range=1.0).item()


def lpips(prediction: torch.Tensor, target: torch.Tensor, crop: int = 0) -> float:
    prediction = prediction.clone()
    target = target.clone()
    if crop > 0:
        target = target[crop:-crop, crop:-crop]
        prediction = prediction[crop:-crop, crop:-crop]
    prediction = prediction.permute(2, 0, 1).unsqueeze(0)
    prediction.clamp_(0.0, 1.0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    target.clamp_(0.0, 1.0)
    return _lpips(prediction, target).item()


def psnr(prediction: torch.Tensor, target: torch.Tensor, crop: int = 0) -> float:
    prediction = prediction.clone()
    target = target.clone()
    if crop > 0:
        target = target[crop:-crop, crop:-crop]
        prediction = prediction[crop:-crop, crop:-crop]
    prediction = prediction.permute(2, 0, 1).unsqueeze(0)
    prediction.clamp_(0.0, 1.0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    target.clamp_(0.0, 1.0)
    return tmf.peak_signal_noise_ratio(prediction, target, data_range=1.0).item()


def depthl1(prediction: torch.Tensor, target: torch.Tensor, crop: int = 0) -> float:
    mask = target != 0
    depthl1 = (prediction[mask] - target[mask]).abs().mean()
    return depthl1.item()


def reconstruction_f1(
    gt_points: np.ndarray,
    rec_points: np.ndarray,
    dist_th: float = 0.05,
) -> float:
    comp = completion_ratio(gt_points, rec_points, dist_th)
    acc = accuracy_ratio(gt_points, rec_points, dist_th)
    return 2.0 / (1.0 / comp + 1.0 / acc)


def completion_ratio(
    gt_points: np.ndarray,
    rec_points: np.ndarray,
    dist_th: float = 0.05,
    rerun_vis: bool = False,
) -> float:
    gen_points_kd_tree = spatial.KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    complete_mask = distances < dist_th
    comp_ratio = np.mean(complete_mask.astype(np.float32))
    if rerun_vis:
        rr.log("world/uncompleted", rr.Points3D(gt_points[~complete_mask]))
    return comp_ratio.item()


def accuracy_ratio(
    gt_points: np.ndarray,
    rec_points: np.ndarray,
    dist_th: float = 0.05,
    rerun_vis: bool = False,
) -> float:
    gt_points_kd_tree = spatial.KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    accurate_mask = distances < dist_th
    acc_ratio = np.mean(accurate_mask.astype(np.float32))
    if rerun_vis:
        rr.log("world/inaccurate", rr.Points3D(rec_points[~accurate_mask]))
    return acc_ratio.item()


def median_accuracy(gt_points: np.ndarray, rec_points: np.ndarray) -> float:
    gt_points_kd_tree = spatial.KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.median(distances)
    return acc.item()


def mean_accuracy(gt_points: np.ndarray, rec_points: np.ndarray) -> float:
    gt_points_kd_tree = spatial.KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc.item()


def mean_completion(gt_points: np.ndarray, rec_points: np.ndarray) -> float:
    gt_points_kd_tree = spatial.KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp.item()


def median_completion(gt_points: np.ndarray, rec_points: np.ndarray) -> float:
    gt_points_kd_tree = spatial.KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.median(distances)
    return comp.item()


def _align_mesh(
    source_mesh_path: pathlib.Path,
    target_mesh_path: pathlib.Path,
    aligned_mesh_path: pathlib.Path,
) -> None:
    """Use ICP to align mesh with ground-truth mesh, assuming good initial estimate."""
    print(f"Aligning mesh {target_mesh_path} with {source_mesh_path}")

    source_mesh = o3d.io.read_triangle_mesh(str(source_mesh_path))
    target_mesh = o3d.io.read_triangle_mesh(str(target_mesh_path))
    o3d_target_pc = o3d.geometry.PointCloud(target_mesh.vertices)
    target_mesh.compute_vertex_normals()
    o3d_target_pc.normals = target_mesh.vertex_normals
    o3d_source_pc = o3d.geometry.PointCloud(source_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_source_pc,
        o3d_target_pc,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    transformation = reg_p2p.transformation.copy()

    # overwrite mesh with aligned mesh
    o3d.io.write_triangle_mesh(str(aligned_mesh_path), source_mesh.transform(transformation))


def _evaluate_postprocessed_meshes(
    est_mesh_path: pathlib.Path,
    gt_mesh_path: pathlib.Path,
    num_points: int,
    rerun_vis: bool,
) -> dict:
    gt_mesh = trimesh.load(gt_mesh_path)
    est_mesh = trimesh.load(est_mesh_path)

    if rerun_vis:
        rr.log(
            "world/gt_mesh",
            rr.Mesh3D(
                vertex_positions=gt_mesh.vertices,
                triangle_indices=gt_mesh.faces,
                vertex_colors=gt_mesh.vertex_normals,
            )
        )
        rr.log(
            "world/est_mesh",
            rr.Mesh3D(
                vertex_positions=est_mesh.vertices,
                triangle_indices=est_mesh.faces,
                vertex_normals=est_mesh.vertex_normals,
            )
        )

    gt_points = trimesh.sample.sample_surface(gt_mesh, num_points)[0]
    est_points = trimesh.sample.sample_surface(est_mesh, num_points)[0]

    if rerun_vis:
        rr.log("world/gt_points", rr.Points3D(gt_points))
        rr.log("world/est_points", rr.Points3D(est_points))

    return {
        "median_acc": median_accuracy(gt_points, est_points),
        "median_comp": median_completion(gt_points, est_points),
        "acc": mean_accuracy(gt_points, est_points),
        "comp": mean_completion(gt_points, est_points),
        "acc_ratio": accuracy_ratio(gt_points, est_points, 0.05, rerun_vis=rerun_vis),
        "acc_ratio_1cm": accuracy_ratio(gt_points, est_points, 0.01),
        "comp_ratio": completion_ratio(gt_points, est_points, 0.05, rerun_vis=rerun_vis),
        "comp_ratio_1cm": completion_ratio(gt_points, est_points, 0.01),
        "f1_5cm": reconstruction_f1(gt_points, est_points, 0.05),
        "f1_1cm": reconstruction_f1(gt_points, est_points, 0.01),
    }


def evaluate_raw_mesh(
    est_mesh_path: pathlib.Path,
    dataset: slam_dataset.SLAMDataset,
    gt_culling_method: mesh_culling.CullingMethod,
    est_culling_method: mesh_culling.CullingMethod,
    mesh_alignment: bool,
    num_points: int,
    rerun_vis: bool,
) -> dict:
    """Evaluate mesh with set of metrics."""
    # Prepare groun-truth
    gt_mesh_path = dataset.gt_mesh_path
    culled_gt_mesh_path = gt_mesh_path.with_stem(
        f"eval_{gt_mesh_path.stem}_culled_{gt_culling_method}"
    )
    if not culled_gt_mesh_path.is_file():
        mesh_culling.cull_mesh(
            gt_mesh_path,
            culled_gt_mesh_path,
            gt_culling_method,
            dataset,
        )

    # Prepare estimated (first align, then cull)
    if mesh_alignment:
        aligned_est_mesh_path = est_mesh_path.with_stem("eval_aligned_" + est_mesh_path.stem)
        if not aligned_est_mesh_path.is_file():
            _align_mesh(est_mesh_path, culled_gt_mesh_path, aligned_est_mesh_path)
        est_mesh_path = aligned_est_mesh_path

    culled_est_mesh_path = est_mesh_path.with_stem(
        f"eval_{est_mesh_path.stem}_culled_{est_culling_method}"
    )
    if not culled_est_mesh_path.is_file():
        mesh_culling.cull_mesh(
            est_mesh_path, culled_est_mesh_path, est_culling_method, dataset
        )

    return _evaluate_postprocessed_meshes(
        culled_est_mesh_path, culled_gt_mesh_path, num_points, rerun_vis
    )
