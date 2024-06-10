"""Abstract SLAMDataset class."""
import abc
import copy
import json
import logging
import os
import pathlib
from typing import List, Literal, Optional, TypedDict

import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
import tqdm
import yoco
from evo.core import trajectory
from scipy.spatial import transform
from torch.utils.data import DataLoader

from neural_graph_mapping import camera, graph, utils


class Mesh:
    """Mesh stored as o3d.geometry.TriangleMesh with properties converting to torch.Tensors."""

    _o3d_mesh: o3d.geometry.TriangleMesh

    def __init__(
        self,
        o3d_mesh: Optional[o3d.geometry.TriangleMesh] = None,
        vertices: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        vertex_colors: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize mesh open3d mesh."""
        assert not (o3d_mesh is not None and vertices is not None)

        if o3d_mesh is not None:
            self._o3d_mesh = o3d_mesh
            self._o3d_mesh.compute_vertex_normals()

        if vertices is not None:
            vertices = vertices.numpy(force=True)
            indices = indices.numpy(force=True)
            self._o3d_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(indices)
            )

        if vertex_colors is not None:
            self._o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                vertex_colors.numpy(force=True)
            )

    @property
    def o3d_mesh(self) -> o3d.geometry.TriangleMesh:
        """Return open3d mesh."""
        return self._o3d_mesh

    def simplify(self, voxel_size: float) -> None:
        """Simplify mesh in-place."""
        self._o3d_mesh = self._o3d_mesh.simplify_vertex_clustering(voxel_size)

    @property
    def vertices(self) -> torch.Tensor:
        """Return vertex positions. Shape (num_vertices, 3)."""
        return torch.from_numpy(np.asarray(self._o3d_mesh.vertices))

    @property
    def face_indices(self) -> torch.Tensor:
        """Return face indices. Shape (num_faces, 3)."""
        return torch.from_numpy(np.asarray(self._o3d_mesh.triangles))

    @property
    def vertex_colors(self) -> Optional[torch.Tensor]:
        """Return vertex_colors or None. Shape (num_vertices, 3)."""
        vertex_colors = torch.from_numpy(np.asarray(self._o3d_mesh.vertex_colors))
        if len(vertex_colors) == 0:
            vertex_colors = None
        return vertex_colors

    @property
    def vertex_normals(self) -> torch.Tensor:
        """Return vertex normals. Shape (num_vertices, 3)."""
        return torch.from_numpy(np.asarray(self._o3d_mesh.vertex_normals))


class SLAMDataset(torch.utils.data.Dataset):
    """Abstract SLAM dataset.

    It assumes that the dataset has a root directory and one or multiple named scenes.
    The exact directory structure of the dataset is defined by derived classes.

    Two modes are defined: sequence and rays.

    - For sequence, each sample is a dictionary containing the following objects:
        time (torch.Tensor): Timestamp of image in seconds. Scalar.
        rgbd (torch.Tensor):
            RGBD image. 0-1 for RGB and in meters for depth. Shape (H,W,4).
        c2w (torch.Tensor):
            Transformation matrix from camera to world coordinates. Shape (4,4).
            OpenGL convention.
    - For ray, each sample is a dictionary containing the following objects:
        rgbd (torch.LongTensor): Shape (4,).
        ij (torch.LongTensor): Row and column of pixel. Shape (2,).
        c2w (torch.Tensor):
            Transformation matrix from camera to world coordinates. Shape (4,4).
            OpenGL convention.
    """

    device: str
    camera: camera.Camera
    gt_c2ws: Optional[torch.Tensor]
    slam_online_c2ws: torch.Tensor
    slam_final_c2ws: Optional[torch.Tensor]
    slam_c2w_dict: Optional[dict]
    slam_pg_dict: Optional[dict]
    up_axis: Optional[Literal["x", "y", "z", "-x", "-y", "-z"]]
    root_dir_path: pathlib.Path
    scene: str

    class Config(TypedDict, total=False):
        """Configuration dictionary for SLAMDataset.

        These attributes are common to all SLAMDatasets. See dataset-specific config for
        dataset specific configuration.

        Attributes:
            root_dir: Root directory of dataset.
            scene: Scene name.
            slam_final_file:
                File name of precomputed SLAM result, containing full estimated trajectory at
                the end of the sequence.
            slam_c2w_file:
                File name of precomputed SLAM result, containing updated c2ws for
                each frame. Optional, only required for pose graph functions.
            slam_pg_file:
                File name of precomputed SLAM result, containing pose graph information
                for each frame. Optional, only required for pose graph functions.
            slam_essential_weight_threshold:
                Only edges with higher or equal weight will be included in essentia
                graph.
            up_axis:
                Defines up-axis in dataset based on ground-truth.
                None, if no ground-truth available.
            device:
                Device to store data on. If enough memory is available, transfer time
                can be reduced with a GPU.
            pose_source:
                Which poses to return. Allows to test algorithms with ground-truth poses.
            pg_source:
                Which pose graph to return.
                "fixed_kf_freq" is only supported when pose_source="gt".
            fixed_kf_freq: How many keyframes are inserted when pg_source="fixed_frequency".
        """

        root_dir: str
        scene: str
        slam_final_file: Optional[str]
        slam_c2w_file: Optional[str]
        slam_pg_file: Optional[str]
        slam_essential_weight_threshold: int
        up_axis: Optional[str]
        device: str
        pose_source: Literal["slam", "gt"]
        pg_source: Literal["slam", "fixed_kf_freq"]
        fixed_kf_freq: int

    default_config: Config = {
        "slam_final_file": None,
        "slam_c2w_file": None,
        "slam_pg_file": None,
        "slam_essential_weight_threshold": 10,
        "up_axis": None,
        "device": "cpu",
        "pose_source": "slam",
        "pg_source": "slam",
        "fixed_kf_freq": 5,
    }

    def __init__(self, config: Config) -> None:
        """Initialize the dataset.

        This will parse SLAMDataset config and load SLAM data if available.

        Derived classes should call super().__init__() after loading their own config.

        Args:
            config:
                Configuration dictionary of dataset. Provided dictionary will be merged
                with default_dict. See SLAMDataset.Config for supported keys.
        """
        self.config = yoco.load_config(config, current_dict=SLAMDataset.default_config)
        self._parse_config()
        self.gt_c2ws = None
        self.slam_online_c2ws = None
        self.slam_final_c2ws = None
        self.slam_pg_dict = None
        self.slam_c2w_dict = None
        self._mode = None

    def _parse_config(self) -> None:
        """Parse configuration dictionary into member variables.

        This function parses a SLAMDataset.Config dictionary.

        Therefore, derived classes should call super()._parse_config() before parsing
        their own config or duplicate the required code.
        """
        self.root_dir_path = pathlib.Path(self.config["root_dir"])
        self.scene = self.config["scene"]
        self._slam_essential_weight_threshold = self.config["slam_essential_weight_threshold"]
        self._slam_final_file = self.config["slam_final_file"]
        self._slam_c2w_file = self.config["slam_c2w_file"]
        self._slam_pg_file = self.config["slam_pg_file"]
        self.device = self.config["device"]
        self.up_axis = self.config["up_axis"]
        self._pose_source = self.config["pose_source"]
        self._pg_source = self.config["pg_source"]
        self._fixed_kf_freq = self.config["fixed_kf_freq"]

        assert self._pose_source in ["gt", "slam"]
        assert self._pg_source in ["fixed_kf_freq", "slam"]

    def _resolve_slam_file(self, filepath: os.PathLike) -> pathlib.Path:
        """Resolve filename / filepath.

        Search scene directory if filepath is not an absolute path already.

        Args:
            filepath: Filename, relative filepath, or absolute filepath.

        Returns: Absolute filepath.
        """
        search_paths = [self.scene_dir_path, "."]
        path = yoco.resolve_path(str(filepath), search_paths=search_paths)
        return pathlib.Path(path)

    @staticmethod
    @abc.abstractmethod
    def get_available_scenes(root_dir: str) -> list[str]:
        """Return available scenes at the given root directory."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def num_images(self) -> int:
        """Return number of images in this dataset."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def scene_dir_path(self) -> pathlib.Path:
        """Return directory for the current scene."""
        raise NotImplementedError()

    @property
    def has_gt_mesh(self) -> bool:
        """Return whether dataset has a ground-truth mesh and it exists."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def gt_mesh_path(self) -> pathlib.Path:
        """Return path of ground-truth mesh.

        Should only be called and / or implemented when has_gt_mesh is True.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load_gt_mesh(self) -> Mesh:
        """Return ground-truth mesh or None if not available for this dataset.

        Should only be called and / or implemented when has_gt_mesh is True.
        """
        raise NotImplementedError()

    def gt_from_est_transform(
        self, alignment_method: Literal["origin", "umeyama"]
    ) -> torch.Tensor:
        """Return transform that aligns final slam c2ws with gt c2ws.

        Return:
            Transform that aligns self.slam_final_c2ws with self.gt_c2ws.
        """
        if self.slam_final_c2ws is None:
            raise ValueError(
                "Cannot align, because final estimated trajectory is not available."
            )
        if self.gt_c2ws is None:
            raise ValueError("Cannot align, because ground-truth trajectory is not available.")

        gt_mask = ~self.gt_c2ws.isnan().reshape(-1, 4 * 4).any(dim=1)
        slam_final_mask = ~self.slam_final_c2ws.isnan().reshape(-1, 4 * 4).any(dim=1)
        mask = gt_mask * slam_final_mask
        gt_c2ws = self.gt_c2ws[mask].numpy(force=True)
        slam_final_c2ws = self.slam_final_c2ws[mask].numpy(force=True)

        gt_trajectory = trajectory.PosePath3D(poses_se3=gt_c2ws)
        slam_final_trajectory = trajectory.PosePath3D(poses_se3=slam_final_c2ws)
        if alignment_method == "umeyama":
            # rot, trans, scale = slam_final_trajectory.align(gt_trajectory, correct_scale=True)
            # print(scale)
            # exit()
            rot, trans, scale = slam_final_trajectory.align(gt_trajectory)
            assert scale == 1.0
            gt_from_est = np.eye(4)
            gt_from_est[:3, :3] = rot
            gt_from_est[:3, 3] = trans
        elif alignment_method == "origin":
            gt_from_est = slam_final_trajectory.align_origin(gt_trajectory)
        else:
            raise ValueError(f"Unknown {alignment_method=}")

        return torch.from_numpy(gt_from_est).to(dtype=torch.float, device=self.device)

    @abc.abstractmethod
    def set_mode(self, mode: Literal["ray", "sequence"]) -> None:
        """Set specified mode and setup / load dataset accordingly.

        If None, nothing should be done.

        Args:
            mode: Mode to set.
        """
        raise NotImplementedError()

    def get_slam_c2ws(
        self, frame_id: Optional[int] = None, at_frame_id: Optional[int] = None
    ) -> torch.Tensor:
        """Return estimated camera pose of a given frame id, at a given frame id.

        Args:
            frame_id:
                Frame ids of keyframe to return c2w for. If None, all c2ws are returned
                as a sparse matrix with keyframe ids as indices.
            at_frame_id:
                Id of frame at which the returned transform was estimated.
                If None, frame_ids == at_frame_id is assumed. Only supported for single
                frame id.

        Returns:
            Transform from camera to world coordinates of frame frame_id, estimated at
            frame at_frame_id. Tensor values will be NaN is transform is not available, e.g.,
            when SLAM system lost track.
        """
        assert not (frame_id is None and at_frame_id is None)

        if at_frame_id is None:
            at_frame_id = frame_id

        if isinstance(frame_id, int):
            return self.slam_c2w_dict[at_frame_id][frame_id]
        else:
            return self.slam_c2w_dict[at_frame_id]

    def get_slam_essential_graph(self, at_frame_id: int) -> dict:
        """Return essential graph (vertices and edges) at a given frame id.

        The essential graph contains loop closure edges and covisibility edges with
        weight greater or equal to specified weight threshold.

        Args:
            at_frame_id:
                Id of the frame for which the pose graph should be returned.
                That is, the pose graph up to and including that frame.
        """
        return self.slam_pg_dict[at_frame_id]

    @utils.benchmark
    def is_keyframe(self, frame_id: int, at_frame_id: Optional[int] = None) -> bool:
        """Check if a frame is a keyframe in the SLAM results."""
        if at_frame_id is None:
            at_frame_id = frame_id

        return frame_id in self.slam_pg_dict[at_frame_id]

    def load_slam_results(self) -> None:
        """Load precomputed SLAM results.

        Side effects:
            Populates the slam_... attributes according to Config.pg_source and
            Config.pose_source.
        """
        logging.debug("SLAMDataset.load_slam_results")

        if self._slam_c2w_file is not None and self._pose_source == "slam":
            self._load_slam_c2w_file()
        elif self._pose_source == "gt":
            self.slam_online_c2ws = self.gt_c2ws
            self._create_gt_c2w_dict()

        if self._slam_pg_file is not None and self._pg_source == "slam":
            self._load_slam_pg_file()
        elif self._pg_source == "fixed_kf_freq":
            self._create_fixed_kf_freq_pg_dict()

        if self._slam_final_file is not None and self._pose_source == "slam":
            self._load_slam_final_file()
        elif self._pose_source == "gt":
            self.slam_final_c2ws = self.gt_c2ws

    def _create_fixed_kf_freq_pg_dict(self) -> None:
        """Create pose graph based on fixed_kf_freq.

        Side-effects:
            Populates the slam_pg_dict attribute.
        """
        self.slam_pg_dict = {}
        current_kf_ids = set()
        for frame_id in range(self.num_images):
            if frame_id % self._fixed_kf_freq == 0:
                current_kf_ids.add(frame_id)
                fully_connected_graph = {
                    kf_id: copy.deepcopy(current_kf_ids) for kf_id in current_kf_ids
                }
                self.slam_pg_dict[frame_id] = fully_connected_graph
            self.slam_pg_dict[frame_id] = fully_connected_graph

    def _load_slam_pg_file(self) -> None:
        """Load pose graph file as specified by slam_pg_file.

        Side-effects:
            Populates the slam_pg_dict attribute.

        Raises:
            ValueError: If slam_pg_file is None.
        """
        if self._slam_pg_file is None:
            raise ValueError("Could not load slam_pg_file because it was not specified.")

        self.slam_pg_dict = {}

        slam_pg_json_path = self._resolve_slam_file(self._slam_pg_file)
        slam_pg_dict_pt_path = slam_pg_json_path.with_name("slam_pg_dict.pt")

        if slam_pg_dict_pt_path.is_file():
            self.slam_pg_dict = torch.load(slam_pg_dict_pt_path, map_location=self.device)
            return

        with open(slam_pg_json_path) as f:
            pg_data = json.load(f)

        for at_frame_id in tqdm.tqdm(
            range(self.num_images), "Load pose graph information", dynamic_ncols=True
        ):
            at_frame_str = str(at_frame_id)
            if at_frame_str in pg_data:
                latest_pg_dicts = pg_data[at_frame_str]
                # make sure the keyframe is not immediately removed again
                if at_frame_id in [pg_dict["KF"] for pg_dict in latest_pg_dicts]:
                    latest_graph = _pg_dicts_to_essential_graph(
                        latest_pg_dicts, self._slam_essential_weight_threshold
                    )

            # ensure pose of all keyframes is available
            removed_frame_ids = []
            for frame_id in list(latest_graph.keys()):
                if frame_id not in self.slam_c2w_dict[at_frame_id].indices():
                    removed_frame_ids.append(frame_id)

            for removed_frame_id in removed_frame_ids:
                latest_graph = graph.remove_vertex(latest_graph, removed_frame_id)

            self.slam_pg_dict[at_frame_id] = latest_graph

        torch.save(self.slam_pg_dict, slam_pg_dict_pt_path)

    def _create_gt_c2w_dict(self) -> None:
        """Create slam_c2w_dict with ground-truth poses.

        Side-effects:
            Populates the slam_c2w_dict attribute.
        """
        gt_c2ws = torch.sparse_coo_tensor(
            [list(range(self.num_images))], self.gt_c2ws, device=self.device
        ).coalesce()
        self.slam_c2w_dict = {at_frame_id: gt_c2ws for at_frame_id in range(self.num_images)}

    def _load_slam_c2w_file(self) -> None:
        """Load final SLAM result from slam_c2ws_file.

        Side-effects:
            Populates the slam_c2w_dict and slam_online_c2ws attributes.

        Raises:
            ValueError: If slam_c2ws_file is None.
        """
        if self._slam_c2w_file is None:
            raise ValueError("Could not load slam_c2w_file because it was not specified.")

        self.slam_c2w_dict = {}

        slam_c2w_path = self._resolve_slam_file(self._slam_c2w_file)
        slam_c2w_dict_pt_path = slam_c2w_path.with_name("slam_c2w_dict.pt")
        slam_online_c2ws_pt_path = slam_c2w_path.with_name("slam_online_c2ws.pt")

        if slam_c2w_dict_pt_path.is_file() and slam_online_c2ws_pt_path.is_file():
            self.slam_c2w_dict = torch.load(slam_c2w_dict_pt_path, map_location=self.device)
            self.slam_online_c2ws = torch.load(
                slam_online_c2ws_pt_path, map_location=self.device
            )
            for k in self.slam_c2w_dict:
                self.slam_c2w_dict[k] = self.slam_c2w_dict[k].coalesce()
            return

        with open(slam_c2w_path) as f:
            c2w_data = json.load(f)

        online_c2ws = torch.full((self.num_images, 4, 4), torch.nan)
        for at_frame_id in tqdm.tqdm(
            range(self.num_images), "Load online trajectory", dynamic_ncols=True
        ):
            at_frame_str = str(at_frame_id)
            frame_ids = []
            frame_ids_set = set()  # to quickly skip duplicates
            c2ws = []
            for frame_str, pose_vector in c2w_data[at_frame_str].items():
                c2w = _pose_vector_to_4x4(pose_vector)
                if frame_str == "cur":
                    frame_str = at_frame_str
                    online_c2ws[at_frame_id] = c2w
                frame_id = int(frame_str)
                if frame_id in frame_ids_set:
                    continue
                frame_ids_set.add(frame_id)
                frame_ids.append(frame_id)
                c2ws.append(c2w)

            # add nan c2w if current is missing
            if at_frame_id not in frame_ids:
                c2ws.append(torch.full((4, 4), torch.nan))
                frame_ids.append(at_frame_id)

            c2ws = torch.stack(c2ws)
            self.slam_c2w_dict[at_frame_id] = torch.sparse_coo_tensor(
                [frame_ids], c2ws, device=self.device
            ).coalesce()

        self.slam_online_c2ws = online_c2ws.to(self.device)

        torch.save(self.slam_online_c2ws, slam_online_c2ws_pt_path)
        torch.save(self.slam_c2w_dict, slam_c2w_dict_pt_path)

    def _load_slam_final_file(self) -> None:
        """Load final SLAM result from slam_final_file.

        Side-effects:
            Populates the slam_final_c2ws attribute.

        Raises:
            ValueError: If slam_final_file is None.
        """
        if self._slam_final_file is None:
            raise ValueError("Could not load slam_final_file, because it was not specified.")

        slam_final_path = self._resolve_slam_file(self._slam_final_file)
        results = np.loadtxt(slam_final_path)

        # FIXME this shouldn't be necessary if final.txt has correct frame numbers
        has_c2w_mask = self.slam_online_c2ws.view(-1, 4 * 4).isfinite().all(dim=-1)
        ncid_from_cid = torch.arange(self._num_images)[has_c2w_mask]

        c2ws = torch.full((self.num_images, 4, 4), torch.nan)
        for i, result in enumerate(results):
            # this is how it should be
            # frame_id = int(result[0])
            frame_id = ncid_from_cid[i]
            pose_vector = result[1:]
            c2ws[frame_id] = _pose_vector_to_4x4(pose_vector)

        self.slam_final_c2ws = c2ws.to(self.device)

    @property
    def scene_bounds(self) -> Optional[torch.Tensor]:
        """Return tight scene bounds according to ground-truth trajectory.

        Returns:
            None if ground-truth are not known for the scene.
            Otherwise array of shape (2, 3). First row containing minimum point, last row
            containing maximum point.

            Returned tensor will be on CPU independent of dataset device.
        """
        if self.gt_c2ws is None:
            return None

        bounds_file_path = self.scene_dir_path / "scene_bounds.txt"
        if bounds_file_path.is_file():
            return torch.from_numpy(np.loadtxt(bounds_file_path)).float()

        aabb_min = None
        aabb_max = None

        data_loader = DataLoader(self, num_workers=16, pin_memory=True)
        for i, item in enumerate(data_loader):
            depth_image = item["rgbd"][0, :, :, 3]
            points_c = self.camera.depth_to_pointcloud(depth_image)
            gt_c2w = self.gt_c2ws[i]
            if gt_c2w.isnan().any():
                continue
            points_w = utils.transform_points(points_c, gt_c2w)
            if aabb_min is None:
                aabb_min = points_w.min(dim=0)[0]
                aabb_max = points_w.max(dim=0)[0]
            else:
                aabb_min = torch.min(aabb_min, points_w.min(dim=0)[0])
                aabb_max = torch.max(aabb_max, points_w.max(dim=0)[0])

        bounds = torch.stack((aabb_min, aabb_max)).cpu()

        np.savetxt(bounds_file_path, bounds.numpy())

        return bounds

    @property
    def custom_scene_bounds(self) -> Optional[torch.Tensor]:
        """Return custom scene bounds if specified."""
        return None


def _pose_vector_to_4x4(pose_vector: npt.ArrayLike) -> torch.Tensor:
    """Convert an x y z qx qy qz qw array to a 4x4 transformation matrix."""
    pose_vector = np.array(pose_vector)
    matrix = torch.eye(4)
    matrix[:3, :3] = torch.as_tensor(
        transform.Rotation.from_quat(pose_vector[3:]).as_matrix(), dtype=torch.float
    )
    matrix[:3, 3] = torch.as_tensor(pose_vector[:3], dtype=torch.float)

    # OpenCV to OpenGL camera convention
    # FIXME this should be a parameter (see FIXME above)
    ogl2ocv = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return matrix @ ogl2ocv


def _pg_dicts_to_essential_graph(pg_dicts: List[dict], weight_threshold: float) -> dict:
    """Convert list of keyframe dictionaries to essential graph."""
    graph = {}
    # first add all keyframes (vertices)
    for pg_dict in pg_dicts:
        frame_id = pg_dict["KF"]
        graph[frame_id] = set()

    # add edges
    for pg_dict in pg_dicts:
        frame_id = pg_dict["KF"]
        edges = set()
        edges.update(pg_dict["LC"])
        cov_edges = [
            to for to, wgt in zip(pg_dict["CV"], pg_dict["WGT"]) if wgt > weight_threshold
        ]
        edges.update(cov_edges)
        graph[frame_id] = edges & graph.keys()  # skip edges to invalid vertices
        # FIXME this shouldn't be necessary
    return graph
