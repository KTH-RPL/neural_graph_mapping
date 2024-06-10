"""Dataset class for a scene.

Currently only dataset format from Neural RGB-D Reconstruction supported.
See: https://github.com/dazinovic/neural-rgbd-surface-reconstruction
"""

import os
import pathlib
import re
from typing import Literal

import numpy as np
import open3d as o3d
import PIL.Image
import torch
import tqdm
import yoco

from neural_graph_mapping import camera, slam_dataset

_ocv2ogl = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class ScanNetDataset(slam_dataset.SLAMDataset):
    """ScanNet RGB-D dataset class.

    Can be downloaded via http://www.scan-net.org/
    Use https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py
    to extract .sens files to the expected directory structure.

    Expected directory format:
        {root_dir}/{scene}/color/
        {root_dir}/{scene}/depth/
        {root_dir}/{scene}/intrinsic/
        {root_dir}/{scene}/pose/
        {root_dir}/{scene}/{scene}_vh_clean.ply
        {root_dir}/{scene}/{pg_file_all}
        {root_dir}/{scene}/{pg_file_kf}

    The script will automatically perform preprocessing of the color frames which are
    stored at
        {root_dir}/{scene}/aligned_color_to_depth/

    Attributes:
        camera (camera.Camera):
            The camera information for the images.
    """

    class Config(slam_dataset.SLAMDataset.Config, total=False):
        """Configuration dictionary for ScanNetDataset.

        Attributes:
            camera:
                Camera parameters of the dataset. Will be passed as kwargs to
                constructor of camera.Camera.
            fps:
                Frames per second used to calculate timestamp in sequence mode.
            frame_skip:
                Number of frames skipped between consecutive frames.
                In ray mode this can be used to reduce memory usage of dataset, and
                shouldn't decrease performance when images are very close together.
                In sequence mode fps and frame_skip together define speed of camera
                motion.  For example, both, changing fps 30->60 and frame_skip 0->1
                double the sequence speed, but only in the second case the information
                of every other frame is lost.
                If 0 all frames are used, 1 every other, 2 every third, ...
            scale:
                Scaling factor for depth and poses.
            prefetch:
                Whether to prefetch the whole dataset (i.e., store in memory).
        """

        fps: int
        frame_skip: int
        scale: float
        prefetch: bool

    default_config: Config = {
        "fps": 30,
        "frame_skip": 0,
        "scale": 1,
        "prefetch": False,
    }

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the dataset.

        Args:
            config:
                Configuration dictionary of dataset. Provided dictionary will be merged
                with default_dict. See ScanNetDataset.Config for supported keys.
        """
        self.config = yoco.load_config(config, current_dict=ScanNetDataset.default_config)
        super().__init__(self.config)

        self._color_dir_path = self.scene_dir_path / "aligned_color_to_depth"
        self._depth_dir_path = self.scene_dir_path / "depth"
        if not self._color_dir_path.exists():
            self._preprocess_color()
        self._num_images = len(list(self._color_dir_path.iterdir()))
        self.camera = self._load_camera()
        self._image_filenames = sorted(self._color_dir_path.iterdir(), key=self._get_id)
        self._depth_filenames = sorted(self._depth_dir_path.iterdir(), key=self._get_id)
        self._image_filenames = self._image_filenames[:: self._frame_skip + 1]
        self._depth_filenames = self._depth_filenames[:: self._frame_skip + 1]
        self._image_filenames = [ifn.name for ifn in self._image_filenames]
        self._depth_filenames = [dfn.name for dfn in self._depth_filenames]

        self.gt_c2ws = self._load_gt_c2ws()

    def __str__(self) -> str:
        """Return identifier name of dataset and scene."""
        return "ScanNet_" + self.scene

    @property
    def num_images(self) -> int:
        """Return number of images in this dataset."""
        return self._num_images

    @property
    def scene_dir_path(self) -> pathlib.Path:
        """Return directory for the current scene."""
        return self.root_dir_path / self.scene

    @property
    def has_gt_mesh(self) -> bool:
        """Return whether dataset has a ground-truth mesh and it exists."""
        return self.gt_mesh_path.is_file()

    @property
    def gt_mesh_path(self) -> pathlib.Path:
        """Return path of ground-truth mesh or None if not available for this dataset."""
        mesh_file_name = f"{self.scene}_vh_clean.ply"
        mesh_file_path: pathlib.Path = self.scene_dir_path / mesh_file_name
        return mesh_file_path

    def load_gt_mesh(self) -> slam_dataset.Mesh:
        """Return ground-truth mesh or None if not available for this dataset."""
        o3d_mesh = o3d.io.read_triangle_mesh(str(self.gt_mesh_path))
        return slam_dataset.Mesh(o3d_mesh)

    def _parse_config(self) -> None:
        """Parse configuration dictionary into member variables."""
        super()._parse_config()
        self._scale = self.config["scale"]
        self._prefetch = self.config["prefetch"]
        self._fps = self.config["fps"]
        self._frame_skip = self.config["frame_skip"]

    @staticmethod
    def get_available_scenes(root_dir: str) -> list[str]:
        """Return available scenes at the given root directory."""
        root_dir_path = pathlib.Path(root_dir)
        scene_dir_paths = [p for p in root_dir_path.iterdir() if p.is_dir()]
        valid_scene_dir_paths = [
            p
            for p in scene_dir_paths
            if (p / "color").exists()
            and (p / "depth").exists()
            and (p / "pose").exists()
            and (p / "intrinsic").exists()
        ]
        scene_names = [p.name for p in valid_scene_dir_paths]
        return scene_names

    def _load_gt_c2ws(self) -> torch.Tensor:
        """Load and returns matrix containing ground-truth c2w matrices."""
        c2ws = []
        for i in tqdm.tqdm(range(0, len(self), self._frame_skip + 1), "Load gt trajectory"):
            c2ws.append(self._load_gt_c2w(i))
        return torch.stack(c2ws).to(self.device)

    def _load_gt_c2w(self, frame_id: int) -> torch.Tensor:
        """Load and returns matrix containing ground-truth c2w matrix for one frame."""
        c2w_file_path = self.scene_dir_path / "pose" / f"{frame_id}.txt"
        c2w = torch.from_numpy(np.loadtxt(c2w_file_path)).float()  # N, 4, 4
        c2w *= self._scale
        c2w @= _ocv2ogl
        c2w = c2w.to(self.device)
        return c2w

    def _get_image_size(self) -> tuple:
        depth_path = self.scene_dir_path / "depth" / "0.png"
        depth_image = PIL.Image.open(depth_path)
        return depth_image.size

    def _load_camera(self) -> camera.Camera:
        intrinsic_path = self.scene_dir_path / "intrinsic" / "intrinsic_depth.txt"
        intrinsic_matrix = np.loadtxt(intrinsic_path)
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        width, height = self._get_image_size()
        #  (1.0 maybe because matlab was used for calibration ??)
        return camera.Camera(width, height, fx, fy, cx, cy, pixel_center=1.0)

    def _preprocess_color(self) -> None:
        self._color_dir_path.mkdir()
        size = self._get_image_size()
        raw_dir_path = self.scene_dir_path / "color"
        print("Preprocessing color images.")
        print(f"Saving to {self._color_dir_path}...")
        for raw_file_path in tqdm.tqdm(list(raw_dir_path.iterdir())):
            raw = PIL.Image.open(raw_file_path)
            processed = raw.resize(size, resample=PIL.Image.Resampling.LANCZOS)
            processed_color_path = self._color_dir_path / raw_file_path.name
            processed.save(processed_color_path)

    def set_mode(self, mode: Literal["ray", "sequence"]) -> None:
        """See slam_dataset.SLAMDataset.set_mode."""
        if mode == self._mode:
            return

        # reset current data
        self._rgbds = self._times = self._ijs = self._frame_ids = None

        self._mode = mode

        if self._mode == "ray":
            self._load_dataset = self._load_ray_dataset
            self._get_item = self._get_ray_item
        elif self._mode == "sequence":
            self._load_dataset = self._load_sequence_dataset
            self._get_item = self._get_sequence_item
        else:
            raise ValueError("Dataset mode must be ray or sequence.")
        if self._prefetch:
            self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset into memory.

        See ScanNetDataset._load_ray_dataset and ScanNetDataset._load_sequence_dataset for
        details.
        """
        raise NotImplementedError(
            "_load_dataset should point to _load_ray_dataset or _load_sequence_dataset"
        )

    def _load_ray_dataset(self) -> None:
        """Load ray dataset into memory."""
        rgb_sample = self._load_color(
            os.path.join(self._color_dir_path, self._image_filenames[0])
        )
        w = rgb_sample.shape[1]
        h = rgb_sample.shape[0]
        n = len(self._image_filenames)

        self._ijs = torch.cartesian_prod(torch.arange(h), torch.arange(w))
        self._rgbds = torch.empty(
            len(self._image_filenames), h, w, 4, device=self.device
        )  # N, H, W, 4
        self._frame_ids = torch.empty(
            len(self._image_filenames), h, w, dtype=torch.long, device=self.device
        )

        for i, (image_filename, depth_filename) in tqdm.tqdm(
            enumerate(zip(self._image_filenames, self._depth_filenames)),
            total=len(self._image_filenames),
            desc="Loading dataset",
            dynamic_ncols=True,
        ):
            image_file = self._color_dir_path / image_filename
            depth_file = self._depth_dir_path, depth_filename
            self._rgbds[i] = self._load_rgbd(image_file, depth_file)
            self._frame_ids[i] = i

        # repeat ijs, memory inefficient, but good enough here
        self._ijs = self._ijs.repeat(n, 1)

        # flatten to rays
        self._rgbds = self._rgbds.reshape(-1, 4)
        self._frame_ids = self._frame_ids.reshape(-1)

        self.print_memory_usage()

    def _load_sequence_dataset(self) -> None:
        """Load sequence dataset into memory."""
        rgb_sample = self._load_color(self._color_dir_path / self._image_filenames[0])
        width = rgb_sample.shape[1]
        height = rgb_sample.shape[0]

        self._times = []
        self._rgbds = torch.empty(
            len(self._image_filenames), height, width, 4, device=self.device
        )

        seconds_per_frame = 1 / self._fps

        for i, (image_filename, depth_filename) in tqdm.tqdm(
            enumerate(zip(self._image_filenames, self._depth_filenames)),
            total=len(self._image_filenames),
            desc="Loading dataset",
        ):
            self._times.append(i * seconds_per_frame)
            image_file = self._color_dir_path / image_filename
            depth_file = self._depth_dir_path / depth_filename
            self._rgbds[i] = self._load_rgbd(image_file, depth_file)

        self._times = torch.Tensor(self._times).to(self.device)

        self.print_memory_usage()

    def __getitem__(self, index: int) -> dict:
        """Return a sample of the dataset.

        See ScanNetDataset._get_ray_item and ScanNetDataset._get_sequence_item for details.
        """
        return self._get_item(index)

    def _get_item(self, index: int) -> dict:
        raise NotImplementedError(
            "_get_item should point to _get_ray_item or _get_sequence_item"
        )

    def _get_ray_item(self, index: int) -> dict:
        if self._prefetch:
            return {
                "ij": self._ijs[index],
                "rgbd": self._rgbds[index],
                "c2w": self.gt_c2ws[self._frame_ids[index]],
            }
        else:
            return self._load_ray_item(index)

    def _load_ray_item(self, index: int) -> dict:
        raise NotImplementedError()

    def _get_sequence_item(self, index: int) -> dict:
        if self._prefetch:
            return {
                "time": self._times[index],
                "rgbd": self._rgbds[index],
                "c2w": self.gt_c2ws[index],
            }
        else:
            return self._load_sequence_item(index)

    def _load_sequence_item(self, index: int) -> dict:
        c2w = self.gt_c2ws[index]

        seconds_per_frame = 1 / self._fps

        image_file_path = self._color_dir_path / self._image_filenames[index]
        depth_file_path = self._depth_dir_path / self._depth_filenames[index]
        rgbd = self._load_rgbd(image_file_path, depth_file_path)
        rgbd = rgbd.to(self.device)

        time = torch.tensor(index * seconds_per_frame, device=self.device)

        return {"time": time, "rgbd": rgbd, "c2w": c2w}

    def _load_rgbd(self, color_path: os.PathLike, depth_path: os.PathLike) -> torch.Tensor:
        """Load RGB-D image from filepath.

        Args:
            color_path: Filepath of color image.
            depth_path: Filepath of depth image.

        Returns:
            Tensor containing RGB-D image. RGBD. Shape (H,W,4).
            First three channels are RGB, 0-1.  Last channel is depth in meters.
        """
        rgb = self._load_color(color_path)
        depth = self._load_depth(depth_path)
        return torch.cat([rgb, depth[..., None]], dim=-1)

    def _load_color(self, color_path: os.PathLike) -> torch.Tensor:
        """Load color image from filepath.

        Args:
            color_path: Filepath of color image.

        Returns:
            Tensor containing color image. RGB. 0-1. Shape (H,W,3).
        """
        color = torch.from_numpy(
            np.asarray(PIL.Image.open(color_path), dtype=np.float32) / 255
        ).to(self.device)
        return color

    def _load_depth(self, depth_path: os.PathLike) -> torch.Tensor:
        """Load depth image from filepath.

        Args:
            depth_path: Filepath of depth image.

        Returns:
            Tensor containing depth image. Depth. Meters. Shape (H,W).
        """
        depth = torch.from_numpy(
            np.asarray(PIL.Image.open(depth_path), dtype=np.float32) * 0.001 * self._scale
        ).to(self.device)
        return depth

    def _get_id(self, path: os.PathLike) -> int:
        """Return last integer found in path."""
        return int(re.findall(r"\d+", str(path))[-1])

    def __len__(self) -> int:
        """Return number of samples in dataset.

        For mode ray this is the total number of pixels.
        For mode sequence
        """
        return self._num_images

    def print_memory_usage(self) -> None:
        """Print memory usage of dataset."""
        if self._mode == "ray":
            mem = (
                self._ijs.element_size() * self._ijs.numel()
                + self._frame_ids.element_size() * self._frame_ids.numel()
                + self._rgbds.element_size() * self._rgbds.numel()
                + self.gt_c2ws.element_size() * self.gt_c2ws.numel()
            )
        elif self._mode == "sequence":
            mem = (
                self._rgbds.element_size() * self._rgbds.numel()
                + self.gt_c2ws.element_size() * self.gt_c2ws.numel()
                + self._times.element_size() * self._times.numel()
            )
        else:
            raise ValueError("ScanNetDataset mode must be ray or sequence.")

        print("Dataset memory usage: ", mem / 1e9, "GB")
