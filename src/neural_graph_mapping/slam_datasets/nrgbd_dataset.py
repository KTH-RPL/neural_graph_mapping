"""Dataset class for NRGBD dataset."""
import os
import pathlib
import re
from typing import Literal, Optional

import numpy as np
import open3d as o3d
import PIL.Image
import torch
import yoco
from tqdm import tqdm

from neural_graph_mapping import camera, slam_dataset


class NRGBDDataset(slam_dataset.SLAMDataset):
    """Neural RGB-D dataset class.

    Can be downloaded at
    https://github.com/dazinovic/neural-rgbd-surface-reconstruction.

    Expected directory format:
        {root_dir}/{scene}/{image_dir}/
        {root_dir}/{scene}/{depth_dir}/
        {root_dir}/{scene}/{poses_file}
        {root_dir}/{scene}/{pg_file_all}
        {root_dir}/{scene}/{pg_file_kf}
        {root_dir}/{scene}/gt_mesh.ply

    Attributes:
        camera (camera.Camera):
            The camera information for the images.
    """

    class Config(slam_dataset.SLAMDataset.Config, total=False):
        """Configuration dictionary for NRGBDDataset.

        Attributes:
            image_dir: See class docstring.
            depth_dir: See class docstring.
            poses_file: See class docstring.
            mesh_file: See class docstring.
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

        image_dir: str
        depth_dir: str
        poses_file: str
        mesh_file: str
        camera: dict
        fps: int
        frame_skip: int
        scale: float
        prefetch: bool

    default_config: Config = {
        "image_dir": "images",
        "depth_dir": "depth_filtered",
        "poses_file": "poses.txt",
        "mesh_file": None,
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
                with default_dict. See NRGBDDataset.Config for supported keys.
        """
        self.config = yoco.load_config(config, current_dict=NRGBDDataset.default_config)
        super().__init__(self.config)

        self._image_dir_path = self.root_dir_path / self.scene / self._image_dir_name
        self._depth_dir_path = self.root_dir_path / self.scene / self._depth_dir_name
        self._poses_file_path = self.root_dir_path / self.scene / self._poses_file_name

        self._num_images = len(sorted(os.listdir(self._image_dir_path), key=self._get_id))
        self.gt_c2ws = self._load_gt_c2ws()

    def _parse_config(self) -> None:
        """Parse configuration dictionary into member variables."""
        super()._parse_config()
        self._image_dir_name = self.config["image_dir"]
        self._depth_dir_name = self.config["depth_dir"]
        self._poses_file_name = self.config["poses_file"]
        self._scale = self.config["scale"]
        self._prefetch = self.config["prefetch"]
        self._depth_bias = self.config["depth_bias"]

        self._fps = self.config["fps"]
        self._frame_skip = self.config["frame_skip"]

        # FIXME should not be configurable
        self.camera = camera.Camera(**self.config["camera"])

    @staticmethod
    def get_available_scenes(root_dir: str) -> list[str]:
        """Return available scenes at the given root directory."""
        root_dir_path = pathlib.Path(root_dir)
        scene_dir_paths = [p for p in root_dir_path.iterdir() if p.is_dir()]
        valid_scene_dir_paths = [p for p in scene_dir_paths if (p / "gt_mesh.ply").exists()]
        scene_names = [p.name for p in valid_scene_dir_paths]
        return scene_names

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
        mesh_file_path: pathlib.Path = self.scene_dir_path / "gt_mesh.ply"
        return mesh_file_path

    def load_gt_mesh(self) -> slam_dataset.Mesh:
        """Return ground-truth mesh or None if not available for this dataset.

        Should only be called and / or implemented when has_gt_mesh is True.
        """
        o3d_mesh = o3d.io.read_triangle_mesh(str(self.gt_mesh_path))
        return slam_dataset.Mesh(o3d_mesh)

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

        See NRGBDDataset._load_ray_dataset and NRGBDDataset._load_sequence_dataset for
        details.
        """
        raise NotImplementedError(
            "_load_dataset should point to _load_ray_dataset or _load_sequence_dataset"
        )

    def _load_gt_c2ws(self) -> torch.Tensor:
        self._ijs = torch.cartesian_prod(
            torch.arange(self.camera.height), torch.arange(self.camera.width)
        )
        gt_c2ws = torch.from_numpy(
            np.loadtxt(self._poses_file_path).reshape(-1, 4, 4)[:: self._frame_skip + 1]
        ).float()  # N, 4, 4
        gt_c2ws[:, :3, 3] *= self._scale
        return gt_c2ws.to(self.device)

    def _load_ray_dataset(self) -> None:
        """Load ray dataset into memory."""
        image_filenames = sorted(os.listdir(self._image_dir_path), key=self._get_id)
        depth_filenames = sorted(os.listdir(self._depth_dir_path), key=self._get_id)
        image_filenames = image_filenames[:: self._frame_skip + 1]
        depth_filenames = depth_filenames[:: self._frame_skip + 1]

        rgb_sample = self._load_color(self._image_dir_path / image_filenames[0])
        w = rgb_sample.shape[1]
        h = rgb_sample.shape[0]
        n = len(image_filenames)

        self._rgbds = torch.empty(
            len(image_filenames), h, w, 4, device=self.device
        )  # N, H, W, 4
        self._frame_ids = torch.empty(
            len(image_filenames), h, w, dtype=torch.long, device=self.device
        )

        for i, (image_filename, depth_filename) in tqdm(
            enumerate(zip(image_filenames, depth_filenames)),
            total=len(image_filenames),
            desc="Loading dataset",
            dynamic_ncols=True,
        ):
            image_file = os.path.join(self._image_dir_path, image_filename)
            depth_file = os.path.join(self._depth_dir_path, depth_filename)
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
        image_filenames = sorted(os.listdir(self._image_dir_path), key=self._get_id)
        depth_filenames = sorted(os.listdir(self._depth_dir_path), key=self._get_id)
        image_filenames = image_filenames[:: self._frame_skip + 1]
        depth_filenames = depth_filenames[:: self._frame_skip + 1]

        rgb_sample = self._load_color(os.path.join(self._image_dir_path, image_filenames[0]))
        width = rgb_sample.shape[1]
        height = rgb_sample.shape[0]

        self._times = []
        self._rgbds = torch.empty(len(image_filenames), height, width, 4, device=self.device)

        seconds_per_frame = 1 / self._fps

        for i, (image_filename, depth_filename) in tqdm(
            enumerate(zip(image_filenames, depth_filenames)),
            total=len(image_filenames),
            desc="Loading dataset",
            dynamic_ncols=True,
        ):
            self._times.append(i * seconds_per_frame)
            image_file = os.path.join(self._image_dir_path, image_filename)
            depth_file = os.path.join(self._depth_dir_path, depth_filename)
            self._rgbds[i] = self._load_rgbd(image_file, depth_file)

        self._times = torch.Tensor(self._times).to(self.device)

        self.print_memory_usage()

    def __getitem__(self, index: int) -> dict:
        """Return a sample of the dataset.

        See NRGBDDataset._get_ray_item and NRGBDDataset._get_sequence_item for details.
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
        image_filenames = sorted(os.listdir(self._image_dir_path), key=self._get_id)
        depth_filenames = sorted(os.listdir(self._depth_dir_path), key=self._get_id)
        image_filenames = image_filenames[:: self._frame_skip + 1]
        depth_filenames = depth_filenames[:: self._frame_skip + 1]

        c2ws = torch.from_numpy(
            np.loadtxt(self._poses_file_path).reshape(-1, 4, 4)[:: self._frame_skip + 1]
        ).float()
        c2ws[:, :3, 3] *= self._scale
        c2ws = c2ws.to(self.device)

        seconds_per_frame = 1 / self._fps

        image_file_path = os.path.join(self._image_dir_path, image_filenames[index])
        depth_file_path = os.path.join(self._depth_dir_path, depth_filenames[index])
        rgbd = self._load_rgbd(image_file_path, depth_file_path)
        rgbd = rgbd.to(self.device)

        time = torch.tensor(index * seconds_per_frame, device=self.device)

        return {"time": time, "rgbd": rgbd, "c2w": c2ws[index]}

    def _load_rgbd(self, color_path: str, depth_path: str) -> torch.Tensor:
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

    def _load_color(self, color_path: str) -> torch.Tensor:
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

    def _load_depth(self, depth_path: str) -> torch.Tensor:
        """Load depth image from filepath.

        Args:
            depth_path: Filepath of depth image.

        Returns:
            Tensor containing depth image. Depth. Meters. Shape (H,W).
        """
        depth = torch.from_numpy(
            np.asarray(PIL.Image.open(depth_path), dtype=np.float32) * 0.001 * self._scale
        ).to(self.device)
        if self._depth_dir_name == "depth_filtered":
            # noise for depth_filtered is biased, here we attempt to reduce the bias
            # this is a fit based on staircase scene (polynomial a * x^2 + b *x on the error)
            depth = 0.00123631 * depth**2 + (1 + 0.00073707) * depth
        return depth

    def _get_id(self, path: str) -> int:
        """Return last integer found in path."""
        return int(re.findall(r"\d+", path)[-1])

    def __len__(self) -> int:
        """Return number of samples in dataset.

        For mode ray this is the total number of pixels.
        For mode sequence
        """
        return self.num_images

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
            raise ValueError("NRGBDDataset mode must be ray or sequence.")

        print("Dataset memory usage: ", mem / 1e9, "GB")

    @property
    def custom_scene_bounds(self) -> Optional[torch.Tensor]:
        """Return scene bounds as used by CO-SLAM and NICE-SLAM.

        Returns:
            None if scene bounds are not known for the scene.
            Otherwise array of shape (2, 3). First row containing minimum point, last row
            containing maximum point.
        """
        if self.scene == "breakfast_room":
            return torch.tensor([[-2.4, 2.0], [-0.6, 2.9], [-1.8, 3.1]]).T
        elif self.scene == "complete_kitchen":
            return torch.tensor([[-5.7, 3.8], [-0.2, 3.3], [-6.6, 3.6]]).T
        elif self.scene == "green_room":
            return torch.tensor([[-2.6, 5.6], [-0.3, 3.0], [0.2, 5.1]]).T
        elif self.scene == "grey_white_room":
            return torch.tensor([[-0.7, 5.4], [-0.2, 3.1], [-3.9, 0.8]]).T
        elif self.scene == "morning_apartment":
            return torch.tensor([[-1.5, 2.2], [-0.3, 2.2], [-2.3, 1.9]]).T
        elif self.scene == "thin_geometry":
            return torch.tensor([[-2.5, 1.1], [-0.3, 1.1], [0.1, 3.9]]).T
        elif self.scene == "whiteroom":
            return torch.tensor([[-2.6, 3.2], [-0.1, 3.6], [0.5, 8.3]]).T
        else:
            return None
