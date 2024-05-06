"""Dataset class for Kintinuous dataset."""
import os
import pathlib
from typing import Literal, Optional

import numpy as np
import PIL.Image
import torch
import yoco
from tqdm import tqdm

from neural_graph_mapping import camera, slam_dataset


class KintinuousDataset(slam_dataset.SLAMDataset):
    """Kintinuous dataset class.

    Can be downloaded at
    http://www.cs.nuim.ie/research/vision/data/loop.klg

    Can be extracted with https://github.com/wonhkim/klg2png/
    Make sure to increase precision to 10 at line 75 in
    https://github.com/wonhkim/klg2png/blob/master/src/LogReader.cpp

    Since the dataset has no ground-truth camera poses the c2ws will all be identity
    matrices and should not be used.

    Expected directory format:
        {root_dir}/{scene}/color/
        {root_dir}/{scene}/depth/
        {root_dir}/{scene}/{pg_file_all}
        {root_dir}/{scene}/{pg_file_kf}

    Attributes:
        camera (camera.Camera):
            The camera information for the images.
    """

    class Config(slam_dataset.SLAMDataset.Config, total=False):
        """Configuration dictionary for KintinuousDataset.

        Attributes:
            root_dir: See class docstring.
            scene: See class docstring.
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

        root_dir: str
        scene: str
        camera: dict
        fps: int
        frame_skip: int
        scale: float
        prefetch: bool

    default_config: Config = {
        "image_dir": "images",
        "scene": "loop",
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
                with default_dict. See KintinuousDataset.Config for supported keys.
        """
        self.config = yoco.load_config(config, current_dict=KintinuousDataset.default_config)
        super().__init__(self.config)

    def __str__(self) -> str:
        """Return identifier name of dataset and scene."""
        return "Kintinuous_" + self._scene

    def _parse_config(self) -> None:
        """Parse configuration dictionary into member variables."""
        super()._parse_config()

        self._root_dir_path = pathlib.Path(self.config["root_dir"])
        self._scene = self.config["scene"]
        self._image_dir_path = self._root_dir_path / self._scene / "color"
        self._depth_dir_path = self._root_dir_path / self._scene / "depth"
        self._num_images = len(os.listdir(self._image_dir_path))

        self._scale = self.config["scale"]
        self._prefetch = self.config["prefetch"]
        self._fps = self.config["fps"]
        self._frame_skip = self.config["frame_skip"]

        self.camera = camera.Camera(**self.config["camera"])
        self._c2ws = torch.eye(4, device=self.device).expand(self._get_num_images(), 4, 4)

    @property
    def num_images(self) -> int:
        """Return number of images in this dataset."""
        return self._num_images

    @property
    def scene_dir_path(self) -> pathlib.Path:
        """Return directory for the current scene."""
        return self._root_dir_path / self._scene

    @property
    def has_gt_mesh(self) -> bool:
        """Return False since there is no ground-truth mesh for Kintinuous dataset."""
        return False

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
            self._get_len = self._get_num_rays
        elif self._mode == "sequence":
            self._load_dataset = self._load_sequence_dataset
            self._get_item = self._get_sequence_item
            self._get_len = self._get_num_images
        else:
            raise ValueError("Dataset mode must be ray or sequence.")

        if self._prefetch:
            self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset into memory.

        See KintinuousDataset._load_ray_dataset and
        KintinuousDataset._load_sequence_dataset for details.
        """
        raise NotImplementedError(
            "_load_dataset should point to _load_ray_dataset or _load_sequence_dataset"
        )

    def _load_ray_dataset(self) -> None:
        """Load ray dataset into memory."""
        image_filenames = sorted(os.listdir(self._image_dir_path), key=self._get_sort_float)
        depth_filenames = sorted(os.listdir(self._depth_dir_path), key=self._get_sort_float)
        image_filenames = image_filenames[:: self._frame_skip + 1]
        depth_filenames = depth_filenames[:: self._frame_skip + 1]

        rgb_sample = self._load_color(self._image_dir_path / image_filenames[0])
        w = rgb_sample.shape[1]
        h = rgb_sample.shape[0]
        n = len(image_filenames)

        self._ijs = torch.cartesian_prod(torch.arange(h), torch.arange(w))
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
        ):
            image_file_path = self._image_dir_path / image_filename
            depth_file_path = self._depth_dir_path / depth_filename
            self._rgbds[i] = self._load_rgbd(image_file_path, depth_file_path)
            self._frame_ids[i] = i

        # repeat ijs, memory inefficient, but good enough here
        self._ijs = self._ijs.repeat(n, 1)

        # flatten to rays
        self._rgbds = self._rgbds.reshape(-1, 4)
        self._frame_ids = self._frame_ids.reshape(-1)

        self.print_memory_usage()

    def _load_sequence_dataset(self) -> None:
        """Load sequence dataset into memory."""
        image_filenames = sorted(os.listdir(self._image_dir_path), key=self._get_sort_float)
        depth_filenames = sorted(os.listdir(self._depth_dir_path), key=self._get_sort_float)
        image_filenames = image_filenames[:: self._frame_skip + 1]
        depth_filenames = depth_filenames[:: self._frame_skip + 1]

        rgb_sample = self._load_color(self._image_dir_path / image_filenames[0])
        width = rgb_sample.shape[1]
        height = rgb_sample.shape[0]

        self._times = []
        self._rgbds = torch.empty(len(image_filenames), height, width, 4, device=self.device)

        seconds_per_frame = 1 / self._fps

        for i, (image_filename, depth_filename) in tqdm(
            enumerate(zip(image_filenames, depth_filenames)),
            total=len(image_filenames),
            desc="Loading dataset",
        ):
            self._times.append(i * seconds_per_frame)
            image_file_path = self._image_dir_path / image_filename
            depth_file_path = self._depth_dir_path / depth_filename
            self._rgbds[i] = self._load_rgbd(image_file_path, depth_file_path)

        self._times = torch.Tensor(self._times).to(self.device)

        self.print_memory_usage()

    def __getitem__(self, index: int) -> dict:
        """Return a sample of the dataset.

        See KintinuousDataset._get_ray_item and KintinuousDataset._get_sequence_item for
        details.
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
                "c2w": self._c2ws[self._frame_ids[index]],
            }
        else:
            return self._load_ray_item(index)

    def _load_ray_item(self, index: int) -> dict:
        raise NotImplementedError("ray mode only implemented in prefetch mode right now")

    def _get_sequence_item(self, index: int) -> dict:
        if self._prefetch:
            return {
                "time": self._times[index],
                "rgbd": self._rgbds[index],
                "c2w": self._c2ws[index],
            }
        else:
            return self._load_sequence_item(index)

    def _load_sequence_item(self, index: int) -> dict:
        image_filenames = sorted(os.listdir(self._image_dir_path), key=self._get_sort_float)
        depth_filenames = sorted(os.listdir(self._depth_dir_path), key=self._get_sort_float)
        image_filenames = image_filenames[:: self._frame_skip + 1]
        depth_filenames = depth_filenames[:: self._frame_skip + 1]

        seconds_per_frame = 1 / self._fps

        image_file_path = self._image_dir_path / image_filenames[index]
        depth_file_path = self._depth_dir_path / depth_filenames[index]
        rgbd = self._load_rgbd(image_file_path, depth_file_path)
        rgbd = rgbd.to(self.device)

        time = torch.tensor(index * seconds_per_frame, device=self.device)

        return {"time": time, "rgbd": rgbd, "c2w": self._c2ws[index]}

    def _load_rgbd(self, color_path: pathlib.Path, depth_path: pathlib.Path) -> torch.Tensor:
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

    def _load_color(self, color_path: pathlib.Path) -> torch.Tensor:
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

    def _load_depth(self, depth_path: pathlib.Path) -> torch.Tensor:
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

    def _get_sort_float(self, file_name: str) -> int:
        """Return last integer found in path."""
        return float(os.path.splitext(file_name)[0])

    def __len__(self) -> int:
        """Return number of samples in dataset.

        For mode ray this is the total number of pixels.
        For mode sequence iit is the total number of images.
        """
        return self._get_len()

    def _get_len(self) -> int:
        raise NotImplementedError("_get_len should point to _get_num_rays or _get_num_images")

    def _get_num_images(self) -> int:
        return self._num_images

    def _get_num_rays(self) -> int:
        return self._num_images * self.camera.width * self.camera.height

    def print_memory_usage(self) -> None:
        """Print memory usage of dataset."""
        if self._mode == "ray":
            mem = (
                self._ijs.element_size() * self._ijs.numel()
                + self._frame_ids.element_size() * self._frame_ids.numel()
                + self._rgbds.element_size() * self._rgbds.numel()
                + self._c2ws.element_size() * self._c2ws.numel()
            )
        elif self._mode == "sequence":
            mem = (
                self._rgbds.element_size() * self._rgbds.numel()
                + self._c2ws.element_size() * self._c2ws.numel()
                + self._times.element_size() * self._times.numel()
            )
        else:
            raise ValueError("KintinuousDataset mode must be ray or sequence.")

        print("Dataset memory usage: ", mem / 1e9, "GB")

    @property
    def scene_bounds(self) -> Optional[np.ndarray]:
        """Return tight scene bounds according to ground-truth trajectory.

        Returns:
            None if scene bounds are not known for the scene.
            Otherwise array of shape (2, 3). First row containing minimum point, last row
            containing maximum point.
        """
        return None
