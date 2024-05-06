"""This module provides various utility functions."""
import copy
import inspect
import time
from pydoc import locate
from typing import Any, Callable, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch
from PIL import Image
from pytorch3d import transforms as p3dt
from tqdm import tqdm


def rr_init(
    application_id: str,
    rrd_path: Optional[str] = None,
    connect_addr: Optional[str] = None,
) -> None:
    """Spawn, connect, or save Rerun.

    If both rrd_path and connect_addr are None, spawn will be used.

    Args:
        application_id: Passed to rr.init.
        rrd_path: Path to save rrd file to.
        connect_addr: ip_address:port to connect to.
    """
    assert not (rrd_path is not None and connect_addr is not None)
    rr.init(application_id)
    if rrd_path is not None:
        rr.save(rrd_path)
    elif connect_addr is not None:
        rr.connect(connect_addr)
    else:
        rr.spawn()


def rr_up_axis(
    up_axis: Literal["x", "y", "z", "-x", "-y", "-z"],
) -> rr.ViewCoordinates:
    """Convert up_axis in SLAMDataset format to rerun format."""
    if up_axis == "x":
        return rr.ViewCoordinates.RIGHT_HAND_X_UP
    elif up_axis == "y":
        return rr.ViewCoordinates.RIGHT_HAND_Y_UP
    elif up_axis == "z":
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP
    elif up_axis == "-x":
        return rr.ViewCoordinates.RIGHT_HAND_X_DOWN
    elif up_axis == "-y":
        return rr.ViewCoordinates.RIGHT_HAND_Y_DOWN
    elif up_axis == "-z":
        return rr.ViewCoordinates.RIGHT_HAND_Z_DOWN
    else:
        ValueError(f"Invalid {up_axis=}")


def benchmark(f: Callable) -> Callable:
    """Decorator to print time it took to execute a function.

    Can be globally enabled / disabled by setting benchmark.enabled to True / False.
    """
    if not hasattr(benchmark, "enabled"):
        benchmark.enabled = True

    if not hasattr(benchmark, "indent"):
        benchmark.indent = 0

    def wrapper(*args, **kwargs):
        if benchmark.enabled:
            benchmark.indent += 1
            torch.cuda.synchronize()
            t1 = time.time()
            result = f(*args, **kwargs)
            torch.cuda.synchronize()
            t2 = time.time()
            benchmark.indent -= 1
            print(f"{'  ' * benchmark.indent}{f.__name__} finished in {t2-t1:.4f}")
        else:
            result = f(*args, **kwargs)
        return result

    return wrapper


def create_layer(
    in_features: int,
    out_features: int,
    activation: Optional[torch.nn.Module] = None,
    bias: bool = True,
) -> torch.nn.Module:
    """Sequence of a linear layer with optional activation.

    Args:
        in_features: number of input connections to the layer
        out_features: number of nodes in the layer
        activation: optional activation after linear layer
        bias: Whether the linear layer includes a bias term.

    Returns:
        sequential_layer: created sequential layer
    """
    layers = [torch.nn.Linear(in_features, out_features, bias=bias)]
    if activation is not None:
        layers.append(activation)
    sequential_layer = torch.nn.Sequential(*layers)

    return sequential_layer


def str_to_object(name: str) -> Any:
    """Try to find object with a given name.

    First scope of calling function is checked for the name, then current environment
    (in which case name has to be a fully qualified name). In the second case, the
    object is imported if found.

    Args:
        name: Name of the object to resolve.

    Returns:
        The object which the provided name refers to. None if no object was found.
    """
    # check callers local variables
    caller_locals = inspect.currentframe().f_back.f_locals
    if name in caller_locals:
        return caller_locals[name]

    # check callers global variables (i.e., imported modules etc.)
    caller_globals = inspect.currentframe().f_back.f_globals
    if name in caller_globals:
        return caller_globals[name]

    # check environment
    return locate(name)


def visualize_pointset(pointset: torch.Tensor, max_points: int = 1000) -> None:
    """Visualize pointset as 3D scatter plot.

    Args:
        pointset:
            The pointset to visualize. Either shape (N,3), xyz, or shape (N,6), xyzrgb.
        max_points:
            Maximum number of points.
            If N>max_points only a random subset will be shown.
    """
    pointset_np = pointset.cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    if len(pointset_np) > max_points:
        indices = np.random.choice(len(pointset_np), replace=False, size=max_points)
        pointset_np = pointset_np[indices]

    if pointset_np.shape[1] >= 4:
        colors = pointset_np[:, 3:].squeeze()
    else:
        colors = None

    sc = ax.scatter(pointset_np[:, 0], pointset_np[:, 1], pointset_np[:, 2], c=colors)
    fig.colorbar(sc)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect(pointset_np[:, :3].max(axis=0) - pointset_np[:, :3].min(axis=0))
    plt.show()


@benchmark
def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Convert a tensor of inhomogeneous coordinates to homogeneous coordinates.

    Adds 1 to the end of last dimension.

    Args:
        x: the tensor containing inhomogeneous coordinates, shape (...,M)

    Returns:
        the tensor containing homogeneous coordinates, shape (...,M+1)
    """
    return torch.nn.functional.pad(x, (0, 1), value=1)


@benchmark
def to_inhomogeneous(x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Convert a tensor of inhomogeneous coordinates to homogeneous coordinates.

    Removes the last element of the trailing dimension.

    Args:
        x: the tensor containing homogeneous coordinates, shape (...,M)
        normalize: whether to normalize the tensor before removing last element

    Returns:
        the tensor containing inhomogeneous coordinates, shape (...,M+1)
    """
    if normalize:
        x /= x[..., -1, None]
    return x[..., :-1]


def save_image(img: torch.Tensor, file_path: str) -> None:
    """Save torch.Tensor as an image.

    Args:
        img: Image tensor to save. Shape (H, W, 3).
        out_path: File path of the image.
    """
    np_img = (img.numpy(force=True) * 255).astype(np.uint8)
    comparison_img = Image.fromarray(np_img)
    comparison_img.save(file_path)


def batched_evaluation(
    model: Callable[..., Tuple[torch.Tensor]],
    inputs: torch.Tensor,
    block_size: int,
    progressbar: bool = False,
) -> Tuple[torch.Tensor]:
    """Evaluate large batch of points in blocks to reduce memory usage.

    Args:
        model: the model to evaluate
        inputs: input tensor of shape (N, ...), will be split along first dimension
        block_size: number of samples evaluated in parallel
        progressbar: Whether to show progressbar.

    Returns:
        output of model when inputs would have been passed
    """
    outs = []
    iterator = range(0, inputs.shape[0], block_size)
    if progressbar:
        iterator = tqdm(iterator)
    for start in iterator:
        end = min(start + block_size, inputs.shape[0])
        outs.append(model(inputs[start:end]))

    if isinstance(outs[0], tuple):
        outs = tuple(
            torch.cat(x) if isinstance(x[0], torch.Tensor) else x for x in zip(*outs)
        )
    elif isinstance(outs[0], torch.Tensor):
        outs = torch.cat(outs)
    return outs


def prepare_dict_for_wandb(x: dict) -> dict:
    """Recursively converts values derived from float and int to base type.

    Returns:
    """
    x = copy.deepcopy(x)
    for key in x:
        if isinstance(x[key], float):
            x[key] = float(x[key])
        elif isinstance(x[key], int):
            x[key] = int(x[key])
        elif isinstance(x[key], dict):
            x[key] = prepare_dict_for_wandb(x[key])
    return x


def transform_quaternions(quaternions: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    transform_matrices = transforms[..., :3, :3]  # rotation matrix from 4x4 transform
    transform_quats = p3dt.matrix_to_quaternion(transform_matrices)
    return p3dt.quaternion_multiply(transform_quats, quaternions)


def transform_points(
    points: torch.Tensor, transforms: torch.Tensor, inv: bool = False
) -> torch.Tensor:
    if inv:
        return torch.einsum(
            "...kd,...k -> ...d", transforms[..., :3, :3], points - transforms[..., :3, 3]
        )
    return (
        torch.einsum("...dk,...k -> ...d", transforms[..., :3, :3], points)
        + transforms[..., :3, 3]
    )
