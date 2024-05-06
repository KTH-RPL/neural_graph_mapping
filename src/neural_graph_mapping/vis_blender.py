"""Helper script to launch blender visualization.

NOTE: this only works if current python environment has the same version as blender
"""
import argparse
import os
import subprocess
import sys
import sysconfig
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yoco

if "BLENDERVIS" in os.environ:
    import bpy
    import mathutils

    from neural_graph_mapping import camera
    from neural_graph_mapping.run_mapping import NeuralGraphMap

num_fields = None
scene_representation = None
scene_changed = False
ngs2ble = None
config = None
c2w = None
cam = None
rendering = None
imshow = None
ijs = None
id_to_name = {}
min_iterations = None
field_ids = None
batch_size = 64  # will automatically be adapted based on computer speed


# FIXME if time: try to render in viewport (this should be optional)
#  this would allow using blender to render animations also
#  some possible resources:
#  https://docs.blender.org/api/current/bpy.types.RenderEngine.html
#  https://blender.stackexchange.com/questions/237428/get-pixel-coords-for-vertex-in-viewport


# FIXME allow freely chosen up_axis and set it based on first frame (currently only 90
#  deg rotations are possible, which might not work nicely for SLAM results)


def blender_main() -> None:
    """Entry point for blender script."""
    global num_fields, scene_representation, ngs2ble, config, fig, imshow_obj
    global min_iterations, field_ids

    print("Adding NGS Vis to blender")

    # Remove default cube, and set up view
    bpy.context.preferences.view.show_splash = False
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    config = yoco.load_config_from_args(
        argparse.ArgumentParser(), os.environ["BLENDERVISARGS"].split()
    )

    mesh_resolution = config.get("mesh_resolution", 0.02)
    min_iterations = config.get("min_iterations", 0)

    config["rerun_field_details"] = None

    # Load scene representation with weights
    scene_representation = NeuralGraphMap(config)
    scene_representation.eval()

    # scene_representation._extract_mesh(
    #     "./temp.ply",
    #     resolution=mesh_resolution,
    #     min_iterations=min_iterations,
    #     threshold=0.1,
    # )
    field_ids = scene_representation.get_field_ids(min_iterations)

    # Blender always uses z as up-axis in viewport
    ngs2ble = remap_axis_transform({config["dataset_config"]["up_axis"]: "z"})
    # ngs2ble = np.eye(3, 3)

    # setup scene in blender
    num_fields = scene_representation._global_map_dict["num"]
    add_map_dict_to_blender(scene_representation._global_map_dict)

    # setup rendering callback
    bpy.app.timers.register(execute_rendering)

    # add callback to detect change in scene
    bpy.app.handlers.depsgraph_update_post.append(depsgraph_callback)

    bpy.data.scenes["Scene"].render.resolution_x = 50
    bpy.data.scenes["Scene"].render.resolution_y = 50

    bpy.data.scenes["Scene"].eevee.taa_render_samples = scene_representation._num_samples
    bpy.data.scenes["Scene"].eevee.taa_samples = scene_representation._model._num_knn


    add_default_camera()

    # cv2.namedWindow("rendering", cv2.WINDOW_NORMAL)


def add_default_camera() -> None:
    """Add default camera add eye height, up assumed to be z (blender default)."""
    bpy.ops.object.camera_add()
    bpy.data.cameras["Camera.001"].angle = 70
    bpy.data.objects["Camera"].location.z = 1.8
    bpy.data.objects["Camera"].rotation_euler.x = np.pi / 2
    bpy.context.scene.camera = bpy.data.objects["Camera"]


def add_map_dict_to_blender(global_map_dict: dict) -> None:
    """Add icospheres for each scene representation embedding to blender scene."""
    global num_fields, config, id_to_name

    for i in range(num_fields):
        position = global_map_dict["positions"][i]
        orientation_wxyz = global_map_dict["orientations"][i]
        orientation_ngs_q = mathutils.Quaternion(orientation_wxyz.numpy(force=True))

        orientation_b_matrix = mathutils.Matrix(
            ngs2ble @ np.asarray(orientation_ngs_q.to_matrix())
        )
        training_iterations = global_map_dict["training_iterations"][i].item()

        bpy.ops.mesh.primitive_ico_sphere_add(
            location=ngs2ble @ position.numpy(force=True),
            rotation=orientation_b_matrix.to_euler(),
            radius=config["field_radius"],
        )
        id_to_name[i] = f"ngs_{i}_{training_iterations}"
        bpy.context.active_object.name = id_to_name[i]


def depsgraph_callback(scene, depsgraph) -> None:
    """Set flag indicating the scene was modified."""
    global scene_changed
    del scene, depsgraph  # unused args
    scene_changed = True


def update_sr() -> None:
    """Update position and orientation from blender to scene representation."""
    global scene_representation, id_to_name, num_fields
    field_positions = torch.empty(num_fields, 3)
    field_orientations = torch.empty(num_fields, 4)

    for i in range(num_fields):
        matrix = bpy.data.objects[id_to_name[i]].matrix_world
        translation = matrix.to_translation()
        quaternion = matrix.to_quaternion()

        field_positions[i, 0] = translation.x
        field_positions[i, 1] = translation.y
        field_positions[i, 2] = translation.z

        field_orientations[i, 0] = quaternion.w
        field_orientations[i, 1] = quaternion.x
        field_orientations[i, 2] = quaternion.y
        field_orientations[i, 3] = quaternion.z

    scene_representation._global_map_dict["positions"] = field_positions.to(config["device"])
    scene_representation._global_map_dict["orientations"] = field_orientations.to(
        config["device"]
    )


def update_cam() -> None:
    """Update camera from blender."""
    global cam, c2w, fig, imshow, rendering, ijs
    blender_camera_obj = bpy.context.scene.camera
    if blender_camera_obj is not None:
        width = bpy.data.scenes["Scene"].render.resolution_x
        height = bpy.data.scenes["Scene"].render.resolution_y

        scene_representation._num_samples = bpy.data.scenes["Scene"].eevee.taa_render_samples
        scene_representation._model._num_knn = max(1,bpy.data.scenes["Scene"].eevee.taa_samples)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        projection_matrix = blender_camera_obj.calc_matrix_camera(depsgraph, x=width, y=height)

        fx = width / 2.0 * projection_matrix[0][0]
        fy = height / 2.0 * projection_matrix[1][1]
        cam = camera.Camera(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=bpy.data.scenes["Scene"].render.resolution_x / 2,
            cy=bpy.data.scenes["Scene"].render.resolution_y / 2,
            pixel_center=0.5,
        )

        c2w = torch.from_numpy(np.asarray(blender_camera_obj.matrix_world)).to(
            config["device"], dtype=torch.float
        )

        rendering = np.random.rand(cam.height, cam.width, 3)

        ijs = torch.cartesian_prod(
            torch.arange(cam.height, device=config["device"]),
            torch.arange(cam.width, device=config["device"]),
        )
    else:
        c2w = None
        cam = None
    print("camera addded")


def execute_rendering() -> float:
    """Sync from blender to scene representation and render pixels."""
    # NOTE: this function runs in GUI thread -> should not block
    global scene_changed, c2w, cam, scene_representation, rendering, imshow

    t1 = time.time()

    if scene_changed:
        update_sr()

        update_cam()

        if cam is not None:
            pass

        scene_changed = False

    if cam is not None:
        with torch.no_grad():
            rgbds, _, _, _, _, _ = scene_representation._render_ijs(
                ijs=ijs, c2ws=c2w, camera=cam, field_ids=field_ids
            )
            rgbds = rgbds.numpy(force=True)
            ijs_np = ijs.numpy(force=True)
            rendering[ijs_np[:, 0], ijs_np[:, 1]] = rgbds[:, :3]

        if imshow is None:
            imshow = plt.imshow(rendering)
            # cv2.waitKey(1)
        else:
            imshow.set_data(rendering)
            width = rendering.shape[1]
            height = rendering.shape[0]
            imshow.set_extent((0,width,0,height))
        plt.draw()
        plt.pause(0.001)

    t2 = time.time()
    print("rendering callback took", t2 - t1)
    return 0.1  # this determines how long until the function is called again


def remap_axis_transform(
    axis_map: dict,
) -> np.ndarray:
    """Return rotation matrix changing coordinate convention.

    Args:
        axis_map:
            Dictionary with zero, one or two elements. Specifying which axis should be
            remapped. If one axis is specified only, the first axes not involved in the
            provided mapping will be set to identity.
            If no axis is specified identity will be returned.
            specified. Keys and values must be one of x, y, z,-x,-y,-z.
    """
    rotation_o2n = np.zeros((3, 3))  # original to new rotation
    alpha_to_num = {"x": 0, "y": 1, "z": 2}

    if len(axis_map) > 2:
        raise ValueError("Maximum two axes remappings can be specified.")
    elif len(axis_map) == 1:  # one degree of freedom -> keep first unpspecified axis
        uninvolved_axis = ["x", "y", "z"]
        try:
            from_, to = list(axis_map.items())[0]
            uninvolved_axis.remove(from_[-1])
            uninvolved_axis.remove(to[-1])
        except ValueError:
            pass
        fixed_axis = uninvolved_axis[0]
        axis_map[fixed_axis] = fixed_axis
    else:  # nothing to be remapped
        axis_map["x"] = "x"
        axis_map["y"] = "y"

    remaining_axes = [0, 1, 2]
    for from_, to in axis_map.items():
        from_axis = alpha_to_num[from_[-1]]
        to_axis = alpha_to_num[to[-1]]
        inv = len(from_) != len(to)  # different length -> different sign
        rotation_o2n[to_axis, from_axis] = -1 if inv else 1
        remaining_axes.remove(from_axis)

    # infer last column
    remaining_axis = remaining_axes[0]
    rotation_o2n[:, remaining_axis] = 1 - np.abs(
        np.sum(rotation_o2n, 1)
    )  # rows must sum to +-1
    rotation_o2n[:, remaining_axis] *= np.linalg.det(rotation_o2n)  # make special orthogonal
    if np.linalg.det(rotation_o2n) != 1.0:  # check if special orthogonal
        raise ValueError("Unsupported combination of remap_{y,x}_axis. det != 1")
    return rotation_o2n


if __name__ == "__main__":
    if "BLENDERVIS" in os.environ:
        blender_main()
    else:
        args = " ".join(sys.argv[1:])

        # NOTE: only tested with pyenv(Python 3.10.9) + blender 3.5
        os.environ["BLENDER_SYSTEM_PYTHON"] = os.path.join(
            sysconfig.get_path("include"), "..", ".."
        )
        # add current environment to PYTHONPATH
        os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)  # needed for virtual envs
        os.environ["BLENDERVIS"] = "1"
        os.environ["BLENDERVISARGS"] = args

        # launch blender such that its python interpreter uses the previously set
        #  PYTHONPATH
        subprocess.call(
            [
                "blender",
                "--python-use-system-env",
                "--python",
                os.path.abspath(__file__),
            ]
        )
