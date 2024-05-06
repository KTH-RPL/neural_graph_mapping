import argparse
import open3d as o3d
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    mesh = o3d.io.read_triangle_mesh(args.filename)
    mesh_sim = mesh.simplify_vertex_clustering(
        voxel_size = 0.05, contraction=o3d.geometry.SimplificationContraction.Average)

    path = pathlib.Path(args.filename)
    out_path = path.with_stem(path.stem + "_simplified")
    o3d.io.write_triangle_mesh(str(out_path), mesh_sim)
