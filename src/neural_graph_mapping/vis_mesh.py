import argparse
import open3d as o3d
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--fieldfile", default=None)
    args = parser.parse_args()
    fieldfile = args.fieldfile
    if fieldfile is not None:
        positions = np.loadtxt(fieldfile)
    geometries = []
    combined_mesh = o3d.geometry.TriangleMesh()
    if fieldfile is not None:
        for position in positions:
            center = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            center.translate(position)
            center.paint_uniform_color(np.array([0.3, 0.3, 1.0]))
            geometries.append(center)
            combined_mesh += center

    mesh = o3d.io.read_triangle_mesh(args.filename)
    mesh.compute_vertex_normals()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0,0,0])

    geometries.append(mesh)

    o3d.visualization.draw_geometries(geometries, width=640, height=480)

    combined_mesh += mesh
    o3d.io.write_triangle_mesh("combined.ply", combined_mesh)
