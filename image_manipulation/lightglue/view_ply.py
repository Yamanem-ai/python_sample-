import sys
import open3d as o3d

if len(sys.argv) < 2:
    print("Usage:")
    print("  python3.10 view_ply.py <ply file>")
    sys.exit(1)

ply_file = sys.argv[1]

pcd = o3d.io.read_point_cloud(ply_file)

o3d.visualization.draw_geometries([pcd])
