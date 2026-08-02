import cv2
import numpy as np
import open3d as o3d

from lightglue_api import LightGlueAPI

NUM_FRAMES = 10

lg = LightGlueAPI()

# ------------------------
# 最初のカメラ
# ------------------------

R_global = np.eye(3)
t_global = np.zeros((3,1))

camera_positions = []

camera_positions.append(
    np.zeros(3)
)

# ------------------------
# フレームを順番に処理
# ------------------------

for idx in range(NUM_FRAMES - 1):

    print(
        f"{idx:03d} -> {idx+1:03d}"
    )

    img_path0 = (
        f"frames/frame_{idx:03d}.jpg"
    )

    img_path1 = (
        f"frames/frame_{idx+1:03d}.jpg"
    )

    img = cv2.imread(img_path0)

    h,w = img.shape[:2]

    cx = w / 2
    cy = h / 2

    f = w

    from motion_utils import estimate_pose

    result = estimate_pose(
        img_path0,
        img_path1
    )

    if result is None:
        continue

    R, t, kpts0, kpts1 = result
    

    # ------------------------
    # 累積
    # ------------------------

    t_global = (
        t_global +
        R_global @ t
    )

    R_global = (
        R @ R_global
    )

    pos = t_global.flatten()

    camera_positions.append(
        pos
    )

    print(
        "camera =",
        pos
    )

# ------------------------
# PLY保存
# ------------------------

camera_positions = np.array(
    camera_positions,
    dtype=np.float64
)

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    camera_positions
)

colors = np.zeros_like(
    camera_positions
)

colors[:,0] = 1.0

pcd.colors = o3d.utility.Vector3dVector(
    colors
)

o3d.io.write_point_cloud(
    "camera_path.ply",
    pcd
)

print(
    "saved camera_path.ply"
)
