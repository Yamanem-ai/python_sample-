import cv2
import numpy as np
import open3d as o3d

from motion_utils import estimate_pose

def reconstruct_pair(idx0, idx1):

    image_path0 = f"frames/frame_{idx0:03d}.jpg"
    image_path1 = f"frames/frame_{idx1:03d}.jpg"

    img0 = cv2.imread(image_path0)
    
# ============================
# Motion推定
# ============================
    result = estimate_pose(
        image_path0,
        image_path1
    )

    if result is None:
        return None

    R, t, kpts0, kpts1 = result
# ============================
# カメラ行列
# ============================

    h, w = img0.shape[:2]

    cx = w / 2
    cy = h / 2
    f = w
    
    K = np.array([
        [f,0,cx],
        [0,f,cy],
        [0,0,1]
    ],dtype=np.float64)

# ============================
# Projection Matrix
# ============================

    P0 = K @ np.hstack([
        np.eye(3),
        np.zeros((3,1))
    ])

    P1 = K @ np.hstack([
        R,
        t
    ])

# ============================
# Triangulation
# ============================
    pts4d = cv2.triangulatePoints(
        P0,
        P1,
        kpts0.T,
        kpts1.T
    )

    pts3d = (
        pts4d[:3] /
        pts4d[3]
    ).T
    
    pts3d[:,1] *= -1
    
    colors = []

    for p in kpts0:

        x = int(round(p[0]))
        y = int(round(p[1]))

        x = np.clip(x, 0, img0.shape[1]-1)
        y = np.clip(y, 0, img0.shape[0]-1)

        colors.append(
            img0[y, x] / 255.0
        )

    colors = np.array(
        colors,
        dtype=np.float64
    )
    
    return pts3d, kpts0, kpts1, colors
    
if __name__ == "__main__":

    pts3d, pts0, pts1, colors = reconstruct_pair(0, 1)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        pts3d
    )
    pcd.colors = o3d.utility.Vector3dVector(
        colors
    )
    o3d.io.write_point_cloud(
        f"pair_000_001.ply",
        pcd
    )
