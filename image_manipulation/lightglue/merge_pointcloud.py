import cv2
import numpy as np
import open3d as o3d

from reconstruct_pair import reconstruct_pair


#=============================
# 座標変換関数
#=============================
def transform_points(points, T):

    points_h = np.hstack([
        points,
        np.ones((len(points),1))
    ])

    return (T @ points_h.T).T[:, :3]

#=============================
# 共通特徴点探索
#=============================
def find_common_points(
    pts3d_a,
    imgpts_a,
    pts3d_b,
    imgpts_b,
    threshold=2.0
):

    common_a = []
    common_b = []

    for i,p1 in enumerate(imgpts_a):

        d = np.linalg.norm(
            imgpts_b - p1,
            axis=1
        )

        j = np.argmin(d)

        if d[j] < threshold:

            common_a.append(
                pts3d_a[i]
            )

            common_b.append(
                pts3d_b[j]
            )

    return (
        np.array(common_a,np.float32),
        np.array(common_b,np.float32)
    )

# =================================
# estimateAffine3D
# =================================
def estimate_pair_transform(
    name_a,
    name_b,
    pts_a_3d,
    pts_a_2d,
    pts_b_3d,
    pts_b_2d
):

    common_a, common_b = find_common_points(
        pts_a_3d,
        pts_a_2d,
        pts_b_3d,
        pts_b_2d
    )

    print(
        f"{name_a}-{name_b} common =",
        len(common_a)
    )

    retval, T, inliers = cv2.estimateAffine3D(
        common_b,
        common_a
    )

    print(
        f"{name_a}-{name_b} inliers =",
        np.count_nonzero(inliers)
    )

    return T

# ============================
# ペア生成
# ============================
NUM_FRAMES = 10

pairs = []

for i in range(NUM_FRAMES-1):

    pairs.append(
        reconstruct_pair(
            i,
            i+1
        )
    )

# ============================
# 隣接ペア間の変換(Transform生成)
# ============================
transforms = []

for i in range(NUM_FRAMES-2):

    pts3d_a, pts0_a, pts1_a, colors_a = pairs[i]
    pts3d_b, pts0_b, pts1_b, colors_b = pairs[i+1]

    T = estimate_pair_transform(
        f"{i}{i+1}",
        f"{i+1}{i+2}",
        pts3d_a,
        pts1_a,
        pts3d_b,
        pts0_b
    )

    transforms.append(T)
# ============================
# pair01座標系へ変換
# ============================
global_T = []

global_T.append(
    np.eye(4)
)

for T in transforms:

    T4 = np.eye(4)
    T4[:3,:4] = T

    global_T.append(
        global_T[-1] @ T4
    )

frame_colors = [
    [1,0,0],      # red
    [0,1,0],      # green
    [0,0,1],      # blue
    [1,1,0],      # yellow
    [1,0,1],      # magenta
    [0,1,1],      # cyan
    [0.5,0,0],    # dark red
    [0,0.5,0],    # dark green
    [0,0,0.5],    # dark blue
]

merged = []
merged_colors = []

for i in range(len(pairs)):

    pts3d, _, _, colors = pairs[i]

    pts_world = transform_points(
        pts3d,
        global_T[i]
    )

    merged.append(
        pts_world
    )
    
    merged_colors.append(
        colors
    )

# ==============================
# 一つの配列へ結合
# ==============================
merged = np.vstack(merged)
merged_colors = np.vstack(merged_colors)

print(
    "merged points =",
    len(merged)
)

# ============================
# Open3D
# ============================

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    merged
)
pcd.colors = o3d.utility.Vector3dVector(
    merged_colors
)
o3d.io.write_point_cloud(
    "sfm_combined.ply",
    pcd
)
