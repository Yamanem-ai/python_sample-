# BEVFormer Demo (Class Colored Version)

import time
import argparse
import torch
import sys
import os
import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

sys.path.append('.')
import projects.mmdet3d_plugin

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset
from mmdet3d.models import build_detector


# -----------------------------
# NuScenes class colors
# -----------------------------
CLASS_COLORS = {
    0:(255,0,0),     # car
    1:(0,255,0),     # truck
    2:(0,0,255),     # bus
    3:(255,255,0),   # trailer
    4:(255,0,255),   # construction_vehicle
    5:(0,255,255),   # pedestrian
    6:(128,128,0),   # motorcycle
    7:(128,0,128),   # bicycle
    8:(0,128,128),   # traffic_cone
    9:(80,80,80)     # barrier
}

NUSCENES_CLASSES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--out-dir", default="demo_out")
    return parser.parse_args()


# -----------------------------
# LiDAR読み込み
# -----------------------------
def load_lidar_points(lidar_path):

    pts = np.fromfile(lidar_path, dtype=np.float32)
    pts = pts.reshape(-1,5)

    return pts[:,:3]


# -----------------------------
# BEV描画
# -----------------------------
def draw_bev(boxes, scores, labels, score_thr, lidar_points):

    bev = np.ones((500,500,3),dtype=np.uint8)*255
    scale = 5

    # -------------------------
    # LiDAR
    # -------------------------
    if lidar_points is not None:

        mask = np.linalg.norm(lidar_points[:,:2],axis=1) < 50
        lidar_points = lidar_points[mask]

        if len(lidar_points) > 10000:
            step = len(lidar_points) // 10000
            lidar_points = lidar_points[::step]

        pts = lidar_points[:,:2]

        px = (250 + pts[:,0]*scale).astype(np.int32)
        py = (250 - pts[:,1]*scale).astype(np.int32)

        mask = (px>=0)&(px<500)&(py>=0)&(py<500)

        bev[py[mask],px[mask]] = (180,180,180)

    # -------------------------
    # Detection
    # -------------------------
    boxes_np = boxes.tensor.cpu().numpy()
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    mask = scores_np > score_thr

    boxes_np = boxes_np[mask]
    labels_np = labels_np[mask]

    for box, label in zip(boxes_np, labels_np):

        x, y, z, w, l, h, yaw = box[:7]

        cx = int(250 + x * scale)
        cy = int(250 - y * scale)

        if cx < 0 or cx >= 500 or cy < 0 or cy >= 500:
            continue

        w_pix = int(w * scale)
        l_pix = int(l * scale)

        angle = -(yaw + np.pi/2) * 180.0 / np.pi

        rect = ((cx, cy), (l_pix, w_pix), angle)
        corners = cv2.boxPoints(rect).astype(np.int32)

        color = CLASS_COLORS.get(int(label),(0,0,0))

        cv2.polylines(bev,[corners],True,color,2)

    # ego vehicle
    cv2.circle(bev,(250,250),6,(0,0,255),-1)

    return bev


def load_camera_images(paths):

    imgs = []

    for p in paths:

        img = cv2.imread(p)
        img = cv2.resize(img,(400,250))
        imgs.append(img)

    return imgs


def compose_frame(cam_imgs, bev):

    canvas = np.zeros((900,1200,3),dtype=np.uint8)

    f, fr, fl, b, br, bl = cam_imgs

    canvas[0:250,0:400] = fl
    canvas[0:250,400:800] = f
    canvas[0:250,800:1200] = fr

    canvas[650:900,0:400] = bl
    canvas[650:900,400:800] = b
    canvas[650:900,800:1200] = br

    bev = cv2.resize(bev,(400,400))
    canvas[250:650,400:800] = bev

    return canvas


def main():
   # cv2.startWindowThread()
    #cv2.namedWindow("BEVFormer Demo", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("BEVFormer Demo",1200,900)

    args = parse_args()

    os.makedirs(args.out_dir,exist_ok=True)

    cfg = Config.fromfile(args.config)

    cfg.model.pretrained=None
    cfg.data.test.test_mode=True
    cfg.model.train_cfg=None

    print("Building dataset...")
    dataset = custom_build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False)

    print("Building model...")
    model = build_detector(cfg.model,test_cfg=cfg.get("test_cfg"))

    if cfg.get("fp16",None):
        wrap_fp16_model(model)

    load_checkpoint(model,args.checkpoint,map_location="cpu")

    model = MMDataParallel(model.cuda(),device_ids=[0])
    model.eval()

    video_path = os.path.join(args.out_dir,"bevformer_demo.mp4")

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (1200,900)
    )

    print("Running demo...")

    for i,data in enumerate(data_loader):

        if i>=args.samples:
            break

        start_inf = time.time()

        with torch.no_grad():
            result = model(return_loss=False,rescale=True,**data)

        end_inf = time.time()

        fps = 1.0/(end_inf-start_inf)

        pts_bbox = result[0]["pts_bbox"]

        boxes = pts_bbox["boxes_3d"]
        scores = pts_bbox["scores_3d"]
        labels = pts_bbox["labels_3d"]

        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()

        mask = scores_np > args.score_thr
        labels_np = labels_np[mask]

        detected_classes = [NUSCENES_CLASSES[int(l)] for l in labels_np]

        print(f"frame {i} detected:", detected_classes)

        img_metas = data["img_metas"][0].data[0][0]
        if 0 in img_metas:
            img_metas = img_metas[0]

        img_paths = img_metas["filename"]

        lidar_path = img_metas.get("pts_filename",None)

        lidar_points = load_lidar_points(lidar_path) if lidar_path else None

        cam_imgs = load_camera_images(img_paths)

        bev = draw_bev(boxes,scores,labels,args.score_thr,lidar_points)

        frame = compose_frame(cam_imgs,bev)

        cv2.rectangle(frame,(10,10),(200,60),(0,0,0),-1)

        cv2.putText(frame,
                    f"Inference FPS: {fps:.1f}",
                    (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        cv2.imshow("BEVFormer Demo",frame)
        cv2.waitKey(5)

        writer.write(frame)

        print("frame",i)

    writer.release()

    print("Dataset length =",len(dataset))
    print("Video saved:",video_path)


if __name__=="__main__":
    main()
