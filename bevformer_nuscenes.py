import os
import cv2
#import json
import numpy as np
#import matplotlib.pyplot as plt

import ailia
import time

from nuscenes.nuscenes import NuScenes

# -----------------------------
# Parameters
# -----------------------------

MODEL_PATH = "bevformer_tiny.onnx.prototxt"
WEIGHT_PATH = "bevformer_tiny.onnx"

NUSCENES_ROOT = "data/nuscenes"
VERSION = "v1.0-mini"

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT"
]

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 800

IMG_MEAN = np.array([123.675,116.28,103.53],dtype=np.float32)
IMG_STD = np.array([58.395,57.12,57.375],dtype=np.float32)

THRESHOLD = 0.3

NUSCENES_CLASSES = [
    "car","truck","construction_vehicle","bus","trailer",
    "barrier","motorcycle","bicycle","pedestrian","traffic_cone"
]


# -----------------------------
# Preprocess
# -----------------------------

def preprocess(img):

    img = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)

    img = (img-IMG_MEAN)/IMG_STD

    img = img.transpose(2,0,1)

    return img


def prepare_input(imgs):

    cams = [preprocess(i) for i in imgs]

    cams = np.stack(cams)

    cams = np.expand_dims(cams,0)

    return cams.astype(np.float32)


# -----------------------------
# Postprocess
# -----------------------------

def sigmoid(x):
    return 1/(1+np.exp(-x))


def decode_bbox(pred):
    cx,cy,cz = pred[0:3]
    w,l,h = np.exp(pred[3:6])
    sin_yaw,cos_yaw = pred[6:8]
    yaw = np.arctan2(sin_yaw,cos_yaw)

    return cx,cy,cz,w,l,h,yaw


def postprocess(cls_scores,bbox_preds):

    cls_scores = cls_scores[0]
    bbox_preds = bbox_preds[0]

    scores = sigmoid(cls_scores)

    detections=[]

    for q in range(scores.shape[0]):

        cls = np.argmax(scores[q])
        score = scores[q,cls]

        if score<THRESHOLD:
            continue

        box = decode_bbox(bbox_preds[q])

        detections.append({
            "label":int(cls),
            "class":NUSCENES_CLASSES[cls],
            "score":float(score),
            "box":box
        })

    detections.sort(key=lambda x:-x["score"])

    return detections[:100]


# -----------------------------
# Visualization
# -----------------------------

CLASS_COLORS = {
    0:(0,0,255),      # car → red
    1:(0,128,255),    # truck → orange
    2:(0,255,255),    # construction_vehicle → yellow
    3:(255,0,0),      # bus → blue
    4:(255,0,255),    # trailer → magenta
    5:(128,128,128),  # barrier → gray
    6:(255,128,0),    # motorcycle → sky blue
    7:(0,255,0),      # bicycle → green
    8:(255,255,0),    # pedestrian → cyan
    9:(0,128,0)       # traffic_cone → dark green
}


def draw_bev(detections, lidar_points=None):

    bev = np.ones((500,500,3),dtype=np.uint8)*255
    scale = 4

    # -------------------------
    # LiDAR描画
    # -------------------------

    if lidar_points is not None:

        pts = lidar_points[:,:2]

        mask = np.linalg.norm(pts,axis=1) < 50
        pts = pts[mask]

        px = (250 + pts[:,0]*scale).astype(np.int32)
        py = (250 - pts[:,1]*scale).astype(np.int32)

        mask = (px>=0)&(px<500)&(py>=0)&(py<500)

        bev[py[mask],px[mask]] = (180,180,180)

    # -------------------------
    # detection
    # -------------------------

    for det in detections:

        cx,cy,cz,w,l,h,yaw = det["box"]

        px = int(250 + cx*scale)
        py = int(250 - cy*scale)

        w_pix = int(w*scale)
        l_pix = int(l*scale)

        rect=((px,py),(l_pix,w_pix),-(yaw+np.pi/2)*180/np.pi)

        pts=cv2.boxPoints(rect).astype(np.int32)

        label = det["label"]

        color = CLASS_COLORS.get(label,(150,150,150))

        cv2.polylines(bev,[pts],True,color,2)

    cv2.circle(bev,(250,250),6,(0,0,255),-1)

    return bev


# -----------------------------
# NuScenes Loader
# -----------------------------

def load_sample(nusc, sample):

    CAMS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"
    ]

    imgs = []

    for cam in CAMS:

        cam_token = sample["data"][cam]
        cam_data = nusc.get("sample_data", cam_token)

        img_path = os.path.join(nusc.dataroot, cam_data["filename"])

        img = cv2.imread(img_path)

        imgs.append(img)

    # LiDAR
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)

    lidar_path = os.path.join(nusc.dataroot, lidar_data["filename"])

    points = np.fromfile(lidar_path,dtype=np.float32).reshape(-1,5)[:,:3]

    return imgs, points
    
def compose_frame(cam_imgs, bev):

    canvas = np.zeros((900,1200,3),dtype=np.uint8)

    f, fl, fr, b, bl, br = cam_imgs
    
    # 表示用resize
    fl = cv2.resize(fl,(400,250))
    f  = cv2.resize(f,(400,250))
    fr = cv2.resize(fr,(400,250))
    bl = cv2.resize(bl,(400,250))
    b  = cv2.resize(b,(400,250))
    br = cv2.resize(br,(400,250))

    # front row
    canvas[0:250,0:400] = fl
    canvas[0:250,400:800] = f
    canvas[0:250,800:1200] = fr

    # back row
    canvas[650:900,0:400] = br
    canvas[650:900,400:800] = b
    canvas[650:900,800:1200] = bl

    # BEV
    bev = cv2.resize(bev,(400,400))
    canvas[250:650,400:800] = bev

    return canvas
    

# -----------------------------
# Main
# -----------------------------

video_path = "bevformer_demo.mp4"

writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    10,                 # FPS
    (1200,900)          # 画面サイズ
)

def main():

    print("Loading NuScenes...")

    nusc = NuScenes(
        version=VERSION,
        dataroot=NUSCENES_ROOT,
        verbose=True
    )

    print("Loading model...")

    # -----------------------------
    # 自動 env_id 選択（GPU優先だがとりあえずCPUに変更）
    # -----------------------------
    #env_id = ailia.get_gpu_environment_id()
    env_id = 2

     #環境情報表示（デバッグ用）
    env = ailia.get_environment(env_id)
    print("Environment:", env)

    # モデルロード

    net = ailia.Net(MODEL_PATH,WEIGHT_PATH,env_id=env_id)
   # net = ailia.Net(MODEL_PATH,WEIGHT_PATH)

    for i,sample in enumerate(nusc.sample):

        print("Frame",i)

        cam_imgs, points = load_sample(nusc,sample)

        input_tensor = prepare_input(cam_imgs)

        start_time = time.time()
        output = net.predict([input_tensor])
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
   
        cls_scores,bbox_preds = output

        detections = postprocess(cls_scores,bbox_preds)

        print("Detected:",len(detections))

        bev = draw_bev(detections, points)
       
        frame = compose_frame(cam_imgs, bev)

        cam_vis = cv2.resize(cam_imgs[0],(500,300))

        canvas=np.zeros((800,1000,3),dtype=np.uint8)

        canvas[0:300,250:750]=cam_vis

        canvas[300:800,250:750]=cv2.resize(bev,(500,500))
        
        cv2.putText(frame, f"FPS:{fps:.2f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        cv2.imshow("BEVFormer demo", frame)
        writer.write(frame)

        if cv2.waitKey(1)==27:
            break

    cv2.destroyAllWindows()
    writer.release()
    
    print("Video saved: bevformer_demo.mp4")

if __name__=="__main__":
    main()
