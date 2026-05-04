import cv2
import time
import numpy as np

import dense_prediction_transformers_ex as dpt_ex   # ← YOLOX → DPTに変更

# ======================
# 設定
# ======================
DETECTION_TIME = 2.0
MOTION_THR = 10000

# ======================
# ライン描画
# ======================
line_points = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    global line_points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        line_points = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if len(line_points) == 1:
            line_points.append((x, y))
        else:
            line_points[1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# ======================
# ラインマスク
# ======================
def create_line_mask(shape, line, thickness=35):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.line(mask, line[0], line[1], 255, thickness)
    return mask


# ======================
# セグメンテーション可視化
# ======================
def visualize_seg(frame, seg):
    color = np.zeros_like(frame)

    # クラス例（モデルにより調整）
    WATER_ID = 22

    for cid in np.unique(seg):
        color[seg == WATER_ID] = [0, 0, 255]
    
    return cv2.addWeighted(frame, 0.6, color, 0.6, 0)

# ======================
# メイン
# ======================
def main():

    # 🔥 DPT初期化
    net = dpt_ex.create_net()

    cap = cv2.VideoCapture("river.mp4")
    cv2.namedWindow("demo")
    
    # マウスでラインを引く (クリック、ドラッグ、離す)
    cv2.setMouseCallback("demo", mouse_callback)

    # 背景と動いているものを分離
    bg_sub = cv2.createBackgroundSubtractorMOG2()

    mode = "line"
    detect_start = 0
    
    # ======================
    # 🔥 最初のフレームを取得して、静止画として使う
    # ======================
    ret, first_frame = cap.read()
    if not ret:
        return
    
    # ======================
    # 🔥 ライン設定フェーズ
    # ======================
    while True:
        display = first_frame.copy()

        if len(line_points) == 2:
            cv2.line(display, line_points[0], line_points[1], (0,255,0), 2)

        cv2.putText(display, "Draw line and press SPACE to start",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("demo", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACEキー
            if len(line_points) == 2:
                break  # ライン確定して開始

        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    # 🔥 motion初期化（これ重要）
    prev_motion = 0
    # ======================
    # メインループ
    # ======================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if mode == "detect" and len(line_points) == 2:
            seg = dpt_ex.infer_frame(net, frame)
            display = visualize_seg(display, seg)   # ← displayを渡す
        if len(line_points) == 2:
            cv2.line(display, line_points[0], line_points[1], (0,255,0), 2)
        
        # ======================
        # 軽量トリガー
        # ======================
        if mode == "line" and len(line_points) == 2:

            fgmask = bg_sub.apply(frame)
            fgmask = cv2.medianBlur(fgmask, 5)

            line_mask = create_line_mask(frame.shape, line_points)
            motion = cv2.bitwise_and(fgmask, fgmask, mask=line_mask)

            motion_count = cv2.countNonZero(motion)

            if motion_count > MOTION_THR and prev_motion <= MOTION_THR:
                mode = "detect"
                detect_start = time.time()

            prev_motion = motion_count

        # ======================
        # DPT推論
        # ======================
        elif mode == "detect":

            seg = dpt_ex.infer_frame(net, frame)
            print("seg unique:", np.unique(seg))
            # 可視化
            display = visualize_seg(frame, seg)

            if time.time() - detect_start > DETECTION_TIME:
                mode = "line"
                prev_motion = 0

        cv2.putText(display, f"MODE: {mode}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("demo", display)
        
        #time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
