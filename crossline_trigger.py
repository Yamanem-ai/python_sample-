import cv2
import time
import numpy as np

import yolox_ex   # ← ここ重要

# ======================
# 設定
# ======================
DETECTION_TIME = 2.0
MOTION_THR = 2000

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
def create_line_mask(shape, line, thickness=40):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.line(mask, line[0], line[1], 255, thickness)
    return mask

# ======================
# メイン
# ======================
def main():

    # 🔥 YOLOXはここで1回だけ初期化
    detector = yolox_ex.create_detector()
    print(type(detector))
    cap = cv2.VideoCapture("animal01.mp4")
    cv2.namedWindow("demo")
    
    # マウスでラインを引く (クリック、ドラッグ、離す)
    cv2.setMouseCallback("demo", mouse_callback)

    # 背景と動いているものを分離 MOG2: Mixture of Gaussians
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

        if len(line_points) == 2:
            cv2.line(display, line_points[0], line_points[1], (0,255,0), 2)

        # ======================
        # 軽量トリガー
        # ======================
        if mode == "line" and len(line_points) == 2:

            fgmask = bg_sub.apply(frame) #背景差分、動いているピクセルを抽出
            fgmask = cv2.medianBlur(fgmask, 5) #小さいノイズを消す

            # ライン周辺だけ監視
            line_mask = create_line_mask(frame.shape, line_points)
            motion = cv2.bitwise_and(fgmask, fgmask, mask=line_mask)
            
            #どれだけ動いたかをピクセル数で表現
            motion_count = cv2.countNonZero(motion)
            
            # エッジ検知（新しく動きが発生した瞬間だけ）
            if motion_count > MOTION_THR and prev_motion <= MOTION_THR:
                mode = "detect"
                detect_start = time.time()

            # 次フレーム用に保存
            prev_motion = motion_count

        # ======================
        # YOLOX
        # ======================
        elif mode == "detect":

            objs = yolox_ex.detect_frame(detector, frame)
            display = yolox_ex.plot_results(objs, display, yolox_ex.COCO_CATEGORY)

            if time.time() - detect_start > DETECTION_TIME:
                mode = "line"

        cv2.putText(display, f"MODE: {mode}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("demo", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
