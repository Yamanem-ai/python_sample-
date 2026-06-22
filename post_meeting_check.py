# post_meeting_check.py

import cv2
import numpy as np

from padim_api import PadimEngine


# ==========================
# Settings
# ==========================

FEATURE_FILE = "meeting.pkl"

THRESHOLD = 30

CAMERA_ID = 0


# ==========================
# Main
# ==========================

def main():

    print("Initialize PaDiM...")

    padim = PadimEngine()

    padim.initialize()

    padim.load(FEATURE_FILE)

    print("Open Camera...")

    #cap = cv2.VideoCapture(CAMERA_ID)
    cap = cv2.VideoCapture("IMG_9239.MOV")

    if not cap.isOpened():
        print("ERROR : camera open failed")
        return

    while True:

        ret, frame = cap.read()
        frame = cv2.rotate(
            frame,
            cv2.ROTATE_90_CLOCKWISE
        )
        print("frame =", frame.shape)
        cv2.imwrite("debug_frame.jpg", frame)
        if not ret:
            break
        
        # ==========================
        # PaDiM Inference
        # ==========================

        score, score_map = padim.infer_and_score(frame)

        # ==========================
        # Heatmap
        # ==========================

        heat_map = score_map.copy()

        heat_map = (
            heat_map - heat_map.min()
        ) / (
            heat_map.max() - heat_map.min() + 1e-8
        )

        heat_map = (heat_map * 255).astype(np.uint8)

        #---------------------------
        # restore crop area
        #---------------------------
        
        h, w = frame.shape[:2]
        
        IMAGE_RESIZE = 256
        IMAGE_SIZE = 224
        
        if h > w:
            resized_h = int(IMAGE_RESIZE * h / w)
            resized_w = IMAGE_RESIZE
        else:
            resized_h = IMAGE_RESIZE
            resized_w = int(IMAGE_RESIZE * w / h)
            
        pad_h = (resized_h - IMAGE_SIZE) // 2
        pad_w = (resized_w - IMAGE_SIZE) // 2
        
        # 黒キャンバス生成
        canvas = np.zeros(
            (resized_h, resized_w),
            dtype=np.uint8
        )
        
        canvas[
            pad_h:pad_h + IMAGE_SIZE,
            pad_w:pad_w + IMAGE_SIZE
        ] = heat_map
        
        heat_map = cv2.applyColorMap(
            canvas,
            cv2.COLORMAP_JET
        )

        heat_map = cv2.resize(
            heat_map,
            (frame.shape[1], frame.shape[0])
        )

        overlay = cv2.addWeighted(
            frame,
            0.6,
            heat_map,
            0.4,
            0
        )

        # ==========================
        # Alarm
        # ==========================

        if score > THRESHOLD:

            cv2.putText(
                overlay,
                "FORGOTTEN ITEM DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            print(
                f"ALARM score={score:.3f}"
            )

        cv2.putText(
            overlay,
            f"Score : {score:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Meeting Room Check",
            overlay
        )
        cv2.imshow(
            "Canvas",
            canvas
        )
        print("overlay =", overlay.shape)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
