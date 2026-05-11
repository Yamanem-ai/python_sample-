import cv2
import numpy as np

from segment_anything_2_ex import load_model, SAM2Tracker, args, preprocess_frame, show_points

# =========================
# グローバル変数
# =========================
click_points = []
click_labels = []

drawing_frame = None
started = False

# =========================
# マウスイベント
# =========================
def mouse_callback(event, x, y, flags, param):
    global click_points, click_labels

    if event == cv2.EVENT_LBUTTONUP:
        print(f"Left click: ({x}, {y})")
        click_points.append([x, y])
        click_labels.append(1)

    elif event == cv2.EVENT_RBUTTONUP:
        print(f"Right click: ({x}, {y})")
        click_points.append([x, y])
        click_labels.append(0)


# =========================
# メイン
# =========================
def main():
    global drawing_frame, started

    # カメラ起動
    cap = cv2.VideoCapture("animal01.mp4")
    
    # ======================
    # 🔥 最初のフレームを取得して、静止画として使う
    # ======================
    ret, first_frame = cap.read()
    if not ret:
        return

    # SAM2初期化（あなたの既存コードを流用）
    image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp = load_model(args)

    tracker = SAM2Tracker(
        image_encoder,
        prompt_encoder,
        mask_decoder,
        memory_attention,
        memory_encoder,
        mlp,
        args
    )

    cv2.namedWindow("frame")
    cv2.imshow("frame", first_frame)
    cv2.setMouseCallback("frame", mouse_callback)

    print("Clickで点を指定して 's' で開始")

    # 👉 最初のフレームを先に登録
    tracker.add_frame(first_frame)
    
    while not started:
            
        display_frame = first_frame.copy()

        if len(click_points) > 0:
            pts = np.array(click_points)
            labels = np.array(click_labels)
            drawing_frame = show_points(pts, labels, display_frame)

        cv2.imshow("frame", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s') and not started:
            if len(click_points) == 0:
                print("点を指定してください")
                continue

            pts = np.array(click_points, dtype=np.float32)
            labels = np.array(click_labels, dtype=np.int32)

            tracker.set_points(pts, labels)
            started = True
            print("Tracking started!")
            
    # =========================
    # トラッキングループ
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracker.add_frame(frame)
        frame = tracker.step(frame)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
