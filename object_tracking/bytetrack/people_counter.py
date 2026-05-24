import cv2
import time
import numpy as np

from bytetrack_ex import (
    create_bytetrack,
    update_bytetrack
)

# ======================
# ライン描画
# ======================
line_points = []
track_history = {}
crossed_ids = set()
cross_count = 0
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

def point_side_of_line(pt, line):

    x, y = pt
    (x1, y1), (x2, y2) = line

    return (
        (x - x1) * (y2 - y1)
        -
        (y - y1) * (x2 - x1)
    )
    
# ======================
# メイン
# ======================
def main():

    global cross_count
    # bytetrack初期化
    net, tracker = create_bytetrack()
    print(type(net))
    cap = cv2.VideoCapture("Paris.mp4")
    cv2.namedWindow("demo")
    
    # マウスでラインを引く (クリック、ドラッグ、離す)
    cv2.setMouseCallback("demo", mouse_callback)

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
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break  # ライン確定して開始

        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return
    
    # ======================
    # メインループ
    # ======================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # ======================
        # bytetrack
        # ======================

        results = update_bytetrack(net, tracker, frame)

        #print(results)
        for obj in results:

            x, y, w, h = obj["bbox"]

            tid = obj["id"]

            foot = obj["foot"]

            # ======================
            # bbox描画
            # ======================
            
            cv2.rectangle(
                display,
                (int(x), int(y)),
                (int(x+w), int(y+h)),
                (0,255,0),
                2
            )
            
            cv2.putText(
                display,
                str(tid),
                (int(x), int(y)-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

            # ======================
            # 足元描画
            # ======================

            cv2.circle(
                display,
                foot,
                5,
                (0,0,255),
                -1
            )

            # ======================
            # ライン横断判定
            # ======================

            if len(line_points) == 2:

                current_side = point_side_of_line(
                    foot,
                    line_points
                )

                if tid in track_history:

                    prev_side = track_history[tid]

                    # 符号変化 = 横断
                    if prev_side * current_side < 0:

                        if tid not in crossed_ids:

                            crossed_ids.add(tid)

                            cross_count += 1

                            print(
                                f"CROSS ID={tid} "
                                f"COUNT={cross_count}"
                            )

                track_history[tid] = current_side

        # ======================
        # ライン描画
        # ======================

        if len(line_points) == 2:

            cv2.line(
                display,
                line_points[0],
                line_points[1],
                (255,0,0),
                3
            )

        # ======================
        # COUNT表示
        # ======================

        cv2.putText(
            display,
            f"COUNT: {cross_count}",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,0,255),
            3
        )

        cv2.imshow("demo", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
