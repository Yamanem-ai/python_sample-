import cv2
import numpy as np
import time
import queue

from yolox_ex import (
    create_detector,
    detect_frame,
)

from clip_ex import (
    create_clip,
    create_text_features,
    predict_clip,
)

from strongsort_ex import create_tracker, update_tracker

# ======================
# Settings
# ======================

YOLO_ENV_ID = 0
CLIP_ENV_ID = 0

ALARM_THRESHOLD = 0.75
ALARM_INTERVAL_SEC = 3.0

CLIP_HISTORY_FRAMES = 5
CLIP_REQUIRED_HITS = 3

# 人物ごとのCLIP判定履歴
clip_histories = []

# ======================
# Settings
# ======================

# suspect追跡後、CLIPによる探索に戻る
RETURN_TO_CLIP = True

# 1人のsuspectを追跡する時間（秒）
SUSPECT_TRACKING_DURATION_SEC = 10.0

TEXT_INPUTS = [
    "a cyclist wearing headphones",
    "a cyclist not wearing headphones",
]

# TARGET_CLASSES = None
TARGET_CLASSES = [
    "person",
    "bicycle",
]

CAMERA_ID = 1

# ======================
# Initialize
# ======================

print("Initializing YOLOX...")

detector = create_detector(
    env_id=YOLO_ENV_ID
)

print("Initializing CLIP...")

net_image, net_text = create_clip(
    model_type='ViTB32',
    env_id=CLIP_ENV_ID
)

print("Creating text features...")

text_feature = create_text_features(
    net_text,
    TEXT_INPUTS
)

print("Initializing StrongSORT...")

tracker, frid_net = create_tracker()


# ======================
# Camera
# ======================

cap = cv2.VideoCapture("headphone3.mov")

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")


last_alarm_time = 0

# ======================
# 違反者の追跡状態
# ======================

# CLIPで違反者を一度検出したか
suspect_active = False

# CLIPで違反者と判定された人物のBBox
# [x, y, width, height]
suspect_bbox = None
suspect_tracking_start_time = None

# ======================
# Main loop
# ======================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    display = frame.copy()

    # ======================
    # YOLOX detection
    # ======================

    objs = detect_frame(
        detector,
        frame,
        target_classes=TARGET_CLASSES
    )

    persons = []
    bicycles = []

    for obj in objs:

        label = int(obj.category)

        if label == 0:
            persons.append(obj)

        elif label == 1:
            bicycles.append(obj)


    # ==========================================================
    # まだ違反者が登録されていない場合
    # → person + bicycle
    # → CLIP
    # → 違反者登録
    # ==========================================================

    if not suspect_active:

        for person in persons:

            px1 = int(person.x * frame.shape[1])
            py1 = int(person.y * frame.shape[0])

            px2 = int(
                (person.x + person.w) * frame.shape[1]
            )

            py2 = int(
                (person.y + person.h) * frame.shape[0]
            )

            # clamp

            px1 = max(0, px1)
            py1 = max(0, py1)

            px2 = min(frame.shape[1], px2)
            py2 = min(frame.shape[0], py2)

            if px2 <= px1 or py2 <= py1:
                continue


            # ======================
            # nearby bicycle check
            # ======================

            is_cyclist = False

            person_center_x = (px1 + px2) / 2
            person_center_y = (py1 + py2) / 2

            for bicycle in bicycles:

                bx1 = int(
                    bicycle.x * frame.shape[1]
                )

                by1 = int(
                    bicycle.y * frame.shape[0]
                )

                bx2 = int(
                    (bicycle.x + bicycle.w) *
                    frame.shape[1]
                )

                by2 = int(
                    (bicycle.y + bicycle.h) *
                    frame.shape[0]
                )

                bicycle_center_x = (bx1 + bx2) / 2
                bicycle_center_y = (by1 + by2) / 2

                dist = np.sqrt(
                    (person_center_x - bicycle_center_x) ** 2 +
                    (person_center_y - bicycle_center_y) ** 2
                )

                if dist < 200:

                    is_cyclist = True

                    break


            if not is_cyclist:
                continue


            # ======================
            # CLIP
            # ======================

            head_y2 = (
                py1 +
                int((py2 - py1) * 0.2)
            )

            crop_x1 = (
                px1 +
                int((px2 - px1) * 0.35)
            )

            crop_x2 = (
                px2 -
                int((px2 - px1) * 0.20)
            )

            crop = frame[
                py1:head_y2,
                crop_x1:crop_x2
            ]

            if crop.size == 0:
                continue

            result = predict_clip(
                net_image,
                crop,
                text_feature,
                TEXT_INPUTS
            )

            best_text = result['best_text']
            best_score = result['best_score']

            print(
                best_text,
                best_score
            )

            # ======================
            # CLIPによる単一フレーム判定
            # ======================
            is_suspect = (
                best_score >= ALARM_THRESHOLD
                and best_text.strip().lower() == "a cyclist wearing headphones"
            )

            # ======================
            # 同じ人物の判定履歴を取得
            # BBoxのIoUで前フレームの人物と対応付け
            # ======================
            current_bbox = [
                px1, py1, px2 - px1, py2 - py1
            ]

            def bbox_iou(a, b):
                ax1, ay1, aw, ah = a
                bx1, by1, bw, bh = b

                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax1 + aw, bx1 + bw)
                iy2 = min(ay1 + ah, by1 + bh)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = iw * ih

                union = aw * ah + bw * bh - intersection
                return intersection / union if union > 0 else 0.0


            best_history = None
            best_iou = 0.0

            for history in clip_histories:
                iou = bbox_iou(current_bbox, history["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_history = history

            # IoUが0.3を超える場合、同じ人物とみなす
            if best_history is None or best_iou < 0.3:
                best_history = {
                    "bbox": current_bbox,
                    "decisions": [],
                }
                clip_histories.append(best_history)

            best_history["bbox"] = current_bbox
            best_history["decisions"].append(is_suspect)

            # 直近5回分だけ保持
            best_history["decisions"] = (
                best_history["decisions"][-CLIP_HISTORY_FRAMES:]
            )

            hit_count = sum(best_history["decisions"])

            # 5回分の判定が揃い、3回以上陽性なら登録
            confirmed_suspect = (
                len(best_history["decisions"]) == CLIP_HISTORY_FRAMES
                and hit_count >= CLIP_REQUIRED_HITS
            )

            print(
                f"CLIP history: {hit_count}/"
                f"{len(best_history['decisions'])}"
            )

            # ======================
            # 違反者登録
            # ======================
            color = (0, 255, 0)

            if confirmed_suspect:
                color = (0, 0, 255)

                # ここから下は既存の違反者登録処理を残す
                suspect_active = True
                suspect_bbox = current_bbox
                suspect_tracking_start_time = time.time()

                print(">>> SUSPECT REGISTERED")
                print("Suspect BBox:", suspect_bbox)

                # ======================
                # Alarm
                # ======================

                current_time = time.time()

                if (
                    current_time - last_alarm_time
                    > ALARM_INTERVAL_SEC
                ):

                    print(
                        "ALARM: "
                        "Cyclist wearing headphones detected!"
                    )

                    print('\a')

                    last_alarm_time = current_time


                # ----------------------------------
                # この人物をStrongSORTへ登録
                # ----------------------------------

                tracks = update_tracker(
                    tracker,
                    frid_net,
                    frame,
                    suspect_bbox,
                    1.0
                )
                
                
                
                # Re-ID特徴量を含む追跡情報を受け取る
                for track in tracks:
                    local_track_id = track["track_id"]
                    tracked_bbox = track["bbox"]
                    feature = track["feature"]
 
                    # 確認用：GlobalTrackerへ渡す情報
                    print(
                        f"Local ID: {local_track_id}, "
                        f"BBox: {tracked_bbox}, "
                        f"Feature shape: {feature.shape}"
                    )


                for track in tracks:

                    track_id = track["track_id"]

                    tlwh = track["bbox"]

                    tx, ty, tw, th = map(
                        int,
                        tlwh
                    )

                    tx2 = tx + tw
                    ty2 = ty + th

                    print(
                        "StrongSORT ID:",
                        track_id,
                        "BBox:",
                        track["bbox"]
                    )

                    cv2.rectangle(
                        display,
                        (tx, ty),
                        (tx2, ty2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        display,
                        f"StrongSORT ID: {track_id}",
                        (tx, ty2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )


            # ======================
            # YOLOX person BBox
            # ======================

            cv2.rectangle(
                display,
                (px1, py1),
                (px2, py2),
                color,
                2
            )

            cv2.putText(
                display,
                f"{best_text}: {best_score:.2f}",
                (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # ----------------------------------
            # 違反者を登録したら
            # このフレームのperson探索を終了
            # ----------------------------------

            if suspect_active:
                break

    # ==========================================================
    # 違反者登録後
    # → CLIPは使用しない
    # → 登録した人物だけStrongSORTで追跡
    # ==========================================================

    else:
    
        # 5秒経過したらCLIP検出へ戻る
        if (
            RETURN_TO_CLIP
            and suspect_tracking_start_time is not None
            and time.time() - suspect_tracking_start_time
                >= SUSPECT_TRACKING_DURATION_SEC
        ):
            print(">>> Suspect tracking finished")
            print(">>> Returning to CLIP detection")

            suspect_active = False
            suspect_bbox = None
            suspect_tracking_start_time = None
            clip_histories.clear()

        else:

            best_person = None
            best_distance = float("inf")

            # 前回の違反者中心
            sx, sy, sw, sh = suspect_bbox

            suspect_center_x = sx + sw / 2
            suspect_center_y = sy + sh / 2

            for person in persons:

                px1 = int(person.x * frame.shape[1])
                py1 = int(person.y * frame.shape[0])

                px2 = int(
                    (person.x + person.w) * frame.shape[1]
                )

                py2 = int(
                    (person.y + person.h) * frame.shape[0]
                )

                px1 = max(0, px1)
                py1 = max(0, py1)
                px2 = min(frame.shape[1], px2)
                py2 = min(frame.shape[0], py2)

                if px2 <= px1 or py2 <= py1:
                    continue

                # person中心
                person_center_x = (px1 + px2) / 2
                person_center_y = (py1 + py2) / 2

                distance = np.sqrt(
                    (suspect_center_x - person_center_x) ** 2 +
                    (suspect_center_y - person_center_y) ** 2
                )

                if distance < best_distance:
                    best_distance = distance
                    best_person = [
                        px1,
                        py1,
                        px2 - px1,
                        py2 - py1
                    ]


            # ======================================
            # 最も近いpersonをStrongSORTへ
            # ======================================

            if best_person is not None:

                suspect_bbox = best_person

                tracks = update_tracker(
                    tracker,
                    frid_net,
                    frame,
                    suspect_bbox,
                    1.0
                )
                
                for track in tracks:

                    track_id = track["track_id"]

                    tlwh = track["bbox"]

                    tx, ty, tw, th = map(
                        int,
                        tlwh
                    )

                    tx2 = tx + tw
                    ty2 = ty + th

                    print(
                        "StrongSORT ID:",
                        track_id,
                        "BBox:",
                        track["bbox"]
                    )

                    cv2.rectangle(
                        display,
                        (tx, ty),
                        (tx2, ty2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        display,
                        f"StrongSORT ID: {track_id}",
                        (tx, ty2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
    # ======================
    # Show
    # ======================

    cv2.imshow("CLIP_StrongSORT連携", display)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# ======================
# Cleanup
# ======================

cap.release()

cv2.destroyAllWindows()

