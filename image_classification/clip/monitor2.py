import cv2
import numpy as np
import time

from yolox_ex import (
    create_detector,
    detect_frame,
)

from clip_ex import (
    create_clip,
    create_text_features,
    predict_clip,
)

# ======================
# Settings
# ======================

YOLO_ENV_ID = 0
CLIP_ENV_ID = 0

ALARM_THRESHOLD = 0.50
ALARM_INTERVAL_SEC = 3.0

HEAD_TEXT_INPUTS = [
    "cyclist wearing headphones",
    "cyclist wearing hat",
]

HAND_TEXT_INPUTS = [
    "cyclist holding a cell phone",
    "cyclist holding nothing",
]

#TARGET_CLASSES = None
TARGET_CLASSES = [
    "person",
    "bicycle",
]

CAMERA_ID = 1

# ======================
# Initialize
# ======================

print("Initializing YOLOX...")
detector = create_detector(env_id=YOLO_ENV_ID)

print("Initializing CLIP...")
net_image, net_text = create_clip(
    model_type='ViTB32',
    env_id=CLIP_ENV_ID
)

print("Creating text features...")

head_text_feature = create_text_features(
    net_text,
    HEAD_TEXT_INPUTS,
)
hand_text_feature = create_text_features(
    net_text,
    HAND_TEXT_INPUTS,
)

# ======================
# Camera
# ======================

cap = cv2.VideoCapture("phone2.MP4")

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

last_alarm_time = 0

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

    # ======================
    # person × bicycle proximity
    # ======================

    for person in persons:

        px1 = int(person.x * frame.shape[1])
        py1 = int(person.y * frame.shape[0])
        px2 = int((person.x + person.w) * frame.shape[1])
        py2 = int((person.y + person.h) * frame.shape[0])

        # clamp
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(frame.shape[1], px2)
        py2 = min(frame.shape[0], py2)

        if px2 <= px1 or py2 <= py1:
            continue

        # nearby bicycle check
        is_cyclist = False

        person_center_x = (px1 + px2) / 2
        person_center_y = (py1 + py2) / 2

        for bicycle in bicycles:
            bx1 = int(bicycle.x * frame.shape[1])
            by1 = int(bicycle.y * frame.shape[0])
            bx2 = int((bicycle.x + bicycle.w) * frame.shape[1])
            by2 = int((bicycle.y + bicycle.h) * frame.shape[0])

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
        # Head Crop
        # ======================
        
        head_y2 = py1 + int((py2 - py1) * 0.20)
        
        head_x1 = px1 + int((px2 - px1) * 0.35)
        head_x2 = px2 - int((px2 - px1) * 0.20)
        
        head_crop = frame[py1:head_y2, head_x1:head_x2]

        # ======================
        # Hand Crop
        # ======================

        hand_y1 = py1 + int((py2 - py1) * 0.30)
        hand_y2 = py1 + int((py2 - py1) * 0.60)
        
        hand_x1 = px1 + int((px2 - px1) * 0.30)
        hand_x2 = px2
        
        hand_crop = frame[hand_y1:hand_y2, hand_x1:hand_x2]

        if head_crop.size != 0:
            cv2.imshow("head_crop", head_crop)
            cv2.moveWindow("head_crop", 100, 0)

            head_result = predict_clip(
                net_image,
                head_crop,
                head_text_feature,
                HEAD_TEXT_INPUTS
            )
            
            head_best_text = head_result['best_text']
            head_best_score = head_result['best_score']
            
            print("Head:", head_best_text, head_best_score)
            
        if hand_crop.size != 0:
            cv2.imshow("hand_crop", hand_crop)
            cv2.moveWindow("hand_crop", 500, 0)
            
            hand_result = predict_clip(
                net_image,
                hand_crop,
                hand_text_feature,
                HAND_TEXT_INPUTS
            )

            hand_best_text = hand_result['best_text']
            hand_best_score = hand_result['best_score']

            print("Hand:", hand_best_text, hand_best_score)

        # ======================
        # Visualization
        # ======================
        head_alarm = (
            head_result["best_score"] >= ALARM_THRESHOLD
            and head_result["best_text"] == "cyclist wearing headphones"
        )
        
        hand_alarm = (
            hand_result["best_score"] >= ALARM_THRESHOLD
            and hand_result["best_text"] == "cyclist holding a cell phone"
        )
        
        color = (0, 0, 255) if (head_alarm or hand_alarm) else (0, 255, 0)


        current_time = time.time()

        if head_alarm:
            if current_time - last_alarm_time > ALARM_INTERVAL_SEC:
                print("ALARM: Cyclist wearing headphones detected!")
                print('\a')
                last_alarm_time = current_time
        elif hand_alarm:
            if current_time - last_alarm_time > ALARM_INTERVAL_SEC:
                print("ALARM: Cyclist holding a cell phone detected!")
                print('\a')
                last_alarm_time = current_time
                
        cv2.rectangle(
            display,
            (px1, py1),
            (px2, py2),
            color,
            2
        )

        cv2.putText(
            display,
            f"Head: {head_result['best_text']} "
            f"{head_result['best_score']:.2f}",
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        cv2.putText(
            display,
            f"Hand: {hand_result['best_text']} "
            f"{hand_result['best_score']:.2f}",
            (px1, py2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ======================
    # Show
    # ======================

    cv2.imshow("monitor", display)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ======================
# Cleanup
# ======================

cap.release()
cv2.destroyAllWindows()
