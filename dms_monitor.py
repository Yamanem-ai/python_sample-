# dms_monitor.py
#
# Simple Driver Monitoring System sample
#
# Usage:
#   python3 dms_monitor.py
#
# Input:
#   edit VIDEO_PATH

import cv2
import facemesh_v2_ex

# =========================================================
# Settings
# =========================================================

VIDEO_PATH = "Driver.mp4"

EYE_CLOSE_THRESHOLD = 0.5
YAWN_THRESHOLD = 0.50

DROWSY_FRAMES = 30
FAINT_FRAMES = 60

# =========================================================
# Initialize
# =========================================================

models = facemesh_v2_ex.initialize_models()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("video open error")
    exit()

# =========================================================
# State
# =========================================================

eye_closed_count = 0
faint_count = 0

# =========================================================
# Main loop
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------------------------------
    # Process frame
    # -----------------------------------------------------

    result = facemesh_v2_ex.process_frame(
        models,
        frame
    )

    if result is not None:

        landmarks = result["landmarks"]
        score = result["blendshape"]

        # draw landmarks
        facemesh_v2_ex.draw_result(
            frame,
            landmarks
        )

        # -------------------------------------------------
        # Blink
        # -------------------------------------------------

        blink = (
            score["eyeBlinkLeft"] +
            score["eyeBlinkRight"]
        ) / 2.0

        if blink > EYE_CLOSE_THRESHOLD:
            eye_closed_count += 1
        else:
            eye_closed_count = 0

        drowsy = eye_closed_count > DROWSY_FRAMES

        # -------------------------------------------------
        # Yawn
        # -------------------------------------------------

        jaw_open = score["jawOpen"]

        # -------------------------------------------------
        # Simple faint detection
        # -------------------------------------------------

        if (
            jaw_open > 0.6 and
            blink < 0.2
        ):
            faint_count += 1
        else:
            faint_count = 0

        fainting = faint_count > FAINT_FRAMES

        # -------------------------------------------------
        # Info
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Blink : {blink:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"JawOpen : {jaw_open:.2f}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # -------------------------------------------------
        # Alerts
        # -------------------------------------------------

        if drowsy:

            cv2.putText(
                frame,
                "DROWSINESS ALERT",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        if jaw_open > YAWN_THRESHOLD:

            cv2.putText(
                frame,
                "YAWNING",
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3
            )

        if fainting:

            cv2.putText(
                frame,
                "FAINTING ALERT",
                (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

    # -----------------------------------------------------
    # Show
    # -----------------------------------------------------

    cv2.imshow(
        "DMS Monitor",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# =========================================================
# End
# =========================================================

cap.release()
cv2.destroyAllWindows()
