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

import numpy as np
import math

# =========================================================
# Settings
# =========================================================

VIDEO_PATH = "Driver.mp4"

EYE_CLOSE_THRESHOLD = 0.5
YAWN_THRESHOLD = 0.50

DROWSY_FRAMES = 30
FAINT_FRAMES = 30

HEAD_DOWN_THRESHOLD = -40
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

#==========================================================
# Head Pose 関数追加
#==========================================================
def estimate_head_pose(landmarks, frame_shape):

    h, w = frame_shape[:2]

    # ----------------------------------------------------
    # FaceMesh landmarks -> image points
    # ----------------------------------------------------

    image_points = np.array([
        landmarks[1][:2],     # nose
        landmarks[33][:2],    # left eye
        landmarks[263][:2],   # right eye
        landmarks[61][:2],    # mouth left
        landmarks[291][:2],   # mouth right
        landmarks[199][:2],   # chin
    ], dtype=np.float32)

    image_points[:,0] *= w
    image_points[:,1] *= h

    # ----------------------------------------------------
    # Simple 3D face model
    # ----------------------------------------------------

    model_points = np.array([
        [0.0,   0.0,    0.0],     # nose
        [-30.0, -30.0, -30.0],    # left eye
        [30.0,  -30.0, -30.0],    # right eye
        [-25.0, 30.0,  -20.0],    # mouth left
        [25.0,  30.0,  -20.0],    # mouth right
        [0.0,   65.0,  -5.0],     # chin
    ], dtype=np.float32)

    # ----------------------------------------------------
    # Camera matrix
    # ----------------------------------------------------

    focal_length = w

    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4,1))

    # ----------------------------------------------------
    # solvePnP
    # ----------------------------------------------------

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0, 0, 0

    # ----------------------------------------------------
    # Rotation matrix
    # ----------------------------------------------------

    rmat, _ = cv2.Rodrigues(rvec)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0]
    yaw   = angles[1]
    roll  = angles[2]

    return pitch, yaw, roll
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
        
        #--------------------------------------------------
        # Head pose 追加分
        #--------------------------------------------------
        pitch, yaw, roll = estimate_head_pose(
            landmarks,
            frame.shape
        )
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
        
        cv2.putText(
            frame,
            f"Pitch : {pitch:.1f}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
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
            
        if pitch < HEAD_DOWN_THRESHOLD:

            cv2.putText(
                frame,
                "HEAD DOWN",
                (30, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 255),
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
