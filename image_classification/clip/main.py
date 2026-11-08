import multiprocessing as mp
import queue
import time

import cv2
import numpy as np

from global_tracker import run_tracker
from monitor4 import run_monitor


def main():
    mp.set_start_method("spawn", force=True)

    tracking_queue = mp.Queue(maxsize=1000)
    display_queue = mp.Queue(maxsize=1000)
    frame_queue = mp.Queue(maxsize=4)

    manager = mp.Manager()
    shared_map = manager.dict()

    global_process = mp.Process(
        target=run_tracker,
        args=(tracking_queue, display_queue, shared_map),
        name="GlobalTracker",
    )
    global_process.start()

    cameras = [
        ("A", "headphone2.mov"),
        ("B", "headphone3.mov"),
    ]

    
    monitor_processes = []

    for camera_id, source in cameras:
        p = mp.Process(
            target=run_monitor,
            args=(camera_id, tracking_queue, source, frame_queue),
            name=f"Monitor-{camera_id}",
        )
        p.start()
        monitor_processes.append(p)


    global_ids = {}
    latest_frames = {}
    windows = {}
    
    # GUIはmain.pyだけで管理
    for camera_id, source in cameras:
        for view in ("Local", "Global"):
            name = f"Camera {camera_id} - {view}"
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 800, 400)

            if camera_id == "A":
                x = 20 if view == "Local" else 1000
                y = 40
            else:
                x = 20 if view == "Local" else 1000
                y = 500

            cv2.moveWindow(name, x, y)
            windows[(camera_id, view)] = name

    try:
        while any(p.is_alive() for p in monitor_processes):
            # Global IDの割り当て結果を受信
            while True:
                try:
                    result = display_queue.get_nowait()
                except queue.Empty:
                    break

                key = (
                    str(result["camera_id"]),
                    int(result["track_id"]),
                )
                global_ids[key] = result["global_id"]

            # 最新のカメラフレームを取得
            while True:
                try:
                    packet = frame_queue.get_nowait()
                except queue.Empty:
                    break

                latest_frames[str(packet["camera_id"])] = packet

            # カメラA・Bを表示
            for camera_id, source in cameras:
                packet = latest_frames.get(camera_id)

                if packet is None:
                    continue

                # Local画面
                local_image = cv2.resize(
                    packet["local_frame"], (800, 400)
                )
                cv2.imshow(
                    windows[(camera_id, "Local")],
                    local_image,
                )

                # Global画面
                global_image = packet["frame"].copy()

                for track in packet["tracks"]:
                    local_id = int(track["track_id"])
                    x, y, w, h = map(int, track["bbox"])
                    gid = global_ids.get((camera_id, local_id))

                    if gid is None:
                        label = f"Local {local_id} / Global pending"
                        color = (0, 200, 255)
                    else:
                        label = f"Global ID: {gid}"
                        color = (0, 0, 255)

                    cv2.rectangle(
                        global_image,
                        (x, y),
                        (x + w, y + h),
                        color,
                        2,
                    )
                    cv2.putText(
                        global_image,
                        label,
                        (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                global_image = cv2.resize(
                    global_image, (800, 400)
                )
                cv2.imshow(
                    windows[(camera_id, "Global")],
                    global_image,
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.005)

    finally:
        for p in monitor_processes:
            if p.is_alive():
                p.terminate()
            p.join()

        # Global Trackerへ終了指示
        tracking_queue.put(None)
        global_process.join(timeout=5)

        if global_process.is_alive():
            global_process.terminate()
            global_process.join()

        cv2.destroyAllWindows()
        manager.shutdown()

        tracking_queue.close()
        display_queue.close()
        frame_queue.close()

if __name__ == "__main__":
    main()
