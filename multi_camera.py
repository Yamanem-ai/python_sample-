from multiprocessing import Process, Queue, Manager
import cv2
import numpy as np
from queue import Empty

import strongsort_ex_A
import strongsort_ex_B
import global_tracker
#import threading
#import queue

#frame_queue = queue.Queue()

if __name__ == '__main__':

    tracking_queue = Queue()
    display_queue = Queue()
    
    manager = Manager()
    shared_global_id_map = manager.dict()
    
    global_process = Process(
        target=global_tracker.run_tracker,
        args=(tracking_queue, display_queue, shared_global_id_map)
    )
    
    global_process.start()
                             
#t1 = threading.Thread(
    p1 = Process(
        target=strongsort_ex_A.recognize_from_video,
        args=("A", "camera_A2.mp4", tracking_queue, shared_global_id_map)
    )

#t2 = threading.Thread(
    p2 = Process(
        target=strongsort_ex_B.recognize_from_video,
        args=("B", "camera_B2.mp4", tracking_queue, shared_global_id_map)
    )

    p1.start()
    p2.start()
    
    latest_tracks = {}
    
    while True:

        try:

            while True:
            
                msg = display_queue.get(timeout=0.01)
            
                print(msg)

                gid = msg["global_id"]

                latest_tracks[gid] = (
                    msg["camera_id"],
                    msg["track_id"]
                )
        
        except Empty:
            pass
    
        global_vis = np.zeros((600, 800, 3), dtype=np.uint8)

        y = 40
        
        for gid, (cam, tid) in latest_tracks.items():
        
            text = f"Global {gid} :Camera {cam} Local {tid}"

            cv2.putText(
                global_vis,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
    
            y += 35

        cv2.imshow("Global IDs", global_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    p1.join()
    p2.join()

    tracking_queue.put(None)
    
    global_process.join()
    
    cv2.destroyAllWindows()
