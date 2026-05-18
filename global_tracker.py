import numpy as np
import time

# ======================
# パラメータ
# ======================
COSINE_THRESHOLD = 0.96
DIST_THRESHOLD = 2.0
TIME_THRESHOLD = 3.0

# ======================
# ユーティリティ
# ======================
def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))

def world_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# ======================
# ワールド変換（仮）
# ======================
CAMERA_PARAM = {
    "A": {"scale": 0.01, "offset": np.array([0.0, 0.0])},
    "B": {"scale": 0.01, "offset": np.array([3.0, 0.5])},
}

def image_to_world(camera_id, bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = y2

    p = CAMERA_PARAM.get(camera_id, {"scale": 0.01, "offset": np.array([0, 0])})
    return np.array([cx * p["scale"], cy * p["scale"]]) + p["offset"]

# ======================
# グローバルトラッカー
# ======================
class GlobalTracker:
    def __init__(self):
        self.global_id_count = 0
        self.tracks = {}
        self.local_to_global = {}

    def _new_id(self):
        self.global_id_count += 1
        return self.global_id_count

    def update(self, det):
        if det["feature"] is None:
            return None, 0.0

        feat = det["feature"]
        feat = feat / (np.linalg.norm(feat) + 1e-6)

        world_pos = image_to_world(det["camera_id"], det["bbox"])
        t = det["timestamp"]
        local_key = (str(det["camera_id"]), int(det["track_id"]))
        # 古いlocal track削除
        expired = []

        for key, gid in self.local_to_global.items():

            if gid not in self.tracks:
                expired.append(key)
                continue

            dt = t - self.tracks[gid]["last_time"]

            if dt > TIME_THRESHOLD:
                expired.append(key)

        for key in expired:
            del self.local_to_global[key]

        # ======================
        # ① ローカル固定ID
        # ======================
        if local_key in self.local_to_global:
            gid = self.local_to_global[local_key]
            
            if gid in self.tracks:
                sim = cosine_similarity(
                    feat,
                    self.tracks[gid]["feature"]
                )
                if sim >= COSINE_THRESHOLD:
                    self._update_track(
                        gid,
                        world_pos,
                        feat,
                        t,
                        det["camera_id"]
                    )
                    return gid
                
            del self.local_to_global[local_key]
                                        
        # ======================
        # ② グローバル探索（単純cosine）
        # ======================
        best_id = None
        best_score = -1.0

        for gid, info in self.tracks.items():
            
            #if info["camera_id"] == det["camera_id"]:
            #    continue


            sim = cosine_similarity(feat, info["feature"])

            print(f"COMPARE: {det['camera_id']}:{det['track_id']} vs G{gid} = {sim:.3f}")

            if sim > best_score:
                best_score = sim
                best_id = gid

        # ======================
        # ③ マッチ判定
        # ======================
        if (
            best_id is not None
            and best_score >= COSINE_THRESHOLD
        ):

            self._update_track(
                best_id,
                world_pos,
                feat,
                t,
                det["camera_id"]
            )

            self.local_to_global[local_key] = best_id

            return best_id
        
        # ======================
        # ④ 新規ID
        # ======================
        new_id = self._new_id()
        self.local_to_global[local_key] = new_id

        self.tracks[new_id] = {
            "feature": feat,
            "feature_history": [feat],
            "last_position": world_pos,
            "last_time": t,
            "camera_id": det["camera_id"]
        }

        return new_id

    def _update_track(self, gid, pos, feat, t, camera_id):
        info = self.tracks[gid]

        alpha = 0.9

        info["feature"] = (
            alpha * info["feature"]
            + (1 - alpha) * feat
        )

        info["feature"] /= (
            np.linalg.norm(info["feature"]) + 1e-6
        )

        info["camera_id"] = camera_id

        info["last_position"] = pos
        info["last_time"] = t

# ======================
# API
# ======================
shared_global_id_map = None

global_tracker = GlobalTracker()

def run_tracker(tracking_queue, display_queue, shared_map):

    global shared_global_id_map
    shared_global_id_map = shared_map

    while True:

        det = tracking_queue.get()

        if det is None:
            break

        gid = global_tracker.update(det)

        print(
            f"[Global] Camera {det['camera_id']} "
            f"Local {det['track_id']} -> Global {gid}"
        )
        
        shared_global_id_map[
            (str(det["camera_id"]), int(det["track_id"]))
        ] = gid
        
        display_queue.put({
            "camera_id": det["camera_id"],
            "track_id": det["track_id"],
            "global_id": gid
        })

    return gid

# ======================
# TEST
# ======================
if __name__ == "__main__":
    f1 = np.random.rand(128)
    f2 = f1 + np.random.normal(0, 0.01, 128)

    detections = [
        {"camera_id": "A", "track_id": 1, "bbox": [100, 100, 150, 200], "feature": f1, "timestamp": 0},
        {"camera_id": "B", "track_id": 3, "bbox": [120, 110, 170, 210], "feature": f2, "timestamp": 1},
    ]

    for d in detections:
        gid = global_tracker.update(d)
        print(f"{d['camera_id']} → G{gid}")
