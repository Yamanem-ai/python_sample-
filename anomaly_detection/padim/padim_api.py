# padim_api.py

import sys
sys.path.append('../../util')
import pickle
import logging
import numpy as np
import cv2

import ailia

from model_utils import check_and_download_models

from padim_utils import (
    get_params,
    training,
    infer,
    preprocess,
)

logger = logging.getLogger(__name__)

# ======================================
# Constants
# ======================================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/padim/'

IMAGE_RESIZE = 256
IMAGE_SIZE = 224
KEEP_ASPECT = True


# ======================================
# Padim Engine
# ======================================

class PadimEngine:

    def __init__(
        self,
        arch="resnet18",
        env_id=0,
        batch_size=32,
        seed=1024,
        aug=False,
        aug_num=5,
    ):

        self.arch = arch
        self.env_id = env_id
        self.batch_size = batch_size
        self.seed = seed
        self.aug = aug
        self.aug_num = aug_num

        self.net = None
        self.params = None
        self.train_outputs = None

    # ======================================
    # Initialize: def main():からコピー
    # ======================================

    def initialize(self):

        weight_path, model_path, params = get_params(self.arch)

        logger.info("Checking model files...")

        check_and_download_models(
            weight_path,
            model_path,
            REMOTE_PATH
        )

        logger.info("Creating ailia.Net...")

        self.net = ailia.Net(
            model_path,
            weight_path,
            env_id=self.env_id
        )

        self.params = params

        logger.info("PaDiM initialized")

    # ======================================
    # Train: def train_from_image_or_videoからコピー
    # ======================================

    def train(self, train_dir):

        if self.net is None:
            raise RuntimeError(
                "initialize() must be called first."
            )

        logger.info(
            f"Training from : {train_dir}"
        )

        self.train_outputs = training(
            self.net,
            self.params,
            IMAGE_RESIZE,
            IMAGE_SIZE,
            KEEP_ASPECT,
            self.batch_size,
            train_dir,
            self.aug,
            self.aug_num,
            self.seed,
            logger
        )

        logger.info("Training completed")

    # ======================================
    # Save: def train_from_image_or_video()の下の方
    # ======================================

    def save(self, pkl_path):

        if self.train_outputs is None:
            raise RuntimeError(
                "No training data available."
            )

        with open(pkl_path, "wb") as f:
            pickle.dump(self.train_outputs, f)

        logger.info(
            f"Saved training feature : {pkl_path}"
        )

    # ======================================
    # Load: def train_and_infer()からコピー
    # ======================================

    def load(self, pkl_path):

        with open(pkl_path, "rb") as f:
            self.train_outputs = pickle.load(f)

        logger.info(
            f"Loaded training feature : {pkl_path}"
        )

    # ======================================
    # Infer OpenCV Frame: def infer_from_video()からコピー
    # ======================================

    def infer_frame(self, frame_bgr):

        if self.train_outputs is None:
            raise RuntimeError(
                "train() or load() must be called first."
            )

        img_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        img = preprocess(
            img_rgb,
            IMAGE_RESIZE,
            IMAGE_SIZE,
            keep_aspect=KEEP_ASPECT
        )

        print("padim =", img.shape)
        
        score_map = infer(
            self.net,
            self.params,
            self.train_outputs,
            img,
            IMAGE_SIZE
        )

        return score_map

    # ======================================
    # Infer Image File: これは単なるラッパー
    # ======================================

    def infer_image(self, image_path):

        frame = cv2.imread(image_path)
        
        if frame is None:
            raise FileNotFoundError(image_path)

        return self.infer_frame(frame)

    # ======================================
    # Score
    # ======================================

    def anomaly_score(self, score_map):

        return float(np.max(score_map))

    # ======================================
    # Alarm
    # ======================================

    def is_anomaly(
        self,
        score_map,
        threshold=0.5
    ):

        score = self.anomaly_score(score_map)

        return score > threshold

    # ======================================
    # Convenience
    # ======================================

    def infer_and_score(
        self,
        frame_bgr
    ):

        score_map = self.infer_frame(frame_bgr)

        score = self.anomaly_score(score_map)

        return score, score_map
