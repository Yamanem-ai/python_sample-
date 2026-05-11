import sys
import os
import time
from logging import getLogger

import numpy as np
import cv2

import os
import numpy as np

# import original modules
sys.path.append('../../util')
import webcamera_utils
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/segment-anything-2/'

IMAGE_PATH = 'truck.jpg'
SAVE_IMAGE_PATH = 'output.png'

POINT1 = (500, 375)
POINT2 = (1125, 625)

TARGET_LENGTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Segment Anything 2', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-p', '--pos', action='append', type=int, metavar="X", nargs=2,
    help='Positive coordinate specified by x,y.'
)
parser.add_argument(
    '--neg', action='append', type=int, metavar="X", nargs=2,
    help='Negative coordinate specified by x,y.'
)
parser.add_argument(
    '--box', type=int, metavar="X", nargs=4,
    help='Box coordinate specified by x1,y1,x2,y2.'
)
parser.add_argument(
    '--num_mask_mem', type=int, default=4, choices=(0, 1, 2, 3, 4, 5, 6, 7),
    help='Number of mask mem. (default 1 input frame + 6 previous frames)'
)
parser.add_argument(
    '--max_obj_ptrs_in_encoder', type=int, default=7, choices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    help='Number of obj ptr in encoder.'
)
parser.add_argument(
    '-m', '--model_type', default='hiera_t', choices=('hiera_l', 'hiera_b+', 'hiera_s', 'hiera_t'),
    help='Select model.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--normal', action='store_true',
    help='Use normal version of onnx model. Normal version requires 6 dim matmul.'
)

args = update_parser(parser)

# ======================
# Utility
# ======================

np.random.seed(3)

def show_mask(mask, image):
    color = np.array([30, 144, 255], dtype=np.uint8)

    mask = mask.astype(bool)

    overlay = image.copy()

    overlay[mask] = (
        overlay[mask].astype(np.float32) * 0.5 +
        color.astype(np.float32) * 0.5
    ).astype(np.uint8)

    return overlay

def show_points(coords, labels, img):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    for p in pos_points:
        cv2.drawMarker(
            img, p, (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)
    for p in neg_points:
        cv2.drawMarker(
            img, p, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)

    return img


def show_box(box, img):
    if box is None:
        return img

    cv2.rectangle(
        img, (box[0], box[1]), (box[2], box[3]), color=(2, 118, 2),
        thickness=3,
        lineType=cv2.LINE_4,
        shift=0)

    return img

# ======================
# Logic
# ======================

from sam2_image_predictor import SAM2ImagePredictor
from sam2_video_predictor import SAM2VideoPredictor

# ======================
# Main
# ======================

def get_input_point():
    pos_points = args.pos
    neg_points = args.neg
    box = args.box

    if pos_points is None:
        if neg_points is None and box is None:
            pos_points = [POINT1]
        else:
            pos_points = []
    if neg_points is None:
        neg_points = []

    input_point = []
    input_label = []
    if pos_points:
        for i in range(len(pos_points)):
            input_point.append(pos_points[i])
            input_label.append(1)
    if neg_points:
        for i in range(len(neg_points)):
            input_point.append(neg_points[i])
            input_label.append(0)
    input_point = np.array(input_point)
    input_label = np.array(input_label)
    input_box = None
    if box:
        input_box = np.array(box)
    return input_point, input_label, input_box

def preprocess_frame(img, image_size):
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = (img / 255.0).astype(np.float32)
    img = img - img_mean
    img = img / img_std
    img = np.transpose(img, (2, 0, 1))
    return img

class SAM2Tracker:
    def __init__(self, image_encoder, prompt_encoder, mask_decoder,
                 memory_attention, memory_encoder, mlp, args):

        self.predictor = SAM2VideoPredictor(args.onnx, args.normal, args.benchmark)

        self.inference_state = self.predictor.init_state(
            args.num_mask_mem, args.max_obj_ptrs_in_encoder
        )
        self.predictor.reset_state(self.inference_state)

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.memory_attention = memory_attention
        self.memory_encoder = memory_encoder
        self.mlp = mlp

        self.internal_idx = 0
        self.initialized = False
        self.last_frame = None

    def add_frame(self, frame):
        image = preprocess_frame(frame, 1024)

        self.predictor.append_image(
            self.inference_state,
            image,
            frame.shape[0],
            frame.shape[1],
            self.image_encoder
        )
        self.internal_idx += 1

    def set_points(self, points, labels):
        
        frame_idx = len(self.inference_state["images"]) - 1
        # GUIから呼ぶ用
        self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.internal_idx - 1,
            obj_id=1,
            points=points,
            labels=labels,
            box=None,
            image_encoder=self.image_encoder,
            prompt_encoder=self.prompt_encoder,
            mask_decoder=self.mask_decoder,
            memory_attention=self.memory_attention,
            memory_encoder=self.memory_encoder,
            mlp=self.mlp
        )

        self.predictor.propagate_in_video_preflight(
            self.inference_state,
            image_encoder=self.image_encoder,
            prompt_encoder=self.prompt_encoder,
            mask_decoder=self.mask_decoder,
            memory_attention=self.memory_attention,
            memory_encoder=self.memory_encoder,
            mlp=self.mlp
        )

        self.initialized = True

    def step(self, frame):
        self.add_frame(frame)
        
        if not self.initialized:
            return frame
            
        frame_idx = len(self.inference_state["images"]) - 1
            
        if frame_idx % 5 != 0:
            if self.last_frame is not None:
                return self.last_frame
            
            return frame

        _, out_obj_ids, out_mask_logits = self.predictor.propagate_in_video(
            self.inference_state,
            image_encoder=self.image_encoder,
            prompt_encoder=self.prompt_encoder,
            mask_decoder=self.mask_decoder,
            memory_attention=self.memory_attention,
            memory_encoder=self.memory_encoder,
            mlp=self.mlp,
            frame_idx=frame_idx
        )

        print("logits max:", out_mask_logits[0].max())
        
        mask = (out_mask_logits[0] > 0.0).astype(np.uint8)
        mask = np.squeeze(mask)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask = mask.astype(bool)
        result = show_mask(mask, frame.copy())
        
        print("frame_idx:", frame_idx)
        print("images:", len(self.inference_state["images"]))
        self.last_frame = result
        return result

def annotate_frame(points, labels, box, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp):
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box,
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        mlp=mlp
    )

    predictor.propagate_in_video_preflight(inference_state,
                                                                            image_encoder = image_encoder,
                                                                            prompt_encoder = prompt_encoder,
                                                                            mask_decoder = mask_decoder,
                                                                            memory_attention = memory_attention,
                                                                            memory_encoder = memory_encoder,
                                                                            mlp = mlp)

def process_frame(image, frame_idx, predictor, inference_state, image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp):
    out_frame_idx, out_obj_ids, out_mask_logits = predictor.propagate_in_video(inference_state,
                                                                                image_encoder = image_encoder,
                                                                                prompt_encoder = prompt_encoder,
                                                                                mask_decoder = mask_decoder,
                                                                                memory_attention = memory_attention,
                                                                                memory_encoder = memory_encoder,
                                                                                mlp = mlp,
                                                                                frame_idx = frame_idx)

    image = show_mask((out_mask_logits[0] > 0.0), image, color = np.array([30, 144, 255]), obj_id = out_obj_ids[0])

    return image

def load_model(args):
    # fetch image encoder model
    WEIGHT_IMAGE_ENCODER_L_PATH = 'image_encoder_'+args.model_type+'.onnx'
    MODEL_IMAGE_ENCODER_L_PATH = 'image_encoder_'+args.model_type+'.onnx.prototxt'

    WEIGHT_PROMPT_ENCODER_L_PATH = 'prompt_encoder_'+args.model_type+'.onnx'
    MODEL_PROMPT_ENCODER_L_PATH = 'prompt_encoder_'+args.model_type+'.onnx.prototxt'

    WEIGHT_MASK_DECODER_L_PATH = 'mask_decoder_'+args.model_type+'.onnx'
    MODEL_MASK_DECODER_L_PATH = 'mask_decoder_'+args.model_type+'.onnx.prototxt'

    if args.normal:
        WEIGHT_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+args.model_type+'.onnx'
        MODEL_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+args.model_type+'.onnx.prototxt'
    else:
        WEIGHT_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+args.model_type+'.opt.onnx'
        MODEL_MEMORY_ATTENTION_L_PATH = 'memory_attention_'+args.model_type+'.opt.onnx.prototxt'

    WEIGHT_MEMORY_ENCODER_L_PATH = 'memory_encoder_'+args.model_type+'.onnx'
    MODEL_MEMORY_ENCODER_L_PATH = 'memory_encoder_'+args.model_type+'.onnx.prototxt'

    WEIGHT_MLP_L_PATH = 'mlp_'+args.model_type+'.onnx'
    MODEL_MLP_L_PATH = 'mlp_'+args.model_type+'.onnx.prototxt'

    # download
    check_and_download_models(WEIGHT_IMAGE_ENCODER_L_PATH, MODEL_IMAGE_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_PROMPT_ENCODER_L_PATH, MODEL_PROMPT_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MASK_DECODER_L_PATH, MODEL_MASK_DECODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MEMORY_ATTENTION_L_PATH, MODEL_MEMORY_ATTENTION_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MEMORY_ENCODER_L_PATH, MODEL_MEMORY_ENCODER_L_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_MLP_L_PATH, MODEL_MLP_L_PATH, REMOTE_PATH)
    
    if args.onnx:
        import onnxruntime
        image_encoder = onnxruntime.InferenceSession(WEIGHT_IMAGE_ENCODER_L_PATH)
        prompt_encoder = onnxruntime.InferenceSession(WEIGHT_PROMPT_ENCODER_L_PATH)
        mask_decoder = onnxruntime.InferenceSession(WEIGHT_MASK_DECODER_L_PATH)
        memory_attention = onnxruntime.InferenceSession(WEIGHT_MEMORY_ATTENTION_L_PATH)
        memory_encoder = onnxruntime.InferenceSession(WEIGHT_MEMORY_ENCODER_L_PATH)
        mlp = onnxruntime.InferenceSession(WEIGHT_MLP_L_PATH)
    else:
        import ailia
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True
        )

        image_encoder = ailia.Net(weight=WEIGHT_IMAGE_ENCODER_L_PATH, stream=MODEL_IMAGE_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        prompt_encoder = ailia.Net(weight=WEIGHT_PROMPT_ENCODER_L_PATH, stream=MODEL_PROMPT_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        mask_decoder = ailia.Net(weight=WEIGHT_MASK_DECODER_L_PATH, stream=MODEL_MASK_DECODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        memory_attention = ailia.Net(weight=WEIGHT_MEMORY_ATTENTION_L_PATH, stream=MODEL_MEMORY_ATTENTION_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        memory_encoder = ailia.Net(weight=WEIGHT_MEMORY_ENCODER_L_PATH, stream=MODEL_MEMORY_ENCODER_L_PATH, memory_mode=memory_mode, env_id=args.env_id)
        mlp = ailia.Net(weight=WEIGHT_MLP_L_PATH, stream=MODEL_MLP_L_PATH, memory_mode=memory_mode, env_id=args.env_id)

    return image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp

def main():
    image_encoder, prompt_encoder, mask_decoder, memory_attention, memory_encoder, mlp = load_model(args)

    if args.video is not None:
        capture = webcamera_utils.get_capture(args.video)

        tracker = SAM2Tracker(
            image_encoder,
            prompt_encoder,
            mask_decoder,
            memory_attention,
            memory_encoder,
            mlp,
            args
        )

        print("Clickで点を指定して 's' で開始")

        # 仮：固定点（あとでGUIに置き換え）
        points = np.array([[500, 375]])
        labels = np.array([1])


        first = True

        while True:
            ret, frame = capture.read()
            if not ret:
                break
                
            #tracker.add_frame(frame)
            
            if first:
                tracker.add_frame(frame)
                tracker.set_points(points, labels)
                first = False
                print("initialized!")
                
                cv2.imshow("frame", frame)
                continue
                
            frame = tracker.step(frame)
                
            cv2.imshow("frame", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    else:
        recognize_from_image(image_encoder, prompt_encoder, mask_decoder)

    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()
