import sys
import time
from typing import List
import random

# logger
from logging import getLogger  # noqa

import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.special import log_softmax
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import ailia

import os
import json
import shutil

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa

from beam_search import BeamSearchScorer
from logit_process import logits_processor
from processing_florence2 import post_process_generation


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_EMB_BASE_PATH = "embeddings_base.onnx"
WEIGHT_IMG_BASE_PATH = "feature_base.onnx"
WEIGHT_ENC_BASE_PATH = "encoder_base.onnx"
WEIGHT_DEC_BASE_PATH = "decoder_base.onnx"
WEIGHT_EMB_LARGE_PATH = "embeddings_large.onnx"
WEIGHT_IMG_LARGE_PATH = "feature_large.onnx"
WEIGHT_ENC_LARGE_PATH = "encoder_large.onnx"
WEIGHT_DEC_LARGE_PATH = "decoder_large.onnx"
MODEL_EMB_BASE_PATH = "embeddings_base.onnx.prototxt"
MODEL_IMG_BASE_PATH = "feature_base.onnx.prototxt"
MODEL_ENC_BASE_PATH = "encoder_base.onnx.prototxt"
MODEL_DEC_BASE_PATH = "decoder_base.onnx.prototxt"
MODEL_EMB_LARGE_PATH = "embeddings_large.onnx.prototxt"
MODEL_IMG_LARGE_PATH = "feature_large.onnx.prototxt"
MODEL_ENC_LARGE_PATH = "encoder_large.onnx.prototxt"
MODEL_DEC_LARGE_PATH = "decoder_large.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/florence2/"

IMAGE_PATH = "car.jpg"
SAVE_IMAGE_PATH = "output.png"

IMG_SIZE = 768

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Florence-2", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-p",
    "--prompt",
    choices=[
        "CAPTION",
        "DETAILED_CAPTION",
        "MORE_DETAILED_CAPTION",
        "CAPTION_TO_PHRASE_GROUNDING",
        "OD",
        "DENSE_REGION_CAPTION",
        "REGION_PROPOSAL",
        "OCR",
        "OCR_WITH_REGION",
    ],
    default="CAPTION",
    help="prompt",
)
parser.add_argument(
    "--text_input",
    type=str,
    default=None,
    help="TEXT_INPUT (use by CAPTION_TO_PHRASE_GROUNDING)",
)
parser.add_argument(
    "--model_type", choices=["base", "large"], default="base", help="model type"
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================

colormap = [
    "blue",
    "orange",
    "green",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "red",
    "lime",
    "indigo",
    "violet",
    "aqua",
    "magenta",
    "coral",
    "gold",
    "tan",
    "skyblue",
]


def save_coco_image(image, bbox, labels, savepath):
    """ 
    Florence-2のbboxを描画した画像を保存 元画像と同じ解像度で保存する。 
    """
    image_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(image_pil)
    
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(v) for v  in box]
        
        # bbox
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=5,
        )
        
        # label
        draw.text(
            (x1 + 5, y1 + 5),
            label,
            fill="red",
        )
        
    image_pil.save(savepath, quality=95)
    
def crop_image_4(image):
    """
    画像を4分割する。

    中央部分に10%程度のオーバーラップを設ける。

    3024 x 3024の場合:

        通常:
            1512 x 1512

        オーバーラップ:
            約302 px

        実際のCrop:
            約1663 x 1663

    戻り値:
        [
            (crop_image, offset_x, offset_y),
            ...
        ]
    """

    height, width = image.shape[:2]

    # --------------------------------------------------
    # 中央位置
    # --------------------------------------------------

    mid_x = width // 2
    mid_y = height // 2

    # --------------------------------------------------
    # 10% overlap
    #
    # 「元画像の幅・高さの10%」を
    # 2つのCrop間の重複幅にする。
    #
    # そのため、中央境界から左右に5%ずつ広げる。
    # --------------------------------------------------

    overlap_x = int(width * 0.10)
    overlap_y = int(height * 0.10)

    expand_x = overlap_x // 2
    expand_y = overlap_y // 2

    # --------------------------------------------------
    # Crop coordinates
    # --------------------------------------------------

    x1_left = 0
    x2_left = min(
        width,
        mid_x + expand_x,
    )

    x1_right = max(
        0,
        mid_x - expand_x,
    )
    x2_right = width

    y1_top = 0
    y2_top = min(
        height,
        mid_y + expand_y,
    )

    y1_bottom = max(
        0,
        mid_y - expand_y,
    )
    y2_bottom = height

    # --------------------------------------------------
    # Create crops
    # --------------------------------------------------

    crops = [

        # ----------------------------------------------
        # 左上
        # ----------------------------------------------
        (
            image[
                y1_top:y2_top,
                x1_left:x2_left
            ].copy(),

            x1_left,
            y1_top,
        ),

        # ----------------------------------------------
        # 右上
        # ----------------------------------------------
        (
            image[
                y1_top:y2_top,
                x1_right:x2_right
            ].copy(),

            x1_right,
            y1_top,
        ),

        # ----------------------------------------------
        # 左下
        # ----------------------------------------------
        (
            image[
                y1_bottom:y2_bottom,
                x1_left:x2_left
            ].copy(),

            x1_left,
            y1_bottom,
        ),

        # ----------------------------------------------
        # 右下
        # ----------------------------------------------
        (
            image[
                y1_bottom:y2_bottom,
                x1_right:x2_right
            ].copy(),

            x1_right,
            y1_bottom,
        ),
    ]

    # --------------------------------------------------
    # Debug information
    # --------------------------------------------------

    print(
        "  4-Crop overlap:",
        overlap_x,
        "x",
        overlap_y,
        "pixels",
    )

    for i, (
        crop,
        offset_x,
        offset_y,
    ) in enumerate(crops):

        crop_height, crop_width = crop.shape[:2]

        print(
            "    Crop %d: %dx%d, offset=(%d,%d)"
            % (
                i + 1,
                crop_width,
                crop_height,
                offset_x,
                offset_y,
            )
        )

    return crops

def is_valid_crop_bbox(
    bbox,
    crop_width,
    crop_height,
):
    """
    Crop内のBBoxが閉じた傘として妥当か確認する。
    """

    x1, y1, x2, y2 = bbox

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    if bbox_width <= 0 or bbox_height <= 0:
        return False

    bbox_area = bbox_width * bbox_height
    crop_area = crop_width * crop_height

    # ------------------------------------------
    # 1. Crop全体を覆う異常なBBoxを除外
    # ------------------------------------------

    area_ratio = bbox_area / crop_area

    if area_ratio >= 0.60:
        return False

    # ------------------------------------------
    # 2. Cropのほぼ全幅・全高のBBoxを除外
    # ------------------------------------------

    if bbox_width / crop_width >= 0.90:
        return False

    if bbox_height / crop_height >= 0.90:
        return False

    # ------------------------------------------
    # 3. 横長すぎる物体を除外
    # ------------------------------------------

    aspect_ratio = bbox_height / bbox_width

    if aspect_ratio < 2.5:
        return False

    return True
def restore_bbox(bbox, offset_x, offset_y):
    """
    Crop画像上のBBoxを元画像座標へ戻す。

    Florence-2:
        [x1, y1, x2, y2]

    戻り値:
        [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = bbox

    return [
        x1 + offset_x,
        y1 + offset_y,
        x2 + offset_x,
        y2 + offset_y,
    ]

def calculate_iou(box1, box2):
    """
    IoUを計算する。

    box:
        [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(
        0.0,
        x2 - x1,
    )

    inter_height = max(
        0.0,
        y2 - y1,
    )

    intersection = (
        inter_width *
        inter_height
    )

    area1 = (
        max(0.0, box1[2] - box1[0]) *
        max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0]) *
        max(0.0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def remove_duplicate_bboxes(bboxes, labels, iou_threshold=0.5):
    """
    重複BBoxを除去する。

    Florence-2から返された順番を維持する。
    """

    if len(bboxes) == 0:
        return [], []

    kept_bboxes = []
    kept_labels = []

    for bbox, label in zip(bboxes, labels):

        duplicate = False

        for kept_bbox in kept_bboxes:

            iou = calculate_iou(
                bbox,
                kept_bbox,
            )

            if iou >= iou_threshold:

                duplicate = True
                break

        if not duplicate:

            kept_bboxes.append(bbox)
            kept_labels.append(label)

    return kept_bboxes, kept_labels
    
def bbox_to_coco(bbox):
    """
    Florence-2:
        [x1, y1, x2, y2]

    COCO:
        [x, y, width, height]
    """
    x1, y1, x2, y2 = bbox

    return [
        float(x1),
        float(y1),
        float(x2 - x1),
        float(y2 - y1),
    ]


def save_annotated_image(image, bboxes, labels, savepath):
    """
    BBoxを描画して画像を保存する。
    元画像のサイズは変更しない。
    """

    image_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(image_pil)

    for bbox, label in zip(bboxes, labels):

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # BBox
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=5,
        )

        # ラベル
        draw.text(
            (x1 + 5, y1 + 5),
            label,
            fill="red",
        )

    image_pil.save(
        savepath,
        quality=95,
    )

def draw_ocr_bboxes(image, prediction, savepath, scale=1):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    bboxes, labels = prediction["quad_boxes"], prediction["labels"]
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text(
            (new_box[0] + 8, new_box[1] + 2),
            "{}".format(label),
            align="right",
            fill=color,
        )

    # Save or display the image
    image.save(savepath)


# ======================
# Main functions
# ======================


def preprocess(img):
    h = w = IMG_SIZE

    img = np.array(Image.fromarray(img).resize((w, h), Image.Resampling.BICUBIC))

    img = normalize_image(img, normalize_type="ImageNet")

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float16)

    return img


def construct_prompts(text):
    task_prompts_without_inputs = {
        "<OCR>": "What is the text in the image?",
        "<OCR_WITH_REGION>": "What is the text in the image, with regions?",
        "<CAPTION>": "What does the image describe?",
        "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
        "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
        "<OD>": "Locate the objects with category name in the image.",
        "<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
        "<REGION_PROPOSAL>": "Locate the region proposals in the image.",
    }
    task_prompts_with_input = {
        "<CAPTION_TO_PHRASE_GROUNDING>": "Locate the phrases in the caption: {input}",
        "<REFERRING_EXPRESSION_SEGMENTATION>": "Locate {input} in the image with mask",
        "<REGION_TO_SEGMENTATION>": "What is the polygon mask of region {input}",
        "<OPEN_VOCABULARY_DETECTION>": "Locate {input} in the image.",
        "<REGION_TO_CATEGORY>": "What is the region {input}?",
        "<REGION_TO_DESCRIPTION>": "What does the region {input} describe?",
        "<REGION_TO_OCR>": "What text is in the region {input}?",
    }

    # replace the task tokens with the task prompts if task token is in the text
    prompts = []
    for _text in text:
        # 1. fixed task prompts without additional inputs
        for task_token, task_prompt in task_prompts_without_inputs.items():
            if task_token in _text:
                assert (
                    _text == task_token
                ), f"Task token {task_token} should be the only token in the text."
                _text = task_prompt
                break
        # 2. task prompts with additional inputs
        for task_token, task_prompt in task_prompts_with_input.items():
            if task_token in _text:
                _text = task_prompt.format(input=_text.replace(task_token, ""))
                break
        prompts.append(_text)

    return prompts


def decode(
    net,
    input_ids: np.ndarray,
    encoder_hidden_states: np.ndarray,
    past_key_values: List[np.ndarray],
):
    if not args.onnx:
        decoder_output = net.predict(
            [input_ids, encoder_hidden_states, *past_key_values]
        )
    else:
        decoder_output = net.run(
            None,
            {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "past_key_values.0.decoder.key": past_key_values[0],
                "past_key_values.0.decoder.value": past_key_values[1],
                "past_key_values.0.encoder.key": past_key_values[2],
                "past_key_values.0.encoder.value": past_key_values[3],
                "past_key_values.1.decoder.key": past_key_values[4],
                "past_key_values.1.decoder.value": past_key_values[5],
                "past_key_values.1.encoder.key": past_key_values[6],
                "past_key_values.1.encoder.value": past_key_values[7],
                "past_key_values.2.decoder.key": past_key_values[8],
                "past_key_values.2.decoder.value": past_key_values[9],
                "past_key_values.2.encoder.key": past_key_values[10],
                "past_key_values.2.encoder.value": past_key_values[11],
                "past_key_values.3.decoder.key": past_key_values[12],
                "past_key_values.3.decoder.value": past_key_values[13],
                "past_key_values.3.encoder.key": past_key_values[14],
                "past_key_values.3.encoder.value": past_key_values[15],
                "past_key_values.4.decoder.key": past_key_values[16],
                "past_key_values.4.decoder.value": past_key_values[17],
                "past_key_values.4.encoder.key": past_key_values[18],
                "past_key_values.4.encoder.value": past_key_values[19],
                "past_key_values.5.decoder.key": past_key_values[20],
                "past_key_values.5.decoder.value": past_key_values[21],
                "past_key_values.5.encoder.key": past_key_values[22],
                "past_key_values.5.encoder.value": past_key_values[23],
            },
        )

    logits, new_past_key_values = decoder_output[0], decoder_output[1:]

    return logits, new_past_key_values


def stopping_criteria(input_ids: np.array) -> bool:
    stopping_criteria.max_length = max_length = 1025

    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    is_done = np.full(input_ids.shape[0], is_done)

    eos_token_id = 2
    is_done = is_done | np.isin(input_ids[:, -1], eos_token_id)

    return is_done


def greedy_search(net, encoder_hidden_states):
    model_type = args.model_type

    pad_token_id = 1
    bos_token_id = 2
    eos_token_id = 2
    batch_size = 1
    num_beams = 3

    # prepare beam search scorer
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        length_penalty=1.0,
        do_early_stopping=True,
        num_beam_hyps_to_keep=1,
        max_length=1025,
    )

    input_ids = np.ones((num_beams, 1), dtype=int) * bos_token_id
    encoder_hidden_states = np.repeat(encoder_hidden_states, repeats=num_beams, axis=0)
    past_key_values = [
        np.zeros(
            (num_beams, 12 if model_type == "base" else 16, 0, 64), dtype=np.float16
        )
    ] * (24 if model_type == "base" else 48)

    # initialise score of first beam with 0 and the rest with -1e9.
    beam_scores = np.zeros((batch_size, num_beams))
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.flatten()

    this_peer_finished = False  # used by synced_gpus only
    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

    while True:
        decoder_input_ids = input_ids
        past_length = past_key_values[0].shape[2]
        decoder_input_ids = decoder_input_ids[:, past_length:]

        logits, past_key_values = decode(
            net,
            decoder_input_ids,
            encoder_hidden_states,
            past_key_values,
        )

        next_token_logits = logits[:, -1, :]
        next_token_scores = log_softmax(next_token_logits, axis=-1)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + np.broadcast_to(
            beam_scores[:, None], next_token_scores_processed.shape
        )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.reshape(1, num_beams * vocab_size)

        # Beam token selection
        n_eos_tokens = 1
        n_tokens_to_keep = (1 + n_eos_tokens) * num_beams

        next_tokens = np.argsort(-next_token_scores, axis=1, kind="stable")[
            :, :n_tokens_to_keep
        ]
        next_token_scores = np.take_along_axis(next_token_scores, next_tokens, axis=1)

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_prompt_len=decoder_prompt_len,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"].astype(int)
        beam_idx = beam_outputs["next_beam_indices"].astype(int)

        input_ids = np.concatenate(
            [input_ids[beam_idx, :], np.expand_dims(beam_next_tokens, axis=-1)], axis=-1
        )

        # temporary_reorder_cache
        reordered_past = []
        n = 24 if model_type == "base" else 48
        for i in range(0, n, 4):
            layer_past = past_key_values[i : i + 4]
            reordered_past += [
                np.take(past_state, beam_idx, axis=0) for past_state in layer_past[:2]
            ] + layer_past[2:]
        past_key_values = reordered_past

        if beam_scorer.is_done or all(stopping_criteria(input_ids)):
            this_peer_finished = True

        if this_peer_finished:
            break

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        decoder_prompt_len=decoder_prompt_len,
    )

    return sequence_outputs["sequences"]


def predict(models, img, task_prompt, text_input=None):
    im_h, im_w, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB
    pixel_values = preprocess(img)

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    text = construct_prompts([prompt])

    tokenizer = models["tokenizer"]
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=False,
        return_token_type_ids=False,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Extra the input embeddings
    net = models["embedding"]
    if not args.onnx:
        output = net.predict([input_ids])
    else:
        output = net.run(None, {"input_ids": input_ids})
    inputs_embeds = output[0]

    # Merge text and images
    net = models["encode_image"]
    if not args.onnx:
        output = net.predict([pixel_values])
    else:
        output = net.run(None, {"pixel_values": pixel_values})
    image_features = output[0]
    inputs_embeds = np.concatenate([image_features, inputs_embeds], axis=1)

    attention_mask = np.ones(inputs_embeds.shape[:2], dtype=int)

    net = models["encoder"]
    if not args.onnx:
        output = net.predict([inputs_embeds, attention_mask])
    else:
        output = net.run(
            None, {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        )
    last_hidden_state = output[0]

    net = models["decoder"]
    generated_ids = greedy_search(net, last_hidden_state)

    tokenizer = models["tokenizer"]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

    answer = post_process_generation(
        generated_text, task=task_prompt, image_size=(im_w, im_h)
    )
    return answer


def recognize_from_image(models):
    """
    Florence-2によるclosed umbrella自動アノテーション。

    元画像:
        3024 x 3024

    処理:
        1. 元画像を4分割
        2. 各CropをFlorence-2へ入力
        3. Crop座標を元画像座標へ戻す
        4. 重複BBoxを除去
        5. COCO annotationを生成
        6. BBox描画画像を保存

    Split:
        2 images -> train
        1 image  -> val
    """

    input_dir = "input_images"
    output_dir = "output"

    prompt = "<CAPTION_TO_PHRASE_GROUNDING>"

    # Florence-2にはumbrellaとして問い合わせる
    text_input = "umbrella"

    category_id = 1

    # データセット上のクラス名
    category_name = "closed_umbrella"

    # --------------------------------------------------
    # Output directories
    # --------------------------------------------------

    # BBox付き画像（annotation_viewer.py用）
    train_image_dir = os.path.join(
        output_dir,
        "train",
        "images",
    )

    val_image_dir = os.path.join(
        output_dir,
        "val",
        "images",
    )

    # BBoxなし元画像（YOLOX学習用）
    train_original_dir = os.path.join(
        output_dir,
        "train2017",
    )

    val_original_dir = os.path.join(
        output_dir,
        "val2017",
    )

    os.makedirs(
        train_image_dir,
        exist_ok=True,
    )

    os.makedirs(
        val_image_dir,
        exist_ok=True,
    )

    os.makedirs(
        train_original_dir,
        exist_ok=True,
    )

    os.makedirs(
        val_original_dir,
        exist_ok=True,
    )

    # --------------------------------------------------
    # COCO data
    # --------------------------------------------------

    train_coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "umbrella",
            }
        ],
    }

    val_coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "umbrella",
            }
        ],
    }

    train_image_id = 1
    train_annotation_id = 1

    val_image_id = 1
    val_annotation_id = 1

    # --------------------------------------------------
    # Input images
    # --------------------------------------------------

    image_files = []

    for filename in os.listdir(input_dir):

        lower = filename.lower()

        if (
            lower.endswith(".jpg")
            or lower.endswith(".jpeg")
            or lower.endswith(".png")
        ):

            image_files.append(filename)

    image_files.sort()

    print("")
    print("========================================")
    print("Florence-2 Closed Umbrella Detection")
    print("4-Crop Mode")
    print("========================================")
    print("")
    print(
        "Input images :",
        len(image_files),
    )
    print("")

    # --------------------------------------------------
    # Process images
    # --------------------------------------------------

    for index, filename in enumerate(image_files):

        input_path = os.path.join(
            input_dir,
            filename,
        )

        # ----------------------------------------------
        # 2 train / 1 val
        # ----------------------------------------------

        if index % 3 < 2:

            split = "train"

            image_dir = train_image_dir

            image_id = train_image_id

        else:

            split = "val"

            image_dir = val_image_dir

            image_id = val_image_id

        print("")
        print(
            "[%d/%d] %s -> %s"
            % (
                index + 1,
                len(image_files),
                filename,
                split,
            )
        )

        # ----------------------------------------------
        # Load original image
        # ----------------------------------------------

        img = load_image(
            input_path
        )

        if img is None:

            print(
                "  ERROR: cannot load image"
            )

            continue

        # ----------------------------------------------
        # Convert to RGB
        # ----------------------------------------------

        if len(img.shape) == 3 and img.shape[2] == 4:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2BGR,
            )

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

        original_height, original_width = (
            img_rgb.shape[:2]
        )

        print(
            "  Original size:",
            original_width,
            "x",
            original_height,
        )

        # ----------------------------------------------
        # Create 4 crops
        # ----------------------------------------------

        crops = crop_image_4(
            img_rgb
        )

        all_bboxes = []
        all_labels = []

        # ----------------------------------------------
        # Florence-2 inference for each crop
        # ----------------------------------------------

        for crop_index, (
            crop,
            offset_x,
            offset_y,
        ) in enumerate(crops):

            crop_height, crop_width = (
                crop.shape[:2]
            )

            print(
                "  Crop %d: %dx%d"
                % (
                    crop_index + 1,
                    crop_width,
                    crop_height,
                )
            )

            # ------------------------------------------
            # Florence-2
            # ------------------------------------------

            answer = predict(
                models,
                crop,
                prompt,
                text_input=text_input,
            )

            # ------------------------------------------
            # Get result
            # ------------------------------------------

            info = next(
                iter(answer.values())
            )

            crop_bboxes = info.get(
                "bboxes",
                [],
            )

            crop_labels = info.get(
                "labels",
                [],
            )

            print(
                "    detected:",
                len(crop_bboxes),
            )

            # ------------------------------------------
            # Restore coordinates
            # ------------------------------------------

            for bbox, label in zip(
                crop_bboxes,
                crop_labels,
            ):

                # ------------------------------------------
                # Remove suspicious large bbox
                # ------------------------------------------

                if not is_valid_crop_bbox(
                    bbox,
                    crop_width,
                    crop_height,
                ):

                    print(
                        "    rejected large bbox:",
                        bbox,
                    )

                    continue

                # ------------------------------------------
                # Restore coordinates
                # ------------------------------------------

                
                restored_bbox = restore_bbox(
                    bbox,
                    offset_x,
                    offset_y,
                )

                # --------------------------------------
                # Clip to original image
                # --------------------------------------

                x1, y1, x2, y2 = restored_bbox

                x1 = max(
                    0.0,
                    min(
                        float(original_width),
                        x1,
                    ),
                )

                y1 = max(
                    0.0,
                    min(
                        float(original_height),
                        y1,
                    ),
                )

                x2 = max(
                    0.0,
                    min(
                        float(original_width),
                        x2,
                    ),
                )

                y2 = max(
                    0.0,
                    min(
                        float(original_height),
                        y2,
                    ),
                )

                if x2 <= x1 or y2 <= y1:

                    continue

                all_bboxes.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )

                all_labels.append(
                    label
                )

        # ----------------------------------------------
        # Remove duplicate detections
        # ----------------------------------------------

        print(
            "  Before duplicate removal:",
            len(all_bboxes),
        )

        all_bboxes, all_labels = (
            remove_duplicate_bboxes(
                all_bboxes,
                all_labels,
                iou_threshold=0.5,
            )
        )

        print(
            "  Final detections:",
            len(all_bboxes),
        )

        # ----------------------------------------------
        # COCO image information
        # ----------------------------------------------

        coco_image = {
            "id": image_id,
            "file_name": filename,
            "width": original_width,
            "height": original_height,
        }

        if split == "train":

            train_coco["images"].append(
                coco_image
            )

        else:

            val_coco["images"].append(
                coco_image
            )

        # ----------------------------------------------
        # Add annotations
        # ----------------------------------------------

        for bbox in all_bboxes:

            coco_bbox = bbox_to_coco(
                bbox
            )

            x, y, w, h = coco_bbox

            annotation_id = (
                train_annotation_id
                if split == "train"
                else val_annotation_id
            )

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_bbox,
                "area": float(w * h),
                "iscrowd": 0,
            }

            if split == "train":

                train_coco["annotations"].append(
                    annotation
                )

                train_annotation_id += 1

            else:

                val_coco["annotations"].append(
                    annotation
                )

                val_annotation_id += 1

        # ----------------------------------------------
        # Save annotated image
        # ----------------------------------------------

        # ----------------------------------------------
        # Save annotated image for annotation_viewer.py
        # ----------------------------------------------

        output_path = os.path.join(
            image_dir,
            filename,
        )

        save_annotated_image(
            img_rgb,
            all_bboxes,
            all_labels,
            output_path,
        )

        # ----------------------------------------------
        # Save original image for YOLOX
        # ----------------------------------------------
        # BBoxは描画せず、input_imagesの元画像をそのままコピーする。

        if split == "train":
            original_output_path = os.path.join(
                train_original_dir,
                filename,
            )
        else:
            original_output_path = os.path.join(
                val_original_dir,
                filename,
            )

        shutil.copy2(
            input_path,
            original_output_path,
        )

        # ----------------------------------------------
        # Next image ID
        # ----------------------------------------------

        if split == "train":

            train_image_id += 1

        else:

            val_image_id += 1

    # --------------------------------------------------
    # Save COCO JSON
    # --------------------------------------------------

    train_json_path = os.path.join(
        output_dir,
        "train",
        "train.json",
    )

    val_json_path = os.path.join(
        output_dir,
        "val",
        "val.json",
    )

    with open(
        train_json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            train_coco,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        val_json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            val_coco,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("")
    print("========================================")
    print("Finished")
    print("========================================")
    print("")

    print(
        "Train images      :",
        len(train_coco["images"]),
    )

    print(
        "Train annotations :",
        len(train_coco["annotations"]),
    )

    print(
        "Val images        :",
        len(val_coco["images"]),
    )

    print(
        "Val annotations   :",
        len(val_coco["annotations"]),
    )

    print("")
    print(
        "Train JSON:",
        train_json_path,
    )

    print(
        "Val JSON:",
        val_json_path,
    )

    print("")

def main():
    if args.model_type == "base":
        WEIGHT_EMB_PATH = WEIGHT_EMB_BASE_PATH
        WEIGHT_IMG_PATH = WEIGHT_IMG_BASE_PATH
        WEIGHT_ENC_PATH = WEIGHT_ENC_BASE_PATH
        WEIGHT_DEC_PATH = WEIGHT_DEC_BASE_PATH
        MODEL_EMB_PATH = MODEL_EMB_BASE_PATH
        MODEL_IMG_PATH = MODEL_IMG_BASE_PATH
        MODEL_ENC_PATH = MODEL_ENC_BASE_PATH
        MODEL_DEC_PATH = MODEL_DEC_BASE_PATH
    else:
        WEIGHT_EMB_PATH = WEIGHT_EMB_LARGE_PATH
        WEIGHT_IMG_PATH = WEIGHT_IMG_LARGE_PATH
        WEIGHT_ENC_PATH = WEIGHT_ENC_LARGE_PATH
        WEIGHT_DEC_PATH = WEIGHT_DEC_LARGE_PATH
        MODEL_EMB_PATH = MODEL_EMB_LARGE_PATH
        MODEL_IMG_PATH = MODEL_IMG_LARGE_PATH
        MODEL_ENC_PATH = MODEL_ENC_LARGE_PATH
        MODEL_DEC_PATH = MODEL_DEC_LARGE_PATH
    check_and_download_models(WEIGHT_EMB_PATH, MODEL_EMB_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_IMG_PATH, MODEL_IMG_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=False, reuse_interstage=True)
        embedding = ailia.Net(MODEL_EMB_PATH, WEIGHT_EMB_PATH, env_id=env_id, memory_mode=memory_mode)
        encode_image = ailia.Net(MODEL_IMG_PATH, WEIGHT_IMG_PATH, env_id=env_id, memory_mode=memory_mode)
        encoder = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id, memory_mode=memory_mode)
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        embedding = onnxruntime.InferenceSession(WEIGHT_EMB_PATH, providers=providers)
        encode_image = onnxruntime.InferenceSession(
            WEIGHT_IMG_PATH, providers=providers
        )
        encoder = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)

    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.BartTokenizerFast.from_pretrained("./tokenizer")

        #tokenizer.additional_special_tokensにすでにSpecialTokensが含まれている
        if len(tokenizer.additional_special_tokens) != 1024:
            raise("special token error")

        #tokens_to_add = {
        #    "additional_special_tokens": tokenizer.additional_special_tokens
        #    + ["<od>", "</od>", "<ocr>", "</ocr>"]
        #    + [f"<loc_{x}>" for x in range(1000)]
        #    + [
        #        # fmt: off
        #        '<cap>', '</cap>', '<ncap>', '</ncap>','<dcap>', '</dcap>', '<grounding>', 
        #        '</grounding>', '<seg>', '</seg>', '<sep>', '<region_cap>', '</region_cap>', 
        #        '<region_to_desciption>', '</region_to_desciption>', '<proposal>', '</proposal>', 
        #        '<poly>', '</poly>', '<and>'
        #        # fmt: on
        #    ]
        #}
        #tokenizer.add_special_tokens(tokens_to_add)
    else:
        import ailia_tokenizer
        tokenizer = ailia_tokenizer.RobertaTokenizer.from_pretrained("./tokenizer")

        tokens_to_add = {
            "additional_special_tokens":
              ["<od>", "</od>", "<ocr>", "</ocr>"]
            + [f"<loc_{x}>" for x in range(1000)]
            + [
                # fmt: off
                '<cap>', '</cap>', '<ncap>', '</ncap>','<dcap>', '</dcap>', '<grounding>', 
                '</grounding>', '<seg>', '</seg>', '<sep>', '<region_cap>', '</region_cap>', 
                '<region_to_desciption>', '</region_to_desciption>', '<proposal>', '</proposal>', 
                '<poly>', '</poly>', '<and>'
                # fmt: on
            ]
        }
        tokenizer.add_special_tokens(tokens_to_add)

    models = {
        "tokenizer": tokenizer,
        "embedding": embedding,
        "encode_image": encode_image,
        "encoder": encoder,
        "decoder": decoder,
    }

    # generate
    recognize_from_image(models)


if __name__ == "__main__":
    main()
