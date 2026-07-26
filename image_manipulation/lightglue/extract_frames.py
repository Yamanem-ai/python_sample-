import os
import cv2
import numpy as np
import argparse

def extract_frames(video_path, output_dir="frames", num_frames=10):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    os.makedirs("frames", exist_ok=True)

    frame_files = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if not ret:
            continue

        filename = f"frames/frame_{i:03d}.jpg"

        cv2.imwrite(filename, frame)

        frame_files.append(filename)

    cap.release()

    return frame_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract equally spaced frames from a video."
    )

    parser.add_argument(
        "video",
        help="Input video file"
    )

    parser.add_argument(
        "-n",
        "--num_frames",
        type=int,
        default=10,
        help="Number of frames to extract"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="frames",
        help="Output directory"
    )

    args = parser.parse_args()

    extract_frames(
        args.video,
        args.output,
        args.num_frames
    )
