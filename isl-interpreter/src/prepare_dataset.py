"""
prepare_dataset.py

Extracts landmark sequences from labeled video clips and builds the
manifest that dataset.py / train.py expect.

Expected input layout — organize (or symlink) your clips like this,
using folder names as the word labels:

    data/raw_videos/
        help/
            clip_001.mp4
            clip_002.mp4
        water/
            clip_001.mp4
            ...

If you're starting from AI4Bharat's INCLUDE dataset, its raw folder
layout varies by category. The simplest path is to copy or symlink each
word's clips into this flat data/raw_videos/<word>/ structure before
running this script — you don't need to touch the original download
beyond that one reorganizing step. Start with a small subset (your ~50
target words) rather than the full vocabulary.

Run:
    python src/prepare_dataset.py

Produces:
    data/landmarks/<word>/<clip>.npy   — one landmark sequence per clip
    data/manifest.json                  — [[npy_path, label_index], ...]
    data/labels.json                    — {"help": 0, "water": 1, ...}
"""

import json
from pathlib import Path

import cv2
import numpy as np

from capture import extract_landmarks, mp_holistic

RAW_VIDEO_DIR = Path("data/raw_videos")
LANDMARK_DIR = Path("data/landmarks")
MANIFEST_PATH = Path("data/manifest.json")
LABELS_PATH = Path("data/labels.json")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def extract_landmarks_from_video(video_path: Path) -> np.ndarray:
    """Runs MediaPipe Holistic over every frame of a video file and
    returns the landmark sequence as an (num_frames, feature_dim) array."""
    cap = cv2.VideoCapture(str(video_path))
    sequence = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            sequence.append(extract_landmarks(results))

    cap.release()
    return np.array(sequence)


def build_dataset() -> None:
    if not RAW_VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_VIDEO_DIR} not found. Organize your clips as "
            f"{RAW_VIDEO_DIR}/<word>/<clip>.mp4 before running this script "
            f"(see the module docstring)."
        )

    words = sorted(d.name for d in RAW_VIDEO_DIR.iterdir() if d.is_dir())
    if not words:
        raise ValueError(f"No word subfolders found inside {RAW_VIDEO_DIR}.")

    labels = {word: idx for idx, word in enumerate(words)}
    manifest = []

    for word in words:
        word_dir = RAW_VIDEO_DIR / word
        out_dir = LANDMARK_DIR / word
        out_dir.mkdir(parents=True, exist_ok=True)

        # Case-insensitive: INCLUDE ships uppercase .MOV / .MP4 filenames
        clips = sorted(p for p in word_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
        if not clips:
            print(f"Warning: no video clips found for '{word}', skipping.")
            continue

        for clip_path in clips:
            out_path = out_dir / f"{clip_path.stem}.npy"

            if out_path.exists():
                # Already processed in a previous run — skip so re-runs are cheap
                manifest.append([str(out_path), labels[word]])
                continue

            print(f"Processing {clip_path} ...")
            sequence = extract_landmarks_from_video(clip_path)
            if len(sequence) == 0:
                print(f"Warning: no frames extracted from {clip_path}, skipping.")
                continue

            np.save(out_path, sequence)
            manifest.append([str(out_path), labels[word]])

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone. {len(manifest)} clips processed across {len(words)} words.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Labels:   {LABELS_PATH}")


if __name__ == "__main__":
    build_dataset()
