"""
dataset.py

PyTorch Dataset for loading landmark sequences + labels.

Expects a "manifest": a list of (landmark_npy_path, label_index) pairs,
produced by prepare_dataset.py.

Normalization happens here at load time rather than during extraction,
so the cached .npy files stay raw and you can change the normalization
scheme without re-running the slow landmark extraction step.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# Feature layout, matching capture.py's extract_landmarks:
#   pose: 33 landmarks x 4 (x, y, z, visibility)  = 132
#   left hand:  21 x 3                            = 63
#   right hand: 21 x 3                            = 63
POSE_COUNT, POSE_VALS = 33, 4
HAND_COUNT, HAND_VALS = 21, 3
POSE_DIM = POSE_COUNT * POSE_VALS
HAND_DIM = HAND_COUNT * HAND_VALS

# MediaPipe pose landmark indices for the shoulders
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Makes landmarks translation- and scale-invariant by re-expressing every
    point relative to the signer's own body.

    MediaPipe returns coordinates normalized to the image frame, which means
    the same sign performed closer to the camera, or standing slightly to
    one side, produces very different numbers. The model then has to waste
    capacity learning that those are the same gesture. Instead:

      - recenter every point on the midpoint between the shoulders
      - divide by shoulder width, so a person close to the camera and a
        person further away produce the same values

    This matters a lot when training data and webcam conditions differ,
    which is exactly our situation (INCLUDE was shot in a fixed studio
    setup; your webcam is not).

    Frames where the pose wasn't detected (all zeros) are left as zeros.
    """
    seq = seq.copy()

    pose = seq[:, :POSE_DIM].reshape(-1, POSE_COUNT, POSE_VALS)
    left = seq[:, POSE_DIM:POSE_DIM + HAND_DIM].reshape(-1, HAND_COUNT, HAND_VALS)
    right = seq[:, POSE_DIM + HAND_DIM:].reshape(-1, HAND_COUNT, HAND_VALS)

    shoulder_l = pose[:, LEFT_SHOULDER, :3]
    shoulder_r = pose[:, RIGHT_SHOULDER, :3]
    center = (shoulder_l + shoulder_r) / 2.0                      # (frames, 3)
    width = np.linalg.norm(shoulder_l - shoulder_r, axis=1)       # (frames,)

    # Frames with no pose detection have zero shoulder width — skip those
    # rather than dividing by zero.
    valid = width > 1e-6
    scale = np.where(valid, width, 1.0)[:, None]

    for block, vals in ((pose, 3), (left, 3), (right, 3)):
        present = np.any(block != 0, axis=2)                      # (frames, points)
        centered = (block[:, :, :vals] - center[:, None, :]) / scale[:, None, :]
        # Keep missing landmarks at zero instead of shifting them to -center
        block[:, :, :vals] = np.where(present[:, :, None], centered, 0.0)

    # Frames without a valid pose get zeroed entirely — they carry no usable
    # spatial information once we can't anchor them.
    pose[~valid] = 0.0
    left[~valid] = 0.0
    right[~valid] = 0.0

    return np.concatenate(
        [pose.reshape(len(seq), -1), left.reshape(len(seq), -1), right.reshape(len(seq), -1)],
        axis=1,
    )


class ISLSequenceDataset(Dataset):
    def __init__(self, manifest: list[tuple[str, int]], max_seq_len: int = 60, normalize: bool = True):
        self.manifest = manifest
        self.max_seq_len = max_seq_len
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        path, label = self.manifest[idx]
        seq = np.load(path)  # shape: (num_frames, feature_dim)

        if self.normalize:
            seq = normalize_sequence(seq)

        # Pad or truncate to a fixed length so batches are uniform.
        # Signs are short (typically 1-3 seconds), so a fixed window is
        # simpler than variable-length batching for a first version.
        # If your clips are consistently longer than max_seq_len, raise it —
        # truncating cuts off the end of the sign, which is often where the
        # distinguishing motion happens.
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
        else:
            pad = np.zeros((self.max_seq_len - len(seq), seq.shape[1]))
            seq = np.concatenate([seq, pad], axis=0)

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
