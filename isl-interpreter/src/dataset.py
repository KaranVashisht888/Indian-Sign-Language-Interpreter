"""
dataset.py

PyTorch Dataset for loading landmark sequences + labels.

Expects a "manifest": a list of (landmark_npy_path, label_index) pairs.
You build this manifest once you've run landmark extraction over your
dataset's video clips — e.g. AI4Bharat's INCLUDE dataset — using the
same extract_landmarks() logic from capture.py, adapted to read from
video files frame-by-frame instead of a live webcam feed. Save each
clip's extracted sequence as a .npy file and point the manifest at it.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ISLSequenceDataset(Dataset):
    def __init__(self, manifest: list[tuple[str, int]], max_seq_len: int = 60):
        self.manifest = manifest
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        path, label = self.manifest[idx]
        seq = np.load(path)  # shape: (num_frames, feature_dim)

        # Pad or truncate to a fixed length so batches are uniform.
        # Signs are short (typically 1-3 seconds), so a fixed window
        # is simpler than dealing with variable-length batching for
        # a first version.
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
        else:
            pad = np.zeros((self.max_seq_len - len(seq), seq.shape[1]))
            seq = np.concatenate([seq, pad], axis=0)

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
