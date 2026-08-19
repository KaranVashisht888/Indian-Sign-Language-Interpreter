"""
train.py

Training loop for the ISL sign classifier. Loads the manifest and label
vocabulary produced by prepare_dataset.py — run that first if you haven't:

    python src/prepare_dataset.py
    python src/train.py

Saves the best-validation-accuracy checkpoint to sign_classifier.pt,
which inference.py loads for real-time prediction.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import ISLSequenceDataset
from model import SignClassifier
from capture import FEATURE_DIM

MANIFEST_PATH = Path("data/manifest.json")
LABELS_PATH = Path("data/labels.json")
MODEL_OUT_PATH = Path("sign_classifier.pt")

EPOCHS = 60
BATCH_SIZE = 16
# 1e-3 is too hot for an LSTM on a small dataset — it causes the model to
# lurch in and out of learning (loss dropping, then snapping back to
# chance level). 3e-4 trains slower but far more steadily.
LR = 3e-4
WEIGHT_DECAY = 1e-4
# LSTMs are prone to exploding gradients. Clipping is cheap insurance and
# is the single most effective fix for training that collapses mid-run.
GRAD_CLIP = 1.0
SEED = 42


def load_manifest_and_labels():
    if not MANIFEST_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "data/manifest.json or data/labels.json not found. Run "
            "prepare_dataset.py first to extract landmarks and build these."
        )
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    return manifest, labels


def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    manifest, labels = load_manifest_and_labels()
    num_classes = len(labels)
    chance = 1.0 / num_classes

    dataset = ISLSequenceDataset(manifest)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    print(f"Loaded {len(manifest)} samples across {num_classes} words.")
    print(f"Train: {train_size} | Val: {val_size} "
          f"(each val sample is worth {100 / val_size:.1f} percentage points)")
    print(f"Random-chance accuracy is {chance:.1%} — beat this and the model is learning.\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignClassifier(input_dim=FEATURE_DIM, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = torch.nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters for {train_size} training samples.")
    if param_count > train_size * 1000:
        print("(That's a lot of capacity for this much data — expect overfitting "
              "until you add more words/clips.)\n")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, num_batches = 0.0, 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        # Average, not sum — a summed loss changes meaning whenever the
        # batch count changes, which makes runs impossible to compare.
        # Reference point: random guessing gives ln(num_classes).
        avg_loss = total_loss / max(num_batches, 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                preds = model(seqs).argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = correct / total if total else 0.0
        scheduler.step(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT_PATH)
            marker = "  <- best, saved"

        print(f"Epoch {epoch + 1:>3}/{EPOCHS} | Loss: {avg_loss:.4f} "
              f"(chance={np.log(num_classes):.2f}) | Val: {val_acc:.1%}{marker}")

    print(f"\nBest val accuracy: {best_val_acc:.1%} (chance is {chance:.1%})")
    print(f"Saved best model to {MODEL_OUT_PATH}")

    if best_val_acc < chance * 1.5:
        print("\nThat's barely above chance. More data is the fix — add words "
              "and clips rather than tuning hyperparameters further.")


if __name__ == "__main__":
    train()
