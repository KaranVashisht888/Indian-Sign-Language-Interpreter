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

import torch
from torch.utils.data import DataLoader, random_split

from dataset import ISLSequenceDataset
from model import SignClassifier
from capture import FEATURE_DIM

MANIFEST_PATH = Path("data/manifest.json")
LABELS_PATH = Path("data/labels.json")
MODEL_OUT_PATH = Path("sign_classifier.pt")

EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3


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
    manifest, labels = load_manifest_and_labels()
    num_classes = len(labels)
    print(f"Loaded {len(manifest)} samples across {num_classes} words.")

    dataset = ISLSequenceDataset(manifest)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignClassifier(input_dim=FEATURE_DIM, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                outputs = model(seqs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = correct / total if total else 0.0
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT_PATH)

    print(f"\nDone. Best val accuracy: {best_val_acc:.2%}. Saved best model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    train()
