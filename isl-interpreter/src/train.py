"""
train.py

Training loop for the ISL sign classifier. This is a skeleton — fill in
`manifest` with your real (landmark_path, label) pairs once you've
extracted landmarks for your chosen word subset (see dataset.py).

Suggested first vocabulary: pick ~50 high-utility words from the
INCLUDE dataset (greetings, numbers, "help", "doctor", "water",
"yes"/"no", family terms) rather than trying to cover the full
vocabulary — smaller, focused vocabulary = faster to get a working,
demoable model.
"""

import torch
from torch.utils.data import DataLoader, random_split

from dataset import ISLSequenceDataset
from model import SignClassifier
from capture import FEATURE_DIM

# TODO: replace with your real manifest, e.g.:
# manifest = [("data/landmarks/help_001.npy", 0), ("data/landmarks/water_002.npy", 1), ...]
manifest: list[tuple[str, int]] = []

NUM_CLASSES = 50  # match this to however many words are in your manifest
EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3


def train() -> None:
    if not manifest:
        raise ValueError(
            "manifest is empty — build it from your extracted landmark "
            "sequences first (see dataset.py docstring)."
        )

    dataset = ISLSequenceDataset(manifest)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignClassifier(input_dim=FEATURE_DIM, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total else 0.0
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2%}")

    torch.save(model.state_dict(), "sign_classifier.pt")
    print("Saved model to sign_classifier.pt")


if __name__ == "__main__":
    train()
