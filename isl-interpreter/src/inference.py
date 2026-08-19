"""
inference.py

Real-time inference: webcam -> landmarks -> trained model -> predicted
word, shown live on screen.

Requires a trained model (run prepare_dataset.py then train.py first,
which produce sign_classifier.pt) and data/labels.json (also produced
by prepare_dataset.py) to map predicted class indices back to words.

Run:
    python src/inference.py

Press 'q' to quit.
"""

import json
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from capture import extract_landmarks, FEATURE_DIM
from model import SignClassifier

MODEL_PATH = Path("sign_classifier.pt")
LABELS_PATH = Path("data/labels.json")

# Must match dataset.py's max_seq_len used during training — the model
# was trained on fixed-length windows of this size.
SEQUENCE_LENGTH = 60
# Run a prediction every N frames rather than every single frame — signs
# take a while to perform, so predicting every frame is wasted work and
# makes the on-screen text flicker.
PREDICTION_INTERVAL = 30
# Below this confidence, show nothing rather than a noisy guess.
CONFIDENCE_THRESHOLD = 0.6

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def load_labels() -> dict:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"{LABELS_PATH} not found — run prepare_dataset.py and train.py first.")
    with open(LABELS_PATH) as f:
        word_to_idx = json.load(f)
    return {idx: word for word, idx in word_to_idx.items()}


def load_model(num_classes: int, device: torch.device) -> SignClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found — train the model first with train.py.")
    model = SignClassifier(input_dim=FEATURE_DIM, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference() -> None:
    idx_to_word = load_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(num_classes=len(idx_to_word), device=device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # A sliding window of the last SEQUENCE_LENGTH frames. deque(maxlen=...)
    # automatically drops the oldest frame once full, so once len() ==
    # SEQUENCE_LENGTH it's always exactly a full window — no manual
    # padding/truncation needed here (unlike dataset.py, which pads
    # short *training* clips).
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    last_prediction = ""

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            frame_buffer.append(extract_landmarks(results))
            frame_count += 1

            if frame_count % PREDICTION_INTERVAL == 0 and len(frame_buffer) == SEQUENCE_LENGTH:
                sequence = np.array(frame_buffer)
                input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    confidence, pred_idx = probs.max(dim=1)

                if confidence.item() >= CONFIDENCE_THRESHOLD:
                    word = idx_to_word[pred_idx.item()]
                    last_prediction = f"{word} ({confidence.item():.0%})"

            cv2.putText(image, last_prediction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("ISL Interpreter (press q to quit)", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()
