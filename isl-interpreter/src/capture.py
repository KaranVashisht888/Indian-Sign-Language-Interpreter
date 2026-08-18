"""
capture.py

Real-time webcam capture with MediaPipe Holistic landmark extraction.
This is the first building block of the ISL interpreter pipeline:

    webcam -> landmarks -> (later) sequence model -> predicted word

Run this file directly to see your webcam feed with pose/hand landmarks
drawn live. It also gives you `extract_landmarks`, which turns a single
frame's detection results into a fixed-length feature vector — this is
the function you'll reuse later to build a training dataset from
recorded clips.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Feature vector layout: pose (33 landmarks x 4 values) + left hand
# (21 x 3) + right hand (21 x 3). Face landmarks are skipped for now to
# keep the feature dimension small — hands + pose carry most of the
# signal for isolated-word signs. Revisit if accuracy plateaus.
POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
FEATURE_DIM = POSE_DIM + HAND_DIM + HAND_DIM


def extract_landmarks(results) -> np.ndarray:
    """
    Flattens pose and both-hand landmarks from a MediaPipe Holistic
    result into a single feature vector of fixed length FEATURE_DIM.
    Missing landmark sets (e.g. a hand out of frame) become zeros
    rather than crashing or shrinking the vector.
    """
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(POSE_DIM)
    )
    left_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(HAND_DIM)
    )
    right_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(HAND_DIM)
    )
    return np.concatenate([pose, left_hand, right_hand])


def run_capture(save_sequence: bool = False, output_path: str = "sequence.npy") -> None:
    """
    Opens the default webcam, runs MediaPipe Holistic on each frame,
    draws the landmarks live, and (optionally) records the extracted
    feature vectors to a .npy file — useful later for recording your
    own sign samples if the public dataset needs supplementing.

    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/index.")

    sequence = []

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

            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if save_sequence:
                sequence.append(extract_landmarks(results))

            cv2.imshow("ISL Capture (press q to quit)", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if save_sequence and sequence:
        np.save(output_path, np.array(sequence))
        print(f"Saved {len(sequence)} frames to {output_path}")


if __name__ == "__main__":
    run_capture()
