# ISL Interpreter

Real-time Indian Sign Language (ISL) recognition — captures webcam video,
extracts hand/pose landmarks with MediaPipe Holistic, and classifies short
signs into a set of high-utility words/phrases (greetings, numbers, "help",
"doctor", "water", "yes/no", family terms, etc).

Scope note: this targets isolated-word recognition, not full continuous
sentence translation. ISL has its own grammar (topic-comment structure,
grammatical facial expressions) that doesn't map word-for-word to English —
that's an open research problem, not something to solve in a resume project.
A focused word/phrase-level tool that actually works is more valuable (and
more honest) than an inflated claim.

## Status
🚧 Full pipeline is scaffolded end-to-end (capture → dataset prep → train →
inference). What's missing is your actual video data — none of the scripts
past `capture.py` will produce real results until you supply labeled clips.

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/capture.py           # opens your webcam and overlays landmarks
```

## Pipeline, in order
```bash
# 1. Organize labeled clips as data/raw_videos/<word>/<clip>.mp4
#    (see the docstring in prepare_dataset.py)
python src/prepare_dataset.py   # extracts landmarks, builds manifest + labels

python src/train.py             # trains on data/manifest.json, saves sign_classifier.pt

python src/inference.py         # real-time webcam prediction using the trained model
```

## Roadmap
- [x] Live landmark extraction (MediaPipe Holistic)
- [x] Dataset prep script (video clips → landmark manifest) — needs real clips to run
- [x] Training script (loads manifest, saves best checkpoint) — needs a manifest to run
- [x] Real-time inference script (webcam → prediction → text) — needs a trained model to run
- [ ] Simple web frontend + WebSocket streaming
- [ ] Stretch: rule-based/LLM cleanup of word sequences into natural sentences

## Project layout
```
isl-interpreter/
├── requirements.txt
├── src/
│   ├── capture.py           # webcam -> landmarks (run this first, it works today)
│   ├── prepare_dataset.py   # video clips -> landmark manifest + label vocabulary
│   ├── model.py              # sequence classifier (LSTM)
│   ├── dataset.py            # PyTorch Dataset for landmark sequences
│   ├── train.py              # training loop (consumes prepare_dataset.py's output)
│   └── inference.py          # real-time webcam -> prediction, using the trained model
└── data/                      # gitignored — raw_videos/, landmarks/, manifest.json, labels.json go here
```
