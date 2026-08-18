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
🚧 Early scaffold — landmark capture pipeline is working; sequence model and
dataset integration are in progress.

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/capture.py           # opens your webcam and overlays landmarks
```

## Roadmap
- [x] Live landmark extraction (MediaPipe Holistic)
- [ ] Build a manifest from AI4Bharat's INCLUDE dataset (or your own recordings)
- [ ] Train the sequence classifier on ~50 core words
- [ ] Real-time inference loop (webcam → prediction → text output)
- [ ] Simple web frontend + WebSocket streaming
- [ ] Stretch: rule-based/LLM cleanup of word sequences into natural sentences

## Project layout
```
isl-interpreter/
├── requirements.txt
├── src/
│   ├── capture.py    # webcam -> landmarks (run this first, it works today)
│   ├── model.py       # sequence classifier (LSTM)
│   ├── dataset.py     # PyTorch Dataset for landmark sequences
│   └── train.py       # training loop (needs a manifest — see dataset.py)
└── data/               # gitignored — put extracted landmarks/datasets here
```
