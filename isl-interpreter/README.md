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


## Dataset: AI4Bharat INCLUDE

This project trains on [INCLUDE](https://huggingface.co/datasets/ai4bharat/INCLUDE)
(AI4Bharat, IIT Madras) — 4,287 videos across 263 ISL word signs. It's
CC-BY-4.0 and freely downloadable, no access request needed.

- **Metadata** (class labels, video paths, `include_50` flag) lives on Hugging Face.
- **The videos themselves** live on Zenodo, record `4010759` — the HF dataset
  card has a bash download script.
- **INCLUDE-50** is a 50-word subset chosen across categories for faster
  iteration, ~25 videos per word. Start here, not with the full 263.

Note: these videos were shot in Chennai, Tamil Nadu. ISL varies across India,
so a model trained on this won't generalize perfectly to signing from other
regions — worth stating honestly rather than claiming universal ISL coverage.


## Roadmap
- [x] Live landmark extraction (MediaPipe Holistic)
- [x] INCLUDE reorganizer (category tree → flat word folders, subset filtering)
- [x] Dataset prep script (video clips → landmark manifest)
- [x] Training script (loads manifest, saves best checkpoint)
- [x] Real-time inference script (webcam → prediction → text)
- [ ] Scale from the 5-word starter subset up to INCLUDE-50
- [ ] Simple web frontend + WebSocket streaming
- [ ] Stretch: rule-based/LLM cleanup of word sequences into natural sentences

## Project layout
```
isl-interpreter/
├── requirements.txt
├── src/
│   ├── capture.py           # webcam -> landmarks (run this first, it works today)
│   ├── organize_include.py  # INCLUDE's folder tree -> flat data/raw_videos/<word>/
│   ├── prepare_dataset.py   # video clips -> landmark manifest + label vocabulary
│   ├── model.py              # sequence classifier (LSTM)
│   ├── dataset.py            # PyTorch Dataset for landmark sequences
│   ├── train.py              # training loop (consumes prepare_dataset.py's output)
│   └── inference.py          # real-time webcam -> prediction, using the trained model
└── data/                      # gitignored — raw_videos/, landmarks/, manifest.json, labels.json go here
```

## Citation
```
Sridhar, A., Ganesan, R.G., Kumar, P., & Khapra, M. (2020).
INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.
Proceedings of the 28th ACM International Conference on Multimedia, 1366–1375.
https://doi.org/10.1145/3394171.3413528
```



Windows users should install torch from the CPU index to avoid the long-path issue


noting that the project pins MediaPipe below 0.10.30 because the legacy Solutions API was removed, and that migrating to Tasks is future work.
