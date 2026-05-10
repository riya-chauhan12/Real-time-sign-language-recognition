<div align="center">

# 🤟 ISL Real-Time Sign Language Recognition

**A real-time Indian Sign Language recognition system built with MediaPipe, LSTM, TensorFlow, Flask, and Server-Sent Events**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13–2.21+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9+-0097A7?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Prototype-brightgreen?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)]()

<br/>

> *Translating hand gestures into words — one landmark at a time.*

</div>

---

## What This Project Is

This is a real-time Indian Sign Language (ISL) recognition system that uses your webcam to detect hand gestures and translate them into text — live, in your browser.

It's not a production app or a research paper. It's an **actively developed prototype** built to explore how sign language recognition actually works end-to-end — from raw camera frames to predicted words, streamed to a frontend without any page refresh.

The project currently focuses on a **curated set of ISL gestures**. That's intentional — a smaller, well-defined gesture set lets the system prioritise real-time performance, stable landmark extraction, and solid sequence-learning experiments rather than chasing a large vocabulary that sacrifices everything else.

---

## What This Project Tries to Achieve

- Capture hand landmarks in real-time using **MediaPipe HandLandmarker**
- Build temporal sequences from per-frame landmark features
- Run those sequences through an **LSTM model** to classify ISL gestures
- Automatically detect when a gesture starts and ends — no button presses needed
- Stream predictions live to the browser using **Server-Sent Events (SSE)**
- Keep the whole pipeline working reliably across different TensorFlow versions

The project is as much about making real-time ML inference actually work end-to-end as it is about the gestures themselves.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Hand Tracking | MediaPipe HandLandmarker (Task API) |
| Feature Extraction | Per-frame 3D landmark coordinates + exponential smoothing |
| Sequence Model | LSTM via TensorFlow / Keras |
| Backend | Flask + Flask-CORS |
| Live Updates | Server-Sent Events (SSE) |
| Video Streaming | MJPEG over HTTP |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Camera | OpenCV VideoCapture |

---

## Architecture & Pipeline

```
Webcam Frame (OpenCV)
        │
        ▼
  MediaPipe HandLandmarker
  (21 landmarks × 2 hands × XYZ)
        │
        ▼
  FrameFeatureExtractor
  (normalise + smooth landmarks → feature vector)
        │
        ▼
  Sliding Window Buffer  ──→  SEQ_LEN frames
        │
        ▼
  Gesture Segmenter (SegState machine)
  IDLE → SIGNING → HOLD → COMMIT
        │
        ▼
  LSTM Model (TensorFlow / Keras)
  softmax → label + confidence
        │
        ├──→  SSE stream  ──→  Browser (live word predictions)
        │
        └──→  MJPEG stream  ──→  Browser (annotated video feed)
```

The inference loop runs in a background thread, keeping Flask free to handle browser connections without blocking.

---

## Features

**Real-Time Gesture Inference**
Landmark extraction, sequence building, and LSTM prediction all happen at webcam frame rate with no manual triggering required.

**Automatic Gesture Segmentation**
A simple state machine (IDLE → SIGNING → HOLD → COMMIT) figures out when a gesture starts and ends on its own — so you just sign naturally and the system picks it up.

**Live Predictions via SSE**
Predicted words are pushed to the browser the moment they're confirmed — no polling, no refresh. The connection stays open and updates arrive as you sign.

**Annotated Video Feed**
The webcam stream appears in the browser with hand skeleton overlaid and a live state indicator (IDLE / SIGNING / HOLD) drawn directly on the frame using OpenCV.

**TensorFlow Version Compatibility**
The model loader handles `.h5` files across TF 2.13 through 2.21+, working around a breaking change in how Keras 3 reads older model configs. It tries several approaches automatically before falling back to patching the model file directly.

**One-Click Setup (Windows)**
`setup_and_run.bat` handles everything — virtual environment, dependencies, TF version detection, and server startup — in a single double-click.

---

## Screenshots / Demo

> 📸 *Demo visuals coming soon — GIFs and screenshots will be added here.*

```
┌─────────────────────────────────────────┐
│  [ Webcam feed with hand skeleton ]     │
│                          ┌────────────┐ │
│                          │ SIGNING    │ │
│  ●  (hand detected)      └────────────┘ │
│                                         │
│  Predicted Word:  "HELLO"  (87%)        │
│  Sentence:  HELLO  HOW  ARE  YOU        │
└─────────────────────────────────────────┘
```



---

## Folder Structure

```
isl-demo/
├── backend/
│   └── app.py                  # Flask server, inference loop, SSE, video feed
├── utils/
│   ├── landmarks.py            # landmark extraction + smoothing (FrameFeatureExtractor)
│   ├── predictor.py            # LSTM inference + confidence gating (GesturePredictor)
│   └── segmenter.py            # gesture segmentation state machine (SegState)
├── models/
│   ├── isl_model_final.h5      # trained LSTM model (not tracked in git)
│   ├── model_meta.json         # gesture labels + model config
│   └── hand_landmarker.task    # MediaPipe hand tracking binary
├── frontend/
│   ├── templates/
│   │   └── index.html          # browser UI
│   └── static/                 # CSS + JS
├── setup_and_run.bat           # Windows one-click setup & launch
└── requirements.txt
```

---

## Installation & Setup

### Prerequisites

- Python **3.10, 3.11, or 3.12**
- A working webcam
- The model files (see note below)

### Model Files Required

Copy the following into the `models/` folder before running:

| File | Source |
|---|---|
| `isl_model_final.h5` | Your Kaggle/Colab training output |
| `model_meta.json` | Your Kaggle/Colab training output |
| `hand_landmarker.task` | [Download from MediaPipe](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |

### Install Dependencies

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# If you're on TF 2.16 or later, also install:
pip install tf_keras
```

> **Note:** If you see a Keras compatibility error on startup, the backend will try to resolve it automatically. If it can't, check that `tf_keras` is installed.

---

## Running the Project

**Option A — Windows (easiest)**

```
Double-click setup_and_run.bat
```

The script handles everything: venv creation, dependency installation, and server startup.

**Option B — Manual**

```bash
# Activate your venv first, then:
python backend/app.py
```

Then open your browser at:

```
http://localhost:5050
```

The webcam feed, gesture state indicator, and live predictions will all appear in the browser. Just start signing.

---

## Technical Implementation Highlights

### MediaPipe Landmark Extraction

Each video frame is passed to MediaPipe's `HandLandmarker`, which detects up to 2 hands and returns 21 landmarks per hand — each with X, Y, Z coordinates in normalised image space. The `FrameFeatureExtractor` applies a bit of smoothing to reduce jitter between frames, then flattens everything into a fixed-size feature vector ready for the model.

### Landmark → Sequence → LSTM → Prediction

Feature vectors are collected frame by frame into a sliding window. Once the gesture segmenter decides the gesture has ended (more on that below), the buffered sequence gets passed to the LSTM model. The model outputs a probability distribution over gesture classes, and a prediction is only confirmed if the confidence is high enough *and* there's a clear enough gap between the top two candidates — this helps avoid false positives.

### Gesture Segmentation

Rather than using fixed time windows, the system watches for changes in landmark motion to decide when a gesture has started and stopped. It moves through four states:

- **IDLE** — no hand movement detected
- **SIGNING** — movement picked up, recording frames
- **HOLD** — movement has settled, waiting to confirm
- **COMMIT** — sequence sent to the LSTM, prediction emitted

This makes signing feel natural — you don't need to press anything or hold a pose for a fixed duration.

### Real-Time Updates with SSE

When a word is predicted, it's immediately pushed to the browser via Server-Sent Events — a simple, lightweight alternative to WebSockets for one-way server-to-client updates. The connection stays open the whole time, so predictions just appear as you sign without any polling or page refresh.

### Keras Compatibility

Old `.h5` model files can have issues loading on newer versions of TensorFlow (2.16+) due to changes in how Keras handles model configs. The backend works around this automatically using a few different loading strategies, so the same model file should work regardless of which TF version you have installed.

---

## Challenges Faced During Development

**TensorFlow / Keras version fragmentation**
This was probably the most frustrating part. The Keras 2 → Keras 3 transition in TF 2.16 broke `.h5` model loading in a subtle way — a `batch_shape` field in the model config that older Keras versions wrote gets rejected by newer ones. Figuring out what was actually wrong meant digging into the saved model's internal JSON and writing several different loading approaches until one stuck.

**MediaPipe API changes between versions**
At some point between mediapipe releases, a parameter got renamed from `min_hand_presence_score` to `min_hand_presence_confidence`. This caused silent failures on some setups. The fix was to check which name the installed version actually accepts at runtime using `inspect.signature` — small thing, but it took a while to track down.

**Gesture boundary detection**
During training, sequences are neatly labelled and pre-segmented. At inference time, the system has to figure out on its own where each gesture starts and ends. Getting the segmenter to not fire on accidental movement, not miss short gestures, and not chop a gesture mid-motion required a lot of manual tuning and real-world testing.

**Keeping the video feed and predictions in sync**
The webcam frames and the LSTM predictions are produced in a background loop, but they need to be served to the browser across multiple HTTP connections simultaneously. Getting this to work without blocking, dropping frames, or causing weird timing issues took more thought than expected.

**NumPy 2.x compatibility**
NumPy 2.x broke compatibility with older TensorFlow versions in non-obvious ways. Pinning `numpy<2.0` in the requirements file was the straightforward fix, but it wasn't immediately obvious that NumPy was the culprit.

---

## Limitations

The system currently recognises a **curated subset of ISL gestures**. This is intentional — starting with a smaller, well-defined gesture set made it much easier to focus on getting the real-time pipeline right, tuning the segmenter, and building stable frontend-backend integration. Expanding the vocabulary means collecting more training data, which is planned.

Prediction quality is sensitive to lighting and background. The model was trained under specific conditions, so it may not generalise perfectly to every environment straight away.

Currently tested primarily on Windows. The backend itself should run on Linux and macOS with minor adjustments, but the `setup_and_run.bat` script is Windows-only for now.

---

## Future Improvements

- Expand the gesture vocabulary with more ISL signs and training data
- Add a data collection mode in the UI to record and label new gestures without leaving the app
- Experiment with bidirectional LSTM or attention-based models for better sequence understanding
- Export to TensorFlow Lite for mobile or offline use
- Improve sentence-level output — smarter duplicate filtering, punctuation handling
- Linux/macOS setup script (`setup_and_run.sh`)
- Real-time confidence graph and per-class prediction breakdown in the UI

---

## Credits & Acknowledgements

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — hand landmark detection
- [TensorFlow / Keras](https://www.tensorflow.org/) — model training and inference
- [Flask](https://flask.palletsprojects.com/) — lightweight Python web backend
- [OpenCV](https://opencv.org/) — camera capture and frame annotation
- The open-source ISL research community for gesture references and dataset inspiration

---

<div align="center">

*Built frame by frame, landmark by landmark. Still signing. Still learning.*

</div>
