"""
backend/app.py —  Flask Backend ( TF 2.13–2.21 + Keras 2/3)
"""

import os, sys, json, time, threading, queue, inspect
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS

from utils.landmarks import FrameFeatureExtractor, hand_detected, FEATURE_DIM, SEQ_LEN
from utils.predictor  import GesturePredictor
from utils.segmenter  import SegState

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "isl_model_final.keras")
META_PATH  = os.path.join(ROOT, "models", "model_meta.json")
TASK_PATH  = os.path.join(ROOT, "models", "hand_landmarker.task")
STATIC_DIR = os.path.join(ROOT, "frontend", "static")
TMPL_DIR   = os.path.join(ROOT, "frontend", "templates")

# ── File checks ────────────────────────────────────────────────────────────────
for fpath, hint in [
    (MODEL_PATH, "isl_model_final.h5  — copy from Kaggle output"),
    (META_PATH,  "model_meta.json     — copy from Kaggle output"),
    (TASK_PATH,  "hand_landmarker.task — download from MediaPipe"),
]:
    if not os.path.exists(fpath):
        print(f"\n  Missing: {fpath}")
        print(f"  Needed:  {hint}\n")
        sys.exit(1)

# ── Model loading  ────────────
print("Loading model ...")

def load_keras_model(path):
    """
    Robust loader that handles:
      - TF 2.13-2.15  (Keras 2, standard load_model)
      - TF 2.16+      (Keras 3, requires tf_keras for .h5)
      - Old .h5 files with 'batch_shape' in InputLayer config (Keras < 2.4)
    """
    import tensorflow as tf
    print(f"  TensorFlow {tf.__version__}")

    tf_major, tf_minor = [int(x) for x in tf.__version__.split(".")[:2]]

    # ── Strategy 1: tf_keras (best for TF 2.16+ with old .h5 files) ──────────
    try:
        import tf_keras
        print(f"  Trying tf_keras {tf_keras.__version__} ...")
        from tf_keras.src.layers import InputLayer as TFKInputLayer

        class PatchedInputLayer(TFKInputLayer):
            def __init__(self, *args, **kwargs):
                kwargs.pop("batch_shape", None)
                kwargs.pop("batch_input_shape", None)
                super().__init__(*args, **kwargs)

        m = tf_keras.models.load_model(
            path,
            custom_objects={"InputLayer": PatchedInputLayer},
            compile=False,
        )
        print("  ✓ Loaded via tf_keras with InputLayer patch")
        return m
    except ImportError:
        print("  tf_keras not installed — trying next strategy")
    except Exception as e:
        print(f"  tf_keras failed ({e}) — trying next strategy")

    # ── Strategy 2: tensorflow.keras with InputLayer patch ────────────────────
    try:
        from tensorflow.python.keras.layers import InputLayer as LegacyInputLayer

        class PatchedInputLayer2(LegacyInputLayer):
            def __init__(self, *args, **kwargs):
                kwargs.pop("batch_shape", None)
                kwargs.pop("batch_input_shape", None)
                super().__init__(*args, **kwargs)

        from tensorflow.keras.models import load_model as _lm
        m = _lm(path, custom_objects={"InputLayer": PatchedInputLayer2}, compile=False)
        print("  ✓ Loaded via tensorflow.python.keras with InputLayer patch")
        return m
    except Exception as e:
        print(f"  Legacy keras patch failed ({e}) — trying next strategy")

    # ── Strategy 3: plain tensorflow.keras.models.load_model ─────────────────
    try:
        from tensorflow.keras.models import load_model as _lm
        from tensorflow.keras.layers import InputLayer

        class PatchedInputLayer3(InputLayer):
            def __init__(self, *args, **kwargs):
                kwargs.pop("batch_shape", None)
                kwargs.pop("batch_input_shape", None)
                super().__init__(*args, **kwargs)

        m = _lm(path, custom_objects={"InputLayer": PatchedInputLayer3}, compile=False)
        print("  ✓ Loaded via tensorflow.keras with InputLayer patch")
        return m
    except Exception as e:
        print(f"  tensorflow.keras patch failed ({e}) — trying next strategy")

    # ── Strategy 4: h5py raw config surgery (last resort) ─────────────────────
    try:
        print("  Attempting h5py config surgery ...")
        import h5py, tempfile, shutil

        tmp = path + ".patched.h5"
        shutil.copy2(path, tmp)

        with h5py.File(tmp, "r+") as f:
            # The model config is stored as a JSON string attribute
            if "model_config" in f.attrs:
                cfg_str = f.attrs["model_config"]
                if isinstance(cfg_str, bytes):
                    cfg_str = cfg_str.decode("utf-8")
                cfg = json.loads(cfg_str)

                def _patch_config(obj):
                    if isinstance(obj, dict):
                        if obj.get("class_name") == "InputLayer":
                            c = obj.get("config", {})
                            bs = c.pop("batch_shape", None)
                            bis = c.pop("batch_input_shape", None)
                            # reconstruct batch_size + shape from batch_shape
                            if bs is not None and "batch_size" not in c:
                                c["batch_size"] = bs[0]  # may be None
                                if "sparse" not in c:
                                    c["sparse"] = False
                                if "ragged" not in c:
                                    c["ragged"] = False
                            elif bis is not None and "batch_size" not in c:
                                c["batch_size"] = bis[0]
                        for v in obj.values():
                            _patch_config(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            _patch_config(item)

                _patch_config(cfg)
                new_cfg = json.dumps(cfg)
                f.attrs["model_config"] = new_cfg.encode("utf-8")

        from tensorflow.keras.models import load_model as _lm
        m = _lm(tmp, compile=False)
        os.remove(tmp)
        print("  ✓ Loaded via h5py config surgery")
        return m
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"All model loading strategies failed.\n"
            f"Last error: {e}\n\n"
            f"SOLUTION: Convert your model to the newer .keras format by running:\n"
            f"  python fix_model.py\n"
            f"(The fix_model.py script was provided alongside this fix.)"
        )


try:
    model = load_keras_model(MODEL_PATH)
except Exception as e:
    print(f"\n  ✗ Model load failed:\n  {e}\n")
    sys.exit(1)

with open(META_PATH) as f:
    META = json.load(f)

LABELS      = META["labels"]
TEMPERATURE = META.get("temperature", 1.0)

import tensorflow as tf
print(f"  Classes: {len(LABELS)} | Input shape: {model.input_shape}")

# ── MediaPipe HandLandmarker  ────────────────────────────────────
print("Initialising MediaPipe ...")
_base = mp_python.BaseOptions(model_asset_path=TASK_PATH)

# Detect correct parameter name — changed between mediapipe versions
_landmarker_params = inspect.signature(mp_vision.HandLandmarkerOptions.__init__).parameters
_presence_kwarg = (
    "min_hand_presence_score"       # mediapipe < 0.10.8
    if "min_hand_presence_score"    in _landmarker_params
    else "min_hand_presence_confidence"   # mediapipe >= 0.10.8
)

_opts_kwargs = dict(
    base_options                  = _base,
    num_hands                     = 2,
    min_hand_detection_confidence = 0.60,
    min_tracking_confidence       = 0.55,
    running_mode                  = mp_vision.RunningMode.IMAGE,
)
_opts_kwargs[_presence_kwarg] = 0.55   # inject the right param name

_opts = mp_vision.HandLandmarkerOptions(**_opts_kwargs)
landmarker = mp_vision.HandLandmarker.create_from_options(_opts)
print(f"  HandLandmarker ready (mediapipe {mp.__version__}, presence param: {_presence_kwarg})")


# ── Shared inference state ─────────────────────────────────────────────────────
class InferenceState:
    def __init__(self):
        self._lock      = threading.Lock()
        self.seg_state  = "IDLE"
        self.hold_pct   = 0.0
        self.word       = ""
        self.conf       = 0.0
        self.margin     = 0.0
        self.hand       = False
        self.sentence   = []
        self.last_commit = ""
        self.commit_time = 0.0
        self._sse_queues = []

    def snapshot(self):
        with self._lock:
            return dict(
                seg_state   = self.seg_state,
                hold_pct    = round(self.hold_pct, 3),
                word        = self.word,
                conf        = round(self.conf, 3),
                margin      = round(self.margin, 3),
                hand        = self.hand,
                sentence    = list(self.sentence),
                last_commit = self.last_commit,
                commit_age  = round(time.time() - self.commit_time, 2),
            )

    def update_frame(self, seg_state, hold_pct, word, conf, margin, hand):
        with self._lock:
            self.seg_state = seg_state
            self.hold_pct  = hold_pct
            self.word      = word
            self.conf      = conf
            self.margin    = margin
            self.hand      = hand

    def commit_word(self, word, conf):
        with self._lock:
            self.sentence.append(word)
            self.last_commit = word
            self.commit_time = time.time()
            payload = json.dumps({"type": "commit", "word": word,
                                  "conf": round(conf, 3),
                                  "sentence": list(self.sentence)})
            for q in self._sse_queues:
                try: q.put_nowait(payload)
                except queue.Full: pass

    def clear(self):
        with self._lock:
            self.sentence.clear()
            self.last_commit = ""
            payload = json.dumps({"type": "clear"})
            for q in self._sse_queues:
                try: q.put_nowait(payload)
                except queue.Full: pass

    def subscribe(self):
        q = queue.Queue(maxsize=20)
        with self._lock:
            self._sse_queues.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            try: self._sse_queues.remove(q)
            except ValueError: pass

STATE = InferenceState()

# ── MJPEG frame buffer ─────────────────────────────────────────────────────────
_frame_lock   = threading.Lock()
_latest_frame = None

def _set_frame(bgr):
    global _latest_frame
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if ok:
        with _frame_lock:
            _latest_frame = buf.tobytes()

def _get_frame():
    with _frame_lock:
        return _latest_frame

# ── Skeleton drawing ───────────────────────────────────────────────────────────
_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
         (13,17),(17,18),(18,19),(19,20),(0,17)]

def _draw_skeleton(frame, result):
    h, w = frame.shape[:2]
    for hand in result.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in hand]
        for a, b in _CONN:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)

def _draw_state_pill(frame, snap):
    h, w = frame.shape[:2]
    seg = snap["seg_state"]
    col = {"IDLE": (90, 90, 90), "SIGNING": (0, 200, 80),
           "HOLD": (0, 165, 255)}.get(seg, (90, 90, 90))
    label = seg if seg != "HOLD" else f"HOLD {int(snap['hold_pct']*100)}%"
    pad, th = 10, 28
    tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
    x1, y1 = w - tw - 2 * pad - 10, 10
    x2, y2 = w - 10, th + 10
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
    cv2.putText(frame, label, (x1 + pad, y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    dot = (0, 200, 80) if snap["hand"] else (60, 60, 200)
    cv2.circle(frame, (20, 20), 8, dot, -1)

# ── Inference thread ───────────────────────────────────────────────────────────
def _inference_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("  ERROR: Cannot open webcam (index 0).")
        print("  Try changing VideoCapture(0) to VideoCapture(1) in backend/app.py")
        return

    extractor = FrameFeatureExtractor(smooth_alpha=0.65)
    predictor = GesturePredictor(
        model            = model,
        labels           = LABELS,
        seq_len          = SEQ_LEN,
        conf_threshold   = 0.72,
        margin_threshold = 0.18,
        temperature      = TEMPERATURE,
        cooldown         = 2.5,
        segmenter_kwargs = dict(
            onset_thresh  = 0.008,
            offset_thresh = 0.005,
            hold_frames   = 8,
            min_frames    = 12,
            max_frames    = 90,
        ),
    )

    print("  Inference thread started — open http://localhost:5050")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
        hv     = hand_detected(result)

        if hv:
            _draw_skeleton(frame, result)

        feat   = extractor.process(result)
        commit = predictor.update(feat)

        if commit:
            word, conf = commit
            STATE.commit_word(word, conf)
            print(f"    Committed: '{word}'  ({conf:.0%})")

        STATE.update_frame(
            seg_state = predictor.seg_state.name,
            hold_pct  = predictor.hold_progress(),
            word      = predictor.current_word,
            conf      = predictor.current_conf,
            margin    = predictor.current_margin,
            hand      = hv,
        )

        _draw_state_pill(frame, STATE.snapshot())
        _set_frame(frame)

    cap.release()


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TMPL_DIR)
CORS(app)

@app.route("/")
def index():
    return send_from_directory(TMPL_DIR, "index.html")

@app.route("/api/state")
def api_state():
    return jsonify(STATE.snapshot())

@app.route("/api/stream")
def api_stream():
    def generate():
        q = STATE.subscribe()
        try:
            yield "data: {\"type\":\"ping\"}\n\n"
            while True:
                try:
                    event = q.get(timeout=15)
                    yield f"data: {event}\n\n"
                except queue.Empty:
                    yield "data: {\"type\":\"ping\"}\n\n"
        except GeneratorExit:
            STATE.unsubscribe(q)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Access-Control-Allow-Origin": "*"})

@app.route("/api/video_feed")
def video_feed():
    def generate():
        while True:
            buf = _get_frame()
            if buf:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
            else:
                time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/clear", methods=["POST"])
def api_clear():
    STATE.clear()
    return jsonify({"ok": True})

@app.route("/api/config")
def api_config():
    return jsonify({"labels": LABELS, "num_classes": len(LABELS)})


if __name__ == "__main__":
    t = threading.Thread(target=_inference_loop, daemon=True)
    t.start()
    time.sleep(2)
    print("\n  Open your browser at  http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, threaded=True, debug=False)
