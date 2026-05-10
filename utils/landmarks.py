"""
landmarks.py — Canonical feature extraction.

SHARED between training (train_kaggle_final.py) and inference (main.py).
Any change here must be reflected in the training script and vice versa.

Feature vector layout per frame (258 dimensions total):
  [0  :126]  Position   — 2 hands × 21 landmarks × 3 coords, wrist-normalised
  [126:252]  Velocity   — frame-to-frame delta of ALL positions (both hands, 126)
  [252:258]  Meta       — [present_h0, conf_h0, vis_h0, present_h1, conf_h1, vis_h1]

Uses MediaPipe Tasks API (HandLandmarker) — avoids version clashes on Kaggle
and on local systems where mediapipe ≥ 0.10 is installed.
"""

import numpy as np

# ── Dimension constants  ──────────────────────────────
POS_DIM     = 126   # 2 × 21 × 3
VEL_DIM     = 126   # 2 × 21 × 3  (both hands)
META_DIM    = 6     # 2 × 3
FEATURE_DIM = POS_DIM + VEL_DIM + META_DIM   # 258
SEQ_LEN     = 30


# ── EMA landmark smoother ─────────────────────────────────────────────────────
class LandmarkSmoother:
    """
    Exponential moving average per coordinate to suppress MediaPipe jitter.
    alpha=1.0 → no smoothing (raw output).
    alpha=0.65 → moderate smoothing used in live inference.
    """

    def __init__(self, alpha: float = 0.65):
        self.alpha  = alpha
        self._state = None

    def smooth(self, lm: np.ndarray) -> np.ndarray:
        """lm: (126,) raw position vector."""
        if self._state is None:
            self._state = lm.copy()
            return lm
        # Only smooth hands that are currently detected (non-zero)
        mask        = (lm != 0).astype(np.float32)
        self._state = self.alpha * lm + (1.0 - self.alpha) * self._state
        return self._state * mask + lm * (1.0 - mask)

    def reset(self):
        self._state = None


# ── Core per-frame extraction  ─────────────────────────────
def extract_from_task_result(result) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract raw position + meta from a HandLandmarker result object.

    Parameters
    ----------
    result : mediapipe.tasks.python.vision.HandLandmarkerResult

    Returns
    -------
    lm   : (126,) raw (un-normalised) landmark coordinates
    meta : (6,)
    """
    lm   = np.zeros(POS_DIM,  dtype=np.float32)
    meta = np.zeros(META_DIM, dtype=np.float32)

    if not result.hand_landmarks:
        return lm, meta

    handedness = result.handedness if result.handedness else []

    for i, hand in enumerate(result.hand_landmarks[:2]):
        base = i * 63
        for j, pt in enumerate(hand):
            lm[base + j*3]     = pt.x
            lm[base + j*3 + 1] = pt.y
            lm[base + j*3 + 2] = pt.z

        conf              = handedness[i][0].score if i < len(handedness) else 0.5
        meta[i*3]         = 1.0    # hand present
        meta[i*3 + 1]     = conf
        meta[i*3 + 2]     = 1.0   # assume all landmarks visible (Tasks API)

    return lm, meta


# ── Normalisation ─────────────────────────────────────────────────────────────
def normalize(lm: np.ndarray) -> np.ndarray:
    """
    Per-hand: wrist (landmark 0) → origin, scale to [-1, +1] unit cube.
    Absent hands (all zeros) are left as zeros.
    """
    out = lm.copy().reshape(2, 21, 3)
    for h in range(2):
        if np.all(out[h] == 0.0):
            continue
        out[h] -= out[h][0]
        s = np.max(np.abs(out[h]))
        if s > 1e-6:
            out[h] /= s
    return out.flatten()


# ── Velocity ──────────────────────────────────────────────────────────────────
def compute_velocity(prev_lm: np.ndarray, curr_lm: np.ndarray) -> np.ndarray:
    """
    Frame-to-frame position delta for BOTH hands (126-dim).
    Returns zeros if either frame had no detected hands.
    """
    if np.all(prev_lm == 0) or np.all(curr_lm == 0):
        return np.zeros(VEL_DIM, dtype=np.float32)
    return (curr_lm - prev_lm).astype(np.float32)


# ── Stateful frame extractor (used in inference main.py) ─────────────────────
class FrameFeatureExtractor:
    """
    Wraps one call per webcam frame and maintains the previous-frame state
    needed for velocity computation and EMA smoothing.

    Usage
    -----
        extractor = FrameFeatureExtractor()
        extractor.reset()          # call when starting a new session

        # inside webcam loop:
        result  = landmarker.detect(mp_image)
        feature = extractor.process(result)   # → (258,) numpy array
    """

    def __init__(self, smooth_alpha: float = 0.65):
        self._smoother  = LandmarkSmoother(alpha=smooth_alpha)
        self._prev_norm = np.zeros(POS_DIM, dtype=np.float32)

    def process(self, result) -> np.ndarray:
        """result: HandLandmarkerResult from mediapipe Tasks API."""
        raw, meta    = extract_from_task_result(result)
        smoothed     = self._smoother.smooth(raw)
        norm         = normalize(smoothed)
        vel          = compute_velocity(self._prev_norm, norm)
        self._prev_norm = norm.copy()
        return np.concatenate([norm, vel, meta]).astype(np.float32)   # (258,)

    def reset(self):
        self._smoother.reset()
        self._prev_norm = np.zeros(POS_DIM, dtype=np.float32)


# ── Hand presence helper ──────────────────────────────────────────────────────
def hand_detected(result) -> bool:
    """True if at least one hand is visible in this Tasks API result."""
    return bool(result.hand_landmarks)
