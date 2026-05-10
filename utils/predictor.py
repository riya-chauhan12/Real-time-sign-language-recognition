"""
predictor.py — Stable gesture prediction engine.

Compatible with:
  • FEATURE_DIM = 258  (pos=126, vel=126, meta=6)
  • SEQ_LEN = 30
  • HandLandmarker Tasks API (inference pipeline)
  • Model trained by train_kaggle_final.py
"""

import time
import numpy as np
from .segmenter import GestureSegmenter, SegState


class GesturePredictor:
    """
    Parameters
    ----------
    model            : Keras model, input (1, 30, 258) → (num_classes,)
    labels           : list of class name strings (order must match training)
    seq_len          : 30 (must match training)
    conf_threshold   : minimum softmax confidence to commit a word
    margin_threshold : minimum gap between top-1 and top-2 probability
                       (prevents committing when two classes are nearly tied)
    temperature      : softmax temperature from model_meta.json
    cooldown         : min seconds before same word can be committed again
    segmenter_kwargs : passed to GestureSegmenter constructor
    """

    def __init__(
        self,
        model,
        labels:            list,
        seq_len:           int   = 30,
        conf_threshold:    float = 0.70,
        margin_threshold:  float = 0.18,
        temperature:       float = 1.0,
        cooldown:          float = 2.0,
        segmenter_kwargs:  dict  = None,
    ):
        self.model             = model
        self.labels            = labels
        self.seq_len           = seq_len
        self.conf_threshold    = conf_threshold
        self.margin_threshold  = margin_threshold
        self.temperature       = temperature
        self.cooldown          = cooldown

        seg_kw = segmenter_kwargs or {}
        self.segmenter = GestureSegmenter(target_len=seq_len, **seg_kw)

        self._last_word  = ""
        self._last_add_t = 0.0

        # Always-available display values
        self.current_word   = ""
        self.current_conf   = 0.0
        self.current_margin = 0.0

    # ── Main update loop ───────────────────────────────────────────────────────

    def update(self, feature: np.ndarray) -> tuple | None:
        """
        Feed one frame's (258,) feature vector.
        Returns (word, confidence) when a complete gesture is committed, else None.
        """
        segment = self.segmenter.update(feature)
        if segment is None:
            return None
        return self._predict_segment(segment)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        if abs(self.temperature - 1.0) < 1e-4:
            return probs
        logits  = np.log(np.clip(probs, 1e-9, 1.0)) / self.temperature
        logits -= logits.max()
        exp     = np.exp(logits)
        return exp / exp.sum()

    def _predict_segment(self, segment: np.ndarray) -> tuple | None:
        """segment: (seq_len, 258)"""
        inp   = segment[np.newaxis]                        # (1, 30, 258)
        raw   = self.model.predict(inp, verbose=0)[0]      # (num_classes,)
        probs = self._apply_temperature(raw)

        sorted_idx = np.argsort(probs)[::-1]
        idx    = sorted_idx[0]
        conf   = float(probs[idx])
        margin = float(probs[idx] - probs[sorted_idx[1]]) if len(probs) > 1 else 1.0
        word   = self.labels[idx]

        self.current_word   = word
        self.current_conf   = conf
        self.current_margin = margin

        now = time.time()
        if (conf   >= self.conf_threshold
                and margin >= self.margin_threshold
                and not (word == self._last_word
                         and (now - self._last_add_t) < self.cooldown)):
            self._last_word  = word
            self._last_add_t = now
            return (word, conf)

        return None

    # ── State helpers ──────────────────────────────────────────────────────────

    @property
    def seg_state(self) -> SegState:
        return self.segmenter.state

    def hold_progress(self) -> float:
        return self.segmenter.hold_progress

    def reset(self):
        self.segmenter.reset()
        self._last_word    = ""
        self._last_add_t   = 0.0
        self.current_word  = ""
        self.current_conf  = 0.0
        self.current_margin = 0.0

    def set_threshold(self, t: float):
        self.conf_threshold = max(0.0, min(1.0, t))

    def set_cooldown(self, s: float):
        self.cooldown = max(0.0, s)
