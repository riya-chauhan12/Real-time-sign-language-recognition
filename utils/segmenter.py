"""
segmenter.py — Gesture start/end detector (unchanged from v3).
Works on the (258,) feature vector — reads only the position slice [0:126].
"""

import numpy as np
import collections
from enum import Enum, auto


class SegState(Enum):
    IDLE    = auto()
    SIGNING = auto()
    HOLD    = auto()


class GestureSegmenter:
    """
    Three-state machine that detects gesture boundaries.

    onset_thresh  : motion energy threshold to leave IDLE and start capturing
    offset_thresh : motion must fall below this to begin the hold countdown
    hold_frames   : consecutive low-motion frames needed to declare gesture complete
    min_frames    : minimum captured frames to be accepted (rejects accidental triggers)
    max_frames    : hard cap on buffer length (force-complete very long gestures)
    target_len    : output length after resampling (must equal SEQ_LEN = 30)
    """

    def __init__(
        self,
        onset_thresh:  float = 0.008,
        offset_thresh: float = 0.005,
        hold_frames:   int   = 8,
        min_frames:    int   = 12,
        max_frames:    int   = 90,
        target_len:    int   = 30,
    ):
        self.onset_thresh  = onset_thresh
        self.offset_thresh = offset_thresh
        self.hold_frames   = hold_frames
        self.min_frames    = min_frames
        self.max_frames    = max_frames
        self.target_len    = target_len

        self._state       = SegState.IDLE
        self._gesture_buf = []
        self._hold_count  = 0
        self._motion_hist = collections.deque(maxlen=5)

    @property
    def state(self) -> SegState:
        return self._state

    @property
    def buffer_len(self) -> int:
        return len(self._gesture_buf)

    @property
    def hold_progress(self) -> float:
        if self._state != SegState.HOLD:
            return 0.0
        return min(1.0, self._hold_count / self.hold_frames)

    def update(self, feature: np.ndarray) -> np.ndarray | None:
        """
        Feed one frame. Returns (target_len, feat_dim) when gesture completes.
        Uses position slice [0:126] for motion detection.
        """
        pos    = feature[:126]
        self._motion_hist.append(float(np.mean(np.abs(pos))))
        motion = float(np.std(list(self._motion_hist))) \
                 if len(self._motion_hist) >= 3 else 0.0

        if self._state == SegState.IDLE:
            if motion >= self.onset_thresh:
                self._state       = SegState.SIGNING
                self._gesture_buf = [feature]
                self._hold_count  = 0

        elif self._state == SegState.SIGNING:
            self._gesture_buf.append(feature)
            if len(self._gesture_buf) >= self.max_frames:
                return self._try_complete()
            if motion < self.offset_thresh:
                self._state      = SegState.HOLD
                self._hold_count = 1

        elif self._state == SegState.HOLD:
            self._gesture_buf.append(feature)
            if motion >= self.offset_thresh:
                self._state      = SegState.SIGNING
                self._hold_count = 0
            else:
                self._hold_count += 1
                if self._hold_count >= self.hold_frames:
                    return self._try_complete()

        return None

    def reset(self):
        self._state       = SegState.IDLE
        self._gesture_buf = []
        self._hold_count  = 0
        self._motion_hist.clear()

    def _try_complete(self) -> np.ndarray | None:
        buf = self._gesture_buf.copy()
        self.reset()
        if len(buf) < self.min_frames:
            return None
        return _resample(np.array(buf, dtype=np.float32), self.target_len)


def _resample(seq: np.ndarray, target_len: int) -> np.ndarray:
    n = len(seq)
    if n == target_len:
        return seq
    src_t = np.linspace(0, 1, n)
    dst_t = np.linspace(0, 1, target_len)
    out   = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(dst_t, src_t, seq[:, d])
    return out
