"""Landmark stabilisation utilities (anti-jitter / anti-"mleyot" / anti-missing).

Pure-numpy, unit-testable building blocks used by the ``holistic_stabilized``
extraction method (offline studio + live):

- :class:`OneEuroFilter` — vectorised 1€ filter: low jitter, low lag.
- :class:`LandmarkStreamStabilizer` — forward-only per-stream wrapper that adds
  velocity-gated outlier rejection (drops a frame that teleports too far, the
  "mleyot" case) and short-gap holding with score decay (rides out brief
  detection dropouts instead of snapping to nothing).
- :func:`resolve_hand_swap` — left<->right swap guard, ported from the v6 tracker.
- :func:`fill_gaps_bidirectional` — offline-only gap interpolation that uses
  future frames, so it is never used live.

All geometry is expressed in the same shoulder scale smart180 uses, so jump
thresholds are resolution-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Vectorised 1€ filter (Casiez et al.) over an arbitrary-shaped array.

    The cutoff frequency rises with the estimated speed, so slow motion is
    smoothed hard (low jitter) while fast motion passes through (low lag).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()

        dt = max(float(dt), 1e-6)
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        # Per-element adaptive alpha; vectorised across the array.
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
        a_x = 1.0 / (1.0 + tau / dt)
        x_hat = a_x * x + (1.0 - a_x) * self._x_prev

        self._x_prev = x_hat.astype(np.float32)
        self._dx_prev = dx_hat.astype(np.float32)
        return self._x_prev.copy()


@dataclass(frozen=True)
class StabilizerConfig:
    min_cutoff: float = 1.0
    beta: float = 0.5
    d_cutoff: float = 1.0
    # Max wrist speed (in shoulder-scales per second) before a frame is rejected.
    jump_factor: float = 8.0
    # Hold a vanished landmark group for at most this many frames before dropping.
    hold_frames: int = 5
    # Confidence multiplier applied each held frame.
    hold_decay: float = 0.60


@dataclass
class StabilizerStep:
    points: np.ndarray | None
    present: bool
    held: bool
    rejected_jump: bool
    score: float


class LandmarkStreamStabilizer:
    """Forward-only stabiliser for one landmark group (e.g. one hand).

    Live-safe: it never looks at future frames. Combines 1€ smoothing, a
    velocity gate that rejects teleporting detections, and a hold-with-decay
    window that survives brief dropouts.
    """

    def __init__(self, config: StabilizerConfig | None = None) -> None:
        self.cfg = config or StabilizerConfig()
        self._filter = OneEuroFilter(self.cfg.min_cutoff, self.cfg.beta, self.cfg.d_cutoff)
        self._last: np.ndarray | None = None
        self._held = 0
        self._score = 0.0

    def reset(self) -> None:
        self._filter.reset()
        self._last = None
        self._held = 0
        self._score = 0.0

    def _is_jump(self, pts: np.ndarray, shoulder_scale: float, dt: float) -> bool:
        if self._last is None:
            return False
        wrist_disp = float(np.linalg.norm(pts[0, :2] - self._last[0, :2]))
        threshold = self.cfg.jump_factor * max(float(shoulder_scale), 1e-4) * max(float(dt), 1e-3)
        return wrist_disp > threshold

    def update(
        self,
        points: np.ndarray | None,
        present: bool,
        shoulder_scale: float,
        dt: float,
    ) -> StabilizerStep:
        rejected = False
        if present and points is not None:
            pts = np.asarray(points, dtype=np.float32)
            if self._is_jump(pts, shoulder_scale, dt):
                rejected = True  # teleport -> treat this frame as missing
            else:
                filtered = self._filter(pts, dt)
                self._last = filtered.copy()
                self._held = 0
                self._score = 1.0
                return StabilizerStep(filtered, True, False, False, 1.0)

        # Missing this frame (truly absent, or rejected as a jump): try to hold.
        if self._last is not None and self._held < self.cfg.hold_frames:
            self._held += 1
            self._score *= self.cfg.hold_decay
            return StabilizerStep(self._last.copy(), True, True, rejected, self._score)

        # Gap too long -> drop and reset so the next detection starts clean.
        self._filter.reset()
        self._last = None
        self._held = 0
        self._score = 0.0
        return StabilizerStep(None, False, False, rejected, 0.0)


def resolve_hand_swap(
    left: np.ndarray | None,
    right: np.ndarray | None,
    prev_left: np.ndarray | None,
    prev_right: np.ndarray | None,
    margin: float = 0.030,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Swap left/right if the cross assignment matches the previous frame better.

    Ported from the v6 tracker ``_swap_guard``: compares total wrist displacement
    for the direct vs swapped assignment and flips only when the swap wins by
    ``margin``. No-op when either current or previous hand is missing.
    """
    if left is None or right is None or prev_left is None or prev_right is None:
        return left, right
    lw = np.asarray(left)[0, :2]
    rw = np.asarray(right)[0, :2]
    plw = np.asarray(prev_left)[0, :2]
    prw = np.asarray(prev_right)[0, :2]
    direct = float(np.linalg.norm(lw - plw) + np.linalg.norm(rw - prw))
    cross = float(np.linalg.norm(lw - prw) + np.linalg.norm(rw - plw))
    if cross + float(margin) < direct:
        return right, left
    return left, right


def fill_gaps_bidirectional(
    points_seq: np.ndarray,
    present_seq: np.ndarray,
    max_gap: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate missing frames bounded on both sides (offline only).

    A run of missing frames is filled only when both neighbours exist and the run
    length is ``<= max_gap``. Leading/trailing gaps (unbounded on one side) and
    runs longer than ``max_gap`` are left missing. Returns ``(filled_points,
    filled_present)`` without mutating the inputs.
    """
    points = np.array(points_seq, dtype=np.float32, copy=True)
    present = np.array(present_seq, dtype=bool, copy=True)
    n = present.shape[0]
    idx = 0
    while idx < n:
        if present[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and not present[idx]:
            idx += 1
        end = idx  # first present frame after the gap (exclusive end of gap)
        gap_len = end - start
        if start > 0 and end < n and gap_len <= int(max_gap):
            left_pts = points[start - 1]
            right_pts = points[end]
            for k in range(gap_len):
                t = (k + 1) / (gap_len + 1)
                points[start + k] = (1.0 - t) * left_pts + t * right_pts
                present[start + k] = True
    return points, present
