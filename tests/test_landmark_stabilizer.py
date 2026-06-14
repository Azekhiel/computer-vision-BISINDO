"""Tests for src/landmark_stabilizer.py (anti-jitter / anti-mleyot / anti-missing)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import landmark_stabilizer as ls  # noqa: E402


def _hand(wrist_xy, spread=0.02):
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[:, 0] = wrist_xy[0]
    pts[:, 1] = wrist_xy[1]
    pts[0] = np.array([wrist_xy[0], wrist_xy[1], 0.0], dtype=np.float32)
    pts[1:, 0] += np.linspace(-spread, spread, 20)
    return pts


def test_one_euro_reduces_variance_while_tracking_trend():
    rng = np.random.default_rng(0)
    clean = np.linspace(0.0, 1.0, 240, dtype=np.float32)  # slow ramp (near-DC, minimal lag)
    noisy = clean + rng.normal(0, 0.08, size=clean.shape).astype(np.float32)

    filt = ls.OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    out = np.array([filt(np.array([noisy[i]], dtype=np.float32), dt=1 / 60)[0] for i in range(len(noisy))])

    # jitter (frame-to-frame diff) must drop substantially
    raw_jitter = np.mean(np.abs(np.diff(noisy)))
    filt_jitter = np.mean(np.abs(np.diff(out)))
    assert filt_jitter < raw_jitter * 0.5
    # denoising: filtered output tracks the ramp better than the noisy input
    rmse_noisy = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    rmse_filt = float(np.sqrt(np.mean((out[5:] - clean[5:]) ** 2)))
    assert rmse_filt < rmse_noisy * 0.7


def test_one_euro_first_call_is_passthrough():
    filt = ls.OneEuroFilter()
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(filt(x, dt=0.1), x)


def test_jump_frame_is_rejected_and_held():
    cfg = ls.StabilizerConfig(jump_factor=8.0, hold_frames=5)
    stab = ls.LandmarkStreamStabilizer(cfg)
    scale, dt = 0.2, 0.1

    s0 = stab.update(_hand((0.4, 0.5)), True, scale, dt)
    assert s0.present and not s0.rejected_jump
    s1 = stab.update(_hand((0.41, 0.5)), True, scale, dt)
    assert s1.present and not s1.rejected_jump

    # huge teleport: 0.4 shoulder-units >> 8 * 0.2 * 0.1 = 0.16 threshold
    s2 = stab.update(_hand((0.95, 0.5)), True, scale, dt)
    assert s2.rejected_jump
    assert s2.held  # held the previous good frame instead of snapping
    np.testing.assert_allclose(s2.points[0, :2], s1.points[0, :2], atol=1e-5)
    assert s2.score < 1.0  # decayed


def test_small_motion_is_not_rejected():
    cfg = ls.StabilizerConfig(jump_factor=8.0)
    stab = ls.LandmarkStreamStabilizer(cfg)
    stab.update(_hand((0.4, 0.5)), True, 0.2, 0.1)
    step = stab.update(_hand((0.43, 0.5)), True, 0.2, 0.1)  # within threshold 0.16
    assert step.present and not step.rejected_jump and not step.held


def test_hold_then_drop_after_max_frames():
    cfg = ls.StabilizerConfig(hold_frames=3, hold_decay=0.5)
    stab = ls.LandmarkStreamStabilizer(cfg)
    stab.update(_hand((0.4, 0.5)), True, 0.2, 0.1)

    scores = []
    for _ in range(3):
        s = stab.update(None, False, 0.2, 0.1)
        assert s.present and s.held
        scores.append(s.score)
    assert scores == [0.5, 0.25, 0.125]
    # 4th missing frame exhausts the hold window -> truly gone
    s = stab.update(None, False, 0.2, 0.1)
    assert not s.present and s.points is None and s.score == 0.0


def test_swap_guard_swaps_when_cross_matches_previous():
    prev_left = _hand((0.3, 0.5))
    prev_right = _hand((0.7, 0.5))
    # current hands arrive with labels swapped relative to previous positions
    cur_left = _hand((0.7, 0.5))
    cur_right = _hand((0.3, 0.5))
    l, r = ls.resolve_hand_swap(cur_left, cur_right, prev_left, prev_right)
    np.testing.assert_allclose(l[0, :2], prev_left[0, :2], atol=1e-6)
    np.testing.assert_allclose(r[0, :2], prev_right[0, :2], atol=1e-6)


def test_swap_guard_keeps_direct_when_already_consistent():
    prev_left = _hand((0.3, 0.5))
    prev_right = _hand((0.7, 0.5))
    cur_left = _hand((0.31, 0.5))
    cur_right = _hand((0.69, 0.5))
    l, r = ls.resolve_hand_swap(cur_left, cur_right, prev_left, prev_right)
    np.testing.assert_allclose(l[0, :2], cur_left[0, :2], atol=1e-6)
    np.testing.assert_allclose(r[0, :2], cur_right[0, :2], atol=1e-6)


def test_swap_guard_noop_when_any_missing():
    left = _hand((0.3, 0.5))
    assert ls.resolve_hand_swap(left, None, left, left) == (left, None)


def test_fill_gaps_bidirectional_fills_short_gap_linearly():
    T, N = 6, 21
    pts = np.zeros((T, N, 3), dtype=np.float32)
    present = np.array([True, False, False, True, True, True], dtype=bool)
    pts[0] = _hand((0.0, 0.0))
    pts[3] = _hand((0.3, 0.3))
    pts[4] = _hand((0.4, 0.4))
    pts[5] = _hand((0.5, 0.5))

    filled, filled_present = fill = ls.fill_gaps_bidirectional(pts, present, max_gap=4)
    assert filled_present[1] and filled_present[2]
    # linear interpolation at thirds between frame 0 and frame 3
    np.testing.assert_allclose(filled[1, 0, :2], [0.1, 0.1], atol=1e-5)
    np.testing.assert_allclose(filled[2, 0, :2], [0.2, 0.2], atol=1e-5)


def test_fill_gaps_leaves_long_gap_and_edges_missing():
    T, N = 8, 21
    pts = np.zeros((T, N, 3), dtype=np.float32)
    present = np.array([False, True, False, False, False, False, False, True], dtype=bool)
    pts[1] = _hand((0.1, 0.1))
    pts[7] = _hand((0.7, 0.7))
    filled, filled_present = ls.fill_gaps_bidirectional(pts, present, max_gap=4)
    assert not filled_present[0]              # leading edge stays missing
    assert not filled_present[2:7].any()      # 5-frame gap > max_gap stays missing
    assert filled_present[1] and filled_present[7]


def test_fill_gaps_does_not_mutate_inputs():
    T, N = 4, 21
    pts = np.zeros((T, N, 3), dtype=np.float32)
    present = np.array([True, False, True, True], dtype=bool)
    pts[0] = _hand((0.0, 0.0))
    pts[2] = _hand((0.2, 0.2))
    pts[3] = _hand((0.3, 0.3))
    present_before = present.copy()
    ls.fill_gaps_bidirectional(pts, present, max_gap=4)
    np.testing.assert_array_equal(present, present_before)
