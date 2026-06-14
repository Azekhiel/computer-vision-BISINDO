"""Face-reference feature blocks for smart180-based BISINDO schemas.

The face here is a *position reference for the hands*, not a motion source.
Every face/hand point is normalised in the same shoulder frame used by smart180
(``shoulder_anchor_scale``), so a face-ref vector is just ``smart180[0:180]``
followed by one extra block. ``build_face_schema_vector`` is the single entry
point shared by the offline converter and the live extractor, which gives
offline/live parity by construction.

When the face (or a hand) is missing the corresponding sub-block is zeroed and a
``*_present`` flag drops to 0 — never ``(0 - anchor)/scale`` leaking a fake
position. Velocity terms are per-second (``delta / dt``) so offline 10 fps and a
variable live frame-rate stay comparable; the first valid frame (and the frame
after a gap) reports zero velocity.
"""

from __future__ import annotations

import numpy as np

from smart_extract.live_bisindo_mp_real_shoulder_v6 import (
    palm_center,
    shoulder_anchor_scale,
)


# --- FaceMesh landmark indices (all < 468, valid for refine on/off) -----------
NOSE_TIP = 1
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
EYE_L_OUTER = 33
EYE_L_INNER = 133
EYE_R_INNER = 362
EYE_R_OUTER = 263

# 9 face points used by the mouthdyn schema (relative block).
MOUTHDYN_POINTS = (
    NOSE_TIP,
    MOUTH_TOP,
    MOUTH_BOTTOM,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    EYE_L_OUTER,
    EYE_L_INNER,
    EYE_R_INNER,
    EYE_R_OUTER,
)
MOUTHDYN_POINT_NAMES = (
    "nose",
    "mouth_top",
    "mouth_bottom",
    "mouth_left",
    "mouth_right",
    "eye_l_outer",
    "eye_l_inner",
    "eye_r_inner",
    "eye_r_outer",
)

SMART_BASE_DIM = 180

# Selected hand points used by smart180 global slices (for velocity column names).
_SMART180_SELECTED_POINTS = (
    "wrist",
    "palm_center",
    "thumb_tip",
    "index_mcp",
    "index_tip",
    "middle_mcp",
    "middle_tip",
    "ring_mcp",
    "ring_tip",
    "pinky_mcp",
    "pinky_tip",
)
# smart180 [6:39] = left global, [39:72] = right global (11 points x 3).
_SLICE_LEFT_GLOBAL = slice(6, 39)
_SLICE_RIGHT_GLOBAL = slice(39, 72)

# Extra block size appended after smart180[0:180] for each schema.
FACE_REF_BLOCK_DIMS = {
    "smart180_mouthdyn214": 34,
    "smart180_mouthstat206": 26,
    "smart180_handface220": 40,
    "smart180_handface_vel286": 106,
}
FACE_REF_SCHEMA_DIMS = {name: SMART_BASE_DIM + block for name, block in FACE_REF_BLOCK_DIMS.items()}
FACE_REF_SCHEMA_NAMES = tuple(FACE_REF_BLOCK_DIMS.keys())
# Schemas that keep temporal state across frames (must run single-worker).
FACE_REF_STATEFUL_SCHEMAS = ("smart180_mouthdyn214", "smart180_handface_vel286")


def _shoulder_ref(shoulders) -> tuple[np.ndarray, float]:
    anchor, scale, _, _ = shoulder_anchor_scale(np.asarray(shoulders, dtype=np.float32))
    return anchor.astype(np.float32), float(scale)


def _face_points(face_xyz: np.ndarray, indices) -> np.ndarray:
    return np.stack([np.asarray(face_xyz[i], dtype=np.float32) for i in indices]).astype(np.float32)


def _rel(point: np.ndarray, anchor: np.ndarray, scale: float) -> np.ndarray:
    return ((np.asarray(point, dtype=np.float32) - anchor) / scale).astype(np.float32)


def _build_mouthdyn(face_xyz, anchor, scale, face_present, state, dt) -> np.ndarray:
    """27 face-rel + (openness,width,aspect,eye_dist) + (d_openness,d_mouth_y) + present."""
    block = np.zeros(34, dtype=np.float32)
    if not face_present:
        if state is not None:
            state.pop("mouthdyn_prev_openness", None)
            state.pop("mouthdyn_prev_mouth_y", None)
        return block

    rel = (_face_points(face_xyz, MOUTHDYN_POINTS) - anchor) / scale  # (9, 3)
    block[0:27] = rel.reshape(-1)
    top, bottom, m_left, m_right = rel[1], rel[2], rel[3], rel[4]
    eye_left = (rel[5] + rel[6]) * 0.5
    eye_right = (rel[7] + rel[8]) * 0.5
    openness = float(np.linalg.norm(top - bottom))
    width = float(np.linalg.norm(m_left - m_right))
    aspect = openness / max(width, 1e-4)
    eye_dist = float(np.linalg.norm(eye_left - eye_right))
    mouth_center_y = float((top[1] + bottom[1] + m_left[1] + m_right[1]) / 4.0)
    block[27] = openness
    block[28] = width
    block[29] = aspect
    block[30] = eye_dist

    dt_s = max(float(dt), 1e-3)
    prev_open = state.get("mouthdyn_prev_openness") if state is not None else None
    prev_y = state.get("mouthdyn_prev_mouth_y") if state is not None else None
    if prev_open is not None:
        block[31] = (openness - float(prev_open)) / dt_s
    if prev_y is not None:
        block[32] = (mouth_center_y - float(prev_y)) / dt_s
    block[33] = 1.0
    if state is not None:
        state["mouthdyn_prev_openness"] = openness
        state["mouthdyn_prev_mouth_y"] = mouth_center_y
    return block


def _build_mouthstat(face_xyz, anchor, scale, face_present) -> np.ndarray:
    """nose(3) + mouth line (left,right,center = 9) + 4 eye corners(12) + eye_dist + present."""
    block = np.zeros(26, dtype=np.float32)
    if not face_present:
        return block

    nose = _rel(face_xyz[NOSE_TIP], anchor, scale)
    m_left = _rel(face_xyz[MOUTH_LEFT], anchor, scale)
    m_right = _rel(face_xyz[MOUTH_RIGHT], anchor, scale)
    mouth_center_raw = (
        np.asarray(face_xyz[MOUTH_TOP], dtype=np.float32)
        + np.asarray(face_xyz[MOUTH_BOTTOM], dtype=np.float32)
        + np.asarray(face_xyz[MOUTH_LEFT], dtype=np.float32)
        + np.asarray(face_xyz[MOUTH_RIGHT], dtype=np.float32)
    ) / 4.0
    mouth_center = _rel(mouth_center_raw, anchor, scale)
    e_lo = _rel(face_xyz[EYE_L_OUTER], anchor, scale)
    e_li = _rel(face_xyz[EYE_L_INNER], anchor, scale)
    e_ri = _rel(face_xyz[EYE_R_INNER], anchor, scale)
    e_ro = _rel(face_xyz[EYE_R_OUTER], anchor, scale)
    eye_left = (e_lo + e_li) * 0.5
    eye_right = (e_ri + e_ro) * 0.5

    block[0:3] = nose
    block[3:6] = m_left
    block[6:9] = m_right
    block[9:12] = mouth_center
    block[12:15] = e_lo
    block[15:18] = e_li
    block[18:21] = e_ri
    block[21:24] = e_ro
    block[24] = float(np.linalg.norm(eye_left - eye_right))
    block[25] = 1.0
    return block


def _build_handface(face_xyz, anchor, scale, left_xyz, right_xyz, face_present) -> np.ndarray:
    """anchor rel (9) + per-hand wrist/palm->face vectors (24) + 4 dist + 3 flags."""
    block = np.zeros(40, dtype=np.float32)
    has_face = bool(face_present)
    left_present = left_xyz is not None
    right_present = right_xyz is not None

    if has_face:
        nose_raw = np.asarray(face_xyz[NOSE_TIP], dtype=np.float32)
        mouth_raw = (
            np.asarray(face_xyz[MOUTH_TOP], dtype=np.float32)
            + np.asarray(face_xyz[MOUTH_BOTTOM], dtype=np.float32)
            + np.asarray(face_xyz[MOUTH_LEFT], dtype=np.float32)
            + np.asarray(face_xyz[MOUTH_RIGHT], dtype=np.float32)
        ) / 4.0
        eye_raw = (
            np.asarray(face_xyz[EYE_L_OUTER], dtype=np.float32)
            + np.asarray(face_xyz[EYE_L_INNER], dtype=np.float32)
            + np.asarray(face_xyz[EYE_R_INNER], dtype=np.float32)
            + np.asarray(face_xyz[EYE_R_OUTER], dtype=np.float32)
        ) / 4.0
        block[0:3] = _rel(nose_raw, anchor, scale)
        block[3:6] = _rel(mouth_raw, anchor, scale)
        block[6:9] = _rel(eye_raw, anchor, scale)

        for hand_idx, hand in enumerate((left_xyz, right_xyz)):
            if hand is None:
                continue
            base = 9 + hand_idx * 12
            wrist = np.asarray(hand[0], dtype=np.float32)
            palm = palm_center(np.asarray(hand, dtype=np.float32))
            block[base + 0:base + 3] = (nose_raw - wrist) / scale
            block[base + 3:base + 6] = (mouth_raw - wrist) / scale
            block[base + 6:base + 9] = (eye_raw - wrist) / scale
            block[base + 9:base + 12] = (nose_raw - palm) / scale

        if left_present:
            block[33] = float(np.linalg.norm((nose_raw - np.asarray(left_xyz[0], dtype=np.float32)) / scale))
            block[35] = float(np.linalg.norm((mouth_raw - palm_center(np.asarray(left_xyz, dtype=np.float32))) / scale))
        if right_present:
            block[34] = float(np.linalg.norm((nose_raw - np.asarray(right_xyz[0], dtype=np.float32)) / scale))
            block[36] = float(np.linalg.norm((mouth_raw - palm_center(np.asarray(right_xyz, dtype=np.float32))) / scale))

    block[37] = float(has_face)
    block[38] = float(left_present)
    block[39] = float(right_present)
    return block


def _build_handface_vel(smart_vec180, handface_block, state, dt) -> np.ndarray:
    """handface(40) + per-second velocity of smart180 left/right global slices (33+33)."""
    block = np.zeros(106, dtype=np.float32)
    block[0:40] = handface_block
    cur_left = np.asarray(smart_vec180[_SLICE_LEFT_GLOBAL], dtype=np.float32)
    cur_right = np.asarray(smart_vec180[_SLICE_RIGHT_GLOBAL], dtype=np.float32)
    dt_s = max(float(dt), 1e-3)
    prev_left = state.get("handfacevel_prev_left") if state is not None else None
    prev_right = state.get("handfacevel_prev_right") if state is not None else None
    if prev_left is not None:
        block[40:73] = (cur_left - np.asarray(prev_left, dtype=np.float32)) / dt_s
    if prev_right is not None:
        block[73:106] = (cur_right - np.asarray(prev_right, dtype=np.float32)) / dt_s
    if state is not None:
        state["handfacevel_prev_left"] = cur_left.copy()
        state["handfacevel_prev_right"] = cur_right.copy()
    return block


def _schema_name(schema) -> str:
    if isinstance(schema, str):
        return schema
    return getattr(schema, "name", str(schema))


def is_face_ref_schema(schema) -> bool:
    return _schema_name(schema) in FACE_REF_BLOCK_DIMS


def build_face_schema_vector(
    schema_name,
    smart_vec180: np.ndarray,
    face_xyz: np.ndarray | None,
    shoulders: np.ndarray,
    left_xyz: np.ndarray | None = None,
    right_xyz: np.ndarray | None = None,
    face_present: bool = True,
    state: dict | None = None,
    dt: float = 0.1,
) -> np.ndarray:
    """Return ``smart180[0:180]`` concatenated with the face-ref block for ``schema_name``.

    Shared by the offline NPZ converter and the live extractor. ``state`` (a mutable
    dict the caller keeps per video / per live session) carries velocity history;
    pass ``None`` for stateless one-shot use (velocity terms become 0).
    """
    name = _schema_name(schema_name)
    if name not in FACE_REF_BLOCK_DIMS:
        raise ValueError(f"Bukan schema face-reference: {name}")

    smart = np.asarray(smart_vec180, dtype=np.float32).reshape(-1)
    if smart.shape[0] != SMART_BASE_DIM:
        raise ValueError(f"smart_vec180 harus {SMART_BASE_DIM}-D, got {smart.shape[0]}")

    anchor, scale = _shoulder_ref(shoulders)
    face_ok = bool(
        face_present
        and face_xyz is not None
        and np.isfinite(np.asarray(face_xyz, dtype=np.float32)).all()
    )

    if name == "smart180_mouthdyn214":
        block = _build_mouthdyn(face_xyz, anchor, scale, face_ok, state, dt)
    elif name == "smart180_mouthstat206":
        block = _build_mouthstat(face_xyz, anchor, scale, face_ok)
    elif name == "smart180_handface220":
        block = _build_handface(face_xyz, anchor, scale, left_xyz, right_xyz, face_ok)
    elif name == "smart180_handface_vel286":
        handface = _build_handface(face_xyz, anchor, scale, left_xyz, right_xyz, face_ok)
        block = _build_handface_vel(smart, handface, state, dt)
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Bukan schema face-reference: {name}")

    vector = np.concatenate([smart, block]).astype(np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def face_block_column_names(schema_name) -> list[str]:
    """Column names for the extra block only (smart180 names handled by feature_schemas)."""
    name = _schema_name(schema_name)
    axes = ("x", "y", "z")
    if name == "smart180_mouthdyn214":
        names = [f"face_{pt}_{ax}" for pt in MOUTHDYN_POINT_NAMES for ax in axes]
        names += ["mouth_openness", "mouth_width", "mouth_aspect", "eye_dist"]
        names += ["d_mouth_openness", "d_mouth_y", "face_present"]
        return names
    if name == "smart180_mouthstat206":
        names: list[str] = []
        names += [f"face_nose_{ax}" for ax in axes]
        names += [f"face_mouth_left_{ax}" for ax in axes]
        names += [f"face_mouth_right_{ax}" for ax in axes]
        names += [f"face_mouth_center_{ax}" for ax in axes]
        for corner in ("eye_l_outer", "eye_l_inner", "eye_r_inner", "eye_r_outer"):
            names += [f"face_{corner}_{ax}" for ax in axes]
        names += ["eye_dist", "face_present"]
        return names
    if name in {"smart180_handface220", "smart180_handface_vel286"}:
        names = []
        for anchor_name in ("nose", "mouth_center", "eye_center"):
            names += [f"face_anchor_{anchor_name}_{ax}" for ax in axes]
        for hand in ("left", "right"):
            for target in ("wrist_to_nose", "wrist_to_mouth", "wrist_to_eye", "palm_to_nose"):
                names += [f"{hand}_{target}_{ax}" for ax in axes]
        names += ["dist_left_wrist_nose", "dist_right_wrist_nose", "dist_left_palm_mouth", "dist_right_palm_mouth"]
        names += ["face_present", "left_present", "right_present"]
        if name == "smart180_handface_vel286":
            for hand in ("left", "right"):
                for point in _SMART180_SELECTED_POINTS:
                    names += [f"vel_{hand}_{point}_{ax}" for ax in axes]
        return names
    raise ValueError(f"Bukan schema face-reference: {name}")
