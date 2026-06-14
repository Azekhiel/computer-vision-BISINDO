"""
feature_extractor.py
====================
Core MediaPipe feature extraction module.
Platform: Jetson Orin Nano (JetPack 6.x, CUDA 12.6, Ubuntu 22.04)
"""

import numpy as np
import mediapipe as mp
import cv2

# ── Pose keypoints ──────────────────────────────────────────────────────────
POSE_KP_IDX = [11, 12, 13, 14, 23, 24]   # L/R shoulder, elbow, hip
POSE_KP_NAMES = ["L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_hip", "R_hip"]

HAND_ANGLE_TRIPLETS = [
    (5,  0,  1), (0,  1,  2),  (1,  2,  3),  (2,  3,  4),       # thumb
    (0,  5,  6),  (5,  6,  7),  (6,  7,  8),                   # index
    (0,  9, 10),  (9, 10, 11), (10, 11, 12),                   # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),                   # ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),                   # pinky
]

mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic       = mp.solutions.holistic

def make_holistic(model_complexity: int = 0, smooth: bool = True,
                  det_conf: float = 0.5, track_conf: float = 0.5) -> mp_holistic.Holistic:
    return mp_holistic.Holistic(
        model_complexity=model_complexity,
        smooth_landmarks=smooth,
        enable_segmentation=False,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )

def _lm_xyz(lm_list, idx: int) -> np.ndarray:
    lm = lm_list[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def _angle_at_b(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    ba, bc = pa - pb, pc - pb
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-7: return 0.0
    cos_val = np.dot(ba, bc) / denom
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))

def _hand_angles(lm_list) -> np.ndarray:
    angles = np.zeros(16, dtype=np.float32)
    for j, (a, b, c) in enumerate(HAND_ANGLE_TRIPLETS):
        angles[j] = _angle_at_b(_lm_xyz(lm_list, a), _lm_xyz(lm_list, b), _lm_xyz(lm_list, c))
    return angles

def extract_features(results) -> np.ndarray:
    feat = np.zeros(179, dtype=np.float32)
    pose_ok = lh_ok = rh_ok = 0

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        for i, idx in enumerate(POSE_KP_IDX):
            feat[i*3:i*3+3] = [lm[idx].x, lm[idx].y, lm[idx].z]
        pose_ok = 1

    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark
        for i in range(21):
            feat[18+i*3:18+i*3+3] = [lm[i].x, lm[i].y, lm[i].z]
        feat[144:160] = _hand_angles(lm)
        lh_ok = 1

    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark
        for i in range(21):
            feat[81+i*3:81+i*3+3] = [lm[i].x, lm[i].y, lm[i].z]
        feat[160:176] = _hand_angles(lm)
        rh_ok = 1

    feat[176:179] = [pose_ok, lh_ok, rh_ok]
    return feat

def draw_landmarks(frame: np.ndarray, results) -> np.ndarray:
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
    return frame

def draw_hud(frame: np.ndarray, feat: np.ndarray, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    pose_ok, lh_ok, rh_ok = int(feat[176]), int(feat[177]), int(feat[178])
    flag_color = lambda v: (0, 220, 80) if v else (0, 60, 220)

    overlay = frame.copy()
    cv2.rectangle(overlay, (4, 4), (210, 98), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 22), font, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Pose : {'OK' if pose_ok else '--'}", (10, 46), font, 0.5, flag_color(pose_ok), 1)
    cv2.putText(frame, f"L_Hand: {'OK' if lh_ok else '--'}", (10, 66), font, 0.5, flag_color(lh_ok), 1)
    cv2.putText(frame, f"R_Hand: {'OK' if rh_ok else '--'}", (10, 86), font, 0.5, flag_color(rh_ok), 1)

    norm = float(np.linalg.norm(feat[:144]))
    bar_w = min(int(norm * 30), w - 20)
    cv2.rectangle(frame, (10, h - 16), (10 + bar_w, h - 8), (60, 180, 255), -1)
    cv2.putText(frame, f"feat||{norm:.2f}", (10, h - 20), font, 0.38, (160, 160, 160), 1)
    return frame