"""
feature_engine.py  —  Spatio-Temporal Gap-Free Keypoint Extraction
===========================================================
Gabungan: 
1. Spatial: Anchor Mid-Shoulder & Skala Lebar Bahu (Scale & Translation Invariant)
2. Temporal: Interpolasi Linear & Savitzky-Golay Smoothing
"""

import numpy as np
from scipy.signal import savgol_filter

# ──────────────────────────────────────────────
# Konstanta dimensi
# ──────────────────────────────────────────────
N_POSE  = 18    # 6 joints (bahu, siku, pergelangan) × 3 (X, Y, Z)
N_HAND  = 63    # 21 landmark × 3 (X, Y, Z)
N_TOTAL = N_POSE + N_HAND + N_HAND  # 144

SLICE_POSE = slice(0,        N_POSE)
SLICE_LH   = slice(N_POSE,   N_POSE + N_HAND)
SLICE_RH   = slice(N_POSE + N_HAND, N_TOTAL)

IDX_POSE, IDX_LH, IDX_RH = 0, 1, 2

# ──────────────────────────────────────────────
# Fungsi Ekstraksi Primitif (Spatio-Temporal Invariant)
# ──────────────────────────────────────────────

def extract_keypoints_relative(results):
    """
    Ekstrak vektor fitur 144-D dengan Skala Lebar Bahu & Masking.
    Returns:
      vector : np.array(144,)
      mask   : np.array([pose_ok, lh_ok, rh_ok], dtype=bool)
    """
    pose_coords = []
    lh_coords = []
    rh_coords = []
    
    p_ok, lh_ok, rh_ok = False, False, False

    # 1. ANALISIS REFERENSI SKALA (LEBAR BAHU)
    shoulder_width = 1.0 
    mid_shoulder = np.array([0.0, 0.0, 0.0])
    
    if results.pose_landmarks:
        p_ok = True
        ls = results.pose_landmarks.landmark[11] 
        rs = results.pose_landmarks.landmark[12] 
        
        p_ls = np.array([ls.x, ls.y, ls.z])
        p_rs = np.array([rs.x, rs.y, rs.z])
        
        mid_shoulder = (p_ls + p_rs) / 2.0
        dist = np.linalg.norm(p_ls - p_rs)
        if dist > 0.01: 
            shoulder_width = dist

        # Ekstrak POSE (Bahu, Siku, Pergelangan)
        for idx in range(11, 17):
            res = results.pose_landmarks.landmark[idx]
            norm_x = (res.x - mid_shoulder[0]) / shoulder_width
            norm_y = (res.y - mid_shoulder[1]) / shoulder_width
            norm_z = (res.z - mid_shoulder[2]) / shoulder_width
            pose_coords.extend([norm_x, norm_y, norm_z])
    else:
        pose_coords = list(np.zeros(18))

    # 2. EKSTRAK TANGAN KIRI
    if results.left_hand_landmarks:
        lh_ok = True
        wrist = results.left_hand_landmarks.landmark[0]
        wrist_anchor = np.array([wrist.x, wrist.y, wrist.z])
        
        for res in results.left_hand_landmarks.landmark:
            norm_x = (res.x - wrist_anchor[0]) / shoulder_width
            norm_y = (res.y - wrist_anchor[1]) / shoulder_width
            norm_z = (res.z - wrist_anchor[2]) / shoulder_width
            lh_coords.extend([norm_x, norm_y, norm_z])
    else:
        lh_coords = list(np.zeros(63))

    # 3. EKSTRAK TANGAN KANAN
    if results.right_hand_landmarks:
        rh_ok = True
        wrist = results.right_hand_landmarks.landmark[0]
        wrist_anchor = np.array([wrist.x, wrist.y, wrist.z])
        
        for res in results.right_hand_landmarks.landmark:
            norm_x = (res.x - wrist_anchor[0]) / shoulder_width
            norm_y = (res.y - wrist_anchor[1]) / shoulder_width
            norm_z = (res.z - wrist_anchor[2]) / shoulder_width
            rh_coords.extend([norm_x, norm_y, norm_z])
    else:
        rh_coords = list(np.zeros(63))
        
    vector = np.array(pose_coords + lh_coords + rh_coords, dtype=np.float32)
    mask = np.array([p_ok, lh_ok, rh_ok], dtype=bool)
    
    return vector, mask

def calculate_movement_score(prev_vector, curr_vector):
    if prev_vector is None or curr_vector is None: return 0.0
    return float(np.linalg.norm(curr_vector - prev_vector))

# ──────────────────────────────────────────────
# SequenceBuilder  —  Penambal Celah & Penghalus Jitter
# ──────────────────────────────────────────────

class SequenceBuilder:
    def __init__(self, max_interp_gap: int = 6, smooth_window: int = 9, smooth_poly: int = 3):
        self.max_interp_gap = max_interp_gap
        self.smooth_window  = smooth_window
        self.smooth_poly    = smooth_poly
        self._vectors: list[np.ndarray] = []
        self._masks:   list[np.ndarray] = []

    def add_frame(self, vector: np.ndarray, mask: np.ndarray):
        self._vectors.append(vector.astype(np.float32))
        self._masks.append(mask.astype(bool))

    def reset(self):
        self._vectors.clear()
        self._masks.clear()

    def build(self) -> tuple[list[np.ndarray], list[float]]:
        if not self._vectors: return [], []

        seq = np.stack(self._vectors)   
        msk = np.stack(self._masks)     

        seq = self._fill_gaps(seq, msk)
        seq = self._smooth(seq)

        scores = self._compute_scores(seq)
        return [seq[i] for i in range(len(seq))], scores

    def _fill_gaps(self, seq: np.ndarray, msk: np.ndarray) -> np.ndarray:
        T, _ = seq.shape
        result = seq.copy()

        part_info = [(IDX_POSE, SLICE_POSE), (IDX_LH, SLICE_LH), (IDX_RH, SLICE_RH)]

        for part_idx, part_sl in part_info:
            detected = msk[:, part_idx]   

            if detected.all(): continue   
            if not detected.any(): continue   

            valid_t = np.where(detected)[0]

            for t in np.where(~detected)[0]:
                before = valid_t[valid_t < t]
                after  = valid_t[valid_t > t]

                if before.size == 0:
                    result[t, part_sl] = result[after[0], part_sl]
                elif after.size == 0:
                    result[t, part_sl] = result[before[-1], part_sl]
                else:
                    t0, t1 = before[-1], after[0]
                    gap = t1 - t0   

                    if gap <= self.max_interp_gap + 1:
                        alpha = (t - t0) / gap
                        result[t, part_sl] = ((1.0 - alpha) * result[t0, part_sl] + alpha * result[t1, part_sl])
                    else:
                        if (t - t0) <= (t1 - t): result[t, part_sl] = result[t0, part_sl]
                        else: result[t, part_sl] = result[t1, part_sl]
        return result

    def _smooth(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T < 3: return seq

        win = min(self.smooth_window, T)
        if win % 2 == 0: win -= 1
        win = max(win, 3)
        poly = min(self.smooth_poly, win - 1)

        try:
            return savgol_filter(seq, window_length=win, polyorder=poly, axis=0).astype(np.float32)
        except Exception:
            return seq

    @staticmethod
    def _compute_scores(seq: np.ndarray) -> list[float]:
        scores = [0.0]
        for i in range(1, len(seq)):
            scores.append(float(np.linalg.norm(seq[i] - seq[i - 1])))
        return scores