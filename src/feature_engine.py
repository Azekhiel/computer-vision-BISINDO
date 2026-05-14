import numpy as np
from scipy.signal import savgol_filter

N_POSE  = 18    
N_HAND  = 63    
N_TOTAL = N_POSE + N_HAND + N_HAND  # 144

SLICE_POSE = slice(0, N_POSE)
SLICE_LH   = slice(N_POSE, N_POSE + N_HAND)
SLICE_RH   = slice(N_POSE + N_HAND, N_TOTAL)
IDX_POSE, IDX_LH, IDX_RH = 0, 1, 2

def extract_keypoints_relative(results):
    pose_coords, lh_coords, rh_coords = [], [], []
    p_ok, lh_ok, rh_ok = False, False, False
    shoulder_width = 1.0 
    mid_shoulder = np.array([0.0, 0.0, 0.0])
    
    pose_lw, pose_rw = np.zeros(3), np.zeros(3)
    
    if results.pose_landmarks:
        p_ok = True
        lm = results.pose_landmarks.landmark
        ls, rs = lm[11], lm[12]
        
        p_ls = np.array([ls.x, ls.y, ls.z])
        p_rs = np.array([rs.x, rs.y, rs.z])
        mid_shoulder = (p_ls + p_rs) / 2.0
        
        dist = np.linalg.norm(p_ls - p_rs)
        if dist > 0.01: shoulder_width = dist

        # Ekstrak Pose Wrist (Landmark 15 & 16) untuk Fallback
        pose_lw = np.array([(lm[15].x - mid_shoulder[0])/shoulder_width, 
                            (lm[15].y - mid_shoulder[1])/shoulder_width, 
                            (lm[15].z - mid_shoulder[2])/shoulder_width])
        pose_rw = np.array([(lm[16].x - mid_shoulder[0])/shoulder_width, 
                            (lm[16].y - mid_shoulder[1])/shoulder_width, 
                            (lm[16].z - mid_shoulder[2])/shoulder_width])

        for idx in range(11, 17):
            norm_x = (lm[idx].x - mid_shoulder[0]) / shoulder_width
            norm_y = (lm[idx].y - mid_shoulder[1]) / shoulder_width
            norm_z = (lm[idx].z - mid_shoulder[2]) / shoulder_width
            pose_coords.extend([norm_x, norm_y, norm_z])
    else:
        pose_coords = list(np.zeros(18))

    if results.left_hand_landmarks:
        lh_ok = True
        wrist_anchor = np.array([results.left_hand_landmarks.landmark[0].x, 
                                 results.left_hand_landmarks.landmark[0].y, 
                                 results.left_hand_landmarks.landmark[0].z])
        for res in results.left_hand_landmarks.landmark:
            lh_coords.extend([(res.x - wrist_anchor[0])/shoulder_width, 
                              (res.y - wrist_anchor[1])/shoulder_width, 
                              (res.z - wrist_anchor[2])/shoulder_width])
    else: lh_coords = list(np.zeros(63))

    if results.right_hand_landmarks:
        rh_ok = True
        wrist_anchor = np.array([results.right_hand_landmarks.landmark[0].x, 
                                 results.right_hand_landmarks.landmark[0].y, 
                                 results.right_hand_landmarks.landmark[0].z])
        for res in results.right_hand_landmarks.landmark:
            rh_coords.extend([(res.x - wrist_anchor[0])/shoulder_width, 
                              (res.y - wrist_anchor[1])/shoulder_width, 
                              (res.z - wrist_anchor[2])/shoulder_width])
    else: rh_coords = list(np.zeros(63))
        
    vector = np.array(pose_coords + lh_coords + rh_coords, dtype=np.float32)
    mask = np.array([p_ok, lh_ok, rh_ok], dtype=bool)
    
    return vector, mask, pose_lw, pose_rw

def calculate_movement_score(prev_vector, curr_vector):
    if prev_vector is None or curr_vector is None: return 0.0
    # Hitung gerakan hanya dari 144 fitur utama (Abaikan 3 Flag)
    return float(np.linalg.norm(curr_vector[:144] - prev_vector[:144]))

class SequenceBuilder:
    NOISE_GAP_MAX = 3    
    INTERACT_GAP_MAX = 20 

    def __init__(self, smooth_window: int = 9, smooth_poly: int = 3):
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self._vectors, self._masks = [], []
        self._pose_lw, self._pose_rw = [], []

    def add_frame(self, vector, mask, pose_lw=None, pose_rw=None):
        self._vectors.append(vector.astype(np.float32))
        self._masks.append(mask.astype(bool))
        self._pose_lw.append(pose_lw)
        self._pose_rw.append(pose_rw)

    def reset(self):
        self._vectors.clear()
        self._masks.clear()
        self._pose_lw.clear()
        self._pose_rw.clear()

    def build(self):
        if not self._vectors: return [], []
        seq = np.stack(self._vectors)       
        msk = np.stack(self._masks)         
        flags = msk.astype(np.float32) # 1 = Asli, 0 = Oklusi/Estimasi

        gap_map = self._classify_all_gaps(msk)
        seq = self._occlusion_aware_fill(seq, msk, gap_map)
        seq = self._smooth(seq)

        # GABUNGKAN 144 FITUR + 3 FLAG = 147 DIMENSI
        augmented_seq = np.concatenate([seq, flags], axis=1)
        scores = self._compute_scores(seq)
        return [augmented_seq[i] for i in range(len(augmented_seq))], scores

    def _classify_all_gaps(self, msk):
        T = msk.shape[0]
        gap_map = {}
        for part_idx in range(3):
            detected = msk[:, part_idx]
            in_gap, gap_start = False, 0
            for t in range(T + 1):
                is_end = (t == T)
                curr_detected = False if is_end else detected[t]
                if not curr_detected and not in_gap:
                    in_gap, gap_start = True, t
                elif curr_detected and in_gap:
                    gap_len = t - gap_start
                    in_gap = False
                    both_missing = (not msk[gap_start:t, IDX_LH].any() and not msk[gap_start:t, IDX_RH].any())
                    
                    if gap_len <= self.NOISE_GAP_MAX: kind = 'noise'
                    elif gap_len <= self.INTERACT_GAP_MAX and both_missing: kind = 'interaction'
                    elif gap_len <= self.INTERACT_GAP_MAX: kind = 'noise'
                    else: kind = 'catastrophic'
                    gap_map[(part_idx, gap_start, t - 1)] = kind
                elif is_end and in_gap:
                    kind = 'noise' if (T - gap_start) <= self.NOISE_GAP_MAX else 'catastrophic'
                    gap_map[(part_idx, gap_start, T - 1)] = kind
        return gap_map

    def _occlusion_aware_fill(self, seq, msk, gap_map):
        result = seq.copy()
        for (part_idx, t_start, t_end), kind in gap_map.items():
            part_sl = [SLICE_POSE, SLICE_LH, SLICE_RH][part_idx]
            gap_frames = list(range(t_start, t_end + 1))

            if kind == 'noise':
                result = self._fill_linear(result, msk, part_idx, part_sl, gap_frames)
            elif kind == 'interaction':
                result = self._fill_interaction(result, msk, part_idx, part_sl, gap_frames)
            else:
                result = self._fill_nearest(result, part_sl, gap_frames)
        return result

    def _fill_linear(self, result, msk, part_idx, part_sl, gap_frames):
        valid = np.where(msk[:, part_idx])[0]
        if not valid.size: return result
        for t in gap_frames:
            before = valid[valid < t]
            after  = valid[valid > t]
            if not before.size: result[t, part_sl] = result[after[0], part_sl]
            elif not after.size: result[t, part_sl] = result[before[-1], part_sl]
            else:
                t0, t1 = before[-1], after[0]
                alpha = (t - t0) / (t1 - t0)
                result[t, part_sl] = (1 - alpha) * result[t0, part_sl] + alpha * result[t1, part_sl]
        return result

    def _fill_interaction(self, result, msk, part_idx, part_sl, gap_frames):
        valid = np.where(msk[:, part_idx])[0]
        last_valid_before = valid[valid < gap_frames[0]]
        first_valid_after = valid[valid > gap_frames[-1]]
        t_before = last_valid_before[-1] if last_valid_before.size else None
        t_after  = first_valid_after[0] if first_valid_after.size else None

        for t in gap_frames:
            # FREEZE JARI
            if t_before is not None: result[t, part_sl] = result[t_before, part_sl]
            elif t_after is not None: result[t, part_sl] = result[t_after, part_sl]

            # POSE-GUIDED WRIST RECOVERY
            pose_wrist_list = self._pose_lw if part_idx == IDX_LH else self._pose_rw
            pw = pose_wrist_list[t]
            if pw is not None:
                result[t, part_sl.start : part_sl.start + 3] = pw
        return result

    def _fill_nearest(self, result, part_sl, gap_frames):
        T = result.shape[0]
        for t in gap_frames:
            all_valid = [i for i in range(T) if i not in gap_frames]
            if not all_valid: continue
            nearest = min(all_valid, key=lambda i: abs(i - t))
            result[t, part_sl] = result[nearest, part_sl]
        return result

    def _smooth(self, seq):
        T = seq.shape[0]
        if T < 3: return seq
        win = max(min(self.smooth_window, T) - (1 if min(self.smooth_window, T) % 2 == 0 else 0), 3)
        try: return savgol_filter(seq, window_length=win, polyorder=min(self.smooth_poly, win - 1), axis=0).astype(np.float32)
        except Exception: return seq

    @staticmethod
    def _compute_scores(seq):
        scores = [0.0]
        for i in range(1, len(seq)):
            scores.append(float(np.linalg.norm(seq[i] - seq[i - 1])))
        return scores