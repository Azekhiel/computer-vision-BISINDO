import numpy as np

def extract_keypoints_relative(results):
    """
    Mengekstrak tepat 144-dimensi fitur spasial dari MediaPipe Holistic.
    DIPERBAIKI:
    - Pose dihitung relatif terhadap Hidung.
    - Tangan Kiri dihitung relatif terhadap Pergelangan Tangan Kiri.
    - Tangan Kanan dihitung relatif terhadap Pergelangan Tangan Kanan.
    Ini mengisolasi 'Bentuk Tangan' dari 'Posisi Tangan'.
    """
    
    # 1. POSE (Relatif ke Hidung)
    pose_coords = []
    if results.pose_landmarks:
        # Hidung adalah landmark 0
        nose_x = results.pose_landmarks.landmark[0].x
        nose_y = results.pose_landmarks.landmark[0].y
        nose_z = results.pose_landmarks.landmark[0].z
        
        pose_indices = [11, 12, 13, 14, 15, 16] # Bahu, Siku, Pergelangan
        for idx in pose_indices:
            res = results.pose_landmarks.landmark[idx]
            pose_coords.extend([res.x - nose_x, res.y - nose_y, res.z - nose_z])
    else:
        pose_coords = list(np.zeros(18)) # 6 titik x 3 dimensi

    # 2. TANGAN KIRI (Relatif ke Pergelangan Tangan Kiri)
    lh_coords = []
    if results.left_hand_landmarks:
        # Wrist adalah landmark 0 dari tangan
        wrist_x = results.left_hand_landmarks.landmark[0].x
        wrist_y = results.left_hand_landmarks.landmark[0].y
        wrist_z = results.left_hand_landmarks.landmark[0].z
        
        for res in results.left_hand_landmarks.landmark:
            lh_coords.extend([res.x - wrist_x, res.y - wrist_y, res.z - wrist_z])
    else:
        lh_coords = list(np.zeros(63)) # 21 titik x 3 dimensi

    # 3. TANGAN KANAN (Relatif ke Pergelangan Tangan Kanan)
    rh_coords = []
    if results.right_hand_landmarks:
        # Wrist adalah landmark 0 dari tangan
        wrist_x = results.right_hand_landmarks.landmark[0].x
        wrist_y = results.right_hand_landmarks.landmark[0].y
        wrist_z = results.right_hand_landmarks.landmark[0].z
        
        for res in results.right_hand_landmarks.landmark:
            rh_coords.extend([res.x - wrist_x, res.y - wrist_y, res.z - wrist_z])
    else:
        rh_coords = list(np.zeros(63)) # 21 titik x 3 dimensi
        
    # Total Vektor: 18 + 63 + 63 = 144 Dimensi
    return np.array(pose_coords + lh_coords + rh_coords)

def calculate_movement_score(prev_vector, curr_vector):
    """
    Menghitung jarak L2 Norm dari vektor 144-D antar 2 frame berurutan.
    Fungsi ini menjadi inti dari sistem Visual Activity Detection (VAD).
    """
    if prev_vector is None or curr_vector is None:
        return 0.0
        
    # Selisih Euclidean antar vektor 144-dimensi
    diff = prev_vector - curr_vector
    return np.linalg.norm(diff)