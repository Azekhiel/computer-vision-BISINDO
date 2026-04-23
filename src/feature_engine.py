import numpy as np

def extract_keypoints_relative(results):
    """
    Mengekstrak tepat 144-dimensi fitur spasial dari MediaPipe Holistic.
    Koordinat dibuat relatif terhadap hidung untuk menghilangkan bias posisi kamera.
    """
    # Set default anchor di tengah jika pose tubuh tidak terdeteksi
    anchor_x, anchor_y, anchor_z = 0.5, 0.5, 0.0 
    
    if results.pose_landmarks:
        # Gunakan hidung (landmark 0) sebagai titik pusat (0,0,0)
        anchor_x = results.pose_landmarks.landmark[0].x
        anchor_y = results.pose_landmarks.landmark[0].y
        anchor_z = results.pose_landmarks.landmark[0].z

    def get_relative_coords(landmarks, num_points, specific_indices=None):
        if not landmarks:
            if specific_indices:
                return np.zeros(len(specific_indices) * 3)
            return np.zeros(num_points * 3)
        
        rel_coords = []
        if specific_indices:
            # Hanya ekstrak titik tertentu (untuk pose tubuh)
            for idx in specific_indices:
                res = landmarks.landmark[idx]
                rel_coords.extend([
                    res.x - anchor_x, 
                    res.y - anchor_y, 
                    res.z - anchor_z
                ])
        else:
            # Ekstrak semua titik (untuk jari tangan)
            for res in landmarks.landmark:
                rel_coords.extend([
                    res.x - anchor_x, 
                    res.y - anchor_y, 
                    res.z - anchor_z
                ])
        return rel_coords

    # 1. Ekstrak 6 titik Pose (Bahu Kiri-Kanan, Siku Kiri-Kanan, Pergelangan Kiri-Kanan)
    # 6 titik * 3 (X, Y, Z) = 18 Dimensi
    pose_indices = [11, 12, 13, 14, 15, 16]
    pose = get_relative_coords(results.pose_landmarks, 33, specific_indices=pose_indices)
    
    # 2. Ekstrak 21 titik Tangan Kiri
    # 21 titik * 3 (X, Y, Z) = 63 Dimensi
    lh = get_relative_coords(results.left_hand_landmarks, 21)
    
    # 3. Ekstrak 21 titik Tangan Kanan
    # 21 titik * 3 (X, Y, Z) = 63 Dimensi
    rh = get_relative_coords(results.right_hand_landmarks, 21)
    
    # Total Vektor: 18 + 63 + 63 = 144 Dimensi
    return np.concatenate([pose, lh, rh])

def calculate_movement_score(prev_vector, curr_vector):
    """
    Menghitung jarak L2 Norm dari vektor 144-D antar 2 frame berurutan.
    Fungsi ini menjadi inti dari sistem Visual Activity Detection (VAD) 
    untuk mendeteksi kapan isyarat dimulai dan kapan tangan kembali diam.
    """
    if prev_vector is None or curr_vector is None:
        return 0.0
        
    # Selisih Euclidean antar vektor 144-dimensi
    diff = prev_vector - curr_vector
    return np.linalg.norm(diff)