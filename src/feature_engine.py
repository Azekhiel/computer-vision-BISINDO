import numpy as np

def normalize_scale(coords):
    """
    Membagi seluruh koordinat dengan nilai absolut terbesarnya 
    agar rentangnya selalu proporsional (kebal terhadap jarak kamera).
    """
    coords_array = np.array(coords)
    max_val = np.max(np.abs(coords_array))
    if max_val > 0:
        return (coords_array / max_val).tolist()
    return coords_array.tolist()

def extract_keypoints_relative(results):
    """
    Mengekstrak tepat 144-dimensi fitur spasial dari MediaPipe Holistic.
    Logika yang diperbaiki:
    - Pose dihitung relatif terhadap Hidung, lalu dinormalisasi.
    - Tangan Kiri dihitung relatif terhadap Pergelangan Tangan Kiri, lalu dinormalisasi.
    - Tangan Kanan dihitung relatif terhadap Pergelangan Tangan Kanan, lalu dinormalisasi.
    """
    
    # ==========================================
    # 1. POSE BADAN (Anchor: Hidung)
    # ==========================================
    pose_coords = []
    if results.pose_landmarks:
        # Landmark 0 adalah Hidung
        nose_x = results.pose_landmarks.landmark[0].x
        nose_y = results.pose_landmarks.landmark[0].y
        nose_z = results.pose_landmarks.landmark[0].z
        
        # Ekstrak Bahu, Siku, dan Pergelangan Tangan (Indeks 11 sampai 16)
        pose_indices = [11, 12, 13, 14, 15, 16]
        for idx in pose_indices:
            res = results.pose_landmarks.landmark[idx]
            pose_coords.extend([res.x - nose_x, res.y - nose_y, res.z - nose_z])
            
        pose_coords = normalize_scale(pose_coords)
    else:
        pose_coords = list(np.zeros(18)) # 6 titik x 3 (X,Y,Z)

    # ==========================================
    # 2. TANGAN KIRI (Anchor: Pergelangan Tangan Kiri)
    # ==========================================
    lh_coords = []
    if results.left_hand_landmarks:
        # Landmark 0 adalah Wrist (Pergelangan Tangan)
        wrist_x = results.left_hand_landmarks.landmark[0].x
        wrist_y = results.left_hand_landmarks.landmark[0].y
        wrist_z = results.left_hand_landmarks.landmark[0].z
        
        # Ekstrak semua 21 titik jari
        for res in results.left_hand_landmarks.landmark:
            lh_coords.extend([res.x - wrist_x, res.y - wrist_y, res.z - wrist_z])
            
        lh_coords = normalize_scale(lh_coords)
    else:
        lh_coords = list(np.zeros(63)) # 21 titik x 3 (X,Y,Z)

    # ==========================================
    # 3. TANGAN KANAN (Anchor: Pergelangan Tangan Kanan)
    # ==========================================
    rh_coords = []
    if results.right_hand_landmarks:
        # Landmark 0 adalah Wrist (Pergelangan Tangan)
        wrist_x = results.right_hand_landmarks.landmark[0].x
        wrist_y = results.right_hand_landmarks.landmark[0].y
        wrist_z = results.right_hand_landmarks.landmark[0].z
        
        # Ekstrak semua 21 titik jari
        for res in results.right_hand_landmarks.landmark:
            rh_coords.extend([res.x - wrist_x, res.y - wrist_y, res.z - wrist_z])
            
        rh_coords = normalize_scale(rh_coords)
    else:
        rh_coords = list(np.zeros(63)) # 21 titik x 3 (X,Y,Z)
        
    # Total Vektor Array: 18 + 63 + 63 = 144 Dimensi
    return np.array(pose_coords + lh_coords + rh_coords)


def calculate_movement_score(prev_vector, curr_vector):
    """
    Menghitung jarak L2 Norm dari vektor 144-D antar 2 frame berurutan.
    Fungsi ini menjadi inti dari sistem Visual Activity Detection (VAD) 
    untuk mendeteksi kapan gerakan isyarat dimulai dan ditahan.
    """
    if prev_vector is None or curr_vector is None:
        return 0.0
        
    # Selisih Euclidean antar vektor 144-dimensi
    diff = prev_vector - curr_vector
    return float(np.linalg.norm(diff))