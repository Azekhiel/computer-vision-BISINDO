import numpy as np

def extract_keypoints_relative(results):
    """
    Ekstraksi 144-D fitur spasial dengan tingkat akurasi matematis tertinggi (Flawless).
    Menggunakan teknik:
    1. Translation Invariance: Anchor di Mid-Shoulder (Tengah Bahu) untuk badan, 
       dan Pergelangan (Wrist) untuk masing-masing tangan.
    2. Scale Invariance: Normalisasi frame-by-frame menggunakan 'Shoulder Width' (Lebar Bahu). 
       Jarak tubuh dari kamera tidak akan merusak proporsi fitur.
    """
    pose_coords = []
    lh_coords = []
    rh_coords = []

    # ==========================================
    # 1. ANALISIS REFERENSI SKALA (LEBAR BAHU)
    # ==========================================
    shoulder_width = 1.0 # Default fallback jika bahu tidak terdeteksi
    mid_shoulder = np.array([0.0, 0.0, 0.0])
    
    if results.pose_landmarks:
        # Landmark 11 = Bahu Kiri, 12 = Bahu Kanan
        ls = results.pose_landmarks.landmark[11] 
        rs = results.pose_landmarks.landmark[12] 
        
        p_ls = np.array([ls.x, ls.y, ls.z])
        p_rs = np.array([rs.x, rs.y, rs.z])
        
        # Titik gravitasi pusat tubuh (Center of Mass untuk Pose)
        mid_shoulder = (p_ls + p_rs) / 2.0
        
        # Euclidean distance antara bahu kiri dan kanan
        dist = np.linalg.norm(p_ls - p_rs)
        if dist > 0.01: # Cegah error pembagian dengan nol jika pose glitch
            shoulder_width = dist

    # ==========================================
    # 2. EKSTRAK POSE (18 Dimensi)
    # ==========================================
    if results.pose_landmarks:
        # Ekstrak Bahu(11,12), Siku(13,14), Pergelangan(15,16)
        for idx in range(11, 17):
            res = results.pose_landmarks.landmark[idx]
            # KUNCI AKURASI: Anchor di tengah bahu, lalu skalakan jaraknya dengan lebar bahu
            norm_x = (res.x - mid_shoulder[0]) / shoulder_width
            norm_y = (res.y - mid_shoulder[1]) / shoulder_width
            norm_z = (res.z - mid_shoulder[2]) / shoulder_width
            pose_coords.extend([norm_x, norm_y, norm_z])
    else:
        pose_coords = list(np.zeros(18)) # 6 titik x 3 (X,Y,Z)

    # ==========================================
    # 3. EKSTRAK TANGAN KIRI (63 Dimensi)
    # ==========================================
    if results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        wrist_anchor = np.array([wrist.x, wrist.y, wrist.z])
        
        for res in results.left_hand_landmarks.landmark:
            # KUNCI AKURASI: Anchor di pergelangan tangan agar kebal putaran lengan, 
            # lalu skalakan dengan lebar bahu TUBUH agar ukuran tangan sinkron dengan badan.
            norm_x = (res.x - wrist_anchor[0]) / shoulder_width
            norm_y = (res.y - wrist_anchor[1]) / shoulder_width
            norm_z = (res.z - wrist_anchor[2]) / shoulder_width
            lh_coords.extend([norm_x, norm_y, norm_z])
    else:
        lh_coords = list(np.zeros(63)) # 21 titik x 3 (X,Y,Z)

    # ==========================================
    # 4. EKSTRAK TANGAN KANAN (63 Dimensi)
    # ==========================================
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        wrist_anchor = np.array([wrist.x, wrist.y, wrist.z])
        
        for res in results.right_hand_landmarks.landmark:
            norm_x = (res.x - wrist_anchor[0]) / shoulder_width
            norm_y = (res.y - wrist_anchor[1]) / shoulder_width
            norm_z = (res.z - wrist_anchor[2]) / shoulder_width
            rh_coords.extend([norm_x, norm_y, norm_z])
    else:
        rh_coords = list(np.zeros(63)) # 21 titik x 3 (X,Y,Z)
        
    return np.array(pose_coords + lh_coords + rh_coords, dtype=np.float32)

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