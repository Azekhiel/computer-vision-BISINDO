import numpy as np

# ekstrak koordinat mediapipe jadi relatif ke hidung
# biar posisi orang di kamera ga ngaruh ke akurasi
def extract_keypoints_relative(results):
    # set default anchor di tengah kalo pose ga kedetect
    anchor_x, anchor_y, anchor_z = 0.5, 0.5, 0.0 
    
    if results.pose_landmarks:
        # pake hidung (landmark 0) sbg titik pusat 0,0,0
        anchor_x = results.pose_landmarks.landmark[0].x
        anchor_y = results.pose_landmarks.landmark[0].y
        anchor_z = results.pose_landmarks.landmark[0].z

    def get_relative_coords(landmarks, num_points):
        if not landmarks:
            return np.zeros(num_points * 3)
        
        rel_coords = []
        for res in landmarks.landmark:
            # nilai asli dikurangin nilai jangkar
            rel_coords.extend([
                res.x - anchor_x, 
                res.y - anchor_y, 
                res.z - anchor_z
            ])
        return rel_coords

    pose = get_relative_coords(results.pose_landmarks, 33)
    face = get_relative_coords(results.face_landmarks, 468)
    lh = get_relative_coords(results.left_hand_landmarks, 21)
    rh = get_relative_coords(results.right_hand_landmarks, 21)
    
    return np.concatenate([pose, face, lh, rh])

# ngitung jarak l2 norm dari 2 frame buat nentuin user lagi gerak atau diem
def calculate_movement_score(prev_vector, curr_vector):
    if prev_vector is None or curr_vector is None:
        return 0.0
    # selisih antar vektor
    diff = prev_vector - curr_vector
    return np.linalg.norm(diff)