import matplotlib
# WAJIB: Gunakan backend 'Agg' agar tidak terjadi konflik GUI dengan threading Tkinter
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import os

# ==========================================
# KONFIGURASI PATH (Tahan Banting & Partisi)
# ==========================================
# Mengambil path direktori utama (root) secara absolut
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
GIF_OUTPUT_DIR = os.path.join(ROOT_DIR, 'assets', 'gifs')

def generate_vocab_gif(vocab_name):
    """
    Mengambil satu sampel video asli dari database partisi untuk sebuah vocab,
    merender kerangkanya dari vektor 144-D, dan menyimpannya sebagai GIF.
    """
    # Buat folder assets/gifs jika belum ada
    os.makedirs(GIF_OUTPUT_DIR, exist_ok=True)
    gif_path = os.path.join(GIF_OUTPUT_DIR, f"{vocab_name}.gif")
    
    # 1. OPTIMALISASI: Jika GIF sudah pernah dibuat, langsung return (INSTAN)
    if os.path.exists(gif_path):
        return gif_path

    # 2. JIKA BELUM ADA, baca file parquet spesifik vocab tersebut
    vocab_file = os.path.join(DATABASE_DIR, f"{vocab_name}.parquet")
    
    if not os.path.exists(vocab_file):
        return None

    try:
        # BACA DATA (Hanya memuat data vocab yang bersangkutan)
        df_vocab = pd.read_parquet(vocab_file)
        
        if df_vocab.empty:
            return None

        # CARI VIDEO ASLI TERBAIK (Abaikan data augmentasi agar gerakan di GIF akurat)
        asli_df = df_vocab[~df_vocab['video_id'].astype(str).str.contains('_aug_')]
        
        # Fallback jika hanya ada data augmentasi
        if asli_df.empty:
            asli_df = df_vocab

        # Ambil video_id pertama dan urutkan framenya
        target_video_id = asli_df['video_id'].iloc[0]
        video_data = asli_df[asli_df['video_id'] == target_video_id].sort_values('frame_num')
        
        # Ekstrak sequence matriksnya (Frame x 144)
        sequence = np.array([list(map(float, f.split(','))) for f in video_data['features']])

        # 3. PROSES RENDERING MATPLOTLIB
        # Ukuran figsize yang lebih kecil (3,3) akan mempercepat proses render pertama kali
        fig, ax = plt.subplots(figsize=(3, 3))
        
        def update(frame_idx):
            ax.clear()
            # Set batasan sumbu (Sesuai koordinat relatif MediaPipe)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(1.0, -1.0) # Y dibalik agar kepala di atas
            ax.set_title(f"Vocab: {vocab_name.upper()}", fontweight='bold', fontsize=10)
            ax.axis('off')
            
            vector = sequence[frame_idx]
            
            # --- BONGKAR 144 DIMENSI ---
            # Pose (0-17), Left Hand (18-80), Right Hand (81-143)
            pose = vector[0:18].reshape(-1, 3) # 6 titik x 3 (X,Y,Z)
            lh = vector[18:81].reshape(-1, 3)  # 21 titik x 3 (X,Y,Z)
            rh = vector[81:144].reshape(-1, 3) # 21 titik x 3 (X,Y,Z)

            # Fungsi helper untuk menggambar tulang tangan
            def draw_hand(hand_points, color):
                if np.all(hand_points == 0): return
                
                # Koneksi standar MediaPipe Hand
                connections = [
                    (0,1), (1,2), (2,3), (3,4),       # Jempol
                    (0,5), (5,6), (6,7), (7,8),       # Telunjuk
                    (5,9), (9,10), (10,11), (11,12),  # Tengah
                    (9,13), (13,14), (14,15), (15,16),# Manis
                    (13,17), (17,18), (18,19), (19,20),# Kelingking
                    (0,17)                            # Telapak
                ]
                for start, end in connections:
                    ax.plot([hand_points[start][0], hand_points[end][0]], 
                            [hand_points[start][1], hand_points[end][1]], color=color, linewidth=2)
                ax.scatter(hand_points[:, 0], hand_points[:, 1], color='black', s=5)

            # Gambar Tangan Kiri (Merah) dan Kanan (Biru)
            draw_hand(lh, 'red')
            draw_hand(rh, 'blue')
            
            # Gambar Bahu & Lengan
            if not np.all(pose == 0):
                # Garis bahu
                ax.plot([pose[0][0], pose[1][0]], [pose[0][1], pose[1][1]], color='gray', linestyle='--')
                # Lengan Kiri & Kanan
                ax.plot([pose[0][0], pose[2][0], pose[4][0]], [pose[0][1], pose[2][1], pose[4][1]], color='gray')
                ax.plot([pose[1][0], pose[3][0], pose[5][0]], [pose[1][1], pose[3][1], pose[5][1]], color='gray')

        # Buat animasi
        # Interval 50ms = 20 FPS (Sesuai kecepatan standar MediaPipe)
        anim = animation.FuncAnimation(fig, update, frames=len(sequence), interval=50)
        
        # Simpan menggunakan writer Pillow
        anim.save(gif_path, writer='pillow')
        plt.close(fig) # Tutup figure untuk menghemat memori
        
        return gif_path

    except Exception as e:
        print(f"Error rendering GIF untuk {vocab_name}: {e}")
        return None