import matplotlib
# WAJIB: Gunakan backend 'Agg' agar tidak terjadi konflik GUI
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
GIF_OUTPUT_DIR = os.path.join(ROOT_DIR, 'assets', 'gifs')

def generate_vocab_gif(vocab_name):
    os.makedirs(GIF_OUTPUT_DIR, exist_ok=True)
    gif_path = os.path.join(GIF_OUTPUT_DIR, f"{vocab_name}.gif")
    
    if os.path.exists(gif_path):
        return gif_path

    vocab_file = os.path.join(DATABASE_DIR, f"{vocab_name}.parquet")
    if not os.path.exists(vocab_file):
        return None

    try:
        df_vocab = pd.read_parquet(vocab_file)
        if df_vocab.empty: return None

        asli_df = df_vocab[~df_vocab['video_id'].astype(str).str.contains('_aug_')]
        if asli_df.empty: asli_df = df_vocab

        target_video_id = asli_df['video_id'].iloc[0]
        video_data = asli_df[asli_df['video_id'] == target_video_id].sort_values('frame_num')
        
        sequence = np.array([list(map(float, f.split(','))) for f in video_data['features']])

        fig, ax = plt.subplots(figsize=(3, 3))
        
        def update(frame_idx):
            ax.clear()
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(1.2, -1.2) # Y dibalik
            ax.set_title(f"Vocab: {vocab_name.upper()}", fontweight='bold', fontsize=10)
            ax.axis('off')
            
            vector = sequence[frame_idx]
            
            pose = vector[0:18].reshape(-1, 3) 
            lh = vector[18:81].reshape(-1, 3)  
            rh = vector[81:144].reshape(-1, 3) 

            # PERBAIKAN: Tempelkan tangan ke pergelangan dan sesuaikan skala visualnya
            if not np.all(pose == 0):
                left_wrist = pose[4]
                right_wrist = pose[5]
                # Perkecil tangan (0.15) lalu geser ke titik pergelangan
                lh = (lh * 0.15) + left_wrist
                rh = (rh * 0.15) + right_wrist

            def draw_hand(hand_points, color):
                if np.all(hand_points == 0): return
                connections = [
                    (0,1), (1,2), (2,3), (3,4),       
                    (0,5), (5,6), (6,7), (7,8),       
                    (5,9), (9,10), (10,11), (11,12),  
                    (9,13), (13,14), (14,15), (15,16),
                    (13,17), (17,18), (18,19), (19,20),
                    (0,17)                            
                ]
                for start, end in connections:
                    ax.plot([hand_points[start][0], hand_points[end][0]], 
                            [hand_points[start][1], hand_points[end][1]], color=color, linewidth=2)
                ax.scatter(hand_points[:, 0], hand_points[:, 1], color='black', s=5)

            draw_hand(lh, 'red')
            draw_hand(rh, 'blue')
            
            if not np.all(pose == 0):
                ax.plot([pose[0][0], pose[1][0]], [pose[0][1], pose[1][1]], color='gray', linestyle='--')
                ax.plot([pose[0][0], pose[2][0], pose[4][0]], [pose[0][1], pose[2][1], pose[4][1]], color='gray')
                ax.plot([pose[1][0], pose[3][0], pose[5][0]], [pose[1][1], pose[3][1], pose[5][1]], color='gray')

        anim = animation.FuncAnimation(fig, update, frames=len(sequence), interval=50)
        anim.save(gif_path, writer='pillow')
        plt.close(fig) 
        
        return gif_path
    except Exception as e:
        print(f"Error rendering GIF untuk {vocab_name}: {e}")
        return None