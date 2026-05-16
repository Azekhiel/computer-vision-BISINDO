BISINDO_Project/
│
├── src/
│   │   # --- 1. DATA & PROCESSING LAYER ---
│   ├── data_ingestion.py         # [SELESAI] Import video massal, Auto-Trimmer, label split (Train/Val/Test)
│   ├── augmentation_factory.py   # [SELESAI] Generate 200+ data spasial-temporal HANYA untuk data 'train'
│   ├── database_manager.py       # [SELESAI] Otak CRUD, kalkulasi statistik, tracker status model
│   ├── feature_engine.py         # [V3.1] Body-frame kinematic preprocessing, segment-aware occlusion handling
│   │
│   │   # --- 2. CORE AI ENGINES ---
│   ├── faiss_manager.py          # [V3.1] FAISS descriptor DCT posisi+velocity, IP threshold
│   ├── lstm_manager.py           # [V3.1] Bi-Directional LSTM dengan masked attention
│   ├── transformer_manager.py    # [V3.1] Spatial-Temporal Transformer dengan padding mask
│   │
│   │   # --- 3. INFERENCE & UI LAYER ---
│   ├── inference_engine.py       # [V3.1] Worker-thread live inference, VAD 30-frame domain-matched
│   └── main_ui.py                # [V3.1] Dashboard Tkinter non-blocking
│
├── data_raw/                     # Folder tempat naruh video/GIF mentah sebelum di-import
│   ├── train/                    # └─ misal: /train/terima_kasih/video1.mp4
│   ├── val/                      # └─ misal: /val/terima_kasih/video_test1.mp4
│   └── test/
│
├── database/                     
│   ├── dataset_dynamic.csv       # (Akan terbuat otomatis) Database utama deret waktu
│   └── vocab_list.txt            # Daftar kosakata terdaftar
│
├── models/                       
│   ├── model_status.json         # (Akan terbuat otomatis) Menyimpan log kapan DB dan Model terakhir diupdate
│   ├── sign_language.index       # Index pencarian FAISS
│   ├── label_map.npy             # Pemetaan label FAISS
│   ├── lstm_weights.pth          # Bobot hasil training Model 2 (LSTM)
│   └── transformer_weights.pth   # Bobot hasil training Model 3 (Transformer)
│
└── generated_gifs/               # Folder render GIF otomatis buat preview di UI
