BISINDO_Project/
│
├── src/
│   │   # --- 1. DATA & PROCESSING LAYER ---
│   ├── data_ingestion.py         # [SELESAI] Import video massal, Auto-Trimmer, label split (Train/Val/Test)
│   ├── augmentation_factory.py   # [SELESAI] Generate 200+ data spasial-temporal HANYA untuk data 'train'
│   ├── database_manager.py       # [SELESAI] Otak CRUD, kalkulasi statistik, tracker status model
│   ├── feature_engine.py         # [ADA] Ekstrak 144-D fitur relatif Mediapipe & velocity
│   │
│   │   # --- 2. CORE AI ENGINES ---
│   ├── faiss_manager.py          # [REVISI NANTI] Model 1: FAISS dengan Interpolasi Waktu Dinamis
│   ├── lstm_manager.py           # [BELUM] Model 2: Bi-Directional LSTM dengan Attention Mechanism
│   ├── transformer_manager.py    # [BELUM] Model 3: Spatial-Temporal Transformer (SOTA)
│   │
│   │   # --- 3. INFERENCE & UI LAYER ---
│   ├── inference_engine.py       # [BELUM] Mesin Live Test Seamless (Sliding window, Action spotting)
│   └── main_ui.py                # [REVISI NANTI] Dashboard Tkinter yang akan membungkus semua file di atas
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