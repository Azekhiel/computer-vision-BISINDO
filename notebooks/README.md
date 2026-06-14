# BISINDO Training Notebooks

Folder ini berisi notebook training GRU BISINDO.

## File Utama

- `train_bisindo_gru_colab_v4.ipynb`
  - Untuk Colab atau Kaggle.
  - Multi-select schema, model, mode data, dan suite.
  - Setelah tiap kombinasi selesai, model langsung disimpan ke target Drive/output, dibuat zip, lalu di-download otomatis jika berjalan di Colab.

- `train_bisindo_gru_local_v4.ipynb`
  - Untuk laptop lokal, termasuk Windows lewat VS Code/Jupyter.
  - Pakai dataset lokal dulu jika ada.
  - Kalau schema yang dipilih belum ada, notebook mencoba download dari link yang diisi.
  - Zip hasil training dibuat di folder `downloads/`.

## Struktur Dataset

Dataset path/root harus berisi folder per schema:

```text
dataset_parquets/
  smart180/
    *.parquet
  smart180_handface_vel286/
    *.parquet
```

Notebook hanya mengambil schema yang dicentang. Kalau root berupa Google Drive folder link, notebook mencoba mencari subfolder bernama schema, misalnya `smart180_handface_vel286`. Kalau root link tidak bisa dilist karena permission/auth, isi fallback `SCHEMA_DATASET_LINKS_TEXT`.

Format fallback link per schema:

```text
smart180=https://drive.google.com/drive/folders/...,smart180_handface_vel286=https://drive.google.com/drive/folders/...
```

## Output Model

Output mengikuti struktur repo:

```text
<target>/gru/<schema>/
  gru_<variant>.pth
  gru_<variant>_labels.json
  gru_<variant>_metadata.json
  experts/
```

Selain disimpan ke target, setiap kombinasi juga dibuat zip:

```text
downloads/<schema>__<variant>__<suite>.zip
```

## Colab

1. Buka `train_bisindo_gru_colab_v4.ipynb`.
2. Aktifkan GPU: `Runtime > Change runtime type > GPU`.
3. Jalankan cell environment dan mount Drive.
4. Isi `DRIVE_DATASET_DIR` dengan path Drive mounted atau Google Drive folder link.
5. Isi `DRIVE_MODEL_DIR` dengan folder output Drive/path.
6. Centang schema/model/suite yang ingin dilatih.
7. Jalankan cell berurutan.

Jika `DRIVE_MODEL_DIR` berupa link, Drive API harus bisa auth. Jika tidak, pakai path mounted seperti `/content/drive/MyDrive/bisindo_models`.

## Kaggle

Gunakan `train_bisindo_gru_colab_v4.ipynb`.

- Untuk dataset Kaggle, set `DRIVE_DATASET_DIR` ke path seperti `/kaggle/input/<dataset-name>/dataset_parquets`.
- Untuk output, set `DRIVE_MODEL_DIR` ke `/kaggle/working/bisindo_models`.
- Download otomatis browser biasanya tidak tersedia; ambil zip dari folder `DOWNLOAD_DIR`.

## Windows Local

1. Buka repo di VS Code atau Jupyter.
2. Buka `train_bisindo_gru_local_v4.ipynb`.
3. Pastikan environment Python punya dependency training repo, terutama `torch`, `pandas`, `pyarrow`, `scikit-learn`, `tqdm`, dan `gdown`.
4. Letakkan dataset di `dataset_parquets/<schema>/`.
5. Kalau dataset belum ada, isi `SCHEMA_DATASET_LINKS_TEXT`.
6. Jalankan cell berurutan.

Default local:

- Dataset: `dataset_parquets/`
- Model: `models/`
- Zip: `downloads/`

## Catatan Aman

- `OVERWRITE_EXISTING` default `False`, jadi checkpoint yang sudah ada akan di-skip.
- `SAVE_TO_DRIVE` default `True` di notebook Colab/Kaggle agar hasil tidak cuma tertinggal di runtime sementara.
- `DOWNLOAD_AFTER_EACH_COMBO` bisa dimatikan kalau browser terlalu sering membuka dialog download.
