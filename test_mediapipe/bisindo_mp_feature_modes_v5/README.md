# BISINDO Pure MediaPipe Feature Modes V5

Target: live test 640x480, kamera wide RGB biasa, pure MediaPipe, fokus **bahu + telapak tangan + jari**.

Tidak pakai YOLO/TensorRT. Tujuannya menjaga akurasi landmark dan memberi beberapa format fitur untuk dibandingkan saat training.

## Mode fitur

| Mode | Dimensi | Isi utama | Kapan dipakai |
|---|---:|---|---|
| `84` | 84 | shoulder/body + palm/wrist + inter-hand + finger angles | baseline kecil, dataset masih sedikit |
| `179` | 179 | shoulder + full hand global xyz + finger angles + meta | rekomendasi awal |
| `228` | 228 | 179 + selected local palm/finger + geometry | kalau 179 kurang detail pada bentuk/arah jari |
| `268` | 268 | shoulder + full hand global + full hand local + meta | full global+local, tanpa extra geometry |
| `288` | 288 | 268 + 20 geometry/quality/pseudo-depth | paling lengkap, dataset harus lebih banyak |
| `btj_global` | 114 | bahu + telapak+jari selected keypoints global + angles + meta | hanya posisi relatif tubuh |
| `btj_local` | 114 | bahu + telapak+jari selected keypoints local + angles + meta | hanya bentuk tangan relatif wrist |
| `btj_global_local` | 180 | gabungan selected global + selected local + angles + meta | kompromi bagus: posisi + bentuk |

`btj` = bahu + telapak tangan + jari. Selected keypoints per tangan: wrist, palm center, thumb tip, index MCP/tip, middle MCP/tip, ring MCP/tip, pinky MCP/tip.

## Install

```bash
pip install mediapipe opencv-python numpy imageio
```

Kalau di Jetson sudah pakai OpenCV bawaan sistem, cukup:

```bash
pip install mediapipe numpy imageio
```

## Run rekomendasi awal

```bash
bash run_179.sh
```

Kalau mau paling fokus ke bahu+telapak+jari global+local:

```bash
bash run_btj_global_local.sh
```

Kalau mau full global+local seperti sebelumnya:

```bash
bash run_268.sh
```

## Tombol live

- `Q` / `ESC`: keluar
- `R`: start/stop record ke NPZ + CSV
- `S`: simpan snapshot 1 frame
- `O`: overlay on/off
- `G`: simpan GIF kalau run pakai `--enable-gif-buffer`
- `H`: hold-last-good on/off
- `M`: mirror preview only
- `X`: swap left/right label kalau kebalik

## Catatan akurasi/FPS

- FPS terutama dipengaruhi MediaPipe Hands + Pose, bukan jumlah fitur.
- Untuk akurasi, gunakan `--hand-model-complexity 1`, `--proc-width 384/416`, `--hand-every 1`.
- Untuk FPS, gunakan `--hand-model-complexity 0`, `--proc-width 288/320`, `--shoulder-every 12`, `--no-overlay`.
- Karena kamera wide, default `--center-crop 0.92` untuk mengurangi distorsi pinggir.

## Output

Output masuk ke folder `runs_mp_feature_modes_v5/`, berisi `.npz`, `.csv`, dan metadata JSON.
