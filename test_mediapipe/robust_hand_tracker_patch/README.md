# BISINDO Pure MediaPipe Accurate 288

Target: **akurasi dulu**, FPS cukup sekitar **10 FPS**.

Tidak pakai YOLO/TensorRT. Fokus:
- MediaPipe Hands model complexity 1
- input tangan 384 px default
- shoulder anchor dari MediaPipe Pose tiap beberapa frame
- low latency latest-frame-only camera
- hold-last-good-frame untuk occlusion pendek/self-handshake
- fitur 288 dimensi per frame

## Run utama

```bash
bash run_accurate_10fps.sh
```

Kalau FPS masih terlalu berat, tapi tangan sudah akurat:

```bash
bash run_accurate_10fps_no_pose.sh
```

Kalau FPS masih aman dan mau tambah akurat:

```bash
bash run_max_accuracy_if_fps_ok.sh
```

## Feature dimension

Total = **288 dimensi/frame**.

Layout:

```text
0:6       bahu kiri/kanan normalized
6:69      tangan kiri global 21*xyz
69:132    tangan kanan global 21*xyz
132:195   tangan kiri local 21*xyz
195:258   tangan kanan local 21*xyz
258:268   metadata/status 10 dim
268:278   geometri/kualitas tangan kiri 10 dim
278:288   geometri/kualitas tangan kanan 10 dim
```

Extra 20 dimensi baru:

```text
per tangan:
present, detected, held, handedness_score,
bbox_w, bbox_h, bbox_area,
palm_size, scale_vs_shoulder, pseudo_z
```

Kenapa 288? Karena 268 fitur utama tetap dipakai, lalu ditambah 20 fitur kualitas/geometri supaya model lebih paham:
- tangan valid atau hasil hold
- tangan sedang besar/kecil di kamera
- estimasi maju/mundur dari ukuran telapak
- confidence handedness

## Keys

```text
Q / ESC = keluar
R       = start/stop record ke NPZ + CSV
S       = snapshot 1 frame
O       = overlay on/off
G       = save GIF kalau run dengan --enable-gif-buffer
H       = hold last good on/off
M       = mirror preview on/off
X       = swap label kiri/kanan
```

## Kalau label kiri/kanan kebalik

Tekan `X` saat live, atau run dengan:

```bash
--swap-handedness
```

## Kalau kamera wide terlalu distorsi

Turunkan area crop:

```bash
--center-crop 0.88
```

Kalau tangan sering keluar frame, naikkan:

```bash
--center-crop 0.96
```

## Kalau FPS kurang dari 10

Turunkan bertahap:

```bash
--proc-width 352
--shoulder-every 12
--shoulder-backend none
```

Jangan langsung pakai `--hand-every 2` kalau dataset untuk training, karena frame tangan jadi hasil hold.

## Kalau mau GIF

Tambahkan:

```bash
--enable-gif-buffer
```

Lalu tekan `G` saat live. GIF buffer dimatikan default supaya FPS stabil.
