# BISINDO MediaPipe Ultra Full

Pure MediaPipe live extractor yang dibuat khusus untuk target **10 FPS+** di Jetson/CPU dengan kamera wide 640x480.

Fokus optimasi:

- Tidak pakai YOLO/TensorRT.
- Tidak pakai Holistic.
- MediaPipe Hands tetap jadi proses utama per-frame.
- MediaPipe Pose untuk bahu bisa dimatikan total atau dijalankan jarang.
- Kamera memakai latest-frame-only supaya tidak ada frame lama yang numpuk.
- Overlay/GIF/recording mati default atau hanya aktif saat diminta.
- Fitur banyak mode: 84, 179, 228, 268, 288, `btj_global`, `btj_local`, `btj_global_local`.

## Install

```bash
pip install mediapipe opencv-python numpy imageio
```

Kalau OpenCV sistem sudah ada di Jetson, cukup:

```bash
pip install mediapipe numpy imageio
```

## Run pertama, target 10 FPS+

```bash
bash run_ultra_10fps.sh
```

Mode ini:

- feature mode: `btj_global_local` = 180 dimensi
- shoulder: fixed/manual anchor, bukan pose model
- proc width: 256
- overlay: off
- GIF: off

## Run lebih akurat

```bash
bash run_ultra_accurate_10fps.sh
```

Mode ini:

- feature mode: `268`
- shoulder: MediaPipe Pose, tapi cuma tiap 20 frame
- proc width: 288
- overlay: off

## Benchmark murni tanpa display

```bash
bash run_benchmark_no_display.sh btj_global_local
```

Ganti mode:

```bash
bash run_benchmark_no_display.sh 84
bash run_benchmark_no_display.sh 179
bash run_benchmark_no_display.sh 268
```

Benchmark ini paling bersih karena tidak ada `imshow`, overlay, atau drawing.

## Bandingkan semua mode

```bash
bash run_modes_compare.sh
```

Tutup window dengan `Q`, nanti lanjut mode berikutnya.

## Tombol saat live

| Tombol | Fungsi |
|---|---|
| `Q` / `ESC` | keluar |
| `R` | start/stop record fitur ke NPZ + CSV |
| `S` | simpan snapshot 1 frame |
| `O` | toggle overlay skeleton |
| `H` | toggle hold last good hand |
| `X` | swap kiri/kanan |
| `G` | save GIF, hanya kalau run dengan `--enable-gif-buffer` |

## Mode fitur dan dimensi

| Mode | Dimensi | Isi utama |
|---|---:|---|
| `84` | 84 | bahu + palm/wrist descriptor + pair distance + angle jari |
| `179` | 179 | bahu + full hand global 21 titik kiri/kanan + angle jari + meta |
| `228` | 228 | bahu + full global + compact local telapak/jari + angle jari + meta kecil |
| `268` | 268 | bahu + full hand global + full hand local + meta |
| `288` | 288 | 268 + geometry/quality/pseudo-depth tambahan |
| `btj_global` | 114 | bahu + telapak/jari selected keypoints global + angle jari + meta |
| `btj_local` | 114 | bahu + telapak/jari selected keypoints local + angle jari + meta |
| `btj_global_local` | 180 | bahu + telapak/jari selected keypoints global+local + angle jari + meta |

`btj` = **bahu + telapak tangan + jari**.

Selected keypoints per tangan untuk `btj_*`:

- wrist
- palm center
- thumb tip
- index MCP + index tip
- middle MCP + middle tip
- ring MCP + ring tip
- pinky MCP + pinky tip

## Local vs global

Global:

```text
posisi titik tangan relatif ke tengah bahu / lebar bahu
```

Local:

```text
bentuk tangan relatif ke wrist / ukuran telapak
```

Jadi:

- `btj_global` tahu posisi tangan terhadap badan.
- `btj_local` tahu bentuk tangan/jari.
- `btj_global_local` tahu posisi + bentuk.

## Kalau FPS masih di bawah 10

Turunkan bertahap:

```bash
--proc-width 224
```

Lalu:

```bash
--center-crop 0.80
```

Lalu:

```bash
--hand-every 2
```

Catatan: `--hand-every 2` berarti tangan dihitung tiap 2 frame, frame sisanya pakai hold. Ini menaikkan FPS, tapi fitur temporal jadi tidak sehalus `hand-every 1`.

## Kalau skeleton terasa ketinggalan

Naikkan responsivitas:

```bash
--smooth-alpha 0.90 --hold-frames 1
```

Kalau terlalu jitter:

```bash
--smooth-alpha 0.75 --hold-frames 3
```

## Rekomendasi eksperimen

Mulai dari:

```text
btj_global_local = 180 dimensi
```

Kalau gesture yang mirip bentuk jari sering salah, naik ke:

```text
268 dimensi
```

Kalau dataset kecil dan model overfit, turun ke:

```text
84 dimensi
```

## Output record

Saat tekan `R`, hasil disimpan ke folder:

```text
runs_mp_ultra_full/
```

Format:

- `.npz` untuk training Python
- `.csv` untuk inspeksi cepat
- `_meta.json` untuk status frame
