# BISINDO Video Skeleton Extractor (10 FPS)

Script ini mengambil video input lalu:

1. men-sample video ke **target FPS tetap** (default 10 FPS)
2. mengekstrak skeleton tangan+bahu dengan **pure MediaPipe**
3. menyimpan:
   - GIF **overlay** (video asli/crop + skeleton)
   - GIF **skeleton only**
   - fitur `.npz`
   - fitur `.csv`
   - metadata `.json`

## Kenapa sampling berbasis waktu?

Bukan sekadar "ambil tiap N frame". Script ini memakai **timestamp video**:
- cocok untuk video 15 FPS, 24 FPS, 25 FPS, 30 FPS, dll
- output tetap 10 FPS
- frame yang diambil mengikuti grid waktu 0.0s, 0.1s, 0.2s, ...

Jadi lebih stabil untuk dataset temporal.

## Pakai cepat

```bash
cd bisindo_video_extract_10fps
bash run_extract_10fps.sh /path/ke/video.mp4
```

Mode default: `btj_global_local` (180 dimensi).

## Pakai manual

```bash
python3 extract_video_skeleton_10fps.py /path/ke/video.mp4 \
  --feature-mode btj_global_local \
  --target-fps 10 \
  --width 640 --height 480 \
  --proc-width 256 \
  --center-crop 0.86 \
  --shoulder-backend none
```

## Opsi penting

- `--feature-mode`: `84`, `179`, `228`, `268`, `288`, `btj_global`, `btj_local`, `btj_global_local`
- `--target-fps`: default `10`
- `--center-crop`: default `0.86` untuk wide camera
- `--shoulder-backend none`: paling cepat
- `--shoulder-backend mp-pose`: bahu dari MediaPipe Pose, lebih akurat tapi lebih lambat
- `--gif-width`: resize output GIF biar file size tidak terlalu besar

## Output

Secara default output masuk ke folder:

```text
<nama_video>_extract_10fps/
```

Isi file:
- `*_overlay.gif`
- `*_skeleton_only.gif`
- `*.npz`
- `*.csv`
- `*_meta.json`

