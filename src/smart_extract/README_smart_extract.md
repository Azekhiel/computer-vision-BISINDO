# BISINDO Smart Extract V8

Ini extractor untuk video yang sudah terlanjur direkam dan kualitasnya kadang lebih jelek daripada live.

Fitur utama:
- output tetap 10 FPS
- default `center-crop 1.0`, jadi tangan pinggir tidak kepotong
- bahu real pakai MediaPipe Pose
- auto brightness/gamma/CLAHE/sharpen
- mode `smart`: kalau frame target lemah, coba beberapa versi enhancement
- mode `best`: cari frame terbaik di sekitar timestamp target, cocok untuk video blur/dropframe
- batch folder untuk ribuan video

## Quick test satu video

```bash
bash run_smart_extract_one.sh "/path/video.mp4"
```

## Best quality satu video

Lebih lambat, tapi paling kuat untuk video lama/compressed/blur.

```bash
bash run_best_extract_one.sh "/path/video.mp4"
```

## Batch ribuan video

Default batch tidak menyimpan GIF supaya tidak berat. Batch sekarang memakai preset `best`
yang sama dengan `run_best_extract_one.sh`, supaya semua video import/migrasi identik.

```bash
bash run_smart_extract_batch.sh "/path/folder_video"
```

Output batch masuk ke folder:

```text
/path/folder_video/_smart_extract_v8_10fps/
```

## Mode fitur

Default `btj_global_local` = 180 dimensi.

Bisa juga:

```bash
bash run_smart_extract_one.sh "/path/video.mp4" 268
```

## Kapan pakai smart vs best?

- `smart`: untuk sebagian besar video, lebih cepat.
- `best`: kalau skeleton masih sering hilang. Dia cek frame sekitar target dan pilih frame+enhancement terbaik.

## Setting paling penting

- `--center-crop 1.0`: jangan potong tangan.
- `--proc-width 320` atau `384`: lebih akurat dari 256.
- `--det-conf 0.40` sampai `0.45`: video compressed lebih gampang dideteksi.
- `--hold-frames 4/5`: skeleton tidak gampang hilang.
