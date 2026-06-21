# BISINDO + Sherpa Lightweight GUI

Jalankan dari root repo:

```bash
PYTHONPATH=src:LLM:. ./env_bisindo_cuda126/bin/python -m src_integrasi.app
```

GUI ini tidak mengganti program live yang sudah ada. Jalur BISINDO tetap memakai
Smart180 ADI augmentasi, `route=main`, MediaPipe holistic, profil live
`lossless1080_10`, LLM `bisindo-prompt-qwen3b`, dan TTS profile default
`cewek_dewasa_default`.

## Shortcut

- `p`: pause/resume live BISINDO.
- `v`: toggle preview.
- `space`: tambah token `{spasi}` ke buffer.
- `q` atau `Esc`: keluar bersih.

## Sherpa

Sherpa di-load lazy saat tombol `Start STT` dipakai. Model default:

```text
Sherpa/models/sherpa-onnx-streaming-zipformer2-id
```

UDP default:

```text
listen 0.0.0.0:8080
audio PCM 16-bit mono little-endian 16000 Hz
```

Tombol `Konek UDP` menyalakan Zeroconf broadcast. Default broadcast mengikuti
`Sherpa/hotspot_broadcast2.py`, yaitu `10.42.0.1:8080`. Kalau IP hotspot itu
belum aktif, GUI fallback ke auto-detect seperti `Sherpa/hotspot_broadcast.py`.

## Dependency

Dependency/version existing tidak diubah. Untuk fitur Sherpa, pastikan paket
optional ini tersedia di environment yang dipakai:

```text
sherpa_onnx
onnxruntime
zeroconf
jiwer
soundfile
scipy  # hanya perlu kalau evaluasi WAV butuh resample
```

Kalau dependency belum tersedia, GUI tetap bisa terbuka dan akan menampilkan
error saat fitur terkait dipakai.

## Hide Tab Testing

Untuk deployment ringan, ubah satu baris di `src_integrasi/app.py`:

```python
ENABLE_SHERPA_TEST_TAB = False
```

Saat `False`, tab WER/file-evaluation tidak dibuat dan dependency testing tidak
di-import sampai dipakai oleh kode lain.

