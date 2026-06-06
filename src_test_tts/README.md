# Standalone Indonesian TTS Test Lab

`src_test_tts/` adalah modul testing terpisah untuk eksperimen TTS Bahasa Indonesia. Modul ini belum terhubung ke aplikasi utama dan tidak membutuhkan perubahan apa pun di `src/`.

Backend utama memakai repo `https://github.com/drat/TTS-Indonesia-Gratis`, yang menggunakan G2P Indonesia dan Coqui TTS. Speaker dasar:

- `Wibowo`: baseline cowok dewasa.
- `Gadis`: baseline cewek dewasa.

Preset remaja dan anak-anak hanya simulasi post-processing dari speaker dewasa, jadi wajib divalidasi dengan mendengarkan langsung.

## Setup Venv

Wajib gunakan Python 3.10 dan venv `env_bisindo_cuda126`.

```bash
bash src_test_tts/setup_env.sh
source env_bisindo_cuda126/bin/activate
```

Cek venv aktif:

```bash
source env_bisindo_cuda126/bin/activate
which python
python --version
pip --version
```

Kalau ingin recreate venv:

```bash
rm -rf env_bisindo_cuda126
bash src_test_tts/setup_env.sh
source env_bisindo_cuda126/bin/activate
```

Semua contoh command di bawah diasumsikan dijalankan setelah:

```bash
source env_bisindo_cuda126/bin/activate
```

## Command Utama

```bash
python src_test_tts/cli.py init
python src_test_tts/cli.py download-model
python src_test_tts/cli.py list
python src_test_tts/cli.py show cowok_dewasa_default
python src_test_tts/cli.py generate --profile cewek_remaja_default --text "Halo, ini percobaan suara remaja perempuan."
python src_test_tts/cli.py generate --profile cewek_remaja_default --text "Halo, ini percobaan suara remaja perempuan." --play
python src_test_tts/cli.py live --profile cewek_remaja_default
python src_test_tts/cli.py baseline-tests
python src_test_tts/cli.py analyze --input src_test_tts/outputs/baseline_tests
python src_test_tts/app.py
```

`cli.py` dan `app.py` akan menampilkan warning kalau venv yang aktif bukan `env_bisindo_cuda126`. Pakai `--strict-venv` untuk memaksa program berhenti jika venv salah.

## Dependency Safety

Modul ini memakai venv yang sama dengan main project, jadi dependency sensitif dikunci di `src_test_tts/constraints.txt`. Baseline yang sudah diuji:

- `torch==2.8.0`
- `torchaudio==2.8.0`
- CUDA PyTorch: `12.6`
- `numpy==1.22.0`
- `scipy==1.11.4`

Kalau `torchaudio` rusak karena mismatch CUDA, perbaiki hanya `torchaudio` dan jangan ikutkan dependency turunannya:

```bash
source env_bisindo_cuda126/bin/activate
python -m pip install --no-deps --force-reinstall torchaudio==2.8.0
```

Jangan upgrade/downgrade `torch`, `numpy`, atau `scipy` kecuali benar-benar diperlukan dan sudah dicek tidak mengganggu main program.

## Model

Lokasi model lokal:

```text
src_test_tts/models/tts_indonesia_gratis/
```

Auto-download:

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py download-model
```

File yang dipakai:

- `checkpoint_1260000-inference.pth`
- `config.json`
- `speakers.pth`
- `languages.json` opsional

Downloader memakai release `Wikidepia/indonesian-tts` yang dirujuk oleh upstream `TTS-Indonesia-Gratis`. Kalau auto-download gagal, download manual dari:

```text
https://github.com/Wikidepia/indonesian-tts/releases/tag/v1.2
```

Lalu taruh file di:

```text
src_test_tts/models/tts_indonesia_gratis/
```

Verifikasi:

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py download-model
```

Hapus model untuk download ulang:

```bash
rm -f src_test_tts/models/tts_indonesia_gratis/checkpoint_1260000-inference.pth
rm -f src_test_tts/models/tts_indonesia_gratis/config.json
rm -f src_test_tts/models/tts_indonesia_gratis/speakers.pth
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py download-model
```

Alternatif checkpoint manual:

```bash
source env_bisindo_cuda126/bin/activate
export TTS_INDONESIA_GRATIS_MODEL_URL="https://alamat-manual/checkpoint_1260000-inference.pth"
python src_test_tts/cli.py download-model
```

## Profile Voice

Semua profile disimpan di:

```text
src_test_tts/configs/voice_profiles.json
```

Format nama wajib:

```text
{gender}_{demografi}_{variasi_conf}
```

Contoh valid:

- `cowok_dewasa_default`
- `cewek_remaja_natural_01`
- `cowok_anak_anak_ceria_01`

Simpan profile baru:

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py save-profile --gender cewek --demografi remaja --variasi-conf soft_01 --base-speaker Gadis --pitch 0.6 --speed 1.03 --volume 0 --formant 0.04
```

Overwrite hanya jika sengaja:

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py save-profile --gender cewek --demografi remaja --variasi-conf soft_01 --base-speaker Gadis --overwrite
```

## Generate Dan Output

Generate satu suara:

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py generate --profile cewek_remaja_default --text "Halo, ini percobaan suara remaja perempuan."
```

Generate lalu langsung putar ke speaker default:

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py generate --profile cewek_remaja_default --text "Halo, ini percobaan suara remaja perempuan." --play
```

Live text-to-speech terminal:

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py live --profile cewek_remaja_default
```

Ketik teks lalu Enter. Audio akan digenerate ke `src_test_tts/outputs/live_tests/` dan langsung diputar. Ketik `:q` atau `exit` untuk keluar.

Live mode memuat model sekali di awal dan menjalankan warmup singkat. Load/warmup pertama bisa lebih lama, tetapi input berikutnya tidak reload model lagi sehingga targetnya teks pendek menjadi suara dalam sekitar 2 detik atau kurang, tergantung panjang teks dan setting efek audio. Untuk latency terendah, live mode default tidak menjalankan audio analysis. Jika butuh analisis saat live:

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py live --profile cewek_remaja_default --analysis
```

Jika device CUDA bermasalah, paksa CPU:

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py live --profile cewek_remaja_default --device cpu
```

Setiap generate menyimpan:

- raw WAV dari TTS model.
- final WAV hasil post-processing.
- metadata JSON di sebelah WAV.

Output default:

```text
src_test_tts/outputs/
```

Metadata JSON berisi teks, profile, config lengkap, speaker, raw/final path, waktu pembuatan, hasil audio analysis, dan warning.

## Baseline Listening/Test Pack

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py baseline-tests
```

Output:

```text
src_test_tts/outputs/baseline_tests/
```

Baseline yang dibuat:

- Wibowo raw/default
- Gadis raw/default
- `cowok_dewasa_default`
- `cewek_dewasa_default`
- `cowok_remaja_default`
- `cewek_remaja_default`
- `cowok_anak_anak_default`
- `cewek_anak_anak_default`

Kalimat uji dibuat sama untuk semua profile agar mudah dibandingkan.

## Audio Analysis

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/cli.py analyze --input src_test_tts/outputs/baseline_tests
```

Report:

```text
src_test_tts/outputs/baseline_tests/analysis_report.json
src_test_tts/outputs/baseline_tests/analysis_report.csv
```

Metric:

- estimated F0 mean/median
- durasi
- RMS loudness
- peak dB
- spectral centroid
- sample rate
- jumlah sample
- clipping detection
- speech rate estimate jika teks tersedia

Analisis objektif hanya alat bantu. Keputusan natural/tidak tetap harus dari pendengaran manusia.

## UI Gradio

```bash
source env_bisindo_cuda126/bin/activate
python src_test_tts/app.py
```

UI menyediakan dropdown profile, gender, demografi, speaker Wibowo/Gadis, tuning pitch/speed/volume/formant/filter/normalize/compressor/sample-rate, generate, audio preview, save profile, baseline tests, dan tabel analysis report. Bagian `Live Test` memakai runtime yang sama dan tidak reload model tiap teks setelah warmup. Tombol `Warmup Live` tersedia jika ingin memanaskan model sebelum mengetik teks pertama.

## Dependency Eksternal

Python dependency ada di `src_test_tts/requirements.txt` dan harus diinstall di venv.

Dependency sistem yang membantu audio processing:

- `ffmpeg`: dipakai untuk pitch shift yang lebih stabil jika tersedia.
- `sox`: opsional untuk inspeksi/manual audio.
- `rubberband`: opsional, belum wajib. Jika tidak tersedia, `formant_shift` disimpan di config tetapi belum diterapkan dan akan muncul warning.

Jangan install dependency global. Selalu aktifkan venv dulu.

## Keterbatasan

- Tuning umur remaja/anak-anak bukan voice model asli, hanya simulasi dari Wibowo/Gadis.
- Pitch/formant/speed tuning tidak akan senatural model yang memang dilatih untuk umur tersebut.
- Preset awal adalah starting point, bukan final.
- Output anak-anak paling rawan terdengar tidak natural, jadi perlu preview dan tuning manual.
- Model dan output besar tidak boleh masuk git; sudah di-ignore oleh `src_test_tts/.gitignore`.
