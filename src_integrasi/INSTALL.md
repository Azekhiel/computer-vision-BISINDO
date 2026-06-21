# INSTALL — Integrasi BISINDO (Sherpa STT + MQTT)

Panduan lengkap menyiapkan komponen integrasi di Jetson:
1. **Sherpa-ONNX** (speech-to-text Bahasa Indonesia, dipakai tab Sherpa & WER).
2. **MQTT** (Mosquitto broker + client Python untuk komunikasi HP Android ↔ Jetson).

Semua perintah dijalankan dari root repo:
`/home/ta14/Proyek_TA14/computer-vision-BISINDO`

Virtualenv proyek: **`env_bisindo_cuda126`** (Python 3.10, aarch64).
Selalu pakai interpreter venv ini: `./env_bisindo_cuda126/bin/python` dan
`./env_bisindo_cuda126/bin/pip`.

---

## 0. Ringkas (TL;DR)

```bash
cd ~/Proyek_TA14/computer-vision-BISINDO

# Python deps (ke dalam venv proyek). LANGKAH AMAN: cek dulu pakai --dry-run.
# 1) Dry-run: HARUS "Requirement already satisfied" tanpa baris "Would install/uninstall".
./env_bisindo_cuda126/bin/pip install --dry-run "paho-mqtt>=1.6.0" jiwer sherpa-onnx
# 2) Kalau dry-run bersih, baru install (TANPA --upgrade -> yang sudah ada di-skip, tidak ditimpa).
#    Kalau dry-run mau upgrade/uninstall numpy/onnxruntime dkk -> BATALKAN, jangan lanjut.
./env_bisindo_cuda126/bin/pip install "paho-mqtt>=1.6.0" jiwer sherpa-onnx

# Broker MQTT (system service) — biasanya sudah terpasang & jalan
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto

# Cek
./env_bisindo_cuda126/bin/python -c "import sherpa_onnx, jiwer, paho.mqtt.client; print('deps OK')"
systemctl is-active mosquitto && hostname -I
```

> **Aman / no-overwrite.** Tanpa `--upgrade`, pip **tidak menyentuh** paket yang sudah terpasang —
> jadi di Jetson yang sudah jalan, perintah di atas efektif cuma verifikasi. **JANGAN** pakai
> `--upgrade`, **JANGAN** reinstall `onnxruntime`/`numpy` (rawan merusak build GPU/CPU Jetson).
> Status terpasang saat ini (Jun 2026): paho-mqtt 2.1.0, jiwer 4.0.0, sherpa-onnx 1.13.3,
> numpy 1.22.0, onnxruntime 1.23.2, soundfile 0.13.1 — semua sudah ada.
>
> Pemakaian MQTT end-to-end (HP ↔ Jetson): lihat [`tutor_mqtt.md`](tutor_mqtt.md).

Model Sherpa **sudah ada di repo** (tidak perlu download), lihat bagian 2.

---

## 1. Python dependencies (venv)

Status yang diharapkan di venv Jetson:

| Paket          | Status di Jetson ini | Catatan |
|----------------|----------------------|---------|
| `onnxruntime`  | sudah ada            | Jangan reinstall sembarangan di Jetson (urusan build GPU/CPU). |
| `soundfile`    | sudah ada            | Untuk baca WAV di tab WER. |
| `numpy`        | sudah ada            | — |
| `sherpa-onnx`  | **install**          | Ada wheel aarch64 (`sherpa_onnx-1.13.x-...aarch64.whl`). |
| `jiwer`        | **install**          | Hitung WER/MER/WIL. |
| `paho-mqtt`    | **install**          | Client MQTT Python. |

Install yang kurang (cek dulu, jangan menimpa yang sudah ada):

```bash
# Cek rencana pip — aman, tidak mengubah apa pun:
./env_bisindo_cuda126/bin/pip install --dry-run "paho-mqtt>=1.6.0" jiwer sherpa-onnx
# Kalau output semuanya "Requirement already satisfied" -> tidak perlu install apa-apa.
# Kalau ada yang benar-benar kurang, install TANPA --upgrade (yang sudah ada di-skip):
./env_bisindo_cuda126/bin/pip install "paho-mqtt>=1.6.0" jiwer sherpa-onnx
```

> ⚠️ Jangan tambah `--upgrade` dan jangan reinstall `onnxruntime`/`numpy`/`soundfile` yang sudah ada —
> pip tanpa `--upgrade` otomatis melewati paket yang sudah terpenuhi, jadi tidak ada yang ditimpa.

Verifikasi:

```bash
./env_bisindo_cuda126/bin/python - <<'EOF'
import importlib
for m in ["sherpa_onnx","onnxruntime","jiwer","soundfile","paho.mqtt.client","numpy"]:
    importlib.import_module(m); print(m, "OK")
EOF
```

> Catatan onnxruntime: file `Sherpa/onnxruntime_gpu-1.11.0-cp36-...aarch64.whl` di repo
> adalah build lama untuk **Python 3.6** — JANGAN dipakai untuk venv Python 3.10.
> Provider dipilih otomatis (`cuda` kalau tersedia, kalau tidak `cpu`).

---

## 2. Model Sherpa (sudah tersedia)

Model streaming zipformer2 ID sudah ada di repo, tidak perlu download:

```
Sherpa/models/sherpa-onnx-streaming-zipformer2-id/
├── tokens.txt
├── encoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx
├── decoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx
└── joiner-iter-100000-avg-15-chunk-32-left-256.int8.onnx
```

Path ini sudah jadi default di `src_integrasi/sherpa_backend.py`
(`DEFAULT_SHERPA_MODEL_DIR`). Backend memakai varian **int8**.

Verifikasi recognizer bisa load model lokal:

```bash
./env_bisindo_cuda126/bin/python - <<'EOF'
from src_integrasi.sherpa_backend import LazySherpaRecognizer, SherpaModelConfig
r = LazySherpaRecognizer(SherpaModelConfig())
print("provider:", r.load(), "| loaded:", r.loaded)
EOF
```

Output diharapkan: `provider: cpu | loaded: True` (atau `cuda`).

### Alur audio Sherpa
Tab Sherpa **tidak** merekam mic langsung — ia menerima audio via **UDP** (port `8080`)
dari sumber audio (HP/mic embedded), lalu di-decode streaming. Tab "BISINDO" juga
mem-broadcast target UDP via Zeroconf (`hotspot_broadcast.py`).

---

## 3. MQTT — Mosquitto broker (Jetson = broker)

### 3.1 Install (sekali)

```bash
sudo apt update
sudo apt install -y mosquitto mosquitto-clients
```

### 3.2 Konfigurasi listen di semua interface

Supaya HP bisa connect lewat hotspot, broker harus listen di `0.0.0.0`.
Di Jetson ini sudah disetel lewat `/etc/mosquitto/conf.d/default.conf`:

```conf
listener 1883 0.0.0.0
allow_anonymous true
```

> Kalau file itu belum ada di Jetson lain, salin dari repo:
> ```bash
> sudo cp src_integrasi/mqtt/mosquitto.conf /etc/mosquitto/conf.d/bisindo.conf
> sudo systemctl restart mosquitto
> ```
> (Repo juga punya `src_integrasi/mqtt/mosquitto.conf` untuk dipakai manual:
> `mosquitto -c src_integrasi/mqtt/mosquitto.conf -v`.)

### 3.3 Jalankan service

```bash
sudo systemctl enable --now mosquitto
systemctl is-active mosquitto          # -> active
ss -tln | grep 1883                    # -> 0.0.0.0:1883 LISTEN
```

### 3.4 Cari IP Jetson (untuk diisi di MqttDroid)

```bash
hostname -I
```

Pakai IP di jaringan/hotspot yang sama dengan HP (mis. `10.42.0.1` saat Jetson jadi
hotspot, atau IP lain sesuai jaringan).

---

## 4. Topic & payload MQTT

HP **publish**, Jetson **subscribe** (QoS 1, retain false):

| Topic        | Payload valid                                  |
|--------------|------------------------------------------------|
| `MODE`       | `STT`, `STS_AM`, `STS_TF`, `STS_AF`, `STS_TM`  |
| `SENTENCEOK` | `GAS`, `NO`                                     |

Jetson **publish**, HP **subscribe** (QoS 1, retain true):

| Topic      | Payload                              |
|------------|--------------------------------------|
| `STT`      | hasil speech-to-text                 |
| `CAMPOS`   | `OK`, `UP`, `DOWN`, `LEFT`, `RIGHT`  |
| `WORD`     | kata/gesture terbaru                 |
| `SENTENCE` | buffer kalimat berjalan              |

Mode → voice: `STS_AM`=cowok dewasa, `STS_AF`=cewek dewasa, `STS_TM`=cowok remaja,
`STS_TF`=cewek remaja, `STT`=default (cewek dewasa untuk GAS).

Detail alur (GAS-gated, client ID, dll) ada di [`mqtt/README.md`](mqtt/README.md).

---

## 5. Menjalankan

### 5.1 GUI integrasi (kalibrasi posisi, profile, Sherpa, WER)

```bash
PYTHONPATH=src:LLM:. ./env_bisindo_cuda126/bin/python -m src_integrasi.app
```

Tab:
- **BISINDO**: Start live (kamera + MediaPipe + GRU).
- **Profile**: pilih voice TTS.
- **Posisi**: kalibrasi posisi tubuh acuan (dari bahu) → `CAMPOS`. Tekan Start dulu,
  berdiri di posisi pas, klik **Kalibrasi**.
- **Sherpa / WER**: STT via UDP & evaluasi WER.

### 5.2 Runtime MQTT headless (dikendalikan HP)

```bash
PYTHONPATH=src:LLM:. ./env_bisindo_cuda126/bin/python -m src_integrasi.mqtt_runtime
```

Tombol `c` di terminal = kalibrasi posisi kamera (kalau dijalankan interaktif).

---

## 6. Verifikasi end-to-end MQTT (simulasi HP dari Jetson)

Terminal A — lihat status yang dipublish Python:

```bash
mosquitto_sub -h localhost -v -t WORD -t SENTENCE -t CAMPOS -t STT
```

Terminal B — kirim command seolah dari HP:

```bash
mosquitto_pub -h localhost -t MODE -m STS_TM        # ganti voice
mosquitto_pub -h localhost -t SENTENCEOK -m GAS      # suarakan + clear buffer
mosquitto_pub -h localhost -t SENTENCEOK -m NO       # clear tanpa suara
```

Cek cepat broker tanpa menjalankan runtime:

```bash
mosquitto_sub -h localhost -t demo -C 1 &
mosquitto_pub -h localhost -t demo -m halo
```

---

## 7. Troubleshooting

- **`ModuleNotFoundError: paho` / `sherpa_onnx`** → lupa pakai interpreter venv.
  Gunakan `./env_bisindo_cuda126/bin/python`, bukan `python3` sistem.
- **HP tidak bisa connect** → pastikan `ss -tln | grep 1883` menunjukkan `0.0.0.0:1883`
  (bukan `127.0.0.1`), HP & Jetson di hotspot sama, dan client ID HP **beda** dari
  `jetson-bisindo-py`.
- **`File model Sherpa tidak lengkap`** → cek folder model di bagian 2 lengkap (4 file).
- **onnxruntime warning GPU discovery failed** → aman diabaikan, jatuh ke CPU.
- **Mosquitto tidak jalan** → `sudo systemctl status mosquitto` dan
  `sudo tail -n 50 /var/log/mosquitto/mosquitto.log`.
