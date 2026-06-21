# Tutorial MQTT BISINDO — HP (MqttDroid) ↔ Jetson

Panduan lengkap menjalankan kontrol MQTT: **HP Android sebagai remote**, **Jetson sebagai broker +
runtime BISINDO**. Sudah diverifikasi end-to-end (broker + jembatan MQTT) di Jetson ini.

Untuk instalasi dependency & broker, lihat [`INSTALL.md`](INSTALL.md) dulu. Dokumen ini fokus ke
**pemakaian**.

---

## 1. Gambaran sistem

```
        ┌──────────────── HP Android (MqttDroid) ────────────────┐
        │  PUBLISH:  MODE, SENTENCEOK                             │
        │  SUBSCRIBE: WORD, SENTENCE, CAMPOS, STT                 │
        └───────────────▲───────────────────────┬────────────────┘
                        │ (Wi-Fi / hotspot, 1883)│
        ┌───────────────┴───────────────────────▼────────────────┐
        │  Jetson                                                 │
        │  • Mosquitto broker (port 1883, 0.0.0.0)                │
        │  • mqtt_runtime.py  (kamera + MediaPipe + GRU + TTS)    │
        │     PUBLISH:  WORD, SENTENCE, CAMPOS, STT               │
        │     SUBSCRIBE: MODE, SENTENCEOK                         │
        └─────────────────────────────────────────────────────────┘
```

**Arah pesan:**

| Arah | Topic | Payload | Arti |
|------|-------|---------|------|
| HP → Jetson | `MODE` | `STT`, `STS_AM`, `STS_AF`, `STS_TM`, `STS_TF` | Ganti mode + suara TTS |
| HP → Jetson | `SENTENCEOK` | `GAS` / `NO` | `GAS`=rapikan buffer via LLM + **suarakan**, lalu clear. `NO`=clear tanpa suara |
| HP → Jetson | `WORDDEL` | `DEL` (bebas) | Hapus **kata terakhir** (backspace), kirim ulang `SENTENCE` terkoreksi |
| Jetson → HP | `WORD` | kata terbaru | tiap gesture dikenali |
| Jetson → HP | `SENTENCE` | kalimat berjalan | isi buffer sekarang |
| Jetson → HP | `CAMPOS` | `OK`/`UP`/`DOWN`/`LEFT`/`RIGHT` | panduan posisi badan ke kamera |
| Jetson → HP | `STT` | teks | hasil speech-to-text (kalau dipakai) |

Mode → voice: `STS_AM`=cowok dewasa, `STS_AF`=cewek dewasa, `STS_TM`=cowok remaja,
`STS_TF`=cewek remaja, `STT`=default (cewek dewasa untuk GAS).

> **Penting — GAS-gated.** Runtime MQTT (`mqtt_runtime.py`) **TIDAK** bersuara otomatis. Suara hanya
> keluar saat HP kirim `SENTENCEOK=GAS`. (Beda dari GUI `app.py` yang auto-suara tiap diam 3 detik.)

---

## 2. Prasyarat

- HP dan Jetson di **jaringan yang sama** (hotspot Jetson, atau Wi-Fi yang sama).
- Broker Mosquitto **aktif** di Jetson (lihat INSTALL.md bagian 3).
- App **MqttDroid** di HP: <https://github.com/LightJockey/MqttDroid> (rilis APK di tab Releases).

---

## 3. Sisi Jetson

### 3.1 Pastikan broker jalan
```bash
systemctl is-active mosquitto          # -> active
ss -tln | grep 1883                    # -> 0.0.0.0:1883 LISTEN  (BUKAN 127.0.0.1)
```
Kalau belum `0.0.0.0`, lihat INSTALL.md bagian 3.2 (salin `mqtt/mosquitto.conf`).

### 3.2 Cari IP Jetson (buat diisi di HP)
```bash
hostname -I
```
- Jetson **jadi hotspot** → biasanya `10.42.0.1`.
- Jetson **nyambung ke Wi-Fi** → IP jaringan itu (mis. `10.8.107.214`). Pakai IP yang **satu subnet**
  dengan HP.

### 3.3 (Opsional) atur konfigurasi
Edit [`configuration.json`](configuration.json) untuk schema/model/suite/augmentasi/specialist/
threshold/llm (lihat [`isi_konfiguration.md`](isi_konfiguration.md)). CLI `--threshold`/`--specialist`
override file kalau diisi.

### 3.4 Jalankan runtime MQTT
```bash
PYTHONPATH=src:LLM:. ./env_bisindo_cuda126/bin/python -m src_integrasi.mqtt_runtime
```
Opsi berguna:
```bash
# broker di host lain / port lain:
... -m src_integrasi.mqtt_runtime --mqtt-host 10.42.0.1 --mqtt-port 1883
# matikan specialist / set device:
... --no-specialist --device cuda
```
Saat start, runtime connect ke broker sebagai client id **`jetson-bisindo-py`** dan reset display HP
(`SENTENCE=""`). Tombol `c` di terminal = kalibrasi posisi kamera acuan (CAMPOS).

---

## 4. Sisi HP (MqttDroid)

### 4.1 Koneksi broker
Di MqttDroid → **Settings / Broker**:
- **Address / Host**: IP Jetson dari langkah 3.2 (mis. `10.42.0.1`).
- **Port**: `1883`.
- **Username / Password**: kosong (broker `allow_anonymous true`).
- **Client ID**: **unik, JANGAN `jetson-bisindo-py`** (mis. `hp-bisindo`). Kalau sama, salah satu
  ke-disconnect terus.
- **Clean session**: on. **Auto reconnect**: on (kalau ada).

### 4.2 Tombol PUBLISH (HP → Jetson), QoS 1
Bikin tile/tombol publish, **payload persis** (huruf besar):

| Label | Topic | Payload |
|-------|-------|---------|
| Mode STT | `MODE` | `STT` |
| Voice cowok dewasa | `MODE` | `STS_AM` |
| Voice cewek dewasa | `MODE` | `STS_AF` |
| Voice cowok remaja | `MODE` | `STS_TM` |
| Voice cewek remaja | `MODE` | `STS_TF` |
| **Hapus kata (backspace)** | `WORDDEL` | `DEL` |
| **GAS (suarakan)** | `SENTENCEOK` | `GAS` |
| **NO (clear)** | `SENTENCEOK` | `NO` |

> `MODE` harus persis salah satu di atas (case-sensitive) — kalau ngaco, diabaikan. `SENTENCEOK`
> toleran huruf kecil (`gas`/`no` otomatis di-upper), tapi biasakan kirim `GAS`/`NO`.

### 4.3 Tile SUBSCRIBE (Jetson → HP), QoS 1
Bikin tile yang menampilkan teks dari topic ini:

| Tampilan | Topic |
|----------|-------|
| Kata terbaru | `WORD` |
| Kalimat berjalan | `SENTENCE` |
| Panduan posisi | `CAMPOS` |
| Hasil STT | `STT` |

Topic Jetson→HP di-publish **retain=true**, jadi HP yang baru connect langsung dapat status terakhir.

---

## 5. Alur pakai nyata

1. **Start** `mqtt_runtime.py` di Jetson; connect MqttDroid di HP.
2. HP kirim **`MODE`** (mis. `STS_AF`) → Jetson ganti voice (TTS reload di background).
3. Atur posisi: lihat tile **`CAMPOS`** di HP (`UP`/`DOWN`/`LEFT`/`RIGHT`) sampai **`OK`**.
4. Mulai **isyarat**. Tiap gesture dikenali → **`WORD`** muncul, **`SENTENCE`** terupdate di HP.
5. Selesai satu kalimat → tekan **`GAS`** → Jetson rapikan buffer pakai LLM lalu **suara keluar di
   Jetson**, buffer di-clear. Salah/mau batal → tekan **`NO`**.

---

## 6. Tes end-to-end tanpa HP (simulasi dari Jetson)

Cara cepat memastikan runtime + broker jalan, sebelum nyobain HP. Buka 2 terminal.

**Terminal A** — lihat yang dipublish runtime (Jetson → HP):
```bash
mosquitto_sub -h localhost -v -t WORD -t SENTENCE -t CAMPOS -t STT
```
**Terminal B** — kirim command seolah dari HP:
```bash
mosquitto_pub -h localhost -t MODE -m STS_TM      # ganti voice
mosquitto_pub -h localhost -t SENTENCEOK -m GAS    # suarakan + clear buffer
mosquitto_pub -h localhost -t SENTENCEOK -m NO     # clear tanpa suara
```
Cek broker mentah (tanpa runtime):
```bash
mosquitto_sub -h localhost -t demo -C 1 &
mosquitto_pub -h localhost -t demo -m halo
```

> Status verifikasi di Jetson ini: broker round-trip ✓, callback `MODE`→diterima ✓,
> `SENTENCEOK` `gas`→dinormalkan `GAS` ✓, publish `WORD`/`SENTENCE`/`CAMPOS` ✓.

---

## 7. Troubleshooting

| Gejala | Sebab / solusi |
|--------|----------------|
| HP tidak connect | `ss -tln \| grep 1883` harus `0.0.0.0:1883` (bukan `127.0.0.1`); HP & Jetson satu hotspot; IP benar |
| Connect lalu putus terus | **Client ID HP sama** dengan `jetson-bisindo-py` → ganti jadi unik (mis. `hp-bisindo`) |
| Kirim MODE tapi diam | Payload harus persis (`STS_AM`/`STS_AF`/`STS_TM`/`STS_TF`/`STT`); selain itu diabaikan |
| GAS tapi tak ada suara | Buffer kosong (belum ada WORD), atau TTS belum loaded — cek log runtime; pastikan voice (MODE) sudah dipilih |
| Suara keluar otomatis tanpa GAS | Kamu menjalankan **GUI `app.py`**, bukan `mqtt_runtime.py`. Untuk kontrol HP, pakai `mqtt_runtime.py` |
| `WORD`/`SENTENCE` tidak update di HP | Tile subscribe topic salah ketik; atau live belum jalan (kamera/model) — cek log runtime |
| `ModuleNotFoundError: paho` | Lupa pakai interpreter venv `./env_bisindo_cuda126/bin/python` |
| Broker mati | `sudo systemctl status mosquitto`; `sudo tail -n 50 /var/log/mosquitto/mosquitto.log` |

> Catatan paho-mqtt v2: kode bridge (`mqtt_bridge.py`) sudah kompatibel v1 & v2 (`_make_client`),
> jadi paho 2.x aman.
