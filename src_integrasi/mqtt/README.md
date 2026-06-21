# MQTT BISINDO (HP Android ↔ Jetson)

Arsitektur:

- **HP Android (MqttDroid)** = UI kontrol + display. Connect ke **IP Jetson : 1883**.
- **Jetson** menjalankan:
  1. **Mosquitto** sebagai broker (listen `0.0.0.0:1883`).
  2. **`src_integrasi/mqtt_runtime.py`** sebagai MQTT client (connect `localhost:1883`).

Pakai hotspot yang sama untuk HP dan Jetson.

## 1. Install broker (sekali saja)

```bash
sudo apt update
sudo apt install -y mosquitto mosquitto-clients
```

## 2. Jalankan broker dengan config ini

```bash
# Hentikan service default kalau bentrok di port 1883
sudo systemctl stop mosquitto

# Jalankan manual pakai config repo
mosquitto -c src_integrasi/mqtt/mosquitto.conf -v
```

Cari IP Jetson untuk diisi di MqttDroid (Host):

```bash
hostname -I
```

## 3. Topic

Dari **HP → Jetson** (HP publish, Python subscribe), QoS 1, retain **false**:

| Topic        | Payload valid                                   |
|--------------|-------------------------------------------------|
| `MODE`       | `STT`, `STS_AM`, `STS_TF`, `STS_AF`, `STS_TM`   |
| `SENTENCEOK` | `GAS`, `NO`                                     |

Dari **Jetson → HP** (Python publish, HP subscribe), QoS 1, retain **true**:

| Topic      | Payload                                  |
|------------|------------------------------------------|
| `STT`      | hasil speech-to-text                     |
| `CAMPOS`   | `OK`, `UP`, `DOWN`, `LEFT`, `RIGHT`      |
| `WORD`     | kata/gesture terbaru yang dikenali       |
| `SENTENCE` | buffer kalimat berjalan (belum disuarakan)|

### Mode → voice TTS

| MODE     | Voice                  |
|----------|------------------------|
| `STT`    | speech-to-text (default voice cewek dewasa untuk GAS) |
| `STS_AM` | cowok dewasa           |
| `STS_AF` | cewek dewasa           |
| `STS_TM` | cowok remaja           |
| `STS_TF` | cewek remaja           |

### Alur sentence (GAS-gated)

- Tiap gesture dikenali → `WORD` dipublish, buffer di-update → `SENTENCE` dipublish.
- Kalimat **tidak disuarakan** sampai HP kirim `SENTENCEOK=GAS`.
- `GAS` → buffer dirapikan LLM (Sailor2) → disuarakan → buffer di-clear.
- `NO`  → buffer di-clear tanpa disuarakan.

## 4. Client ID

- Python Jetson: `jetson-bisindo-py` (sudah diset di `mqtt_bridge.py`).
- HP MqttDroid: pakai client ID **berbeda** (mis. `hp-bisindo`). Jangan sama — MQTT
  akan memutus salah satu koneksi kalau client ID kembar.

## 5. Tes cepat dari Jetson (simulasi HP)

```bash
# Terminal A: lihat status dari Python
mosquitto_sub -h localhost -v -t WORD -t SENTENCE -t CAMPOS -t STT

# Terminal B: kirim command seolah dari HP
mosquitto_pub -h localhost -t MODE -m STS_TM
mosquitto_pub -h localhost -t SENTENCEOK -m GAS
mosquitto_pub -h localhost -t SENTENCEOK -m NO
```

## 6. Setup MqttDroid di HP

- Host: IP Jetson (mis. `192.168.x.x` / `10.42.0.1`), Port `1883`, client ID `hp-bisindo`.
- Buat 5 **Trigger** untuk `MODE` (payload `STT`, `STS_AM`, `STS_TF`, `STS_AF`, `STS_TM`).
- `SENTENCEOK`: 1 Toggle (On=`GAS`, Off=`NO`) atau 2 Trigger (`GAS` / `NO`).
- Subscribe display: `STT`, `CAMPOS`, `WORD`, `SENTENCE`.
