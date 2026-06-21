# Tutorial IoT MQTT Panel (HP) ↔ Jetson BISINDO

Panduan lengkap setup **IoT MQTT Panel** di HP Android sebagai remote untuk runtime BISINDO di Jetson.
Pakai ini sebagai ganti MqttDroid (lebih jelas, ada tombol publish + tile subscribe).

Prasyarat: broker & runtime sudah jalan di Jetson — lihat [`INSTALL.md`](INSTALL.md) dan
[`tutor_mqtt.md`](tutor_mqtt.md). Dokumen ini fokus ke **konfigurasi app di HP**.

> Catatan: nama tombol/label di app bisa beda sedikit antar versi, tapi alurnya (Connection → Add panel
> → isi Topic/Payload) sama.

---

## 0. Konsep IoT MQTT Panel

Tiga lapis:
1. **Connection** — koneksi ke broker (1 koneksi ke Jetson).
2. **Dashboard** — halaman berisi panel (boleh 1 dashboard aja).
3. **Panel** — komponen: tombol kirim (Button), penampil pesan (Text Log), indikator (LED), dll.

Yang kita butuh:
- **7 Button** untuk PUBLISH (HP → Jetson): 5× `MODE`, 2× `SENTENCEOK`.
- **4 penampil** untuk SUBSCRIBE (Jetson → HP): `WORD`, `SENTENCE`, `CAMPOS`, `STT`.

---

## 1. Install
Play Store → **"IoT MQTT Panel"** (oleh Rahul Kundu). Install, buka.

---

## 2. Bikin Connection ke broker Jetson

1. Di layar **Connections**, tap **+** (atau "Add connection").
2. Isi:
   | Field | Nilai |
   |-------|-------|
   | Connection Name | `Jetson BISINDO` (bebas) |
   | Client ID | biarkan auto-generate (yang penting **BUKAN** `jetson-bisindo-py`) |
   | Broker Web/IP Address | **IP Jetson**, mis. `10.42.0.1` (dari `hostname -I` di Jetson) |
   | Port | `1883` |
   | Network protocol | `TCP` |
   | Username / Password | **kosongkan** (broker `allow_anonymous true`) |
   | SSL/TLS | **OFF** |
   | Auto reconnect | **ON** |
   | Persistent / Clean session | Clean session **ON** |
3. Tap **Create / Save**. Status harus jadi **Connected** (hijau). Kalau gagal → lihat Troubleshooting.

> HP **wajib** di WiFi/hotspot yang sama dengan Jetson. Kalau Jetson jadi hotspot, sambungkan HP ke
> WiFi Jetson dulu, IP broker = `10.42.0.1`.

---

## 3. Tambah panel PUBLISH (tombol kirim) — tipe **Button**

Masuk ke connection → **Edit** (ikon pensil) → **+ Add panel** → pilih **Button**. Ulangi untuk tiap
tombol di tabel. Konfig tiap Button:

- **Panel name**: sesuai kolom Label.
- **Topic** (publish topic): sesuai kolom Topic.
- **Payload** (yang dikirim saat ditekan): kolom Payload.
- **QoS**: `1`.
- **Retain**: **OFF**.

| Label tombol | Topic | Payload |
|--------------|-------|---------|
| Mode STT | `MODE` | `STT` |
| Voice Cowok Dewasa | `MODE` | `STS_AM` |
| Voice Cewek Dewasa | `MODE` | `STS_AF` |
| Voice Cowok Remaja | `MODE` | `STS_TM` |
| Voice Cewek Remaja | `MODE` | `STS_TF` |
| ⌫ Hapus Kata | `WORDDEL` | `DEL` |
| ✅ GAS (suarakan) | `SENTENCEOK` | `GAS` |
| ❌ NO (clear) | `SENTENCEOK` | `NO` |

> **Hapus Kata (`WORDDEL`)** = backspace: hapus **kata terakhir** di buffer kalau salah deteksi.
> Payload bebas (`DEL`/`1`/apa pun) — aksinya selalu hapus 1 kata terakhir. Setelah dihapus, Jetson
> kirim ulang **`SENTENCE`** yang sudah terkoreksi (cuma sekali, pas ada perubahan). Beda dari `NO`
> yang menghapus **seluruh** buffer.

> **Payload harus PERSIS** (huruf besar). `MODE` selain 5 nilai itu diabaikan Jetson.
> Kalau panel Button memintamu isi "Payload on/off" (mode toggle), pakai mode **single/normal** dan isi
> hanya satu payload (nilai di tabel). Jangan pakai mode switch yang punya 2 nilai.

---

## 4. Tambah panel SUBSCRIBE (penampil pesan) — tipe **Text Log**

**+ Add panel** → pilih **Text Log** (atau "Text" / "Log Panel"). Buat 4 panel, konfig tiap-nya:

- **Panel name**: sesuai Label.
- **Topic** (subscribe topic): sesuai Topic.
- **QoS**: `1`.

| Label | Topic |
|-------|-------|
| Kata (WORD) | `WORD` |
| Kalimat (SENTENCE) | `SENTENCE` |
| Posisi (CAMPOS) | `CAMPOS` |
| STT | `STT` |

> Topik dari Jetson dikirim **retain=true**, jadi begitu connect, tile langsung nampilin status terakhir.

### (Opsional) CAMPOS lebih visual
Daripada Text Log, `CAMPOS` enak pakai **LED Indicator** atau **Multi State Indicator**:
- Topic `CAMPOS`, lalu petakan teks → warna: `OK`=hijau, `UP`/`DOWN`/`LEFT`/`RIGHT`=kuning/merah.
- Bikin gampang lihat "udah pas atau belum" tanpa baca teks.

---

## 5. Tes & verifikasi

### 5.1 Jalankan runtime di Jetson
```bash
PYTHONPATH=src:LLM:. ./env_bisindo_cuda126/bin/python -m src_integrasi.mqtt_runtime
```

### 5.2 Tes dari HP
- Tekan tombol **Mode Cewek Dewasa** (`MODE`=`STS_AF`) → di terminal Jetson muncul log `MODE: STS_AF`.
- Mulai isyarat → tile **WORD** & **SENTENCE** di HP keisi.
- Tekan **GAS** → Jetson rapikan kalimat pakai LLM lalu **suara keluar di Jetson**, buffer clear.

### 5.3 Tes tile subscribe tanpa isyarat (dari Jetson)
Buat mastiin tile WORD/SENTENCE/CAMPOS nyala, publish manual dari Jetson:
```bash
mosquitto_pub -h localhost -t WORD -m halo
mosquitto_pub -h localhost -t SENTENCE -m "halo apa kabar"
mosquitto_pub -h localhost -t CAMPOS -m OK
```
Tile di HP harus berubah. (Ini cuma tes tampilan; saat runtime jalan, topik ini diisi otomatis.)

---

## 6. Alur pakai harian
1. Jetson: nyalakan hotspot → jalankan `mqtt_runtime`. HP: connect IoT MQTT Panel.
2. HP: pilih **MODE** (voice).
3. Atur posisi badan lihat **CAMPOS** sampai `OK`.
4. Isyarat → pantau **WORD**/**SENTENCE**.
5. Selesai kalimat → **GAS** (suara keluar) atau **NO** (batal/clear).

---

## 7. Troubleshooting

| Gejala | Solusi |
|--------|--------|
| Connection "Disconnected"/gagal | HP belum di WiFi/hotspot Jetson; IP broker salah (cek `hostname -I`); port bukan `1883`; isi username/password padahal harusnya kosong |
| Connect lalu putus-nyambung terus | Client ID **sama** dengan `jetson-bisindo-py` → ganti jadi unik |
| Tombol MODE ditekan tapi Jetson diam | Payload tidak persis (`STS_AM`/`STS_AF`/`STS_TM`/`STS_TF`/`STT`) |
| GAS tapi gak ada suara | Buffer kosong (belum ada WORD) atau voice belum dipilih (kirim MODE dulu); cek TTS di log runtime |
| Tile subscribe kosong terus | Topic salah ketik (huruf besar semua), atau runtime/live belum jalan |
| Suara keluar otomatis tanpa GAS | Itu GUI `app.py`, bukan `mqtt_runtime.py`. Untuk kontrol HP pakai `mqtt_runtime.py` |
| HP gak konek padahal app lain juga gagal | Cek broker dengar di `0.0.0.0`: di Jetson `ss -tln \| grep 1883` harus `0.0.0.0:1883` |

---

## 8. Ringkasan topic (referensi cepat)

**HP → Jetson (publish, QoS 1):**
- `MODE` = `STT` / `STS_AM` / `STS_AF` / `STS_TM` / `STS_TF`
- `SENTENCEOK` = `GAS` / `NO`
- `WORDDEL` = `DEL` (hapus kata terakhir / backspace)

**Jetson → HP (subscribe, QoS 1, retain):**
- `WORD`, `SENTENCE`, `CAMPOS` (`OK`/`UP`/`DOWN`/`LEFT`/`RIGHT`), `STT`
