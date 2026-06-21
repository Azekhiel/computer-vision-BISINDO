# Isi Konfigurasi Integrasi (`configuration.json`)

File `src_integrasi/configuration.json` mengatur **schema, model, suite, augmentasi, specialist,
threshold, dan llm** yang dipakai integrasi (GUI `app.py` maupun runtime `mqtt_runtime.py`) — tanpa
perlu mengubah kode.

Contoh isi default:

```json
{
  "schema": "smart180",
  "model": "adi",
  "suite": "main",
  "augmentasi": true,
  "specialist": "all",
  "threshold": "default",
  "llm": "default"
}
```

Cara kerja:
- `model` + `augmentasi` digabung jadi nama varian: `augmentasi: true` → `adi_dengan_augmentasi`,
  `augmentasi: false` → `adi`.
- CLI args di `mqtt_runtime.py` **hanya override kalau diisi**. `--camera`/`--device`/`--tts-*` selalu
  ada nilainya, tapi `--threshold` & `--specialist` default kosong → kalau tak disetel, isi file yang
  dipakai. (Di GUI `app.py` semua selalu dari file.)
- Kalau file hilang / rusak / field kosong → fallback ke default AssistantConfig, tidak crash.
- Field bernilai `"default"` (threshold/llm) atau kosong → dilewati (pakai default).
- Nilai tidak valid → error jelas dengan daftar pilihan yang benar.
- Kombinasi hanya jalan kalau **checkpoint-nya ada** di `models/.../<schema>/`. Kalau tidak ada,
  integrasi gagal load model (cek folder model dulu).

---

## 1. `schema` — skema fitur (9 pilihan)

Sumber: `src/feature_schemas.py`. Default: `smart180`.

| Nilai | Dimensi | Keterangan |
|---|---|---|
| `smart180` | 180-D | Smart V8: tangan + bahu, **tanpa wajah** (default) |
| `khukuh1629` | 1629-D | Holistic Khukuh, **termasuk wajah** |
| `adi1662` | 1662-D | Holistic Adi, **termasuk wajah** |
| `smart268` | 268-D | Smart full-hand: 21 titik tangan global+lokal + bahu |
| `smart180_face1584` | 1584-D | Smart180 + fitur wajah 1404-D |
| `smart180_mouthdyn214` | 214-D | Smart180 + mouth dynamics |
| `smart180_mouthstat206` | 206-D | Smart180 + mouth static |
| `smart180_handface220` | 220-D | Smart180 + relasi tangan-wajah |
| `smart180_handface_vel286` | 286-D | Smart180 + relasi tangan-wajah + velocity |

Alias grup (kalau dipakai sebagai schema tunggal akan dinormalisasi): `default` → `smart180`.

---

## 2. `model` — varian model (7 pilihan base)

Sumber: `src/gru_manager.py`. Default: `adi`. Augmentasi diatur lewat field `augmentasi`, **jangan**
tulis suffix `_dengan_augmentasi` di sini.

| Nilai | Arsitektur | Target frames |
|---|---|---|
| `khukuh` | GRU Khukuh | 30 |
| `adi` | GRU Adi (default) | 60 |
| `hybrid` | GRU Hybrid | 30 |
| `biattn` | BiGRU + Attention | 60 |
| `convfront` | Conv1d front + GRU | 60 |
| `tcn` | Temporal ConvNet (TCN) | 60 |
| `transformer` | Mini Transformer | 60 |

Alias: `bigru_attn` → `biattn`, `conv` → `convfront`, `xformer` → `transformer`.

---

## 3. `suite` — route inference (7 pilihan)

Sumber: `src/gru_manager.py` (`EVAL_SUITE_NAMES`). Default: `main`.

| Nilai | Keterangan |
|---|---|
| `main` | Main GRU saja, tanpa expert (default) |
| `chunk10` | Expert chunk10 |
| `threshold` | Expert threshold |
| `main_chunk10` | Main + chunk10 |
| `main_threshold` | Main + threshold |
| `vote_all` | Voting semua expert |
| `boosted_stack` | Boosted stack |

Alias: `utama` / `main_gru` → `main`, `boosted` → `boosted_stack`.

---

## 4. `augmentasi` — pakai data augmentasi atau tidak (boolean)

| Nilai | Efek |
|---|---|
| `true` | Varian `_dengan_augmentasi` (mis. `adi_dengan_augmentasi`) — **default** |
| `false` | Varian base (mis. `adi`) |

---

## 5. `specialist` — model spesialis untuk huruf/vocab yang sering ketuker

Specialist = model kecil khusus beberapa vocab (mis. `c_l` untuk c vs l). Dinamai dari folder di
`models/gru/<schema>/specialists/`.

| Nilai | Efek |
|---|---|
| `all` | Muat **semua** specialist yang punya checkpoint untuk schema+model aktif — **default** |
| `c_l` | Muat **satu** specialist saja |
| `c_l, m_masalah` | Muat **beberapa** specialist (dipisah koma) |
| `off` / `false` / `none` | **Matikan** specialist (live jalan tanpa specialist) |

Cek specialist yang tersedia:
```bash
ls models/gru/<schema>/specialists/
```
Specialist yang diminta tapi checkpoint-nya tak ada → dilewati dengan `warning`, tidak crash.

---

## 6. `threshold` — ambang kepercayaan prediksi

| Nilai | Efek |
|---|---|
| `default` | Pakai default `0.65` |
| angka 0–1 (mis. `0.7`) | Ambang sendiri; makin tinggi makin ketat (lebih sedikit prediksi salah, tapi bisa lebih sering "tak yakin") |

Di luar rentang `(0, 1]` → error.

---

## 7. `llm` — model Ollama untuk merangkai kalimat

| Nilai | Efek |
|---|---|
| `default` | Pakai `bisindo-sailor2` |
| nama model Ollama (mis. `bisindo-sailor2`) | Model lain yang sudah ada di `ollama list` |

Model harus sudah ter-build/ter-pull (`ollama list`). Kalau belum ada, runtime cuma memperingatkan
saat warmup, tidak crash. Build contoh: `ollama create bisindo-sailor2 -f LLM/Modelfile.sailor2`.

---

## Contoh kombinasi

```json
{ "schema": "smart180", "model": "biattn", "suite": "chunk10", "augmentasi": false }
```
→ live_schema=`smart180`, live_variant=`biattn`, live_route=`chunk10`.

```json
{ "schema": "adi1662", "model": "adi", "suite": "boosted", "augmentasi": true }
```
→ live_schema=`adi1662`, live_variant=`adi_dengan_augmentasi`, live_route=`boosted_stack`.

```json
{
  "schema": "smart180", "model": "adi", "suite": "main", "augmentasi": true,
  "specialist": "c_l, m_masalah", "threshold": 0.7, "llm": "bisindo-sailor2"
}
```
→ muat 2 specialist (`c_l`+`m_masalah`), confidence_threshold=`0.7`, llm_model=`bisindo-sailor2`.
