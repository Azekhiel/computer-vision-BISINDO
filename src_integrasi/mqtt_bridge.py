"""MQTT bridge antara Python Jetson dan HP Android (MqttDroid).

Python Jetson = MQTT client yang connect ke broker lokal (Mosquitto di localhost:1883).
- Subscribe command dari HP: MODE, SENTENCEOK, WORDDEL.
- Publish status ke HP: STT, CAMPOS, WORD, SENTENCE.

Lihat src_integrasi/mqtt/README.md untuk arsitektur lengkap.
"""

from __future__ import annotations

import threading
from typing import Callable

try:  # paho-mqtt wajib di-install (lihat requirements.txt)
    import paho.mqtt.client as mqtt
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "paho-mqtt belum terpasang. Install dengan: pip install paho-mqtt"
    ) from exc


# --- Topic (uppercase, sesuai spesifikasi HP) ---
TOPIC_MODE = "MODE"
TOPIC_SENTENCEOK = "SENTENCEOK"
TOPIC_WORDDEL = "WORDDEL"
TOPIC_STT = "STT"
TOPIC_CAMPOS = "CAMPOS"
TOPIC_WORD = "WORD"
TOPIC_SENTENCE = "SENTENCE"

# --- Payload valid ---
VALID_MODES = ("STT", "STS_AM", "STS_TF", "STS_AF", "STS_TM")
VALID_SENTENCEOK = ("GAS", "NO")
VALID_CAMPOS = ("OK", "UP", "DOWN", "LEFT", "RIGHT")

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1883
DEFAULT_CLIENT_ID = "jetson-bisindo-py"
QOS = 1

# Command (HP -> Jetson) retain false; status (Jetson -> HP) retain true.
RETAIN_STATUS = True


def _make_client(client_id: str) -> "mqtt.Client":
    """Buat paho Client kompatibel dengan paho-mqtt v1 maupun v2."""
    callback_api = getattr(mqtt, "CallbackAPIVersion", None)
    if callback_api is not None:  # paho-mqtt >= 2.0
        return mqtt.Client(callback_api.VERSION1, client_id=client_id, clean_session=True)
    return mqtt.Client(client_id=client_id, clean_session=True)


class MqttBridge:
    """Wrapper paho-mqtt: subscribe command, publish status.

    Callback didaftarkan lewat ``on_mode`` dan ``on_sentenceok``. Keduanya
    dipanggil dari thread network paho, jadi callback harus thread-safe / cepat.
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        client_id: str = DEFAULT_CLIENT_ID,
        on_mode: Callable[[str], None] | None = None,
        on_sentenceok: Callable[[str], None] | None = None,
        on_worddel: Callable[[str], None] | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.client_id = str(client_id)
        self.on_mode = on_mode
        self.on_sentenceok = on_sentenceok
        self.on_worddel = on_worddel
        self._log = logger or (lambda msg: print(msg, flush=True))
        self._connected = threading.Event()

        self.client = _make_client(self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    # --- lifecycle ---
    def connect(self, timeout: float = 10.0) -> bool:
        self.client.connect(self.host, self.port, keepalive=30)
        self.client.loop_start()
        ok = self._connected.wait(timeout=timeout)
        if not ok:
            self._log(f"MQTT: gagal connect ke {self.host}:{self.port} dalam {timeout:.0f}s")
        return ok

    def disconnect(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # --- paho callbacks ---
    def _on_connect(self, client, _userdata, _flags, rc, *args) -> None:
        if rc == 0:
            self._connected.set()
            client.subscribe([(TOPIC_MODE, QOS), (TOPIC_SENTENCEOK, QOS), (TOPIC_WORDDEL, QOS)])
            self._log(f"MQTT connected ({self.host}:{self.port}) sebagai {self.client_id}")
        else:
            self._log(f"MQTT connect gagal rc={rc}")

    def _on_disconnect(self, _client, _userdata, rc, *args) -> None:
        self._connected.clear()
        self._log(f"MQTT disconnected rc={rc}")

    def _on_message(self, _client, _userdata, message) -> None:
        topic = message.topic
        payload = message.payload.decode("utf-8", errors="replace").strip()
        if topic == TOPIC_MODE:
            if payload in VALID_MODES:
                self._dispatch(self.on_mode, payload)
            else:
                self._log(f"MQTT MODE payload invalid diabaikan: {payload!r}")
        elif topic == TOPIC_SENTENCEOK:
            up = payload.upper()
            if up in VALID_SENTENCEOK:
                self._dispatch(self.on_sentenceok, up)
            else:
                self._log(f"MQTT SENTENCEOK payload invalid diabaikan: {payload!r}")
        elif topic == TOPIC_WORDDEL:
            # Payload bebas (mis. "DEL"); aksinya selalu sama: hapus kata terakhir di buffer.
            self._dispatch(self.on_worddel, payload or "DEL")

    def _dispatch(self, callback: Callable[[str], None] | None, payload: str) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except Exception as exc:  # jangan biarkan callback error membunuh loop paho
            self._log(f"MQTT callback error ({payload}): {exc}")

    # --- publish helpers (Jetson -> HP), QoS 1, retain true ---
    def _publish(self, topic: str, text: str) -> None:
        try:
            self.client.publish(topic, str(text), qos=QOS, retain=RETAIN_STATUS)
        except Exception as exc:
            self._log(f"MQTT publish {topic} gagal: {exc}")

    def publish_stt(self, text: str) -> None:
        self._publish(TOPIC_STT, text)

    def publish_campos(self, status: str) -> None:
        self._publish(TOPIC_CAMPOS, status)

    def publish_word(self, word: str) -> None:
        self._publish(TOPIC_WORD, word)

    def publish_sentence(self, sentence: str) -> None:
        self._publish(TOPIC_SENTENCE, sentence)
