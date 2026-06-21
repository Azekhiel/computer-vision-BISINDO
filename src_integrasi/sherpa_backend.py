"""Lazy Sherpa-ONNX backend used by the lightweight integration GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import socket
import string
import threading
import time
from typing import Any, Callable

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SHERPA_MODEL_DIR = ROOT_DIR / "Sherpa" / "models" / "sherpa-onnx-streaming-zipformer2-id"
UDP_IP = "0.0.0.0"
UDP_PORT = 8080
UDP_BUFFER_SIZE = 1024
UDP_AUDIO_SAMPLE_RATE = 16000
SHERPA_SAMPLE_RATE = 16000
SILENCE_TIMEOUT_SEC = 1.5


class SherpaBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class SherpaModelConfig:
    model_dir: Path = DEFAULT_SHERPA_MODEL_DIR
    num_threads: int = 2
    sample_rate: int = SHERPA_SAMPLE_RATE
    feature_dim: int = 80

    def files(self) -> dict[str, Path]:
        model_dir = Path(self.model_dir)
        return {
            "tokens": model_dir / "tokens.txt",
            "encoder": model_dir / "encoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx",
            "decoder": model_dir / "decoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx",
            "joiner": model_dir / "joiner-iter-100000-avg-15-chunk-32-left-256.int8.onnx",
        }


@dataclass(frozen=True)
class UdpConfig:
    host: str = UDP_IP
    port: int = UDP_PORT
    buffer_size: int = UDP_BUFFER_SIZE
    audio_sample_rate: int = UDP_AUDIO_SAMPLE_RATE
    receive_buffer_bytes: int = 1024 * 1024
    socket_timeout_sec: float = 0.5
    silence_timeout_sec: float = SILENCE_TIMEOUT_SEC


def normalize_text(text: str) -> str:
    value = str(text or "").lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return " ".join(value.split())


def diff_new_words(confirmed: list[str], current: list[str]) -> list[str]:
    if len(current) < len(confirmed):
        return []
    for idx, word in enumerate(confirmed):
        if idx >= len(current) or current[idx] != word:
            return []
    return list(current[len(confirmed) :])


def pcm16le_to_float32(
    data: bytes,
    *,
    input_sample_rate: int = UDP_AUDIO_SAMPLE_RATE,
    target_sample_rate: int = SHERPA_SAMPLE_RATE,
) -> np.ndarray:
    if not data:
        return np.empty((0,), dtype=np.float32)
    if len(data) % 2:
        data = data[:-1]
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if input_sample_rate != target_sample_rate:
        ratio = int(input_sample_rate) // int(target_sample_rate)
        if ratio > 1 and input_sample_rate % target_sample_rate == 0:
            samples = samples[::ratio]
    return samples.astype(np.float32, copy=False)


def select_provider(ort_module: Any) -> str:
    providers = set(ort_module.get_available_providers())
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"


class LazySherpaRecognizer:
    def __init__(self, config: SherpaModelConfig | None = None) -> None:
        self.config = config or SherpaModelConfig()
        self.recognizer: Any | None = None
        self.provider = ""

    @property
    def loaded(self) -> bool:
        return self.recognizer is not None

    def load(self) -> str:
        if self.recognizer is not None:
            return self.provider
        try:
            import onnxruntime as ort
            import sherpa_onnx
        except Exception as exc:
            raise SherpaBackendError(
                "Dependency Sherpa belum tersedia: butuh sherpa_onnx dan onnxruntime."
            ) from exc

        files = self.config.files()
        missing = [str(path) for path in files.values() if not Path(path).exists()]
        if missing:
            raise SherpaBackendError("File model Sherpa tidak lengkap: " + ", ".join(missing))

        provider = select_provider(ort)
        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(files["tokens"]),
                encoder=str(files["encoder"]),
                decoder=str(files["decoder"]),
                joiner=str(files["joiner"]),
                num_threads=int(self.config.num_threads),
                sample_rate=int(self.config.sample_rate),
                feature_dim=int(self.config.feature_dim),
                provider=provider,
            )
        except Exception as exc:
            raise SherpaBackendError(f"Gagal load recognizer Sherpa: {exc}") from exc
        self.provider = provider
        return provider

    def create_stream(self) -> Any:
        self.load()
        return self.recognizer.create_stream()

    def is_ready(self, stream: Any) -> bool:
        return bool(self.recognizer.is_ready(stream))

    def decode_stream(self, stream: Any) -> None:
        self.recognizer.decode_stream(stream)

    def get_result(self, stream: Any) -> str:
        return str(self.recognizer.get_result(stream) or "")

    def is_endpoint(self, stream: Any) -> bool:
        return bool(self.recognizer.is_endpoint(stream))

    def reset(self, stream: Any) -> None:
        self.recognizer.reset(stream)


class UdpSherpaWorker:
    def __init__(
        self,
        *,
        recognizer: LazySherpaRecognizer | None = None,
        udp_config: UdpConfig | None = None,
        event_queue: queue.Queue | None = None,
        socket_factory: Callable[..., socket.socket] = socket.socket,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.recognizer = recognizer or LazySherpaRecognizer()
        self.udp_config = udp_config or UdpConfig()
        self.event_queue = event_queue or queue.Queue()
        self.socket_factory = socket_factory
        self.monotonic = monotonic
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self.packet_count = 0
        self.last_sender = "-"

    def start(self) -> None:
        if self.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sherpa-udp-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _put(self, event: dict[str, Any]) -> None:
        self.event_queue.put(event)

    def _run(self) -> None:
        sock = None
        try:
            provider = self.recognizer.load()
            self._put({"type": "recognizer", "provider": provider, "message": f"Sherpa siap ({provider})"})
            stream = self.recognizer.create_stream()
            sock = self.socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.udp_config.receive_buffer_bytes)
            sock.bind((self.udp_config.host, self.udp_config.port))
            sock.settimeout(float(self.udp_config.socket_timeout_sec))
            self._put(
                {
                    "type": "listening",
                    "host": self.udp_config.host,
                    "port": self.udp_config.port,
                    "message": f"UDP listen {self.udp_config.host}:{self.udp_config.port}",
                }
            )
            self._decode_loop(sock, stream)
        except OSError as exc:
            self._put({"type": "error", "message": f"UDP error {self.udp_config.host}:{self.udp_config.port}: {exc}"})
        except Exception as exc:
            self._put({"type": "error", "message": str(exc)})
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            self._sock = None
            self._put({"type": "stopped", "message": "Sherpa UDP stopped"})

    def _decode_loop(self, sock: socket.socket, stream: Any) -> None:
        confirmed_words: list[str] = []
        last_text = ""
        last_change_at = self.monotonic()
        chunk_total_sec = 0.0
        chunk_count = 0

        while not self._stop_event.is_set():
            try:
                data, addr = sock.recvfrom(int(self.udp_config.buffer_size))
            except socket.timeout:
                data = b""
                addr = None
            except OSError:
                if self._stop_event.is_set():
                    break
                raise

            if data:
                self.packet_count += 1
                self.last_sender = f"{addr[0]}:{addr[1]}" if addr else "-"
                samples = pcm16le_to_float32(
                    data,
                    input_sample_rate=self.udp_config.audio_sample_rate,
                    target_sample_rate=SHERPA_SAMPLE_RATE,
                )
                if samples.size:
                    stream.accept_waveform(SHERPA_SAMPLE_RATE, samples)
                self._put({"type": "packet", "count": self.packet_count, "last_sender": self.last_sender})

            processed = False
            while self.recognizer.is_ready(stream):
                start = time.perf_counter()
                self.recognizer.decode_stream(stream)
                chunk_total_sec += time.perf_counter() - start
                chunk_count += 1
                processed = True
            if processed and chunk_count:
                avg_ms = (chunk_total_sec / chunk_count) * 1000.0
                self._put({"type": "latency", "value_ms": avg_ms})
                if chunk_count > 10:
                    chunk_total_sec = 0.0
                    chunk_count = 0

            text = self.recognizer.get_result(stream)
            current_words = text.split() if text else []
            if text != last_text:
                new_words = diff_new_words(confirmed_words, current_words)
                for word in new_words:
                    confirmed_words.append(word)
                    self._put({"type": "word", "text": word})
                tail = " ".join(current_words[len(confirmed_words) :])
                self._put({"type": "partial", "text": tail})
                last_text = text
                last_change_at = self.monotonic()

            if confirmed_words and (self.monotonic() - last_change_at) > self.udp_config.silence_timeout_sec:
                segment = " ".join(confirmed_words)
                self._put({"type": "segment_end", "text": segment})
                confirmed_words = []
                last_text = ""
                last_change_at = self.monotonic()
                self.recognizer.reset(stream)

            if self.recognizer.is_endpoint(stream):
                current_words = (self.recognizer.get_result(stream) or "").split()
                for word in diff_new_words(confirmed_words, current_words):
                    confirmed_words.append(word)
                    self._put({"type": "word", "text": word})
                if confirmed_words:
                    self._put({"type": "segment_end", "text": " ".join(confirmed_words)})
                confirmed_words = []
                last_text = ""
                last_change_at = self.monotonic()
                self.recognizer.reset(stream)


def calculate_wer(reference: str, hypothesis: str) -> dict[str, float | str]:
    try:
        import jiwer
    except Exception as exc:
        raise SherpaBackendError("Dependency jiwer belum tersedia untuk WER.") from exc
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    return {
        "reference": ref,
        "hypothesis": hyp,
        "wer": float(jiwer.wer(ref, hyp)),
        "mer": float(jiwer.mer(ref, hyp)),
        "wil": float(jiwer.wil(ref, hyp)),
    }


def transcribe_wav(
    wav_path: Path | str,
    *,
    recognizer: LazySherpaRecognizer | None = None,
) -> dict[str, float | str]:
    try:
        import soundfile as sf
    except Exception as exc:
        raise SherpaBackendError("Dependency soundfile belum tersedia untuk evaluasi WAV.") from exc

    recognizer = recognizer or LazySherpaRecognizer()
    audio, sample_rate = sf.read(str(wav_path), dtype="float32")
    if len(getattr(audio, "shape", ())) > 1:
        audio = audio.mean(axis=1)
    if int(sample_rate) != SHERPA_SAMPLE_RATE:
        try:
            import scipy.signal as scipy_signal
        except Exception as exc:
            raise SherpaBackendError(
                f"Audio {sample_rate}Hz butuh scipy untuk resample ke {SHERPA_SAMPLE_RATE}Hz."
            ) from exc
        audio = scipy_signal.resample_poly(audio, SHERPA_SAMPLE_RATE, int(sample_rate)).astype(np.float32)

    stream = recognizer.create_stream()
    stream.accept_waveform(SHERPA_SAMPLE_RATE, audio.astype(np.float32, copy=False))
    stream.accept_waveform(SHERPA_SAMPLE_RATE, np.zeros(int(0.3 * SHERPA_SAMPLE_RATE), dtype=np.float32))

    started = time.perf_counter()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    process_sec = time.perf_counter() - started
    duration_sec = float(len(audio)) / SHERPA_SAMPLE_RATE if len(audio) else 0.0
    return {
        "text": recognizer.get_result(stream),
        "process_sec": process_sec,
        "duration_sec": duration_sec,
        "rtf": process_sec / duration_sec if duration_sec > 0 else 0.0,
    }

