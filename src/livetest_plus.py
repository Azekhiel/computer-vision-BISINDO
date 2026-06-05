"""Sentence buffering, local LLM, and TTS helpers for LiveTest Plus."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib import request


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_MAX_WORDS = 5
DEFAULT_IDLE_NO_HAND_SEC = 5.0


@dataclass(frozen=True)
class FlushResult:
    words: tuple[str, ...]
    reason: str

    @property
    def text(self) -> str:
        return words_to_text(self.words)


@dataclass(frozen=True)
class AudioSink:
    index: int
    name: str
    driver: str = ""

    @property
    def display(self) -> str:
        return f"{self.index}: {self.name}"


@dataclass(frozen=True)
class SpeechJob:
    text: str
    sink_name: str | None = None


def normalize_word(label: str) -> str:
    return str(label or "").strip()


def words_to_text(words: tuple[str, ...] | list[str]) -> str:
    return " ".join(str(word).replace("_", " ").strip() for word in words if str(word).strip()).strip()


class SentenceBuffer:
    """Collect stable live predictions and flush them as short word batches."""

    def __init__(
        self,
        max_words: int = DEFAULT_MAX_WORDS,
        idle_no_hand_sec: float = DEFAULT_IDLE_NO_HAND_SEC,
        min_confidence: float = 1e-6,
    ) -> None:
        self.max_words = max(1, int(max_words))
        self.idle_no_hand_sec = max(0.0, float(idle_no_hand_sec))
        self.min_confidence = float(min_confidence)
        self.words: list[str] = []
        self.last_prediction_id = 0
        self.no_hand_started_at: float | None = None

    def reset(self) -> None:
        self.words.clear()
        self.last_prediction_id = 0
        self.no_hand_started_at = None

    def pending_words(self) -> tuple[str, ...]:
        return tuple(self.words)

    def pending_text(self) -> str:
        return words_to_text(self.words)

    def flush(self, reason: str) -> FlushResult | None:
        if not self.words:
            self.no_hand_started_at = None
            return None
        result = FlushResult(tuple(self.words), reason=reason)
        self.words.clear()
        self.no_hand_started_at = None
        return result

    def observe_status(
        self,
        *,
        label: str,
        confidence: float,
        prediction_id: int | None,
        visible: bool,
        now: float | None = None,
    ) -> FlushResult | None:
        current_time = time.perf_counter() if now is None else float(now)
        if visible:
            self.no_hand_started_at = None

        if prediction_id is not None:
            try:
                parsed_id = int(prediction_id)
            except (TypeError, ValueError):
                parsed_id = 0
            if parsed_id > 0 and parsed_id <= self.last_prediction_id:
                return self._maybe_idle_flush(visible=visible, now=current_time)
        else:
            parsed_id = 0

        word = normalize_word(label)
        if word and word != "-" and float(confidence) >= self.min_confidence:
            self.words.append(word)
            if parsed_id > 0:
                self.last_prediction_id = parsed_id
            if len(self.words) >= self.max_words:
                return self.flush("max_words")

        return self._maybe_idle_flush(visible=visible, now=current_time)

    def _maybe_idle_flush(self, *, visible: bool, now: float) -> FlushResult | None:
        if visible or not self.words:
            return None
        if self.no_hand_started_at is None:
            self.no_hand_started_at = now
            return None
        if now - self.no_hand_started_at >= self.idle_no_hand_sec:
            return self.flush("idle_no_hand")
        return None


class OllamaSentenceClient:
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 180.0,
    ) -> None:
        self.model = str(model or DEFAULT_OLLAMA_MODEL)
        self.url = str(url or DEFAULT_OLLAMA_URL)
        self.timeout = float(timeout)

    def build_prompt(self, words: tuple[str, ...] | list[str], allow_word_correction: bool = False) -> str:
        word_text = words_to_text(words)
        correction_rule = (
            "Jika ada kata input yang jelas tidak nyambung, boleh ganti dengan kata yang lebih cocok."
            if allow_word_correction
            else "Jangan mengganti kata inti dari input; hanya rapikan urutan, grammar, dan kata penghubung/pendukung seperlunya."
        )
        return (
            "Tugas: susun kata BISINDO menjadi satu kalimat bahasa Indonesia yang natural.\n"
            f"{correction_rule}\n"
            "Jawab hanya kalimat akhirnya, tanpa penjelasan, tanpa bullet, tanpa tanda kutip.\n"
            f"Input kata: {word_text}"
        )

    def compose(self, words: tuple[str, ...] | list[str], allow_word_correction: bool = False) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "prompt": self.build_prompt(words, allow_word_correction=allow_word_correction),
            "options": {
                "temperature": 0.1 if allow_word_correction else 0.0,
                "num_predict": 80,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self.url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        return sanitize_llm_output(str(parsed.get("response", "")))


def sanitize_llm_output(text: str) -> str:
    value = str(text or "").strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines:
        value = lines[0]
    return value.strip().strip('"').strip("'").strip("`").strip()


def parse_pactl_sinks(output: str) -> list[AudioSink]:
    sinks: list[AudioSink] = []
    for line in str(output or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        driver = parts[2] if len(parts) > 2 else ""
        sinks.append(AudioSink(index=index, name=parts[1], driver=driver))
    return sinks


def list_audio_sinks(timeout: float = 3.0) -> list[AudioSink]:
    pactl = shutil.which("pactl")
    if not pactl:
        return []
    try:
        proc = subprocess.run(
            [pactl, "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=float(timeout),
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return parse_pactl_sinks(proc.stdout)


def sink_display_options(sinks: list[AudioSink]) -> tuple[dict[str, str | None], list[str]]:
    display_to_name: dict[str, str | None] = {"auto/default": None}
    options = ["auto/default"]
    for sink in sinks:
        display_to_name[sink.display] = sink.name
        options.append(sink.display)
    return display_to_name, options


class EspeakSpeaker:
    """Small non-blocking espeak worker with optional PulseAudio sink routing."""

    def __init__(self, binary: str | None = None, rate: int = 150) -> None:
        self.binary = binary or shutil.which("espeak") or "/usr/bin/espeak"
        self.rate = int(rate)
        self.queue: queue.Queue[SpeechJob | None] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return bool(self.binary and os.path.exists(self.binary))

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def say(self, text: str, sink_name: str | None = None) -> None:
        clean_text = str(text or "").strip()
        if not clean_text:
            return
        self.start()
        self.queue.put(SpeechJob(clean_text, sink_name=sink_name))

    def stop(self) -> None:
        self.stop_event.set()
        self.queue.put(None)

    def join(self, timeout: float | None = 1.0) -> None:
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def run_once(self, text: str, sink_name: str | None = None) -> subprocess.CompletedProcess | None:
        if not self.available:
            return None
        env = os.environ.copy()
        if sink_name:
            env["PULSE_SINK"] = sink_name
        return subprocess.run(
            [self.binary, "-s", str(self.rate), str(text)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=30,
            check=False,
        )

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            job = self.queue.get()
            if job is None:
                break
            try:
                self.run_once(job.text, sink_name=job.sink_name)
            except Exception:
                pass
