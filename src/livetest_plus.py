"""Sentence buffering, local LLM, and TTS helpers for LiveTest Plus."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "bisindo-gemma1b"
DEFAULT_MAX_WORDS = 8
DEFAULT_IDLE_NO_HAND_SEC = 3.0
DEFAULT_OLLAMA_KEEP_ALIVE = "10m"
LLM_WARMUP_TOKENS = ("makan", "aku", "suka")
ROOT_DIR = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT_DIR / "LLM"


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
        return _load_bisindo_llm().build_prompt(words, allow_word_correction=allow_word_correction)

    def build_system_prompt(self) -> str:
        return "System prompt ada di LLM/Modelfile.gemma1b."

    def compose(self, words: tuple[str, ...] | list[str], allow_word_correction: bool = False) -> str:
        raw = _load_bisindo_llm().gloss_to_sentence(
            words,
            model=self.model,
            url=self.url,
            timeout=self.timeout,
            keep_alive=DEFAULT_OLLAMA_KEEP_ALIVE,
            allow_word_correction=allow_word_correction,
        )
        return guarded_llm_output(
            raw,
            words,
            allow_word_correction=allow_word_correction,
            require_core_words=not _is_bisindo_sentence_model(self.model),
        )


def _load_bisindo_llm():
    if str(LLM_DIR) not in sys.path:
        sys.path.insert(0, str(LLM_DIR))
    import bisindo_llm

    return bisindo_llm


def _is_bisindo_sentence_model(model: str) -> bool:
    name = str(model or "").strip().lower()
    return name in {"bisindo-gemma1b", "bisindo-gemma1b:latest"}


def warmup_sentence_llm(
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 30.0,
) -> str:
    return OllamaSentenceClient(model=model, url=url, timeout=timeout).compose(
        list(LLM_WARMUP_TOKENS),
        allow_word_correction=False,
    )


def sanitize_llm_output(text: str) -> str:
    value = str(text or "").strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines:
        value = lines[0]
    return value.strip().strip('"').strip("'").strip("`").strip()


def guarded_llm_output(
    text: str,
    words: tuple[str, ...] | list[str],
    allow_word_correction: bool = False,
    require_core_words: bool = True,
) -> str:
    fallback = words_to_text(words)
    value = sanitize_llm_output(text)
    if not value:
        return fallback
    if _looks_like_meta_output(value) or _has_repeated_phrase(value):
        return fallback
    if require_core_words and not allow_word_correction and _missing_core_words(value, words):
        return fallback
    return value


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text or "").replace("_", " ").lower())


def _looks_like_meta_output(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if "bisindo" in lowered or "buffer" in lowered:
        return True
    return bool(re.search(r"(^|\b)(input|output|final|kata|kalimat)\s*:", lowered))


def _has_repeated_phrase(text: str) -> bool:
    tokens = _word_tokens(text)
    if len(tokens) < 4 or len(tokens) % 2 != 0:
        return False
    midpoint = len(tokens) // 2
    return tokens[:midpoint] == tokens[midpoint:]


def _missing_core_words(text: str, words: tuple[str, ...] | list[str]) -> bool:
    output_tokens = set(_word_tokens(text))
    input_tokens = set(_word_tokens(words_to_text(words)))
    return bool(input_tokens and not input_tokens.issubset(output_tokens))


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
