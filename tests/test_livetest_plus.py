import os
import sys
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
LLM_DIR = os.path.join(ROOT_DIR, "LLM")
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

import bisindo_llm
import livetest_plus as lp


def test_sentence_buffer_flushes_at_five_words():
    buffer = lp.SentenceBuffer(max_words=5, idle_no_hand_sec=5.0)

    result = None
    for idx, word in enumerate(["saya", "mau", "pergi", "ke", "sekolah"], start=1):
        result = buffer.observe_status(label=word, confidence=0.9, prediction_id=idx, visible=True, now=float(idx))

    assert result is not None
    assert result.reason == "max_words"
    assert result.words == ("saya", "mau", "pergi", "ke", "sekolah")
    assert buffer.pending_words() == ()


def test_sentence_buffer_flushes_after_idle_no_hand():
    buffer = lp.SentenceBuffer(max_words=5, idle_no_hand_sec=5.0)

    assert buffer.observe_status(label="saya", confidence=0.9, prediction_id=1, visible=True, now=0.0) is None
    assert buffer.observe_status(label="-", confidence=0.0, prediction_id=1, visible=False, now=1.0) is None
    result = buffer.observe_status(label="-", confidence=0.0, prediction_id=1, visible=False, now=6.1)

    assert result is not None
    assert result.reason == "idle_no_hand"
    assert result.words == ("saya",)


def test_sentence_buffer_ignores_duplicate_prediction_id():
    buffer = lp.SentenceBuffer(max_words=5, idle_no_hand_sec=5.0)

    buffer.observe_status(label="saya", confidence=0.9, prediction_id=7, visible=True, now=0.0)
    buffer.observe_status(label="saya", confidence=0.9, prediction_id=7, visible=True, now=0.1)

    assert buffer.pending_words() == ("saya",)


def test_sentence_buffer_hand_visible_resets_idle_timer():
    buffer = lp.SentenceBuffer(max_words=5, idle_no_hand_sec=5.0)

    buffer.observe_status(label="saya", confidence=0.9, prediction_id=1, visible=True, now=0.0)
    buffer.observe_status(label="-", confidence=0.0, prediction_id=1, visible=False, now=1.0)
    buffer.observe_status(label="-", confidence=0.0, prediction_id=1, visible=True, now=4.0)
    assert buffer.observe_status(label="-", confidence=0.0, prediction_id=1, visible=False, now=7.0) is None

    result = buffer.observe_status(label="-", confidence=0.0, prediction_id=1, visible=False, now=12.1)
    assert result is not None
    assert result.words == ("saya",)


def test_ollama_prompt_modes():
    client = lp.OllamaSentenceClient(model="bisindo-gemma1b")

    strict = client.build_prompt(["SAYA", "MAKAN", "RUMAH"], allow_word_correction=False)
    corrective = client.build_prompt(["SAYA", "MAKAN", "RUMAH"], allow_word_correction=True)

    assert "Mode: pertahankan token inti" in strict
    assert "Mode: boleh memperbaiki token" in corrective
    assert "Token BISINDO: SAYA MAKAN RUMAH" in strict
    assert "System prompt ada di LLM/Modelfile.gemma1b" in client.build_system_prompt()


def test_ollama_compose_sanitizes_mocked_response(monkeypatch):
    calls = {}

    class FakeBisindoLLM:
        @staticmethod
        def gloss_to_sentence(words, **kwargs):
            calls["words"] = words
            calls["kwargs"] = kwargs
            return '"Saya makan di rumah."\nPenjelasan tidak dipakai.'

    monkeypatch.setattr(lp, "_load_bisindo_llm", lambda: FakeBisindoLLM)

    client = lp.OllamaSentenceClient(model="bisindo-gemma1b", url="http://ollama", timeout=1.0)
    assert client.compose(["SAYA", "MAKAN", "RUMAH"]) == "Saya makan di rumah."
    assert calls["words"] == ["SAYA", "MAKAN", "RUMAH"]
    assert calls["kwargs"]["model"] == "bisindo-gemma1b"
    assert calls["kwargs"]["url"] == "http://ollama"
    assert calls["kwargs"]["timeout"] == 1.0
    assert calls["kwargs"]["keep_alive"] == "10m"
    assert calls["kwargs"]["allow_word_correction"] is False


def test_ollama_compose_keeps_bisindo_naturalization(monkeypatch):
    class FakeBisindoLLM:
        @staticmethod
        def gloss_to_sentence(words, **_kwargs):
            assert words == ["makan", "aku", "suka"]
            return "Saya suka makan."

    monkeypatch.setattr(lp, "_load_bisindo_llm", lambda: FakeBisindoLLM)

    client = lp.OllamaSentenceClient(model="bisindo-gemma1b", timeout=1.0)

    assert client.compose(["makan", "aku", "suka"], allow_word_correction=False) == "Saya suka makan."


def test_bisindo_llm_clean_output_and_parameter_override(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            calls["raised"] = True

        def json(self):
            return {"response": "Kalimat Indonesia: Saya suka makan.\nCatatan tidak dipakai."}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(bisindo_llm.requests, "post", fake_post)

    output = bisindo_llm.gloss_to_sentence(
        ["makan", "aku", "suka"],
        model="custom-bisindo",
        url="http://local/api/generate",
        timeout=12,
        keep_alive="2m",
        allow_word_correction=True,
    )

    assert output == "Saya suka makan."
    assert calls["url"] == "http://local/api/generate"
    assert calls["json"]["model"] == "custom-bisindo"
    assert calls["json"]["keep_alive"] == "2m"
    assert "Mode: boleh memperbaiki token" in calls["json"]["prompt"]
    assert "Token BISINDO: makan aku suka" in calls["json"]["prompt"]
    assert calls["timeout"] == 12
    assert calls["raised"] is True


def test_bisindo_llm_default_keep_alive(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "Saya suka makan."}

    def fake_post(url, json, timeout):
        calls["json"] = json
        return FakeResponse()

    monkeypatch.setattr(bisindo_llm.requests, "post", fake_post)

    assert bisindo_llm.gloss_to_sentence(["makan", "aku", "suka"]) == "Saya suka makan."
    assert calls["json"]["keep_alive"] == "10m"


def test_guarded_llm_output_falls_back_for_bad_output():
    words = ["SAYA", "MAU", "MINUM"]

    assert lp.guarded_llm_output("", words) == "SAYA MAU MINUM"
    assert lp.guarded_llm_output("Final: Saya mau minum.", words) == "SAYA MAU MINUM"
    assert lp.guarded_llm_output("Terima kasih Terima kasih", ["TERIMA", "KASIH"]) == "TERIMA KASIH"
    assert lp.guarded_llm_output("Saya minum.", words, allow_word_correction=False) == "SAYA MAU MINUM"


def test_guarded_llm_output_keeps_valid_and_allows_word_fix():
    words = ["SAYA", "MAKAN", "RUMAH"]

    assert lp.guarded_llm_output("Saya makan di rumah.", words) == "Saya makan di rumah."
    assert lp.guarded_llm_output("Saya minum di rumah.", words, allow_word_correction=True) == "Saya minum di rumah."
    assert lp.guarded_llm_output("Saya suka makan.", ["makan", "aku", "suka"], require_core_words=False) == "Saya suka makan."


def test_parse_pactl_sinks_and_display_options():
    sinks = lp.parse_pactl_sinks(
        "0\talsa_output.hdmi\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
        "1\talsa_output.analog\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
    )

    display_to_name, options = lp.sink_display_options(sinks)

    assert [sink.name for sink in sinks] == ["alsa_output.hdmi", "alsa_output.analog"]
    assert options == ["auto/default", "0: alsa_output.hdmi", "1: alsa_output.analog"]
    assert display_to_name["auto/default"] is None
    assert display_to_name["1: alsa_output.analog"] == "alsa_output.analog"


def test_espeak_speaker_run_once_uses_selected_sink(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return None

    monkeypatch.setattr(lp.os.path, "exists", lambda path: Path(path).name == "espeak")
    monkeypatch.setattr(lp.subprocess, "run", fake_run)

    speaker = lp.EspeakSpeaker(binary="/usr/bin/espeak", rate=155)
    speaker.run_once("Saya makan.", sink_name="alsa_output.analog")

    assert calls[0][0] == ["/usr/bin/espeak", "-s", "155", "Saya makan."]
    assert calls[0][1]["env"]["PULSE_SINK"] == "alsa_output.analog"
