import os
import sys
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

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
    client = lp.OllamaSentenceClient(model="qwen2.5:1.5b")

    strict = client.build_prompt(["SAYA", "MAKAN", "RUMAH"], allow_word_correction=False)
    corrective = client.build_prompt(["SAYA", "MAKAN", "RUMAH"], allow_word_correction=True)

    assert "Mode struktur saja" in strict
    assert "jangan mengganti kata inti" in strict
    assert "Mode perbaiki kata" in corrective
    assert "boleh mengganti kata" in corrective
    assert "Kata: SAYA MAKAN RUMAH" in strict
    assert "Jangan menyebut BISINDO" in client.build_system_prompt()


def test_ollama_compose_sanitizes_mocked_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"response":"\\"Saya makan di rumah.\\"\\nPenjelasan tidak dipakai."}'

    monkeypatch.setattr(lp.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    client = lp.OllamaSentenceClient(model="qwen2.5:1.5b", timeout=1.0)

    assert client.compose(["SAYA", "MAKAN", "RUMAH"]) == "Saya makan di rumah."


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
