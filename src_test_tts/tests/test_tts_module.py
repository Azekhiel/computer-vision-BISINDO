import json
import sys
from pathlib import Path

import pytest

import audio_playback
import env_check
import model_downloader
import tts_engine
from cli import main as cli_main
from profile_manager import (
    ProfileError,
    VoiceProfileManager,
    build_profile_name,
    display_demografi,
    display_gender,
    make_profile,
    normalize_demografi_choice,
    normalize_gender_choice,
    preferred_profile_name,
    profile_names_for,
    validate_profile_name,
)


def test_profile_name_validation_accepts_requested_format():
    validate_profile_name("cowok_dewasa_default")
    validate_profile_name("cewek_remaja_natural_01")
    validate_profile_name("cowok_anak_anak_ceria_01")


def test_profile_picker_filters_by_display_gender_and_age_default_first():
    names = [
        "cowok_anak_anak_default",
        "cewek_dewasa_soft_01",
        "cewek_dewasa_default",
        "cewek_remaja_default",
    ]

    assert normalize_gender_choice("Cewek") == "cewek"
    assert normalize_demografi_choice("Anak-Anak") == "anak_anak"
    assert display_gender("cewek") == "Cewek"
    assert display_demografi("dewasa") == "Dewasa"
    assert profile_names_for(names, "Cewek", "Dewasa") == [
        "cewek_dewasa_default",
        "cewek_dewasa_soft_01",
    ]
    assert preferred_profile_name(names, "Cowok", "Anak-Anak") == "cowok_anak_anak_default"


@pytest.mark.parametrize(
    "name",
    [
        "cowok dewasa default",
        "male_adult_default",
        "cewek_tua_default",
        "cowok_anak-anak_default",
        "cowok_anak_anak_",
    ],
)
def test_profile_name_validation_rejects_invalid_examples(name):
    with pytest.raises(ProfileError):
        validate_profile_name(name)


def test_profile_manager_does_not_overwrite_by_default(tmp_path):
    path = tmp_path / "voice_profiles.json"
    manager = VoiceProfileManager(path)
    assert manager.init_defaults()
    original = json.loads(path.read_text(encoding="utf-8"))
    original["cowok_dewasa_default"]["notes"] = "custom"
    path.write_text(json.dumps(original), encoding="utf-8")
    assert not manager.init_defaults(overwrite=False)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["cowok_dewasa_default"]["notes"] == "custom"


def test_save_profile_prevents_overwrite(tmp_path):
    path = tmp_path / "voice_profiles.json"
    manager = VoiceProfileManager(path)
    manager.init_defaults()
    profile = make_profile("cewek", "remaja", "soft_01", "Gadis", pitch=0.6, speed=1.03)
    manager.save("cewek_remaja_soft_01", profile)
    with pytest.raises(ProfileError):
        manager.save("cewek_remaja_soft_01", profile)


def test_env_check_expected_venv_from_virtual_env(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/env_bisindo_cuda126")
    assert env_check.is_expected_venv()
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/other_env")
    assert not env_check.is_expected_venv()


def test_model_downloader_mocked_download(tmp_path, monkeypatch):
    calls = []

    def fake_release_urls():
        return {}

    def fake_download(url, target, min_size):
        calls.append((url, target.name, min_size))
        target.write_bytes(b"x" * max(min_size, 32))

    monkeypatch.setattr(model_downloader, "_release_asset_urls", fake_release_urls)
    monkeypatch.setattr(model_downloader, "_download_url", fake_download)
    paths = model_downloader.download_model_if_missing(model_dir=tmp_path)
    assert paths["checkpoint_1260000-inference.pth"].exists()
    assert any(call[1] == "checkpoint_1260000-inference.pth" for call in calls)
    calls.clear()
    model_downloader.download_model_if_missing(model_dir=tmp_path)
    assert calls == []


def test_generate_from_profile_metadata_mocked(tmp_path, monkeypatch):
    profile = make_profile("cowok", "remaja", "soft_01", "Wibowo", pitch=1.0, speed=1.02)

    def fake_synthesize(text, base_speaker, output_path, speed=None):
        output_path.write_bytes(b"RIFFfakewav")
        return tts_engine.SynthesisResult(output_path, "halo", "wibowo")

    def fake_effects(raw, final, profile_data):
        final.write_bytes(b"RIFFfinalwav")
        return final, ["formant_shift belum diterapkan"]

    def fake_analysis(path):
        return {
            "path": str(path),
            "estimated_f0_mean_hz": 120.0,
            "estimated_f0_median_hz": 118.0,
            "duration_sec": 1.0,
            "rms_loudness": 0.1,
            "peak_db": -3.0,
            "spectral_centroid_mean": 1500.0,
            "sample_rate": 22050,
            "num_samples": 22050,
            "clipping_detected": False,
            "warnings": [],
        }

    monkeypatch.setattr(tts_engine, "synthesize_raw", fake_synthesize)
    monkeypatch.setattr(tts_engine, "apply_profile_effects", fake_effects)
    monkeypatch.setattr(tts_engine, "analyze_wav", fake_analysis)
    result = tts_engine.generate_from_profile("Halo", "cowok_remaja_soft_01", profile, tmp_path, timestamp="t")
    assert result.raw_wav_path.exists()
    assert result.final_wav_path.exists()
    assert result.metadata_path.exists()
    assert result.metadata["warnings"] == ["formant_shift belum diterapkan"]


def test_cli_init_and_list_use_tmp_profile(tmp_path, monkeypatch, capsys):
    profile_path = tmp_path / "voice_profiles.json"

    def manager_factory():
        return VoiceProfileManager(profile_path)

    monkeypatch.setattr("cli.VoiceProfileManager", manager_factory)
    assert cli_main(["init"]) == 0
    assert cli_main(["list"]) == 0
    out = capsys.readouterr().out
    assert "cowok_dewasa_default" in out


def test_audio_playback_auto_uses_available_player(tmp_path, monkeypatch):
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFFfake")
    commands = []

    def fake_which(name):
        return f"/usr/bin/{name}" if name == "paplay" else None

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        commands.append(cmd)
        return Result()

    monkeypatch.setattr(audio_playback.shutil, "which", fake_which)
    monkeypatch.setattr(audio_playback.subprocess, "run", fake_run)
    assert audio_playback.play_audio(wav) == "paplay"
    assert commands == [["paplay", str(wav)]]


def test_cli_generate_play_is_mocked(tmp_path, monkeypatch):
    profile_path = tmp_path / "voice_profiles.json"
    VoiceProfileManager(profile_path).init_defaults()
    final = tmp_path / "final.wav"
    raw = tmp_path / "raw.wav"
    meta = tmp_path / "final.json"
    final.write_bytes(b"RIFFfinal")
    raw.write_bytes(b"RIFFraw")
    meta.write_text("{}", encoding="utf-8")
    played = []

    def manager_factory():
        return VoiceProfileManager(profile_path)

    def fake_generate(text, profile_name, profile, output_dir, timestamp=None, analyze=True):
        metadata = {
            "raw_wav_path": str(raw),
            "final_wav_path": str(final),
            "warnings": [],
        }
        return tts_engine.GenerationResult(raw, final, meta, metadata)

    def fake_play(path, player="auto", wait=True):
        played.append((path, player, wait))
        return "paplay"

    monkeypatch.setattr("cli.VoiceProfileManager", manager_factory)
    monkeypatch.setattr("cli.generate_from_profile", fake_generate)
    monkeypatch.setattr("cli.play_audio", fake_play)
    assert cli_main(["generate", "--profile", "cewek_remaja_default", "--text", "Halo", "--play"]) == 0
    assert played == [(final, "auto", True)]
