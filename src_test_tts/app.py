import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from audio_analysis import save_analysis_reports
from audio_playback import play_audio
from cli import BASELINE_TEXTS, cmd_baseline_tests
from env_check import require_expected_venv, warn_if_wrong_venv
from paths import OUTPUT_DIR, display_path, ensure_base_dirs
from profile_manager import (
    DEMOGRAFI_DISPLAY_OPTIONS,
    GENDER_DISPLAY_OPTIONS,
    ProfileError,
    VoiceProfileManager,
    build_profile_name,
    default_profile_name,
    display_demografi,
    display_gender,
    make_profile,
    normalize_demografi_choice,
    normalize_gender_choice,
    preferred_profile_name,
    profile_names_for,
)
from tts_engine import InProcessTTSRuntime, TTSEngineError, generate_from_profile


DEFAULT_TEXT = "Halo, ini percobaan text to speech bahasa Indonesia."


def main(argv: List[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    strict = "--strict-venv" in argv
    warn_if_wrong_venv()
    if strict:
        try:
            require_expected_venv()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    ensure_base_dirs()
    try:
        import gradio as gr
    except ImportError:
        print(
            "ERROR: Gradio belum terinstall. Jalankan:\n"
            "source env_bisindo_cuda126/bin/activate\n"
            "pip install -r src_test_tts/requirements.txt",
            file=sys.stderr,
        )
        return 1
    app = build_app(gr)
    app.launch()
    return 0


def build_app(gr: Any) -> Any:
    manager = VoiceProfileManager()
    manager.merge_defaults()
    profile_names = manager.list_names()
    initial_gender = "cewek"
    initial_demografi = "dewasa"
    initial_profile_name = (
        preferred_profile_name(profile_names, initial_gender, initial_demografi)
        or default_profile_name(initial_gender, initial_demografi)
    )
    initial = manager.get(initial_profile_name)
    initial_profile_names = profile_names_for(profile_names, initial["gender"], initial["demografi"])
    runtime_holder: Dict[str, InProcessTTSRuntime] = {}

    with gr.Blocks(title="TTS Indonesia Test Lab") as app:
        gr.Markdown("## TTS Indonesia Test Lab")
        with gr.Row():
            gender = gr.Dropdown(
                list(GENDER_DISPLAY_OPTIONS),
                value=display_gender(initial["gender"]),
                label="Gender",
            )
            demografi = gr.Dropdown(
                list(DEMOGRAFI_DISPLAY_OPTIONS),
                value=display_demografi(initial["demografi"]),
                label="Usia",
            )
            profile_dropdown = gr.Dropdown(
                initial_profile_names,
                value=initial_profile_name,
                label="Profile/Variasi",
            )
            refresh_profiles = gr.Button("Refresh Profiles")
        with gr.Row():
            base_speaker = gr.Dropdown(["Wibowo", "Gadis"], value=initial["base_speaker"], label="Base Speaker")
            variasi_conf = gr.Textbox(value=initial["variasi_conf"], label="variasi_conf")
        text = gr.Textbox(value=DEFAULT_TEXT, lines=3, label="Teks TTS")
        with gr.Row():
            pitch = gr.Slider(-6.0, 6.0, value=initial["pitch_semitones"], step=0.1, label="Pitch Semitones")
            speed = gr.Slider(0.75, 1.30, value=initial["speed"], step=0.01, label="Speed/Tempo")
            volume = gr.Slider(-12.0, 12.0, value=initial["volume_gain_db"], step=0.5, label="Volume Gain dB")
            formant = gr.Slider(-0.3, 0.3, value=initial["formant_shift"], step=0.01, label="Formant Shift")
        with gr.Row():
            highpass = gr.Number(value=initial["highpass_hz"], label="Highpass Hz", precision=0)
            lowpass = gr.Number(value=initial["lowpass_hz"], label="Lowpass Hz", precision=0)
            sample_rate = gr.Number(value=initial["output_sample_rate"], label="Output Sample Rate", precision=0)
        with gr.Row():
            normalize = gr.Checkbox(value=initial["normalize"], label="Normalize")
            compressor = gr.Checkbox(value=initial["compressor"], label="Compressor")
        notes = gr.Textbox(value=initial.get("notes", ""), lines=2, label="Notes")
        with gr.Row():
            generate_btn = gr.Button("Generate")
            save_btn = gr.Button("Save Profile")
            baseline_btn = gr.Button("Generate Baseline Tests")
        with gr.Row():
            auto_play = gr.Checkbox(value=False, label="Auto Play Speaker")
            player = gr.Dropdown(["auto", "paplay", "aplay", "ffplay"], value="auto", label="Audio Player")
        audio = gr.Audio(label="Preview Output", type="filepath")
        raw_path = gr.Textbox(label="Raw WAV Path")
        final_path = gr.Textbox(label="Final WAV Path")
        metadata = gr.JSON(label="Metadata")
        status = gr.Textbox(label="Status")
        analysis_table = gr.Dataframe(label="Latest Analysis Report")
        gr.Markdown("### Live Test")
        live_text = gr.Textbox(value="", lines=2, label="Live Text")
        with gr.Row():
            warmup_live = gr.Button("Warmup Live")
            live_generate = gr.Button("Live Generate")

        fields = [
            gender,
            demografi,
            base_speaker,
            variasi_conf,
            pitch,
            speed,
            volume,
            formant,
            highpass,
            lowpass,
            normalize,
            compressor,
            sample_rate,
            notes,
        ]

        def profile_field_values(profile: Dict[str, Any]) -> Tuple[Any, ...]:
            return (
                display_gender(profile["gender"]),
                display_demografi(profile["demografi"]),
                profile["base_speaker"],
                profile["variasi_conf"],
                profile["pitch_semitones"],
                profile["speed"],
                profile["volume_gain_db"],
                profile["formant_shift"],
                profile["highpass_hz"],
                profile["lowpass_hz"],
                profile["normalize"],
                profile["compressor"],
                profile["output_sample_rate"],
                profile.get("notes", ""),
            )

        def load_profile(name: str) -> Tuple[Any, ...]:
            profile = VoiceProfileManager().get(name)
            return profile_field_values(profile)

        def fallback_profile(gender_value: str, demografi_value: str) -> Dict[str, Any]:
            base_speaker_value = "Gadis" if gender_value == "cewek" else "Wibowo"
            return make_profile(
                gender=gender_value,
                demografi=demografi_value,
                variasi_conf="default",
                base_speaker=base_speaker_value,
            )

        def refresh_picker(gender_choice: str, demografi_choice: str) -> Tuple[Any, ...]:
            gender_v = normalize_gender_choice(gender_choice)
            demografi_v = normalize_demografi_choice(demografi_choice)
            names = VoiceProfileManager().list_names()
            filtered = profile_names_for(names, gender_v, demografi_v)
            selected = filtered[0] if filtered else None
            if selected:
                profile = VoiceProfileManager().get(selected)
                status_text = f"OK: profile terpilih {selected}"
            else:
                profile = fallback_profile(gender_v, demografi_v)
                status_text = f"ERROR: tidak ada profile untuk {display_gender(gender_v)} / {display_demografi(demografi_v)}"
            return (
                gr.Dropdown(choices=filtered, value=selected),
                *profile_field_values(profile),
                status_text,
            )

        def profile_from_ui(*values: Any) -> Tuple[str, Dict[str, Any]]:
            (
                gender_choice,
                demografi_choice,
                base_speaker_v,
                variasi_v,
                pitch_v,
                speed_v,
                volume_v,
                formant_v,
                highpass_v,
                lowpass_v,
                normalize_v,
                compressor_v,
                sample_rate_v,
                notes_v,
            ) = values
            gender_v = normalize_gender_choice(gender_choice)
            demografi_v = normalize_demografi_choice(demografi_choice)
            profile = make_profile(
                gender=gender_v,
                demografi=demografi_v,
                variasi_conf=variasi_v,
                base_speaker=base_speaker_v,
                pitch=float(pitch_v),
                speed=float(speed_v),
                volume=float(volume_v),
                formant=float(formant_v),
                highpass_hz=int(highpass_v),
                lowpass_hz=int(lowpass_v),
                normalize=bool(normalize_v),
                compressor=bool(compressor_v),
                output_sample_rate=int(sample_rate_v),
                notes=notes_v or "",
            )
            return build_profile_name(gender_v, demografi_v, variasi_v), profile

        def get_runtime() -> InProcessTTSRuntime:
            runtime = runtime_holder.get("runtime")
            if runtime is None:
                runtime = InProcessTTSRuntime(device="auto")
                runtime.load()
                runtime_holder["runtime"] = runtime
            return runtime

        def warmup_ui() -> str:
            try:
                seconds = get_runtime().warmup(OUTPUT_DIR / "live_tests")
                return f"OK: live warmup selesai {seconds:.2f}s"
            except Exception as exc:  # noqa: BLE001 - keep UI responsive.
                return f"ERROR warmup: {exc}"

        def generate_ui(text_value: str, auto_play_value: bool, player_value: str, *values: Any) -> Tuple[str, str, str, Dict[str, Any], str]:
            try:
                name, profile = profile_from_ui(*values)
                result = generate_from_profile(text_value, name, profile, OUTPUT_DIR, runtime=get_runtime())
                play_status = ""
                if auto_play_value:
                    played_with = play_audio(result.final_wav_path, player=player_value, wait=True)
                    play_status = f" Played with {played_with}."
                return (
                    str(result.final_wav_path),
                    display_path(result.raw_wav_path),
                    display_path(result.final_wav_path),
                    result.metadata,
                    f"OK: {display_path(result.metadata_path)}.{play_status}",
                )
            except Exception as exc:  # noqa: BLE001 - keep UI responsive.
                return "", "", "", {}, f"ERROR: {exc}"

        def live_generate_ui(text_value: str, auto_play_value: bool, player_value: str, *values: Any) -> Tuple[str, str, str, Dict[str, Any], str]:
            try:
                name, profile = profile_from_ui(*values)
                result = generate_from_profile(text_value, name, profile, OUTPUT_DIR / "live_tests", analyze=False, runtime=get_runtime())
                play_status = ""
                if auto_play_value:
                    played_with = play_audio(result.final_wav_path, player=player_value, wait=True)
                    play_status = f" Played with {played_with}."
                return (
                    str(result.final_wav_path),
                    display_path(result.raw_wav_path),
                    display_path(result.final_wav_path),
                    result.metadata,
                    f"OK Live: {display_path(result.metadata_path)}.{play_status}",
                )
            except Exception as exc:  # noqa: BLE001 - keep UI responsive.
                return "", "", "", {}, f"ERROR: {exc}"

        def save_profile_ui(*values: Any) -> str:
            try:
                name, profile = profile_from_ui(*values)
                VoiceProfileManager().save(name, profile, overwrite=False)
                return f"OK: profile tersimpan {name}"
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {exc}"

        def baseline_ui() -> Tuple[str, List[List[Any]]]:
            try:
                import argparse

                args = argparse.Namespace(output_dir=str(OUTPUT_DIR / "baseline_tests"), no_analysis=False)
                cmd_baseline_tests(args)
                report = OUTPUT_DIR / "baseline_tests" / "analysis_report.json"
                rows: List[List[Any]] = []
                if report.exists():
                    data = json.loads(report.read_text(encoding="utf-8"))
                    for item in data:
                        rows.append(
                            [
                                item.get("path"),
                                item.get("estimated_f0_median_hz"),
                                item.get("duration_sec"),
                                item.get("peak_db"),
                                item.get("clipping_detected"),
                            ]
                        )
                return f"OK: baseline tests di {display_path(OUTPUT_DIR / 'baseline_tests')}", rows
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {exc}", []

        profile_dropdown.change(load_profile, inputs=[profile_dropdown], outputs=fields)
        gender.change(refresh_picker, inputs=[gender, demografi], outputs=[profile_dropdown, *fields, status])
        demografi.change(refresh_picker, inputs=[gender, demografi], outputs=[profile_dropdown, *fields, status])
        refresh_profiles.click(refresh_picker, inputs=[gender, demografi], outputs=[profile_dropdown, *fields, status])
        app.load(warmup_ui, outputs=[status])
        warmup_live.click(warmup_ui, outputs=[status])
        generate_btn.click(generate_ui, inputs=[text, auto_play, player, *fields], outputs=[audio, raw_path, final_path, metadata, status])
        live_generate.click(live_generate_ui, inputs=[live_text, auto_play, player, *fields], outputs=[audio, raw_path, final_path, metadata, status])
        live_text.submit(live_generate_ui, inputs=[live_text, auto_play, player, *fields], outputs=[audio, raw_path, final_path, metadata, status])
        save_btn.click(save_profile_ui, inputs=fields, outputs=[status])
        baseline_btn.click(baseline_ui, outputs=[status, analysis_table])
    return app


if __name__ == "__main__":
    raise SystemExit(main())
