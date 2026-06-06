import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from audio_analysis import analyze_input, analyze_wav, save_analysis_reports
from audio_effects import apply_profile_effects
from audio_playback import AudioPlaybackError, available_players, play_audio
from env_check import require_expected_venv, warn_if_wrong_venv
from model_downloader import ModelDownloadError, ensure_model_available, model_status, print_manual_download_instructions
from paths import OUTPUT_DIR, display_path, ensure_base_dirs
from profile_manager import ProfileError, VoiceProfileManager, build_profile_name, make_profile
from tts_engine import InProcessTTSRuntime, TTSEngineError, generate_from_profile, synthesize_raw


BASELINE_TEXTS = [
    "Halo, ini adalah pengujian suara untuk sistem text to speech bahasa Indonesia.",
    "Nama saya sedang diuji untuk melihat apakah suara terdengar natural, jelas, dan nyaman didengar.",
    "Saya bisa membantu membaca teks dengan intonasi yang stabil dan pelafalan yang mudah dipahami.",
]


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warn_if_wrong_venv()
    if getattr(args, "strict_venv", False):
        try:
            require_expected_venv()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    ensure_base_dirs()
    try:
        return int(args.func(args) or 0)
    except (ProfileError, TTSEngineError, ModelDownloadError, AudioPlaybackError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Indonesian TTS testing module.")
    parser.add_argument("--strict-venv", action="store_true", help="Berhenti kalau bukan env_bisindo_cuda126.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Buat folder dan inisialisasi preset default.")
    p_init.add_argument("--overwrite", action="store_true", help="Timpa voice_profiles.json existing.")
    p_init.set_defaults(func=cmd_init)

    p_download = sub.add_parser("download-model", help="Cek/download model TTS Indonesia.")
    p_download.set_defaults(func=cmd_download_model)

    p_list = sub.add_parser("list", help="Tampilkan semua profile.")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Tampilkan detail profile.")
    p_show.add_argument("profile")
    p_show.set_defaults(func=cmd_show)

    p_generate = sub.add_parser("generate", help="Generate TTS dari profile.")
    p_generate.add_argument("--profile", required=True)
    p_generate.add_argument("--text", required=True)
    p_generate.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p_generate.add_argument("--no-analysis", action="store_true")
    p_generate.add_argument("--play", action="store_true", help="Putar audio final setelah generate.")
    p_generate.add_argument("--player", default="auto", help="auto, paplay, aplay, atau ffplay.")
    p_generate.set_defaults(func=cmd_generate)

    p_live = sub.add_parser("live", help="Live text-to-speech: ketik teks, generate, lalu putar.")
    p_live.add_argument("--profile", required=True)
    p_live.add_argument("--output-dir", default=str(OUTPUT_DIR / "live_tests"))
    p_live.add_argument("--analysis", action="store_true", help="Aktifkan audio analysis di live mode.")
    p_live.add_argument("--no-analysis", action="store_true", help=argparse.SUPPRESS)
    p_live.add_argument("--no-play", action="store_true")
    p_live.add_argument("--player", default="auto", help="auto, paplay, aplay, atau ffplay.")
    p_live.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device in-process TTS runtime.")
    p_live.add_argument("--no-warmup", action="store_true", help="Lewati warmup awal model.")
    p_live.set_defaults(func=cmd_live)

    p_save = sub.add_parser("save-profile", help="Simpan profile baru.")
    add_profile_args(p_save, require_identity=True)
    p_save.add_argument("--overwrite", action="store_true")
    p_save.set_defaults(func=cmd_save_profile)

    p_update = sub.add_parser("update-profile", help="Update profile existing.")
    p_update.add_argument("profile")
    add_profile_args(p_update, require_identity=False)
    p_update.set_defaults(func=cmd_update_profile)

    p_delete = sub.add_parser("delete-profile", help="Hapus profile.")
    p_delete.add_argument("profile")
    p_delete.set_defaults(func=cmd_delete_profile)

    p_baseline = sub.add_parser("baseline-tests", help="Generate baseline listening/test pack.")
    p_baseline.add_argument("--output-dir", default=str(OUTPUT_DIR / "baseline_tests"))
    p_baseline.add_argument("--no-analysis", action="store_true")
    p_baseline.set_defaults(func=cmd_baseline_tests)

    p_analyze = sub.add_parser("analyze", help="Analisis WAV atau folder WAV.")
    p_analyze.add_argument("--input", required=True)
    p_analyze.add_argument("--text", default="")
    p_analyze.set_defaults(func=cmd_analyze)
    return parser


def add_profile_args(parser: argparse.ArgumentParser, require_identity: bool) -> None:
    parser.add_argument("--gender", required=require_identity, choices=["cowok", "cewek"])
    parser.add_argument("--demografi", required=require_identity, choices=["dewasa", "remaja", "anak_anak"])
    parser.add_argument("--variasi-conf", required=require_identity)
    parser.add_argument("--base-speaker", required=require_identity, choices=["Wibowo", "Gadis"])
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--volume", type=float, default=None)
    parser.add_argument("--formant", type=float, default=None)
    parser.add_argument("--highpass-hz", type=int, default=None)
    parser.add_argument("--lowpass-hz", type=int, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compressor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-sample-rate", type=int, default=None)
    parser.add_argument("--notes", default=None)


def cmd_init(args: argparse.Namespace) -> int:
    manager = VoiceProfileManager()
    created = manager.init_defaults(overwrite=args.overwrite)
    print("Folder testing TTS siap.")
    if created:
        print(f"Preset default ditulis ke {display_path(manager.path)}")
    else:
        added = manager.merge_defaults()
        if added:
            print(f"Menambahkan preset yang belum ada: {', '.join(added)}")
        else:
            print(f"Config existing dipakai ulang: {display_path(manager.path)}")
    return 0


def cmd_download_model(args: argparse.Namespace) -> int:
    try:
        paths = ensure_model_available()
    except ModelDownloadError:
        print_manual_download_instructions()
        raise
    print("Status model:")
    for row in model_status():
        flag = "OK" if row["valid"] else "MISSING"
        print(f"- [{flag}] {row['name']} -> {display_path(Path(row['path']))}")
    print("Model siap dipakai:")
    for path in paths.values():
        if path.exists():
            print(f"- {display_path(path)}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    manager = VoiceProfileManager()
    for name in manager.list_names():
        profile = manager.get(name)
        print(f"{name}\t{profile['base_speaker']}\t{profile['gender']}/{profile['demografi']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    manager = VoiceProfileManager()
    print(json.dumps(manager.get(args.profile), indent=2, ensure_ascii=False))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    manager = VoiceProfileManager()
    profile = manager.get(args.profile)
    result = generate_from_profile(
        args.text,
        args.profile,
        profile,
        Path(args.output_dir),
        analyze=not args.no_analysis,
    )
    _print_generation_result(result.metadata)
    if args.play:
        player = play_audio(result.final_wav_path, player=args.player, wait=True)
        print(f"Played with: {player}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    manager = VoiceProfileManager()
    profile = manager.get(args.profile)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    players = ", ".join(available_players()) or "tidak ada"
    print(f"Live TTS profile: {args.profile}")
    print(f"Output: {display_path(output_dir)}")
    print(f"Audio players: {players}")
    runtime = InProcessTTSRuntime(device=args.device)
    print(f"Memuat model TTS sekali untuk live mode (device={args.device})...")
    runtime.load()
    if not args.no_warmup:
        warmup_sec = runtime.warmup(output_dir)
        print(f"Warmup selesai: {warmup_sec:.2f}s. Input berikutnya tidak perlu load model ulang.")
    print("Ketik teks lalu Enter. Ketik :q atau exit untuk keluar.")
    while True:
        try:
            text = input("tts> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {":q", "q", "quit", "exit"}:
            break
        if not text:
            continue
        result = generate_from_profile(
            text,
            args.profile,
            profile,
            output_dir,
            analyze=bool(args.analysis),
            runtime=runtime,
        )
        _print_generation_result(result.metadata)
        if not args.no_play:
            player = play_audio(result.final_wav_path, player=args.player, wait=True)
            print(f"Played with: {player}")
    return 0


def cmd_save_profile(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args, require_identity=True)
    name = build_profile_name(args.gender, args.demografi, args.variasi_conf)
    VoiceProfileManager().save(name, profile, overwrite=args.overwrite)
    print(f"Profile tersimpan: {name}")
    return 0


def cmd_update_profile(args: argparse.Namespace) -> int:
    manager = VoiceProfileManager()
    current = manager.get(args.profile)
    updates: Dict[str, Any] = {}
    mapping = {
        "base_speaker": args.base_speaker,
        "pitch_semitones": args.pitch,
        "speed": args.speed,
        "volume_gain_db": args.volume,
        "formant_shift": args.formant,
        "highpass_hz": args.highpass_hz,
        "lowpass_hz": args.lowpass_hz,
        "normalize": args.normalize,
        "compressor": args.compressor,
        "output_sample_rate": args.output_sample_rate,
        "notes": args.notes,
    }
    updates.update({key: value for key, value in mapping.items() if value is not None})
    for key in ("gender", "demografi", "variasi_conf"):
        value = getattr(args, key if key != "variasi_conf" else "variasi_conf")
        if value is not None:
            updates[key] = value
    updated = manager.update(args.profile, updates)
    print(json.dumps(updated, indent=2, ensure_ascii=False))
    return 0


def cmd_delete_profile(args: argparse.Namespace) -> int:
    VoiceProfileManager().delete(args.profile)
    print(f"Profile dihapus: {args.profile}")
    return 0


def cmd_baseline_tests(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = VoiceProfileManager()
    manager.merge_defaults()
    profile_names = [
        "cowok_dewasa_default",
        "cewek_dewasa_default",
        "cowok_remaja_default",
        "cewek_remaja_default",
        "cowok_anak_anak_default",
        "cewek_anak_anak_default",
    ]
    metadata_files: List[Path] = []
    all_analysis: List[Dict[str, Any]] = []
    for idx, text in enumerate(BASELINE_TEXTS, start=1):
        stamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_%f_text{idx:02d}")
        for speaker in ("Wibowo", "Gadis"):
            meta = _generate_raw_speaker_baseline(text, speaker, output_dir, stamp, analyze=not args.no_analysis)
            metadata_files.append(Path(meta["metadata_path"]))
            if meta.get("audio_analysis"):
                all_analysis.append(meta["audio_analysis"])
        for name in profile_names:
            result = generate_from_profile(
                text,
                name,
                manager.get(name),
                output_dir,
                timestamp=stamp,
                analyze=not args.no_analysis,
            )
            metadata_files.append(result.metadata_path)
            if result.metadata.get("audio_analysis"):
                all_analysis.append(result.metadata["audio_analysis"])
    if all_analysis:
        json_path, csv_path = save_analysis_reports(all_analysis, output_dir)
        print(f"Analysis report: {display_path(json_path)}")
        print(f"Analysis CSV: {display_path(csv_path)}")
    print(f"Baseline test pack selesai: {display_path(output_dir)}")
    print(f"Metadata dibuat: {len(metadata_files)} file")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        raise RuntimeError(f"Input tidak ditemukan: {input_path}")
    rows = analyze_input(input_path, text=args.text)
    output_dir = input_path if input_path.is_dir() else input_path.parent
    json_path, csv_path = save_analysis_reports(rows, output_dir)
    print(f"Analysis JSON: {display_path(json_path)}")
    print(f"Analysis CSV: {display_path(csv_path)}")
    print(f"File WAV dianalisis: {len(rows)}")
    return 0


def _profile_from_args(args: argparse.Namespace, require_identity: bool) -> Dict[str, Any]:
    if require_identity:
        defaults = {
            "pitch": 0.0,
            "speed": 1.0,
            "volume": 0.0,
            "formant": 0.0,
            "highpass_hz": 80,
            "lowpass_hz": 9500,
            "normalize": True if args.normalize is None else args.normalize,
            "compressor": False if args.compressor is None else args.compressor,
            "output_sample_rate": 22050,
            "notes": args.notes or "",
        }
        return make_profile(
            gender=args.gender,
            demografi=args.demografi,
            variasi_conf=args.variasi_conf,
            base_speaker=args.base_speaker,
            pitch=args.pitch if args.pitch is not None else defaults["pitch"],
            speed=args.speed if args.speed is not None else defaults["speed"],
            volume=args.volume if args.volume is not None else defaults["volume"],
            formant=args.formant if args.formant is not None else defaults["formant"],
            highpass_hz=args.highpass_hz if args.highpass_hz is not None else defaults["highpass_hz"],
            lowpass_hz=args.lowpass_hz if args.lowpass_hz is not None else defaults["lowpass_hz"],
            normalize=defaults["normalize"],
            compressor=defaults["compressor"],
            output_sample_rate=args.output_sample_rate if args.output_sample_rate is not None else defaults["output_sample_rate"],
            notes=defaults["notes"],
        )
    raise RuntimeError("Internal error: require_identity=False belum didukung untuk pembuatan profile.")


def _generate_raw_speaker_baseline(text: str, speaker: str, output_dir: Path, timestamp: str, analyze: bool) -> Dict[str, Any]:
    name = f"{speaker.lower()}_raw_default"
    raw_path = output_dir / f"{timestamp}_{name}_raw.wav"
    final_path = output_dir / f"{timestamp}_{name}.wav"
    metadata_path = output_dir / f"{timestamp}_{name}.json"
    synth = synthesize_raw(text, speaker, raw_path)
    shutil.copyfile(raw_path, final_path)
    analysis = analyze_wav(final_path, text=text) if analyze else None
    metadata = {
        "text": text,
        "profile_name": name,
        "config": {
            "base_speaker": speaker,
            "raw_baseline": True,
            "pitch_semitones": 0.0,
            "speed": 1.0,
            "volume_gain_db": 0.0,
            "formant_shift": 0.0,
            "normalize": False,
            "compressor": False,
        },
        "base_speaker": speaker,
        "speaker_id": synth.speaker_id,
        "raw_wav_path": str(raw_path),
        "final_wav_path": str(final_path),
        "metadata_path": str(metadata_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "text_for_tts": synth.text_for_tts,
        "audio_analysis": analysis,
        "warnings": [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def _print_generation_result(metadata: Dict[str, Any]) -> None:
    print(f"Raw WAV: {display_path(Path(metadata['raw_wav_path']))}")
    print(f"Final WAV: {display_path(Path(metadata['final_wav_path']))}")
    print(f"Metadata: {display_path(Path(metadata['final_wav_path']).with_suffix('.json'))}")
    warnings = metadata.get("warnings") or []
    timing = metadata.get("timing_sec") or {}
    if timing:
        print(
            "Timing: "
            f"synth={timing.get('synthesis', 0):.3f}s "
            f"effects={timing.get('effects', 0):.3f}s "
            f"analysis={timing.get('analysis', 0):.3f}s "
            f"total={timing.get('total', 0):.3f}s"
        )
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
