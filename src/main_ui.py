"""Minimal GRU-focused BISINDO dashboard."""

from __future__ import annotations

import queue
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Iterable

import feature_schemas as fs
import gru_manager as gm
import live_session
import livetest_plus as lp
import live_gru_fast
import reinforcement_learning as rl
import tts_profile_runtime as tts_rt


DEFAULT_LIVE_SCHEMA = fs.DEFAULT_SCHEMA
LIVE_SCHEMA_CHOICES = fs.SCHEMA_NAMES
LIVE_MP_METHOD_CHOICES = ("holistic", "holistic_stabilized")
LIVE_MEDIAPIPE_HELP_TEXT = (
    "Workers: Stream=baca kamera, MediaPipe=jumlah worker ekstraksi landmark/feature, Infer=jumlah worker prediksi GRU; default 1/1/1. "
    "MP Method adalah mode extractor: holistic=Holistic standar; "
    "holistic_stabilized=Holistic + stabilizer tangan, lebih halus tapi sedikit lebih berat. "
    "Schema velocity/stateful tetap pakai MediaPipe=1. "
    "Mulai dari lossless1080_10 + holistic + 1/1/1; kalau jitter coba holistic_stabilized; "
    "kalau berat/lag turunkan profile ke accurate10 atau fast10."
)
LIVE_ROUTE_CHOICES = (
    ("Main GRU (tanpa expert)", "main"),
    ("Expert chunk10", "chunk10"),
    ("Expert threshold", "threshold"),
    ("Main + chunk10", "main_chunk10"),
    ("Main + threshold", "main_threshold"),
    ("Vote all experts", "vote_all"),
    ("Boosted stack", "boosted_stack"),
)
LIVE_ROUTE_DISPLAY_TO_VALUE = dict(LIVE_ROUTE_CHOICES)
LIVE_ROUTE_VALUE_TO_DISPLAY = {value: display for display, value in LIVE_ROUTE_CHOICES}
MODEL_DATA_CHOICES = ("Original", "Dengan augmentasi")
MODEL_DATA_DISPLAY_TO_MODE = {
    "Original": "original",
    "Dengan augmentasi": "with_augmentation",
}


def _is_all_specialist_request(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"", "all", "*", "semua"}


def _clean_extract_label(value: str) -> str:
    return str(value or "").strip().replace(" ", "_").lower()


def discover_extract_full_vocabs(source: str | Path) -> tuple[str, ...]:
    root = Path(source).expanduser()
    if (root / "features_holistic" / "fps_10").exists():
        root = root / "features_holistic" / "fps_10"
    labels: set[str] = set()
    for split_name in ("train", "val", "test"):
        split_dir = root / split_name
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if child.is_dir():
                label = _clean_extract_label(child.name)
                if label:
                    labels.add(label)
    return tuple(sorted(labels))


def resolve_live_schema_name(value: str | None) -> str:
    return fs.normalize_schema_name(value or fs.DEFAULT_SCHEMA)


def model_data_mode_from_display(value: str | None) -> str:
    raw = str(value or "Original").strip()
    if raw in MODEL_DATA_DISPLAY_TO_MODE:
        return MODEL_DATA_DISPLAY_TO_MODE[raw]
    return gm.normalize_train_data_mode(raw)


def variant_pool_for_model_data(model_data: str | None) -> tuple[str, ...]:
    mode = model_data_mode_from_display(model_data)
    return gm.AUGMENTED_VARIANT_NAMES if mode == "with_augmentation" else gm.BASE_VARIANT_NAMES


def map_variant_to_model_data(variant: str, model_data: str | None) -> str:
    value = str(variant or "auto").strip()
    if value in {"auto", "best"}:
        return "auto"
    normalized = gm.normalize_variant_name(value)
    mode = model_data_mode_from_display(model_data)
    if mode == "with_augmentation":
        return gm.augmented_variant_name(normalized)
    return gm.base_variant_name(normalized)


def validate_live_checkpoint(schema: str, variant: str, model_dir: str | Path = gm.MODEL_DIR, model_data: str | None = "original") -> str:
    schema_name = resolve_live_schema_name(schema)
    variant_name = live_gru_fast.normalize_live_variant(variant)
    if variant_name == "auto":
        try:
            return gm.select_best_available_variant(model_dir=model_dir, schema=schema_name, variants=variant_pool_for_model_data(model_data))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Checkpoint live belum ada untuk schema {schema_name}.") from exc
    variant_name = map_variant_to_model_data(variant_name, model_data)
    if not gm.checkpoint_exists(variant_name, model_dir=model_dir, schema=schema_name):
        raise FileNotFoundError(f"Checkpoint live belum ada untuk {schema_name}/gru_{variant_name}.")
    return variant_name


def resolve_live_route_name(value: str | None) -> str:
    raw = str(value or "main").strip()
    return LIVE_ROUTE_DISPLAY_TO_VALUE.get(raw, raw)


def display_live_route_name(value: str | None) -> str:
    route = resolve_live_route_name(value)
    return LIVE_ROUTE_VALUE_TO_DISPLAY.get(route, route)


class AppUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("BISINDO GRU Live Test")
        self.root.geometry("980x820")
        self.root.minsize(900, 720)

        self.live_worker = None
        self.live_queue: queue.Queue = queue.Queue()
        self.live_poll_job = None
        self.live_reset_job = None
        self.live_plus_worker = None
        self.live_plus_queue: queue.Queue = queue.Queue()
        self.live_plus_poll_job = None
        self.live_plus_reset_job = None
        self.live_plus_buffer = lp.SentenceBuffer()
        self.live_plus_speaker = lp.EspeakSpeaker()
        self.live_plus_sentence_pending = 0
        self.tts_runtime: tts_rt.LoadedTTSProfile | None = None
        self.tts_load_thread: threading.Thread | None = None

        self.variant_display_to_value: dict[str, str] = {}
        self.variant_options = ["auto"]
        self.live_variant_display_to_value: dict[str, str] = {}
        self.live_variant_options = ["auto"]
        self.dataset_dir_var = tk.StringVar(value=str(gm.DATASET_DIR))
        self.overwrite_existing_var = tk.BooleanVar(value=False)
        self.schema_var = tk.StringVar(value=fs.DEFAULT_SCHEMA)
        self.train_schema_vars = {name: tk.BooleanVar(value=(name == fs.DEFAULT_SCHEMA)) for name in fs.SCHEMA_NAMES}
        self.variant_var = tk.StringVar(value="auto")
        self.train_variant_vars = {
            variant: tk.BooleanVar(value=(variant == "adi"))
            for variant in gm.VARIANT_NAMES
        }
        self.specialist_name_var = tk.StringVar(value="d_p")
        self.specialist_vocab_var = tk.StringVar(value="d,p")
        self.specialist_epochs_var = tk.StringVar(value=str(gm.SPECIALIST_DEFAULT_EPOCHS))
        self.specialist_batch_var = tk.StringVar(value=str(gm.SPECIALIST_DEFAULT_BATCH_SIZE))
        self.specialist_patience_var = tk.StringVar(value=str(gm.SPECIALIST_DEFAULT_PATIENCE))
        self.specialist_schema_vars = {name: tk.BooleanVar(value=(name == fs.DEFAULT_SCHEMA)) for name in fs.SCHEMA_NAMES}
        self.specialist_variant_vars = {
            variant: tk.BooleanVar(value=(variant == "adi_dengan_augmentasi"))
            for variant in gm.VARIANT_NAMES
        }
        self.specialist_vocab_vars: dict[str, tk.BooleanVar] = {}
        self.specialist_vocab_items: list[str] = []
        self._specialist_last_auto_name = "d_p"
        self.specialist_train_thread: threading.Thread | None = None
        self.live_schema_var = tk.StringVar(value=DEFAULT_LIVE_SCHEMA)
        self.live_variant_var = tk.StringVar(value="auto")
        self.live_model_data_var = tk.StringVar(value="Original")
        self.live_specialist_enabled_var = tk.BooleanVar(value=True)
        self.live_specialist_name_var = tk.StringVar(value="all")
        self.mode_var = tk.StringVar(value=live_gru_fast.DEFAULT_LIVE_PROFILE)
        self.device_var = tk.StringVar(value="auto")
        self.live_device_var = tk.StringVar(value="auto")
        self.epochs_var = tk.StringVar(value="100")
        self.batch_var = tk.StringVar(value="64")
        self.record_label_var = tk.StringVar(value="")
        self.record_split_var = tk.StringVar(value="train")
        self.record_schema_var = tk.StringVar(value="all")
        self.record_count_var = tk.StringVar(value="1")
        self.record_duration_var = tk.StringVar(value="2.5")
        self.record_camera_var = tk.StringVar(value="0")
        self.record_profile_var = tk.StringVar(value="fast10")
        self.record_target_fps_var = tk.StringVar(value="10")
        self.record_preview_width_var = tk.StringVar(value="640")
        self.record_window_var = tk.BooleanVar(value=True)
        self.record_keep_short_var = tk.BooleanVar(value=False)
        self.extract_full_source_var = tk.StringVar(value=str(gm.ROOT_DIR / "dataset_full_mediapipe"))
        self.extract_full_schema_var = tk.StringVar(value="full")
        self.extract_full_schema_vars = {name: tk.BooleanVar(value=True) for name in fs.SCHEMA_NAMES}
        self.extract_full_clean_var = tk.StringVar(value="backup")
        self.extract_full_all_vocab_var = tk.BooleanVar(value=True)
        self.extract_full_vocab_vars: dict[str, tk.BooleanVar] = {}
        self.extract_full_vocab_items: list[str] = []
        self.extract_full_process = None
        self.photo_source_var = tk.StringVar(value=str(gm.ROOT_DIR / "record" / "photo"))
        self.photo_workers_var = tk.StringVar(value="1")
        self.photo_profile_var = tk.StringVar(value="fast10")
        self.photo_schema_vars = {name: tk.BooleanVar(value=(name == fs.DEFAULT_SCHEMA)) for name in fs.SCHEMA_NAMES}
        self.photo_process = None
        self.photo_last_session = ""
        self.gif_schema_var = tk.StringVar(value="all")
        self.gif_split_var = tk.StringVar(value="all")
        self.gif_limit_var = tk.StringVar(value="5")
        self.gif_width_var = tk.StringVar(value="420")
        self.gif_force_var = tk.BooleanVar(value=False)
        self.gif_draw_face_var = tk.StringVar(value="auto")
        self.gif_vocab_items: list[str] = []
        self.gif_process = None
        self.sample_schema_var = tk.StringVar(value="all")
        self.sample_split_var = tk.StringVar(value="all")
        self.sample_vocab_var = tk.StringVar(value="")
        self.sample_dry_run_var = tk.BooleanVar(value=True)
        self.sample_items: list[dict[str, str]] = []
        self.sample_process = None
        self.augment_schema_var = tk.StringVar(value="all")
        self.augment_schema_vars = {name: tk.BooleanVar(value=True) for name in fs.SCHEMA_NAMES}
        self.augment_split_var = tk.StringVar(value="train")
        self.augment_split_vars = {
            "train": tk.BooleanVar(value=True),
            "val": tk.BooleanVar(value=False),
            "test": tk.BooleanVar(value=False),
        }
        self.augment_vocab_var = tk.StringVar(value="")
        self.augment_all_vocab_var = tk.BooleanVar(value=False)
        self.augment_vocab_vars: dict[str, tk.BooleanVar] = {}
        self.augment_vocab_items: list[str] = []
        self.augment_copies_var = tk.StringVar(value="2")
        self.augment_min_var = tk.StringVar(value="5")
        self.augment_target_var = tk.StringVar(value="0")
        self.augment_seed_var = tk.StringVar(value="")
        self.augment_delete_dry_var = tk.BooleanVar(value=True)
        self.augment_process = None
        self.train_suite_schema_var = tk.StringVar(value="full")
        self.train_suite_schema_vars = {
            name: tk.BooleanVar(value=(name in fs.expand_schema_names("full")))
            for name in fs.SCHEMA_NAMES
        }
        self.train_suite_var = tk.StringVar(value="main,chunk10,threshold,boosted")
        self.train_suite_variant_vars = {
            variant: tk.BooleanVar(value=(variant == "adi"))
            for variant in gm.VARIANT_NAMES
        }
        self.train_suite_suite_vars = {
            "main": tk.BooleanVar(value=True),
            "chunk10": tk.BooleanVar(value=True),
            "threshold": tk.BooleanVar(value=True),
            "boosted": tk.BooleanVar(value=True),
        }
        self.train_suite_process = None
        self.eval_schema_var = tk.StringVar(value="all")
        self.eval_variant_var = tk.StringVar(value="all")
        self.eval_suite_var = tk.StringVar(value="all")
        self.eval_split_var = tk.StringVar(value="test")
        self.eval_process = None
        self.rl_schema_var = tk.StringVar(value="smart_face")
        self.rl_variant_var = tk.StringVar(value="auto")
        self.rl_model_data_var = tk.StringVar(value="Original")
        self.rl_route_var = tk.StringVar(value=display_live_route_name("main"))
        self.rl_profile_var = tk.StringVar(value=live_gru_fast.DEFAULT_LIVE_PROFILE)
        self.rl_device_var = tk.StringVar(value="auto")
        self.rl_camera_var = tk.StringVar(value="0")
        self.rl_lr_var = tk.StringVar(value="0.0001")
        self.rl_steps_var = tk.StringVar(value="2")
        self.rl_correct_label_var = tk.StringVar(value="")
        self.rl_status_var = tk.StringVar(value="Reinforcement: idle")
        self.rl_prediction_var = tk.StringVar(value="Prediksi: -")
        self.rl_checkpoint_var = tk.StringVar(value="Checkpoint RL: -")
        self.rl_backup_var = tk.StringVar(value="Backup: -")
        self.rl_session: rl.ReinforcementSession | None = None
        self.rl_worker = None
        self.rl_queue: queue.Queue = queue.Queue()
        self.rl_poll_job = None
        self.rl_reset_job = None
        self.rl_train_thread: threading.Thread | None = None
        self.rl_last_sequence = None
        self.rl_last_prediction: dict[str, object] = {}
        self.live_route_var = tk.StringVar(value=display_live_route_name("main"))
        self.live_stream_workers_var = tk.StringVar(value="1")
        self.live_mp_workers_var = tk.StringVar(value="1")
        self.live_inference_workers_var = tk.StringVar(value="1")
        self.live_mp_method_var = tk.StringVar(value="holistic")
        self.live_threshold_var = tk.StringVar(value="default")
        self.live_stop_started_at: float | None = None
        self.live_plus_stop_started_at: float | None = None
        self.live_plus_use_llm_var = tk.BooleanVar(value=False)
        self.live_plus_allow_word_fix_var = tk.BooleanVar(value=False)
        self.live_plus_ollama_model_var = tk.StringVar(value=lp.DEFAULT_OLLAMA_MODEL)
        self.live_plus_audio_sink_var = tk.StringVar(value="auto/default")
        self.live_plus_audio_sink_display_to_name: dict[str, str | None] = {"auto/default": None}
        self.tts_gender_var = tk.StringVar(value="Cewek")
        self.tts_demografi_var = tk.StringVar(value="Dewasa")
        self.tts_profile_var = tk.StringVar(value="")
        self.tts_device_var = tk.StringVar(value="auto")
        self.tts_player_var = tk.StringVar(value="auto")
        self.tts_use_loaded_var = tk.BooleanVar(value=True)
        self.tts_test_text_var = tk.StringVar(value="Halo, ini percobaan suara dari profil TTS.")
        self.tts_status_var = tk.StringVar(value="TTS: idle (belum load)")
        self.tts_detail_var = tk.StringVar(value="Profile: -")
        self.status_var = tk.StringVar(value="Siap.")
        self.live_status_var = tk.StringVar(value="Live: idle")
        self.live_plus_status_var = tk.StringVar(value="LiveTest Plus: idle")
        self.live_plus_buffer_var = tk.StringVar(value="Buffer: -")
        self.live_plus_output_var = tk.StringVar(value="Output akhir: -")
        self.live_plus_llm_var = tk.StringVar(value="LLM: -")
        self.live_plus_tts_var = tk.StringVar(value="Suara/TTS: -")

        self._build_ui()
        if hasattr(self.root, "protocol"):
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh_status()

    def _scroll_tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        outer = ttk.Frame(notebook)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=10)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_configure(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window_id, width=canvas.winfo_width())

        inner.bind("<Configure>", on_configure)
        canvas.bind("<Configure>", on_configure)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        notebook.add(outer, text=title)
        inner.columnconfigure(0, weight=1)
        return inner

    def _on_close(self) -> None:
        for worker in (getattr(self, "live_worker", None), getattr(self, "live_plus_worker", None), getattr(self, "rl_worker", None)):
            try:
                live_session.stop_worker(worker, join_timeout=0.5, force=True)
            except Exception:
                pass
        if getattr(self, "rl_session", None) is not None:
            try:
                backup_dir = self.rl_session.finish()
                self.rl_backup_var.set(f"Backup: {backup_dir}")
            except Exception:
                pass
        try:
            self.unload_tts_profile(silent=True)
        except Exception:
            pass
        self.root.destroy()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        title = ttk.Label(outer, text="BISINDO GRU Dashboard", font=("Arial", 18, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        dataset_bar = ttk.LabelFrame(outer, text="Dataset Folder")
        dataset_bar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        dataset_bar.columnconfigure(1, weight=1)
        ttk.Label(dataset_bar, text="Root").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(dataset_bar, textvariable=self.dataset_dir_var).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(dataset_bar, text="Browse", command=self.browse_dataset_dir).grid(row=0, column=2, sticky="ew", padx=8, pady=6)
        ttk.Button(dataset_bar, text="Refresh", command=self.refresh_status).grid(row=0, column=3, sticky="ew", padx=8, pady=6)
        ttk.Checkbutton(
            dataset_bar,
            text="Overwrite existing outputs",
            variable=self.overwrite_existing_var,
        ).grid(row=1, column=1, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        notebook = ttk.Notebook(outer)
        notebook.grid(row=2, column=0, sticky="nsew")
        dataset_tab = self._scroll_tab(notebook, "Dataset")
        training_tab = self._scroll_tab(notebook, "Training")
        augment_tab = self._scroll_tab(notebook, "Augmentasi")
        multi_tab = self._scroll_tab(notebook, "Multi-Model")
        live_tab = self._scroll_tab(notebook, "Live")
        live_plus_tab = self._scroll_tab(notebook, "LiveTest Plus")
        reinforcement_tab = self._scroll_tab(notebook, "Reinforcement")
        tts_profile_tab = self._scroll_tab(notebook, "TTS Profile")
        maintenance_tab = self._scroll_tab(notebook, "Maintenance")
        logs_tab = ttk.Frame(notebook, padding=10)
        logs_tab.columnconfigure(0, weight=1)
        logs_tab.rowconfigure(0, weight=1)
        notebook.add(logs_tab, text="Logs")

        controls = ttk.LabelFrame(training_tab, text="Model")
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            controls.columnconfigure(idx, weight=1)

        ttk.Label(controls, text="Target model").grid(row=0, column=0, sticky="nw", padx=8, pady=8)
        train_variant_frame = ttk.Frame(controls)
        train_variant_frame.grid(row=0, column=1, columnspan=7, sticky="ew", padx=8, pady=8)
        for idx, variant in enumerate(gm.VARIANT_NAMES):
            ttk.Checkbutton(
                train_variant_frame,
                text=f"gru_{variant}",
                variable=self.train_variant_vars[variant],
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=(0, 14), pady=2)

        ttk.Label(controls, text="Schema").grid(row=1, column=0, sticky="nw", padx=8, pady=8)
        train_schema_frame = ttk.Frame(controls)
        train_schema_frame.grid(row=1, column=1, columnspan=7, sticky="ew", padx=8, pady=8)
        for idx, schema_name in enumerate(fs.SCHEMA_NAMES):
            ttk.Checkbutton(
                train_schema_frame,
                text=schema_name,
                variable=self.train_schema_vars[schema_name],
                command=self.refresh_status,
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=(0, 14), pady=2)

        ttk.Label(controls, text="Train device").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        ttk.Combobox(
            controls,
            textvariable=self.device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=8,
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=8)

        ttk.Label(controls, text="Epochs").grid(row=2, column=2, sticky="w", padx=8, pady=8)
        ttk.Entry(controls, textvariable=self.epochs_var, width=10).grid(row=2, column=3, sticky="ew", padx=8, pady=8)
        ttk.Label(controls, text="Batch").grid(row=2, column=4, sticky="w", padx=8, pady=8)
        ttk.Entry(controls, textvariable=self.batch_var, width=10).grid(row=2, column=5, sticky="ew", padx=8, pady=8)

        specialist_box = ttk.LabelFrame(training_tab, text="Vocab Khusus")
        specialist_box.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            specialist_box.columnconfigure(idx, weight=1)

        ttk.Label(specialist_box, text="Nama").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(specialist_box, textvariable=self.specialist_name_var, width=14).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(specialist_box, text="Manual vocab").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(specialist_box, textvariable=self.specialist_vocab_var, width=18).grid(row=0, column=3, columnspan=2, sticky="ew", padx=8, pady=6)
        self.btn_load_specialist_vocab = ttk.Button(specialist_box, text="Muat Vocab", command=self.refresh_specialist_vocab_list)
        self.btn_load_specialist_vocab.grid(row=0, column=5, sticky="ew", padx=8, pady=6)
        self.btn_train_specialist = ttk.Button(specialist_box, text="Train Vocab Khusus", command=self.train_specialist_selected)
        self.btn_train_specialist.grid(row=0, column=6, columnspan=2, sticky="ew", padx=8, pady=6)

        ttk.Label(specialist_box, text="Schema").grid(row=1, column=0, sticky="nw", padx=8, pady=6)
        specialist_schema_frame = ttk.Frame(specialist_box)
        specialist_schema_frame.grid(row=1, column=1, columnspan=7, sticky="ew", padx=8, pady=6)
        for idx, schema_name in enumerate(fs.SCHEMA_NAMES):
            ttk.Checkbutton(
                specialist_schema_frame,
                text=schema_name,
                variable=self.specialist_schema_vars[schema_name],
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=(0, 14), pady=2)

        ttk.Label(specialist_box, text="Model").grid(row=2, column=0, sticky="nw", padx=8, pady=6)
        specialist_variant_frame = ttk.Frame(specialist_box)
        specialist_variant_frame.grid(row=2, column=1, columnspan=7, sticky="ew", padx=8, pady=6)
        for idx, variant in enumerate(gm.VARIANT_NAMES):
            ttk.Checkbutton(
                specialist_variant_frame,
                text=f"gru_{variant}",
                variable=self.specialist_variant_vars[variant],
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=(0, 14), pady=2)

        ttk.Label(specialist_box, text="Epochs").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(specialist_box, textvariable=self.specialist_epochs_var, width=8).grid(row=3, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(specialist_box, text="Batch").grid(row=3, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(specialist_box, textvariable=self.specialist_batch_var, width=8).grid(row=3, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(specialist_box, text="Patience").grid(row=3, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(specialist_box, textvariable=self.specialist_patience_var, width=8).grid(row=3, column=5, sticky="ew", padx=8, pady=6)
        self.specialist_vocab_frame = ttk.Frame(specialist_box)
        self.specialist_vocab_frame.grid(row=4, column=0, columnspan=8, sticky="ew", padx=8, pady=(0, 6))
        self._set_specialist_vocab_items(["d", "p"])

        extract_box = ttk.LabelFrame(dataset_tab, text="Extract Full MediaPipe Dataset")
        extract_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            extract_box.columnconfigure(idx, weight=1)
        ttk.Label(extract_box, text="Source").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(extract_box, textvariable=self.extract_full_source_var, width=42).grid(row=0, column=1, columnspan=3, sticky="ew", padx=8, pady=6)
        ttk.Label(extract_box, text="Clean").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Combobox(extract_box, textvariable=self.extract_full_clean_var, values=["backup", "none"], state="readonly", width=8).grid(row=0, column=5, sticky="ew", padx=8, pady=6)
        ttk.Button(extract_box, text="Muat Vocab", command=self.refresh_extract_full_vocab_list).grid(row=0, column=6, sticky="ew", padx=8, pady=6)
        self.btn_extract_full = ttk.Button(extract_box, text="Extract Full Dataset", command=self.extract_full_dataset)
        self.btn_extract_full.grid(row=0, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(extract_box, text="Schema").grid(row=1, column=0, sticky="nw", padx=8, pady=6)
        extract_schema_frame = ttk.Frame(extract_box)
        extract_schema_frame.grid(row=1, column=1, columnspan=7, sticky="ew", padx=8, pady=6)
        for idx, schema_name in enumerate(fs.SCHEMA_NAMES):
            ttk.Checkbutton(
                extract_schema_frame,
                text=schema_name,
                variable=self.extract_full_schema_vars[schema_name],
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=(0, 14), pady=2)

        ttk.Label(extract_box, text="Vocab").grid(row=2, column=0, sticky="nw", padx=8, pady=6)
        ttk.Checkbutton(
            extract_box,
            text="Pilih semua vocab",
            variable=self.extract_full_all_vocab_var,
            command=self._toggle_all_extract_full_vocab,
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=8, pady=6)
        self.extract_full_vocab_frame = ttk.Frame(extract_box)
        self.extract_full_vocab_frame.grid(row=3, column=0, columnspan=8, sticky="ew", padx=8, pady=(0, 6))
        self._set_extract_full_vocab_items(discover_extract_full_vocabs(self.extract_full_source_var.get()))

        recorder = ttk.LabelFrame(dataset_tab, text="Record Dataset Live — feature-only, tanpa simpan video")
        recorder.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            recorder.columnconfigure(idx, weight=1)

        ttk.Label(recorder, text="Label vocab").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(recorder, textvariable=self.record_label_var, width=16).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(recorder, text="Split").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            recorder,
            textvariable=self.record_split_var,
            values=["train", "val", "test"],
            state="readonly",
            width=8,
        ).grid(row=0, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(recorder, text="Schema simpan").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            recorder,
            textvariable=self.record_schema_var,
            values=["all", "face", "full", *fs.SCHEMA_NAMES],
            state="readonly",
            width=12,
        ).grid(row=0, column=5, sticky="ew", padx=8, pady=6)
        self.btn_record_live = ttk.Button(recorder, text="Record Live Dataset", command=self.record_live_dataset)
        self.btn_record_live.grid(row=0, column=6, columnspan=2, sticky="ew", padx=8, pady=6)

        ttk.Label(recorder, text="Count").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(recorder, textvariable=self.record_count_var, width=8).grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(recorder, text="Durasi/sample").grid(row=1, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(recorder, textvariable=self.record_duration_var, width=8).grid(row=1, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(recorder, text="Camera").grid(row=1, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(recorder, textvariable=self.record_camera_var, width=8).grid(row=1, column=5, sticky="ew", padx=8, pady=6)
        ttk.Checkbutton(recorder, text="Preview window", variable=self.record_window_var).grid(row=1, column=6, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(recorder, text="Keep short", variable=self.record_keep_short_var).grid(row=1, column=7, sticky="w", padx=8, pady=6)

        ttk.Label(recorder, text="Record profile").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            recorder,
            textvariable=self.record_profile_var,
            values=sorted(live_gru_fast.LIVE_PROFILES),
            state="readonly",
            width=10,
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(recorder, text="Target FPS simpan").grid(row=2, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(recorder, textvariable=self.record_target_fps_var, width=8).grid(row=2, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(recorder, text="Preview width").grid(row=2, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(recorder, textvariable=self.record_preview_width_var, width=8).grid(row=2, column=5, sticky="ew", padx=8, pady=6)
        ttk.Label(recorder, text="Preview menampilkan FPS capture + FPS simpan.").grid(row=2, column=6, columnspan=2, sticky="w", padx=8, pady=6)

        photo_box = ttk.LabelFrame(dataset_tab, text="Import Foto Dataset")
        photo_box.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            photo_box.columnconfigure(idx, weight=1)

        ttk.Label(photo_box, text="Source").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(photo_box, textvariable=self.photo_source_var, width=42).grid(row=0, column=1, columnspan=3, sticky="ew", padx=8, pady=6)
        ttk.Label(photo_box, text="Workers").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(photo_box, textvariable=self.photo_workers_var, width=8).grid(row=0, column=5, sticky="ew", padx=8, pady=6)
        ttk.Label(photo_box, text="Profile").grid(row=0, column=6, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            photo_box,
            textvariable=self.photo_profile_var,
            values=sorted(live_gru_fast.LIVE_PROFILES),
            state="readonly",
            width=10,
        ).grid(row=0, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(photo_box, text="Schema").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        schema_frame = ttk.Frame(photo_box)
        schema_frame.grid(row=1, column=1, columnspan=5, sticky="ew", padx=8, pady=6)
        for idx, schema_name in enumerate(fs.SCHEMA_NAMES):
            ttk.Checkbutton(schema_frame, text=schema_name, variable=self.photo_schema_vars[schema_name]).grid(row=0, column=idx, sticky="w", padx=(0, 10))
        self.btn_photo_extract = ttk.Button(photo_box, text="Extract Foto", command=self.extract_photo_dataset)
        self.btn_photo_extract.grid(row=1, column=6, columnspan=2, sticky="ew", padx=8, pady=6)

        gif_box = ttk.LabelFrame(dataset_tab, text="Hasilkan GIF Dataset — dari feature parquet, ringan tanpa video")
        gif_box.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            gif_box.columnconfigure(idx, weight=1)

        ttk.Label(gif_box, text="Schema GIF").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            gif_box,
            textvariable=self.gif_schema_var,
            values=["all", "face", "full", *fs.SCHEMA_NAMES],
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(gif_box, text="Split").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            gif_box,
            textvariable=self.gif_split_var,
            values=["all", "train", "val", "test"],
            state="readonly",
            width=8,
        ).grid(row=0, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(gif_box, text="Limit/split").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(gif_box, textvariable=self.gif_limit_var, width=8).grid(row=0, column=5, sticky="ew", padx=8, pady=6)
        self.btn_load_gif_vocab = ttk.Button(gif_box, text="Muat List Vocab", command=self.load_gif_vocab_list)
        self.btn_load_gif_vocab.grid(row=0, column=6, sticky="ew", padx=8, pady=6)
        self.btn_make_gif = ttk.Button(gif_box, text="Hasilkan GIF", command=self.generate_dataset_gifs)
        self.btn_make_gif.grid(row=0, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(gif_box, text="Vocab dari dataset").grid(row=1, column=0, sticky="nw", padx=8, pady=6)
        self.gif_vocab_listbox = tk.Listbox(gif_box, height=5, exportselection=False)
        self.gif_vocab_listbox.grid(row=1, column=1, columnspan=5, sticky="ew", padx=8, pady=6)
        gif_scroll = ttk.Scrollbar(gif_box, orient="vertical", command=self.gif_vocab_listbox.yview)
        gif_scroll.grid(row=1, column=6, sticky="ns", pady=6)
        self.gif_vocab_listbox.configure(yscrollcommand=gif_scroll.set)
        opts = ttk.Frame(gif_box)
        opts.grid(row=1, column=7, sticky="nsew", padx=8, pady=6)
        ttk.Label(opts, text="Width").pack(anchor="w")
        ttk.Entry(opts, textvariable=self.gif_width_var, width=8).pack(fill="x", pady=(0, 6))
        ttk.Label(opts, text="Face GIF").pack(anchor="w")
        ttk.Combobox(opts, textvariable=self.gif_draw_face_var, values=["auto", "off", "sparse", "mesh"], state="readonly", width=8).pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(opts, text="Force overwrite", variable=self.gif_force_var).pack(anchor="w")

        sample_box = ttk.LabelFrame(maintenance_tab, text="Hapus Sample Dataset — semua schema/dimensi + GIF terkait")
        sample_box.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            sample_box.columnconfigure(idx, weight=1)

        ttk.Label(sample_box, text="Schema target").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            sample_box,
            textvariable=self.sample_schema_var,
            values=["all", "face", "full", *fs.SCHEMA_NAMES],
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(sample_box, text="Split").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            sample_box,
            textvariable=self.sample_split_var,
            values=["all", "train", "val", "test"],
            state="readonly",
            width=8,
        ).grid(row=0, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(sample_box, text="Vocab").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(sample_box, textvariable=self.sample_vocab_var, width=16).grid(row=0, column=5, sticky="ew", padx=8, pady=6)
        self.btn_load_samples = ttk.Button(sample_box, text="Muat Sample", command=self.load_sample_list)
        self.btn_load_samples.grid(row=0, column=6, sticky="ew", padx=8, pady=6)
        self.btn_delete_sample = ttk.Button(sample_box, text="Hapus Sample", command=self.delete_selected_sample)
        self.btn_delete_sample.grid(row=0, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(sample_box, text="Pilih sample").grid(row=1, column=0, sticky="nw", padx=8, pady=6)
        self.sample_listbox = tk.Listbox(sample_box, height=5, exportselection=False)
        self.sample_listbox.grid(row=1, column=1, columnspan=5, sticky="ew", padx=8, pady=6)
        sample_scroll = ttk.Scrollbar(sample_box, orient="vertical", command=self.sample_listbox.yview)
        sample_scroll.grid(row=1, column=6, sticky="ns", pady=6)
        self.sample_listbox.configure(yscrollcommand=sample_scroll.set)
        sample_opts = ttk.Frame(sample_box)
        sample_opts.grid(row=1, column=7, sticky="nsew", padx=8, pady=6)
        ttk.Checkbutton(sample_opts, text="Dry run dulu", variable=self.sample_dry_run_var).pack(anchor="w")
        ttk.Label(sample_opts, text="Isi vocab, muat sample, pilih video_id, lalu hapus.").pack(anchor="w", pady=(8, 0))


        augment_box = ttk.LabelFrame(augment_tab, text="Augmentasi Dataset - feature-only, bisa hapus hasil augmentasi")
        augment_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            augment_box.columnconfigure(idx, weight=1)
        ttk.Label(augment_box, text="Schema").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        schema_frame = ttk.Frame(augment_box)
        schema_frame.grid(row=0, column=1, columnspan=7, sticky="ew", padx=8, pady=6)
        for idx, schema_name in enumerate(fs.SCHEMA_NAMES):
            ttk.Checkbutton(
                schema_frame,
                text=schema_name,
                variable=self.augment_schema_vars[schema_name],
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=(0, 14), pady=2)
        ttk.Label(augment_box, text="Split").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        split_frame = ttk.Frame(augment_box)
        split_frame.grid(row=1, column=1, columnspan=4, sticky="ew", padx=8, pady=6)
        for idx, split_name in enumerate(("train", "val", "test")):
            ttk.Checkbutton(split_frame, text=split_name, variable=self.augment_split_vars[split_name]).grid(row=0, column=idx, sticky="w", padx=(0, 10))
        self.btn_refresh_augment_vocab = ttk.Button(augment_box, text="Muat Vocab", command=self.refresh_augment_vocab_list)
        self.btn_refresh_augment_vocab.grid(row=1, column=5, sticky="ew", padx=8, pady=6)
        self.btn_augment = ttk.Button(augment_box, text="Augmentasi", command=self.run_augmentation)
        self.btn_augment.grid(row=1, column=6, sticky="ew", padx=8, pady=6)
        self.btn_augment_delete = ttk.Button(augment_box, text="Hapus Augmentasi", command=self.delete_augmentation)
        self.btn_augment_delete.grid(row=1, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(augment_box, text="Copy/sample").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(augment_box, textvariable=self.augment_copies_var, width=8).grid(row=2, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(augment_box, text="Min asli").grid(row=2, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(augment_box, textvariable=self.augment_min_var, width=8).grid(row=2, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(augment_box, text="Target/class").grid(row=2, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(augment_box, textvariable=self.augment_target_var, width=8).grid(row=2, column=5, sticky="ew", padx=8, pady=6)
        ttk.Label(augment_box, text="Seed").grid(row=2, column=6, sticky="w", padx=8, pady=6)
        seed_row = ttk.Frame(augment_box)
        seed_row.grid(row=2, column=7, sticky="ew", padx=8, pady=6)
        ttk.Entry(seed_row, textvariable=self.augment_seed_var, width=8).pack(side="left", fill="x", expand=True)
        ttk.Checkbutton(seed_row, text="Dry delete", variable=self.augment_delete_dry_var).pack(side="left", padx=(6, 0))

        vocab_box = ttk.LabelFrame(augment_tab, text="Vocab yang diaugmentasi")
        vocab_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            vocab_box.columnconfigure(idx, weight=1)
        ttk.Checkbutton(vocab_box, text="Pilih semua vocab", variable=self.augment_all_vocab_var, command=self._toggle_all_augment_vocab).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=6)
        ttk.Label(vocab_box, text="Manual").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(vocab_box, textvariable=self.augment_vocab_var, width=18).grid(row=0, column=3, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Label(vocab_box, text="Kosong/manual kosong = semua vocab yang cocok").grid(row=0, column=5, columnspan=3, sticky="w", padx=8, pady=6)
        self.augment_vocab_frame = ttk.Frame(vocab_box)
        self.augment_vocab_frame.grid(row=1, column=0, columnspan=8, sticky="ew", padx=8, pady=(0, 6))

        multi_box = ttk.LabelFrame(multi_tab, text="Train Multi-Model Suite")
        multi_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            multi_box.columnconfigure(idx, weight=1)
        ttk.Label(multi_box, text="Schema").grid(row=0, column=0, sticky="nw", padx=8, pady=6)
        suite_schema_frame = ttk.Frame(multi_box)
        suite_schema_frame.grid(row=0, column=1, columnspan=7, sticky="ew", padx=8, pady=6)
        for idx, schema_name in enumerate(fs.SCHEMA_NAMES):
            ttk.Checkbutton(
                suite_schema_frame,
                text=schema_name,
                variable=self.train_suite_schema_vars[schema_name],
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=(0, 14), pady=2)
        ttk.Label(multi_box, text="Target model").grid(row=1, column=0, sticky="nw", padx=8, pady=6)
        suite_variant_frame = ttk.Frame(multi_box)
        suite_variant_frame.grid(row=1, column=1, columnspan=7, sticky="ew", padx=8, pady=6)
        for idx, variant in enumerate(gm.VARIANT_NAMES):
            ttk.Checkbutton(
                suite_variant_frame,
                text=f"gru_{variant}",
                variable=self.train_suite_variant_vars[variant],
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=(0, 14), pady=2)
        ttk.Label(multi_box, text="Suite").grid(row=2, column=0, sticky="nw", padx=8, pady=6)
        suite_frame = ttk.Frame(multi_box)
        suite_frame.grid(row=2, column=1, columnspan=5, sticky="ew", padx=8, pady=6)
        for idx, suite_name in enumerate(("main", "chunk10", "threshold", "boosted")):
            ttk.Checkbutton(
                suite_frame,
                text=suite_name,
                variable=self.train_suite_suite_vars[suite_name],
            ).grid(row=0, column=idx, sticky="w", padx=(0, 14), pady=2)
        self.btn_train_suite_one = ttk.Button(multi_box, text="Train Suite Dipilih", command=self.train_suite_selected)
        self.btn_train_suite_one.grid(row=2, column=6, sticky="ew", padx=8, pady=6)
        self.btn_train_suite_all = ttk.Button(multi_box, text="Centang Semua + Train", command=self.train_suite_all)
        self.btn_train_suite_all.grid(row=2, column=7, sticky="ew", padx=8, pady=6)

        eval_box = ttk.LabelFrame(multi_tab, text="Evaluate Test Metrics")
        eval_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            eval_box.columnconfigure(idx, weight=1)
        ttk.Label(eval_box, text="Schema").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(eval_box, textvariable=self.eval_schema_var, values=["all", "full", "face", *fs.SCHEMA_NAMES], state="readonly", width=12).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(eval_box, text="GRU").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Combobox(eval_box, textvariable=self.eval_variant_var, values=["all", *gm.VARIANT_NAMES], state="readonly", width=10).grid(row=0, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(eval_box, text="Suite").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            eval_box,
            textvariable=self.eval_suite_var,
            values=["all", *(value for _display, value in LIVE_ROUTE_CHOICES)],
            state="readonly",
            width=14,
        ).grid(row=0, column=5, sticky="ew", padx=8, pady=6)
        ttk.Label(eval_box, text="Split").grid(row=0, column=6, sticky="w", padx=8, pady=6)
        ttk.Combobox(eval_box, textvariable=self.eval_split_var, values=["test", "val", "train"], state="readonly", width=8).grid(row=0, column=7, sticky="ew", padx=8, pady=6)
        self.btn_eval_test = ttk.Button(eval_box, text="Evaluate Test", command=self.evaluate_test)
        self.btn_eval_test.grid(row=1, column=6, columnspan=2, sticky="ew", padx=8, pady=6)

        buttons = ttk.Frame(training_tab)
        buttons.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        buttons.columnconfigure((0, 1, 2, 3), weight=1)

        self.btn_train_one = ttk.Button(buttons, text="Train Varian", command=self.train_selected)
        self.btn_train_one.grid(row=0, column=0, sticky="ew", padx=4)
        self.btn_train_all = ttk.Button(buttons, text="Train Semua", command=self.train_all)
        self.btn_train_all.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(buttons, text="Evaluator", command=self.open_evaluator).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(buttons, text="Refresh", command=self.refresh_status).grid(row=0, column=3, sticky="ew", padx=4)

        live_box = ttk.LabelFrame(live_tab, text="Live Inference")
        live_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            live_box.columnconfigure(idx, weight=1)

        ttk.Label(live_box, text="Model/schema").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.live_schema_combo = ttk.Combobox(
            live_box,
            textvariable=self.live_schema_var,
            values=list(LIVE_SCHEMA_CHOICES),
            state="readonly",
            width=12,
        )
        self.live_schema_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        self.live_schema_combo.bind("<<ComboboxSelected>>", lambda _event: self._build_live_variant_options())

        ttk.Label(live_box, text="GRU variant").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        self.live_variant_combo = ttk.Combobox(
            live_box,
            textvariable=self.live_variant_var,
            values=self.live_variant_options,
            state="readonly",
            width=14,
        )
        self.live_variant_combo.grid(row=0, column=3, sticky="ew", padx=8, pady=6)

        ttk.Label(live_box, text="Profile").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            live_box,
            textvariable=self.mode_var,
            values=sorted(live_gru_fast.LIVE_PROFILES),
            state="readonly",
            width=16,
        ).grid(row=0, column=5, sticky="ew", padx=8, pady=6)

        ttk.Label(live_box, text="Device").grid(row=0, column=6, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            live_box,
            textvariable=self.live_device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=8,
        ).grid(row=0, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(live_box, text="Route").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            live_box,
            textvariable=self.live_route_var,
            values=[display for display, _value in LIVE_ROUTE_CHOICES],
            state="readonly",
            width=24,
        ).grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(live_box, text="Stream").grid(row=1, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(live_box, textvariable=self.live_stream_workers_var, width=6).grid(row=1, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(live_box, text="MediaPipe").grid(row=1, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(live_box, textvariable=self.live_mp_workers_var, width=6).grid(row=1, column=5, sticky="ew", padx=8, pady=6)
        ttk.Label(live_box, text="Infer").grid(row=1, column=6, sticky="w", padx=8, pady=6)
        ttk.Entry(live_box, textvariable=self.live_inference_workers_var, width=6).grid(row=1, column=7, sticky="ew", padx=8, pady=6)
        ttk.Label(live_box, text="Threshold").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            live_box,
            textvariable=self.live_threshold_var,
            values=["default", "0.25", "0.50", "0.65", "0.80"],
            width=10,
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(live_box, text="Model data").grid(row=2, column=2, sticky="w", padx=8, pady=6)
        self.live_model_data_combo = ttk.Combobox(
            live_box,
            textvariable=self.live_model_data_var,
            values=list(MODEL_DATA_CHOICES),
            state="readonly",
            width=18,
        )
        self.live_model_data_combo.grid(row=2, column=3, sticky="ew", padx=8, pady=6)
        self.live_model_data_combo.bind("<<ComboboxSelected>>", lambda _event: self._build_live_variant_options())
        ttk.Label(live_box, text="MP Method").grid(row=2, column=4, sticky="w", padx=8, pady=6)
        self.live_mp_method_combo = ttk.Combobox(
            live_box,
            textvariable=self.live_mp_method_var,
            values=list(LIVE_MP_METHOD_CHOICES),
            state="readonly",
            width=18,
        )
        self.live_mp_method_combo.grid(row=2, column=5, sticky="ew", padx=8, pady=6)
        self.btn_live = ttk.Button(live_box, text="Start Live Test", command=self.toggle_live)
        self.btn_live.grid(row=2, column=6, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Checkbutton(live_box, text="Auto specialist", variable=self.live_specialist_enabled_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=6)
        ttk.Label(live_box, text="Specialist (all=semua)").grid(row=3, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(live_box, textvariable=self.live_specialist_name_var, width=14).grid(row=3, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(
            live_box,
            text=LIVE_MEDIAPIPE_HELP_TEXT,
            wraplength=900,
        ).grid(row=4, column=0, columnspan=8, sticky="ew", padx=8, pady=(0, 6))

        live_plus_box = ttk.LabelFrame(live_plus_tab, text="LiveTest Plus")
        live_plus_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            live_plus_box.columnconfigure(idx, weight=1)

        ttk.Label(live_plus_box, text="Model/schema").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.live_plus_schema_combo = ttk.Combobox(
            live_plus_box,
            textvariable=self.live_schema_var,
            values=list(LIVE_SCHEMA_CHOICES),
            state="readonly",
            width=12,
        )
        self.live_plus_schema_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        self.live_plus_schema_combo.bind("<<ComboboxSelected>>", lambda _event: self._build_live_variant_options())

        ttk.Label(live_plus_box, text="GRU variant").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        self.live_plus_variant_combo = ttk.Combobox(
            live_plus_box,
            textvariable=self.live_variant_var,
            values=self.live_variant_options,
            state="readonly",
            width=14,
        )
        self.live_plus_variant_combo.grid(row=0, column=3, sticky="ew", padx=8, pady=6)

        ttk.Label(live_plus_box, text="Profile").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            live_plus_box,
            textvariable=self.mode_var,
            values=sorted(live_gru_fast.LIVE_PROFILES),
            state="readonly",
            width=16,
        ).grid(row=0, column=5, sticky="ew", padx=8, pady=6)

        ttk.Label(live_plus_box, text="Device").grid(row=0, column=6, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            live_plus_box,
            textvariable=self.live_device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=8,
        ).grid(row=0, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(live_plus_box, text="Route").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            live_plus_box,
            textvariable=self.live_route_var,
            values=[display for display, _value in LIVE_ROUTE_CHOICES],
            state="readonly",
            width=24,
        ).grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(live_plus_box, text="Stream").grid(row=1, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(live_plus_box, textvariable=self.live_stream_workers_var, width=6).grid(row=1, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(live_plus_box, text="MediaPipe").grid(row=1, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(live_plus_box, textvariable=self.live_mp_workers_var, width=6).grid(row=1, column=5, sticky="ew", padx=8, pady=6)
        ttk.Label(live_plus_box, text="Infer").grid(row=1, column=6, sticky="w", padx=8, pady=6)
        ttk.Entry(live_plus_box, textvariable=self.live_inference_workers_var, width=6).grid(row=1, column=7, sticky="ew", padx=8, pady=6)

        ttk.Checkbutton(live_plus_box, text="Use local LLM", variable=self.live_plus_use_llm_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(live_plus_box, text="Allow word fix", variable=self.live_plus_allow_word_fix_var).grid(row=2, column=2, columnspan=2, sticky="w", padx=8, pady=6)
        ttk.Label(live_plus_box, text="Ollama model").grid(row=2, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(live_plus_box, textvariable=self.live_plus_ollama_model_var, width=16).grid(row=2, column=5, sticky="ew", padx=8, pady=6)
        ttk.Label(live_plus_box, text="Threshold").grid(row=2, column=6, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            live_plus_box,
            textvariable=self.live_threshold_var,
            values=["default", "0.25", "0.50", "0.65", "0.80"],
            width=10,
        ).grid(row=2, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(live_plus_box, text="Speaker").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        self.live_plus_audio_combo = ttk.Combobox(
            live_plus_box,
            textvariable=self.live_plus_audio_sink_var,
            values=["auto/default"],
            state="readonly",
            width=28,
        )
        self.live_plus_audio_combo.grid(row=3, column=1, columnspan=3, sticky="ew", padx=8, pady=6)
        ttk.Button(live_plus_box, text="Refresh Speaker", command=self.refresh_live_plus_audio_sinks).grid(row=3, column=4, sticky="ew", padx=8, pady=6)
        ttk.Button(live_plus_box, text="Clear Buffer", command=self.clear_live_plus_buffer).grid(row=3, column=5, sticky="ew", padx=8, pady=6)
        self.btn_live_plus = ttk.Button(live_plus_box, text="Start LiveTest Plus", command=self.toggle_live_plus)
        self.btn_live_plus.grid(row=3, column=6, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Label(live_plus_box, text="Model data").grid(row=4, column=0, sticky="w", padx=8, pady=6)
        self.live_plus_model_data_combo = ttk.Combobox(
            live_plus_box,
            textvariable=self.live_model_data_var,
            values=list(MODEL_DATA_CHOICES),
            state="readonly",
            width=18,
        )
        self.live_plus_model_data_combo.grid(row=4, column=1, columnspan=3, sticky="ew", padx=8, pady=6)
        self.live_plus_model_data_combo.bind("<<ComboboxSelected>>", lambda _event: self._build_live_variant_options())
        ttk.Label(live_plus_box, text="MP Method").grid(row=4, column=4, sticky="w", padx=8, pady=6)
        self.live_plus_mp_method_combo = ttk.Combobox(
            live_plus_box,
            textvariable=self.live_mp_method_var,
            values=list(LIVE_MP_METHOD_CHOICES),
            state="readonly",
            width=18,
        )
        self.live_plus_mp_method_combo.grid(row=4, column=5, sticky="ew", padx=8, pady=6)
        ttk.Checkbutton(live_plus_box, text="Auto specialist", variable=self.live_specialist_enabled_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=8, pady=6)
        ttk.Label(live_plus_box, text="Specialist (all=semua)").grid(row=5, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(live_plus_box, textvariable=self.live_specialist_name_var, width=14).grid(row=5, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(
            live_plus_box,
            text=LIVE_MEDIAPIPE_HELP_TEXT,
            wraplength=900,
        ).grid(row=6, column=0, columnspan=8, sticky="ew", padx=8, pady=(0, 6))

        plus_output = ttk.LabelFrame(live_plus_tab, text="Output")
        plus_output.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        plus_output.columnconfigure(0, weight=1)
        ttk.Label(plus_output, textvariable=self.live_plus_status_var, foreground="#0d6efd").grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(plus_output, textvariable=self.live_plus_buffer_var).grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(plus_output, textvariable=self.live_plus_output_var).grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(plus_output, textvariable=self.live_plus_llm_var).grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(plus_output, textvariable=self.live_plus_tts_var).grid(row=4, column=0, sticky="ew", padx=8, pady=6)
        self.refresh_live_plus_audio_sinks()
        self._build_reinforcement_tab(reinforcement_tab)
        self._build_tts_profile_tab(tts_profile_tab)

        body = ttk.Frame(logs_tab)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, textvariable=self.status_var, foreground="#333333").grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.status_text = tk.Text(body, height=15, wrap="word")
        self.status_text.grid(row=1, column=0, sticky="nsew")
        self.status_text.configure(state="disabled")

        ttk.Label(live_tab, textvariable=self.live_status_var, foreground="#0d6efd").grid(row=1, column=0, sticky="ew", pady=(0, 10))

    def _build_reinforcement_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        session_box = ttk.LabelFrame(tab, text="Reinforcement Learning")
        session_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(8):
            session_box.columnconfigure(idx, weight=1)

        schema_values = list(dict.fromkeys(["smart", "khukuh", "adi", "smart_face", *fs.SCHEMA_NAMES]))
        ttk.Label(session_box, text="Schema").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(session_box, textvariable=self.rl_schema_var, values=schema_values, state="readonly", width=12).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(session_box, text="GRU variant").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Combobox(session_box, textvariable=self.rl_variant_var, values=["auto", *gm.VARIANT_NAMES], state="readonly", width=10).grid(row=0, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(session_box, text="Suite/route").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            session_box,
            textvariable=self.rl_route_var,
            values=[display for display, _value in LIVE_ROUTE_CHOICES],
            state="readonly",
            width=22,
        ).grid(row=0, column=5, sticky="ew", padx=8, pady=6)
        ttk.Label(session_box, text="Device").grid(row=0, column=6, sticky="w", padx=8, pady=6)
        ttk.Combobox(session_box, textvariable=self.rl_device_var, values=["auto", "cpu", "cuda"], state="readonly", width=8).grid(row=0, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(session_box, text="Profile/mode").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(session_box, textvariable=self.rl_profile_var, values=sorted(live_gru_fast.LIVE_PROFILES), state="readonly", width=16).grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(session_box, text="Camera").grid(row=1, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(session_box, textvariable=self.rl_camera_var, width=8).grid(row=1, column=3, sticky="ew", padx=8, pady=6)
        ttk.Label(session_box, text="LR").grid(row=1, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(session_box, textvariable=self.rl_lr_var, width=10).grid(row=1, column=5, sticky="ew", padx=8, pady=6)
        ttk.Label(session_box, text="Steps/koreksi").grid(row=1, column=6, sticky="w", padx=8, pady=6)
        ttk.Entry(session_box, textvariable=self.rl_steps_var, width=8).grid(row=1, column=7, sticky="ew", padx=8, pady=6)

        self.btn_rl_start = ttk.Button(session_box, text="Start Session", command=self.start_reinforcement_session)
        self.btn_rl_start.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=6)
        self.btn_rl_finish = ttk.Button(session_box, text="Stop/Finish Session", command=self.finish_reinforcement_session, state="disabled")
        self.btn_rl_finish.grid(row=2, column=2, columnspan=2, sticky="ew", padx=8, pady=6)
        self.btn_rl_correct = ttk.Button(session_box, text="Correct", command=self.mark_reinforcement_correct, state="disabled")
        self.btn_rl_correct.grid(row=2, column=4, sticky="ew", padx=8, pady=6)
        self.btn_rl_wrong = ttk.Button(session_box, text="Wrong -> Apply Label", command=self.mark_reinforcement_wrong, state="disabled")
        self.btn_rl_wrong.grid(row=2, column=5, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Button(session_box, text="Refresh Labels", command=self.refresh_reinforcement_labels).grid(row=2, column=7, sticky="ew", padx=8, pady=6)

        ttk.Label(session_box, text="Corrected label").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        self.rl_label_combo = ttk.Combobox(session_box, textvariable=self.rl_correct_label_var, values=[], state="readonly", width=18)
        self.rl_label_combo.grid(row=3, column=1, columnspan=3, sticky="ew", padx=8, pady=6)
        ttk.Label(session_box, text="Model data").grid(row=3, column=4, sticky="w", padx=8, pady=6)
        self.rl_model_data_combo = ttk.Combobox(
            session_box,
            textvariable=self.rl_model_data_var,
            values=list(MODEL_DATA_CHOICES),
            state="readonly",
            width=18,
        )
        self.rl_model_data_combo.grid(row=3, column=5, columnspan=3, sticky="ew", padx=8, pady=6)
        self.rl_model_data_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_reinforcement_labels())

        status_box = ttk.LabelFrame(tab, text="Status")
        status_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        status_box.columnconfigure(0, weight=1)
        ttk.Label(status_box, textvariable=self.rl_status_var, foreground="#0d6efd").grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(status_box, textvariable=self.rl_prediction_var).grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(status_box, textvariable=self.rl_checkpoint_var).grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(status_box, textvariable=self.rl_backup_var).grid(row=3, column=0, sticky="ew", padx=8, pady=6)

    def _build_tts_profile_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        profile_box = ttk.LabelFrame(tab, text="TTS Profile")
        profile_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(6):
            profile_box.columnconfigure(idx, weight=1)

        ttk.Label(profile_box, text="Gender").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.tts_gender_combo = ttk.Combobox(
            profile_box,
            textvariable=self.tts_gender_var,
            values=list(tts_rt.GENDER_DISPLAY_OPTIONS),
            state="readonly",
            width=10,
        )
        self.tts_gender_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        self.tts_gender_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_tts_profiles())

        ttk.Label(profile_box, text="Usia").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        self.tts_demografi_combo = ttk.Combobox(
            profile_box,
            textvariable=self.tts_demografi_var,
            values=list(tts_rt.DEMOGRAFI_DISPLAY_OPTIONS),
            state="readonly",
            width=12,
        )
        self.tts_demografi_combo.grid(row=0, column=3, sticky="ew", padx=8, pady=6)
        self.tts_demografi_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_tts_profiles())

        ttk.Label(profile_box, text="Profile/Variasi").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.tts_profile_combo = ttk.Combobox(
            profile_box,
            textvariable=self.tts_profile_var,
            values=[],
            state="readonly",
            width=28,
        )
        self.tts_profile_combo.grid(row=1, column=1, columnspan=3, sticky="ew", padx=8, pady=6)
        self.tts_profile_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_tts_profile_detail())

        ttk.Label(profile_box, text="Device").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        self.tts_device_combo = ttk.Combobox(
            profile_box,
            textvariable=self.tts_device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=8,
        )
        self.tts_device_combo.grid(row=0, column=5, sticky="ew", padx=8, pady=6)
        self.tts_device_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_tts_profile_detail())

        ttk.Label(profile_box, text="Player").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            profile_box,
            textvariable=self.tts_player_var,
            values=["auto", "paplay", "aplay", "ffplay"],
            state="readonly",
            width=10,
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=6)
        ttk.Checkbutton(
            profile_box,
            text="Use loaded TTS for LiveTest Plus",
            variable=self.tts_use_loaded_var,
        ).grid(row=2, column=2, columnspan=3, sticky="w", padx=8, pady=6)

        ttk.Button(profile_box, text="Refresh Profiles", command=self.refresh_tts_profiles).grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        self.btn_tts_load = ttk.Button(profile_box, text="Preload + Warmup", command=self.load_tts_profile)
        self.btn_tts_load.grid(row=3, column=1, sticky="ew", padx=8, pady=6)
        self.btn_tts_unload = ttk.Button(profile_box, text="Unload", command=self.unload_tts_profile)
        self.btn_tts_unload.grid(row=3, column=2, sticky="ew", padx=8, pady=6)
        ttk.Button(profile_box, text="Test Speak", command=self.test_tts_profile).grid(row=3, column=3, sticky="ew", padx=8, pady=6)

        ttk.Label(profile_box, text="Test text").grid(row=4, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(profile_box, textvariable=self.tts_test_text_var).grid(row=4, column=1, columnspan=5, sticky="ew", padx=8, pady=6)

        status_box = ttk.LabelFrame(tab, text="Status")
        status_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        status_box.columnconfigure(0, weight=1)
        ttk.Label(status_box, textvariable=self.tts_status_var, foreground="#0d6efd").grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(status_box, textvariable=self.tts_detail_var).grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(
            status_box,
            text=(
                "Start Live Test/LiveTest Plus akan auto-load TTS. Preload opsional kalau ingin memanaskan model sebelum start; Unload melepas runtime."
            ),
            wraplength=860,
        ).grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        self.refresh_tts_profiles()

    def refresh_tts_profiles(self) -> None:
        try:
            gender = self.tts_gender_var.get()
            demografi = self.tts_demografi_var.get()
            profiles = tts_rt.profiles_for_picker(gender, demografi)
        except Exception as exc:
            self.tts_status_var.set(f"TTS: profile gagal dibaca | {exc}")
            profiles = []
        if hasattr(self, "tts_profile_combo"):
            self.tts_profile_combo.configure(values=profiles)
        current = self.tts_profile_var.get()
        if profiles and current not in profiles:
            self.tts_profile_var.set(profiles[0])
        elif not profiles:
            self.tts_profile_var.set("")
            self.tts_status_var.set(
                f"TTS: tidak ada profile untuk {self.tts_gender_var.get()} / {self.tts_demografi_var.get()}"
            )
        self._update_tts_profile_detail()

    def _update_tts_profile_detail(self) -> None:
        profile_name = self.tts_profile_var.get().strip()
        if not profile_name:
            self.tts_detail_var.set("Profile: -")
            return
        try:
            profile = tts_rt.profile_detail(profile_name)
            self.tts_gender_var.set(tts_rt.display_gender(str(profile.get("gender", ""))))
            self.tts_demografi_var.set(tts_rt.display_demografi(str(profile.get("demografi", ""))))
            self.tts_detail_var.set(
                f"Profile: {profile_name} | speaker {profile.get('base_speaker', '-')} | "
                f"pitch {profile.get('pitch_semitones', '-')} | speed {profile.get('speed', '-')}"
            )
            runtime = getattr(self, "tts_runtime", None)
            selected_device = self.tts_device_var.get().strip() or "auto"
            if runtime is not None and runtime.is_loaded and (
                runtime.profile_name != profile_name or runtime.device != selected_device
            ):
                self.tts_status_var.set(
                    f"TTS: loaded {runtime.profile_name} ({runtime.device}); pilihan baru "
                    f"{profile_name} ({selected_device}) dipakai saat Start/Preload berikutnya."
                )
        except Exception as exc:
            self.tts_detail_var.set(f"Profile: {profile_name} | detail gagal: {exc}")

    def _tts_is_loaded(self) -> bool:
        runtime = getattr(self, "tts_runtime", None)
        return bool(runtime is not None and runtime.is_loaded)

    def _selected_tts_profile_name(self) -> str:
        profile_name = self.tts_profile_var.get().strip()
        if not profile_name:
            self.refresh_tts_profiles()
            profile_name = self.tts_profile_var.get().strip()
        if not profile_name:
            raise RuntimeError("Profile TTS belum tersedia. Cek tab TTS Profile dan jalankan init/download model TTS.")
        return profile_name

    def _tts_runtime_matches(self, profile_name: str, device: str) -> bool:
        runtime = getattr(self, "tts_runtime", None)
        return bool(
            runtime is not None
            and runtime.is_loaded
            and runtime.profile_name == profile_name
            and runtime.device == device
        )

    def ensure_tts_loaded_for_live(self, context: str, on_ready) -> bool:
        """Load/warm TTS before starting a live worker."""
        if self.tts_load_thread is not None and self.tts_load_thread.is_alive():
            self.tts_status_var.set("TTS: load masih berjalan")
            return False
        try:
            profile_name = self._selected_tts_profile_name()
            device = self.tts_device_var.get().strip() or "auto"
        except Exception as exc:
            messagebox.showerror("TTS Profile", str(exc))
            return False

        if self._tts_runtime_matches(profile_name, device):
            self.tts_status_var.set(f"TTS: reuse loaded {profile_name} ({device})")
            on_ready()
            return True

        self.unload_tts_profile(silent=True)
        if context == "plus":
            self.btn_live_plus.configure(text="Loading TTS...", state="disabled")
            self.live_plus_status_var.set(f"LiveTest Plus: loading TTS {profile_name}")
        else:
            self.btn_live.configure(text="Loading TTS...", state="disabled")
            self.live_status_var.set(f"Live: loading TTS {profile_name}")
        self.tts_status_var.set(f"TTS: loading {profile_name} ({device}) for live...")

        def task() -> None:
            ok = True
            error = ""
            elapsed = 0.0
            runtime: tts_rt.LoadedTTSProfile | None = None
            try:
                runtime = tts_rt.LoadedTTSProfile(profile_name, device=device)
                elapsed = runtime.load(warmup=True)
            except Exception as exc:
                ok = False
                error = str(exc)
                if runtime is not None:
                    runtime.unload()
                    runtime = None
            self.root.after(0, lambda: self._finish_tts_load_for_live(context, ok, profile_name, device, elapsed, runtime, error, on_ready))

        self.tts_load_thread = threading.Thread(target=task, daemon=True)
        self.tts_load_thread.start()
        return True

    def _finish_tts_load_for_live(
        self,
        context: str,
        ok: bool,
        profile_name: str,
        device: str,
        elapsed: float,
        runtime: tts_rt.LoadedTTSProfile | None,
        error: str,
        on_ready,
    ) -> None:
        if ok and runtime is not None:
            self.tts_runtime = runtime
            self.tts_status_var.set(f"TTS: loaded {profile_name} ({device}) for live in {elapsed:.2f}s")
            try:
                on_ready()
            except Exception as exc:
                self.release_tts_after_live(context)
                self._reset_live_start_button(context)
                messagebox.showerror("Live Test" if context == "live" else "LiveTest Plus", str(exc))
            return

        self.tts_runtime = None
        self.tts_status_var.set(f"TTS: load gagal | {error}")
        self._reset_live_start_button(context)
        messagebox.showerror("TTS Profile", error)

    def _reset_live_start_button(self, context: str) -> None:
        if context == "plus":
            self.btn_live_plus.configure(text="Start LiveTest Plus", state="normal")
            self.live_plus_status_var.set("LiveTest Plus: idle")
        else:
            self.btn_live.configure(text="Start Live Test", state="normal")
            self.live_status_var.set("Live: idle")

    def release_tts_after_live(self, context: str) -> None:
        if self._tts_is_loaded():
            self.unload_tts_profile(silent=True)
            self.tts_status_var.set(f"TTS: unloaded after {'LiveTest Plus' if context == 'plus' else 'Live Test'}")

    def load_tts_profile(self) -> None:
        profile_name = self.tts_profile_var.get().strip()
        if not profile_name:
            messagebox.showwarning("TTS Profile", "Pilih profile TTS dulu.")
            return
        if self.tts_load_thread is not None and self.tts_load_thread.is_alive():
            self.tts_status_var.set("TTS: load masih berjalan")
            return
        self.unload_tts_profile(silent=True)
        device = self.tts_device_var.get().strip() or "auto"
        self.tts_status_var.set(f"TTS: loading {profile_name} ({device})...")
        self.btn_tts_load.configure(state="disabled")
        self.btn_tts_unload.configure(state="disabled")

        def task() -> None:
            ok = True
            error = ""
            elapsed = 0.0
            runtime: tts_rt.LoadedTTSProfile | None = None
            try:
                runtime = tts_rt.LoadedTTSProfile(profile_name, device=device)
                elapsed = runtime.load(warmup=True)
            except Exception as exc:
                ok = False
                error = str(exc)
                if runtime is not None:
                    runtime.unload()
                    runtime = None
            self.root.after(0, lambda: self._finish_tts_load(ok, profile_name, device, elapsed, runtime, error))

        self.tts_load_thread = threading.Thread(target=task, daemon=True)
        self.tts_load_thread.start()

    def _finish_tts_load(
        self,
        ok: bool,
        profile_name: str,
        device: str,
        elapsed: float,
        runtime: tts_rt.LoadedTTSProfile | None,
        error: str,
    ) -> None:
        self.btn_tts_load.configure(state="normal")
        self.btn_tts_unload.configure(state="normal")
        if ok and runtime is not None:
            self.tts_runtime = runtime
            self.tts_status_var.set(f"TTS: loaded {profile_name} ({device}) in {elapsed:.2f}s")
        else:
            self.tts_runtime = None
            self.tts_status_var.set(f"TTS: load gagal | {error}")
            messagebox.showerror("TTS Profile", error)

    def unload_tts_profile(self, silent: bool = False) -> None:
        runtime = getattr(self, "tts_runtime", None)
        if runtime is not None:
            runtime.unload()
        self.tts_runtime = None
        if not silent:
            self.tts_status_var.set("TTS: unloaded (RAM/GPU dilepas)")

    def test_tts_profile(self) -> None:
        if not self._tts_is_loaded():
            messagebox.showwarning("TTS Profile", "Preload + Warmup profile dulu supaya test cepat.")
            return
        text = self.tts_test_text_var.get().strip()
        if not text:
            messagebox.showwarning("TTS Profile", "Isi test text dulu.")
            return
        self.tts_status_var.set("TTS: test speaking...")

        def task() -> None:
            ok = True
            error = ""
            result = None
            try:
                result = self.tts_runtime.speak(text, play=True, player=self.tts_player_var.get().strip() or "auto")
            except Exception as exc:
                ok = False
                error = str(exc)
            self.root.after(0, lambda: self._finish_tts_test(ok, result, error))

        threading.Thread(target=task, daemon=True).start()

    def _finish_tts_test(self, ok: bool, result: tts_rt.TTSGenerateResult | None, error: str) -> None:
        if ok and result is not None:
            total = result.timing_sec.get("total", 0.0)
            self.tts_status_var.set(f"TTS: test done {total:.2f}s | {result.final_wav_path}")
        else:
            self.tts_status_var.set(f"TTS: test gagal | {error}")
            messagebox.showerror("TTS Profile", error)

    def _set_text(self, content: str) -> None:
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", content)
        self.status_text.configure(state="disabled")

    def _append_text(self, content: str) -> None:
        self.status_text.configure(state="normal")
        self.status_text.insert("end", content)
        self.status_text.see("end")
        self.status_text.configure(state="disabled")

    def _cli_path(self) -> str:
        return str(gm.ROOT_DIR / "src" / "bisindo_cli.py")

    def selected_dataset_dir(self) -> str:
        return self.dataset_dir_var.get().strip() or str(gm.DATASET_DIR)

    def selected_full_root(self) -> str:
        return str(Path(self.selected_dataset_dir()) / "full_features")

    def _overwrite_existing_args(self) -> list[str]:
        return ["--overwrite-existing"] if bool(self.overwrite_existing_var.get()) else []

    def _reset_overwrite_existing(self) -> None:
        self.overwrite_existing_var.set(False)

    def selected_live_schema(self) -> str:
        return resolve_live_schema_name(self.live_schema_var.get())

    def selected_live_variant_value(self) -> str:
        value = self.live_variant_display_to_value.get(self.live_variant_var.get())
        if value:
            return value
        raw = self.live_variant_var.get().split(" ", 1)[0]
        if raw == "auto":
            return "auto"
        return raw

    def selected_live_route_value(self) -> str:
        return resolve_live_route_name(self.live_route_var.get())

    def selected_live_model_data_mode(self) -> str:
        return model_data_mode_from_display(self.live_model_data_var.get())

    def selected_rl_model_data_mode(self) -> str:
        return model_data_mode_from_display(self.rl_model_data_var.get())

    def selected_train_schemas(self) -> tuple[str, ...]:
        schema_vars = getattr(self, "train_schema_vars", None)
        if schema_vars:
            return tuple(name for name in fs.SCHEMA_NAMES if schema_vars.get(name) is not None and schema_vars[name].get())
        schema_var = getattr(self, "schema_var", None)
        if schema_var is not None:
            return fs.expand_schema_names(schema_var.get())
        return (fs.DEFAULT_SCHEMA,)

    def selected_extract_full_schemas(self) -> tuple[str, ...]:
        schema_vars = getattr(self, "extract_full_schema_vars", None)
        if schema_vars:
            return tuple(name for name in fs.SCHEMA_NAMES if schema_vars.get(name) is not None and schema_vars[name].get())
        schema_var = getattr(self, "extract_full_schema_var", None)
        if schema_var is not None:
            return fs.expand_schema_names(schema_var.get())
        return fs.FULL_SCHEMA_NAMES

    def selected_extract_full_vocabs(self) -> tuple[str, ...]:
        all_var = getattr(self, "extract_full_all_vocab_var", None)
        if all_var is not None and all_var.get():
            return ()
        vocab_vars = getattr(self, "extract_full_vocab_vars", {})
        selected = [
            _clean_extract_label(label)
            for label, var in vocab_vars.items()
            if var.get() and _clean_extract_label(label)
        ]
        out: list[str] = []
        for label in selected:
            if label not in out:
                out.append(label)
        return tuple(out)

    def _set_extract_full_vocab_items(self, labels: Iterable[str]) -> None:
        old_values = {label: var.get() for label, var in getattr(self, "extract_full_vocab_vars", {}).items()}
        all_var = getattr(self, "extract_full_all_vocab_var", None)
        all_selected = True if all_var is None else bool(all_var.get())
        self.extract_full_vocab_items = []
        self.extract_full_vocab_vars = {}
        frame = getattr(self, "extract_full_vocab_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        for idx, raw_label in enumerate(labels):
            label = _clean_extract_label(str(raw_label))
            if not label or label in self.extract_full_vocab_vars:
                continue
            checked = all_selected or bool(old_values.get(label, False))
            self.extract_full_vocab_items.append(label)
            self.extract_full_vocab_vars[label] = tk.BooleanVar(value=checked)
            ttk.Checkbutton(
                self.extract_full_vocab_frame,
                text=label,
                variable=self.extract_full_vocab_vars[label],
                command=self._sync_extract_full_all_vocab,
            ).grid(row=idx // 8, column=idx % 8, sticky="w", padx=(0, 12), pady=2)
        if not self.extract_full_vocab_items:
            ttk.Label(self.extract_full_vocab_frame, text="Belum ada vocab full dataset terbaca.").grid(row=0, column=0, sticky="w")

    def refresh_extract_full_vocab_list(self) -> None:
        source = self.extract_full_source_var.get().strip()
        if not source:
            messagebox.showerror("Extract Full", "Source wajib diisi.")
            return
        try:
            labels = discover_extract_full_vocabs(source)
            self._set_extract_full_vocab_items(labels)
            self.status_var.set(f"Vocab extract full dimuat: {len(labels)} label.")
        except Exception as exc:
            self.status_var.set(f"Gagal memuat vocab extract full: {exc}")
            messagebox.showerror("Extract Full", str(exc))

    def _toggle_all_extract_full_vocab(self) -> None:
        value = bool(self.extract_full_all_vocab_var.get())
        for var in self.extract_full_vocab_vars.values():
            var.set(value)

    def _sync_extract_full_all_vocab(self) -> None:
        all_var = getattr(self, "extract_full_all_vocab_var", None)
        if all_var is None:
            return
        vocab_vars = getattr(self, "extract_full_vocab_vars", {})
        if not vocab_vars:
            all_var.set(True)
            return
        all_var.set(all(var.get() for var in vocab_vars.values()))

    def selected_train_variants(self) -> tuple[str, ...]:
        return tuple(variant for variant, var in self.train_variant_vars.items() if var.get())

    def selected_specialist_schemas(self) -> tuple[str, ...]:
        return tuple(name for name in fs.SCHEMA_NAMES if self.specialist_schema_vars.get(name) is not None and self.specialist_schema_vars[name].get())

    def selected_specialist_variants(self) -> tuple[str, ...]:
        return tuple(variant for variant, var in self.specialist_variant_vars.items() if var.get())

    def selected_specialist_labels(self) -> tuple[str, ...]:
        selected = [label for label, var in self.specialist_vocab_vars.items() if var.get()]
        manual = self.specialist_vocab_var.get().strip()
        if manual:
            for part in manual.replace(";", ",").split(","):
                label = part.strip()
                if label and label not in selected:
                    selected.append(label)
        return gm.normalize_specialist_labels(selected)

    def selected_specialist_name(self, labels: Iterable[str] | str | None = None) -> str:
        label_values = labels if labels is not None else self.selected_specialist_labels()
        generated = gm.specialist_name_from_labels(label_values)
        current = self.specialist_name_var.get().strip()
        if not current or current == getattr(self, "_specialist_last_auto_name", ""):
            self.specialist_name_var.set(generated)
            if not _is_all_specialist_request(self.live_specialist_name_var.get()):
                self.live_specialist_name_var.set(generated)
            self._specialist_last_auto_name = generated
            return generated
        return gm.normalize_specialist_name(current, label_values)

    def _update_specialist_auto_name(self) -> None:
        try:
            labels = self.selected_specialist_labels()
        except Exception:
            return
        current = self.specialist_name_var.get().strip()
        generated = gm.specialist_name_from_labels(labels)
        if not current or current == getattr(self, "_specialist_last_auto_name", ""):
            self.specialist_name_var.set(generated)
            if not _is_all_specialist_request(self.live_specialist_name_var.get()):
                self.live_specialist_name_var.set(generated)
            self._specialist_last_auto_name = generated

    def _set_specialist_vocab_items(self, labels: Iterable[str]) -> None:
        current_selected = {label for label, var in self.specialist_vocab_vars.items() if var.get()}
        defaults = set(gm.normalize_specialist_labels(["d", "p"]))
        self.specialist_vocab_items = []
        self.specialist_vocab_vars = {}
        frame = getattr(self, "specialist_vocab_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        for idx, raw_label in enumerate(labels):
            label = gm.normalize_specialist_name(str(raw_label))
            if label in self.specialist_vocab_vars:
                continue
            checked = label in current_selected or (not current_selected and label in defaults)
            self.specialist_vocab_items.append(label)
            self.specialist_vocab_vars[label] = tk.BooleanVar(value=checked)
            ttk.Checkbutton(
                self.specialist_vocab_frame,
                text=label,
                variable=self.specialist_vocab_vars[label],
                command=self._update_specialist_auto_name,
            ).grid(row=idx // 8, column=idx % 8, sticky="w", padx=(0, 12), pady=2)
        self._update_specialist_auto_name()

    def refresh_specialist_vocab_list(self) -> None:
        schemas = self.selected_specialist_schemas() or (fs.DEFAULT_SCHEMA,)
        labels: set[str] = set()
        try:
            for schema_name in schemas:
                summary = gm.dataset_summary(dataset_dir=self.selected_dataset_dir(), schema=schema_name)
                labels.update(str(label) for label in summary.get("labels", []) if str(label).lower() not in gm.EXCLUDED_LABELS)
            if not labels:
                labels.update(["d", "p"])
            ordered = sorted(labels)
            self._set_specialist_vocab_items(ordered)
            self.status_var.set(f"Vocab spesialis dimuat: {len(ordered)} label.")
        except Exception as exc:
            self.status_var.set(f"Gagal memuat vocab spesialis: {exc}")
            messagebox.showerror("Vocab Khusus", str(exc))

    def selected_train_suite_schemas(self) -> tuple[str, ...]:
        schema_vars = getattr(self, "train_suite_schema_vars", None)
        if schema_vars:
            return tuple(name for name in fs.SCHEMA_NAMES if schema_vars.get(name) is not None and schema_vars[name].get())
        schema_var = getattr(self, "train_suite_schema_var", None)
        if schema_var is not None:
            return fs.expand_schema_names(schema_var.get())
        return fs.expand_schema_names("full")

    def selected_train_suite_variants(self) -> tuple[str, ...]:
        return tuple(variant for variant, var in self.train_suite_variant_vars.items() if var.get())

    def selected_train_suite_names(self) -> str:
        suites = [suite for suite, var in self.train_suite_suite_vars.items() if var.get()]
        return ",".join(suites)

    def selected_augment_splits(self) -> tuple[str, ...]:
        splits = tuple(split for split, var in self.augment_split_vars.items() if var.get())
        return splits or ("train",)

    def selected_augment_schemas(self) -> tuple[str, ...]:
        schema_vars = getattr(self, "augment_schema_vars", None)
        if schema_vars:
            return tuple(name for name in fs.SCHEMA_NAMES if schema_vars.get(name) is not None and schema_vars[name].get())
        schema_var = getattr(self, "augment_schema_var", None)
        if schema_var is not None:
            return fs.expand_schema_names(schema_var.get())
        return fs.FULL_SCHEMA_NAMES

    def selected_augment_vocabs(self) -> tuple[str, ...]:
        if self.augment_all_vocab_var.get():
            return ()
        selected = [label for label, var in self.augment_vocab_vars.items() if var.get()]
        manual = self.augment_vocab_var.get().strip()
        if manual:
            for part in manual.replace(";", ",").split(","):
                label = part.strip().replace(" ", "_").lower()
                if label and label not in selected:
                    selected.append(label)
        return tuple(selected)

    def selected_live_confidence_threshold(self) -> float | None:
        raw = str(self.live_threshold_var.get() or "").strip().lower()
        if raw in {"", "default"}:
            return None
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError("Threshold live harus 'default' atau angka 0.0 sampai 1.0.") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold live harus berada di rentang 0.0 sampai 1.0.")
        return value

    def selected_live_plus_audio_sink(self) -> str | None:
        return self.live_plus_audio_sink_display_to_name.get(self.live_plus_audio_sink_var.get())

    def refresh_live_plus_audio_sinks(self) -> None:
        previous = self.live_plus_audio_sink_var.get()
        self.live_plus_audio_sink_display_to_name, options = lp.sink_display_options(lp.list_audio_sinks())
        if hasattr(self, "live_plus_audio_combo"):
            self.live_plus_audio_combo.configure(values=options)
        self.live_plus_audio_sink_var.set(previous if previous in self.live_plus_audio_sink_display_to_name else options[0])

    def clear_live_plus_buffer(self) -> None:
        self.live_plus_buffer.reset()
        self._update_live_plus_buffer_text()
        self.live_plus_output_var.set("Output akhir: -")
        self.live_plus_llm_var.set("LLM: -")
        self.live_plus_tts_var.set("Suara/TTS: -")
        self.live_plus_status_var.set("LiveTest Plus: buffer cleared")

    def browse_dataset_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.selected_dataset_dir(), title="Pilih folder dataset_parquets")
        if selected:
            self.dataset_dir_var.set(selected)
            self.refresh_status()

    def _selected_gif_vocab(self) -> str | None:
        sel = self.gif_vocab_listbox.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if 0 <= idx < len(self.gif_vocab_items):
            return self.gif_vocab_items[idx]
        return None

    def load_gif_vocab_list(self) -> None:
        self.btn_load_gif_vocab.configure(state="disabled")
        self.status_var.set("Memuat list vocab untuk GIF...")
        args = self._gif_vocab_args()

        def task() -> None:
            try:
                proc = subprocess.run(args, capture_output=True, text=True)
                output = (proc.stdout or "") + (proc.stderr or "")
                self.root.after(0, lambda: self._gif_vocab_loaded(proc.returncode, output))
            except Exception as exc:
                self.root.after(0, lambda: self._gif_vocab_failed(str(exc)))

        threading.Thread(target=task, daemon=True).start()

    def _gif_vocab_args(self) -> list[str]:
        return [
            sys.executable,
            self._cli_path(),
            "gif",
            "vocab",
            "--schema",
            self.gif_schema_var.get(),
            "--dataset-dir",
            self.selected_dataset_dir(),
            "--plain",
        ]

    def _gif_vocab_loaded(self, code: int, output: str) -> None:
        self.btn_load_gif_vocab.configure(state="normal")
        self._set_text(output.strip() or f"gif vocab selesai dengan exit code {code}")
        self.gif_vocab_listbox.delete(0, "end")
        self.gif_vocab_items = []
        if code != 0:
            self.status_var.set("Gagal memuat list vocab GIF. Cek log.")
            messagebox.showwarning("GIF Dataset", f"gif vocab exit code {code}")
            return
        for line in output.splitlines():
            if not line.strip() or line.startswith("label\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            label, train, val, test, total = parts[:5]
            self.gif_vocab_items.append(label)
            self.gif_vocab_listbox.insert("end", f"{label}    train={train}  val={val}  test={test}  total={total}")
        if self.gif_vocab_items:
            self.gif_vocab_listbox.selection_set(0)
            self.status_var.set(f"List vocab GIF dimuat: {len(self.gif_vocab_items)} vocab.")
        else:
            self.status_var.set("Tidak ada vocab dataset untuk dibuat GIF.")

    def _gif_vocab_failed(self, message: str) -> None:
        self.btn_load_gif_vocab.configure(state="normal")
        self.status_var.set("Gagal memuat list vocab GIF.")
        self._set_text(message)
        messagebox.showerror("GIF Dataset", message)

    def _parse_gif_args(self) -> list[str]:
        vocab = self._selected_gif_vocab()
        if not vocab:
            raise ValueError("Pilih satu vocab dari list. Klik 'Muat List Vocab' dulu kalau list masih kosong.")
        try:
            limit = int(self.gif_limit_var.get())
            width = int(self.gif_width_var.get())
        except ValueError as exc:
            raise ValueError("Limit/split dan width harus angka.") from exc
        if limit < 0:
            raise ValueError("Limit/split minimal 0. Isi 0 kalau mau semua sample.")
        if width <= 0:
            raise ValueError("Width GIF harus > 0.")
        args = [
            sys.executable,
            self._cli_path(),
            "gif",
            "dataset",
            "--schema",
            self.gif_schema_var.get(),
            "--vocab",
            vocab,
            "--split",
            self.gif_split_var.get(),
            "--limit",
            str(limit),
            "--width",
            str(width),
            "--height",
            str(width),
            "--fps",
            "10",
            "--draw-face",
            self.gif_draw_face_var.get(),
            "--dataset-dir",
            self.selected_dataset_dir(),
        ]
        if self.gif_force_var.get():
            args.append("--force")
        return args

    def generate_dataset_gifs(self) -> None:
        if self.gif_process is not None:
            messagebox.showinfo("GIF Dataset", "Generate GIF masih berjalan.")
            return
        try:
            args = self._parse_gif_args()
        except ValueError as exc:
            messagebox.showerror("GIF Dataset", str(exc))
            return
        self.btn_make_gif.configure(state="disabled")
        self.btn_load_gif_vocab.configure(state="disabled")
        self.status_var.set("Generate GIF dataset berjalan... progress muncul di panel log.")
        self._set_text("Menjalankan:\n" + " ".join(args) + "\n\n")

        def task() -> None:
            code = 1
            try:
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.gif_process = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.root.after(0, lambda line=line: self._append_text(line))
                code = proc.wait()
                self.root.after(0, lambda: self._gif_generation_done(code))
            except Exception as exc:
                self.root.after(0, lambda: self._gif_generation_failed(str(exc)))
            finally:
                self.gif_process = None

        threading.Thread(target=task, daemon=True).start()

    def _gif_generation_done(self, code: int) -> None:
        self.btn_make_gif.configure(state="normal")
        self.btn_load_gif_vocab.configure(state="normal")
        if code == 0:
            self.status_var.set("Generate GIF dataset selesai. Cek assets/gifs/samples/<schema>/<vocab>.")
            messagebox.showinfo("GIF Dataset", "GIF selesai dibuat / ditemukan existing.")
        else:
            self.status_var.set(f"Generate GIF dataset selesai dengan exit code {code}. Cek log.")
            messagebox.showwarning("GIF Dataset", f"Selesai dengan exit code {code}. Cek log di panel.")

    def _gif_generation_failed(self, message: str) -> None:
        self.gif_process = None
        self.btn_make_gif.configure(state="normal")
        self.btn_load_gif_vocab.configure(state="normal")
        self.status_var.set("Generate GIF dataset gagal.")
        self._append_text("\nERROR: " + message)
        messagebox.showerror("GIF Dataset", message)


    def _run_cli_stream(self, args: list[str], done_callback, fail_callback, process_attr: str | None = None) -> None:
        self._set_text("Menjalankan:\n" + " ".join(args) + "\n\n")

        def task() -> None:
            code = 1
            try:
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                if process_attr:
                    setattr(self, process_attr, proc)
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.root.after(0, lambda line=line: self._append_text(line))
                code = proc.wait()
                self.root.after(0, lambda: done_callback(code))
            except Exception as exc:
                self.root.after(0, lambda: fail_callback(str(exc)))
            finally:
                if process_attr:
                    setattr(self, process_attr, None)

        threading.Thread(target=task, daemon=True).start()

    def _extract_full_args(self) -> list[str]:
        source = self.extract_full_source_var.get().strip()
        if not source:
            raise ValueError("Source wajib diisi.")
        schemas = self.selected_extract_full_schemas()
        if not schemas:
            raise ValueError("Pilih minimal satu schema extract.")
        vocabs = self.selected_extract_full_vocabs()
        all_vocab_var = getattr(self, "extract_full_all_vocab_var", None)
        if all_vocab_var is not None and not all_vocab_var.get() and not vocabs:
            raise ValueError("Pilih minimal satu vocab extract.")
        args = [
            sys.executable,
            self._cli_path(),
            "extract-full",
            "--source",
            source,
            "--clean",
            self.extract_full_clean_var.get(),
            "--dataset-dir",
            self.selected_dataset_dir(),
        ]
        if tuple(schemas) == tuple(fs.FULL_SCHEMA_NAMES):
            args += ["--schema", "full"]
        else:
            for schema_name in schemas:
                args += ["--schema", schema_name]
        for vocab in vocabs:
            args += ["--vocab", vocab]
        return args

    def extract_full_dataset(self) -> None:
        if self.extract_full_process is not None:
            messagebox.showinfo("Extract Full", "Extract full dataset masih berjalan.")
            return
        try:
            args = self._extract_full_args()
        except ValueError as exc:
            messagebox.showerror("Extract Full", str(exc))
            return
        self.btn_extract_full.configure(state="disabled")
        self.status_var.set("Extract full dataset berjalan... progress muncul di log.")
        self._run_cli_stream(args, self._extract_full_done, self._extract_full_failed, process_attr="extract_full_process")

    def _extract_full_done(self, code: int) -> None:
        self.btn_extract_full.configure(state="normal")
        self.status_var.set(f"Extract full dataset selesai dengan exit code {code}.")
        self.refresh_status()

    def _extract_full_failed(self, message: str) -> None:
        self.extract_full_process = None
        self.btn_extract_full.configure(state="normal")
        self.status_var.set("Extract full dataset gagal.")
        self._append_text("\nERROR: " + message)
        messagebox.showerror("Extract Full", message)

    def _train_suite_args(self, variant: str) -> list[str]:
        try:
            epochs, batch = self._parse_training_args()
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        schemas = self.selected_train_suite_schemas()
        if not schemas:
            raise ValueError("Pilih minimal satu schema training.")
        suites = self.selected_train_suite_names()
        if not suites:
            raise ValueError("Pilih minimal satu suite.")
        args = [
            sys.executable,
            self._cli_path(),
            "train-suite",
            "--schema",
            ",".join(schemas),
            "--variant",
            variant,
            "--suite",
            suites,
            "--device",
            self.device_var.get(),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch),
            "--dataset-dir",
            self.selected_dataset_dir(),
        ]
        args += self._overwrite_existing_args()
        return args

    def train_suite_selected(self) -> None:
        if self.train_suite_process is not None:
            messagebox.showinfo("Train Suite", "Train suite masih berjalan.")
            return
        variants = self.selected_train_suite_variants()
        if not variants:
            messagebox.showwarning("Train Suite", "Centang minimal satu target model.")
            return
        try:
            args = self._train_suite_args(",".join(variants))
        except ValueError as exc:
            messagebox.showerror("Train Suite", str(exc))
            return
        self.btn_train_suite_one.configure(state="disabled")
        self.btn_train_suite_all.configure(state="disabled")
        self.status_var.set("Train suite berjalan... progress muncul di log.")
        self._run_cli_stream(args, self._train_suite_done, self._train_suite_failed, process_attr="train_suite_process")

    def train_suite_all(self) -> None:
        if self.train_suite_process is not None:
            messagebox.showinfo("Train Suite", "Train suite masih berjalan.")
            return
        for var in self.train_suite_variant_vars.values():
            var.set(True)
        for var in self.train_suite_suite_vars.values():
            var.set(True)
        try:
            args = self._train_suite_args(",".join(gm.VARIANT_NAMES))
        except ValueError as exc:
            messagebox.showerror("Train Suite", str(exc))
            return
        self.btn_train_suite_one.configure(state="disabled")
        self.btn_train_suite_all.configure(state="disabled")
        self.status_var.set("Train suite semua varian berjalan... progress muncul di log.")
        self._run_cli_stream(args, self._train_suite_done, self._train_suite_failed, process_attr="train_suite_process")

    def _train_suite_done(self, code: int) -> None:
        self.btn_train_suite_one.configure(state="normal")
        self.btn_train_suite_all.configure(state="normal")
        self._reset_overwrite_existing()
        self.status_var.set(f"Train suite selesai dengan exit code {code}.")
        self.refresh_status()

    def _train_suite_failed(self, message: str) -> None:
        self.train_suite_process = None
        self.btn_train_suite_one.configure(state="normal")
        self.btn_train_suite_all.configure(state="normal")
        self._reset_overwrite_existing()
        self.status_var.set("Train suite gagal.")
        self._append_text("\nERROR: " + message)
        messagebox.showerror("Train Suite", message)

    def _eval_args(self) -> list[str]:
        return [
            sys.executable,
            self._cli_path(),
            "eval",
            "--schema",
            self.eval_schema_var.get(),
            "--variant",
            self.eval_variant_var.get(),
            "--suite",
            self.eval_suite_var.get(),
            "--split",
            self.eval_split_var.get(),
            "--device",
            self.device_var.get(),
            "--dataset-dir",
            self.selected_dataset_dir(),
        ]

    def evaluate_test(self) -> None:
        if self.eval_process is not None:
            messagebox.showinfo("Evaluate Test", "Evaluasi masih berjalan.")
            return
        args = self._eval_args()
        self.btn_eval_test.configure(state="disabled")
        self.status_var.set("Evaluasi test berjalan... hasil muncul di log.")
        self._run_cli_stream(args, self._eval_done, self._eval_failed, process_attr="eval_process")

    def _eval_done(self, code: int) -> None:
        self.btn_eval_test.configure(state="normal")
        if code == 0:
            self.status_var.set("Evaluasi selesai. Akurasi/precision/recall/F1 ada di log.")
        else:
            self.status_var.set(f"Evaluasi selesai dengan exit code {code}. Cek log.")

    def _eval_failed(self, message: str) -> None:
        self.eval_process = None
        self.btn_eval_test.configure(state="normal")
        self.status_var.set("Evaluasi gagal.")
        self._append_text("\nERROR: " + message)
        messagebox.showerror("Evaluate Test", message)

    def _selected_photo_schemas(self) -> list[str]:
        return [name for name, var in self.photo_schema_vars.items() if var.get()]

    def _photo_extract_args(self) -> list[str]:
        source = self.photo_source_var.get().strip()
        if not source:
            raise ValueError("Source foto wajib diisi.")
        try:
            workers = int(self.photo_workers_var.get())
        except ValueError as exc:
            raise ValueError("Workers harus angka.") from exc
        if workers <= 0:
            raise ValueError("Workers harus > 0.")
        schemas = self._selected_photo_schemas()
        if not schemas:
            raise ValueError("Pilih minimal satu schema foto.")
        args = [
            sys.executable,
            self._cli_path(),
            "photo",
            "extract",
            "--path",
            source,
            "--workers",
            str(workers),
            "--profile",
            self.photo_profile_var.get(),
            "--dataset-dir",
            self.selected_dataset_dir(),
        ]
        args += self._overwrite_existing_args()
        for schema_name in schemas:
            args += ["--schema", schema_name]
        return args

    def extract_photo_dataset(self) -> None:
        if self.photo_process is not None:
            messagebox.showinfo("Import Foto", "Extract foto masih berjalan.")
            return
        try:
            args = self._photo_extract_args()
        except ValueError as exc:
            messagebox.showerror("Import Foto", str(exc))
            return
        self.btn_photo_extract.configure(state="disabled")
        self.status_var.set("Extract foto berjalan... progress muncul di log.")
        self._set_text("Menjalankan:\n" + " ".join(args) + "\n\n")

        def task() -> None:
            output_parts: list[str] = []
            code = 1
            try:
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                self.photo_process = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    output_parts.append(line)
                    self.root.after(0, lambda line=line: self._append_text(line))
                code = proc.wait()
                output = "".join(output_parts)
                self.root.after(0, lambda: self._photo_extract_done(code, output))
            except Exception as exc:
                self.root.after(0, lambda: self._photo_extract_failed(str(exc)))
            finally:
                self.photo_process = None

        threading.Thread(target=task, daemon=True).start()

    @staticmethod
    def _parse_photo_session(output: str) -> str:
        session = ""
        for line in output.splitlines():
            if line.startswith("session\t"):
                session = line.split("\t", 1)[1].strip()
            elif "photo-extract: session=" in line:
                part = line.split("photo-extract: session=", 1)[1]
                session = part.split(" ", 1)[0].strip()
        return session

    def _photo_extract_done(self, code: int, output: str) -> None:
        self.btn_photo_extract.configure(state="normal")
        self._reset_overwrite_existing()
        session = self._parse_photo_session(output)
        if session:
            self.photo_last_session = session
        if code == 0:
            self.status_var.set("Extract foto selesai. Review hasil sebelum commit.")
            if session:
                self._open_photo_review(session)
            else:
                messagebox.showinfo("Import Foto", "Extract selesai, tapi session tidak terbaca dari log.")
        else:
            self.status_var.set(f"Extract foto selesai dengan exit code {code}.")
            if session:
                self._open_photo_review(session)
            else:
                messagebox.showwarning("Import Foto", f"Extract gagal dengan exit code {code}. Cek log.")

    def _photo_extract_failed(self, message: str) -> None:
        self.photo_process = None
        self.btn_photo_extract.configure(state="normal")
        self._reset_overwrite_existing()
        self.status_var.set("Extract foto gagal.")
        self._append_text("\nERROR: " + message)
        messagebox.showerror("Import Foto", message)

    def _open_photo_review(self, session: str) -> None:
        manifest_path = Path(session) / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Review Foto", f"Gagal baca manifest: {exc}")
            return
        entries = list(manifest.get("entries", []))
        success_entries = [entry for entry in entries if entry.get("status") == "success"]
        failed_entries = [entry for entry in entries if entry.get("status") == "failed"]
        skipped_entries = [entry for entry in entries if entry.get("status") == "skipped_existing"]

        win = tk.Toplevel(self.root)
        win.title("Review Import Foto")
        win.geometry("1100x680")
        win.minsize(900, 560)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(win)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        summary_tab = ttk.Frame(notebook)
        ok_tab = ttk.Frame(notebook)
        fail_tab = ttk.Frame(notebook)
        notebook.add(summary_tab, text="Summary")
        notebook.add(ok_tab, text="Berhasil")
        notebook.add(fail_tab, text="Gagal")

        summary_tab.columnconfigure(0, weight=1)
        summary_tab.rowconfigure(1, weight=1)
        summary = manifest.get("summary", {})
        ttk.Label(
            summary_tab,
            text=(
                f"Session: {session} | success={summary.get('success', 0)} "
                f"failed={summary.get('failed', 0)} skipped={summary.get('skipped_existing', 0)}"
            ),
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        group_frame = ttk.Frame(summary_tab)
        group_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        group_frame.columnconfigure(0, weight=1)
        canvas = tk.Canvas(group_frame, highlightthickness=0)
        scroll = ttk.Scrollbar(group_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        group_frame.rowconfigure(0, weight=1)

        group_vars: dict[tuple[str, str, str], tk.BooleanVar] = {}
        group_counts: dict[tuple[str, str, str], int] = {}
        for entry in success_entries:
            key = (str(entry["schema"]), str(entry["split"]), str(entry["label"]))
            group_counts[key] = group_counts.get(key, 0) + 1
        for row, key in enumerate(sorted(group_counts)):
            schema_name, split, label = key
            var = tk.BooleanVar(value=True)
            group_vars[key] = var
            ttk.Checkbutton(
                inner,
                variable=var,
                text=f"{schema_name} | {split}/{label} | {group_counts[key]} sample",
            ).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        if skipped_entries:
            ttk.Label(inner, text=f"Skipped existing: {len(skipped_entries)} entry").grid(row=len(group_counts) + 1, column=0, sticky="w", padx=6, pady=8)

        ok_tab.columnconfigure(0, weight=1)
        ok_tab.rowconfigure(0, weight=1)
        ok_cols = ("schema", "split", "label", "frames", "video_id", "source")
        ok_tree = ttk.Treeview(ok_tab, columns=ok_cols, show="headings", height=16)
        for col in ok_cols:
            ok_tree.heading(col, text=col)
            ok_tree.column(col, width=120 if col != "source" else 420, stretch=(col == "source"))
        ok_scroll = ttk.Scrollbar(ok_tab, orient="vertical", command=ok_tree.yview)
        ok_tree.configure(yscrollcommand=ok_scroll.set)
        ok_tree.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        ok_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        for entry in success_entries:
            ok_tree.insert(
                "",
                "end",
                iid=str(entry["entry_id"]),
                values=(
                    entry.get("schema", ""),
                    entry.get("split", ""),
                    entry.get("label", ""),
                    entry.get("frames", ""),
                    entry.get("video_id", ""),
                    entry.get("source_image_path", ""),
                ),
            )

        fail_tab.columnconfigure(0, weight=1)
        fail_tab.rowconfigure(0, weight=1)
        fail_cols = ("schema", "split", "label", "video_id", "error", "source")
        fail_tree = ttk.Treeview(fail_tab, columns=fail_cols, show="headings", height=16)
        for col in fail_cols:
            fail_tree.heading(col, text=col)
            fail_tree.column(col, width=120 if col not in {"error", "source"} else 300, stretch=(col in {"error", "source"}))
        fail_scroll = ttk.Scrollbar(fail_tab, orient="vertical", command=fail_tree.yview)
        fail_tree.configure(yscrollcommand=fail_scroll.set)
        fail_tree.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        fail_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        for entry in failed_entries:
            fail_tree.insert(
                "",
                "end",
                values=(
                    entry.get("schema", ""),
                    entry.get("split", ""),
                    entry.get("label", ""),
                    entry.get("video_id", ""),
                    entry.get("error", ""),
                    entry.get("source_image_path", ""),
                ),
            )

        buttons = ttk.Frame(win)
        buttons.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(buttons, text="Generate GIF Sample", command=lambda: self._photo_generate_gif(session, ok_tree)).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(buttons, text="Simpan yang Dicentang", command=lambda: self._photo_commit_checked(session, group_vars, win)).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(buttons, text="Tutup", command=win.destroy).grid(row=0, column=2, sticky="ew", padx=4)

    def _photo_generate_gif(self, session: str, tree: ttk.Treeview) -> None:
        selected = tree.selection()
        if not selected:
            messagebox.showerror("GIF Foto", "Pilih satu sample berhasil dulu.")
            return
        entry_id = str(selected[0])
        args = [
            sys.executable,
            self._cli_path(),
            "photo",
            "gif",
            "--session",
            session,
            "--entry-id",
            entry_id,
            "--force",
        ]
        self.status_var.set("Generate GIF temp foto...")

        def task() -> None:
            try:
                proc = subprocess.run(args, capture_output=True, text=True)
                output = (proc.stdout or "") + (proc.stderr or "")
                self.root.after(0, lambda: self._photo_gif_done(proc.returncode, output))
            except Exception as exc:
                self.root.after(0, lambda: self._photo_gif_failed(str(exc)))

        threading.Thread(target=task, daemon=True).start()

    def _photo_gif_done(self, code: int, output: str) -> None:
        self._append_text("\n" + output)
        if code == 0:
            self.status_var.set("GIF temp foto selesai.")
            messagebox.showinfo("GIF Foto", "GIF temp selesai dibuat.")
        else:
            self.status_var.set(f"GIF temp foto selesai dengan exit code {code}.")
            messagebox.showwarning("GIF Foto", f"Exit code {code}. Cek log.")

    def _photo_gif_failed(self, message: str) -> None:
        self.status_var.set("GIF temp foto gagal.")
        self._append_text("\nERROR: " + message)
        messagebox.showerror("GIF Foto", message)

    def _photo_commit_checked(self, session: str, group_vars: dict[tuple[str, str, str], tk.BooleanVar], window: tk.Toplevel) -> None:
        manifest_path = Path(session) / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Commit Foto", f"Gagal baca manifest: {exc}")
            return
        accepted_ids = []
        for entry in manifest.get("entries", []):
            if entry.get("status") != "success":
                continue
            key = (str(entry["schema"]), str(entry["split"]), str(entry["label"]))
            var = group_vars.get(key)
            if var is not None and var.get():
                accepted_ids.append(str(entry["entry_id"]))
        if not accepted_ids:
            messagebox.showerror("Commit Foto", "Tidak ada sample yang dicentang.")
            return
        accept_path = Path(session) / "accepted_entry_ids.json"
        accept_path.write_text(json.dumps({"entry_ids": accepted_ids}, indent=2), encoding="utf-8")
        args = self._photo_commit_args(session, accept_path)
        self.status_var.set("Commit foto berjalan...")
        self._run_cli_stream(
            args,
            lambda code: self._photo_commit_done(code, window),
            self._photo_commit_failed,
        )

    def _photo_commit_args(self, session: str, accept_path: Path) -> list[str]:
        args = [
            sys.executable,
            self._cli_path(),
            "photo",
            "commit",
            "--session",
            session,
            "--accept-file",
            str(accept_path),
            "--dataset-dir",
            self.selected_dataset_dir(),
        ]
        args += self._overwrite_existing_args()
        return args

    def _photo_commit_done(self, code: int, window: tk.Toplevel) -> None:
        self._reset_overwrite_existing()
        if code == 0:
            self.status_var.set("Commit foto selesai.")
            self.refresh_status()
            messagebox.showinfo("Commit Foto", "Sample foto terpilih sudah disimpan.")
            window.destroy()
        else:
            self.status_var.set(f"Commit foto selesai dengan exit code {code}.")
            messagebox.showwarning("Commit Foto", f"Exit code {code}. Cek log.")

    def _photo_commit_failed(self, message: str) -> None:
        self._reset_overwrite_existing()
        self.status_var.set("Commit foto gagal.")
        self._append_text("\nERROR: " + message)
        messagebox.showerror("Commit Foto", message)

    def _augment_base_args(self) -> list[str]:
        return [
            sys.executable,
            self._cli_path(),
            "--schema_PLACEHOLDER",
        ]

    def refresh_augment_vocab_list(self) -> None:
        try:
            labels: set[str] = set()
            schema_names = self.selected_augment_schemas()
            if not schema_names:
                raise ValueError("Pilih minimal satu schema augmentasi.")
            for schema_name in schema_names:
                summary = gm.dataset_summary(dataset_dir=self.selected_dataset_dir(), schema=schema_name)
                labels.update(str(label) for label in summary.get("labels", []) if str(label).lower() not in gm.EXCLUDED_LABELS)
            self.augment_vocab_items = sorted(labels)
            old_values = {label: var.get() for label, var in self.augment_vocab_vars.items()}
            self.augment_vocab_vars = {
                label: tk.BooleanVar(value=bool(self.augment_all_vocab_var.get() or old_values.get(label, False)))
                for label in self.augment_vocab_items
            }
            for child in self.augment_vocab_frame.winfo_children():
                child.destroy()
            if not self.augment_vocab_items:
                ttk.Label(self.augment_vocab_frame, text="Belum ada vocab terbaca untuk schema ini.").grid(row=0, column=0, sticky="w")
            for idx, label in enumerate(self.augment_vocab_items):
                ttk.Checkbutton(
                    self.augment_vocab_frame,
                    text=label,
                    variable=self.augment_vocab_vars[label],
                ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=(0, 14), pady=2)
            self.status_var.set(f"Vocab augmentasi dimuat: {len(self.augment_vocab_items)} label.")
        except Exception as exc:
            self.status_var.set(f"Gagal memuat vocab augmentasi: {exc}")
            messagebox.showerror("Augmentasi", str(exc))

    def _toggle_all_augment_vocab(self) -> None:
        value = bool(self.augment_all_vocab_var.get())
        for var in self.augment_vocab_vars.values():
            var.set(value)

    def _augment_args(self, delete: bool = False) -> list[str]:
        schemas = self.selected_augment_schemas()
        if not schemas:
            raise ValueError("Pilih minimal satu schema augmentasi.")
        schema = ",".join(schemas)
        split = ",".join(self.selected_augment_splits())
        vocabs = self.selected_augment_vocabs()
        args = [
            sys.executable,
            self._cli_path(),
            "augment-delete" if delete else "augment",
            "--schema",
            schema,
            "--split",
            split,
            "--dataset-dir",
            self.selected_dataset_dir(),
        ]
        if self.augment_all_vocab_var.get():
            args.append("--all-vocab")
        else:
            for vocab in vocabs:
                args += ["--vocab", vocab]
        if delete:
            if self.augment_delete_dry_var.get():
                args.append("--dry-run")
            return args
        try:
            copies = int(self.augment_copies_var.get())
            min_source = int(self.augment_min_var.get())
            target = int(self.augment_target_var.get())
        except ValueError as exc:
            raise ValueError("Copy/sample, min asli, dan target/class harus angka.") from exc
        args += ["--copies-per-sample", str(copies), "--min-source-samples", str(min_source), "--target-per-class", str(target)]
        args += self._overwrite_existing_args()
        seed = self.augment_seed_var.get().strip()
        if seed:
            try:
                int(seed)
            except ValueError as exc:
                raise ValueError("Seed harus angka atau kosong.") from exc
            args += ["--seed", seed]
        return args

    def run_augmentation(self) -> None:
        if self.augment_process is not None:
            messagebox.showinfo("Augmentasi", "Proses augmentasi masih berjalan.")
            return
        try:
            args = self._augment_args(delete=False)
        except ValueError as exc:
            messagebox.showerror("Augmentasi", str(exc))
            return
        self.btn_augment.configure(state="disabled")
        self.btn_augment_delete.configure(state="disabled")
        self.status_var.set("Augmentasi berjalan... progress muncul di log.")
        self._run_cli_stream(args, self._augment_done, self._augment_failed, process_attr="augment_process")

    def delete_augmentation(self) -> None:
        if self.augment_process is not None:
            messagebox.showinfo("Hapus augmentasi", "Proses augmentasi masih berjalan.")
            return
        if not self.augment_delete_dry_var.get():
            if not messagebox.askyesno("Konfirmasi", "Hapus hasil augmentasi sesuai schema/split/vocab? Sample asli tidak dihapus."):
                return
        try:
            args = self._augment_args(delete=True)
        except ValueError as exc:
            messagebox.showerror("Hapus augmentasi", str(exc))
            return
        self.btn_augment.configure(state="disabled")
        self.btn_augment_delete.configure(state="disabled")
        self.status_var.set("Hapus augmentasi berjalan... progress muncul di log.")
        self._run_cli_stream(args, self._augment_done, self._augment_failed, process_attr="augment_process")

    def _augment_done(self, code: int) -> None:
        self.btn_augment.configure(state="normal")
        self.btn_augment_delete.configure(state="normal")
        self._reset_overwrite_existing()
        self.status_var.set(f"Augmentasi/hapus augmentasi selesai dengan exit code {code}. Cek log.")
        self.refresh_status()

    def _augment_failed(self, message: str) -> None:
        self.augment_process = None
        self.btn_augment.configure(state="normal")
        self.btn_augment_delete.configure(state="normal")
        self._reset_overwrite_existing()
        self.status_var.set("Augmentasi gagal.")
        self._append_text("\nERROR: " + message)
        messagebox.showerror("Augmentasi", message)

    def _selected_sample_item(self) -> dict[str, str] | None:
        sel = self.sample_listbox.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if 0 <= idx < len(self.sample_items):
            return self.sample_items[idx]
        return None

    def load_sample_list(self) -> None:
        vocab = self.sample_vocab_var.get().strip()
        if not vocab:
            messagebox.showerror("Hapus Sample", "Isi vocab dulu.")
            return
        self.btn_load_samples.configure(state="disabled")
        self.status_var.set("Memuat sample dataset...")
        args = self._sample_list_args(vocab)

        def task() -> None:
            try:
                proc = subprocess.run(args, capture_output=True, text=True)
                output = (proc.stdout or "") + (proc.stderr or "")
                self.root.after(0, lambda: self._sample_list_loaded(proc.returncode, output))
            except Exception as exc:
                self.root.after(0, lambda: self._sample_list_failed(str(exc)))

        threading.Thread(target=task, daemon=True).start()

    def _sample_list_args(self, vocab: str) -> list[str]:
        return [
            sys.executable,
            self._cli_path(),
            "sample",
            "list",
            "--schema",
            self.sample_schema_var.get(),
            "--split",
            self.sample_split_var.get(),
            "--vocab",
            vocab,
            "--dataset-dir",
            self.selected_dataset_dir(),
            "--plain",
        ]

    def _sample_list_loaded(self, code: int, output: str) -> None:
        self.btn_load_samples.configure(state="normal")
        self._set_text(output.strip() or f"sample list selesai dengan exit code {code}")
        self.sample_listbox.delete(0, "end")
        self.sample_items = []
        if code != 0:
            self.status_var.set("Gagal memuat sample. Cek log.")
            messagebox.showwarning("Hapus Sample", f"sample list exit code {code}")
            return
        for line in output.splitlines():
            if not line.strip() or line.startswith("label\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            item = {
                "label": parts[0],
                "split": parts[1],
                "video_id": parts[2],
                "rows": parts[3],
                "schemas": parts[4],
                "frames_by_schema": parts[5],
                "is_augmented": parts[6],
            }
            self.sample_items.append(item)
            aug = " aug" if parts[6] == "1" else ""
            self.sample_listbox.insert("end", f"{parts[1]} | rows={parts[3]} | {parts[2]} | {parts[4]}{aug}")
        if self.sample_items:
            self.sample_listbox.selection_set(0)
            self.status_var.set(f"Sample dimuat: {len(self.sample_items)}. Pilih salah satu untuk dihapus.")
        else:
            self.status_var.set("Tidak ada sample cocok.")

    def _sample_list_failed(self, message: str) -> None:
        self.btn_load_samples.configure(state="normal")
        self.status_var.set("Gagal memuat sample.")
        self._set_text(message)
        messagebox.showerror("Hapus Sample", message)

    def delete_selected_sample(self) -> None:
        if self.sample_process is not None:
            messagebox.showinfo("Hapus Sample", "Proses hapus sample masih berjalan.")
            return
        item = self._selected_sample_item()
        if not item:
            messagebox.showerror("Hapus Sample", "Pilih sample dari list dulu.")
            return
        dry_run = bool(self.sample_dry_run_var.get())
        if not dry_run:
            ok = messagebox.askyesno(
                "Konfirmasi hapus sample",
                "Hapus sample ini dari semua schema target dan hapus GIF terkait?\n\n"
                f"vocab: {item['label']}\n"
                f"split: {item['split']}\n"
                f"video_id: {item['video_id']}\n\n"
                "Backup parquet akan dibuat otomatis.",
            )
            if not ok:
                return
        args = self._sample_delete_args(item, dry_run)
        self.btn_delete_sample.configure(state="disabled")
        self.btn_load_samples.configure(state="disabled")
        self.status_var.set("Hapus sample berjalan... cek log.")
        self._set_text("Menjalankan:\n" + " ".join(args) + "\n\n")

        def task() -> None:
            code = 1
            try:
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                self.sample_process = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.root.after(0, lambda line=line: self._append_text(line))
                code = proc.wait()
                self.root.after(0, lambda: self._sample_delete_done(code, dry_run))
            except Exception as exc:
                self.root.after(0, lambda: self._sample_delete_failed(str(exc)))
            finally:
                self.sample_process = None

        threading.Thread(target=task, daemon=True).start()

    def _sample_delete_args(self, item: dict[str, str], dry_run: bool) -> list[str]:
        args = [
            sys.executable,
            self._cli_path(),
            "sample",
            "delete",
            "--schema",
            self.sample_schema_var.get(),
            "--vocab",
            item["label"],
            "--split",
            item["split"],
            "--video-id",
            item["video_id"],
            "--dataset-dir",
            self.selected_dataset_dir(),
        ]
        if dry_run:
            args.append("--dry-run")
        return args

    def _sample_delete_done(self, code: int, dry_run: bool) -> None:
        self.btn_delete_sample.configure(state="normal")
        self.btn_load_samples.configure(state="normal")
        if code == 0:
            self.status_var.set("Dry-run selesai." if dry_run else "Sample berhasil dihapus.")
            messagebox.showinfo("Hapus Sample", "Dry-run selesai, belum ada yang dihapus." if dry_run else "Sample berhasil dihapus.")
            if not dry_run:
                self.load_sample_list()
        else:
            self.status_var.set(f"Hapus sample selesai dengan exit code {code}. Cek log.")
            messagebox.showwarning("Hapus Sample", f"Selesai dengan exit code {code}. Cek log.")

    def _sample_delete_failed(self, message: str) -> None:
        self.sample_process = None
        self.btn_delete_sample.configure(state="normal")
        self.btn_load_samples.configure(state="normal")
        self.status_var.set("Hapus sample gagal.")
        self._append_text("\nERROR: " + message)
        messagebox.showerror("Hapus Sample", message)

    def _parse_training_args(self) -> tuple[int, int]:
        try:
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
        except ValueError as exc:
            raise ValueError("Epochs dan batch harus angka.") from exc
        if epochs <= 0 or batch <= 0:
            raise ValueError("Epochs dan batch harus > 0.")
        return epochs, batch

    def _parse_specialist_training_args(self) -> tuple[int, int, int]:
        try:
            epochs = int(self.specialist_epochs_var.get())
            batch = int(self.specialist_batch_var.get())
            patience = int(self.specialist_patience_var.get())
        except ValueError as exc:
            raise ValueError("Epochs, batch, dan patience spesialis harus angka.") from exc
        if epochs <= 0 or batch <= 0:
            raise ValueError("Epochs dan batch spesialis harus > 0.")
        if patience < 0:
            raise ValueError("Patience spesialis minimal 0.")
        return epochs, batch, patience

    def train_specialist_selected(self) -> None:
        if self.specialist_train_thread is not None and self.specialist_train_thread.is_alive():
            messagebox.showinfo("Vocab Khusus", "Training vocab khusus masih berjalan.")
            return
        try:
            labels = self.selected_specialist_labels()
            specialist_name = self.selected_specialist_name(labels)
            epochs, batch, patience = self._parse_specialist_training_args()
        except ValueError as exc:
            messagebox.showerror("Vocab Khusus", str(exc))
            return
        schemas = self.selected_specialist_schemas()
        variants = self.selected_specialist_variants()
        if not schemas:
            messagebox.showwarning("Vocab Khusus", "Centang minimal satu schema spesialis.")
            return
        if not variants:
            messagebox.showwarning("Vocab Khusus", "Centang minimal satu model spesialis.")
            return

        self.btn_train_specialist.configure(state="disabled")
        self.btn_load_specialist_vocab.configure(state="disabled")
        self.status_var.set(f"Training spesialis {specialist_name} berjalan...")
        self._set_text(
            "Training spesialis:\n"
            f"name={specialist_name}\nlabels={','.join(labels)}\n"
            f"schemas={','.join(schemas)}\nvariants={','.join(variants)}\n"
            f"epochs={epochs} batch={batch} patience={patience}\n\n"
        )
        overwrite_existing = bool(self.overwrite_existing_var.get())

        def task() -> None:
            lines: list[str] = []
            try:
                for schema_name in schemas:
                    for variant in variants:
                        ok, msg = gm.train_specialist_variant(
                            variant,
                            labels,
                            specialist_name=specialist_name,
                            dataset_dir=self.selected_dataset_dir(),
                            epochs=epochs,
                            batch_size=batch,
                            patience=patience,
                            device=self.device_var.get(),
                            schema=schema_name,
                            overwrite_existing=overwrite_existing,
                            train_data=gm.variant_train_data_mode(variant),
                        )
                        lines.append(("OK " if ok else "ERR ") + msg)
            except Exception as exc:
                lines.append("ERR " + str(exc))
            self.root.after(0, lambda: self._specialist_training_done("\n".join(lines)))

        self.specialist_train_thread = threading.Thread(target=task, daemon=True)
        self.specialist_train_thread.start()

    def _specialist_training_done(self, message: str) -> None:
        self.btn_train_specialist.configure(state="normal")
        self.btn_load_specialist_vocab.configure(state="normal")
        self._reset_overwrite_existing()
        self.status_var.set("Training spesialis selesai.")
        self._set_text(message)
        messagebox.showinfo("Vocab Khusus", message)

    def _variant_options_for_schema(self, schema: str, previous_value: str, model_data: str | None = "original") -> tuple[dict[str, str], list[str], str]:
        schema_name = fs.normalize_schema_name(schema)
        display_to_value: dict[str, str] = {}
        options: list[str] = []
        variant_pool = variant_pool_for_model_data(model_data)
        try:
            best = gm.select_best_available_variant(schema=schema_name, variants=variant_pool)
            best_label = f"auto (best: gru_{best})"
        except Exception:
            best_label = "auto (no checkpoint)"
        display_to_value[best_label] = "auto"
        options.append(best_label)

        for variant in variant_pool:
            display = f"gru_{variant}"
            if not gm.checkpoint_exists(variant, schema=schema_name):
                display += " (missing)"
            else:
                try:
                    metadata = gm.load_metadata(variant, schema=schema_name)
                    val_acc = metadata.get("best_val_acc")
                    if isinstance(val_acc, (float, int)):
                        display += f" (val {float(val_acc):.3f})"
                        if float(val_acc) < 0.70:
                            display += " LOW"
                except Exception:
                    display += " (metadata?)"
            display_to_value[display] = variant
            options.append(display)

        selected_display = next((label for label, value in display_to_value.items() if value == previous_value), options[0])
        return display_to_value, options, selected_display

    def _build_variant_options(self) -> None:
        schema_names = self.selected_train_schemas()
        if len(schema_names) != 1:
            return
        previous_value = self.selected_variant_value()
        self.variant_display_to_value, options, selected_display = self._variant_options_for_schema(schema_names[0], previous_value)
        self.variant_options = options
        if hasattr(self, "variant_combo"):
            self.variant_combo.configure(values=options)
        self.variant_var.set(selected_display)

    def _build_live_variant_options(self) -> None:
        previous_value = self.selected_live_variant_value()
        self.live_variant_display_to_value, options, selected_display = self._variant_options_for_schema(
            self.selected_live_schema(),
            previous_value,
            model_data=self.selected_live_model_data_mode(),
        )
        self.live_variant_options = options
        if hasattr(self, "live_variant_combo"):
            self.live_variant_combo.configure(values=options)
        if hasattr(self, "live_plus_variant_combo"):
            self.live_plus_variant_combo.configure(values=options)
        self.live_variant_var.set(selected_display)

    def selected_variant_value(self) -> str:
        value = self.variant_display_to_value.get(self.variant_var.get())
        if value:
            return value
        raw = self.variant_var.get().split(" ", 1)[0]
        if raw == "auto":
            return "auto"
        return raw

    def refresh_status(self) -> None:
        try:
            self._build_variant_options()
            self._build_live_variant_options()
            schema_names = self.selected_train_schemas()
            live_schema = self.selected_live_schema()
            dataset_dir = self.selected_dataset_dir()
            lines = [
                f"Dataset folder: {dataset_dir}",
            ]
            if not schema_names:
                lines.append("Schema training: belum ada yang dipilih.")
            for schema_name in schema_names:
                summary = gm.dataset_summary(dataset_dir=dataset_dir, schema=schema_name)
                model_status = gm.list_model_status(schema=schema_name)
                lines.extend([
                    "",
                    f"Dataset {schema_name}: {summary['total_samples']} sampel",
                    f"Kelas classifier non-idle: {summary['num_classifier_classes']}",
                    "Checkpoint:",
                ])
                for variant in gm.VARIANT_NAMES:
                    lines.append(f"- gru_{variant}: {model_status.get(variant, '-')}")
            if live_schema not in schema_names:
                lines.extend(["", f"Live schema {live_schema}:"])
                live_status = gm.list_model_status(schema=live_schema)
                for variant in gm.VARIANT_NAMES:
                    lines.append(f"- gru_{variant}: {live_status.get(variant, '-')}")
            self._set_text("\n".join(lines))
            self.status_var.set("Status diperbarui.")
        except Exception as exc:
            self._set_text(str(exc))
            self.status_var.set("Gagal membaca status.")

    def _run_training(self, variants: tuple[str, ...]) -> None:
        try:
            epochs, batch = self._parse_training_args()
        except ValueError as exc:
            messagebox.showerror("Input training tidak valid", str(exc))
            return
        schema_names = self.selected_train_schemas()
        if not schema_names:
            messagebox.showerror("Training GRU", "Pilih minimal satu schema training.")
            return

        self.btn_train_one.configure(state="disabled")
        self.btn_train_all.configure(state="disabled")
        self.status_var.set("Training berjalan...")
        overwrite_existing = bool(self.overwrite_existing_var.get())

        def task() -> None:
            lines = []
            try:
                for schema_name in schema_names:
                    for variant in variants:
                        ok, msg = gm.train_variant(
                            variant,
                            dataset_dir=self.selected_dataset_dir(),
                            epochs=epochs,
                            batch_size=batch,
                            device=self.device_var.get(),
                            schema=schema_name,
                            overwrite_existing=overwrite_existing,
                        )
                        lines.append(("OK " if ok else "ERR ") + msg)
            except Exception as exc:
                lines.append("ERR " + str(exc))
            self.root.after(0, lambda: self._training_done("\n".join(lines)))

        threading.Thread(target=task, daemon=True).start()

    def _training_done(self, message: str) -> None:
        self.btn_train_one.configure(state="normal")
        self.btn_train_all.configure(state="normal")
        self._reset_overwrite_existing()
        self.status_var.set("Training selesai.")
        self.refresh_status()
        messagebox.showinfo("Training GRU", message)

    def train_selected(self) -> None:
        variants = self.selected_train_variants()
        if not variants:
            messagebox.showwarning("Training GRU", "Centang minimal satu target model.")
            return
        self._run_training(variants)

    def train_all(self) -> None:
        for var in self.train_variant_vars.values():
            var.set(True)
        self._run_training(gm.VARIANT_NAMES)


    def _record_live_args(self) -> list[str]:
        label = self.record_label_var.get().strip().replace(" ", "_").lower()
        if not label:
            raise ValueError("Label vocab wajib diisi.")
        try:
            count = int(self.record_count_var.get())
            duration = float(self.record_duration_var.get())
            camera = int(self.record_camera_var.get())
            target_fps = float(self.record_target_fps_var.get())
            preview_width = int(self.record_preview_width_var.get())
        except ValueError as exc:
            raise ValueError("Count, durasi, camera, target FPS, dan preview width harus angka.") from exc
        if count <= 0:
            raise ValueError("Count harus > 0.")
        if duration <= 0:
            raise ValueError("Durasi harus > 0 detik.")
        if target_fps <= 0:
            raise ValueError("Target FPS harus > 0.")
        if preview_width <= 0:
            raise ValueError("Preview width harus > 0.")

        args = [
            sys.executable,
            self._cli_path(),
            "record-live",
            "--label",
            label,
            "--split",
            self.record_split_var.get(),
            "--schema",
            self.record_schema_var.get(),
            "--count",
            str(count),
            "--duration",
            str(duration),
            "--camera",
            str(camera),
            "--profile",
            self.record_profile_var.get(),
            "--target-fps",
            str(target_fps),
            "--preview-width",
            str(preview_width),
            "--dataset-dir",
            self.selected_dataset_dir(),
            "--full-root",
            self.selected_full_root(),
        ]
        if self.record_window_var.get():
            args.append("--window")
        if self.record_keep_short_var.get():
            args.append("--keep-short")
        args += self._overwrite_existing_args()
        return args

    def record_live_dataset(self) -> None:
        try:
            args = self._record_live_args()
        except ValueError as exc:
            messagebox.showerror("Record dataset", str(exc))
            return

        self.btn_record_live.configure(state="disabled")
        self.status_var.set("Record dataset live berjalan... preview akan tampilkan FPS capture/simpan; video tidak disimpan.")
        self._set_text("Menjalankan:\n" + " ".join(args))

        def task() -> None:
            try:
                proc = subprocess.run(args, capture_output=True, text=True)
                output = (proc.stdout or "") + (proc.stderr or "")
                self.root.after(0, lambda: self._record_live_done(proc.returncode, output))
            except Exception as exc:
                self.root.after(0, lambda: self._record_live_failed(str(exc)))

        threading.Thread(target=task, daemon=True).start()

    def _record_live_done(self, code: int, output: str) -> None:
        self.btn_record_live.configure(state="normal")
        self._reset_overwrite_existing()
        self._set_text(output.strip() or f"record-live selesai dengan exit code {code}")
        if code == 0:
            self.status_var.set("Record dataset live selesai. Dataset/parquet sudah diperbarui. Klik Refresh untuk hitung ulang status.")
            try:
                self._build_variant_options()
                self._build_live_variant_options()
            except Exception:
                pass
            messagebox.showinfo("Record dataset", "Selesai. Feature masuk ke schema parquet dan full_features.")
        else:
            self.status_var.set(f"Record dataset live selesai dengan error code {code}.")
            messagebox.showwarning("Record dataset", f"Selesai dengan exit code {code}. Cek log di panel.")

    def _record_live_failed(self, message: str) -> None:
        self.btn_record_live.configure(state="normal")
        self._reset_overwrite_existing()
        self.status_var.set("Record dataset live gagal.")
        self._set_text(message)
        messagebox.showerror("Record dataset", message)

    def _format_live_capture_status(self, item: dict) -> str:
        profile = item.get("profile") or self.mode_var.get()
        width = item.get("capture_width")
        height = item.get("capture_height")
        fps = item.get("camera_fps")
        proc = item.get("proc_width")
        pose_proc = item.get("pose_proc_width")
        display_width = item.get("display_width")
        display_height = item.get("display_height")
        if width and height:
            fps_text = f"@{fps}" if fps else ""
            proc_text = f" | proc {proc}px pose {pose_proc}px" if proc and pose_proc else ""
            display_text = f" | display {display_width}x{display_height}" if display_width and display_height else ""
            return f"profile {profile} | cap {width}x{height}{fps_text}{proc_text}{display_text}"
        return f"profile {profile}"

    def _format_live_specialist_status(self, item: dict) -> str:
        if not item.get("specialist_enabled"):
            return "specialist off"
        name = item.get("specialist_name") or self.live_specialist_name_var.get() or "all"
        state = "ready" if item.get("specialist_ready") else "missing"
        message = str(item.get("specialist_message") or "").strip()
        suffix = f" {message}" if message and message not in {state, "not loaded"} else ""
        count = int(item.get("specialist_count") or 0)
        selected = str(item.get("specialist_selected") or "").strip()
        selected_text = f" -> {selected}" if selected else ""
        count_text = f"[{count}]" if count else ""
        return f"specialist {name}{count_text}:{state}{selected_text}{suffix}"

    def _finish_live_reset(self) -> None:
        self.live_reset_job = None
        self.live_poll_job = None
        self.btn_live.configure(text="Start Live Test", state="normal")

    def _reset_live_ui(self, status_message: str | None = None, cooldown_ms: int = 0) -> None:
        self.release_tts_after_live("live")
        if self.live_poll_job is not None:
            try:
                self.root.after_cancel(self.live_poll_job)
            except Exception:
                pass
            self.live_poll_job = None
        if getattr(self, "live_reset_job", None) is not None:
            try:
                self.root.after_cancel(self.live_reset_job)
            except Exception:
                pass
            self.live_reset_job = None
        if self.live_worker is not None:
            # Beri worker waktu penuh agar finally-nya yang melepas kamera (single owner),
            # sama seperti integrasi (bisindo_live_assistant.stop_worker). Timeout pendek
            # bikin force_cleanup jalan saat worker masih cap.release() -> double-release.
            live_session.stop_worker(self.live_worker, join_timeout=3.0, force=True)
        self.live_worker = None
        self.live_stop_started_at = None
        if status_message is not None:
            self.live_status_var.set(status_message)
        if cooldown_ms > 0:
            self.btn_live.configure(text="Releasing camera...", state="disabled")
            self.live_reset_job = self.root.after(int(cooldown_ms), self._finish_live_reset)
        else:
            self._finish_live_reset()

    def _normal_live_running_or_releasing(self) -> bool:
        return (
            getattr(self, "live_reset_job", None) is not None
            or (self.live_worker is not None and self.live_worker.is_alive())
        )

    def _live_plus_running_or_releasing(self) -> bool:
        return (
            getattr(self, "live_plus_reset_job", None) is not None
            or (self.live_plus_worker is not None and self.live_plus_worker.is_alive())
        )

    def _finish_live_plus_reset(self) -> None:
        self.live_plus_reset_job = None
        self.live_plus_poll_job = None
        self.btn_live_plus.configure(text="Start LiveTest Plus", state="normal")

    def _reset_live_plus_ui(self, status_message: str | None = None, cooldown_ms: int = 0) -> None:
        self.release_tts_after_live("plus")
        if self.live_plus_poll_job is not None:
            try:
                self.root.after_cancel(self.live_plus_poll_job)
            except Exception:
                pass
            self.live_plus_poll_job = None
        if getattr(self, "live_plus_reset_job", None) is not None:
            try:
                self.root.after_cancel(self.live_plus_reset_job)
            except Exception:
                pass
            self.live_plus_reset_job = None
        if self.live_plus_worker is not None:
            # Lihat catatan di _reset_live_ui: tunggu finally worker melepas kamera dulu.
            live_session.stop_worker(self.live_plus_worker, join_timeout=3.0, force=True)
        self.live_plus_worker = None
        self.live_plus_stop_started_at = None
        if status_message is not None:
            self.live_plus_status_var.set(status_message)
        if cooldown_ms > 0:
            self.btn_live_plus.configure(text="Releasing camera...", state="disabled")
            self.live_plus_reset_job = self.root.after(int(cooldown_ms), self._finish_live_plus_reset)
        else:
            self._finish_live_plus_reset()

    def _update_live_plus_buffer_text(self) -> None:
        words = self.live_plus_buffer.pending_words()
        text = lp.words_to_text(words) or "-"
        self.live_plus_buffer_var.set(f"Buffer ({len(words)}/{self.live_plus_buffer.max_words}): {text}")

    def _use_loaded_tts_for_live_plus(self) -> bool:
        use_var = getattr(self, "tts_use_loaded_var", None)
        if use_var is not None and not bool(use_var.get()):
            return False
        return self._tts_is_loaded()

    def _speak_live_plus_text(
        self,
        text: str,
        *,
        sink_name: str | None = None,
        background: bool = True,
        report_event: bool = False,
        source: str = "buffer",
    ) -> dict | None:
        clean_text = str(text or "").strip()
        if not clean_text:
            return None
        if self._use_loaded_tts_for_live_plus():
            player = self.tts_player_var.get().strip() or "auto"

            def run_tts() -> dict:
                ok = True
                error = ""
                ready_sec = 0.0
                done_sec = 0.0
                wav_path = ""
                engine = "loaded TTS"
                started_at = time.perf_counter()
                try:
                    result = self.tts_runtime.speak(clean_text, play=True, player=player, sink_name=sink_name)
                    done_sec = time.perf_counter() - started_at
                    ready_sec = float(result.timing_sec.get("total", 0.0))
                    wav_path = str(result.final_wav_path)
                except Exception as exc:
                    done_sec = time.perf_counter() - started_at
                    ok = False
                    error = str(exc)
                    engine = "eSpeak fallback"
                    try:
                        self.live_plus_speaker.say(clean_text, sink_name=sink_name)
                    except Exception:
                        pass
                event_data = {
                    "event": "tts_done",
                    "ok": ok,
                    "source": source,
                    "text": clean_text,
                    "spoken_text": clean_text,
                    "error": error,
                    "timing_total": ready_sec,
                    "tts_ready_sec": ready_sec,
                    "tts_done_sec": done_sec,
                    "tts_engine": engine,
                    "wav_path": wav_path,
                }
                if report_event:
                    self.live_plus_queue.put(event_data)
                return event_data

            if background:
                threading.Thread(target=run_tts, daemon=True).start()
                return None
            return run_tts()

        started_at = time.perf_counter()
        self.live_plus_speaker.say(clean_text, sink_name=sink_name)
        done_sec = time.perf_counter() - started_at
        event_data = {
            "event": "tts_done",
            "ok": True,
            "source": source,
            "text": clean_text,
            "spoken_text": clean_text,
            "error": "",
            "timing_total": done_sec,
            "tts_ready_sec": done_sec,
            "tts_done_sec": done_sec,
            "tts_engine": "eSpeak",
            "wav_path": "",
        }
        if report_event:
            self.live_plus_queue.put(event_data)
        return event_data

    def _format_live_plus_tts_timing(self, item: dict) -> str:
        ready = item.get("tts_ready_sec", item.get("timing_total"))
        done = item.get("tts_done_sec")
        parts = []
        try:
            if ready is not None:
                parts.append(f"TTS ready {float(ready):.2f}s")
        except (TypeError, ValueError):
            pass
        try:
            if done is not None:
                parts.append(f"done {float(done):.2f}s")
        except (TypeError, ValueError):
            pass
        return " | ".join(parts)

    def _format_live_plus_sentence_timing(self, item: dict) -> str:
        parts = []
        try:
            llm_sec = item.get("llm_sec")
            if llm_sec is not None:
                parts.append(f"LLM {float(llm_sec):.2f}s")
        except (TypeError, ValueError):
            pass
        tts_timing = self._format_live_plus_tts_timing(item)
        if tts_timing:
            parts.append(tts_timing)
        return " | ".join(parts)

    def _live_plus_source_label(self, source: str | None) -> str:
        return "LLM" if str(source or "").lower() == "llm" else "buffer"

    def _format_live_plus_llm_detail(self, item: dict) -> str:
        model = str(item.get("llm_model") or "-")
        llm_input = str(item.get("llm_input") or item.get("input") or "-")
        llm_output = str(item.get("llm_output") or item.get("output") or "-")
        try:
            delay = f"{float(item.get('llm_sec')):.2f}s"
        except (TypeError, ValueError):
            delay = "-s"
        if item.get("ok"):
            return f"LLM: {model} | delay {delay} | buffer: {llm_input} | hasil: {llm_output}"
        return f"LLM: {model} gagal | delay {delay} | buffer: {llm_input} | error: {item.get('error', '-')}"

    def _format_live_plus_tts_detail(self, item: dict) -> str:
        source = self._live_plus_source_label(item.get("source"))
        spoken_text = str(item.get("spoken_text") or item.get("text") or item.get("output") or item.get("input") or "-")
        engine = str(item.get("tts_engine") or "-")
        timing = self._format_live_plus_tts_timing(item)
        timing_suffix = f" | {timing}" if timing else ""
        return f"Suara/TTS: {source} -> {spoken_text} | {engine}{timing_suffix}"

    def _warmup_live_plus_llm(self, model_name: str) -> None:
        def task() -> None:
            try:
                lp.warmup_sentence_llm(model=model_name)
            except Exception:
                pass

        threading.Thread(target=task, daemon=True).start()

    def _submit_live_plus_sentence(self, result: lp.FlushResult) -> None:
        input_text = result.text
        sink_name = self.selected_live_plus_audio_sink()
        if not bool(self.live_plus_use_llm_var.get()):
            self.live_plus_output_var.set(f"Output akhir: {input_text}")
            self.live_plus_llm_var.set("LLM: off")
            self.live_plus_tts_var.set(f"Suara/TTS: buffer -> {input_text} | speaking...")
            engine = "loaded TTS" if self._use_loaded_tts_for_live_plus() else "eSpeak"
            self.live_plus_status_var.set(f"LiveTest Plus: speaking via {engine} ({result.reason})")
            self._speak_live_plus_text(input_text, sink_name=sink_name, background=True, report_event=True, source="buffer")
            return

        self.live_plus_sentence_pending += 1
        self.live_plus_output_var.set(f"Output akhir: menyusun kalimat dari {input_text}")
        self.live_plus_status_var.set("LiveTest Plus: waiting local LLM")
        words = result.words
        allow_word_fix = bool(self.live_plus_allow_word_fix_var.get())
        model_name = self.live_plus_ollama_model_var.get().strip() or lp.DEFAULT_OLLAMA_MODEL
        self.live_plus_llm_var.set(f"LLM: waiting {model_name} | buffer: {input_text}")
        self.live_plus_tts_var.set("Suara/TTS: menunggu hasil LLM")

        def task() -> None:
            ok = True
            error = ""
            output_text = input_text
            llm_sec = 0.0
            tts_info: dict = {}
            llm_started_at = time.perf_counter()
            try:
                output_text = lp.OllamaSentenceClient(model=model_name).compose(words, allow_word_correction=allow_word_fix)
                llm_sec = time.perf_counter() - llm_started_at
            except Exception as exc:
                llm_sec = time.perf_counter() - llm_started_at
                ok = False
                error = str(exc)
            source = "llm" if ok else "buffer"
            try:
                tts_info = self._speak_live_plus_text(
                    output_text,
                    sink_name=sink_name,
                    background=False,
                    report_event=False,
                    source=source,
                ) or {}
            except Exception:
                pass
            self.live_plus_queue.put(
                {
                    "event": "plus_sentence",
                    "ok": ok,
                    "source": tts_info.get("source") or source,
                    "input": input_text,
                    "output": output_text,
                    "spoken_text": tts_info.get("spoken_text") or output_text,
                    "error": error,
                    "llm_model": model_name,
                    "llm_input": input_text,
                    "llm_output": output_text,
                    "llm_sec": llm_sec,
                    "tts_ready_sec": tts_info.get("tts_ready_sec"),
                    "tts_done_sec": tts_info.get("tts_done_sec"),
                    "tts_engine": tts_info.get("tts_engine"),
                }
            )

        threading.Thread(target=task, daemon=True).start()

    def _handle_live_plus_status(self, item: dict) -> None:
        label = item.get("prediction", "-")
        conf = float(item.get("confidence", 0.0))
        result = self.live_plus_buffer.observe_status(
            label=str(label),
            confidence=conf,
            prediction_id=item.get("prediction_id"),
            visible=bool(item.get("visible", False)),
            now=time.perf_counter(),
        )
        self._update_live_plus_buffer_text()
        if result is not None:
            self._submit_live_plus_sentence(result)

    def _build_live_start_request(self, status_queue: queue.Queue) -> tuple[str, dict]:
        live_schema = self.selected_live_schema()
        live_variant = self.selected_live_variant_value()
        live_variant = validate_live_checkpoint(live_schema, live_variant, model_data=self.selected_live_model_data_mode())
        stream_workers = int(self.live_stream_workers_var.get())
        mp_workers = int(self.live_mp_workers_var.get())
        inference_workers = int(self.live_inference_workers_var.get())
        if stream_workers <= 0 or mp_workers <= 0 or inference_workers <= 0:
            raise ValueError("Worker count harus > 0.")
        threshold = self.selected_live_confidence_threshold()
        specialist_enabled = bool(self.live_specialist_enabled_var.get())
        raw_specialist = self.live_specialist_name_var.get().strip()
        if _is_all_specialist_request(raw_specialist):
            specialist_name = "all"
        else:
            specialist_name = gm.normalize_specialist_name(raw_specialist or self.specialist_name_var.get().strip() or "all")
        live_kwargs = {
            "status_queue": status_queue,
            "profile": self.mode_var.get(),
            "device": self.live_device_var.get(),
            "schema": live_schema,
            "route": self.selected_live_route_value(),
            "stream_workers": stream_workers,
            "mp_workers": mp_workers,
            "inference_workers": inference_workers,
            "mp_method": self.live_mp_method_var.get(),
            "specialist_enabled": specialist_enabled,
            "specialist_name": specialist_name,
        }
        if threshold is not None:
            live_kwargs["confidence_threshold"] = threshold
        return live_variant, live_kwargs

    def toggle_live_plus(self) -> None:
        if getattr(self, "live_plus_reset_job", None) is not None:
            self.live_plus_status_var.set("LiveTest Plus: waiting camera release")
            return
        if self.live_plus_worker is not None and self.live_plus_worker.is_alive():
            live_session.request_stop(self.live_plus_worker)
            self.live_plus_stop_started_at = time.perf_counter()
            self.btn_live_plus.configure(text="Stopping...", state="disabled")
            self.live_plus_status_var.set("LiveTest Plus: stopping")
            if self.live_plus_poll_job is None:
                self.live_plus_poll_job = self.root.after(100, self._poll_live_plus)
            return
        if self.live_plus_worker is not None:
            self._reset_live_plus_ui()
        if self._normal_live_running_or_releasing():
            messagebox.showwarning("LiveTest Plus", "Stop Live Test biasa dulu sebelum menjalankan LiveTest Plus.")
            return

        self.live_plus_queue = queue.Queue()
        self.live_plus_stop_started_at = None
        self.live_plus_sentence_pending = 0
        self.live_plus_buffer.reset()
        self._update_live_plus_buffer_text()
        self.live_plus_output_var.set("Output akhir: -")
        self.live_plus_llm_var.set("LLM: -")
        self.live_plus_tts_var.set("Suara/TTS: -")
        if bool(self.live_plus_use_llm_var.get()):
            model_name = self.live_plus_ollama_model_var.get().strip() or lp.DEFAULT_OLLAMA_MODEL
            self.live_plus_llm_var.set(f"LLM: warming {model_name}")
            self._warmup_live_plus_llm(model_name)
        try:
            live_variant, live_kwargs = self._build_live_start_request(self.live_plus_queue)
        except Exception as exc:
            messagebox.showerror("LiveTest Plus", str(exc))
            return

        def start_after_tts() -> None:
            try:
                self.live_plus_worker = live_gru_fast.start_live_inference(
                    live_variant,
                    **live_kwargs,
                )
            except Exception as exc:
                self.release_tts_after_live("plus")
                self._reset_live_start_button("plus")
                messagebox.showerror("LiveTest Plus", str(exc))
                return
            self.btn_live_plus.configure(text="Stop LiveTest Plus", state="normal")
            self.live_plus_status_var.set("LiveTest Plus: starting")
            if self.live_plus_poll_job is not None:
                try:
                    self.root.after_cancel(self.live_plus_poll_job)
                except Exception:
                    pass
                self.live_plus_poll_job = None
            self._poll_live_plus()

        self.ensure_tts_loaded_for_live("plus", start_after_tts)

    def _poll_live_plus(self) -> None:
        self.live_plus_poll_job = None
        error_message = None
        stopped_message = None

        def process_event(item: dict) -> None:
            nonlocal error_message, stopped_message
            event = item.get("event")
            if event == "plus_sentence":
                self.live_plus_sentence_pending = max(0, self.live_plus_sentence_pending - 1)
                output = str(item.get("output") or item.get("input") or "")
                self.live_plus_output_var.set(f"Output akhir: {output or '-'}")
                self.live_plus_llm_var.set(self._format_live_plus_llm_detail(item))
                self.live_plus_tts_var.set(self._format_live_plus_tts_detail(item))
                if item.get("ok"):
                    self.live_plus_status_var.set("LiveTest Plus: LLM/TTS output ready")
                else:
                    self.live_plus_status_var.set(f"LiveTest Plus: LLM gagal, fallback audio | {item.get('error', '-')}")
            elif event == "tts_done":
                if item.get("ok"):
                    spoken_text = str(item.get("spoken_text") or item.get("text") or "-")
                    self.live_plus_output_var.set(f"Output akhir: {spoken_text}")
                    self.live_plus_llm_var.set("LLM: off")
                    self.live_plus_tts_var.set(self._format_live_plus_tts_detail(item))
                    self.live_plus_status_var.set("LiveTest Plus: TTS output ready")
                else:
                    self.live_plus_tts_var.set(self._format_live_plus_tts_detail(item))
                    self.live_plus_status_var.set(f"LiveTest Plus: TTS gagal, fallback eSpeak | {item.get('error', '-')}")
            elif event == "error":
                error_message = str(item.get("message"))
                self.live_plus_status_var.set(f"LiveTest Plus error: {error_message}")
            elif event == "camera_retry":
                self.live_plus_status_var.set(f"LiveTest Plus: {item.get('message')}")
            elif event == "stopped":
                closed = " | window closed" if item.get("window_closed") else ""
                released = " | camera released" if item.get("camera_released") else ""
                thread_alive = " | camera thread still alive" if item.get("camera_thread_alive_after_release") else ""
                mp_alive = " | MediaPipe thread still alive" if item.get("mp_thread_alive_after_stop") else ""
                release_ms = item.get("camera_release_ms")
                release_text = f"{released}{thread_alive}{mp_alive}"
                if release_ms is not None:
                    release_text += f" | release {float(release_ms):.0f} ms"
                stopped_message = f"LiveTest Plus: {item.get('message')}{closed}{release_text} | {self._format_live_capture_status(item)}"
                self.live_plus_status_var.set(stopped_message)
            elif event == "started":
                warning = item.get("warning") or ""
                suffix = f" | {warning}" if warning else ""
                reason = item.get("device_reason") or "-"
                capture_text = self._format_live_capture_status(item)
                route_text = display_live_route_name(item.get("route", self.selected_live_route_value()))
                mp_backend = item.get("mp_backend") or "-"
                specialist_text = self._format_live_specialist_status(item)
                self.live_plus_status_var.set(
                    f"LiveTest Plus: {item.get('schema', '-')}:{item.get('feature_dim', '-')}D | gru_{item.get('variant')} | "
                    f"route {route_text} | mp {mp_backend} | {specialist_text} | {capture_text} | device {item.get('device')} | "
                    f"{item.get('runtime', '-')} | {reason}{suffix}"
                )
            elif event == "status":
                label = item.get("prediction", "-")
                conf = float(item.get("confidence", 0.0))
                fps_camera = float(item.get("fps_camera", item.get("fps", 0.0)))
                fps_predict = float(item.get("fps_predict", 0.0))
                raw = item.get("raw_prediction", "-")
                visible = bool(item.get("visible", False))
                specialist_text = self._format_live_specialist_status(item)
                self.live_plus_status_var.set(
                    f"LiveTest Plus: {label} ({conf:.2f}) raw {raw} | {specialist_text} | visible {int(visible)} | "
                    f"cam {fps_camera:.1f} fps | pred {fps_predict:.1f} fps"
                )
                self._handle_live_plus_status(item)

        while not self.live_plus_queue.empty():
            process_event(self.live_plus_queue.get())

        worker_alive = self.live_plus_worker is not None and self.live_plus_worker.is_alive()
        if worker_alive:
            stop_started_at = getattr(self, "live_plus_stop_started_at", None)
            if stop_started_at is not None:
                elapsed = time.perf_counter() - stop_started_at
                if elapsed >= 6.0:
                    live_session.request_stop(self.live_plus_worker)
                    self.btn_live_plus.configure(text="Releasing camera...", state="disabled")
                    self.live_plus_status_var.set(f"LiveTest Plus: forcing camera cleanup ({elapsed:.1f}s)")
                if elapsed >= 10.0:
                    stale_message = "LiveTest Plus: force reset after stop timeout. Camera release requested."
                    live_session.stop_worker(self.live_plus_worker, join_timeout=3.0, force=True)
                    while not self.live_plus_queue.empty():
                        process_event(self.live_plus_queue.get())
                    self._reset_live_plus_ui(stale_message, cooldown_ms=500)
                    return
            self.live_plus_poll_job = self.root.after(250, self._poll_live_plus)
        else:
            if self.live_plus_worker is not None:
                live_session.join_worker(self.live_plus_worker, timeout=0.2)
                while not self.live_plus_queue.empty():
                    process_event(self.live_plus_queue.get())
                cooldown_ms = 500 if stopped_message else 0
                self._reset_live_plus_ui(stopped_message or self.live_plus_status_var.get(), cooldown_ms=cooldown_ms)
            if self.live_plus_sentence_pending > 0:
                self.live_plus_poll_job = self.root.after(250, self._poll_live_plus)
            if error_message:
                messagebox.showwarning("LiveTest Plus", error_message)

    def toggle_live(self) -> None:
        if getattr(self, "live_reset_job", None) is not None:
            self.live_status_var.set("Live: waiting camera release")
            return
        if self.live_worker is not None and self.live_worker.is_alive():
            live_session.request_stop(self.live_worker)
            self.live_stop_started_at = time.perf_counter()
            self.btn_live.configure(text="Stopping...", state="disabled")
            self.live_status_var.set("Live: stopping")
            if self.live_poll_job is None:
                self.live_poll_job = self.root.after(100, self._poll_live)
            return
        if self.live_worker is not None:
            self._reset_live_ui()
        if self._live_plus_running_or_releasing():
            messagebox.showwarning("Live Test", "Stop LiveTest Plus dulu sebelum menjalankan Live Test biasa.")
            return

        self.live_queue = queue.Queue()
        self.live_stop_started_at = None
        try:
            live_variant, live_kwargs = self._build_live_start_request(self.live_queue)
        except Exception as exc:
            messagebox.showerror("Live Test", str(exc))
            return

        def start_after_tts() -> None:
            try:
                self.live_worker = live_gru_fast.start_live_inference(
                    live_variant,
                    **live_kwargs,
                )
            except Exception as exc:
                self.release_tts_after_live("live")
                self._reset_live_start_button("live")
                messagebox.showerror("Live Test", str(exc))
                return
            self.btn_live.configure(text="Stop Live Test", state="normal")
            self.live_status_var.set("Live: starting")
            if self.live_poll_job is not None:
                try:
                    self.root.after_cancel(self.live_poll_job)
                except Exception:
                    pass
                self.live_poll_job = None
            self._poll_live()

        self.ensure_tts_loaded_for_live("live", start_after_tts)

    def _poll_live(self) -> None:
        self.live_poll_job = None
        error_message = None
        stopped_message = None

        def process_event(item: dict) -> None:
            nonlocal error_message, stopped_message
            event = item.get("event")
            if event == "error":
                error_message = str(item.get("message"))
                self.live_status_var.set(f"Live error: {error_message}")
            elif event == "camera_retry":
                self.live_status_var.set(f"Live: {item.get('message')}")
            elif event == "stopped":
                closed = " | window closed" if item.get("window_closed") else ""
                released = " | camera released" if item.get("camera_released") else ""
                thread_alive = " | camera thread still alive" if item.get("camera_thread_alive_after_release") else ""
                mp_alive = " | MediaPipe thread still alive" if item.get("mp_thread_alive_after_stop") else ""
                release_ms = item.get("camera_release_ms")
                release_text = f"{released}{thread_alive}{mp_alive}"
                if release_ms is not None:
                    release_text += f" | release {float(release_ms):.0f} ms"
                stopped_message = f"Live: {item.get('message')}{closed}{release_text} | {self._format_live_capture_status(item)}"
                self.live_status_var.set(stopped_message)
            elif event == "started":
                warning = item.get("warning") or ""
                suffix = f" | {warning}" if warning else ""
                reason = item.get("device_reason") or "-"
                capture_text = self._format_live_capture_status(item)
                route_text = display_live_route_name(item.get("route", self.selected_live_route_value()))
                mp_backend = item.get("mp_backend") or "-"
                specialist_text = self._format_live_specialist_status(item)
                self.live_status_var.set(
                    f"Live: {item.get('schema', '-')}:{item.get('feature_dim', '-')}D | gru_{item.get('variant')} | "
                    f"route {route_text} | mp {mp_backend} | {specialist_text} | {capture_text} | device {item.get('device')} | "
                    f"{item.get('runtime', '-')} | {reason}{suffix}"
                )
            elif event == "status":
                label = item.get("prediction", "-")
                conf = float(item.get("confidence", 0.0))
                fps_camera = float(item.get("fps_camera", item.get("fps", 0.0)))
                fps_predict = float(item.get("fps_predict", 0.0))
                extract_ms = float(item.get("extract_ms", 0.0))
                model_ms = float(item.get("model_ms", 0.0))
                buf = int(item.get("buffer", 0))
                target = int(item.get("target_frames", 0))
                raw = item.get("raw_prediction", "-")
                capture_text = self._format_live_capture_status(item)
                mp_backend = item.get("mp_backend") or "-"
                mp_ready = "ready" if item.get("mp_ready") else "warming"
                specialist_text = self._format_live_specialist_status(item)
                self.live_status_var.set(
                    f"Live: {item.get('schema', '-')}:{item.get('feature_dim', '-')}D | gru_{item.get('variant')} | "
                    f"mp {mp_backend}:{mp_ready} | {specialist_text} | {capture_text} | {label} ({conf:.2f}) raw {raw} | cam {fps_camera:.1f} fps | pred {fps_predict:.1f} fps | "
                    f"extract {extract_ms:.1f} ms | model {model_ms:.1f} ms | buffer {buf}/{target}"
                )

        while not self.live_queue.empty():
            process_event(self.live_queue.get())

        if self.live_worker is not None and self.live_worker.is_alive():
            stop_started_at = getattr(self, "live_stop_started_at", None)
            if stop_started_at is not None:
                elapsed = time.perf_counter() - stop_started_at
                if elapsed >= 6.0:
                    live_session.request_stop(self.live_worker)
                    self.btn_live.configure(text="Releasing camera...", state="disabled")
                    self.live_status_var.set(f"Live: forcing camera cleanup ({elapsed:.1f}s)")
                if elapsed >= 10.0:
                    stale_message = "Live: force reset after stop timeout. Camera release requested."
                    live_session.stop_worker(self.live_worker, join_timeout=3.0, force=True)
                    while not self.live_queue.empty():
                        process_event(self.live_queue.get())
                    self._reset_live_ui(stale_message, cooldown_ms=500)
                    return
            self.live_poll_job = self.root.after(250, self._poll_live)
        else:
            if self.live_worker is not None:
                live_session.join_worker(self.live_worker, timeout=0.2)
                while not self.live_queue.empty():
                    process_event(self.live_queue.get())
                cooldown_ms = 500 if stopped_message else 0
                self._reset_live_ui(stopped_message or self.live_status_var.get(), cooldown_ms=cooldown_ms)
            if error_message:
                messagebox.showwarning("Live Test", error_message)

    def _parse_reinforcement_args(self) -> tuple[str, str, str, str, int, float, int]:
        schema = resolve_live_schema_name(self.rl_schema_var.get())
        variant = self.rl_variant_var.get().strip() or "auto"
        route = resolve_live_route_name(self.rl_route_var.get())
        profile = self.rl_profile_var.get().strip() or live_gru_fast.DEFAULT_LIVE_PROFILE
        try:
            camera = int(self.rl_camera_var.get())
            lr_value = float(self.rl_lr_var.get())
            steps = int(self.rl_steps_var.get())
        except ValueError as exc:
            raise ValueError("Camera, LR, dan Steps/koreksi harus angka.") from exc
        if lr_value <= 0:
            raise ValueError("LR harus > 0.")
        if steps <= 0:
            raise ValueError("Steps/koreksi harus > 0.")
        return schema, variant, route, profile, camera, lr_value, steps

    def refresh_reinforcement_labels(self) -> None:
        labels: list[str] = []
        try:
            if self.rl_session is not None and self.rl_session.started:
                labels = self.rl_session.available_labels()
            else:
                schema, variant, _route, _profile, _camera, _lr_value, _steps = self._parse_reinforcement_args()
                resolved = validate_live_checkpoint(schema, variant, model_data=self.selected_rl_model_data_mode())
                labels_map = gm.load_labels(resolved, schema=schema)
                labels = [labels_map[idx] for idx in sorted(labels_map)]
        except Exception as exc:
            self.rl_status_var.set(f"Reinforcement: label belum siap | {exc}")
        if hasattr(self, "rl_label_combo"):
            self.rl_label_combo.configure(values=labels)
        current = self.rl_correct_label_var.get()
        if labels and current not in labels:
            self.rl_correct_label_var.set(labels[0])
        elif not labels:
            self.rl_correct_label_var.set("")

    def _set_reinforcement_buttons(self, *, running: bool, can_correct: bool = False) -> None:
        if hasattr(self, "btn_rl_start"):
            self.btn_rl_start.configure(state="disabled" if running else "normal")
        if hasattr(self, "btn_rl_finish"):
            self.btn_rl_finish.configure(state="normal" if running else "disabled")
        correction_state = "normal" if running and can_correct else "disabled"
        if hasattr(self, "btn_rl_correct"):
            self.btn_rl_correct.configure(state=correction_state)
        if hasattr(self, "btn_rl_wrong"):
            self.btn_rl_wrong.configure(state=correction_state)

    def start_reinforcement_session(self) -> None:
        if self.rl_worker is not None and self.rl_worker.is_alive():
            self.rl_status_var.set("Reinforcement: session masih berjalan")
            return
        if self._normal_live_running_or_releasing() or self._live_plus_running_or_releasing():
            messagebox.showwarning("Reinforcement", "Stop Live Test/LiveTest Plus dulu sebelum menjalankan Reinforcement.")
            return
        try:
            schema, variant, route, profile, camera, lr_value, steps = self._parse_reinforcement_args()
            variant = validate_live_checkpoint(schema, variant, model_data=self.selected_rl_model_data_mode())
            session = rl.ReinforcementSession(
                schema=schema,
                variant=variant,
                route=route,
                device=self.rl_device_var.get(),
                lr=lr_value,
                steps_per_correction=steps,
            )
            info = session.start()
        except Exception as exc:
            self.rl_status_var.set(f"Reinforcement: gagal start | {exc}")
            messagebox.showerror("Reinforcement", str(exc))
            self._set_reinforcement_buttons(running=False)
            return

        self.rl_session = session
        self.rl_queue = queue.Queue()
        self.rl_last_sequence = None
        self.rl_last_prediction = {}
        self.rl_prediction_var.set("Prediksi: -")
        self.rl_backup_var.set("Backup: -")
        self.rl_checkpoint_var.set(f"Checkpoint RL: {info.get('checkpoint', '-')}")
        self.refresh_reinforcement_labels()
        try:
            self.rl_worker = live_gru_fast.start_live_inference(
                session.resolved_variant,
                status_queue=self.rl_queue,
                profile=profile,
                device=self.rl_device_var.get(),
                camera_index=camera,
                model_dir=str(session.model_root),
                schema=schema,
                route=route,
                show_window=True,
                include_sequences=True,
                stream_workers=1,
                mp_workers=1,
                inference_workers=1,
            )
        except Exception as exc:
            try:
                backup_dir = session.finish()
                self.rl_backup_var.set(f"Backup: {backup_dir}")
            except Exception:
                pass
            self.rl_session = None
            self.rl_worker = None
            self.rl_status_var.set(f"Reinforcement: live gagal | {exc}")
            messagebox.showerror("Reinforcement", str(exc))
            self._set_reinforcement_buttons(running=False)
            return
        self.rl_status_var.set(f"Reinforcement: running {schema}/gru_{session.resolved_variant} route {route}")
        self._set_reinforcement_buttons(running=True, can_correct=False)
        if self.rl_poll_job is not None:
            try:
                self.root.after_cancel(self.rl_poll_job)
            except Exception:
                pass
        self._poll_reinforcement()

    def finish_reinforcement_session(self) -> None:
        if self.rl_poll_job is not None:
            try:
                self.root.after_cancel(self.rl_poll_job)
            except Exception:
                pass
            self.rl_poll_job = None
        worker = self.rl_worker
        alive = live_session.stop_worker(worker, join_timeout=1.0, force=True) if worker is not None else False
        if alive:
            self.rl_status_var.set("Reinforcement: masih melepas kamera")
            self.rl_poll_job = self.root.after(300, self._poll_reinforcement)
            return
        self._finish_reinforcement_backup("Reinforcement: session selesai")

    def _finish_reinforcement_backup(self, status_prefix: str) -> None:
        session = self.rl_session
        self.rl_worker = None
        self.rl_last_sequence = None
        self.rl_last_prediction = {}
        backup_message = ""
        if session is not None:
            try:
                backup_dir = session.finish()
                backup_message = str(backup_dir)
                self.rl_backup_var.set(f"Backup: {backup_dir}")
            except Exception as exc:
                backup_message = f"backup gagal: {exc}"
                self.rl_backup_var.set(f"Backup: {backup_message}")
        self.rl_session = None
        self.rl_status_var.set(f"{status_prefix} | {backup_message}" if backup_message else status_prefix)
        self._set_reinforcement_buttons(running=False)

    def _poll_reinforcement(self) -> None:
        self.rl_poll_job = None
        error_message = None
        stopped_message = None

        def process_event(item: dict) -> None:
            nonlocal error_message, stopped_message
            event = item.get("event")
            if event == "error":
                error_message = str(item.get("message"))
                self.rl_status_var.set(f"Reinforcement error: {error_message}")
            elif event == "camera_retry":
                self.rl_status_var.set(f"Reinforcement: {item.get('message')}")
            elif event == "stopped":
                closed = " | window closed" if item.get("window_closed") else ""
                released = " | camera released" if item.get("camera_released") else ""
                stopped_message = f"Reinforcement: {item.get('message')}{closed}{released}"
                self.rl_status_var.set(stopped_message)
            elif event == "started":
                self.rl_status_var.set(
                    f"Reinforcement: started {item.get('schema', '-')}:{item.get('feature_dim', '-')}D | "
                    f"gru_{item.get('variant')} | route {item.get('route', '-')}"
                )
            elif event == "status":
                label = str(item.get("prediction") or "-")
                conf = float(item.get("confidence") or 0.0)
                raw_label = str(item.get("raw_prediction") or "-")
                raw_conf = float(item.get("raw_confidence") or 0.0)
                chosen_label = label if label != "-" else raw_label
                chosen_conf = conf if label != "-" else raw_conf
                top = item.get("top") or []
                top_text = ", ".join(f"{name}:{float(score):.2f}" for name, score in top[:3]) if top else "-"
                self.rl_prediction_var.set(
                    f"Prediksi: {chosen_label} ({chosen_conf:.2f}) | raw {raw_label} ({raw_conf:.2f}) | top {top_text}"
                )
                sequence = item.get("sequence")
                if sequence is not None and chosen_label and chosen_label != "-":
                    self.rl_last_sequence = sequence
                    self.rl_last_prediction = {
                        "label": chosen_label,
                        "confidence": chosen_conf,
                        "prediction_id": int(item.get("prediction_id") or item.get("raw_prediction_id") or 0),
                    }
                    if self.rl_session is not None:
                        self._set_reinforcement_buttons(running=True, can_correct=True)
                        labels = self.rl_session.available_labels()
                        if chosen_label in labels:
                            self.rl_correct_label_var.set(chosen_label)

        while not self.rl_queue.empty():
            process_event(self.rl_queue.get())

        worker_alive = self.rl_worker is not None and self.rl_worker.is_alive()
        if worker_alive:
            self.rl_poll_job = self.root.after(250, self._poll_reinforcement)
            return
        if self.rl_worker is not None:
            try:
                self.rl_worker.join(timeout=0.2)
            except Exception:
                pass
            while not self.rl_queue.empty():
                process_event(self.rl_queue.get())
            self._finish_reinforcement_backup(stopped_message or self.rl_status_var.get())
        if error_message:
            messagebox.showwarning("Reinforcement", error_message)

    def _apply_reinforcement_correction(self, corrected_label: str) -> None:
        if self.rl_session is None or not self.rl_session.started:
            messagebox.showwarning("Reinforcement", "Sesi reinforcement belum berjalan.")
            return
        if self.rl_last_sequence is None:
            messagebox.showwarning("Reinforcement", "Belum ada sequence prediksi yang bisa dikoreksi.")
            return
        if self.rl_train_thread is not None and self.rl_train_thread.is_alive():
            self.rl_status_var.set("Reinforcement: koreksi sebelumnya masih training")
            return
        label = str(corrected_label or "").strip()
        if not label:
            messagebox.showwarning("Reinforcement", "Pilih label koreksi dulu.")
            return
        sequence = self.rl_last_sequence
        prediction = dict(self.rl_last_prediction)
        self._set_reinforcement_buttons(running=True, can_correct=False)
        self.rl_status_var.set(f"Reinforcement: applying correction -> {label}")

        def task() -> None:
            ok = True
            message = ""
            try:
                result = self.rl_session.apply_correction(
                    sequence=sequence,
                    corrected_label=label,
                    predicted_label=str(prediction.get("label", "-")),
                    confidence=float(prediction.get("confidence", 0.0)),
                    prediction_id=int(prediction.get("prediction_id", 0)),
                )
                message = (
                    f"Reinforcement: koreksi #{result['correction_count']} -> {label} | "
                    f"loss {float(result['loss']):.4f}"
                )
                checkpoint = str(result.get("checkpoint", "-"))
            except Exception as exc:
                ok = False
                message = f"Reinforcement: koreksi gagal | {exc}"
                checkpoint = ""

            def finish() -> None:
                self.rl_status_var.set(message)
                if checkpoint:
                    self.rl_checkpoint_var.set(f"Checkpoint RL: {checkpoint}")
                self._set_reinforcement_buttons(running=self.rl_session is not None, can_correct=ok and self.rl_last_sequence is not None)
                if not ok:
                    messagebox.showerror("Reinforcement", message)

            self.root.after(0, finish)

        self.rl_train_thread = threading.Thread(target=task, daemon=True)
        self.rl_train_thread.start()

    def mark_reinforcement_correct(self) -> None:
        label = str(self.rl_last_prediction.get("label", "") if self.rl_last_prediction else "")
        self._apply_reinforcement_correction(label)

    def mark_reinforcement_wrong(self) -> None:
        self._apply_reinforcement_correction(self.rl_correct_label_var.get())

    def open_evaluator(self) -> None:
        subprocess.Popen([sys.executable, str(gm.ROOT_DIR / "src" / "eva_dashboard.py")])


if __name__ == "__main__":
    root = tk.Tk()
    AppUI(root)
    root.mainloop()
