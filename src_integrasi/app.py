"""Lightweight Tkinter GUI for BISINDO live assistant + Sherpa UDP STT."""

from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src_integrasi.bisindo_panel import (
    AGE_OPTIONS,
    GENDER_OPTIONS,
    GuiBISINDOController,
    build_tts_profile_name,
    shortcut_action,
)
from src_integrasi.configuration import load_configuration
from src_integrasi.hotspot_broadcast import BroadcastError, HotspotBroadcaster
from src_integrasi.sherpa_backend import (
    DEFAULT_SHERPA_MODEL_DIR,
    LazySherpaRecognizer,
    SherpaBackendError,
    SherpaModelConfig,
    UdpConfig,
    UdpSherpaWorker,
    calculate_wer,
    transcribe_wav,
)


ENABLE_SHERPA_TEST_TAB = True
POLL_MS = 80
DEFAULT_WINDOW_SIZE = "820x620"


def enabled_tab_names(enable_sherpa_test_tab: bool = ENABLE_SHERPA_TEST_TAB) -> tuple[str, ...]:
    names = ["BISINDO", "Profile", "Posisi", "Sherpa"]
    if enable_sherpa_test_tab:
        names.append("WER")
    return tuple(names)


def _set_text(widget: ScrolledText, text: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)
    widget.insert(tk.END, text)
    widget.configure(state="disabled")


def _append_text(widget: ScrolledText, text: str) -> None:
    widget.configure(state="normal")
    widget.insert(tk.END, text)
    widget.see(tk.END)
    widget.configure(state="disabled")


class IntegrationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("BISINDO + Sherpa")
        self.root.geometry(DEFAULT_WINDOW_SIZE)
        self.root.minsize(720, 520)

        self.bisindo = GuiBISINDOController(config=load_configuration())
        self.specialist_enabled_var = tk.BooleanVar(value=True)
        self.broadcast = HotspotBroadcaster()
        self.sherpa_queue: queue.Queue = queue.Queue()
        self.sherpa_recognizer = LazySherpaRecognizer(SherpaModelConfig(model_dir=DEFAULT_SHERPA_MODEL_DIR))
        self.sherpa_worker: UdpSherpaWorker | None = None
        self.sherpa_segments: list[str] = []
        self.wav_path: Path | None = None

        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(POLL_MS, self._poll)

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.tabs: dict[str, ttk.Frame] = {}
        for name in enabled_tab_names():
            frame = ttk.Frame(self.notebook, padding=8)
            self.tabs[name] = frame
            self.notebook.add(frame, text=name)
        self.notebook.pack(fill="both", expand=True)
        self._build_bisindo_tab(self.tabs["BISINDO"])
        self._build_profile_tab(self.tabs["Profile"])
        self._build_position_tab(self.tabs["Posisi"])
        self._build_sherpa_tab(self.tabs["Sherpa"])
        if ENABLE_SHERPA_TEST_TAB and "WER" in self.tabs:
            self._build_test_tab(self.tabs["WER"])

    def _build_bisindo_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill="x")
        ttk.Button(controls, text="Start", command=self.bisindo.start).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Pause/Resume", command=self.bisindo.pause_resume).pack(side="left", padx=6)
        ttk.Button(controls, text="Preview", command=self.bisindo.toggle_preview).pack(side="left", padx=6)
        ttk.Button(controls, text="Space {spasi}", command=self.bisindo.append_space).pack(side="left", padx=6)
        ttk.Checkbutton(
            controls,
            text="Auto specialist",
            variable=self.specialist_enabled_var,
            command=self.apply_specialist,
        ).pack(side="left", padx=6)
        ttk.Button(controls, text="Quit", command=self.on_close).pack(side="right")

        self.live_status = tk.StringVar(value="Status: belum start")
        self.live_buffer = tk.StringVar(value="Buffer: -")
        self.live_profile = tk.StringVar(value="TTS: cewek_dewasa_default")
        ttk.Label(parent, textvariable=self.live_status).pack(anchor="w", pady=(10, 2))
        ttk.Label(parent, textvariable=self.live_buffer).pack(anchor="w", pady=2)
        ttk.Label(parent, textvariable=self.live_profile).pack(anchor="w", pady=2)

        shortcuts = ttk.LabelFrame(parent, text="Shortcut", padding=8)
        shortcuts.pack(fill="x", pady=8)
        text = "p: pause/resume | v: preview | space: tambah {spasi} | q/Esc: keluar"
        ttk.Label(shortcuts, text=text).pack(anchor="w")

        self.bisindo_output = tk.StringVar(value="Output akhir: -")
        ttk.Label(parent, textvariable=self.bisindo_output, wraplength=760).pack(anchor="w", pady=(2, 8))

        ttk.Label(parent, text="Log ringkas").pack(anchor="w")
        self.bisindo_log = ScrolledText(parent, height=12, wrap=tk.WORD, state="disabled")
        self.bisindo_log.pack(fill="both", expand=True)

    def _build_profile_tab(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(anchor="w", fill="x", pady=4)
        ttk.Label(row, text="Gender", width=12).pack(side="left")
        self.gender_var = tk.StringVar(value="Cewek")
        ttk.Combobox(row, textvariable=self.gender_var, values=GENDER_OPTIONS, state="readonly", width=16).pack(
            side="left", padx=6
        )

        row2 = ttk.Frame(parent)
        row2.pack(anchor="w", fill="x", pady=4)
        ttk.Label(row2, text="Usia", width=12).pack(side="left")
        self.age_var = tk.StringVar(value="Dewasa")
        ttk.Combobox(row2, textvariable=self.age_var, values=AGE_OPTIONS, state="readonly", width=16).pack(
            side="left", padx=6
        )

        ttk.Button(parent, text="Terapkan Profile", command=self.apply_profile).pack(anchor="w", pady=10)
        self.profile_status = tk.StringVar(value="Aktif: cewek_dewasa_default")
        ttk.Label(parent, textvariable=self.profile_status).pack(anchor="w", pady=4)
        ttk.Label(parent, text="Variasi tetap default. Nama profile: {gender}_{usia}_default.").pack(anchor="w", pady=4)

    def _build_position_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Acuan posisi tubuh diambil dari posisi BAHU (MediaPipe). "
                "Berdiri di posisi yang pas dengan bahu terlihat kamera, "
                "lalu klik Kalibrasi untuk menyimpannya sebagai acuan CAMPOS."
            ),
            wraplength=760,
        ).pack(anchor="w", pady=(0, 8))

        self.pos_live = tk.StringVar(value="Bahu sekarang: - (live belum jalan)")
        self.pos_ref = tk.StringVar(value="Acuan tersimpan: -")
        self.pos_campos = tk.StringVar(value="CAMPOS: -")
        ttk.Label(parent, textvariable=self.pos_campos, font=("TkDefaultFont", 12, "bold")).pack(
            anchor="w", pady=4
        )
        ttk.Label(parent, textvariable=self.pos_live).pack(anchor="w", pady=2)
        ttk.Label(parent, textvariable=self.pos_ref).pack(anchor="w", pady=2)

        tol = ttk.LabelFrame(parent, text="Acceptable error (toleransi, fraksi frame 0-1)", padding=8)
        tol.pack(fill="x", pady=8)
        snap = self.bisindo.camera_snapshot()
        row = ttk.Frame(tol)
        row.pack(anchor="w", fill="x")
        ttk.Label(row, text="Toleransi X", width=12).pack(side="left")
        self.tol_x_var = tk.StringVar(value=f"{snap['tol_x']:.2f}")
        ttk.Entry(row, textvariable=self.tol_x_var, width=8).pack(side="left", padx=6)
        ttk.Label(row, text="Toleransi Y", width=12).pack(side="left", padx=(12, 0))
        self.tol_y_var = tk.StringVar(value=f"{snap['tol_y']:.2f}")
        ttk.Entry(row, textvariable=self.tol_y_var, width=8).pack(side="left", padx=6)
        ttk.Button(row, text="Terapkan toleransi", command=self.apply_camera_tolerance).pack(
            side="left", padx=12
        )

        self.flip_var = tk.BooleanVar(value=bool(snap["flip_horizontal"]))
        ttk.Checkbutton(
            parent,
            text="Preview mirror (balik arah KIRI/KANAN)",
            variable=self.flip_var,
            command=self.apply_camera_flip,
        ).pack(anchor="w", pady=4)

        ttk.Button(parent, text="Kalibrasi (simpan posisi sekarang)", command=self.calibrate_camera).pack(
            anchor="w", pady=10
        )
        self.pos_status = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.pos_status, wraplength=760).pack(anchor="w", pady=4)

    def _build_sherpa_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill="x")
        self.udp_button = ttk.Button(controls, text="Konek UDP", command=self.toggle_udp_broadcast)
        self.udp_button.pack(side="left", padx=(0, 6))
        self.stt_button = ttk.Button(controls, text="Start STT", command=self.toggle_stt)
        self.stt_button.pack(side="left", padx=6)

        self.broadcast_status = tk.StringVar(value="Broadcast: mati")
        self.udp_status = tk.StringVar(value="UDP: belum listen")
        self.recognizer_status = tk.StringVar(value=f"Model: {DEFAULT_SHERPA_MODEL_DIR}")
        self.packet_status = tk.StringVar(value="Packet: 0 | sender: -")
        self.latency_status = tk.StringVar(value="Delay decode: -")
        self.partial_status = tk.StringVar(value="Mengetik: -")
        for var in (
            self.broadcast_status,
            self.udp_status,
            self.recognizer_status,
            self.packet_status,
            self.latency_status,
            self.partial_status,
        ):
            ttk.Label(parent, textvariable=var, wraplength=760).pack(anchor="w", pady=2)

        ttk.Label(parent, text="Transcript").pack(anchor="w", pady=(8, 2))
        self.sherpa_text = ScrolledText(parent, height=14, wrap=tk.WORD, state="disabled")
        self.sherpa_text.pack(fill="both", expand=True)

    def _build_test_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Tab ini bisa dimatikan dengan ENABLE_SHERPA_TEST_TAB = False di app.py").pack(
            anchor="w", pady=(0, 8)
        )
        grid = ttk.Frame(parent)
        grid.pack(fill="both", expand=True)

        left = ttk.Frame(grid)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = ttk.Frame(grid)
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        ttk.Label(left, text="Reference").pack(anchor="w")
        self.wer_reference = ScrolledText(left, height=5, wrap=tk.WORD)
        self.wer_reference.pack(fill="x", pady=2)
        ttk.Label(left, text="Hypothesis").pack(anchor="w")
        self.wer_hypothesis = ScrolledText(left, height=5, wrap=tk.WORD)
        self.wer_hypothesis.pack(fill="x", pady=2)
        ttk.Button(left, text="Ambil Hyp dari Sherpa", command=self.copy_sherpa_to_hypothesis).pack(anchor="w", pady=4)
        ttk.Button(left, text="Hitung WER", command=self.calculate_current_wer).pack(anchor="w", pady=4)

        ttk.Label(right, text="File WAV").pack(anchor="w")
        ttk.Button(right, text="Pilih WAV", command=self.choose_wav_file).pack(anchor="w", pady=2)
        self.wav_status = tk.StringVar(value="File: -")
        ttk.Label(right, textvariable=self.wav_status, wraplength=360).pack(anchor="w", pady=2)
        ttk.Button(right, text="Transcribe WAV + WER", command=self.start_wav_eval).pack(anchor="w", pady=4)

        self.wer_result = ScrolledText(parent, height=9, wrap=tk.WORD, state="disabled")
        self.wer_result.pack(fill="both", expand=False, pady=(8, 0))

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Key>", self._on_key)

    def _on_key(self, event: tk.Event) -> str | None:
        widget_class = ""
        try:
            widget_class = str(event.widget.winfo_class())
        except Exception:
            pass
        if widget_class in {"Text", "Entry", "TEntry", "TCombobox"} and event.keysym != "Escape":
            return None
        action = shortcut_action(event.keysym, getattr(event, "char", ""))
        if action is None:
            return None
        if action == "p":
            self.bisindo.pause_resume()
        elif action == "v":
            self.bisindo.toggle_preview()
        elif action == "space":
            self.bisindo.append_space()
        elif action == "quit":
            self.on_close()
        return "break"

    def apply_profile(self) -> None:
        profile = build_tts_profile_name(self.gender_var.get(), self.age_var.get())
        self.profile_status.set(f"Aktif: {profile}")
        self.live_profile.set(f"TTS: {profile}")
        self.bisindo.set_tts_profile(profile)

    def apply_specialist(self) -> None:
        self.bisindo.set_specialist_enabled(bool(self.specialist_enabled_var.get()))

    def apply_camera_tolerance(self) -> None:
        try:
            tol_x = float(self.tol_x_var.get())
            tol_y = float(self.tol_y_var.get())
        except ValueError:
            messagebox.showerror("Toleransi", "Toleransi harus angka (mis. 0.10).")
            return
        self.bisindo.set_camera_tolerance(tol_x, tol_y)
        self.pos_status.set(f"Toleransi disimpan: X={tol_x:.2f}, Y={tol_y:.2f}.")

    def apply_camera_flip(self) -> None:
        self.bisindo.set_camera_flip(bool(self.flip_var.get()))
        self.pos_status.set(f"Preview mirror: {'on' if self.flip_var.get() else 'off'}.")

    def calibrate_camera(self) -> None:
        ok, message = self.bisindo.calibrate_camera()
        self.pos_status.set(message)
        if not ok:
            messagebox.showwarning("Kalibrasi posisi", message)

    def toggle_udp_broadcast(self) -> None:
        if self.broadcast.running:
            self.broadcast.stop()
            self.broadcast_status.set("Broadcast: mati")
            self.udp_button.configure(text="Konek UDP")
            return
        try:
            target = self.broadcast.start()
            self.broadcast_status.set(self.broadcast.status_text())
            self.udp_button.configure(text="Putus UDP")
            if target.source == "auto":
                messagebox.showinfo(
                    "UDP Broadcast",
                    "IP hotspot 10.42.0.1 belum aktif, jadi broadcast pakai IP auto-detect.",
                )
        except BroadcastError as exc:
            self.broadcast_status.set(f"Broadcast error: {exc}")
            messagebox.showerror("Broadcast UDP", str(exc))

    def toggle_stt(self) -> None:
        if self.sherpa_worker is not None and self.sherpa_worker.is_alive():
            self.sherpa_worker.stop()
            self.sherpa_worker.join(timeout=1.0)
            self.stt_button.configure(text="Start STT")
            self.udp_status.set("UDP: dihentikan")
            return
        self.sherpa_segments = []
        _set_text(self.sherpa_text, "")
        self.sherpa_worker = UdpSherpaWorker(
            recognizer=self.sherpa_recognizer,
            udp_config=UdpConfig(),
            event_queue=self.sherpa_queue,
        )
        self.sherpa_worker.start()
        self.stt_button.configure(text="Stop STT")
        self.udp_status.set("UDP: mulai listen...")

    def copy_sherpa_to_hypothesis(self) -> None:
        if not ENABLE_SHERPA_TEST_TAB:
            return
        text = self.sherpa_text.get("1.0", tk.END).strip()
        self.wer_hypothesis.delete("1.0", tk.END)
        self.wer_hypothesis.insert(tk.END, text)

    def calculate_current_wer(self) -> None:
        ref = self.wer_reference.get("1.0", tk.END).strip()
        hyp = self.wer_hypothesis.get("1.0", tk.END).strip()
        try:
            result = calculate_wer(ref, hyp)
            text = (
                f"WER: {float(result['wer']) * 100:.2f}%\n"
                f"MER: {float(result['mer']) * 100:.2f}%\n"
                f"WIL: {float(result['wil']) * 100:.2f}%\n\n"
                f"Ref: {result['reference']}\n"
                f"Hyp: {result['hypothesis']}\n"
            )
            _set_text(self.wer_result, text)
        except SherpaBackendError as exc:
            messagebox.showerror("WER", str(exc))

    def choose_wav_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("WAV audio", "*.wav")])
        if not path:
            return
        self.wav_path = Path(path)
        self.wav_status.set(f"File: {self.wav_path.name}")

    def start_wav_eval(self) -> None:
        if self.wav_path is None:
            messagebox.showerror("WAV", "Pilih file WAV dulu.")
            return
        ref = self.wer_reference.get("1.0", tk.END).strip()
        _set_text(self.wer_result, "Memproses WAV dengan Sherpa...\n")
        threading.Thread(target=self._wav_eval_worker, args=(self.wav_path, ref), daemon=True).start()

    def _wav_eval_worker(self, wav_path: Path, reference: str) -> None:
        try:
            result = transcribe_wav(wav_path, recognizer=self.sherpa_recognizer)
            report = (
                f"Hasil: {result['text']}\n"
                f"RTF: {float(result['rtf']):.3f} "
                f"(proses {float(result['process_sec']):.2f}s, audio {float(result['duration_sec']):.2f}s)\n"
            )
            if reference:
                wer = calculate_wer(reference, str(result["text"]))
                report += (
                    f"\nWER: {float(wer['wer']) * 100:.2f}%\n"
                    f"MER: {float(wer['mer']) * 100:.2f}%\n"
                    f"WIL: {float(wer['wil']) * 100:.2f}%\n"
                )
            self.root.after(0, lambda: _set_text(self.wer_result, report))
        except Exception as exc:
            self.root.after(0, lambda: _set_text(self.wer_result, f"ERROR: {exc}\n"))

    def _poll(self) -> None:
        for event in self.bisindo.poll():
            self._handle_bisindo_event(event)
        self._update_bisindo_snapshot()
        self._update_position_snapshot()
        self._drain_sherpa_events()
        self.root.after(POLL_MS, self._poll)

    _CAMPOS_HINT = {
        "OK": "OK (posisi sudah pas)",
        "UP": "UP (badan/kamera perlu naik)",
        "DOWN": "DOWN (badan/kamera perlu turun)",
        "LEFT": "LEFT (geser kiri)",
        "RIGHT": "RIGHT (geser kanan)",
    }

    def _update_position_snapshot(self) -> None:
        snap = self.bisindo.camera_snapshot()
        if snap["has_signal"]:
            self.pos_live.set(
                f"Bahu sekarang: x={snap['mid_x']:.3f}, y={snap['mid_y']:.3f}, lebar={snap['width']:.3f}"
            )
            self.pos_campos.set(f"CAMPOS: {self._CAMPOS_HINT.get(snap['campos'], snap['campos'])}")
        else:
            self.pos_live.set("Bahu sekarang: - (bahu tak terdeteksi / live belum jalan)")
            self.pos_campos.set("CAMPOS: -")
        if snap["calibrated"]:
            self.pos_ref.set(
                f"Acuan tersimpan: x={snap['ref_mid_x']:.3f}, y={snap['ref_mid_y']:.3f} "
                f"| toleransi X={snap['tol_x']:.2f}, Y={snap['tol_y']:.2f}"
            )
        else:
            self.pos_ref.set("Acuan tersimpan: belum dikalibrasi (pakai default)")

    def _handle_bisindo_event(self, event: dict[str, Any]) -> None:
        message = str(event.get("message", ""))
        if event.get("type") == "bisindo_output":
            self.bisindo_output.set(message)
        if message:
            _append_text(self.bisindo_log, message + "\n")

    def _update_bisindo_snapshot(self) -> None:
        snap = self.bisindo.snapshot()
        if snap["starting"]:
            state = "loading runtime"
        elif not snap["prepared"]:
            state = "belum start"
        elif snap["paused"]:
            state = "pause"
        else:
            state = "live"
        preview = "on" if snap["preview"] else "off"
        self.live_status.set(f"Status: {state} | preview {preview}")
        self.live_buffer.set(f"Buffer: {snap['buffer']}")
        self.live_profile.set(f"TTS: {snap['profile']}")

    def _drain_sherpa_events(self) -> None:
        while True:
            try:
                event = self.sherpa_queue.get_nowait()
            except queue.Empty:
                return
            event_type = event.get("type")
            if event_type == "recognizer":
                self.recognizer_status.set(str(event.get("message", "Sherpa siap")))
            elif event_type == "listening":
                self.udp_status.set(str(event.get("message", "UDP listen")))
            elif event_type == "packet":
                self.packet_status.set(f"Packet: {event.get('count')} | sender: {event.get('last_sender')}")
            elif event_type == "latency":
                self.latency_status.set(f"Delay decode: {float(event.get('value_ms', 0.0)):.1f} ms")
            elif event_type == "partial":
                text = str(event.get("text", "") or "-")
                self.partial_status.set(f"Mengetik: {text}")
            elif event_type == "word":
                _append_text(self.sherpa_text, str(event.get("text", "")) + " ")
            elif event_type == "segment_end":
                segment = str(event.get("text", "")).strip()
                if segment:
                    self.sherpa_segments.append(segment)
                self.partial_status.set("Mengetik: -")
            elif event_type == "error":
                message = str(event.get("message", "Sherpa error"))
                self.udp_status.set(f"Error: {message}")
                self.stt_button.configure(text="Start STT")
                messagebox.showerror("Sherpa", message)
            elif event_type == "stopped":
                self.udp_status.set(str(event.get("message", "Sherpa UDP stopped")))
                self.stt_button.configure(text="Start STT")

    def on_close(self) -> None:
        try:
            if self.sherpa_worker is not None:
                self.sherpa_worker.stop()
                self.sherpa_worker.join(timeout=1.0)
            self.broadcast.stop()
            self.bisindo.shutdown()
        finally:
            self.root.destroy()


def main() -> int:
    root = tk.Tk()
    IntegrationApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
