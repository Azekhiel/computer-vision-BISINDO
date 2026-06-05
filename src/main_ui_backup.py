"""Minimal GRU-focused BISINDO dashboard."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import feature_schemas as fs
import gru_manager as gm
import live_gru_fast


class AppUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("BISINDO GRU Live Test")
        self.root.geometry("760x520")
        self.root.minsize(700, 460)

        self.live_worker = None
        self.live_queue: queue.Queue = queue.Queue()
        self.live_poll_job = None

        self.variant_display_to_value: dict[str, str] = {}
        self.variant_options = ["auto"]
        self.schema_var = tk.StringVar(value=fs.DEFAULT_SCHEMA)
        self.variant_var = tk.StringVar(value="auto")
        self.mode_var = tk.StringVar(value="accurate10")
        self.device_var = tk.StringVar(value="auto")
        self.live_device_var = tk.StringVar(value="auto")
        self.epochs_var = tk.StringVar(value="100")
        self.batch_var = tk.StringVar(value="64")
        self.status_var = tk.StringVar(value="Siap.")
        self.live_status_var = tk.StringVar(value="Live: idle")

        self._build_ui()
        self.refresh_status()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        title = ttk.Label(outer, text="BISINDO GRU Dashboard", font=("Arial", 18, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        controls = ttk.LabelFrame(outer, text="Model")
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for idx in range(6):
            controls.columnconfigure(idx, weight=1)

        ttk.Label(controls, text="Varian").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.variant_combo = ttk.Combobox(
            controls,
            textvariable=self.variant_var,
            values=self.variant_options,
            state="readonly",
            width=12,
        )
        self.variant_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=8)

        ttk.Label(controls, text="Live mode").grid(row=0, column=2, sticky="w", padx=8, pady=8)
        ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=sorted(live_gru_fast.LIVE_PROFILES),
            state="readonly",
            width=10,
        ).grid(row=0, column=3, sticky="ew", padx=8, pady=8)

        ttk.Label(controls, text="Live device").grid(row=0, column=4, sticky="w", padx=8, pady=8)
        ttk.Combobox(
            controls,
            textvariable=self.live_device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=8,
        ).grid(row=0, column=5, sticky="ew", padx=8, pady=8)

        ttk.Label(controls, text="Train device").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        ttk.Combobox(
            controls,
            textvariable=self.device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=8,
        ).grid(row=1, column=1, sticky="ew", padx=8, pady=8)

        ttk.Label(controls, text="Epochs").grid(row=1, column=2, sticky="w", padx=8, pady=8)
        ttk.Entry(controls, textvariable=self.epochs_var, width=10).grid(row=1, column=3, sticky="ew", padx=8, pady=8)
        ttk.Label(controls, text="Batch").grid(row=1, column=4, sticky="w", padx=8, pady=8)
        ttk.Entry(controls, textvariable=self.batch_var, width=10).grid(row=1, column=5, sticky="ew", padx=8, pady=8)

        ttk.Label(controls, text="Schema").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        ttk.Combobox(
            controls,
            textvariable=self.schema_var,
            values=list(fs.SCHEMA_NAMES),
            state="readonly",
            width=12,
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=8)

        buttons = ttk.Frame(outer)
        buttons.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        buttons.columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.btn_train_one = ttk.Button(buttons, text="Train Varian", command=self.train_selected)
        self.btn_train_one.grid(row=0, column=0, sticky="ew", padx=4)
        self.btn_train_all = ttk.Button(buttons, text="Train Semua", command=self.train_all)
        self.btn_train_all.grid(row=0, column=1, sticky="ew", padx=4)
        self.btn_live = ttk.Button(buttons, text="Start Live Test", command=self.toggle_live)
        self.btn_live.grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(buttons, text="Evaluator", command=self.open_evaluator).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(buttons, text="Refresh", command=self.refresh_status).grid(row=0, column=4, sticky="ew", padx=4)

        body = ttk.Frame(outer)
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, textvariable=self.status_var, foreground="#333333").grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.status_text = tk.Text(body, height=15, wrap="word")
        self.status_text.grid(row=1, column=0, sticky="nsew")
        self.status_text.configure(state="disabled")

        ttk.Label(outer, textvariable=self.live_status_var, foreground="#0d6efd").grid(row=4, column=0, sticky="ew", pady=(10, 0))

    def _set_text(self, content: str) -> None:
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", content)
        self.status_text.configure(state="disabled")

    def _parse_training_args(self) -> tuple[int, int]:
        try:
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
        except ValueError as exc:
            raise ValueError("Epochs dan batch harus angka.") from exc
        if epochs <= 0 or batch <= 0:
            raise ValueError("Epochs dan batch harus > 0.")
        return epochs, batch

    def _build_variant_options(self) -> None:
        previous_value = self.selected_variant_value()
        self.variant_display_to_value = {}
        options: list[str] = []
        try:
            best = gm.select_best_available_variant(schema=self.schema_var.get())
            best_label = f"auto (best: gru_{best})"
        except Exception:
            best_label = "auto (no checkpoint)"
        self.variant_display_to_value[best_label] = "auto"
        options.append(best_label)

        for variant in gm.VARIANT_NAMES:
            display = f"gru_{variant}"
            if not gm.checkpoint_exists(variant, schema=self.schema_var.get()):
                display += " (missing)"
            else:
                try:
                    metadata = gm.load_metadata(variant, schema=self.schema_var.get())
                    val_acc = metadata.get("best_val_acc")
                    if isinstance(val_acc, (float, int)):
                        display += f" (val {float(val_acc):.3f})"
                        if float(val_acc) < 0.70:
                            display += " LOW"
                except Exception:
                    display += " (metadata?)"
            self.variant_display_to_value[display] = variant
            options.append(display)

        self.variant_options = options
        if hasattr(self, "variant_combo"):
            self.variant_combo.configure(values=options)
        selected_display = next((label for label, value in self.variant_display_to_value.items() if value == previous_value), options[0])
        self.variant_var.set(selected_display)

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
            schema = self.schema_var.get()
            summary = gm.dataset_summary(schema=schema)
            model_status = gm.list_model_status(schema=schema)
            lines = [
                f"Dataset {schema}: {summary['total_samples']} sampel",
                f"Kelas classifier non-idle: {summary['num_classifier_classes']}",
                "",
                "Checkpoint:",
            ]
            for variant in gm.VARIANT_NAMES:
                lines.append(f"- gru_{variant}: {model_status.get(variant, '-')}")
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

        self.btn_train_one.configure(state="disabled")
        self.btn_train_all.configure(state="disabled")
        self.status_var.set("Training berjalan...")

        def task() -> None:
            lines = []
            for variant in variants:
                ok, msg = gm.train_variant(
                    variant,
                    epochs=epochs,
                    batch_size=batch,
                    device=self.device_var.get(),
                    schema=self.schema_var.get(),
                )
                lines.append(("OK " if ok else "ERR ") + msg)
            self.root.after(0, lambda: self._training_done("\n".join(lines)))

        threading.Thread(target=task, daemon=True).start()

    def _training_done(self, message: str) -> None:
        self.btn_train_one.configure(state="normal")
        self.btn_train_all.configure(state="normal")
        self.status_var.set("Training selesai.")
        self.refresh_status()
        messagebox.showinfo("Training GRU", message)

    def train_selected(self) -> None:
        variant = self.selected_variant_value()
        if variant == "auto":
            messagebox.showwarning("Training GRU", "Pilih gru_khukuh, gru_adi, atau gru_hybrid untuk training satu varian.")
            return
        self._run_training((variant,))

    def train_all(self) -> None:
        self._run_training(gm.VARIANT_NAMES)

    def toggle_live(self) -> None:
        if self.live_worker is not None and self.live_worker.is_alive():
            self.live_worker.stop()
            self.btn_live.configure(text="Stopping...", state="disabled")
            self.live_status_var.set("Live: stopping")
            return

        self.live_queue = queue.Queue()
        try:
            self.live_worker = live_gru_fast.start_live_inference(
                self.selected_variant_value(),
                status_queue=self.live_queue,
                profile=self.mode_var.get(),
                device=self.live_device_var.get(),
                schema=self.schema_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Live Test", str(exc))
            return
        self.btn_live.configure(text="Stop Live Test")
        self.live_status_var.set("Live: starting")
        self._poll_live()

    def _poll_live(self) -> None:
        self.live_poll_job = None
        while not self.live_queue.empty():
            item = self.live_queue.get()
            event = item.get("event")
            if event == "error":
                self.live_status_var.set(f"Live error: {item.get('message')}")
                messagebox.showwarning("Live Test", str(item.get("message")))
            elif event == "stopped":
                self.live_status_var.set(f"Live: {item.get('message')}")
            elif event == "started":
                warning = item.get("warning") or ""
                suffix = f" | {warning}" if warning else ""
                reason = item.get("device_reason") or "-"
                self.live_status_var.set(
                    f"Live: {item.get('variant')} | profile {item.get('profile')} | device {item.get('device')} | "
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
                self.live_status_var.set(
                    f"Live: {label} ({conf:.2f}) raw {raw} | cam {fps_camera:.1f} fps | pred {fps_predict:.1f} fps | "
                    f"extract {extract_ms:.1f} ms | model {model_ms:.1f} ms | buffer {buf}/{target}"
                )

        if self.live_worker is not None and self.live_worker.is_alive():
            self.live_poll_job = self.root.after(250, self._poll_live)
        else:
            self.btn_live.configure(text="Start Live Test", state="normal")
            self.live_worker = None

    def open_evaluator(self) -> None:
        subprocess.Popen([sys.executable, str(gm.ROOT_DIR / "src" / "eva_dashboard.py")])


if __name__ == "__main__":
    root = tk.Tk()
    AppUI(root)
    root.mainloop()
