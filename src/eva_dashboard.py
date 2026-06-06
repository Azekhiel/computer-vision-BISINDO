"""Evaluation dashboard for the three GRU BISINDO variants."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import feature_schemas as fs
import gru_manager as gm


class EvaluatorBackend:
    def __init__(self) -> None:
        self.test_data: list[gm.SequenceSample] = []
        self.results: dict[str, dict[str, list[str]]] = {}
        self.schema = fs.DEFAULT_SCHEMA

    def load_test_data(self, schema: str = fs.DEFAULT_SCHEMA) -> tuple[bool, str]:
        self.schema = fs.normalize_schema_name(schema)
        try:
            self.test_data = gm.load_sequences(split="test", include_idle=False, schema=self.schema)
        except Exception as exc:
            return False, str(exc)
        if not self.test_data:
            return False, f"Tidak ada data split test untuk schema {self.schema}."
        classes = sorted({sample.label for sample in self.test_data})
        return True, f"Memuat {len(self.test_data)} sampel test {self.schema} dari {len(classes)} kelas."

    def run_evaluations(self, selected_models: list[str]) -> None:
        self.results = {}
        for variant in selected_models:
            try:
                result = gm.evaluate_variant(variant, split="test", schema=self.schema)
                self.results[variant] = {
                    "y_true": list(result["y_true"]),
                    "y_pred": list(result["y_pred"]),
                }
            except Exception as exc:
                print(f"Eval {variant} gagal: {exc}")
                self.results[variant] = {"y_true": [], "y_pred": []}


class EvalUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Evaluasi GRU BISINDO")
        self.root.geometry("1000x740")
        self.backend = EvaluatorBackend()
        self.schema_var = tk.StringVar(value=fs.DEFAULT_SCHEMA)
        self.model_vars = {variant: tk.BooleanVar(value=True) for variant in gm.VARIANT_NAMES}
        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#f8f9fa", pady=10)
        top.pack(fill="x")
        tk.Label(top, text="Evaluasi Model GRU", font=("Arial", 16, "bold"), bg="#f8f9fa").pack()

        check_frame = tk.Frame(top, bg="#f8f9fa")
        check_frame.pack(pady=6)
        ttk.Combobox(
            check_frame,
            textvariable=self.schema_var,
            values=list(fs.SCHEMA_NAMES),
            state="readonly",
            width=14,
        ).pack(side="left", padx=10)
        for variant in gm.VARIANT_NAMES:
            tk.Checkbutton(
                check_frame,
                text=f"gru_{variant}",
                variable=self.model_vars[variant],
                bg="#f8f9fa",
                font=("Arial", 10),
            ).pack(side="left", padx=10)

        self.btn_run = tk.Button(
            top,
            text="Mulai Evaluasi GRU",
            bg="#0d6efd",
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.run_evaluation,
        )
        self.btn_run.pack(pady=8)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.tab_eda = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_eda, text="1. Data Test")
        self.tab_compare = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_compare, text="2. Ringkasan")
        self.tab_detail = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_detail, text="3. Detail")

        self._setup_detail_tab()

    def run_evaluation(self) -> None:
        selected = [variant for variant, var in self.model_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("Evaluasi", "Pilih minimal satu model.")
            return

        self.btn_run.config(state="disabled", text="Evaluasi berjalan...")

        def task() -> None:
            ok, msg = self.backend.load_test_data(self.schema_var.get())
            if not ok:
                self.root.after(0, lambda: messagebox.showerror("Evaluasi", msg))
                self.root.after(0, lambda: self.btn_run.config(state="normal", text="Mulai Evaluasi GRU"))
                return
            self.backend.run_evaluations(selected)
            self.root.after(0, self.render_eda)
            self.root.after(0, self.render_comparison)
            self.root.after(0, lambda: self.combo_model.config(values=selected))
            self.root.after(0, lambda: self.combo_model.set(selected[0]))
            self.root.after(0, self.update_detail_view)
            self.root.after(0, lambda: self.btn_run.config(state="normal", text="Evaluasi Selesai"))

        threading.Thread(target=task, daemon=True).start()

    @staticmethod
    def clear_frame(frame: tk.Frame) -> None:
        for widget in frame.winfo_children():
            widget.destroy()

    def render_eda(self) -> None:
        self.clear_frame(self.tab_eda)
        labels = [sample.label for sample in self.backend.test_data]
        df_counts = pd.Series(labels).value_counts().reset_index()
        df_counts.columns = ["Vocab", "Jumlah Sampel Test"]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_counts, x="Vocab", y="Jumlah Sampel Test", ax=ax, palette="viridis")
        ax.set_title("Distribusi Kelas Data Test", fontweight="bold")
        ax.tick_params(axis="x", labelrotation=45)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.tab_eda)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def render_comparison(self) -> None:
        self.clear_frame(self.tab_compare)
        metrics = {"Model": [], "Macro Precision": [], "Macro Recall": [], "Macro F1": []}
        for model_name, data in self.backend.results.items():
            if not data["y_true"]:
                continue
            p, r, f1, _ = precision_recall_fscore_support(
                data["y_true"],
                data["y_pred"],
                average="macro",
                zero_division=0,
            )
            metrics["Model"].append(f"GRU {model_name.upper()}")
            metrics["Macro Precision"].append(p)
            metrics["Macro Recall"].append(r)
            metrics["Macro F1"].append(f1)

        if not metrics["Model"]:
            tk.Label(self.tab_compare, text="Belum ada hasil evaluasi valid.", bg="white").pack(pady=24)
            return

        df_metrics = pd.DataFrame(metrics).melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_metrics, x="Metric", y="Score", hue="Model", ax=ax, palette="Set2")
        ax.set_title("Perbandingan Macro Metrics", fontweight="bold")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.tab_compare)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _setup_detail_tab(self) -> None:
        control = tk.Frame(self.tab_detail, bg="white")
        control.pack(fill="x", pady=5)
        tk.Label(control, text="Pilih Model:", bg="white", font=("Arial", 11, "bold")).pack(side="left", padx=10)
        self.combo_model = ttk.Combobox(control, values=list(gm.VARIANT_NAMES), state="readonly")
        self.combo_model.set("khukuh")
        self.combo_model.pack(side="left")
        self.combo_model.bind("<<ComboboxSelected>>", lambda _event: self.update_detail_view())

        self.vis_frame = tk.Frame(self.tab_detail, bg="white")
        self.vis_frame.pack(fill="both", expand=True)
        self.cm_frame = tk.Frame(self.vis_frame, bg="white")
        self.cm_frame.pack(side="left", fill="both", expand=True)
        self.table_frame = tk.Frame(self.vis_frame, bg="white", width=300)
        self.table_frame.pack(side="right", fill="y", padx=10)

    def update_detail_view(self) -> None:
        selected = self.combo_model.get()
        res = self.backend.results.get(selected, {})
        self.clear_frame(self.cm_frame)
        self.clear_frame(self.table_frame)
        if not res.get("y_true"):
            tk.Label(self.cm_frame, text="Tidak ada hasil untuk model ini.", bg="white").pack(pady=24)
            return

        y_true = res["y_true"]
        y_pred = res["y_pred"]
        classes = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"Confusion Matrix - GRU {selected.upper()}", fontweight="bold")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
        tk.Label(self.table_frame, text="Skor Per Kelas", font=("Arial", 11, "bold"), bg="white").pack(pady=5)
        columns = ("Class", "Precision", "Recall", "F1")
        tree = ttk.Treeview(self.table_frame, columns=columns, show="headings", height=16)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=76, anchor="center")
        tree.column("Class", width=115, anchor="w")
        for cls in classes:
            if cls in report:
                row = (
                    cls,
                    f"{report[cls]['precision']:.2f}",
                    f"{report[cls]['recall']:.2f}",
                    f"{report[cls]['f1-score']:.2f}",
                )
                tree.insert("", tk.END, values=row)
        tree.pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    EvalUI(root)
    root.mainloop()
