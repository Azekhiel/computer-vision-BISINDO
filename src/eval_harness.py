"""Evaluation harness: train/evaluate/benchmark a schema x variant matrix.

Runs every (schema, variant) combination, reusing the existing
``gru_manager`` primitives:
- ``train_variant`` (skip-if-exists, so re-runs resume for free),
- ``evaluate_variant`` (test-split macro-F1 + per-sample predictions),
- ``benchmark_variant`` (on-device latency = real Jetson latency).

Outputs land in ``reports/eval_<timestamp>/``:
- ``summary.csv`` / ``summary.md``  — one row per combo, ranked by macro-F1,
- ``confusion_<schema>__<variant>.csv`` — confusion matrix per combo,
- ``raw_results.json`` — full machine-readable dump.

The matrix is resumable: trained checkpoints are skipped, so a long run can be
split into stages (start small, widen later).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import feature_schemas as fs
import gru_manager as gm


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_ROOT = ROOT_DIR / "reports"


def expand_schema_request(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        for name in fs.expand_schema_names(token):
            if name not in out:
                out.append(name)
    return out


def expand_variant_request(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        value = str(token or "").strip().lower()
        if value in {"all", "semua"}:
            names: Sequence[str] = gm.VARIANT_NAMES
        elif value in {"base", "original"}:
            names = gm.BASE_VARIANT_NAMES
        elif value in {"augmented", "aug"}:
            names = gm.AUGMENTED_VARIANT_NAMES
        else:
            names = [gm.normalize_variant_name(value)]
        for name in names:
            if name not in out:
                out.append(name)
    return out


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> tuple[list[str], np.ndarray]:
    labels = sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[index[true_label], index[pred_label]] += 1
    return labels, matrix


def _write_confusion_csv(path: Path, labels: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for i, label in enumerate(labels):
            writer.writerow([label, *matrix[i].tolist()])


def _combo_safe_name(schema: str, variant: str) -> str:
    return f"{schema}__{variant}".replace("/", "_")


def run_matrix(
    schemas: Sequence[str],
    variants: Sequence[str],
    dataset_dir: str | Path = gm.DATASET_DIR,
    model_dir: str | Path = gm.MODEL_DIR,
    out_root: str | Path = REPORTS_ROOT,
    train_missing: bool = False,
    epochs: int | None = None,
    limit_per_class: int | None = None,
    device: str = "auto",
    split: str = "test",
    benchmark: bool = True,
    benchmark_device: str = "auto",
    benchmark_runs: int = 30,
    overwrite_existing: bool = False,
) -> dict[str, object]:
    schema_names = expand_schema_request(schemas)
    variant_names = expand_variant_request(variants)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / f"eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for schema in schema_names:
        for variant in variant_names:
            row: dict[str, object] = {"schema": schema, "variant": variant, "status": "ok"}
            try:
                if train_missing:
                    trained, message = gm.train_variant(
                        variant,
                        dataset_dir=dataset_dir,
                        model_dir=model_dir,
                        schema=schema,
                        epochs=epochs,
                        limit_per_class=limit_per_class,
                        device=device,
                        overwrite_existing=overwrite_existing,
                    )
                    row["train_message"] = message
                    if not trained:
                        row["status"] = "train_failed"
                        results.append(row)
                        print(f"[EVAL][train_failed] {schema}/{variant}: {message}", flush=True)
                        continue

                evaluation = gm.evaluate_variant(
                    variant,
                    dataset_dir=dataset_dir,
                    model_dir=model_dir,
                    schema=schema,
                    split=split,
                    device=device,
                )
                metrics = evaluation["metrics"]
                row.update(
                    {
                        "samples": int(evaluation["samples"]),
                        "accuracy": float(metrics["accuracy"]),
                        "f1_macro": float(metrics["f1_macro"]),
                        "precision_macro": float(metrics["precision_macro"]),
                        "recall_macro": float(metrics["recall_macro"]),
                        "f1_micro": float(metrics["f1_micro"]),
                    }
                )
                labels, matrix = confusion_matrix(evaluation["y_true"], evaluation["y_pred"])
                _write_confusion_csv(out_dir / f"confusion_{_combo_safe_name(schema, variant)}.csv", labels, matrix)

                if benchmark:
                    try:
                        bench = gm.benchmark_variant(
                            variant,
                            model_dir=model_dir,
                            schema=schema,
                            device=benchmark_device,
                            runs=benchmark_runs,
                        )
                        row.update(
                            {
                                "bench_device": str(bench["device"]),
                                "mean_ms": float(bench["mean_ms"]),
                                "p50_ms": float(bench["p50_ms"]),
                                "p95_ms": float(bench["p95_ms"]),
                            }
                        )
                    except Exception as exc:  # benchmarking is best-effort
                        row["bench_error"] = str(exc)
                print(
                    f"[EVAL] {schema}/{variant}: f1_macro={row.get('f1_macro', 0.0):.4f} "
                    f"acc={row.get('accuracy', 0.0):.4f} n={row.get('samples', 0)}",
                    flush=True,
                )
            except FileNotFoundError as exc:
                row["status"] = "missing_checkpoint"
                row["error"] = str(exc)
                print(f"[EVAL][missing] {schema}/{variant}: {exc}", flush=True)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                print(f"[EVAL][ERR] {schema}/{variant}: {exc}", flush=True)
            results.append(row)

    ranked = sorted(
        results,
        key=lambda item: (item.get("status") == "ok", float(item.get("f1_macro", 0.0))),
        reverse=True,
    )
    _write_summary_csv(out_dir / "summary.csv", ranked)
    _write_summary_md(out_dir / "summary.md", ranked, timestamp, split)
    (out_dir / "raw_results.json").write_text(
        json.dumps({"timestamp": timestamp, "split": split, "results": ranked}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[EVAL] summary: {out_dir / 'summary.md'}", flush=True)
    return {"out_dir": str(out_dir), "results": ranked}


_SUMMARY_COLUMNS = (
    "schema",
    "variant",
    "status",
    "samples",
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "mean_ms",
    "p50_ms",
    "p95_ms",
)


def _write_summary_csv(path: Path, ranked: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)


def _fmt(value: object, spec: str) -> str:
    if value is None or value == "":
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _write_summary_md(path: Path, ranked: Sequence[dict[str, object]], timestamp: str, split: str) -> None:
    lines = [
        f"# Eval matrix — {timestamp} (split={split})",
        "",
        "Ranked by macro-F1 (descending). Latency = on-device inference (real Jetson latency).",
        "",
        "| # | schema | variant | status | n | acc | macro-F1 | prec | recall | mean ms | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| {rank} | {schema} | {variant} | {status} | {n} | {acc} | {f1} | {prec} | {rec} | {mean} | {p50} | {p95} |".format(
                rank=rank,
                schema=row.get("schema", "-"),
                variant=row.get("variant", "-"),
                status=row.get("status", "-"),
                n=row.get("samples", "-"),
                acc=_fmt(row.get("accuracy"), ".4f"),
                f1=_fmt(row.get("f1_macro"), ".4f"),
                prec=_fmt(row.get("precision_macro"), ".4f"),
                rec=_fmt(row.get("recall_macro"), ".4f"),
                mean=_fmt(row.get("mean_ms"), ".2f"),
                p50=_fmt(row.get("p50_ms"), ".2f"),
                p95=_fmt(row.get("p95_ms"), ".2f"),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate/benchmark a schema x variant matrix.")
    parser.add_argument("--schema", default="smart180", help="Comma list of schemas or groups (smart180,faceref,all,...).")
    parser.add_argument("--variant", default="adi", help="Comma list of variants or groups (adi,biattn,tcn,all,base,...).")
    parser.add_argument("--dataset-dir", default=str(gm.DATASET_DIR))
    parser.add_argument("--model-dir", default=str(gm.MODEL_DIR))
    parser.add_argument("--out-root", default=str(REPORTS_ROOT))
    parser.add_argument("--train-missing", action="store_true", help="Train combos that have no checkpoint yet (skip-if-exists).")
    parser.add_argument("--overwrite-existing", action="store_true", help="Retrain even if a checkpoint already exists.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--benchmark-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--benchmark-runs", type=int, default=30)
    args = parser.parse_args(argv)

    run_matrix(
        schemas=[token for token in args.schema.split(",") if token.strip()],
        variants=[token for token in args.variant.split(",") if token.strip()],
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        out_root=args.out_root,
        train_missing=args.train_missing,
        epochs=args.epochs,
        limit_per_class=args.limit_per_class,
        device=args.device,
        split=args.split,
        benchmark=not args.no_benchmark,
        benchmark_device=args.benchmark_device,
        benchmark_runs=args.benchmark_runs,
        overwrite_existing=args.overwrite_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
