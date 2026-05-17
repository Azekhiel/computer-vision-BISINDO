"""
geometry_benchmark.py - V4.2 extraction/audit harness.

Run from repo root, for example:
  ./env_bisindo/bin/python src/geometry_benchmark.py --reextract

The harness intentionally targets a small stress set before a full rebuild.
It writes body-frame GIFs, pixel-overlay GIFs, contact sheets, and JSON audit
reports under assets/gifs/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import data_ingestion as di
import database_manager as dbm
import feature_engine as fe
import visualization_utils as vu

ROOT_DIR = Path(__file__).resolve().parents[1]
DATABASE_DIR = ROOT_DIR / "dataset_parquets"
VIDEO_ROOT = ROOT_DIR / "record" / "video" / "train"
GIF_DIR = ROOT_DIR / "assets" / "gifs"
REPORT_DIR = GIF_DIR / "audit_reports"
CONTACT_DIR = GIF_DIR / "contact_sheets"

BENCHMARK_VOCABS = ["di_jalan", "bertemu", "perkenalkan", "sehat", "hati_hati", "salam_kenal"]
MEDIA_EXTS = (".mkv", ".mp4", ".avi", ".mov", ".webm", ".jpg", ".jpeg", ".png")


def parse_features(value) -> np.ndarray:
    return np.fromstring(str(value), sep=",", dtype=np.float32)


def _intervals(mask: np.ndarray) -> list[list[int]]:
    out = []
    start = None
    for idx, active in enumerate(list(mask.astype(bool)) + [False]):
        if active and start is None:
            start = idx
        elif start is not None and not active:
            out.append([int(start), int(idx - 1)])
            start = None
    return out


def _hand_total_lengths(hand: np.ndarray) -> np.ndarray:
    pts = np.asarray(hand, dtype=np.float32).reshape(-1, 21, 3)
    totals = []
    for frame in pts:
        total = 0.0
        for parent, child in fe.HAND_TREE_EDGES:
            total += float(np.linalg.norm(frame[child, :2] - frame[parent, :2]))
        totals.append(total)
    return np.asarray(totals, dtype=np.float32)


def _max_anchor_jump(hand: np.ndarray, mask: np.ndarray) -> float:
    pts = np.asarray(hand, dtype=np.float32).reshape(-1, 21, 3)
    mask = np.asarray(mask, dtype=bool)
    max_jump = 0.0
    for idx in range(1, len(mask)):
        if mask[idx] and mask[idx - 1]:
            max_jump = max(max_jump, float(np.linalg.norm(pts[idx, 0, :2] - pts[idx - 1, 0, :2])))
    return float(max_jump)


def _short_segments(mask: np.ndarray, min_len: int = 3) -> list[list[int]]:
    return [span for span in _intervals(mask) if span[1] - span[0] + 1 < min_len]


def _short_segments_without_original(rows: pd.DataFrame, side: str, mask: np.ndarray, min_len: int = 3) -> list[list[int]]:
    out = []
    for start, end in _short_segments(mask, min_len):
        has_original = False
        for _, row in rows.iloc[start : end + 1].iterrows():
            try:
                metadata = json.loads(row.get("tracking_metadata", "{}"))
            except Exception:
                metadata = {}
            hand = metadata.get(side, {}) if isinstance(metadata, dict) else {}
            if bool(hand.get("original_detected", False)):
                has_original = True
                break
        if not has_original:
            out.append([start, end])
    return out


def audit_vocab(vocab: str) -> dict:
    path = DATABASE_DIR / f"{vocab}.parquet"
    report = {"vocab": vocab, "schema": fe.FEATURE_SCHEMA, "exists": path.exists(), "sequences": [], "failures": []}
    if not path.exists():
        report["failures"].append("missing_parquet")
        return report

    df = pd.read_parquet(path)
    df = fe.filter_current_feature_rows(df)
    if df.empty:
        report["failures"].append("no_current_schema_rows")
        return report

    for video_id, group in df.groupby("video_id"):
        group = group.sort_values("frame_num")
        seq = np.vstack([parse_features(value) for value in group["features"]]).astype(np.float32)
        flags = seq[:, fe.SLICE_FLAGS] >= 0.5
        pose = seq[:, fe.SLICE_POSE].reshape(len(seq), 6, 3)
        left = seq[:, fe.SLICE_LH].reshape(len(seq), 21, 3)
        right = seq[:, fe.SLICE_RH].reshape(len(seq), 21, 3)
        left_nonzero = np.max(np.linalg.norm(left[:, :, :2], axis=2), axis=1) > 1e-6
        right_nonzero = np.max(np.linalg.norm(right[:, :, :2], axis=2), axis=1) > 1e-6

        failures = []
        if seq.shape[1] != fe.N_TOTAL_WITH_FLAGS or not np.isfinite(seq).all():
            failures.append("shape_or_finite")
        if np.any((~flags[:, fe.IDX_LH]) & left_nonzero):
            failures.append("left_flag_zero_nonzero")
        if np.any((~flags[:, fe.IDX_RH]) & right_nonzero):
            failures.append("right_flag_zero_nonzero")
        if np.any(flags[:, fe.IDX_LH] & ~left_nonzero):
            failures.append("left_flag_one_zero")
        if np.any(flags[:, fe.IDX_RH] & ~right_nonzero):
            failures.append("right_flag_one_zero")

        metadata_missing_accepted = 0
        for _, row in group.iterrows():
            try:
                metadata = json.loads(row.get("tracking_metadata", "{}"))
            except Exception:
                metadata = {}
            for side in ("left", "right"):
                hand = metadata.get(side, {}) if isinstance(metadata, dict) else {}
                if hand.get("accepted") and hand.get("source") == "missing":
                    metadata_missing_accepted += 1
        if metadata_missing_accepted:
            failures.append("accepted_source_missing")

        for side, hand, idx in (("left", left, fe.IDX_LH), ("right", right, fe.IDX_RH)):
            totals = _hand_total_lengths(hand)
            valid = flags[:, idx] & (totals > 1e-6)
            if valid.sum() > 1:
                cv = float(totals[valid].std() / max(float(totals[valid].mean()), 1e-8))
                if cv >= 1e-4:
                    failures.append(f"{side}_bone_cv")
            jump = _max_anchor_jump(hand, flags[:, idx])
            if jump >= 0.35:
                failures.append(f"{side}_anchor_jump")
            if _short_segments_without_original(group, side, flags[:, idx]):
                failures.append(f"{side}_short_segment")

        for side, idxs in (("left", (2, 4, fe.IDX_LH)), ("right", (3, 5, fe.IDX_RH))):
            elbow_idx, wrist_idx, flag_idx = idxs
            inactive = ~flags[:, flag_idx]
            arm = pose[:, [elbow_idx, wrist_idx], :]
            if np.any(inactive & (np.max(np.linalg.norm(arm[:, :, :2], axis=2), axis=1) > 1e-6)):
                failures.append(f"{side}_pose_arm_not_gated")

        item = {
            "video_id": str(video_id),
            "frames": int(len(seq)),
            "left_intervals": _intervals(flags[:, fe.IDX_LH]),
            "right_intervals": _intervals(flags[:, fe.IDX_RH]),
            "left_anchor_jump": _max_anchor_jump(left, flags[:, fe.IDX_LH]),
            "right_anchor_jump": _max_anchor_jump(right, flags[:, fe.IDX_RH]),
            "failures": sorted(set(failures)),
        }
        report["sequences"].append(item)
        report["failures"].extend([f"{video_id}:{failure}" for failure in sorted(set(failures))])
    return report


def write_audit_report(vocab: str) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = audit_vocab(vocab)
    with open(REPORT_DIR / f"{vocab}_geometry_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)
    return report


def make_contact_sheet(vocab: str, max_frames: int = 12) -> str | None:
    gif_path = GIF_DIR / f"{vocab}.gif"
    overlay_path = GIF_DIR / "pixel_overlay" / f"{vocab}.gif"
    parquet_path = DATABASE_DIR / f"{vocab}.parquet"
    if not gif_path.exists() or not parquet_path.exists():
        return None
    df = fe.filter_current_feature_rows(pd.read_parquet(parquet_path))
    if df.empty:
        return None
    sample = df[~df["video_id"].astype(str).str.contains("_aug_")]
    if sample.empty:
        sample = df
    video_id = sample["video_id"].iloc[0]
    rows = sample[sample["video_id"] == video_id].sort_values("frame_num")
    source_indices = rows["source_frame_num"].astype(int).tolist() if "source_frame_num" in rows.columns else rows["frame_num"].astype(int).tolist()
    raw_path = vu._resolve_raw_video_path(str(video_id), vocab, str(rows["split"].iloc[0]) if "split" in rows.columns else "train")
    if not raw_path or not os.path.exists(raw_path):
        return None

    n = min(len(rows), max_frames)
    frame_ids = np.linspace(0, len(rows) - 1, n, dtype=int).tolist()
    thumb_w, thumb_h = 240, 180
    gif = Image.open(gif_path)
    overlay = Image.open(overlay_path) if overlay_path.exists() else None
    cap = cv2.VideoCapture(raw_path)
    strips = []
    for idx in frame_ids:
        raw_img = Image.new("RGB", (thumb_w, thumb_h), "#111")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_indices[idx]))
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((thumb_w, thumb_h))
            raw_img.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        gif.seek(min(idx, getattr(gif, "n_frames", 1) - 1))
        gif_img = gif.convert("RGB")
        gif_img.thumbnail((thumb_w, thumb_h))
        gif_bg = Image.new("RGB", (thumb_w, thumb_h), "white")
        gif_bg.paste(gif_img, ((thumb_w - gif_img.width) // 2, (thumb_h - gif_img.height) // 2))
        overlay_bg = Image.new("RGB", (thumb_w, thumb_h), "#111")
        if overlay is not None:
            try:
                overlay.seek(min(idx, getattr(overlay, "n_frames", 1) - 1))
                overlay_img = overlay.convert("RGB")
                overlay_img.thumbnail((thumb_w, thumb_h))
                overlay_bg.paste(overlay_img, ((thumb_w - overlay_img.width) // 2, (thumb_h - overlay_img.height) // 2))
            except Exception:
                pass
        ImageDraw.Draw(raw_img).text((5, 5), f"{vocab} raw f{idx}", fill=(255, 255, 0))
        ImageDraw.Draw(gif_bg).text((5, 5), f"body f{idx}", fill=(0, 0, 0))
        ImageDraw.Draw(overlay_bg).text((5, 5), f"overlay f{idx}", fill=(255, 255, 0))
        strips.append((raw_img, gif_bg, overlay_bg))
    cap.release()
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (thumb_w * 3, thumb_h * len(strips)), "#333")
    for row, (raw, gif_img, overlay_img) in enumerate(strips):
        sheet.paste(raw, (0, row * thumb_h))
        sheet.paste(gif_img, (thumb_w, row * thumb_h))
        sheet.paste(overlay_img, (thumb_w * 2, row * thumb_h))
    out = CONTACT_DIR / f"{vocab}_contact.jpg"
    sheet.save(out, quality=92)
    return str(out)


def clear_current_rows(vocabs: list[str]):
    for vocab in vocabs:
        path = DATABASE_DIR / f"{vocab}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "feature_version" not in df.columns:
            continue
        kept = df[df["feature_version"] != fe.FEATURE_SCHEMA].copy()
        kept.to_parquet(path, index=False)
        gif_path = GIF_DIR / f"{vocab}.gif"
        overlay_path = GIF_DIR / "pixel_overlay" / f"{vocab}.gif"
        for stale_path in (gif_path, overlay_path):
            if stale_path.exists():
                stale_path.unlink()


def _append_sequence_to_parquet(vocab: str, video_id: str, split: str, sequence, metadata, source_indices):
    rows = []
    for frame_num, features in enumerate(sequence):
        row = {
            "video_id": video_id,
            "label": vocab,
            "frame_num": int(frame_num),
            "split": split,
            "feature_version": fe.FEATURE_SCHEMA,
            "source_frame_num": int(source_indices[frame_num]) if source_indices else int(frame_num),
            "features": ",".join(map(str, features)),
        }
        if metadata and frame_num < len(metadata):
            row.update(fe.flatten_tracking_metadata(metadata[frame_num]))
        rows.append(row)
    if not rows:
        return
    path = DATABASE_DIR / f"{vocab}.parquet"
    df_new = pd.DataFrame(rows)
    if path.exists():
        df = pd.read_parquet(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(path, index=False)


def reextract_benchmark_stream(vocabs: list[str], limit_files_per_vocab: int | None = None) -> dict:
    di._init_worker()
    summary = {"processed": 0, "failed": []}
    for vocab in vocabs:
        folder = VIDEO_ROOT / vocab
        if not folder.exists():
            summary["failed"].append({"vocab": vocab, "file": None, "message": "missing_video_folder"})
            continue
        media_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTS])
        if limit_files_per_vocab is not None:
            media_files = media_files[: int(limit_files_per_vocab)]
        extracted = []
        for path in media_files:
            video_id = f"train_manual_{path.name}"
            try:
                vid, out_vocab, split, sequence, metadata, source_indices, msg = di._process_media_task(
                    (str(path), vocab, "train", video_id)
                )
            except Exception as exc:
                summary["failed"].append({"vocab": vocab, "file": str(path), "message": str(exc)})
                continue
            if sequence is None:
                summary["failed"].append({"vocab": vocab, "file": str(path), "message": msg})
                continue
            extracted.append((out_vocab, vid, split, sequence, metadata, source_indices))
        if extracted:
            clear_current_rows([vocab])
            for out_vocab, vid, split, sequence, metadata, source_indices in extracted:
                _append_sequence_to_parquet(out_vocab, vid, split, sequence, metadata, source_indices)
            summary["processed"] += len(extracted)
    dbm.update_metadata("db_update")
    return summary


def run_benchmark(
    vocabs: list[str],
    reextract: bool = False,
    limit_files_per_vocab: int | None = None,
    render_body: bool = True,
    render_overlay: bool = True,
    make_contacts: bool = True,
) -> dict:
    if reextract:
        extraction = reextract_benchmark_stream(vocabs, limit_files_per_vocab)
    else:
        extraction = {"processed": 0, "failed": []}
    summary = {"schema": fe.FEATURE_SCHEMA, "extraction": extraction, "vocabs": {}}
    for vocab in vocabs:
        body_gif = vu.generate_vocab_gif(vocab, force_regenerate=True) if render_body else None
        overlay_gif = vu.generate_vocab_pixel_overlay_gif(vocab, force_regenerate=True) if render_overlay else None
        report = write_audit_report(vocab)
        contact = make_contact_sheet(vocab) if make_contacts else None
        summary["vocabs"][vocab] = {
            "failures": report.get("failures", []),
            "body_gif": body_gif,
            "overlay_gif": overlay_gif,
            "contact_sheet": contact,
            "report": str(REPORT_DIR / f"{vocab}_geometry_audit.json"),
        }
    dbm.update_metadata("db_update")
    with open(REPORT_DIR / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reextract", action="store_true", help="Clear current V4.2 rows and re-extract benchmark videos.")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional per-vocab file limit for fast iteration.")
    parser.add_argument("--vocabs", nargs="*", default=BENCHMARK_VOCABS)
    parser.add_argument("--skip-render", action="store_true", help="Skip all GIF/contact-sheet rendering and only extract/audit.")
    parser.add_argument("--skip-body-gif", action="store_true", help="Skip body-frame skeleton GIF rendering.")
    parser.add_argument("--skip-overlay", action="store_true", help="Skip raw pixel-overlay GIF rendering.")
    parser.add_argument("--skip-contact-sheet", action="store_true", help="Skip contact sheet generation.")
    parser.add_argument("--report-only", action="store_true", help="Only write audit reports; implies no extraction or rendering.")
    args = parser.parse_args()
    render_body = not (args.skip_render or args.skip_body_gif or args.report_only)
    render_overlay = not (args.skip_render or args.skip_overlay or args.report_only)
    make_contacts = not (args.skip_render or args.skip_contact_sheet or args.report_only)
    summary = run_benchmark(
        args.vocabs,
        reextract=(args.reextract and not args.report_only),
        limit_files_per_vocab=args.limit_files,
        render_body=render_body,
        render_overlay=render_overlay,
        make_contacts=make_contacts,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
