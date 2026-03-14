#!/usr/bin/env python3
"""
build_openeye_shards.py

Create windowed shard files from the OpenNeuro complementary eyes-open EEG dataset
(preprocessed .set files inside derivatives/).

This script:
- reads participants.tsv
- keeps only AD and CN subjects
- finds derivatives/sub-*/eeg/*.set files
- performs subject-level train/eval split
- loads preprocessed EEG from .set files
- optionally resamples and normalizes
- cuts recordings into overlapping windows
- writes shards:
    X_0.npy, y_0.npy, meta_0.csv, ...

Expected dataset structure:
  <dataset_root>/
      participants.tsv
      derivatives/
          sub-001/eeg/*.set
          sub-002/eeg/*.set
          ...

Output structure:
  <out_dir>/
      train/
          X_0.npy, y_0.npy, meta_0.csv, ...
      eval/
          X_0.npy, y_0.npy, meta_0.csv, ...

Example:
python build_openeye_shards.py \
  --dataset_root /home/e20212/FYP/datasets/Openneuro_openeyes \
  --out_dir /home/e20212/FYP/data/ds_openeyes_ad_cn_shards \
  --target_fs 250 \
  --win_sec 20 \
  --overlap 0.5 \
  --eval_ratio 0.2 \
  --norm_mode recording_standard
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mne


# ============================================================
# Logging
# ============================================================
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("build_openeye_shards")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to: {log_path}")
    return logger


# ============================================================
# Utilities
# ============================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_set_files(dataset_root: Path) -> List[Path]:
    files = list((dataset_root / "derivatives").glob("sub-*/eeg/*.set"))
    if not files:
        files = list((dataset_root / "derivatives").glob("*/sub-*/eeg/*.set"))
    return sorted(files)


def get_subject_id_from_path(set_path: Path) -> str:
    for part in set_path.parts:
        if part.startswith("sub-"):
            return part
    raise ValueError(f"Could not infer subject ID from path: {set_path}")


def infer_binary_label(group_name: str) -> Optional[int]:
    grp = str(group_name).strip().upper()
    if grp in ("AD", "A"):
        return 1
    if grp in ("CN", "C", "HC", "CONTROL", "HEALTHY"):
        return 0
    return None


def load_participants_labels(dataset_root: Path) -> Dict[str, str]:
    p = dataset_root / "participants.tsv"
    if not p.exists():
        raise FileNotFoundError(f"participants.tsv not found: {p}")

    df = pd.read_csv(p, sep="\t")

    if "participant_id" not in df.columns:
        raise ValueError(f"'participant_id' not found. Columns: {list(df.columns)}")

    candidates = ["group", "diagnosis", "condition", "participant_group", "clinical_group"]
    lower_map = {c.lower(): c for c in df.columns}

    group_col = None
    for cand in candidates:
        if cand in lower_map:
            group_col = lower_map[cand]
            break

    if group_col is None:
        raise ValueError(f"Could not find group column. Columns: {list(df.columns)}")

    out = {}
    for _, row in df.iterrows():
        sid = str(row["participant_id"]).strip()
        grp = str(row[group_col]).strip()
        out[sid] = grp

    return out


def standard_recording_normalize(data_ct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = data_ct.mean(axis=1, keepdims=True)
    std = data_ct.std(axis=1, keepdims=True) + eps
    return (data_ct - mean) / std


def robust_recording_normalize(data_ct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    med = np.median(data_ct, axis=1, keepdims=True)
    mad = np.median(np.abs(data_ct - med), axis=1, keepdims=True)
    scale = 1.4826 * mad + eps
    return (data_ct - med) / scale


def normalize_recording(data_ct: np.ndarray, mode: str) -> np.ndarray:
    if mode == "recording_standard":
        return standard_recording_normalize(data_ct)
    if mode == "recording_robust":
        return robust_recording_normalize(data_ct)
    raise ValueError(f"Unknown norm_mode: {mode}")


# ============================================================
# Shard builder
# ============================================================
def build_windowed_shards(
    dataset_root: Path,
    out_dir: Path,
    split: str,
    subjects: List[str],
    label_map: Dict[str, str],
    target_fs: int,
    win_sec: float,
    overlap: float,
    max_minutes_per_subject: Optional[float],
    shard_target_mb: int,
    logger: logging.Logger,
    norm_mode: str,
    apply_bandpass: bool = False,
    bandpass: Tuple[float, float] = (0.5, 45.0),
):
    assert split in ("train", "eval")
    split_dir = out_dir / split
    ensure_dir(split_dir)

    set_files = list_set_files(dataset_root)
    subject_set = set(subjects)
    selected_files = [p for p in set_files if get_subject_id_from_path(p) in subject_set]

    if not selected_files:
        raise FileNotFoundError(f"No .set files found for split={split}")

    logger.info(f"[{split}] subjects={len(subjects)} | set_files={len(selected_files)}")

    step_sec = win_sec * (1.0 - overlap)
    if step_sec <= 0:
        raise ValueError("overlap is too high; step size became non-positive")

    win_len = int(round(win_sec * target_fs))
    step_len = int(round(step_sec * target_fs))
    if win_len <= 0 or step_len <= 0:
        raise ValueError("Invalid window settings")

    shard_target_bytes = shard_target_mb * 1024 * 1024

    shard_id = 0
    X_buf: List[np.ndarray] = []
    y_buf: List[int] = []
    meta_buf: List[Dict[str, Any]] = []
    bytes_per_window = None

    stats = {
        "read_fail": 0,
        "no_eeg": 0,
        "too_short": 0,
        "skipped_non_binary": 0,
        "written_windows": 0,
        "written_subjects": set(),
    }

    def flush():
        nonlocal shard_id, X_buf, y_buf, meta_buf
        if not X_buf:
            return

        X = np.stack(X_buf, axis=0).astype(np.float32)    # (N,C,T)
        y = np.array(y_buf, dtype=np.int64)
        meta = pd.DataFrame(meta_buf)

        np.save(split_dir / f"X_{shard_id}.npy", X)
        np.save(split_dir / f"y_{shard_id}.npy", y)
        meta.to_csv(split_dir / f"meta_{shard_id}.csv", index=False)

        logger.info(
            f"[{split}] wrote shard {shard_id} | X={X.shape} | y={y.shape} | meta_rows={len(meta)}"
        )

        shard_id += 1
        X_buf, y_buf, meta_buf = [], [], []

    for set_path in selected_files:
        sid = get_subject_id_from_path(set_path)
        grp = label_map.get(sid, "")
        y_label = infer_binary_label(grp)

        if y_label is None:
            stats["skipped_non_binary"] += 1
            continue

        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
        except Exception as e:
            logger.warning(f"[{split}] failed reading {set_path.name} ({sid}): {e}")
            stats["read_fail"] += 1
            continue

        raw.pick_types(eeg=True)
        if len(raw.ch_names) == 0:
            logger.warning(f"[{split}] no EEG channels in {set_path}")
            stats["no_eeg"] += 1
            continue

        orig_fs = float(raw.info["sfreq"])
        dur_sec = raw.n_times / max(orig_fs, 1e-8)

        logger.info(
            f"[{split}] loaded {set_path.name} | sid={sid} grp={grp} "
            f"| sfreq={orig_fs:.1f} | duration={dur_sec:.2f}s | channels={len(raw.ch_names)}"
        )

        if int(round(orig_fs)) != int(target_fs):
            raw.resample(target_fs, npad="auto", verbose="ERROR")

        if apply_bandpass:
            raw.filter(
                l_freq=bandpass[0],
                h_freq=bandpass[1],
                method="fir",
                verbose="ERROR",
            )

        data = raw.get_data().astype(np.float32)  # (C,T)
        ch_names = [str(ch) for ch in raw.ch_names]
        C, T_total = data.shape
        local_fs = int(round(raw.info["sfreq"]))

        if max_minutes_per_subject is not None:
            T_cap = int(max_minutes_per_subject * 60.0 * local_fs)
            if T_total > T_cap:
                data = data[:, :T_cap]
                T_total = data.shape[1]

        data = normalize_recording(data, norm_mode)

        if T_total < win_len:
            logger.warning(f"[{split}] too short for one window: {set_path.name}")
            stats["too_short"] += 1
            continue

        n_windows = 1 + (T_total - win_len) // step_len

        if bytes_per_window is None:
            bytes_per_window = C * win_len * 4
            logger.info(
                f"[{split}] window shape=(C={C}, T={win_len}) "
                f"~{bytes_per_window/1024:.1f} KB per window"
            )

        local_written = 0
        for w in range(int(n_windows)):
            start = w * step_len
            end = start + win_len
            x = data[:, start:end]

            X_buf.append(x)
            y_buf.append(int(y_label))
            meta_buf.append(
                {
                    "subject": sid,
                    "file": set_path.name,
                    "group": str(grp).upper(),
                    "label": int(y_label),
                    "win_idx": int(w),
                    "win_start_sec": float(start / local_fs),
                    "win_end_sec": float(end / local_fs),
                    "sfreq": int(local_fs),
                    "n_channels": int(C),
                    "channels": "|".join(ch_names),
                    "source_dataset": dataset_root.name,
                    "recording_type": "eyes_open_photic",
                }
            )

            local_written += 1
            stats["written_windows"] += 1
            stats["written_subjects"].add(sid)

            if bytes_per_window is not None and (len(X_buf) * bytes_per_window) >= shard_target_bytes:
                flush()

        logger.info(f"[{split}] {sid} | {set_path.name} | windows_written={local_written}")

    flush()

    x_files = sorted(split_dir.glob("X_*.npy"))
    if not x_files:
        raise RuntimeError(
            f"[{split}] no shards were written. "
            f"stats={json.dumps({k: (list(v) if isinstance(v, set) else v) for k, v in stats.items()}, indent=2)}"
        )

    logger.info(
        f"[{split}] done | shards={len(x_files)} | windows={stats['written_windows']} "
        f"| subjects_with_windows={len(stats['written_subjects'])}"
    )


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--win_sec", type=float, default=20.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_minutes_per_subject", type=float, default=None)
    parser.add_argument("--shard_target_mb", type=int, default=256)

    parser.add_argument(
        "--norm_mode",
        type=str,
        default="recording_standard",
        choices=["recording_standard", "recording_robust"],
    )

    parser.add_argument("--apply_bandpass", action="store_true")
    parser.add_argument("--bandpass_low", type=float, default=0.5)
    parser.add_argument("--bandpass_high", type=float, default=45.0)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    ensure_dir(out_dir)
    logger = setup_logger(out_dir / "build_shards.log")

    logger.info("========== START ==========")
    logger.info(f"dataset_root = {dataset_root}")
    logger.info(f"out_dir      = {out_dir}")

    label_map = load_participants_labels(dataset_root)
    set_files = list_set_files(dataset_root)

    if not set_files:
        raise FileNotFoundError(f"No .set files found in: {dataset_root / 'derivatives'}")

    subjects_present = sorted({get_subject_id_from_path(p) for p in set_files})

    subjects_ad = [s for s in subjects_present if infer_binary_label(label_map.get(s, "")) == 1]
    subjects_cn = [s for s in subjects_present if infer_binary_label(label_map.get(s, "")) == 0]

    if len(subjects_ad) == 0 or len(subjects_cn) == 0:
        raise RuntimeError(f"No AD/CN subjects found. AD={len(subjects_ad)} CN={len(subjects_cn)}")

    subjects = subjects_ad + subjects_cn
    subj_labels = np.array([infer_binary_label(label_map[s]) for s in subjects], dtype=np.int64)

    logger.info(
        f"subjects present in derivatives = {len(subjects_present)} | "
        f"binary usable = {len(subjects)} | AD={len(subjects_ad)} | CN={len(subjects_cn)}"
    )

    train_subj, eval_subj = train_test_split(
        subjects,
        test_size=args.eval_ratio,
        random_state=args.seed,
        stratify=subj_labels,
    )

    train_subj = sorted(train_subj)
    eval_subj = sorted(eval_subj)

    logger.info(f"subject split | train={len(train_subj)} | eval={len(eval_subj)}")

    build_windowed_shards(
        dataset_root=dataset_root,
        out_dir=out_dir,
        split="train",
        subjects=train_subj,
        label_map=label_map,
        target_fs=args.target_fs,
        win_sec=args.win_sec,
        overlap=args.overlap,
        max_minutes_per_subject=args.max_minutes_per_subject,
        shard_target_mb=args.shard_target_mb,
        logger=logger,
        norm_mode=args.norm_mode,
        apply_bandpass=args.apply_bandpass,
        bandpass=(args.bandpass_low, args.bandpass_high),
    )

    build_windowed_shards(
        dataset_root=dataset_root,
        out_dir=out_dir,
        split="eval",
        subjects=eval_subj,
        label_map=label_map,
        target_fs=args.target_fs,
        win_sec=args.win_sec,
        overlap=args.overlap,
        max_minutes_per_subject=args.max_minutes_per_subject,
        shard_target_mb=args.shard_target_mb,
        logger=logger,
        norm_mode=args.norm_mode,
        apply_bandpass=args.apply_bandpass,
        bandpass=(args.bandpass_low, args.bandpass_high),
    )

    logger.info("========== DONE ==========")


if __name__ == "__main__":
    main()