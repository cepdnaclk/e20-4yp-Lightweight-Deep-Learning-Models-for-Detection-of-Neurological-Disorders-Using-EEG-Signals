#!/usr/bin/env python3
"""
Evaluate a trained AD vs CN CNN1D model (from ds004504 training script) on another
OpenNeuro dataset (e.g., OpenNeuro006036) that has EEGLAB derivatives.

Assumptions (based on your screenshot):
- Dataset structure like:
    Openneuro006036/
      participants.tsv
      derivatives/eeglab/sub-*/eeg/*.set   (or sometimes sub-*/**/*.set)

What this script does:
1) Loads your saved model checkpoint (.pt) that contains:
   - model_state_dict
   - in_channels
2) Builds windowed evaluation shards from OpenNeuro006036 derivatives (no train split)
3) Evaluates the model on all AD/CN windows
4) Optionally reports subject-level metrics by aggregating window predictions

Dependencies:
  pip install numpy pandas torch scikit-learn mne
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict, defaultdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)

import mne


# -----------------------------
# 1) Logging
# -----------------------------
def setup_logger(log_dir: Path, run_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    logger = logging.getLogger(run_name)
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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 2) Dataset utils
# -----------------------------
def list_set_files(dataset_root: Path) -> List[Path]:
    """
    ds004504 script used: dataset_root/derivatives/sub-*/eeg/*.set
    Your ds006036 screenshot shows: dataset_root/derivatives/eeglab/sub-*/
    So we support both patterns.
    """
    patterns = [
        dataset_root / "derivatives" / "sub-*" / "eeg" / "*.set",
        dataset_root / "derivatives" / "eeglab" / "sub-*" / "eeg" / "*.set",
        dataset_root / "derivatives" / "eeglab" / "sub-*" / "**" / "*.set",
        dataset_root / "derivatives" / "**" / "*.set",
    ]
    found = []
    for pat in patterns:
        found.extend(list(pat.parent.glob(pat.name)) if pat.name == "*.set" and pat.parent != dataset_root else list(pat.parent.glob("*.set")))
        # The above line is a safe fallback; we’ll do a proper glob below anyway.
    # Proper globbing (robust):
    found = sorted(set(list((dataset_root / "derivatives").glob("sub-*/eeg/*.set"))
                     + list((dataset_root / "derivatives" / "eeglab").glob("sub-*/eeg/*.set"))
                     + list((dataset_root / "derivatives" / "eeglab").glob("sub-*/**/*.set"))
                     + list((dataset_root / "derivatives").glob("**/*.set"))))
    return found


def get_subject_id_from_path(set_path: Path) -> str:
    for part in set_path.parts:
        if part.startswith("sub-"):
            return part
    raise ValueError(f"Could not infer subject from path: {set_path}")


def zscore_window(x_ct: np.ndarray) -> np.ndarray:
    mean = x_ct.mean(axis=1, keepdims=True)
    std = x_ct.std(axis=1, keepdims=True) + 1e-6
    return (x_ct - mean) / std


def load_participants_labels(dataset_root: Path) -> Dict[str, str]:
    """
    participant_id -> group label
    Supports group column auto-detection.
    """
    p = dataset_root / "participants.tsv"
    if not p.exists():
        raise FileNotFoundError(f"participants.tsv not found: {p}")

    df = pd.read_csv(p, sep="\t")

    candidates = ["group", "diagnosis", "condition", "participant_group", "clinical_group"]
    lower_map = {c.lower(): c for c in df.columns}
    group_col = None
    for cand in candidates:
        if cand in lower_map:
            group_col = lower_map[cand]
            break

    if group_col is None:
        raise ValueError(f"Could not detect group column. Columns: {list(df.columns)}")

    if "participant_id" not in df.columns:
        raise ValueError(f"participants.tsv missing 'participant_id'. Columns: {list(df.columns)}")

    label_map = {}
    for _, row in df.iterrows():
        sid = str(row["participant_id"]).strip()
        grp = str(row[group_col]).strip()
        label_map[sid] = grp
    return label_map


def map_to_binary_label(grp: str) -> Optional[int]:
    """
    AD=1, CN=0. Supports both full and single-letter encodings.
    """
    g = (grp or "").strip().upper()
    if g in ("AD", "A"):
        return 1
    if g in ("CN", "C"):
        return 0
    return None


# -----------------------------
# 3) Build eval shards
# -----------------------------
def build_eval_shards(
    dataset_root: Path,
    out_dir: Path,
    label_map: Dict[str, str],
    target_fs: int,
    win_sec: float,
    overlap: float,
    max_minutes_per_subject: Optional[float],
    shard_target_mb: int,
    logger: logging.Logger,
    apply_bandpass: bool = False,
    bandpass: Tuple[float, float] = (0.5, 45.0),
):
    """
    Writes:
      out_dir/eval/X_k.npy (N,C,T), y_k.npy (N,), meta_k.csv (N rows)
    """
    split = "eval"
    split_dir = out_dir / split
    ensure_dir(split_dir)

    set_files = list_set_files(dataset_root)
    if not set_files:
        raise FileNotFoundError(f"No .set files found under: {dataset_root / 'derivatives'}")

    logger.info(f"[eval] Found .set files: {len(set_files)}")

    # Window params
    step_sec = win_sec * (1.0 - overlap)
    if step_sec <= 0:
        raise ValueError("overlap too high -> step_sec <= 0")

    win_len = int(win_sec * target_fs)
    step_len = int(step_sec * target_fs)
    if win_len <= 0 or step_len <= 0:
        raise ValueError("Invalid window config")

    shard_target_bytes = shard_target_mb * 1024 * 1024
    bytes_per_window = None

    shard_id = 0
    X_buf: List[np.ndarray] = []
    y_buf: List[int] = []
    meta_rows: List[Dict[str, Any]] = []

    def flush():
        nonlocal shard_id, X_buf, y_buf, meta_rows
        if not X_buf:
            return
        X = np.stack(X_buf, axis=0).astype(np.float32)
        y = np.array(y_buf, dtype=np.int64)
        meta = pd.DataFrame(meta_rows)

        np.save(split_dir / f"X_{shard_id}.npy", X)
        np.save(split_dir / f"y_{shard_id}.npy", y)
        meta.to_csv(split_dir / f"meta_{shard_id}.csv", index=False)

        logger.info(f"[eval] Wrote shard {shard_id}: X={X.shape} | y={y.shape} | meta={len(meta)}")
        shard_id += 1
        X_buf, y_buf, meta_rows = [], [], []

    # Diagnostics
    read_fail = 0
    no_eeg = 0
    too_short = 0
    skipped_nonbinary = 0
    wrote_windows = 0

    for set_path in set_files:
        sid = get_subject_id_from_path(set_path)
        grp = label_map.get(sid, "")
        y_label = map_to_binary_label(grp)
        if y_label is None:
            skipped_nonbinary += 1
            continue

        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
        except Exception as e:
            logger.warning(f"[eval] Failed reading {set_path} ({sid}): {e}")
            read_fail += 1
            continue

        raw.pick_types(eeg=True)
        if len(raw.ch_names) == 0:
            raw.set_channel_types({ch: "eeg" for ch in raw.info["ch_names"]})
            raw.pick_types(eeg=True)
            if len(raw.ch_names) == 0:
                logger.warning(f"[eval] No EEG channels: {set_path}")
                no_eeg += 1
                continue

        dur_sec_orig = raw.n_times / float(raw.info["sfreq"])
        logger.info(
            f"[eval] Loaded {set_path.name} | sid={sid} grp={str(grp).upper()} "
            f"| sfreq={raw.info['sfreq']:.1f} n_times={raw.n_times} dur_sec={dur_sec_orig:.2f}"
        )

        if int(raw.info["sfreq"]) != int(target_fs):
            raw.resample(target_fs, npad="auto", verbose="ERROR")

        if apply_bandpass:
            l_freq, h_freq = bandpass
            raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose="ERROR")

        data = raw.get_data()  # (C, T_total)
        C, T_total = data.shape

        if max_minutes_per_subject is not None:
            max_sec = max_minutes_per_subject * 60.0
            T_cap = int(max_sec * target_fs)
            if T_total > T_cap:
                data = data[:, :T_cap]
                T_total = data.shape[1]

        if T_total < win_len:
            logger.warning(
                f"[eval] Too short for one window: {set_path.name} | samples={T_total} < win_len={win_len}"
            )
            too_short += 1
            continue

        n_windows = 1 + (T_total - win_len) // step_len
        local_written = 0

        if bytes_per_window is None:
            bytes_per_window = C * win_len * 4
            logger.info(f"Window shape: (C={C}, T={win_len}) ~ {bytes_per_window/1024:.1f} KB/window")

        for w in range(int(n_windows)):
            start = w * step_len
            end = start + win_len
            x = data[:, start:end].astype(np.float32)
            x = zscore_window(x)

            X_buf.append(x)
            y_buf.append(int(y_label))
            meta_rows.append(
                {
                    "subject": sid,
                    "file": set_path.name,
                    "group": str(grp).upper(),
                    "label": int(y_label),
                    "win_start_sec": float(start / target_fs),
                    "win_end_sec": float(end / target_fs),
                    "sfreq": int(target_fs),
                }
            )
            local_written += 1

            if bytes_per_window is not None and (len(X_buf) * bytes_per_window) >= shard_target_bytes:
                flush()

        wrote_windows += local_written
        logger.info(f"[eval] {sid} ({str(grp).upper()}) | {set_path.name} | windows_written={local_written}")

    flush()

    x_files = sorted((out_dir / "eval").glob("X_*.npy"))
    if not x_files:
        raise RuntimeError(
            f"[eval] No shards written. read_fail={read_fail} no_eeg={no_eeg} "
            f"too_short={too_short} skipped_nonbinary={skipped_nonbinary} wrote_windows={wrote_windows}"
        )

    logger.info(
        f"[eval] Total shards: {len(x_files)} | total_windows={wrote_windows} | "
        f"skipped_nonbinary={skipped_nonbinary}"
    )


# -----------------------------
# 4) Sharded Dataset
# -----------------------------
def to_nct(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got {X.shape}")
    return X


class ShardedNpyDataset(Dataset):
    def __init__(self, split_dir: Path, cache_size: int, logger: logging.Logger):
        self.split_dir = split_dir
        self.logger = logger
        self.cache_size = cache_size
        self._x_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

        self.x_paths = sorted(split_dir.glob("X_*.npy"))
        if not self.x_paths:
            raise FileNotFoundError(f"No X_*.npy in {split_dir}")

        self.shards = []
        self.shard_offsets = []
        total = 0

        for xp in self.x_paths:
            sid = xp.stem.split("_")[1]
            yp = split_dir / f"y_{sid}.npy"
            mp = split_dir / f"meta_{sid}.csv"
            if not yp.exists():
                raise FileNotFoundError(f"Missing {yp.name} for {xp.name}")

            x_mmap = to_nct(np.load(xp, mmap_mode="r"))
            y_mmap = np.load(yp, mmap_mode="r")
            n = int(x_mmap.shape[0])

            if int(y_mmap.shape[0]) != n:
                raise ValueError(f"Row mismatch: {xp.name} n={n} vs {yp.name} n={int(y_mmap.shape[0])}")

            self.shards.append({"x_path": xp, "y_path": yp, "meta_path": mp, "n": n})
            self.shard_offsets.append(total)
            total += n

            logger.info(f"[{split_dir.name}] shard {sid}: n={n} | {xp.name}")

        self.total_len = total

    def __len__(self):
        return self.total_len

    def _get_x_mmap(self, x_path: Path) -> np.ndarray:
        key = str(x_path)
        if key in self._x_cache:
            self._x_cache.move_to_end(key)
            return self._x_cache[key]
        x_mmap = to_nct(np.load(x_path, mmap_mode="r"))
        self._x_cache[key] = x_mmap
        if len(self._x_cache) > self.cache_size:
            self._x_cache.popitem(last=False)
        return x_mmap

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.shard_offsets[mid]
            end = start + self.shards[mid]["n"]
            if start <= idx < end:
                return mid, idx - start
            if idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        shard_id, local_i = self._locate(idx)
        shard = self.shards[shard_id]
        x_mmap = self._get_x_mmap(shard["x_path"])
        y_mmap = np.load(shard["y_path"], mmap_mode="r")

        x = x_mmap[local_i].astype(np.float32)  # (C,T)
        y = int(y_mmap[local_i])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# -----------------------------
# 5) Model (must match training)
# -----------------------------
class CNN1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.head(x)


@torch.no_grad()
def evaluate_windows(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        preds = torch.argmax(logits, dim=1)

        y_true.append(yb.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
    cm = confusion_matrix(y_true, y_pred).tolist() if len(y_true) else [[0, 0], [0, 0]]

    report_text = classification_report(
        y_true, y_pred, digits=4, target_names=["CN(0)", "AD(1)"]
    ) if len(y_true) else "No samples."

    report_dict = classification_report(
        y_true, y_pred, digits=4, output_dict=True, target_names=["CN(0)", "AD(1)"]
    ) if len(y_true) else {}

    return {
        "n_windows": int(len(y_true)),
        "acc": acc,
        "balanced_acc": bal_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "report_text": report_text,
        "report_dict": report_dict,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def subject_level_from_meta(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, logger: logging.Logger):
    """
    Aggregates window-level predictions into subject-level predictions using majority vote.
    Uses the meta_*.csv files written alongside the eval shards.
    """
    eval_dir = out_dir / "eval"
    meta_paths = sorted(eval_dir.glob("meta_*.csv"))
    if not meta_paths:
        logger.warning("No meta_*.csv found. Skipping subject-level metrics.")
        return None

    # Build subject list in the same order windows were written:
    # We concatenate meta files in shard order, same as X/y concatenation order in ShardedNpyDataset.
    metas = []
    for mp in meta_paths:
        metas.append(pd.read_csv(mp))
    meta_all = pd.concat(metas, ignore_index=True)

    if len(meta_all) != len(y_true):
        logger.warning(
            f"Meta rows != number of windows: meta={len(meta_all)} vs y_true={len(y_true)}. "
            "Skipping subject-level metrics."
        )
        return None

    # Aggregate
    subj_true = defaultdict(list)
    subj_pred = defaultdict(list)
    for i in range(len(meta_all)):
        sid = str(meta_all.loc[i, "subject"])
        subj_true[sid].append(int(y_true[i]))
        subj_pred[sid].append(int(y_pred[i]))

    y_true_subj = []
    y_pred_subj = []
    subjects = sorted(subj_true.keys())

    for sid in subjects:
        t = subj_true[sid]
        p = subj_pred[sid]
        # true label should be constant; take majority just in case
        true_vote = int(round(np.mean(t)))
        pred_vote = 1 if (np.mean(p) >= 0.5) else 0
        y_true_subj.append(true_vote)
        y_pred_subj.append(pred_vote)

    y_true_subj = np.array(y_true_subj, dtype=np.int64)
    y_pred_subj = np.array(y_pred_subj, dtype=np.int64)

    acc = float(accuracy_score(y_true_subj, y_pred_subj)) if len(y_true_subj) else 0.0
    bal_acc = float(balanced_accuracy_score(y_true_subj, y_pred_subj)) if len(y_true_subj) else 0.0
    macro_f1 = float(f1_score(y_true_subj, y_pred_subj, average="macro")) if len(y_true_subj) else 0.0
    cm = confusion_matrix(y_true_subj, y_pred_subj).tolist() if len(y_true_subj) else [[0, 0], [0, 0]]

    report_text = classification_report(
        y_true_subj, y_pred_subj, digits=4, target_names=["CN(0)", "AD(1)"]
    ) if len(y_true_subj) else "No subjects."

    report_dict = classification_report(
        y_true_subj, y_pred_subj, digits=4, output_dict=True, target_names=["CN(0)", "AD(1)"]
    ) if len(y_true_subj) else {}

    return {
        "n_subjects": int(len(y_true_subj)),
        "acc": acc,
        "balanced_acc": bal_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "report_text": report_text,
        "report_dict": report_dict,
        "subjects": subjects,
    }


# -----------------------------
# 6) Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate ds004504-trained CNN1D model on OpenNeuro006036.")
    parser.add_argument("--fyp_root", type=str, default=str(Path.home() / "FYP"))
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to OpenNeuro006036 root folder")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to trained .pt model checkpoint")
    parser.add_argument("--out_dir", type=str, default="data/ds006036_eval_shards")

    # Preprocessing
    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--win_sec", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max_minutes_per_subject", type=float, default=None)
    parser.add_argument("--shard_target_mb", type=int, default=256)

    parser.add_argument("--apply_bandpass", action="store_true")
    parser.add_argument("--bandpass_low", type=float, default=0.5)
    parser.add_argument("--bandpass_high", type=float, default=45.0)

    # Loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=2)

    # Control
    parser.add_argument("--rebuild_shards", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    fyp_root = Path(args.fyp_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = (fyp_root / args.out_dir).resolve()
    ckpt_path = Path(args.model_ckpt).resolve()

    run_name = f"eval_ds006036_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(fyp_root / "experiments" / "logs", run_name)

    logger.info("========== RUN START ==========")
    logger.info(f"Dataset root : {dataset_root}")
    logger.info(f"Checkpoint   : {ckpt_path}")
    logger.info(f"Out dir      : {out_dir}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in ckpt or "in_channels" not in ckpt:
        raise ValueError("Checkpoint missing required keys: model_state_dict and in_channels")

    in_channels = int(ckpt["in_channels"])
    logger.info(f"Model expects in_channels={in_channels}")

    label_map = load_participants_labels(dataset_root)

    eval_dir = out_dir / "eval"
    need_build = args.rebuild_shards or (not eval_dir.exists()) or (len(list(eval_dir.glob("X_*.npy"))) == 0)

    if need_build:
        logger.info("========== BUILDING EVAL SHARDS ==========")
        ensure_dir(out_dir)

        build_eval_shards(
            dataset_root=dataset_root,
            out_dir=out_dir,
            label_map=label_map,
            target_fs=args.target_fs,
            win_sec=args.win_sec,
            overlap=args.overlap,
            max_minutes_per_subject=args.max_minutes_per_subject,
            shard_target_mb=args.shard_target_mb,
            logger=logger,
            apply_bandpass=args.apply_bandpass,
            bandpass=(args.bandpass_low, args.bandpass_high),
        )
    else:
        logger.info("Eval shards exist. Skipping rebuild.")

    logger.info("========== LOADING EVAL SHARDS ==========")
    eval_ds = ShardedNpyDataset(eval_dir, cache_size=args.cache_size, logger=logger)

    # Sanity check channels
    x0, _ = eval_ds[0]
    if int(x0.shape[0]) != in_channels:
        raise RuntimeError(
            f"Channel mismatch: eval data has C={int(x0.shape[0])}, but model expects C={in_channels}. "
            "If ds006036 has different channel count/order, you must align channels to the training set."
        )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = CNN1D(in_channels=in_channels, num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    logger.info("Loaded model weights.")

    logger.info("========== WINDOW-LEVEL EVALUATION ==========")
    t0 = time.time()
    results_win = evaluate_windows(model, eval_loader, device)
    t_eval = time.time() - t0

    logger.info(f"Eval windows: {results_win['n_windows']} | time_sec={t_eval:.2f}")
    logger.info(f"Window Acc        : {results_win['acc']:.4f}")
    logger.info(f"Window BalancedAcc: {results_win['balanced_acc']:.4f}")
    logger.info(f"Window MacroF1    : {results_win['macro_f1']:.4f}")
    logger.info("Window Confusion matrix:\n" + str(results_win["confusion_matrix"]))
    logger.info("Window Classification report:\n" + results_win["report_text"])

    logger.info("========== SUBJECT-LEVEL (MAJORITY VOTE) ==========")
    results_subj = subject_level_from_meta(out_dir, results_win["y_true"], results_win["y_pred"], logger)
    if results_subj is not None:
        logger.info(f"Eval subjects: {results_subj['n_subjects']}")
        logger.info(f"Subject Acc        : {results_subj['acc']:.4f}")
        logger.info(f"Subject BalancedAcc: {results_subj['balanced_acc']:.4f}")
        logger.info(f"Subject MacroF1    : {results_subj['macro_f1']:.4f}")
        logger.info("Subject Confusion matrix:\n" + str(results_subj["confusion_matrix"]))
        logger.info("Subject Classification report:\n" + results_subj["report_text"])
    else:
        logger.info("Subject-level metrics not computed.")

    # Save metrics
    exp_dir = fyp_root / "experiments" / run_name
    ensure_dir(exp_dir)

    metrics = {
        "run_name": run_name,
        "dataset_root": str(dataset_root),
        "model_ckpt": str(ckpt_path),
        "preprocess": {
            "target_fs": args.target_fs,
            "win_sec": args.win_sec,
            "overlap": args.overlap,
            "max_minutes_per_subject": args.max_minutes_per_subject,
            "apply_bandpass": bool(args.apply_bandpass),
            "bandpass": [args.bandpass_low, args.bandpass_high],
        },
        "window_level": {
            "n_windows": results_win["n_windows"],
            "acc": results_win["acc"],
            "balanced_acc": results_win["balanced_acc"],
            "macro_f1": results_win["macro_f1"],
            "confusion_matrix": results_win["confusion_matrix"],
            "classification_report": results_win["report_dict"],
        },
        "subject_level": None if results_subj is None else {
            "n_subjects": results_subj["n_subjects"],
            "acc": results_subj["acc"],
            "balanced_acc": results_subj["balanced_acc"],
            "macro_f1": results_subj["macro_f1"],
            "confusion_matrix": results_subj["confusion_matrix"],
            "classification_report": results_subj["report_dict"],
        }
    }

    with open(exp_dir / "metrics_ds006036.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics: {exp_dir / 'metrics_ds006036.json'}")
    logger.info("========== RUN END ==========")


if __name__ == "__main__":
    main()