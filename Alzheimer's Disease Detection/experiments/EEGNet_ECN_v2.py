#!/usr/bin/env python3
"""
Binary AD vs CN classifier on OpenNeuro ds004504 (v1.0.2) using derivatives EEGLAB .set files.

Improved pipeline:
- Read derivatives/sub-*/eeg/*.set
- Subject-level split (no leakage)
- Recording-level normalization (not per-window normalization)
- Windowing into (C,T)
- Save shards (X_*.npy, y_*.npy, meta_*.csv)
- Train improved raw+spectral EEG model
- Train feature-correction ECN on top of frozen base encoder
- Evaluate at both window level and subject level

Dependencies:
  pip install numpy pandas torch scikit-learn mne
"""

import sys
import os
import json
import time
import math
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
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

import mne


# ============================================================
# 1) Logging
# ============================================================
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


# ============================================================
# 2) Utils
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_set_files(dataset_root: Path) -> List[Path]:
    return sorted((dataset_root / "derivatives").glob("sub-*/eeg/*.set"))


def get_subject_id_from_path(set_path: Path) -> str:
    for part in set_path.parts:
        if part.startswith("sub-"):
            return part
    raise ValueError(f"Could not infer subject from path: {set_path}")


def load_participants_labels(dataset_root: Path) -> Dict[str, str]:
    """
    Map participant_id -> group label.
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


def robust_recording_normalize(data_ct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    data_ct: (C, T)
    Robust per-channel normalization over full recording.
    """
    med = np.median(data_ct, axis=1, keepdims=True)
    mad = np.median(np.abs(data_ct - med), axis=1, keepdims=True)
    scale = 1.4826 * mad + eps
    return (data_ct - med) / scale


def standard_recording_normalize(data_ct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    data_ct: (C, T)
    Standard per-channel normalization over full recording.
    """
    mean = data_ct.mean(axis=1, keepdims=True)
    std = data_ct.std(axis=1, keepdims=True) + eps
    return (data_ct - mean) / std


def infer_binary_label(group_name: str) -> Optional[int]:
    grp_u = str(group_name).upper()
    if grp_u in ("AD", "A"):
        return 1
    if grp_u in ("CN", "C"):
        return 0
    return None


# ============================================================
# 3) Build shards
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
    apply_bandpass: bool = False,
    bandpass: Tuple[float, float] = (0.5, 45.0),
    norm_mode: str = "recording_standard",  # recording_standard | recording_robust
):
    """
    Writes:
      out_dir/<split>/X_k.npy (N,C,T), y_k.npy (N,), meta_k.csv (N rows)
    """
    assert split in ("train", "eval")
    split_dir = out_dir / split
    ensure_dir(split_dir)

    set_files = list_set_files(dataset_root)
    subject_set_files = [p for p in set_files if get_subject_id_from_path(p) in set(subjects)]
    if not subject_set_files:
        raise FileNotFoundError(f"No .set files for split={split}")

    logger.info(f"[{split}] Subjects: {len(subjects)} | .set files: {len(subject_set_files)}")

    step_sec = win_sec * (1.0 - overlap)
    if step_sec <= 0:
        raise ValueError("overlap too high -> non-positive step size")

    win_len = int(win_sec * target_fs)
    step_len = int(step_sec * target_fs)
    if win_len <= 0 or step_len <= 0:
        raise ValueError("Invalid window settings")

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

        logger.info(f"[{split}] Wrote shard {shard_id}: X={X.shape} y={y.shape} meta={len(meta)}")
        shard_id += 1
        X_buf, y_buf, meta_rows = [], [], []

    read_fail = 0
    no_eeg = 0
    too_short = 0
    wrote_windows = 0

    for set_path in subject_set_files:
        sid = get_subject_id_from_path(set_path)
        grp = label_map.get(sid, "")
        y_label = infer_binary_label(grp)
        if y_label is None:
            continue

        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
        except Exception as e:
            logger.warning(f"[{split}] Failed reading {set_path.name} ({sid}): {e}")
            read_fail += 1
            continue

        raw.pick_types(eeg=True)
        if len(raw.ch_names) == 0:
            raw.set_channel_types({ch: "eeg" for ch in raw.info["ch_names"]})
            raw.pick_types(eeg=True)
            if len(raw.ch_names) == 0:
                logger.warning(f"[{split}] No EEG channels: {set_path}")
                no_eeg += 1
                continue

        orig_sfreq = float(raw.info["sfreq"])
        dur_sec_orig = raw.n_times / orig_sfreq

        logger.info(
            f"[{split}] Loaded {set_path.name} | sid={sid} grp={grp} "
            f"| sfreq={orig_sfreq:.1f} n_times={raw.n_times} dur_sec={dur_sec_orig:.2f}"
        )

        if int(raw.info["sfreq"]) != int(target_fs):
            raw.resample(target_fs, npad="auto", verbose="ERROR")

        if apply_bandpass:
            l_freq, h_freq = bandpass
            raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose="ERROR")

        data = raw.get_data().astype(np.float32)  # (C,T)
        C, T_total = data.shape

        if max_minutes_per_subject is not None:
            T_cap = int(max_minutes_per_subject * 60.0 * target_fs)
            if T_total > T_cap:
                data = data[:, :T_cap]
                T_total = data.shape[1]

        # Recording-level normalization
        if norm_mode == "recording_standard":
            data = standard_recording_normalize(data)
        elif norm_mode == "recording_robust":
            data = robust_recording_normalize(data)
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

        if T_total < win_len:
            logger.warning(f"[{split}] Too short for one window: {set_path.name}")
            too_short += 1
            continue

        n_windows = 1 + (T_total - win_len) // step_len
        local_written = 0

        if bytes_per_window is None:
            bytes_per_window = C * win_len * 4
            logger.info(f"Window shape: (C={C}, T={win_len}) ~ {bytes_per_window/1024:.1f} KB/window")

        ch_names = raw.ch_names
        local_sfreq = int(raw.info["sfreq"])

        for w in range(int(n_windows)):
            start = w * step_len
            end = start + win_len
            x = data[:, start:end].astype(np.float32)

            X_buf.append(x)
            y_buf.append(int(y_label))
            meta_rows.append(
                {
                    "subject": sid,
                    "file": set_path.name,
                    "group": str(grp).upper(),
                    "label": int(y_label),
                    "win_idx": int(w),
                    "win_start_sec": float(start / local_sfreq),
                    "win_end_sec": float(end / local_sfreq),
                    "sfreq": int(local_sfreq),
                    "n_channels": int(C),
                    "channels": "|".join(ch_names),
                }
            )
            local_written += 1

            if (len(X_buf) * bytes_per_window) >= shard_target_bytes:
                flush()

        wrote_windows += local_written
        logger.info(f"[{split}] {sid} ({grp}) | {set_path.name} | windows_written={local_written}")

    flush()

    x_files = sorted(split_dir.glob("X_*.npy"))
    if not x_files:
        raise RuntimeError(
            f"[{split}] No shards written. "
            f"read_fail={read_fail} no_eeg={no_eeg} too_short={too_short} wrote_windows={wrote_windows}"
        )

    logger.info(f"[{split}] Total shards: {len(x_files)} | total_windows={wrote_windows}")


# ============================================================
# 4) Sharded Dataset
# ============================================================
def to_nct(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got {X.shape}")
    return X


class ShardedNpyDataset(Dataset):
    """
    Returns:
      x: tensor (C,T)
      y: tensor scalar
      meta: dict-like python metadata
    """
    def __init__(self, split_dir: Path, cache_size: int, logger: logging.Logger):
        self.split_dir = split_dir
        self.cache_size = cache_size
        self.logger = logger
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
            if not mp.exists():
                raise FileNotFoundError(f"Missing {mp.name} for {xp.name}")

            x_mmap = to_nct(np.load(xp, mmap_mode="r"))
            y_mmap = np.load(yp, mmap_mode="r")
            meta_df = pd.read_csv(mp)

            n = int(x_mmap.shape[0])
            if int(y_mmap.shape[0]) != n or len(meta_df) != n:
                raise ValueError(f"Row mismatch in shard {sid}")

            self.shards.append(
                {
                    "x_path": xp,
                    "y_path": yp,
                    "meta_df": meta_df,
                    "n": n,
                }
            )
            self.shard_offsets.append(total)
            total += n

            logger.info(f"[{split_dir.name}] shard {sid}: n={n} | {xp.name}")

        self.total_len = total

        # Precompute global index -> subject / label for sampling
        self.subjects = []
        self.labels = []
        for shard in self.shards:
            meta_df = shard["meta_df"]
            self.subjects.extend(meta_df["subject"].astype(str).tolist())
            self.labels.extend(meta_df["label"].astype(int).tolist())

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
        meta_row = shard["meta_df"].iloc[local_i].to_dict()

        x = x_mmap[local_i].astype(np.float32)
        y = int(y_mmap[local_i])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), meta_row


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    xb = torch.stack(xs, dim=0)
    yb = torch.stack(ys, dim=0)
    return xb, yb, list(metas)


def build_subject_balanced_sampler(ds: ShardedNpyDataset) -> WeightedRandomSampler:
    """
    Weight windows so that:
    - subjects contribute more equally
    - classes are also balanced
    """
    subjects = ds.subjects
    labels = ds.labels

    subject_counts = defaultdict(int)
    class_counts = defaultdict(int)

    for s in subjects:
        subject_counts[s] += 1
    for y in labels:
        class_counts[int(y)] += 1

    sample_weights = []
    for s, y in zip(subjects, labels):
        w_subject = 1.0 / max(subject_counts[s], 1)
        w_class = 1.0 / max(class_counts[int(y)], 1)
        sample_weights.append(w_subject * w_class)

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ============================================================
# 5) Model blocks
# ============================================================
class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        s = x.mean(dim=-1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)
        return x * s


class DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int, drop: float):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()
        self.se = SE1D(out_ch, reduction=8)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.drop(x)
        return x


class SpectralBandPower(nn.Module):
    """
    Differentiable spectral branch using torch.fft.
    Input: (B,C,T)
    Output: (B, C*5)
    Bands:
      delta: 0.5-4
      theta: 4-8
      alpha: 8-13
      beta : 13-30
      gamma: 30-45
    """
    def __init__(self, fs: int):
        super().__init__()
        self.fs = fs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        B, C, T = x.shape
        Xf = torch.fft.rfft(x, dim=-1)
        psd = (Xf.real ** 2 + Xf.imag ** 2) / max(T, 1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(x.device)

        bands = [
            (0.5, 4.0),
            (4.0, 8.0),
            (8.0, 13.0),
            (13.0, 30.0),
            (30.0, 45.0),
        ]

        out = []
        total_power = psd.sum(dim=-1, keepdim=True) + 1e-8
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() == 0:
                bp = torch.zeros((B, C), device=x.device, dtype=x.dtype)
            else:
                bp = psd[:, :, mask].sum(dim=-1)
            bp_rel = bp / total_power.squeeze(-1)
            out.append(bp_rel)

        feat = torch.cat(out, dim=1)  # (B, C*5)
        return feat


class ImprovedEEGNet(nn.Module):
    """
    Stronger base model:
    - temporal CNN
    - channel attention
    - spectral bandpower branch
    - fused embedding
    """
    def __init__(self, n_channels: int, fs: int, n_classes: int = 2, emb_dim: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.fs = fs
        self.n_classes = n_classes
        self.emb_dim = emb_dim

        self.raw_stem = nn.Sequential(
            nn.Conv1d(n_channels, 48, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            SE1D(48, reduction=8),
        )
        self.block1 = DSConvBlock(48, 64, k=11, s=2, p=5, drop=0.10)
        self.block2 = DSConvBlock(64, 96, k=7, s=2, p=3, drop=0.12)
        self.block3 = DSConvBlock(96, 128, k=5, s=2, p=2, drop=0.15)

        self.attn = nn.Conv1d(128, 1, kernel_size=1)

        self.spectral = SpectralBandPower(fs=fs)
        self.spec_proj = nn.Sequential(
            nn.Linear(n_channels * 5, 96),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 64),
            nn.SiLU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(128 + 64, emb_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.raw_stem(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)

        w = torch.softmax(self.attn(z).squeeze(1), dim=-1)   # (B,T')
        raw_emb = torch.sum(z * w.unsqueeze(1), dim=-1)      # (B,128)

        spec = self.spectral(x)                              # (B,C*5)
        spec_emb = self.spec_proj(spec)                      # (B,64)

        emb = self.fuse(torch.cat([raw_emb, spec_emb], dim=1))  # (B,emb_dim)
        return emb

    def classify_from_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encode(x)
        return self.classify_from_embedding(emb)


class FeatureCorrectionNetwork(nn.Module):
    """
    Feature-level ECN:
    - sees raw EEG + base embedding + base logits + confidence meta
    - outputs delta embedding
    - corrected embedding = base_emb + delta
    - classification done through base head
    """
    def __init__(self, n_channels: int, emb_dim: int, n_classes: int = 2, scale_init: float = 0.25):
        super().__init__()
        self.sig = nn.Sequential(
            nn.Conv1d(n_channels, 24, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(24, 32, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 32 signal + emb_dim + 2 logits + 4 meta
        in_dim = 32 + emb_dim + n_classes + 4

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, max(128, emb_dim)),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(max(128, emb_dim), emb_dim),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    @staticmethod
    def _meta_from_logits(base_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(base_logits, dim=1)
        max_p, _ = probs.max(dim=1, keepdim=True)
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        margin = sorted_probs[:, :1] - sorted_probs[:, 1:2]
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1, keepdim=True)
        pos_prob = probs[:, 1:2]
        return torch.cat([max_p, margin, entropy, pos_prob], dim=1)

    def forward(self, x: torch.Tensor, base_emb: torch.Tensor, base_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sig_feat = self.sig(x).squeeze(-1)  # (B,32)
        meta = self._meta_from_logits(base_logits)
        feat = torch.cat([sig_feat, base_emb, base_logits, meta], dim=1)
        delta = torch.tanh(self.mlp(feat)) * self.scale
        corrected_emb = base_emb + delta
        return corrected_emb, delta


# ============================================================
# 6) Loss + AMP helpers
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def amp_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    class _Null:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return _Null()


# ============================================================
# 7) Evaluation helpers
# ============================================================
def build_subject_metrics(y_true_subj: List[int], y_pred_subj: List[int]) -> Dict[str, Any]:
    y_true_subj = np.asarray(y_true_subj, dtype=np.int64)
    y_pred_subj = np.asarray(y_pred_subj, dtype=np.int64)

    if len(y_true_subj) == 0:
        return {
            "acc": 0.0,
            "balanced_acc": 0.0,
            "macro_f1": 0.0,
            "cm": [[0, 0], [0, 0]],
            "report": {},
            "n_subjects": 0,
        }

    acc = accuracy_score(y_true_subj, y_pred_subj)
    bal = balanced_accuracy_score(y_true_subj, y_pred_subj)
    f1 = f1_score(y_true_subj, y_pred_subj, average="macro")
    cm = confusion_matrix(y_true_subj, y_pred_subj).tolist()
    rep = classification_report(
        y_true_subj,
        y_pred_subj,
        output_dict=True,
        digits=4,
        target_names=["CN(0)", "AD(1)"],
        zero_division=0,
    )

    return {
        "acc": float(acc),
        "balanced_acc": float(bal),
        "macro_f1": float(f1),
        "cm": cm,
        "report": rep,
        "n_subjects": int(len(y_true_subj)),
    }


@torch.no_grad()
def evaluate_base(
    model: ImprovedEEGNet,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
):
    model.eval()

    total_loss, total_correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    subj_probs = defaultdict(list)
    subj_true = {}

    for xb, yb, metas in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with amp_autocast(device, use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.item()) * yb.size(0)
        total_correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

        probs_np = probs.detach().cpu().numpy()
        y_np = yb.detach().cpu().numpy()

        for i, meta in enumerate(metas):
            sid = str(meta["subject"])
            subj_probs[sid].append(float(probs_np[i]))
            subj_true[sid] = int(y_np[i])

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") if len(y_true) else 0.0

    subj_ids = sorted(subj_probs.keys())
    y_true_subj, y_pred_subj, y_prob_subj = [], [], []

    for sid in subj_ids:
        p = float(np.mean(subj_probs[sid]))
        y_prob_subj.append(p)
        y_true_subj.append(int(subj_true[sid]))
        y_pred_subj.append(int(p >= 0.5))

    subj_metrics = build_subject_metrics(y_true_subj, y_pred_subj)

    return {
        "window_loss": float(avg_loss),
        "window_acc": float(acc),
        "window_bal_acc": float(bal_acc),
        "window_macro_f1": float(macro_f1),
        "y_true": y_true,
        "y_pred": y_pred,
        "subject_metrics": subj_metrics,
        "subject_probs": {k: float(np.mean(v)) for k, v in subj_probs.items()},
        "subject_true": subj_true,
    }


@torch.no_grad()
def evaluate_ecn(
    base_model: ImprovedEEGNet,
    ecn: FeatureCorrectionNetwork,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
    reg_lambda: float,
):
    base_model.eval()
    ecn.eval()

    total_loss, total_correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    subj_probs = defaultdict(list)
    subj_true = {}

    for xb, yb, metas in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with amp_autocast(device, use_amp):
            base_emb = base_model.encode(xb)
            base_logits = base_model.classify_from_embedding(base_emb)

            corr_emb, delta = ecn(xb, base_emb, base_logits)
            corr_logits = base_model.classify_from_embedding(corr_emb)

            loss = criterion(corr_logits, yb) + reg_lambda * (delta ** 2).mean()

        probs = torch.softmax(corr_logits, dim=1)[:, 1]
        preds = torch.argmax(corr_logits, dim=1)

        total_loss += float(loss.item()) * yb.size(0)
        total_correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

        probs_np = probs.detach().cpu().numpy()
        y_np = yb.detach().cpu().numpy()

        for i, meta in enumerate(metas):
            sid = str(meta["subject"])
            subj_probs[sid].append(float(probs_np[i]))
            subj_true[sid] = int(y_np[i])

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") if len(y_true) else 0.0

    subj_ids = sorted(subj_probs.keys())
    y_true_subj, y_pred_subj = [], []

    for sid in subj_ids:
        p = float(np.mean(subj_probs[sid]))
        y_true_subj.append(int(subj_true[sid]))
        y_pred_subj.append(int(p >= 0.5))

    subj_metrics = build_subject_metrics(y_true_subj, y_pred_subj)

    return {
        "window_loss": float(avg_loss),
        "window_acc": float(acc),
        "window_bal_acc": float(bal_acc),
        "window_macro_f1": float(macro_f1),
        "y_true": y_true,
        "y_pred": y_pred,
        "subject_metrics": subj_metrics,
        "subject_probs": {k: float(np.mean(v)) for k, v in subj_probs.items()},
        "subject_true": subj_true,
    }


# ============================================================
# 8) Training
# ============================================================
def train_base(
    model: ImprovedEEGNet,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epochs: int,
    lr: float,
    patience: int,
    logger: logging.Logger,
    use_amp: bool,
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_metric = -1.0
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, run_correct, total = 0.0, 0, 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast(device, use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits, dim=1)
            run_loss += float(loss.item()) * yb.size(0)
            run_correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

        train_loss = run_loss / max(total, 1)
        train_acc = run_correct / max(total, 1)

        ev = evaluate_base(model, eval_loader, device, criterion, use_amp)
        val_metric = ev["subject_metrics"]["balanced_acc"]
        scheduler.step(val_metric)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[BASE] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_win_loss={ev['window_loss']:.4f} val_win_balacc={ev['window_bal_acc']:.4f} "
            f"val_subj_balacc={ev['subject_metrics']['balanced_acc']:.4f} "
            f"val_subj_macroF1={ev['subject_metrics']['macro_f1']:.4f} | lr={lr_now:.2e}"
        )

        if val_metric > best_metric + 1e-6:
            best_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("[BASE] Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"[BASE] Restored best model (best_subject_balacc={best_metric:.4f})")

    return model, float(best_metric)


def train_ecn(
    base_model: ImprovedEEGNet,
    ecn: FeatureCorrectionNetwork,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epochs: int,
    lr: float,
    patience: int,
    logger: logging.Logger,
    use_amp: bool,
    reg_lambda: float,
):
    base_model = base_model.to(device)
    ecn = ecn.to(device)

    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    optimizer = optim.AdamW(ecn.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_metric = -1.0
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        ecn.train()
        run_loss, run_correct, total = 0.0, 0, 0
        delta_mag = 0.0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                with amp_autocast(device, use_amp):
                    base_emb = base_model.encode(xb)
                    base_logits = base_model.classify_from_embedding(base_emb)

            with amp_autocast(device, use_amp):
                corr_emb, delta = ecn(xb, base_emb, base_logits)
                corr_logits = base_model.classify_from_embedding(corr_emb)
                loss = criterion(corr_logits, yb) + reg_lambda * (delta ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(corr_logits, dim=1)
            run_loss += float(loss.item()) * yb.size(0)
            run_correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
            delta_mag += float(delta.detach().abs().mean().item())

        train_loss = run_loss / max(total, 1)
        train_acc = run_correct / max(total, 1)
        delta_mag = delta_mag / max(1, len(train_loader))

        ev = evaluate_ecn(base_model, ecn, eval_loader, device, criterion, use_amp, reg_lambda)
        val_metric = ev["subject_metrics"]["balanced_acc"]
        scheduler.step(val_metric)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[ECN ] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} delta_mag={delta_mag:.4f} | "
            f"val_win_loss={ev['window_loss']:.4f} val_win_balacc={ev['window_bal_acc']:.4f} "
            f"val_subj_balacc={ev['subject_metrics']['balanced_acc']:.4f} "
            f"val_subj_macroF1={ev['subject_metrics']['macro_f1']:.4f} | lr={lr_now:.2e}"
        )

        if val_metric > best_metric + 1e-6:
            best_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in ecn.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("[ECN ] Early stopping triggered.")
                break

    if best_state is not None:
        ecn.load_state_dict(best_state)
        logger.info(f"[ECN ] Restored best ECN (best_subject_balacc={best_metric:.4f})")

    return ecn, float(best_metric)


# ============================================================
# 9) Channel consistency check
# ============================================================
def check_channel_consistency(split_dir: Path, logger: logging.Logger):
    meta_files = sorted(split_dir.glob("meta_*.csv"))
    if not meta_files:
        logger.warning(f"No meta files for channel consistency check in {split_dir}")
        return

    patterns = defaultdict(int)
    for mf in meta_files:
        df = pd.read_csv(mf)
        if "channels" not in df.columns:
            continue
        vals = df["channels"].astype(str).tolist()
        for v in vals[: min(len(vals), 5)]:
            patterns[v] += 1

    if not patterns:
        logger.warning("No channel info found in meta files.")
        return

    logger.info(f"[{split_dir.name}] Unique channel patterns found: {len(patterns)}")
    most_common = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (pat, cnt) in enumerate(most_common, 1):
        n_ch = len(str(pat).split("|"))
        logger.info(f"[{split_dir.name}] Pattern {i}: seen={cnt} n_channels={n_ch} channels={pat}")


# ============================================================
# 10) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fyp_root", type=str, default=str(Path.home() / "FYP"))
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/ds004504_ad_cn_shards_improved")

    # Data
    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--win_sec", type=float, default=20.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max_minutes_per_subject", type=float, default=None)
    parser.add_argument("--shard_target_mb", type=int, default=256)

    parser.add_argument("--apply_bandpass", action="store_true")
    parser.add_argument("--bandpass_low", type=float, default=0.5)
    parser.add_argument("--bandpass_high", type=float, default=45.0)

    parser.add_argument("--norm_mode", type=str, default="recording_standard",
                        choices=["recording_standard", "recording_robust"])

    # Split
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Train base
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)

    # Train ECN
    parser.add_argument("--ecn_epochs", type=int, default=15)
    parser.add_argument("--ecn_lr", type=float, default=5e-4)
    parser.add_argument("--ecn_patience", type=int, default=8)
    parser.add_argument("--ecn_reg_lambda", type=float, default=0.01)
    parser.add_argument("--ecn_scale_init", type=float, default=0.25)

    # Runtime
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=2)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--rebuild_shards", action="store_true")
    parser.add_argument("--tag", type=str, default="ds004504_AD_CN_improved")

    args = parser.parse_args()

    fyp_root = Path(args.fyp_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = (fyp_root / args.out_dir).resolve()

    run_name = f"improved_eegnet_fecn_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(fyp_root / "experiments" / "logs", run_name)

    logger.info("========== RUN START ==========")
    logger.info(f"FYP root     : {fyp_root}")
    logger.info(f"Dataset root : {dataset_root}")
    logger.info(f"Out dir      : {out_dir}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    logger.info(f"Device: {device} | AMP: {use_amp}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    label_map = load_participants_labels(dataset_root)

    set_files = list_set_files(dataset_root)
    subjects_present = sorted({get_subject_id_from_path(p) for p in set_files})

    subjects_ad = [s for s in subjects_present if infer_binary_label(label_map.get(s, "")) == 1]
    subjects_cn = [s for s in subjects_present if infer_binary_label(label_map.get(s, "")) == 0]

    if len(subjects_ad) == 0 or len(subjects_cn) == 0:
        raise RuntimeError(f"No AD/CN subjects found. AD={len(subjects_ad)} CN={len(subjects_cn)}")

    subjects = subjects_ad + subjects_cn
    subj_labels = np.array([infer_binary_label(label_map[s]) for s in subjects], dtype=np.int64)

    logger.info(f"Subjects present in derivatives: {len(subjects_present)}")
    logger.info(f"Binary subjects (AD+CN)        : {len(subjects)} | AD={len(subjects_ad)} CN={len(subjects_cn)}")

    train_subj, eval_subj = train_test_split(
        subjects,
        test_size=args.eval_ratio,
        random_state=args.seed,
        stratify=subj_labels,
    )
    train_subj = sorted(train_subj)
    eval_subj = sorted(eval_subj)
    logger.info(f"Split by subject: train={len(train_subj)} eval={len(eval_subj)}")

    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"

    need_build = args.rebuild_shards or (not train_dir.exists()) or (not eval_dir.exists())
    if not need_build:
        if len(list(train_dir.glob("X_*.npy"))) == 0 or len(list(eval_dir.glob("X_*.npy"))) == 0:
            need_build = True

    if need_build:
        logger.info("========== BUILDING SHARDS ==========")
        ensure_dir(out_dir)

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
            apply_bandpass=args.apply_bandpass,
            bandpass=(args.bandpass_low, args.bandpass_high),
            norm_mode=args.norm_mode,
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
            apply_bandpass=args.apply_bandpass,
            bandpass=(args.bandpass_low, args.bandpass_high),
            norm_mode=args.norm_mode,
        )
    else:
        logger.info("Shards already exist. Skipping rebuild.")

    logger.info("========== CHANNEL CONSISTENCY CHECK ==========")
    check_channel_consistency(train_dir, logger)
    check_channel_consistency(eval_dir, logger)

    logger.info("========== LOADING SHARDS ==========")
    train_ds = ShardedNpyDataset(train_dir, cache_size=args.cache_size, logger=logger)
    eval_ds = ShardedNpyDataset(eval_dir, cache_size=args.cache_size, logger=logger)

    x0, y0, m0 = train_ds[0]
    in_channels = int(x0.shape[0])
    win_len = int(x0.shape[1])
    logger.info(f"Input shape: (C={in_channels}, T={win_len}) | Example y={int(y0)} | Example subject={m0['subject']}")

    # Loss weighting from subject counts by class
    train_subject_labels = {}
    for s in train_subj:
        train_subject_labels[s] = infer_binary_label(label_map[s])

    class_counts = defaultdict(int)
    for s, y in train_subject_labels.items():
        class_counts[int(y)] += 1

    classes = sorted(class_counts.keys())
    counts = np.array([class_counts[c] for c in classes], dtype=np.float32)
    weights = 1.0 / (counts + 1e-12)
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    logger.info(f"Train subject counts by class: {dict(class_counts)}")
    logger.info(f"Class weights              : {weights.tolist()}")

    train_sampler = build_subject_balanced_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_meta,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_meta,
    )

    base_model = ImprovedEEGNet(
        n_channels=in_channels,
        fs=args.target_fs,
        n_classes=2,
        emb_dim=128,
    ).to(device)

    ecn_model = FeatureCorrectionNetwork(
        n_channels=in_channels,
        emb_dim=128,
        n_classes=2,
        scale_init=args.ecn_scale_init,
    ).to(device)

    base_params = sum(p.numel() for p in base_model.parameters())
    ecn_params = sum(p.numel() for p in ecn_model.parameters())
    logger.info(f"Base model: ImprovedEEGNet         | params={base_params:,}")
    logger.info(f"ECN model : FeatureCorrectionNetwork | params={ecn_params:,}")

    if args.use_focal:
        logger.info(f"Loss: FocalLoss(gamma={args.focal_gamma}) with class weights")
        criterion = FocalLoss(gamma=float(args.focal_gamma), weight=class_weights)
    else:
        logger.info("Loss: CrossEntropyLoss with class weights")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -------------------------
    # Train base
    # -------------------------
    logger.info("========== TRAINING BASE ==========")
    t0 = time.time()
    base_model, best_base_subj_balacc = train_base(
        base_model,
        train_loader,
        eval_loader,
        device,
        criterion,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        logger=logger,
        use_amp=use_amp,
    )
    base_train_time = time.time() - t0
    logger.info(f"[BASE] Train time (sec): {base_train_time:.2f}")

    base_eval = evaluate_base(base_model, eval_loader, device, criterion, use_amp)
    logger.info(
        f"[BASE] Final eval | "
        f"window_balacc={base_eval['window_bal_acc']:.4f} "
        f"subject_balacc={base_eval['subject_metrics']['balanced_acc']:.4f} "
        f"subject_macroF1={base_eval['subject_metrics']['macro_f1']:.4f}"
    )

    # -------------------------
    # Train ECN
    # -------------------------
    logger.info("========== TRAINING ECN ==========")
    t1 = time.time()
    ecn_model, best_ecn_subj_balacc = train_ecn(
        base_model,
        ecn_model,
        train_loader,
        eval_loader,
        device,
        criterion,
        epochs=args.ecn_epochs,
        lr=args.ecn_lr,
        patience=args.ecn_patience,
        logger=logger,
        use_amp=use_amp,
        reg_lambda=float(args.ecn_reg_lambda),
    )
    ecn_train_time = time.time() - t1
    logger.info(f"[ECN ] Train time (sec): {ecn_train_time:.2f}")

    ecn_eval = evaluate_ecn(base_model, ecn_model, eval_loader, device, criterion, use_amp, float(args.ecn_reg_lambda))
    logger.info(
        f"[ECN ] Final eval | "
        f"window_balacc={ecn_eval['window_bal_acc']:.4f} "
        f"subject_balacc={ecn_eval['subject_metrics']['balanced_acc']:.4f} "
        f"subject_macroF1={ecn_eval['subject_metrics']['macro_f1']:.4f}"
    )

    # Reports
    base_report_text = classification_report(
        base_eval["y_true"], base_eval["y_pred"], digits=4,
        target_names=["CN(0)", "AD(1)"], zero_division=0
    )
    base_report_dict = classification_report(
        base_eval["y_true"], base_eval["y_pred"], digits=4,
        output_dict=True, target_names=["CN(0)", "AD(1)"], zero_division=0
    )
    base_cm = confusion_matrix(base_eval["y_true"], base_eval["y_pred"])

    ecn_report_text = classification_report(
        ecn_eval["y_true"], ecn_eval["y_pred"], digits=4,
        target_names=["CN(0)", "AD(1)"], zero_division=0
    )
    ecn_report_dict = classification_report(
        ecn_eval["y_true"], ecn_eval["y_pred"], digits=4,
        output_dict=True, target_names=["CN(0)", "AD(1)"], zero_division=0
    )
    ecn_cm = confusion_matrix(ecn_eval["y_true"], ecn_eval["y_pred"])

    logger.info("========== REPORTS ==========")
    logger.info("[BASE] Window classification report:\n" + base_report_text)
    logger.info("[BASE] Window confusion matrix:\n" + str(base_cm))
    logger.info("[BASE] Subject metrics:\n" + json.dumps(base_eval["subject_metrics"], indent=2))

    logger.info("[ECN ] Window classification report:\n" + ecn_report_text)
    logger.info("[ECN ] Window confusion matrix:\n" + str(ecn_cm))
    logger.info("[ECN ] Subject metrics:\n" + json.dumps(ecn_eval["subject_metrics"], indent=2))

    # Save
    exp_dir = fyp_root / "experiments" / run_name
    ensure_dir(exp_dir)
    models_dir = fyp_root / "models"
    ensure_dir(models_dir)

    model_path = models_dir / f"{run_name}.pt"
    torch.save(
        {
            "base_state_dict": base_model.state_dict(),
            "ecn_state_dict": ecn_model.state_dict(),
            "model_name": "ImprovedEEGNet+FeatureCorrectionECN",
            "base_model_name": "ImprovedEEGNet",
            "ecn_model_name": "FeatureCorrectionNetwork",
            "in_channels": in_channels,
            "win_len": win_len,
            "target_fs": args.target_fs,
            "num_classes": 2,
            "emb_dim": 128,
            "seed": args.seed,
            "dataset_root": str(dataset_root),
            "out_dir": str(out_dir),
            "tag": args.tag,
            "best_base_subject_balacc": float(best_base_subj_balacc),
            "best_ecn_subject_balacc": float(best_ecn_subj_balacc),
            "config": vars(args),
        },
        model_path,
    )

    metrics = {
        "run_name": run_name,
        "tag": args.tag,
        "train_windows": int(len(train_ds)),
        "eval_windows": int(len(eval_ds)),
        "train_subjects": len(train_subj),
        "eval_subjects": len(eval_subj),
        "base": {
            "train_time_sec": float(base_train_time),
            "window_loss": float(base_eval["window_loss"]),
            "window_acc": float(base_eval["window_acc"]),
            "window_bal_acc": float(base_eval["window_bal_acc"]),
            "window_macro_f1": float(base_eval["window_macro_f1"]),
            "subject_metrics": base_eval["subject_metrics"],
            "classification_report_window": base_report_dict,
            "confusion_matrix_window": base_cm.tolist(),
            "n_params": int(base_params),
        },
        "ecn": {
            "train_time_sec": float(ecn_train_time),
            "window_loss": float(ecn_eval["window_loss"]),
            "window_acc": float(ecn_eval["window_acc"]),
            "window_bal_acc": float(ecn_eval["window_bal_acc"]),
            "window_macro_f1": float(ecn_eval["window_macro_f1"]),
            "subject_metrics": ecn_eval["subject_metrics"],
            "classification_report_window": ecn_report_dict,
            "confusion_matrix_window": ecn_cm.tolist(),
            "n_params": int(ecn_params),
            "reg_lambda": float(args.ecn_reg_lambda),
        },
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.save(exp_dir / "base_confusion_matrix_window.npy", base_cm)
    np.save(exp_dir / "ecn_confusion_matrix_window.npy", ecn_cm)

    with open(exp_dir / "base_subject_metrics.json", "w", encoding="utf-8") as f:
        json.dump(base_eval["subject_metrics"], f, indent=2)

    with open(exp_dir / "ecn_subject_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ecn_eval["subject_metrics"], f, indent=2)

    logger.info(f"Saved model   : {model_path}")
    logger.info(f"Saved metrics : {exp_dir / 'metrics.json'}")
    logger.info("========== RUN END ==========")


if __name__ == "__main__":
    main()