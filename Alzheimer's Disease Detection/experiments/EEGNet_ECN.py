#!/usr/bin/env python3
"""
Binary AD vs CN classifier on OpenNeuro ds004504 (v1.0.2) using derivatives EEGLAB .set files.

Pipeline:
- Read derivatives/sub-*/eeg/*.set (preprocessed / denoised)
- Subject-level split (no leakage)
- Windowing into (C,T) samples + z-score
- Save shards (X_*.npy, y_*.npy, meta_*.csv)
- Train LightweightEEGNet (PyTorch) + evaluate
- Train ECN (ErrorCorrectionNetwork) on top of frozen base logits + evaluate

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
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
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


def zscore_window(x_ct: np.ndarray) -> np.ndarray:
    """
    x_ct: (C, T) -> z-score per channel over time
    """
    mean = x_ct.mean(axis=1, keepdims=True)
    std = x_ct.std(axis=1, keepdims=True) + 1e-6
    return (x_ct - mean) / std


def load_participants_labels(dataset_root: Path) -> Dict[str, str]:
    """
    Map: participant_id ('sub-001') -> group label (e.g., 'AD','CN','FTD')
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


# ============================================================
# 3) Build shards (train/eval)
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
        raise FileNotFoundError(f"No .set files for split={split} under {dataset_root}/derivatives")

    logger.info(f"[{split}] Subjects: {len(subjects)} | .set files: {len(subject_set_files)}")

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

        logger.info(f"[{split}] Wrote shard {shard_id}: X={X.shape} | y={y.shape} | meta={len(meta)}")
        shard_id += 1
        X_buf, y_buf, meta_rows = [], [], []

    # Diagnostics
    read_fail = 0
    no_eeg = 0
    too_short = 0
    wrote_windows = 0

    for set_path in subject_set_files:
        sid = get_subject_id_from_path(set_path)
        grp = label_map.get(sid, "")
        grp_u = grp.upper()

        # Binary mapping (handles both full labels like 'AD'/'CN' and single-letter 'A'/'C')
        if grp_u in ("AD", "A"):
            y_label = 1
        elif grp_u in ("CN", "C"):
            y_label = 0
        else:
            continue

        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
        except Exception as e:
            logger.warning(f"[{split}] Failed reading {set_path.name} ({sid}): {e}")
            read_fail += 1
            continue

        # Keep EEG channels
        raw.pick_types(eeg=True)
        if len(raw.ch_names) == 0:
            raw.set_channel_types({ch: "eeg" for ch in raw.info["ch_names"]})
            raw.pick_types(eeg=True)
            if len(raw.ch_names) == 0:
                logger.warning(f"[{split}] No EEG channels: {set_path}")
                no_eeg += 1
                continue

        dur_sec_orig = raw.n_times / float(raw.info["sfreq"])
        logger.info(
            f"[{split}] Loaded {set_path.name} | sid={sid} grp={grp_u} "
            f"| sfreq={raw.info['sfreq']:.1f} n_times={raw.n_times} dur_sec={dur_sec_orig:.2f}"
        )

        # Resample to target_fs
        if int(raw.info["sfreq"]) != int(target_fs):
            raw.resample(target_fs, npad="auto", verbose="ERROR")

        # OPTIONAL extra bandpass (derivatives already cleaned; default False)
        if apply_bandpass:
            l_freq, h_freq = bandpass
            raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose="ERROR")

        data = raw.get_data()  # (C, T_total)
        C, T_total = data.shape

        # Optional cap per subject
        if max_minutes_per_subject is not None:
            max_sec = max_minutes_per_subject * 60.0
            T_cap = int(max_sec * target_fs)
            if T_total > T_cap:
                data = data[:, :T_cap]
                T_total = data.shape[1]

        if T_total < win_len:
            logger.warning(
                f"[{split}] Too short for one window: {set_path.name} "
                f"| samples={T_total} < win_len={win_len}"
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
            y_buf.append(y_label)
            meta_rows.append(
                {
                    "subject": sid,
                    "file": set_path.name,
                    "group": grp_u,
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
        logger.info(f"[{split}] {sid} ({grp_u}) | {set_path.name} | windows_written={local_written}")

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
            if not yp.exists():
                raise FileNotFoundError(f"Missing {yp.name} for {xp.name}")

            x_mmap = to_nct(np.load(xp, mmap_mode="r"))
            y_mmap = np.load(yp, mmap_mode="r")

            n = int(x_mmap.shape[0])
            if int(y_mmap.shape[0]) != n:
                raise ValueError(f"Row mismatch: {xp.name} n={n} vs {yp.name} n={int(y_mmap.shape[0])}")

            self.shards.append({"x_path": xp, "y_path": yp, "n": n})
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


# ============================================================
# 5) Models: LightweightEEGNet + ECN
# ============================================================
class DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 7, s: int = 1, p: int = 3, drop: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class LightweightEEGNet(nn.Module):
    """
    Input:  (B, C, T)
    Output: (B, num_classes)
    """
    def __init__(self, n_channels: int, n_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        self.block1 = DSConvBlock(32, 48, k=7, s=2, p=3, drop=0.10)
        self.block2 = DSConvBlock(48, 64, k=5, s=2, p=2, drop=0.12)
        self.block3 = DSConvBlock(64, 80, k=3, s=2, p=1, drop=0.15)

        self.attn = nn.Conv1d(80, 1, kernel_size=1)

        self.head = nn.Sequential(
            nn.Linear(80, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        w = torch.softmax(self.attn(x).squeeze(1), dim=-1)  # (B, T')
        emb = torch.sum(x * w.unsqueeze(1), dim=-1)         # (B, 80)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encode(x)
        return self.head(emb)


class ErrorCorrectionNetwork(nn.Module):
    """
    ECN learns an additive residual to correct base logits.

    Inputs:
      - raw EEG (B,C,T)
      - base logits (B,2)

    Output:
      - corrected logits (B,2)
      - residual (B,2)
    """
    def __init__(self, n_channels: int, n_classes: int = 2, scale_init: float = 0.35):
        super().__init__()
        self.sig = nn.Sequential(
            nn.Conv1d(n_channels, 24, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(24, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 32 signal + 2 logits + 4 meta = 38
        self.mlp = nn.Sequential(
            nn.Linear(38, 48),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(48, n_classes),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    @staticmethod
    def _meta_from_logits(base_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(base_logits, dim=1)
        max_p, _ = probs.max(dim=1, keepdim=True)
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        margin = (sorted_probs[:, :1] - sorted_probs[:, 1:2])
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1, keepdim=True)
        pos_prob = probs[:, 1:2]
        return torch.cat([max_p, margin, entropy, pos_prob], dim=1)

    def forward(self, x: torch.Tensor, base_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sig_feat = self.sig(x).squeeze(-1)  # (B,32)
        meta = self._meta_from_logits(base_logits)  # (B,4)
        feat = torch.cat([sig_feat, base_logits, meta], dim=1)  # (B,38)
        residual = torch.tanh(self.mlp(feat)) * self.scale
        corrected = base_logits + residual
        return corrected, residual


# ============================================================
# 6) Loss + AMP helpers
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)            # (B,)
        pt = torch.exp(-ce)                      # (B,)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def amp_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    # nullcontext without importing contextlib to keep file minimal
    class _Null:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return _Null()


# ============================================================
# 7) Evaluation
# ============================================================
@torch.no_grad()
def evaluate_base(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, use_amp: bool):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with amp_autocast(device, use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * yb.size(0)
        total_correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
    return avg_loss, acc, bal_acc, macro_f1, y_true, y_pred


@torch.no_grad()
def evaluate_ecn(
    base_model: nn.Module,
    ecn: nn.Module,
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

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with amp_autocast(device, use_amp):
            base_logits = base_model(xb)
            corr_logits, residual = ecn(xb, base_logits)
            loss = criterion(corr_logits, yb) + reg_lambda * (residual ** 2).mean()

        preds = torch.argmax(corr_logits, dim=1)
        total_loss += float(loss.item()) * yb.size(0)
        total_correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
    return avg_loss, acc, bal_acc, macro_f1, y_true, y_pred


# ============================================================
# 8) Training
# ============================================================
def train_base(
    model: nn.Module,
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val_balacc = -1.0
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, run_correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
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

        val_loss, val_acc, val_balacc, val_macro_f1, _, _ = evaluate_base(
            model, eval_loader, device, criterion, use_amp=use_amp
        )
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[BASE] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_balacc={val_balacc:.4f} val_macroF1={val_macro_f1:.4f} | lr={lr_now:.2e}"
        )

        if val_balacc > best_val_balacc + 1e-6:
            best_val_balacc = val_balacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("[BASE] Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"[BASE] Restored best model (best_val_balacc={best_val_balacc:.4f})")

    return model, float(best_val_balacc)


def train_ecn(
    base_model: nn.Module,
    ecn: nn.Module,
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

    # freeze base model
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    optimizer = optim.AdamW(ecn.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val_balacc = -1.0
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        ecn.train()
        run_loss, run_correct, total = 0.0, 0, 0
        corr_mag = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                with amp_autocast(device, use_amp):
                    base_logits = base_model(xb)

            with amp_autocast(device, use_amp):
                corr_logits, residual = ecn(xb, base_logits)
                loss = criterion(corr_logits, yb) + reg_lambda * (residual ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(corr_logits, dim=1)
            run_loss += float(loss.item()) * yb.size(0)
            run_correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
            corr_mag += float(residual.detach().abs().mean().item())

        train_loss = run_loss / max(total, 1)
        train_acc = run_correct / max(total, 1)
        corr_mag = corr_mag / max(1, len(train_loader))

        val_loss, val_acc, val_balacc, val_macro_f1, _, _ = evaluate_ecn(
            base_model, ecn, eval_loader, device, criterion, use_amp=use_amp, reg_lambda=reg_lambda
        )
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[ECN ] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} corr_mag={corr_mag:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_balacc={val_balacc:.4f} val_macroF1={val_macro_f1:.4f} | lr={lr_now:.2e}"
        )

        if val_balacc > best_val_balacc + 1e-6:
            best_val_balacc = val_balacc
            best_state = {k: v.detach().cpu().clone() for k, v in ecn.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("[ECN ] Early stopping triggered.")
                break

    if best_state is not None:
        ecn.load_state_dict(best_state)
        logger.info(f"[ECN ] Restored best ECN (best_val_balacc={best_val_balacc:.4f})")

    return ecn, float(best_val_balacc)


# ============================================================
# 9) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fyp_root", type=str, default=str(Path.home() / "FYP"))
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/ds004504_ad_cn_shards")

    # Dataset
    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--win_sec", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max_minutes_per_subject", type=float, default=None)
    parser.add_argument("--shard_target_mb", type=int, default=256)

    # Optional bandpass (OFF by default since derivatives already cleaned)
    parser.add_argument("--apply_bandpass", action="store_true")
    parser.add_argument("--bandpass_low", type=float, default=0.5)
    parser.add_argument("--bandpass_high", type=float, default=45.0)

    # Split
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Train base
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)

    # Train ECN
    parser.add_argument("--ecn_epochs", type=int, default=12)
    parser.add_argument("--ecn_lr", type=float, default=5e-4)
    parser.add_argument("--ecn_patience", type=int, default=6)
    parser.add_argument("--ecn_reg_lambda", type=float, default=0.01)
    parser.add_argument("--ecn_scale_init", type=float, default=0.35)

    # Runtime
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=2)
    parser.add_argument("--use_focal", action="store_true", help="Use focal loss instead of weighted CE")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP even if CUDA is available")

    # Control
    parser.add_argument("--rebuild_shards", action="store_true")
    parser.add_argument("--tag", type=str, default="ds004504_AD_CN_derivatives")

    args = parser.parse_args()

    fyp_root = Path(args.fyp_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = (fyp_root / args.out_dir).resolve()

    run_name = f"lweegnet_ecn_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    subjects_ad = [s for s in subjects_present if label_map.get(s, "").upper() in ("AD", "A")]
    subjects_cn = [s for s in subjects_present if label_map.get(s, "").upper() in ("CN", "C")]

    if len(subjects_ad) == 0 or len(subjects_cn) == 0:
        raise RuntimeError(f"No AD/CN subjects found. AD={len(subjects_ad)} CN={len(subjects_cn)}")

    subjects = subjects_ad + subjects_cn
    subj_labels = np.array([1 if label_map[s].upper() in ("AD", "A") else 0 for s in subjects], dtype=np.int64)

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
        )
    else:
        logger.info("Shards already exist. Skipping rebuild.")

    logger.info("========== LOADING SHARDS ==========")
    train_ds = ShardedNpyDataset(train_dir, cache_size=args.cache_size, logger=logger)
    eval_ds = ShardedNpyDataset(eval_dir, cache_size=args.cache_size, logger=logger)

    x0, y0 = train_ds[0]
    in_channels = int(x0.shape[0])
    win_len = int(x0.shape[1])
    logger.info(f"Input shape: (C={in_channels}, T={win_len}) | Example y={int(y0)}")

    # Class weights (same style as your original script)
    train_y = []
    for yp in sorted(train_dir.glob("y_*.npy")):
        y = np.load(yp, mmap_mode="r")
        train_y.append(np.asarray(y, dtype=np.int64))
    train_y = np.concatenate(train_y)
    classes, counts = np.unique(train_y, return_counts=True)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-12)
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    logger.info(f"Train label counts: {dict(zip([int(c) for c in classes], [int(n) for n in counts]))}")
    logger.info(f"Class weights     : {weights.tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Init models
    base_model = LightweightEEGNet(n_channels=in_channels, n_classes=2).to(device)
    ecn_model = ErrorCorrectionNetwork(
        n_channels=in_channels, n_classes=2, scale_init=args.ecn_scale_init
    ).to(device)

    base_params = sum(p.numel() for p in base_model.parameters())
    ecn_params = sum(p.numel() for p in ecn_model.parameters())
    logger.info(f"Base model: LightweightEEGNet | params={base_params:,}")
    logger.info(f"ECN model : ErrorCorrectionNetwork | params={ecn_params:,}")

    # Loss
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
    base_model, best_base_balacc = train_base(
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

    # Base final eval (best checkpoint already restored)
    base_val_loss, base_val_acc, base_val_balacc, base_val_macro_f1, base_y_true, base_y_pred = evaluate_base(
        base_model, eval_loader, device, criterion, use_amp=use_amp
    )
    logger.info(
        f"[BASE] Final eval | loss={base_val_loss:.4f} acc={base_val_acc:.4f} "
        f"balacc={base_val_balacc:.4f} macroF1={base_val_macro_f1:.4f}"
    )

    # -------------------------
    # Train ECN (freeze base)
    # -------------------------
    logger.info("========== TRAINING ECN ==========")
    t1 = time.time()
    ecn_model, best_ecn_balacc = train_ecn(
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

    # ECN final eval (best checkpoint already restored)
    ecn_val_loss, ecn_val_acc, ecn_val_balacc, ecn_val_macro_f1, ecn_y_true, ecn_y_pred = evaluate_ecn(
        base_model,
        ecn_model,
        eval_loader,
        device,
        criterion,
        use_amp=use_amp,
        reg_lambda=float(args.ecn_reg_lambda),
    )
    logger.info(
        f"[ECN ] Final eval | loss={ecn_val_loss:.4f} acc={ecn_val_acc:.4f} "
        f"balacc={ecn_val_balacc:.4f} macroF1={ecn_val_macro_f1:.4f}"
    )

    # Reports
    base_report_text = classification_report(base_y_true, base_y_pred, digits=4, target_names=["CN(0)", "AD(1)"])
    base_report_dict = classification_report(base_y_true, base_y_pred, digits=4, output_dict=True, target_names=["CN(0)", "AD(1)"])
    base_cm = confusion_matrix(base_y_true, base_y_pred)

    ecn_report_text = classification_report(ecn_y_true, ecn_y_pred, digits=4, target_names=["CN(0)", "AD(1)"])
    ecn_report_dict = classification_report(ecn_y_true, ecn_y_pred, digits=4, output_dict=True, target_names=["CN(0)", "AD(1)"])
    ecn_cm = confusion_matrix(ecn_y_true, ecn_y_pred)

    logger.info("========== REPORTS ==========")
    logger.info("[BASE] Classification report:\n" + base_report_text)
    logger.info("[BASE] Confusion matrix:\n" + str(base_cm))
    logger.info("[ECN ] Classification report:\n" + ecn_report_text)
    logger.info("[ECN ] Confusion matrix:\n" + str(ecn_cm))

    # Save artifacts
    exp_dir = fyp_root / "experiments" / run_name
    ensure_dir(exp_dir)
    models_dir = fyp_root / "models"
    ensure_dir(models_dir)

    model_path = models_dir / f"{run_name}.pt"
    torch.save(
        {
            "base_state_dict": base_model.state_dict(),
            "ecn_state_dict": ecn_model.state_dict(),
            "model_name": "LightweightEEGNet+ECN",
            "base_model_name": "LightweightEEGNet",
            "ecn_model_name": "ErrorCorrectionNetwork",
            "in_channels": in_channels,
            "win_len": win_len,
            "num_classes": 2,
            "seed": args.seed,
            "dataset_root": str(dataset_root),
            "out_dir": str(out_dir),
            "tag": args.tag,
            "best_base_val_balanced_acc": float(best_base_balacc),
            "best_ecn_val_balanced_acc": float(best_ecn_balacc),
            "config": vars(args),
        },
        model_path,
    )

    metrics = {
        "run_name": run_name,
        "tag": args.tag,
        "train_windows": int(len(train_ds)),
        "eval_windows": int(len(eval_ds)),
        "base": {
            "train_time_sec": float(base_train_time),
            "final_val_loss": float(base_val_loss),
            "final_val_acc": float(base_val_acc),
            "final_val_balanced_acc": float(base_val_balacc),
            "final_val_macro_f1": float(base_val_macro_f1),
            "best_val_balanced_acc": float(best_base_balacc),
            "classification_report": base_report_dict,
            "confusion_matrix": base_cm.tolist(),
            "n_params": int(base_params),
        },
        "ecn": {
            "train_time_sec": float(ecn_train_time),
            "final_val_loss": float(ecn_val_loss),
            "final_val_acc": float(ecn_val_acc),
            "final_val_balanced_acc": float(ecn_val_balacc),
            "final_val_macro_f1": float(ecn_val_macro_f1),
            "best_val_balanced_acc": float(best_ecn_balacc),
            "classification_report": ecn_report_dict,
            "confusion_matrix": ecn_cm.tolist(),
            "n_params": int(ecn_params),
            "reg_lambda": float(args.ecn_reg_lambda),
        },
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.save(exp_dir / "base_confusion_matrix.npy", base_cm)
    np.save(exp_dir / "ecn_confusion_matrix.npy", ecn_cm)

    logger.info(f"Saved model   : {model_path}")
    logger.info(f"Saved metrics : {exp_dir / 'metrics.json'}")
    logger.info("========== RUN END ==========")


if __name__ == "__main__":
    main()