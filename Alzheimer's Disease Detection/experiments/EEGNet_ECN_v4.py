#!/usr/bin/env python3
"""
EEGNet_ECN_v4.py

5-fold subject-level stratified cross-validation on EXISTING shards only.
Base model + ECN.

Expected existing shard dirs:
  <fyp_root>/data/ds004504_ad_cn_shards_improved/train
  <fyp_root>/data/ds004504_ad_cn_shards_improved/eval

Outputs use the script name exactly:
  logs   : <fyp_root>/experiments/logs/EEGNet_ECN_v4.log
  exp dir: <fyp_root>/experiments/EEGNet_ECN_v4/
  model  : <fyp_root>/models/EEGNet_ECN_v4.pt
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)


# ============================================================
# Logging / utils
# ============================================================
def get_script_stem() -> str:
    return Path(__file__).stem


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


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_nct(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got {X.shape}")
    return X


# ============================================================
# Dataset over existing shard dirs
# ============================================================
class CombinedShardedDataset(Dataset):
    def __init__(self, split_dirs: List[Path], cache_size: int, logger: logging.Logger):
        self.split_dirs = split_dirs
        self.cache_size = cache_size
        self.logger = logger
        self._x_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

        self.shards = []
        self.shard_offsets = []
        total = 0

        for split_dir in split_dirs:
            x_paths = sorted(split_dir.glob("X_*.npy"))
            if not x_paths:
                raise FileNotFoundError(f"No X_*.npy in {split_dir}")

            for xp in x_paths:
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
                    raise ValueError(f"Row mismatch in shard {xp}")

                self.shards.append(
                    {
                        "split_dir": split_dir,
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

        meta_row["global_index"] = int(idx)
        meta_row["source_split"] = shard["split_dir"].name
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), meta_row


class IndexDataset(Dataset):
    def __init__(self, base_ds: Dataset, indices: List[int]):
        self.base_ds = base_ds
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base_ds[int(self.indices[i])]


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    xb = torch.stack(xs, dim=0)
    yb = torch.stack(ys, dim=0)
    return xb, yb, list(metas)


def build_subject_balanced_sampler_from_indices(
    all_subjects: List[str],
    all_labels: List[int],
    indices: List[int],
) -> WeightedRandomSampler:
    subjects = [all_subjects[i] for i in indices]
    labels = [int(all_labels[i]) for i in indices]

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
# Models
# ============================================================
class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, fs: int):
        super().__init__()
        self.fs = fs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return torch.cat(out, dim=1)


class ImprovedEEGNet(nn.Module):
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

        w = torch.softmax(self.attn(z).squeeze(1), dim=-1)
        raw_emb = torch.sum(z * w.unsqueeze(1), dim=-1)

        spec = self.spectral(x)
        spec_emb = self.spec_proj(spec)

        emb = self.fuse(torch.cat([raw_emb, spec_emb], dim=1))
        return emb

    def classify_from_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encode(x)
        return self.classify_from_embedding(emb)


class FeatureCorrectionNetwork(nn.Module):
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

    def forward(self, x: torch.Tensor, base_emb: torch.Tensor, base_logits: torch.Tensor):
        sig_feat = self.sig(x).squeeze(-1)
        meta = self._meta_from_logits(base_logits)
        feat = torch.cat([sig_feat, base_emb, base_logits, meta], dim=1)
        delta = torch.tanh(self.mlp(feat)) * self.scale
        corrected_emb = base_emb + delta
        return corrected_emb, delta


# ============================================================
# Loss / AMP
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
# Evaluation
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
    }


# ============================================================
# Training
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

        ev = evaluate_ecn_identity(model, eval_loader, device, criterion, use_amp)
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


@torch.no_grad()
def evaluate_ecn_identity(
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
    }


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
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fyp_root", type=str, default=str(Path.home() / "FYP"))
    parser.add_argument("--shards_dir", type=str, default="data/ds004504_ad_cn_shards_improved")

    parser.add_argument("--target_fs", type=int, default=250)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--ecn_epochs", type=int, default=15)
    parser.add_argument("--ecn_lr", type=float, default=5e-4)
    parser.add_argument("--ecn_patience", type=int, default=8)
    parser.add_argument("--ecn_reg_lambda", type=float, default=0.01)
    parser.add_argument("--ecn_scale_init", type=float, default=0.25)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=2)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    run_name = get_script_stem()
    fyp_root = Path(args.fyp_root).resolve()
    shards_root = (fyp_root / args.shards_dir).resolve()
    train_dir = shards_root / "train"
    eval_dir = shards_root / "eval"

    logger = setup_logger(fyp_root / "experiments" / "logs", run_name)
    logger.info("========== RUN START ==========")
    logger.info(f"Run name    : {run_name}")
    logger.info(f"FYP root    : {fyp_root}")
    logger.info(f"Shards root : {shards_root}")

    if not train_dir.exists() or not eval_dir.exists():
        raise FileNotFoundError(f"Expected shard dirs not found:\n{train_dir}\n{eval_dir}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    logger.info(f"Device: {device} | AMP: {use_amp}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("========== LOADING EXISTING SHARDS ==========")
    full_ds = CombinedShardedDataset(
        split_dirs=[train_dir, eval_dir],
        cache_size=args.cache_size,
        logger=logger,
    )

    x0, y0, m0 = full_ds[0]
    in_channels = int(x0.shape[0])
    win_len = int(x0.shape[1])
    logger.info(f"Input shape: (C={in_channels}, T={win_len}) | Example y={int(y0)} | Example subject={m0['subject']}")

    subject_to_label = {}
    subject_to_indices = defaultdict(list)

    for idx, (sid, lab) in enumerate(zip(full_ds.subjects, full_ds.labels)):
        sid = str(sid)
        lab = int(lab)
        if sid in subject_to_label and subject_to_label[sid] != lab:
            raise ValueError(f"Inconsistent label for subject {sid}")
        subject_to_label[sid] = lab
        subject_to_indices[sid].append(idx)

    subjects = np.array(sorted(subject_to_label.keys()))
    subj_labels = np.array([subject_to_label[s] for s in subjects], dtype=np.int64)

    logger.info(f"Total windows : {len(full_ds)}")
    logger.info(f"Total subjects: {len(subjects)} | AD={int((subj_labels==1).sum())} CN={int((subj_labels==0).sum())}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    exp_dir = fyp_root / "experiments" / run_name
    ensure_dir(exp_dir)
    models_dir = fyp_root / "models"
    ensure_dir(models_dir)

    fold_results = []
    best_fold_state = None
    best_fold_metric = -1.0
    best_fold_id = None

    for fold_id, (tr_sub_idx, va_sub_idx) in enumerate(skf.split(subjects, subj_labels), start=1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"FOLD {fold_id}/{args.n_splits}")
        logger.info("=" * 70)

        train_subjects = subjects[tr_sub_idx].tolist()
        val_subjects = subjects[va_sub_idx].tolist()

        train_indices = []
        val_indices = []

        for s in train_subjects:
            train_indices.extend(subject_to_indices[s])
        for s in val_subjects:
            val_indices.extend(subject_to_indices[s])

        logger.info(
            f"Fold {fold_id} | train_subjects={len(train_subjects)} val_subjects={len(val_subjects)} "
            f"| train_windows={len(train_indices)} val_windows={len(val_indices)}"
        )

        train_subject_labels = [subject_to_label[s] for s in train_subjects]
        class_counts = defaultdict(int)
        for y in train_subject_labels:
            class_counts[int(y)] += 1

        counts = np.array([class_counts[c] for c in sorted(class_counts.keys())], dtype=np.float32)
        weights = 1.0 / (counts + 1e-12)
        weights = weights / weights.sum() * len(counts)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        logger.info(f"Fold {fold_id} | Train subject counts by class: {dict(class_counts)}")
        logger.info(f"Fold {fold_id} | Class weights: {weights.tolist()}")

        train_ds = IndexDataset(full_ds, train_indices)
        val_ds = IndexDataset(full_ds, val_indices)

        train_sampler = build_subject_balanced_sampler_from_indices(full_ds.subjects, full_ds.labels, train_indices)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_with_meta,
        )
        val_loader = DataLoader(
            val_ds,
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
        logger.info(f"Fold {fold_id} | Base params={base_params:,} | ECN params={ecn_params:,}")

        if args.use_focal:
            logger.info(f"Fold {fold_id} | Loss: FocalLoss(gamma={args.focal_gamma}) with class weights")
            criterion = FocalLoss(gamma=float(args.focal_gamma), weight=class_weights)
        else:
            logger.info(f"Fold {fold_id} | Loss: CrossEntropyLoss with class weights")
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Train base
        t0 = time.time()
        base_model, best_base_subj_balacc = train_base(
            model=base_model,
            train_loader=train_loader,
            eval_loader=val_loader,
            device=device,
            criterion=criterion,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            logger=logger,
            use_amp=use_amp,
        )
        base_train_time = time.time() - t0

        # Train ECN
        t1 = time.time()
        ecn_model, best_ecn_subj_balacc = train_ecn(
            base_model=base_model,
            ecn=ecn_model,
            train_loader=train_loader,
            eval_loader=val_loader,
            device=device,
            criterion=criterion,
            epochs=args.ecn_epochs,
            lr=args.ecn_lr,
            patience=args.ecn_patience,
            logger=logger,
            use_amp=use_amp,
            reg_lambda=float(args.ecn_reg_lambda),
        )
        ecn_train_time = time.time() - t1

        final_eval = evaluate_ecn(
            base_model=base_model,
            ecn=ecn_model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
            reg_lambda=float(args.ecn_reg_lambda),
        )

        fold_metrics = {
            "fold": fold_id,
            "train_subjects": len(train_subjects),
            "val_subjects": len(val_subjects),
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "base_train_time_sec": float(base_train_time),
            "ecn_train_time_sec": float(ecn_train_time),
            "best_base_subject_balacc_during_training": float(best_base_subj_balacc),
            "best_ecn_subject_balacc_during_training": float(best_ecn_subj_balacc),
            "window_loss": float(final_eval["window_loss"]),
            "window_acc": float(final_eval["window_acc"]),
            "window_bal_acc": float(final_eval["window_bal_acc"]),
            "window_macro_f1": float(final_eval["window_macro_f1"]),
            "subject_metrics": final_eval["subject_metrics"],
            "confusion_matrix_window": confusion_matrix(final_eval["y_true"], final_eval["y_pred"]).tolist(),
            "classification_report_window": classification_report(
                final_eval["y_true"],
                final_eval["y_pred"],
                digits=4,
                output_dict=True,
                target_names=["CN(0)", "AD(1)"],
                zero_division=0,
            ),
            "base_n_params": int(base_params),
            "ecn_n_params": int(ecn_params),
        }

        fold_results.append(fold_metrics)

        logger.info(
            f"Fold {fold_id} FINAL | "
            f"window_balacc={fold_metrics['window_bal_acc']:.4f} "
            f"window_macroF1={fold_metrics['window_macro_f1']:.4f} "
            f"subject_balacc={fold_metrics['subject_metrics']['balanced_acc']:.4f} "
            f"subject_macroF1={fold_metrics['subject_metrics']['macro_f1']:.4f}"
        )

        fold_model_path = models_dir / f"{run_name}_fold{fold_id}.pt"
        torch.save(
            {
                "base_state_dict": base_model.state_dict(),
                "ecn_state_dict": ecn_model.state_dict(),
                "model_name": "ImprovedEEGNet+FeatureCorrectionNetwork",
                "fold": fold_id,
                "in_channels": in_channels,
                "win_len": win_len,
                "target_fs": args.target_fs,
                "num_classes": 2,
                "emb_dim": 128,
                "base_n_params": int(base_params),
                "ecn_n_params": int(ecn_params),
                "config": vars(args),
                "metrics": fold_metrics,
            },
            fold_model_path,
        )
        logger.info(f"Fold {fold_id} model saved: {fold_model_path}")

        fold_subj_balacc = float(fold_metrics["subject_metrics"]["balanced_acc"])
        if fold_subj_balacc > best_fold_metric:
            best_fold_metric = fold_subj_balacc
            best_fold_id = fold_id
            best_fold_state = {
                "base_state_dict": {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()},
                "ecn_state_dict": {k: v.detach().cpu().clone() for k, v in ecn_model.state_dict().items()},
            }

    summary = {
        "run_name": run_name,
        "n_splits": args.n_splits,
        "n_subjects_total": int(len(subjects)),
        "n_windows_total": int(len(full_ds)),
        "model_name": "ImprovedEEGNet+FeatureCorrectionNetwork",
        "folds": fold_results,
    }

    def collect(key1, key2=None):
        vals = []
        for fr in fold_results:
            if key2 is None:
                vals.append(float(fr[key1]))
            else:
                vals.append(float(fr[key1][key2]))
        return vals

    win_bal = collect("window_bal_acc")
    win_f1 = collect("window_macro_f1")
    subj_bal = collect("subject_metrics", "balanced_acc")
    subj_f1 = collect("subject_metrics", "macro_f1")

    summary["aggregate"] = {
        "window_bal_acc_mean": float(np.mean(win_bal)),
        "window_bal_acc_std": float(np.std(win_bal, ddof=1)) if len(win_bal) > 1 else 0.0,
        "window_macro_f1_mean": float(np.mean(win_f1)),
        "window_macro_f1_std": float(np.std(win_f1, ddof=1)) if len(win_f1) > 1 else 0.0,
        "subject_bal_acc_mean": float(np.mean(subj_bal)),
        "subject_bal_acc_std": float(np.std(subj_bal, ddof=1)) if len(subj_bal) > 1 else 0.0,
        "subject_macro_f1_mean": float(np.mean(subj_f1)),
        "subject_macro_f1_std": float(np.std(subj_f1, ddof=1)) if len(subj_f1) > 1 else 0.0,
        "best_fold_id": int(best_fold_id) if best_fold_id is not None else None,
        "best_fold_subject_balacc": float(best_fold_metric) if best_fold_id is not None else None,
    }

    with open(exp_dir / "cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_fold_state is not None:
        best_model_path = models_dir / f"{run_name}.pt"
        torch.save(
            {
                "base_state_dict": best_fold_state["base_state_dict"],
                "ecn_state_dict": best_fold_state["ecn_state_dict"],
                "model_name": "ImprovedEEGNet+FeatureCorrectionNetwork",
                "best_fold_id": best_fold_id,
                "best_fold_subject_balacc": float(best_fold_metric),
                "in_channels": in_channels,
                "win_len": win_len,
                "target_fs": args.target_fs,
                "num_classes": 2,
                "emb_dim": 128,
                "config": vars(args),
                "cv_summary": summary["aggregate"],
            },
            best_model_path,
        )
        logger.info(f"Best overall model saved: {best_model_path}")

    logger.info("========== CV SUMMARY ==========")
    logger.info(json.dumps(summary["aggregate"], indent=2))
    logger.info("========== RUN END ==========")


if __name__ == "__main__":
    main()