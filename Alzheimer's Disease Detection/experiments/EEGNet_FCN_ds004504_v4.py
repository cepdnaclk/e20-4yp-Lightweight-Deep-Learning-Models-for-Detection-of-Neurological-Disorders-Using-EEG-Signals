#!/usr/bin/env python3
"""
EEGNet_FCN_ds004504_v4.py

Train EEGNetFCN (EEGNet base + FeatureCorrectionNetwork) on:
  <fyp_root>/data/ds004504_preprocessed_shards/{train,eval}

Key fixes over v3 (based on v3 log analysis):
  ─────────────────────────────────────────────────────────────────
  FIX 1 — Augmentation timing
    v3 bug: dataset created with training=True and never toggled,
    so augmentation was ON during Phase 1. This caused the base to
    collapse to 0.8903 (vs 0.9078 in v1 which had no augmentation).
    v4 fix: dataset starts with training=False; each phase function
    receives train_ds and toggles .training appropriately:
      Phase 1 → train_ds.training = False  (clean base training)
      Phase 2 → train_ds.training = True   (FCN trains with aug)
      Phase 3 → train_ds.training = True   (fine-tune with aug)

  FIX 2 — FCN hyperparameters reverted to v1 values
    v3 "relaxed" params (scale_init=0.30, consistency=0.01,
    gate_lambda=0.001) caused gate_mean=0.4441 on P2 epoch 1,
    immediately destabilizing the FCN. P2 never beat P1 in v3.
    In v1 (scale_init=0.10, consistency=0.02, gate_lambda=0.005)
    FCN improved val_bal_acc from 0.9078 → 0.9291 in one epoch.
    v4 defaults: scale_init=0.10, reg_lambda=0.05,
                 consistency_lambda=0.02, gate_lambda=0.005

  FIX 3 — Gradient clipping (max_norm=1.0) in all phases
    No clipping in v3 allowed occasional large gradient steps that
    destabilised training.

  FIX 4 — Phase 3 learning rate increased 5e-6 → 1e-5
    P3 barely moved in v3 (0.8880 → 0.8914). A slightly higher lr
    gives the end-to-end fine-tune more room to improve.

  FIX 5 — LR-scheduler patience in Phase 1: 2 → 3
    v3 P1 LR dropped 5 times in 15 epochs, starving the model of
    useful exploration. patience=3 gives each LR level more time.

  FIX 6 — Label smoothing 0.05 → 0.03
    Slightly softer smoothing; smaller perturbation to gradients.
  ─────────────────────────────────────────────────────────────────

Three-phase training strategy:
  Phase 1 — Train EEGNet base alone  (augmentation OFF)
  Phase 2 — Freeze base, train FCN   (augmentation ON)
  Phase 3 — Unfreeze all, end-to-end (augmentation ON)

Outputs:
  logs    -> <fyp_root>/experiments/logs/EEGNet_FCN_ds004504_v4.log
  model   -> <fyp_root>/models/EEGNet_FCN_ds004504_v4.pt
  metrics -> <fyp_root>/experiments/EEGNet_FCN_ds004504_v4/metrics.json
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


# ============================================================
# EEG Augmentation  (toggled per-phase via dataset.training flag)
# ============================================================

def augment_eeg(x: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    x : float32 array (C, T). Returns augmented copy.
    Each augmentation applied independently:
      1. Gaussian noise    p=0.5
      2. Channel dropout   p=0.3  (zero out 1 random channel)
      3. Time shift        p=0.5  (circular roll, up to ±200 ms)
      4. Amplitude scale   p=0.5  (uniform in [0.85, 1.15])
    """
    x = x.copy()

    if np.random.rand() < 0.5:
        sigma = 0.01 * float(np.std(x)) + 1e-8
        x = x + np.random.randn(*x.shape).astype(np.float32) * sigma

    if np.random.rand() < 0.3:
        ch = np.random.randint(x.shape[0])
        x[ch] = 0.0

    if np.random.rand() < 0.5:
        max_shift = int(0.2 * fs)
        shift = np.random.randint(-max_shift, max_shift + 1)
        x = np.roll(x, shift, axis=-1)

    if np.random.rand() < 0.5:
        x = x * np.random.uniform(0.85, 1.15)

    return x


# ============================================================
# Dataset
# ============================================================

def to_nct(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got {X.shape}")
    return X


class ShardedNpyDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        cache_size: int,
        logger: logging.Logger,
        training: bool = False,   # toggled externally per phase
        fs: int = 250,
    ):
        self.split_dir  = split_dir
        self.cache_size = cache_size
        self.logger     = logger
        self.training   = training
        self.fs         = fs
        self._x_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

        self.x_paths = sorted(split_dir.glob("X_*.npy"))
        if not self.x_paths:
            raise FileNotFoundError(f"No X_*.npy found in {split_dir}")

        self.shards        = []
        self.shard_offsets = []
        self.subjects      = []
        self.labels        = []
        total              = 0

        for xp in self.x_paths:
            sid = xp.stem.split("_")[1]
            yp  = split_dir / f"y_{sid}.npy"
            mp  = split_dir / f"meta_{sid}.csv"

            if not yp.exists():
                raise FileNotFoundError(f"Missing {yp.name} for {xp.name}")
            if not mp.exists():
                raise FileNotFoundError(f"Missing {mp.name} for {xp.name}")

            x_mmap  = to_nct(np.load(xp, mmap_mode="r"))
            y_mmap  = np.load(yp, mmap_mode="r")
            meta_df = pd.read_csv(mp)

            n = int(x_mmap.shape[0])
            if int(y_mmap.shape[0]) != n or len(meta_df) != n:
                raise ValueError(f"Row mismatch in shard {sid}")

            self.shards.append({"x_path": xp, "y_path": yp, "meta_df": meta_df, "n": n})
            self.shard_offsets.append(total)
            total += n

            self.subjects.extend(meta_df["subject"].astype(str).tolist())
            self.labels.extend(meta_df["label"].astype(int).tolist())

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
            mid   = (lo + hi) // 2
            start = self.shard_offsets[mid]
            end   = start + self.shards[mid]["n"]
            if start <= idx < end:
                return mid, idx - start
            if idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        shard_id, local_i = self._locate(idx)
        shard    = self.shards[shard_id]
        x_mmap   = self._get_x_mmap(shard["x_path"])
        y_mmap   = np.load(shard["y_path"], mmap_mode="r")
        meta_row = shard["meta_df"].iloc[local_i].to_dict()

        x = x_mmap[local_i].astype(np.float32)
        y = int(y_mmap[local_i])

        if self.training:
            x = augment_eeg(x, fs=self.fs)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), meta_row


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), list(metas)


def build_subject_balanced_sampler(ds: ShardedNpyDataset) -> WeightedRandomSampler:
    subject_counts = defaultdict(int)
    class_counts   = defaultdict(int)

    for s in ds.subjects:
        subject_counts[s] += 1
    for y in ds.labels:
        class_counts[int(y)] += 1

    sample_weights = [
        (1.0 / max(subject_counts[s], 1)) * (1.0 / max(class_counts[int(y)], 1))
        for s, y in zip(ds.subjects, ds.labels)
    ]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ============================================================
# Subject-level aggregation
# ============================================================

def subject_level_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    subjects: List[str],
) -> Dict[str, Any]:
    """Mean-pool softmax probabilities per subject → subject-level metrics."""
    subj_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    subj_label: Dict[str, int] = {}

    for prob, label, subj in zip(y_pred_probs, y_true, subjects):
        subj_probs[subj].append(prob)
        subj_label[subj] = int(label)

    y_true_subj, y_pred_subj = [], []
    for subj in sorted(subj_probs.keys()):
        mean_prob = np.mean(subj_probs[subj], axis=0)
        y_pred_subj.append(int(np.argmax(mean_prob)))
        y_true_subj.append(subj_label[subj])

    y_true_s = np.array(y_true_subj, dtype=np.int64)
    y_pred_s = np.array(y_pred_subj, dtype=np.int64)

    return {
        "n_subjects":  len(y_true_s),
        "acc":         float(accuracy_score(y_true_s, y_pred_s)),
        "bal_acc":     float(balanced_accuracy_score(y_true_s, y_pred_s)),
        "macro_f1":    float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)),
        "y_true_subj": y_true_s,
        "y_pred_subj": y_pred_s,
    }


# ============================================================
# Model — shared building blocks
# ============================================================

class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden   = max(channels // reduction, 4)
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
        self.dw   = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw   = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.SiLU()
        self.se   = SE1D(out_ch, reduction=8)
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
        Xf      = torch.fft.rfft(x, dim=-1)
        psd     = (Xf.real ** 2 + Xf.imag ** 2) / max(T, 1)
        freqs   = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(x.device)
        bands   = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]

        out         = []
        total_power = psd.sum(dim=-1, keepdim=True) + 1e-8
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            bp   = psd[:, :, mask].sum(dim=-1) if mask.sum() > 0 else torch.zeros(B, C, device=x.device, dtype=x.dtype)
            out.append(bp / total_power.squeeze(-1))
        return torch.cat(out, dim=1)


# ============================================================
# Model — EEGNet base
# ============================================================

class EEGNet(nn.Module):
    def __init__(self, n_channels: int, fs: int, n_classes: int = 2, emb_dim: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.fs         = fs
        self.n_classes  = n_classes
        self.emb_dim    = emb_dim

        self.raw_stem = nn.Sequential(
            nn.Conv1d(n_channels, 48, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            SE1D(48, reduction=8),
        )
        self.block1 = DSConvBlock(48,  64,  k=11, s=2, p=5,  drop=0.10)
        self.block2 = DSConvBlock(64,  96,  k=7,  s=2, p=3,  drop=0.12)
        self.block3 = DSConvBlock(96,  128, k=5,  s=2, p=2,  drop=0.15)
        self.attn   = nn.Conv1d(128, 1, kernel_size=1)

        self.spectral  = SpectralBandPower(fs=fs)
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
        z        = self.raw_stem(x)
        z        = self.block1(z)
        z        = self.block2(z)
        z        = self.block3(z)
        w        = torch.softmax(self.attn(z).squeeze(1), dim=-1)
        raw_emb  = torch.sum(z * w.unsqueeze(1), dim=-1)
        spec_emb = self.spec_proj(self.spectral(x))
        return self.fuse(torch.cat([raw_emb, spec_emb], dim=1))

    def classify_from_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_from_embedding(self.encode(x))


# ============================================================
# Model — FeatureCorrectionNetwork
# ============================================================

class FeatureCorrectionNetwork(nn.Module):
    def __init__(
        self,
        n_channels: int,
        emb_dim: int,
        n_classes: int    = 2,
        scale_init: float = 0.10,   # v1 value — keeps gate controlled at init
    ):
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim   = emb_dim

        self.sig = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        meta_dim = 6
        in_dim   = 64 + emb_dim + n_classes + meta_dim
        hidden   = max(192, emb_dim)

        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.delta_emb_head   = nn.Linear(hidden, emb_dim)
        self.delta_logit_head = nn.Linear(hidden, n_classes)
        self.gate_head        = nn.Linear(hidden, 1)
        self.scale            = nn.Parameter(torch.tensor(float(scale_init)))

    @staticmethod
    def _meta_from_logits(base_logits: torch.Tensor) -> torch.Tensor:
        probs       = torch.softmax(base_logits, dim=1)
        max_p, _    = probs.max(dim=1, keepdim=True)
        sorted_p, _ = probs.sort(dim=1, descending=True)
        margin      = sorted_p[:, :1] - sorted_p[:, 1:2]
        entropy     = -(probs * (probs + 1e-8).log()).sum(dim=1, keepdim=True)
        pos_prob    = probs[:, 1:2]
        if base_logits.shape[1] == 2:
            logit_gap = (base_logits[:, 1:2] - base_logits[:, 0:1]).abs()
        else:
            top2, _   = base_logits.topk(k=2, dim=1)
            logit_gap = (top2[:, :1] - top2[:, 1:2]).abs()
        logit_norm = torch.norm(base_logits, dim=1, keepdim=True)
        return torch.cat([max_p, margin, entropy, pos_prob, logit_gap, logit_norm], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        base_emb: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sig_feat = self.sig(x).squeeze(-1)
        meta     = self._meta_from_logits(base_logits)
        feat     = torch.cat([sig_feat, base_emb, base_logits, meta], dim=1)
        h        = self.shared(feat)

        delta_emb    = torch.tanh(self.delta_emb_head(h))   * self.scale
        delta_logits = torch.tanh(self.delta_logit_head(h)) * self.scale
        gate         = torch.sigmoid(self.gate_head(h))
        corrected_emb = base_emb + gate * delta_emb
        return corrected_emb, delta_emb, delta_logits, gate


# ============================================================
# Model — EEGNetFCN (hybrid)
# ============================================================

class EEGNetFCN(nn.Module):
    """
    Forward:
        base_emb    = base.encode(x)
        base_logits = base.head(base_emb)
        corrected_emb, delta_emb, delta_logits, gate = fcn(x, base_emb, base_logits)
        final_logits = base.head(corrected_emb) + gate * delta_logits
    """
    def __init__(
        self,
        n_channels: int,
        fs: int,
        n_classes: int    = 2,
        emb_dim: int      = 128,
        scale_init: float = 0.10,
    ):
        super().__init__()
        self.base = EEGNet(n_channels=n_channels, fs=fs, n_classes=n_classes, emb_dim=emb_dim)
        self.fcn  = FeatureCorrectionNetwork(
            n_channels=n_channels, emb_dim=emb_dim,
            n_classes=n_classes, scale_init=scale_init,
        )

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

    def unfreeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = True
        self.base.train()

    def freeze_fcn(self):
        for p in self.fcn.parameters():
            p.requires_grad = False

    def unfreeze_fcn(self):
        for p in self.fcn.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    def param_summary(self) -> Dict[str, int]:
        base_p = sum(p.numel() for p in self.base.parameters())
        fcn_p  = sum(p.numel() for p in self.fcn.parameters())
        return {"base_params": base_p, "fcn_params": fcn_p, "total_params": base_p + fcn_p}

    def forward_base_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def forward_with_details(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_emb    = self.base.encode(x)
        base_logits = self.base.classify_from_embedding(base_emb)
        corrected_emb, delta_emb, delta_logits, gate = self.fcn(x, base_emb, base_logits)
        head_logits  = self.base.classify_from_embedding(corrected_emb)
        final_logits = head_logits + gate * delta_logits
        return {
            "base_emb":      base_emb,
            "base_logits":   base_logits,
            "corrected_emb": corrected_emb,
            "delta_emb":     delta_emb,
            "delta_logits":  delta_logits,
            "gate":          gate,
            "final_logits":  final_logits,
        }

    def forward(self, x: torch.Tensor, base_only: bool = False) -> torch.Tensor:
        if base_only:
            return self.forward_base_only(x)
        return self.forward_with_details(x)["final_logits"]


# ============================================================
# Hybrid loss
# ============================================================

def hybrid_loss(
    out: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    criterion: nn.Module,
    reg_lambda: float         = 0.05,   # v1 value
    consistency_lambda: float = 0.02,   # v1 value
    gate_lambda: float        = 0.005,  # v1 value
) -> Tuple[torch.Tensor, Dict[str, float]]:
    cls_loss  = criterion(out["final_logits"], targets)
    delta_reg = reg_lambda * (out["delta_emb"] ** 2).mean()
    gate_reg  = gate_lambda * (out["gate"] ** 2).mean()

    p_lp  = F.log_softmax(out["final_logits"], dim=1)
    q_lp  = F.log_softmax(out["base_logits"],  dim=1)
    kl_pq = F.kl_div(p_lp, q_lp.exp(), reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(q_lp, p_lp.exp(), reduction="batchmean", log_target=False)
    consistency_loss = consistency_lambda * 0.5 * (kl_pq + kl_qp)

    total = cls_loss + delta_reg + gate_reg + consistency_loss
    return total, {
        "cls_loss":         float(cls_loss.item()),
        "delta_reg":        float(delta_reg.item()),
        "gate_reg":         float(gate_reg.item()),
        "consistency_loss": float(consistency_loss.item()),
        "total_loss":       float(total.item()),
    }


# ============================================================
# AMP helper
# ============================================================

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

@torch.no_grad()
def evaluate(
    model: EEGNetFCN,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
    phase: str            = "hybrid",
    reg_lambda: float         = 0.05,
    consistency_lambda: float = 0.02,
    gate_lambda: float        = 0.005,
) -> Dict[str, Any]:
    model.eval()

    total_loss, total_correct, total = 0.0, 0, 0
    y_true, y_pred, y_probs = [], [], []
    subjects_list = []
    gate_vals = []

    for xb, yb, meta in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with amp_autocast(device, use_amp):
            if phase == "base":
                logits = model.forward_base_only(xb)
                loss   = criterion(logits, yb)
            else:
                out    = model.forward_with_details(xb)
                logits = out["final_logits"]
                loss, _ = hybrid_loss(out, yb, criterion, reg_lambda, consistency_lambda, gate_lambda)
                gate_vals.append(float(out["gate"].mean().item()))

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        total_loss    += float(loss.item()) * yb.size(0)
        total_correct += int((preds == yb).sum().item())
        total         += int(yb.size(0))

        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_probs.extend(probs.detach().cpu().numpy().tolist())
        subjects_list.extend([str(m["subject"]) for m in meta])

    y_true  = np.asarray(y_true,  dtype=np.int64)
    y_pred  = np.asarray(y_pred,  dtype=np.int64)
    y_probs = np.asarray(y_probs, dtype=np.float32)

    return {
        "loss":      float(total_loss / max(total, 1)),
        "acc":       float(accuracy_score(y_true, y_pred))            if len(y_true) else 0.0,
        "bal_acc":   float(balanced_accuracy_score(y_true, y_pred))   if len(y_true) else 0.0,
        "macro_f1":  float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0,
        "y_true":    y_true,
        "y_pred":    y_pred,
        "y_probs":   y_probs,
        "subjects":  subjects_list,
        "gate_mean": float(np.mean(gate_vals)) if gate_vals else None,
    }


# ============================================================
# Phase 1 — Train base  (augmentation OFF)
# ============================================================

def train_phase1(
    model: EEGNetFCN,
    train_ds: ShardedNpyDataset,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    args,
    logger: logging.Logger,
    use_amp: bool,
) -> float:
    logger.info("===== PHASE 1: Training base EEGNet  [augmentation OFF] =====")

    # FIX 1: Ensure augmentation is OFF for clean base training
    train_ds.training = False
    logger.info("[P1] train_ds.training = False")

    model.freeze_fcn()
    model.unfreeze_base()

    optimizer = optim.AdamW(model.base.parameters(), lr=args.lr, weight_decay=1e-3)
    # FIX 5: patience=3 (was 2) — less aggressive LR drops
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_metric   = -1.0
    best_state    = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.base.train()
        run_loss, run_correct, total = 0.0, 0, 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_autocast(device, use_amp):
                logits = model.forward_base_only(xb)
                loss   = criterion(logits, yb)

            scaler.scale(loss).backward()
            # FIX 3: gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.base.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            preds        = torch.argmax(logits, dim=1)
            run_loss    += float(loss.item()) * yb.size(0)
            run_correct += int((preds == yb).sum().item())
            total       += int(yb.size(0))

        train_loss = run_loss / max(total, 1)
        train_acc  = run_correct / max(total, 1)

        ev     = evaluate(model, eval_loader, device, criterion, use_amp, phase="base")
        scheduler.step(ev["bal_acc"])
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[P1] Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={ev['loss']:.4f} val_acc={ev['acc']:.4f} "
            f"val_bal_acc={ev['bal_acc']:.4f} val_macro_f1={ev['macro_f1']:.4f} | "
            f"lr={lr_now:.2e}"
        )

        if ev["bal_acc"] > best_metric + 1e-6:
            best_metric   = ev["bal_acc"]
            best_state    = {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("[P1] Early stopping triggered.")
                break

    if best_state is not None:
        model.base.load_state_dict(best_state)
        logger.info(f"[P1] Restored best base (val_bal_acc={best_metric:.4f})")

    return best_metric


# ============================================================
# Phase 2 — Train FCN  (augmentation ON)
# ============================================================

def train_phase2(
    model: EEGNetFCN,
    train_ds: ShardedNpyDataset,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    args,
    logger: logging.Logger,
    use_amp: bool,
) -> float:
    logger.info("===== PHASE 2: Training FeatureCorrectionNetwork  [base frozen, augmentation ON] =====")

    # FIX 1: Enable augmentation from Phase 2 onwards
    if not args.no_augment:
        train_ds.training = True
        logger.info("[P2] train_ds.training = True")

    model.freeze_base()
    model.unfreeze_fcn()

    optimizer = optim.AdamW(model.fcn.parameters(), lr=args.fcn_lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_metric   = -1.0
    best_state    = None
    patience_left = args.fcn_patience

    for epoch in range(1, args.fcn_epochs + 1):
        model.fcn.train()
        run_loss, run_correct, total = 0.0, 0, 0
        gate_vals, delta_vals = [], []

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_autocast(device, use_amp):
                out         = model.forward_with_details(xb)
                loss, parts = hybrid_loss(
                    out, yb, criterion,
                    reg_lambda=args.fcn_reg_lambda,
                    consistency_lambda=args.fcn_consistency_lambda,
                    gate_lambda=args.fcn_gate_lambda,
                )

            scaler.scale(loss).backward()
            # FIX 3: gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.fcn.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            preds        = torch.argmax(out["final_logits"], dim=1)
            run_loss    += float(loss.item()) * yb.size(0)
            run_correct += int((preds == yb).sum().item())
            total       += int(yb.size(0))
            gate_vals.append(float(out["gate"].detach().mean().item()))
            delta_vals.append(float((out["gate"] * out["delta_emb"]).detach().abs().mean().item()))

        train_loss = run_loss / max(total, 1)
        train_acc  = run_correct / max(total, 1)
        gate_mean  = float(np.mean(gate_vals))
        delta_mag  = float(np.mean(delta_vals))

        ev = evaluate(
            model, eval_loader, device, criterion, use_amp, phase="hybrid",
            reg_lambda=args.fcn_reg_lambda,
            consistency_lambda=args.fcn_consistency_lambda,
            gate_lambda=args.fcn_gate_lambda,
        )
        scheduler.step(ev["bal_acc"])
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[P2] Epoch {epoch:03d}/{args.fcn_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"gate_mean={gate_mean:.4f} delta_mag={delta_mag:.6f} | "
            f"val_loss={ev['loss']:.4f} val_acc={ev['acc']:.4f} "
            f"val_bal_acc={ev['bal_acc']:.4f} val_macro_f1={ev['macro_f1']:.4f} | "
            f"lr={lr_now:.2e}"
        )

        if ev["bal_acc"] > best_metric + 1e-6:
            best_metric   = ev["bal_acc"]
            best_state    = {k: v.detach().cpu().clone() for k, v in model.fcn.state_dict().items()}
            patience_left = args.fcn_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("[P2] Early stopping triggered.")
                break

    if best_state is not None:
        model.fcn.load_state_dict(best_state)
        logger.info(f"[P2] Restored best FCN (val_bal_acc={best_metric:.4f})")

    return best_metric


# ============================================================
# Phase 3 — End-to-end fine-tune  (augmentation ON)
# ============================================================

def train_phase3(
    model: EEGNetFCN,
    train_ds: ShardedNpyDataset,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    args,
    logger: logging.Logger,
    use_amp: bool,
) -> float:
    logger.info("===== PHASE 3: End-to-end fine-tuning  [augmentation ON] =====")

    if not args.no_augment:
        train_ds.training = True

    model.unfreeze_all()

    optimizer = optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_metric   = -1.0
    best_base     = None
    best_fcn      = None
    patience_left = args.finetune_patience

    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        run_loss, run_correct, total = 0.0, 0, 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_autocast(device, use_amp):
                out         = model.forward_with_details(xb)
                loss, parts = hybrid_loss(
                    out, yb, criterion,
                    reg_lambda=args.fcn_reg_lambda,
                    consistency_lambda=args.fcn_consistency_lambda,
                    gate_lambda=args.fcn_gate_lambda,
                )

            scaler.scale(loss).backward()
            # FIX 3: gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            preds        = torch.argmax(out["final_logits"], dim=1)
            run_loss    += float(loss.item()) * yb.size(0)
            run_correct += int((preds == yb).sum().item())
            total       += int(yb.size(0))

        train_loss = run_loss / max(total, 1)
        train_acc  = run_correct / max(total, 1)

        ev = evaluate(
            model, eval_loader, device, criterion, use_amp, phase="hybrid",
            reg_lambda=args.fcn_reg_lambda,
            consistency_lambda=args.fcn_consistency_lambda,
            gate_lambda=args.fcn_gate_lambda,
        )
        scheduler.step(ev["bal_acc"])
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[P3] Epoch {epoch:03d}/{args.finetune_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={ev['loss']:.4f} val_acc={ev['acc']:.4f} "
            f"val_bal_acc={ev['bal_acc']:.4f} val_macro_f1={ev['macro_f1']:.4f} | "
            f"lr={lr_now:.2e}"
        )

        if ev["bal_acc"] > best_metric + 1e-6:
            best_metric   = ev["bal_acc"]
            best_base     = {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()}
            best_fcn      = {k: v.detach().cpu().clone() for k, v in model.fcn.state_dict().items()}
            patience_left = args.finetune_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("[P3] Early stopping triggered.")
                break

    if best_base is not None:
        model.base.load_state_dict(best_base)
        model.fcn.load_state_dict(best_fcn)
        logger.info(f"[P3] Restored best end-to-end model (val_bal_acc={best_metric:.4f})")

    return best_metric


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--fyp_root",  type=str, default=str(Path.home() / "FYP"))
    parser.add_argument("--data_dir",  type=str, default="data/ds004504_preprocessed_shards")
    parser.add_argument("--target_fs", type=int, default=250)

    # Data
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size",  type=int, default=2)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no_amp",      action="store_true")
    parser.add_argument("--no_augment",  action="store_true",
                        help="Disable augmentation in all phases")

    # FIX 6: Label smoothing reduced 0.05 → 0.03
    parser.add_argument("--label_smoothing", type=float, default=0.03)

    # Phase 1 — base (NO augmentation)
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--patience", type=int,   default=12)

    # Phase 2 — FCN; FIX 2: reverted to v1 values that worked
    parser.add_argument("--fcn_epochs",              type=int,   default=30)
    parser.add_argument("--fcn_lr",                  type=float, default=5e-4)
    parser.add_argument("--fcn_patience",            type=int,   default=10)
    parser.add_argument("--fcn_reg_lambda",          type=float, default=0.05)   # v1
    parser.add_argument("--fcn_consistency_lambda",  type=float, default=0.02)   # v1
    parser.add_argument("--fcn_gate_lambda",         type=float, default=0.005)  # v1
    parser.add_argument("--fcn_scale_init",          type=float, default=0.10)   # v1

    # Phase 3 — FIX 4: higher finetune_lr
    parser.add_argument("--skip_phase3",       action="store_true")
    parser.add_argument("--finetune_epochs",   type=int,   default=15)
    parser.add_argument("--finetune_lr",       type=float, default=1e-5)  # was 5e-6
    parser.add_argument("--finetune_patience", type=int,   default=7)

    args = parser.parse_args()

    run_name  = "EEGNet_FCN_ds004504_v4"
    fyp_root  = Path(args.fyp_root).resolve()
    data_dir  = (fyp_root / args.data_dir).resolve()
    train_dir = data_dir / "train"
    eval_dir  = data_dir / "eval"

    logger = setup_logger(fyp_root / "experiments" / "logs", run_name)
    logger.info("========== RUN START ==========")
    logger.info(f"Run name        : {run_name}")
    logger.info(f"FYP root        : {fyp_root}")
    logger.info(f"Data dir        : {data_dir}")
    logger.info(f"Python          : {sys.version}")
    logger.info(f"Augmentation    : P1=OFF, P2/P3={'ON' if not args.no_augment else 'OFF (--no_augment)'}")
    logger.info(f"Label smoothing : {args.label_smoothing}")
    logger.info(f"Phase 3         : {not args.skip_phase3}")
    logger.info(f"Grad clip       : max_norm=1.0 (all phases)")
    logger.info(f"FCN scale_init  : {args.fcn_scale_init}  (v1 value)")
    logger.info(f"FCN consistency : {args.fcn_consistency_lambda}  (v1 value)")
    logger.info(f"FCN gate_lambda : {args.fcn_gate_lambda}  (v1 value)")
    logger.info(f"Finetune lr     : {args.finetune_lr}  (raised from 5e-6)")

    if not train_dir.exists() or not eval_dir.exists():
        raise FileNotFoundError(f"train/ or eval/ not found under {data_dir}")

    set_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    logger.info(f"Device: {device} | AMP: {use_amp}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets ──────────────────────────────────────────────
    # FIX 1: Both datasets start with training=False.
    # Phase functions toggle train_ds.training as needed.
    train_ds = ShardedNpyDataset(
        train_dir, cache_size=args.cache_size, logger=logger,
        training=False, fs=args.target_fs,
    )
    eval_ds = ShardedNpyDataset(
        eval_dir, cache_size=args.cache_size, logger=logger,
        training=False, fs=args.target_fs,
    )

    x0, _, _    = train_ds[0]
    in_channels = int(x0.shape[0])
    win_len     = int(x0.shape[1])

    logger.info(f"Train windows: {len(train_ds)} | Eval windows: {len(eval_ds)}")
    logger.info(f"Input shape  : C={in_channels}, T={win_len}")

    class_counts  = {
        int(k): int(v)
        for k, v in pd.Series(train_ds.labels).value_counts().sort_index().to_dict().items()
    }
    unique_labels = sorted(class_counts.keys())
    logger.info(f"Train class counts: {class_counts}")

    counts        = np.array([class_counts[c] for c in unique_labels], dtype=np.float32)
    weights       = 1.0 / (counts + 1e-12)
    weights       = weights / weights.sum() * len(unique_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Class weights: {weights.tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=build_subject_balanced_sampler(train_ds),
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

    # ── Model ─────────────────────────────────────────────────
    model = EEGNetFCN(
        n_channels=in_channels,
        fs=args.target_fs,
        n_classes=2,
        emb_dim=128,
        scale_init=args.fcn_scale_init,
    ).to(device)

    params = model.param_summary()
    logger.info(f"Base params : {params['base_params']:,}")
    logger.info(f"FCN  params : {params['fcn_params']:,}")
    logger.info(f"Total params: {params['total_params']:,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # ── Training ──────────────────────────────────────────────
    t0 = time.time()

    best_p1 = train_phase1(
        model, train_ds, train_loader, eval_loader,
        device, criterion, args, logger, use_amp,
    )
    best_p2 = train_phase2(
        model, train_ds, train_loader, eval_loader,
        device, criterion, args, logger, use_amp,
    )

    best_p3 = None
    if not args.skip_phase3:
        best_p3 = train_phase3(
            model, train_ds, train_loader, eval_loader,
            device, criterion, args, logger, use_amp,
        )

    train_time = time.time() - t0
    logger.info(f"Total training time: {train_time:.2f}s")

    # ── Final evaluation ──────────────────────────────────────
    logger.info("========== FINAL EVALUATION ==========")
    final_ev = evaluate(
        model, eval_loader, device, criterion, use_amp, phase="hybrid",
        reg_lambda=args.fcn_reg_lambda,
        consistency_lambda=args.fcn_consistency_lambda,
        gate_lambda=args.fcn_gate_lambda,
    )

    # Window-level
    report_text = classification_report(
        final_ev["y_true"], final_ev["y_pred"],
        digits=4, target_names=["CN(0)", "AD(1)"], zero_division=0,
    )
    report_dict = classification_report(
        final_ev["y_true"], final_ev["y_pred"],
        digits=4, target_names=["CN(0)", "AD(1)"], output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(final_ev["y_true"], final_ev["y_pred"])

    logger.info("Window-level classification report:\n" + report_text)
    logger.info("Window-level confusion matrix:\n" + str(cm))
    logger.info(
        f"Window-level | val_loss={final_ev['loss']:.4f} val_acc={final_ev['acc']:.4f} "
        f"val_bal_acc={final_ev['bal_acc']:.4f} val_macro_f1={final_ev['macro_f1']:.4f} "
        f"gate_mean={final_ev['gate_mean']:.4f}"
    )

    # Subject-level
    subj_metrics = subject_level_metrics(
        final_ev["y_true"],
        final_ev["y_probs"],
        final_ev["subjects"],
    )
    subj_report_text = classification_report(
        subj_metrics["y_true_subj"], subj_metrics["y_pred_subj"],
        digits=4, target_names=["CN(0)", "AD(1)"], zero_division=0,
    )
    subj_report_dict = classification_report(
        subj_metrics["y_true_subj"], subj_metrics["y_pred_subj"],
        digits=4, target_names=["CN(0)", "AD(1)"], output_dict=True, zero_division=0,
    )
    subj_cm = confusion_matrix(subj_metrics["y_true_subj"], subj_metrics["y_pred_subj"])

    logger.info(f"Subject-level report (n={subj_metrics['n_subjects']} subjects):\n" + subj_report_text)
    logger.info("Subject-level confusion matrix:\n" + str(subj_cm))
    logger.info(
        f"Subject-level | acc={subj_metrics['acc']:.4f} "
        f"bal_acc={subj_metrics['bal_acc']:.4f} "
        f"macro_f1={subj_metrics['macro_f1']:.4f}"
    )

    # ── Save ──────────────────────────────────────────────────
    models_dir = fyp_root / "models"
    exp_dir    = fyp_root / "experiments" / run_name
    ensure_dir(models_dir)
    ensure_dir(exp_dir)

    model_path   = models_dir / f"{run_name}.pt"
    metrics_path = exp_dir / "metrics.json"
    cm_path      = exp_dir / "confusion_matrix.npy"
    subj_cm_path = exp_dir / "confusion_matrix_subject.npy"

    torch.save(
        {
            "base_state_dict": model.base.state_dict(),
            "fcn_state_dict":  model.fcn.state_dict(),
            "model_name":      "EEGNetFCN_v4",
            "in_channels":     in_channels,
            "win_len":         win_len,
            "target_fs":       args.target_fs,
            "num_classes":     2,
            "emb_dim":         128,
            "best_p1_bal_acc": float(best_p1),
            "best_p2_bal_acc": float(best_p2),
            "best_p3_bal_acc": float(best_p3) if best_p3 is not None else None,
            "config":          vars(args),
        },
        model_path,
    )

    metrics = {
        "run_name":         run_name,
        "train_windows":    int(len(train_ds)),
        "eval_windows":     int(len(eval_ds)),
        "train_counts":     class_counts,
        "train_time_sec":   float(train_time),
        "augmentation":     "P1=OFF, P2/P3=ON",
        "label_smoothing":  args.label_smoothing,
        "best_p1_bal_acc":  float(best_p1),
        "best_p2_bal_acc":  float(best_p2),
        "best_p3_bal_acc":  float(best_p3) if best_p3 is not None else None,
        "model_params":     params,
        "window_eval": {
            "loss":                  float(final_ev["loss"]),
            "acc":                   float(final_ev["acc"]),
            "bal_acc":               float(final_ev["bal_acc"]),
            "macro_f1":              float(final_ev["macro_f1"]),
            "gate_mean":             final_ev["gate_mean"],
            "classification_report": report_dict,
            "confusion_matrix":      cm.tolist(),
        },
        "subject_eval": {
            "n_subjects":            int(subj_metrics["n_subjects"]),
            "acc":                   float(subj_metrics["acc"]),
            "bal_acc":               float(subj_metrics["bal_acc"]),
            "macro_f1":              float(subj_metrics["macro_f1"]),
            "classification_report": subj_report_dict,
            "confusion_matrix":      subj_cm.tolist(),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.save(cm_path,      cm)
    np.save(subj_cm_path, subj_cm)

    logger.info(f"Saved model        : {model_path}")
    logger.info(f"Saved metrics      : {metrics_path}")
    logger.info(f"Saved window CM    : {cm_path}")
    logger.info(f"Saved subject CM   : {subj_cm_path}")
    logger.info("========== RUN END ==========")


if __name__ == "__main__":
    main()