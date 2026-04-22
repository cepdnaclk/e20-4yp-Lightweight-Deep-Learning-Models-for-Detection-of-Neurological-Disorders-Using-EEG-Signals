#!/usr/bin/env python3
"""
EEGNet_FCN_ds004504_v5.py

Trains EEGNetFCN on ds004504_preprocessed_shards with an upgraded
FeatureCorrectionNetwork (FCN).

New in v5 over v4
─────────────────────────────────────────────────────────────────────
UPGRADE 1 — Per-dimension gate
  v4: scalar gate (B×1) applied uniformly to all embedding dims
  v5: gate_head outputs B×emb_dim → element-wise control, allowing
      the FCN to selectively correct individual features

UPGRADE 2 — Multi-scale base features replace raw sig branch
  v4: FCN re-learned shallow conv features from raw EEG (redundant)
  v5: FCN receives z1 (64-ch), z2 (96-ch), z3 (128-ch) intermediate
      feature maps from EEGNet blocks (pooled → projected to 64-d),
      giving richer, already-computed representations

UPGRADE 3 — Directional consistency loss
  v4: symmetric KL(final||base) penalised FCN for any deviation,
      even beneficial corrections when base was wrong
  v5: regression_penalty = ReLU(CE_final − CE_base).mean()
      → only penalises when FCN increases per-sample loss

UPGRADE 4 — Gate entropy regulariser
  v4: gate_reg = gate_lambda × gate² → pushed gates toward 0 (collapse)
  v5: add −gate_entropy_lambda × H(gate) to reward gates near 0.5,
      preventing lazy all-zero gate behaviour

UPGRADE 5 — Delta supervision (residual learning signal)
  v4: no direct supervision on the delta_logits branch
  v5: per-sample CE(delta_logits, targets) × base_wrong_mask
      → gives FCN an explicit gradient to fix mis-classified windows

UPGRADE 6 — Differential learning rates in Phase 3
  v4: single lr=1e-5 for all params; base degraded in P3
  v5: base gets lr × p3_base_lr_scale (default 0.1),
      FCN gets full lr → protects the strong base

UPGRADE 7 — Uncertainty-aware sampler (optional) in Phase 2
  v4: subject+class balanced sampler throughout
  v5: --uncertainty_sampler upweights low-confidence windows
      during P2 where FCN corrections are most valuable
─────────────────────────────────────────────────────────────────────

Three-phase strategy (same as v4):
  Phase 1 — Train EEGNet base alone    (augmentation OFF)
  Phase 2 — Freeze base, train FCN     (augmentation ON)
  Phase 3 — Unfreeze all, end-to-end  (augmentation ON)

Outputs:
  logs    → <fyp_root>/experiments/logs/EEGNet_FCN_ds004504_v5.log
  model   → <fyp_root>/models/EEGNet_FCN_ds004504_v5.pt
  metrics → <fyp_root>/experiments/EEGNet_FCN_ds004504_v5/metrics.json
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, List, Any, Tuple, Optional

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
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(ch)
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
# EEG Augmentation
# ============================================================

def augment_eeg(x: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    x : float32 array (C, T). Returns augmented copy.
      1. Gaussian noise    p=0.5
      2. Channel dropout   p=0.3  (zero out 1 random channel)
      3. Time shift        p=0.5  (circular roll ±200 ms)
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
        training: bool = False,
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

        self.shards, self.shard_offsets = [], []
        self.subjects, self.labels = [], []
        total = 0

        for xp in self.x_paths:
            sid = xp.stem.split("_")[1]
            yp  = split_dir / f"y_{sid}.npy"
            mp  = split_dir / f"meta_{sid}.csv"
            if not yp.exists(): raise FileNotFoundError(f"Missing {yp.name}")
            if not mp.exists(): raise FileNotFoundError(f"Missing {mp.name}")

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
            if start <= idx < end: return mid, idx - start
            if idx < start: hi = mid - 1
            else: lo = mid + 1
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
    return torch.stack(xs, 0), torch.stack(ys, 0), list(metas)


def build_subject_balanced_sampler(ds: ShardedNpyDataset) -> WeightedRandomSampler:
    subject_counts: Dict[str, int] = defaultdict(int)
    class_counts:   Dict[int, int] = defaultdict(int)
    for s in ds.subjects: subject_counts[s] += 1
    for y in ds.labels:   class_counts[int(y)] += 1
    w = [
        (1.0 / max(subject_counts[s], 1)) * (1.0 / max(class_counts[int(y)], 1))
        for s, y in zip(ds.subjects, ds.labels)
    ]
    wt = torch.tensor(w, dtype=torch.double)
    return WeightedRandomSampler(weights=wt, num_samples=len(wt), replacement=True)


# ============================================================
# UPGRADE 7 — Uncertainty-aware sampler utilities
# ============================================================

def build_uncertainty_sampler(
    train_ds: ShardedNpyDataset,
    confidences: np.ndarray,
) -> WeightedRandomSampler:
    """
    Combine subject+class weights with inverse-confidence weights.
    Windows where the base model is uncertain get up-weighted so the
    FCN sees more of the examples it actually needs to correct.
    """
    subject_counts: Dict[str, int] = defaultdict(int)
    class_counts:   Dict[int, int] = defaultdict(int)
    for s in train_ds.subjects: subject_counts[s] += 1
    for y in train_ds.labels:   class_counts[int(y)] += 1
    base_w = np.array([
        (1.0 / max(subject_counts[s], 1)) * (1.0 / max(class_counts[int(y)], 1))
        for s, y in zip(train_ds.subjects, train_ds.labels)
    ], dtype=np.float64)

    # Clip confidences to [0.5, 1.0] → uncertainty in [0, 0.5]
    uncertainty = 1.0 - confidences.clip(0.5, 1.0).astype(np.float64)
    combined    = base_w * (1.0 + uncertainty)
    combined    = combined / combined.sum() * len(combined)
    wt = torch.tensor(combined, dtype=torch.double)
    return WeightedRandomSampler(weights=wt, num_samples=len(wt), replacement=True)


# ============================================================
# Subject-level aggregation
# ============================================================

def subject_level_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    subjects: List[str],
) -> Dict[str, Any]:
    subj_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    subj_label: Dict[str, int] = {}
    for prob, label, subj in zip(y_pred_probs, y_true, subjects):
        subj_probs[subj].append(prob)
        subj_label[subj] = int(label)
    y_true_s, y_pred_s = [], []
    for subj in sorted(subj_probs):
        mean_prob = np.mean(subj_probs[subj], axis=0)
        y_pred_s.append(int(np.argmax(mean_prob)))
        y_true_s.append(subj_label[subj])
    yt = np.array(y_true_s, dtype=np.int64)
    yp = np.array(y_pred_s, dtype=np.int64)
    return {
        "n_subjects":  len(yt),
        "acc":         float(accuracy_score(yt, yp)),
        "bal_acc":     float(balanced_accuracy_score(yt, yp)),
        "macro_f1":    float(f1_score(yt, yp, average="macro", zero_division=0)),
        "y_true_subj": yt,
        "y_pred_subj": yp,
    }


# ============================================================
# Model — shared building blocks  (unchanged)
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
        self.dw   = nn.Conv1d(in_ch, in_ch,  kernel_size=k, stride=s, padding=p,
                               groups=in_ch, bias=False)
        self.pw   = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.SiLU()
        self.se   = SE1D(out_ch, reduction=8)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x); x = self.pw(x); x = self.bn(x)
        x = self.act(x); x = self.se(x); x = self.drop(x)
        return x


class SpectralBandPower(nn.Module):
    def __init__(self, fs: int):
        super().__init__()
        self.fs = fs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        Xf    = torch.fft.rfft(x, dim=-1)
        psd   = (Xf.real ** 2 + Xf.imag ** 2) / max(T, 1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(x.device)
        bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
        out   = []
        total = psd.sum(dim=-1, keepdim=True) + 1e-8
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            bp   = (psd[:, :, mask].sum(dim=-1) if mask.sum() > 0
                    else torch.zeros(B, C, device=x.device, dtype=x.dtype))
            out.append(bp / total.squeeze(-1))
        return torch.cat(out, dim=1)


# ============================================================
# Model — EEGNet base
# UPGRADE 2: added encode_with_intermediates()
# ============================================================

class EEGNet(nn.Module):
    def __init__(self, n_channels: int, fs: int,
                 n_classes: int = 2, emb_dim: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.fs         = fs
        self.n_classes  = n_classes
        self.emb_dim    = emb_dim

        self.raw_stem = nn.Sequential(
            nn.Conv1d(n_channels, 48, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(48), nn.SiLU(), SE1D(48, reduction=8),
        )
        # Block output channels: 64, 96, 128 — must match FCN z_ch args
        self.block1 = DSConvBlock(48,  64,  k=11, s=2, p=5,  drop=0.10)
        self.block2 = DSConvBlock(64,  96,  k=7,  s=2, p=3,  drop=0.12)
        self.block3 = DSConvBlock(96,  128, k=5,  s=2, p=2,  drop=0.15)
        self.attn   = nn.Conv1d(128, 1, kernel_size=1)

        self.spectral  = SpectralBandPower(fs=fs)
        self.spec_proj = nn.Sequential(
            nn.Linear(n_channels * 5, 96), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(96, 64), nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + 64, emb_dim), nn.SiLU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 64), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def encode_with_intermediates(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (emb, z1, z2, z3):
          emb : (B, emb_dim)   — final fused embedding
          z1  : (B, 64,  T/4)  — after block1
          z2  : (B, 96,  T/8)  — after block2
          z3  : (B, 128, T/16) — after block3
        """
        z        = self.raw_stem(x)
        z1       = self.block1(z)
        z2       = self.block2(z1)
        z3       = self.block3(z2)
        w        = torch.softmax(self.attn(z3).squeeze(1), dim=-1)
        raw_emb  = torch.sum(z3 * w.unsqueeze(1), dim=-1)
        spec_emb = self.spec_proj(self.spectral(x))
        emb      = self.fuse(torch.cat([raw_emb, spec_emb], dim=1))
        return emb, z1, z2, z3

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb, _, _, _ = self.encode_with_intermediates(x)
        return emb

    def classify_from_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_from_embedding(self.encode(x))


# ============================================================
# Model — FeatureCorrectionNetwork v5
# UPGRADES 1, 2 (architecture), 4 (entropy regulariser in loss)
# ============================================================

class FeatureCorrectionNetwork(nn.Module):
    """
    v5 changes vs v4:
      • Removed raw-EEG sig branch (shallow 3-layer conv, redundant)
      • Added ms_proj: pools z1/z2/z3 from base → 64-d feature vector
        (richer multi-scale context, same compute budget)
      • gate_head outputs emb_dim per-dimension gates (was scalar B×1)
    """

    def __init__(
        self,
        emb_dim: int,
        n_classes: int    = 2,
        scale_init: float = 0.10,
        z1_ch: int        = 64,   # must match EEGNet block1 out channels
        z2_ch: int        = 96,   # must match EEGNet block2 out channels
        z3_ch: int        = 128,  # must match EEGNet block3 out channels
    ):
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim   = emb_dim

        # UPGRADE 2: multi-scale feature projector (replaces raw sig branch)
        ms_in_dim = z1_ch + z2_ch + z3_ch  # 288
        self.ms_proj = nn.Sequential(
            nn.Linear(ms_in_dim, 128), nn.SiLU(), nn.Dropout(0.20),
            nn.Linear(128, 64),        nn.SiLU(),
        )

        meta_dim = 6
        in_dim   = 64 + emb_dim + n_classes + meta_dim  # 64+128+2+6 = 200
        hidden   = max(192, emb_dim)

        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(0.20),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.15),
        )
        self.delta_emb_head   = nn.Linear(hidden, emb_dim)
        self.delta_logit_head = nn.Linear(hidden, n_classes)
        # UPGRADE 1: per-dimension gate (emb_dim outputs, not 1)
        self.gate_head        = nn.Linear(hidden, emb_dim)
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
        base_emb:    torch.Tensor,
        base_logits: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # UPGRADE 2: pool multi-scale features and project
        p1 = F.adaptive_avg_pool1d(z1, 1).squeeze(-1)   # (B, 64)
        p2 = F.adaptive_avg_pool1d(z2, 1).squeeze(-1)   # (B, 96)
        p3 = F.adaptive_avg_pool1d(z3, 1).squeeze(-1)   # (B, 128)
        ms_feat = self.ms_proj(torch.cat([p1, p2, p3], dim=1))  # (B, 64)

        meta = self._meta_from_logits(base_logits)
        feat = torch.cat([ms_feat, base_emb, base_logits, meta], dim=1)
        h    = self.shared(feat)

        delta_emb    = torch.tanh(self.delta_emb_head(h))   * self.scale  # (B, emb_dim)
        delta_logits = torch.tanh(self.delta_logit_head(h)) * self.scale  # (B, n_classes)
        # UPGRADE 1: per-dimension gate
        gate          = torch.sigmoid(self.gate_head(h))                  # (B, emb_dim)
        corrected_emb = base_emb + gate * delta_emb
        return corrected_emb, delta_emb, delta_logits, gate


# ============================================================
# Model — EEGNetFCN (hybrid)
# Updated forward_with_details to pass intermediates to FCN
# ============================================================

class EEGNetFCN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        fs: int,
        n_classes: int    = 2,
        emb_dim: int      = 128,
        scale_init: float = 0.10,
    ):
        super().__init__()
        self.base = EEGNet(n_channels=n_channels, fs=fs,
                           n_classes=n_classes, emb_dim=emb_dim)
        self.fcn  = FeatureCorrectionNetwork(
            emb_dim=emb_dim, n_classes=n_classes, scale_init=scale_init,
            z1_ch=64, z2_ch=96, z3_ch=128,  # match EEGNet block channels
        )

    def freeze_base(self):
        for p in self.base.parameters(): p.requires_grad = False
        self.base.eval()

    def unfreeze_base(self):
        for p in self.base.parameters(): p.requires_grad = True
        self.base.train()

    def freeze_fcn(self):
        for p in self.fcn.parameters(): p.requires_grad = False

    def unfreeze_fcn(self):
        for p in self.fcn.parameters(): p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True
        self.train()

    def param_summary(self) -> Dict[str, int]:
        base_p = sum(p.numel() for p in self.base.parameters())
        fcn_p  = sum(p.numel() for p in self.fcn.parameters())
        return {"base_params": base_p, "fcn_params": fcn_p,
                "total_params": base_p + fcn_p}

    def forward_base_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def forward_with_details(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # UPGRADE 2: get intermediate feature maps from base
        base_emb, z1, z2, z3 = self.base.encode_with_intermediates(x)
        base_logits = self.base.classify_from_embedding(base_emb)

        corrected_emb, delta_emb, delta_logits, gate = self.fcn(
            base_emb, base_logits, z1, z2, z3
        )
        head_logits  = self.base.classify_from_embedding(corrected_emb)
        # Mean of per-dim gate used as scalar weight for delta_logits
        gate_scalar  = gate.mean(dim=1, keepdim=True)           # (B, 1)
        final_logits = head_logits + gate_scalar * delta_logits

        return {
            "base_emb":      base_emb,
            "base_logits":   base_logits,
            "corrected_emb": corrected_emb,
            "delta_emb":     delta_emb,
            "delta_logits":  delta_logits,
            "gate":          gate,         # (B, emb_dim)  per-dimension
            "gate_scalar":   gate_scalar,  # (B, 1)        mean gate
            "final_logits":  final_logits,
        }

    def forward(self, x: torch.Tensor, base_only: bool = False) -> torch.Tensor:
        if base_only:
            return self.forward_base_only(x)
        return self.forward_with_details(x)["final_logits"]


# ============================================================
# Hybrid loss v5
# UPGRADES 3 (directional consistency), 4 (gate entropy),
#           5 (delta supervision)
# ============================================================

def hybrid_loss(
    out: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    criterion: nn.Module,
    class_weights: Optional[torch.Tensor]   = None,
    label_smoothing: float                  = 0.03,
    reg_lambda: float                       = 0.05,
    consistency_lambda: float               = 0.02,
    gate_lambda: float                      = 0.005,
    gate_entropy_lambda: float              = 0.01,
    delta_supervision_lambda: float         = 0.10,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    # ── Primary classification loss ──────────────────────────
    cls_loss = criterion(out["final_logits"], targets)

    # ── Delta embedding L2 regularisation ────────────────────
    delta_reg = reg_lambda * (out["delta_emb"] ** 2).mean()

    # ── Gate L2 (mild, on mean gate scalar) ──────────────────
    gate_reg = gate_lambda * (out["gate_scalar"] ** 2).mean()

    # ── UPGRADE 4: Gate entropy bonus ────────────────────────
    # Binary entropy H(g) = -(g·log g + (1−g)·log(1−g)) ∈ [0, log 2]
    # Subtract from loss → gradient pushes gates toward 0.5 (not collapse)
    eps    = 1e-8
    g      = out["gate"]
    H_gate = -(g * (g + eps).log() + (1 - g) * (1 - g + eps).log())
    gate_entropy_loss = -gate_entropy_lambda * H_gate.mean()

    # ── UPGRADE 3: Directional consistency loss ───────────────
    # Penalise only when FCN increases per-sample cross-entropy
    # (i.e., makes the prediction worse than the base)
    _ce_kw = dict(label_smoothing=label_smoothing, reduction="none")
    with torch.no_grad():
        base_ce_ref = (
            F.cross_entropy(out["base_logits"].detach(), targets,
                            weight=class_weights, **_ce_kw)
            if class_weights is not None
            else F.cross_entropy(out["base_logits"].detach(), targets, **_ce_kw)
        )
    final_ce = (
        F.cross_entropy(out["final_logits"], targets,
                        weight=class_weights, **_ce_kw)
        if class_weights is not None
        else F.cross_entropy(out["final_logits"], targets, **_ce_kw)
    )
    regression_penalty = F.relu(final_ce - base_ce_ref).mean()
    consistency_loss   = consistency_lambda * regression_penalty

    # ── UPGRADE 5: Delta supervision ─────────────────────────
    # When base was wrong, directly supervise delta_logits to be correct.
    # This gives the FCN an explicit signal about when to intervene.
    if delta_supervision_lambda > 0.0:
        with torch.no_grad():
            base_wrong = (out["base_logits"].detach().argmax(dim=1) != targets).float()
        delta_ce = (
            F.cross_entropy(out["delta_logits"], targets,
                            weight=class_weights, **_ce_kw)
            if class_weights is not None
            else F.cross_entropy(out["delta_logits"], targets, **_ce_kw)
        )
        delta_supervision = delta_supervision_lambda * (base_wrong * delta_ce).mean()
    else:
        delta_supervision = torch.zeros(1, device=targets.device)[0]

    # ── Total ─────────────────────────────────────────────────
    total = (cls_loss + delta_reg + gate_reg + gate_entropy_loss +
             consistency_loss + delta_supervision)

    return total, {
        "cls_loss":             float(cls_loss.item()),
        "delta_reg":            float(delta_reg.item()),
        "gate_reg":             float(gate_reg.item()),
        "gate_entropy_loss":    float(gate_entropy_loss.item()),
        "consistency_loss":     float(consistency_loss.item()),
        "delta_supervision":    float(delta_supervision.item()),
        "total_loss":           float(total.item()),
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
# Compute base confidences (for UPGRADE 7 sampler)
# ============================================================

@torch.no_grad()
def compute_base_confidences(
    model: EEGNetFCN,
    train_ds: ShardedNpyDataset,
    device: torch.device,
    use_amp: bool,
    batch_size: int = 64,
    num_workers: int = 2,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Returns max-softmax confidence for every training window (base only)."""
    orig_training      = train_ds.training
    train_ds.training  = False
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_meta,
    )
    model.eval()
    all_confs: List[float] = []
    for xb, _, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with amp_autocast(device, use_amp):
            logits = model.forward_base_only(xb)
        conf = torch.softmax(logits, dim=1).max(dim=1).values
        all_confs.extend(conf.cpu().numpy().tolist())
    train_ds.training = orig_training
    confs = np.array(all_confs, dtype=np.float32)
    if logger:
        logger.info(
            f"[P2] Base confidences | mean={confs.mean():.4f} | "
            f"low-conf(<0.75): {int((confs < 0.75).sum())}/{len(confs)}"
        )
    return confs


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
    phase: str                              = "hybrid",
    class_weights: Optional[torch.Tensor]  = None,
    label_smoothing: float                 = 0.03,
    reg_lambda: float                      = 0.05,
    consistency_lambda: float              = 0.02,
    gate_lambda: float                     = 0.005,
    gate_entropy_lambda: float             = 0.01,
    delta_supervision_lambda: float        = 0.10,
) -> Dict[str, Any]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    y_true, y_pred, y_probs = [], [], []
    subjects_list: List[str] = []
    gate_vals: List[float]   = []

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
                loss, _ = hybrid_loss(
                    out, yb, criterion,
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                    reg_lambda=reg_lambda,
                    consistency_lambda=consistency_lambda,
                    gate_lambda=gate_lambda,
                    gate_entropy_lambda=gate_entropy_lambda,
                    delta_supervision_lambda=delta_supervision_lambda,
                )
                gate_vals.append(float(out["gate_scalar"].mean().item()))

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        total_loss    += float(loss.item()) * yb.size(0)
        total_correct += int((preds == yb).sum().item())
        total         += int(yb.size(0))
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_probs.extend(probs.cpu().numpy().tolist())
        subjects_list.extend([str(m["subject"]) for m in meta])

    yt = np.asarray(y_true,  dtype=np.int64)
    yp = np.asarray(y_pred,  dtype=np.int64)
    ypr = np.asarray(y_probs, dtype=np.float32)

    return {
        "loss":      float(total_loss / max(total, 1)),
        "acc":       float(accuracy_score(yt, yp))            if len(yt) else 0.0,
        "bal_acc":   float(balanced_accuracy_score(yt, yp))   if len(yt) else 0.0,
        "macro_f1":  float(f1_score(yt, yp, average="macro")) if len(yt) else 0.0,
        "y_true":    yt,
        "y_pred":    yp,
        "y_probs":   ypr,
        "subjects":  subjects_list,
        "gate_mean": float(np.mean(gate_vals)) if gate_vals else None,
    }


# ============================================================
# Phase 1 — Train base EEGNet  [augmentation OFF]  (unchanged)
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
    train_ds.training = False
    logger.info("[P1] train_ds.training = False")

    model.freeze_fcn()
    model.unfreeze_base()

    optimizer = optim.AdamW(model.base.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_metric, best_state, patience_left = -1.0, None, args.patience

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
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.base.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            preds        = torch.argmax(logits, dim=1)
            run_loss    += float(loss.item()) * yb.size(0)
            run_correct += int((preds == yb).sum().item())
            total       += int(yb.size(0))

        train_loss = run_loss / max(total, 1)
        train_acc  = run_correct / max(total, 1)
        ev = evaluate(model, eval_loader, device, criterion, use_amp, phase="base")
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
            best_state    = {k: v.detach().cpu().clone()
                             for k, v in model.base.state_dict().items()}
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
# Phase 2 — Train FCN  [augmentation ON]
# UPGRADE 7: optional uncertainty-aware sampler
# ============================================================

def train_phase2(
    model: EEGNetFCN,
    train_ds: ShardedNpyDataset,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    class_weights: torch.Tensor,
    args,
    logger: logging.Logger,
    use_amp: bool,
) -> float:
    logger.info("===== PHASE 2: Training FCN  [base frozen, augmentation ON] =====")
    if not args.no_augment:
        train_ds.training = True
        logger.info("[P2] train_ds.training = True")

    # UPGRADE 7: optionally rebuild sampler with uncertainty weighting
    if args.uncertainty_sampler:
        logger.info("[P2] Building uncertainty-aware sampler ...")
        confs      = compute_base_confidences(
            model, train_ds, device, use_amp,
            batch_size=64, num_workers=args.num_workers, logger=logger,
        )
        p2_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=build_uncertainty_sampler(train_ds, confs),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_with_meta,
        )
        logger.info("[P2] Uncertainty-aware sampler active.")
    else:
        p2_loader = train_loader

    model.freeze_base()
    model.unfreeze_fcn()

    optimizer = optim.AdamW(model.fcn.parameters(),
                             lr=args.fcn_lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_metric, best_state, patience_left = -1.0, None, args.fcn_patience

    for epoch in range(1, args.fcn_epochs + 1):
        model.fcn.train()
        run_loss, run_correct, total = 0.0, 0, 0
        gate_vals, delta_vals = [], []

        for xb, yb, _ in p2_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, use_amp):
                out = model.forward_with_details(xb)
                loss, _ = hybrid_loss(
                    out, yb, criterion,
                    class_weights=class_weights,
                    label_smoothing=args.label_smoothing,
                    reg_lambda=args.fcn_reg_lambda,
                    consistency_lambda=args.fcn_consistency_lambda,
                    gate_lambda=args.fcn_gate_lambda,
                    gate_entropy_lambda=args.fcn_gate_entropy_lambda,
                    delta_supervision_lambda=args.fcn_delta_supervision_lambda,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.fcn.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()

            preds        = torch.argmax(out["final_logits"], dim=1)
            run_loss    += float(loss.item()) * yb.size(0)
            run_correct += int((preds == yb).sum().item())
            total       += int(yb.size(0))
            gate_vals.append(float(out["gate_scalar"].detach().mean().item()))
            delta_vals.append(float(
                (out["gate_scalar"] * out["delta_emb"]).detach().abs().mean().item()
            ))

        train_loss = run_loss / max(total, 1)
        train_acc  = run_correct / max(total, 1)
        gate_mean  = float(np.mean(gate_vals))
        delta_mag  = float(np.mean(delta_vals))

        ev = evaluate(
            model, eval_loader, device, criterion, use_amp, phase="hybrid",
            class_weights=class_weights, label_smoothing=args.label_smoothing,
            reg_lambda=args.fcn_reg_lambda,
            consistency_lambda=args.fcn_consistency_lambda,
            gate_lambda=args.fcn_gate_lambda,
            gate_entropy_lambda=args.fcn_gate_entropy_lambda,
            delta_supervision_lambda=args.fcn_delta_supervision_lambda,
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
            best_state    = {k: v.detach().cpu().clone()
                             for k, v in model.fcn.state_dict().items()}
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
# Phase 3 — End-to-end fine-tune  [augmentation ON]
# UPGRADE 6: differential learning rates
# ============================================================

def train_phase3(
    model: EEGNetFCN,
    train_ds: ShardedNpyDataset,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    class_weights: torch.Tensor,
    args,
    logger: logging.Logger,
    use_amp: bool,
) -> float:
    logger.info("===== PHASE 3: End-to-end fine-tuning  [augmentation ON] =====")
    if not args.no_augment:
        train_ds.training = True

    model.unfreeze_all()

    # UPGRADE 6: differential LR — protect base with much lower LR
    base_lr = args.finetune_lr * args.p3_base_lr_scale
    fcn_lr  = args.finetune_lr
    logger.info(f"[P3] Differential LR | base={base_lr:.2e}  FCN={fcn_lr:.2e}")

    optimizer = optim.AdamW([
        {"params": model.base.parameters(), "lr": base_lr, "weight_decay": 1e-4},
        {"params": model.fcn.parameters(),  "lr": fcn_lr,  "weight_decay": 1e-4},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_metric                = -1.0
    best_base: Optional[dict]  = None
    best_fcn:  Optional[dict]  = None
    patience_left              = args.finetune_patience

    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        run_loss, run_correct, total = 0.0, 0, 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, use_amp):
                out = model.forward_with_details(xb)
                loss, _ = hybrid_loss(
                    out, yb, criterion,
                    class_weights=class_weights,
                    label_smoothing=args.label_smoothing,
                    reg_lambda=args.fcn_reg_lambda,
                    consistency_lambda=args.fcn_consistency_lambda,
                    gate_lambda=args.fcn_gate_lambda,
                    gate_entropy_lambda=args.fcn_gate_entropy_lambda,
                    delta_supervision_lambda=args.fcn_delta_supervision_lambda,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()

            preds        = torch.argmax(out["final_logits"], dim=1)
            run_loss    += float(loss.item()) * yb.size(0)
            run_correct += int((preds == yb).sum().item())
            total       += int(yb.size(0))

        train_loss = run_loss / max(total, 1)
        train_acc  = run_correct / max(total, 1)

        ev = evaluate(
            model, eval_loader, device, criterion, use_amp, phase="hybrid",
            class_weights=class_weights, label_smoothing=args.label_smoothing,
            reg_lambda=args.fcn_reg_lambda,
            consistency_lambda=args.fcn_consistency_lambda,
            gate_lambda=args.fcn_gate_lambda,
            gate_entropy_lambda=args.fcn_gate_entropy_lambda,
            delta_supervision_lambda=args.fcn_delta_supervision_lambda,
        )
        scheduler.step(ev["bal_acc"])
        lr_base = optimizer.param_groups[0]["lr"]
        lr_fcn  = optimizer.param_groups[1]["lr"]

        logger.info(
            f"[P3] Epoch {epoch:03d}/{args.finetune_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={ev['loss']:.4f} val_acc={ev['acc']:.4f} "
            f"val_bal_acc={ev['bal_acc']:.4f} val_macro_f1={ev['macro_f1']:.4f} | "
            f"lr_base={lr_base:.2e} lr_fcn={lr_fcn:.2e}"
        )

        if ev["bal_acc"] > best_metric + 1e-6:
            best_metric   = ev["bal_acc"]
            best_base     = {k: v.detach().cpu().clone()
                             for k, v in model.base.state_dict().items()}
            best_fcn      = {k: v.detach().cpu().clone()
                             for k, v in model.fcn.state_dict().items()}
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
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--num_workers",     type=int,   default=2)
    parser.add_argument("--cache_size",      type=int,   default=2)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--no_amp",          action="store_true")
    parser.add_argument("--no_augment",      action="store_true",
                        help="Disable augmentation in all phases")
    parser.add_argument("--label_smoothing", type=float, default=0.03)

    # Phase 1 — base (augmentation OFF)
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--patience", type=int,   default=12)

    # Phase 2 — FCN (v1 hyperparameters retained)
    parser.add_argument("--fcn_epochs",             type=int,   default=30)
    parser.add_argument("--fcn_lr",                 type=float, default=5e-4)
    parser.add_argument("--fcn_patience",           type=int,   default=10)
    parser.add_argument("--fcn_reg_lambda",         type=float, default=0.05)
    parser.add_argument("--fcn_consistency_lambda", type=float, default=0.02)
    parser.add_argument("--fcn_gate_lambda",        type=float, default=0.005)
    parser.add_argument("--fcn_scale_init",         type=float, default=0.10)

    # v5 — new FCN loss hyperparameters
    parser.add_argument("--fcn_gate_entropy_lambda",      type=float, default=0.01,
                        help="[v5] Entropy bonus on gate — prevents gate collapse")
    parser.add_argument("--fcn_delta_supervision_lambda", type=float, default=0.10,
                        help="[v5] Delta supervision weight on base-wrong windows")
    parser.add_argument("--uncertainty_sampler",          action="store_true",
                        help="[v5] Up-weight low-confidence windows in P2 sampler")

    # Phase 3
    parser.add_argument("--skip_phase3",       action="store_true")
    parser.add_argument("--finetune_epochs",   type=int,   default=15)
    parser.add_argument("--finetune_lr",       type=float, default=1e-5)
    parser.add_argument("--finetune_patience", type=int,   default=7)
    # v5 — differential LR
    parser.add_argument("--p3_base_lr_scale",  type=float, default=0.1,
                        help="[v5] LR multiplier for base params in P3 (FCN gets full finetune_lr)")

    args = parser.parse_args()

    run_name  = "EEGNet_FCN_ds004504_v5"
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
    logger.info(f"FCN scale_init  : {args.fcn_scale_init}")
    logger.info(f"FCN consistency : {args.fcn_consistency_lambda}  [v5: directional]")
    logger.info(f"FCN gate_lambda : {args.fcn_gate_lambda}")
    logger.info(f"FCN gate_entropy_lambda      : {args.fcn_gate_entropy_lambda}  [v5 new]")
    logger.info(f"FCN delta_supervision_lambda : {args.fcn_delta_supervision_lambda}  [v5 new]")
    logger.info(f"Uncertainty sampler (P2)     : {args.uncertainty_sampler}  [v5 new]")
    logger.info(f"P3 base_lr_scale             : {args.p3_base_lr_scale}  [v5 new]")
    logger.info(f"Finetune lr     : {args.finetune_lr}")

    if not train_dir.exists() or not eval_dir.exists():
        raise FileNotFoundError(f"train/ or eval/ not found under {data_dir}")

    set_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    logger.info(f"Device: {device} | AMP: {use_amp}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Datasets — both start training=False; phase functions toggle as needed
    train_ds = ShardedNpyDataset(train_dir, cache_size=args.cache_size,
                                  logger=logger, training=False, fs=args.target_fs)
    eval_ds  = ShardedNpyDataset(eval_dir,  cache_size=args.cache_size,
                                  logger=logger, training=False, fs=args.target_fs)

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
        train_ds, batch_size=args.batch_size,
        sampler=build_subject_balanced_sampler(train_ds),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), collate_fn=collate_with_meta,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_meta,
    )

    model = EEGNetFCN(
        n_channels=in_channels, fs=args.target_fs, n_classes=2,
        emb_dim=128, scale_init=args.fcn_scale_init,
    ).to(device)

    params = model.param_summary()
    logger.info(f"Base params : {params['base_params']:,}")
    logger.info(f"FCN  params : {params['fcn_params']:,}")
    logger.info(f"Total params: {params['total_params']:,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing,
    )

    # ── Training ──────────────────────────────────────────────
    t0 = time.time()

    best_p1 = train_phase1(
        model, train_ds, train_loader, eval_loader,
        device, criterion, args, logger, use_amp,
    )
    best_p2 = train_phase2(
        model, train_ds, train_loader, eval_loader,
        device, criterion, class_weights, args, logger, use_amp,
    )

    best_p3 = None
    if not args.skip_phase3:
        best_p3 = train_phase3(
            model, train_ds, train_loader, eval_loader,
            device, criterion, class_weights, args, logger, use_amp,
        )

    train_time = time.time() - t0
    logger.info(f"Total training time: {train_time:.2f}s")

    # ── Final evaluation ──────────────────────────────────────
    logger.info("========== FINAL EVALUATION ==========")
    final_ev = evaluate(
        model, eval_loader, device, criterion, use_amp, phase="hybrid",
        class_weights=class_weights, label_smoothing=args.label_smoothing,
        reg_lambda=args.fcn_reg_lambda,
        consistency_lambda=args.fcn_consistency_lambda,
        gate_lambda=args.fcn_gate_lambda,
        gate_entropy_lambda=args.fcn_gate_entropy_lambda,
        delta_supervision_lambda=args.fcn_delta_supervision_lambda,
    )

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

    subj_metrics = subject_level_metrics(
        final_ev["y_true"], final_ev["y_probs"], final_ev["subjects"],
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

    logger.info(f"Subject-level report (n={subj_metrics['n_subjects']} subjects):\n"
                + subj_report_text)
    logger.info("Subject-level confusion matrix:\n" + str(subj_cm))
    logger.info(
        f"Subject-level | acc={subj_metrics['acc']:.4f} "
        f"bal_acc={subj_metrics['bal_acc']:.4f} "
        f"macro_f1={subj_metrics['macro_f1']:.4f}"
    )

    # ── Save ──────────────────────────────────────────────────
    models_dir = fyp_root / "models"
    exp_dir    = fyp_root / "experiments" / run_name
    ensure_dir(models_dir); ensure_dir(exp_dir)

    model_path   = models_dir / f"{run_name}.pt"
    metrics_path = exp_dir / "metrics.json"
    cm_path      = exp_dir / "confusion_matrix.npy"
    subj_cm_path = exp_dir / "confusion_matrix_subject.npy"

    torch.save(
        {
            "base_state_dict": model.base.state_dict(),
            "fcn_state_dict":  model.fcn.state_dict(),
            "model_name":      "EEGNetFCN_v5",
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
        "run_name":        run_name,
        "train_windows":   int(len(train_ds)),
        "eval_windows":    int(len(eval_ds)),
        "train_counts":    class_counts,
        "train_time_sec":  float(train_time),
        "augmentation":    "P1=OFF, P2/P3=ON",
        "label_smoothing": args.label_smoothing,
        "best_p1_bal_acc": float(best_p1),
        "best_p2_bal_acc": float(best_p2),
        "best_p3_bal_acc": float(best_p3) if best_p3 is not None else None,
        "model_params":    params,
        "v5_upgrades": {
            "per_dim_gate":              True,
            "multiscale_base_features":  True,
            "directional_consistency":   True,
            "gate_entropy_regulariser":  True,
            "delta_supervision":         True,
            "differential_lr_p3":        True,
            "uncertainty_sampler_p2":    args.uncertainty_sampler,
        },
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