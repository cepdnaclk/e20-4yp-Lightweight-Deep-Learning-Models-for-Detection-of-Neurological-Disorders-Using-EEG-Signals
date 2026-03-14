#!/usr/bin/env python3
"""
Binary AD vs CN classifier on OpenNeuro ds004504 (v1.0.2) using derivatives EEGLAB .set files.

v5 improvements over v4:
  - 5-fold stratified subject-level cross-validation
  - Subject-level majority voting evaluation
  - Mixup augmentation (alpha=0.4) in training loop
  - Scaled-up model (~250-400K params):
      node_emb_dim: 32→64, gat_hidden: 64→128, gat_heads: 4→8, temporal_kernel: 64→128
  - Much stronger regularization:
      frontend_dropout: 0.2→0.5, gnn_dropout: 0.3→0.5,
      classifier_dropout: 0.4→0.5, teecn_dropout: 0.1→0.3,
      weight_decay: 0.01→0.05, label_smoothing: 0.1
  - LR warmup (5 epochs) + ReduceLROnPlateau
  - Stronger data augmentation: time masking, frequency-band zeroing
  - Reuses v4 shards (no rebuild necessary)

Pipeline:
  - Load pre-built shards from v4 out_dir (train + eval combined)
  - 5-fold subject-level stratified CV
  - Train EEGNet-GNN-TEECN (PyTorch + PyG) + evaluate per fold
  - Report per-fold & mean±std metrics + subject-level majority voting accuracy
"""

import sys
import json
import time
import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections import OrderedDict, defaultdict, Counter
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import mne

# PyG (Graph Neural Network)
from torch_geometric.nn import GATv2Conv, global_mean_pool


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


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def to_nct(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got {X.shape}")
    return X


# ============================================================
# 3) Unified Sharded Dataset with Augmentation (loads all shards)
# ============================================================
class UnifiedShardedDataset(Dataset):
    """
    Loads ALL shards from multiple split directories (train + eval)
    and exposes a single flat index. Also loads metadata for
    subject-level majority voting.
    """
    def __init__(
        self,
        shard_dirs: List[Path],
        cache_size: int,
        logger: logging.Logger,
        augment: bool = False,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        channel_drop_prob: float = 0.1,
        time_mask_max: int = 100,
    ):
        self.logger = logger
        self.cache_size = cache_size
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.channel_drop_prob = channel_drop_prob
        self.time_mask_max = time_mask_max
        self._x_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

        self.shards = []
        self.shard_offsets = []
        self.subjects = []       # per-window subject ID
        self.all_labels = []     # per-window label
        total = 0

        for shard_dir in shard_dirs:
            x_paths = sorted(shard_dir.glob("X_*.npy"))
            for xp in x_paths:
                sid_str = xp.stem.split("_")[1]
                yp = shard_dir / f"y_{sid_str}.npy"
                mp = shard_dir / f"meta_{sid_str}.csv"
                if not yp.exists():
                    raise FileNotFoundError(f"Missing {yp.name} for {xp.name}")

                x_mmap = to_nct(np.load(xp, mmap_mode="r"))
                y_arr = np.load(yp, mmap_mode="r")
                n = int(x_mmap.shape[0])

                # Load metadata for subject info
                meta_df = pd.read_csv(mp)
                assert len(meta_df) == n, f"Metadata row count mismatch: {mp.name} has {len(meta_df)} vs {n}"

                self.shards.append({
                    "x_path": xp, "y_path": yp, "n": n,
                    "subjects": meta_df["subject"].values.tolist(),
                    "labels": y_arr[:].tolist(),
                })
                self.shard_offsets.append(total)
                self.subjects.extend(meta_df["subject"].values.tolist())
                self.all_labels.extend(y_arr[:].tolist())
                total += n

                logger.info(f"[load] shard {shard_dir.name}/{xp.name}: n={n}")

        self.total_len = total
        logger.info(f"Unified dataset: {self.total_len} windows, {len(set(self.subjects))} subjects")

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

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to a single (C, T) tensor."""
        # 1) Gaussian noise injection
        x = x + torch.randn_like(x) * self.noise_std

        # 2) Random amplitude scaling per channel
        lo, hi = self.scale_range
        scale = lo + (hi - lo) * torch.rand(x.shape[0], 1, device=x.device)
        x = x * scale

        # 3) Random channel dropout (zero out entire channels)
        if self.channel_drop_prob > 0:
            mask = torch.rand(x.shape[0], 1, device=x.device) > self.channel_drop_prob
            x = x * mask.float()

        # 4) Random time shift (circular shift by up to 50 samples)
        max_shift = min(50, x.shape[1] // 10)
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift != 0:
            x = torch.roll(x, shifts=int(shift), dims=1)

        # 5) Time masking (zero out a random segment) — NEW in v5
        if self.time_mask_max > 0:
            T = x.shape[1]
            mask_len = torch.randint(0, min(self.time_mask_max, T // 4) + 1, (1,)).item()
            if mask_len > 0:
                start = torch.randint(0, T - mask_len, (1,)).item()
                x[:, start:start + mask_len] = 0

        # 6) Random frequency-band zeroing (zero out random channel) — NEW in v5
        if torch.rand(1).item() < 0.1:
            ch_to_zero = torch.randint(0, x.shape[0], (1,)).item()
            x[ch_to_zero, :] = 0

        return x

    def __getitem__(self, idx: int):
        shard_id, local_i = self._locate(idx)
        shard = self.shards[shard_id]
        x_mmap = self._get_x_mmap(shard["x_path"])
        y_mmap = np.load(shard["y_path"], mmap_mode="r")

        x = x_mmap[local_i].astype(np.float32)  # (C,T)
        y = int(y_mmap[local_i])
        x = torch.from_numpy(x)

        if self.augment:
            x = self._apply_augmentation(x)

        return x, torch.tensor(y, dtype=torch.long)

    def get_subject(self, idx: int) -> str:
        """Return the subject ID for a given index."""
        return self.subjects[idx]

    def get_unique_subjects(self) -> List[str]:
        return sorted(set(self.subjects))

    def get_subject_label_map(self) -> Dict[str, int]:
        """Return a dict mapping subject_id -> label."""
        result = {}
        for subj, label in zip(self.subjects, self.all_labels):
            result[subj] = int(label)
        return result


# ============================================================
# 4) Model: EEGNet -> GNN -> TEECN (v5 — scaled up)
# ============================================================

@torch.no_grad()
def pearson_topk_graph_batch(
    x: torch.Tensor,
    topk: int = 8,
    add_self_loops: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a batched graph from raw EEG windows using Pearson correlation.
    """
    assert x.dim() == 3, "x must be (B, C, T)"
    B, C, T = x.shape
    device = x.device

    batch_vec = torch.arange(B, device=device).repeat_interleave(C)

    edge_src_all, edge_dst_all, edge_w_all = [], [], []

    for b in range(B):
        xb = x[b]  # (C, T)
        xb = xb - xb.mean(dim=1, keepdim=True)
        xb = xb / (xb.std(dim=1, keepdim=True) + eps)

        corr = (xb @ xb.t()) / max(T - 1, 1)
        corr = torch.clamp(corr, -1.0, 1.0)

        score = corr.abs()
        if not add_self_loops:
            score.fill_diagonal_(-math.inf)

        k = min(topk, C - (0 if add_self_loops else 1))
        vals, idx = torch.topk(score, k=k, dim=1)

        src = torch.arange(C, device=device).unsqueeze(1).expand(C, k).reshape(-1)
        dst = idx.reshape(-1)
        w = corr[src, dst].reshape(-1)

        offset = b * C
        edge_src_all.append(src + offset)
        edge_dst_all.append(dst + offset)
        edge_w_all.append(w)

        if add_self_loops:
            ii = torch.arange(C, device=device)
            edge_src_all.append(ii + offset)
            edge_dst_all.append(ii + offset)
            edge_w_all.append(torch.ones(C, device=device))

    edge_src = torch.cat(edge_src_all, dim=0)
    edge_dst = torch.cat(edge_dst_all, dim=0)
    edge_w = torch.cat(edge_w_all, dim=0)

    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    edge_attr = edge_w.unsqueeze(-1)
    return edge_index, edge_attr, batch_vec


class EEGNetTemporalFrontend(nn.Module):
    """
    EEGNet-inspired temporal frontend with spatial filter (v5 — scaled up).
    Input : (B, C, T)
    Output: (B, C, F)
    """
    def __init__(
        self,
        in_ch: int,
        emb_dim: int = 64,            # v5: 32→64
        temporal_kernel: int = 128,     # v5: 64→128
        depth_multiplier: int = 2,
        dropout: float = 0.5,          # v5: 0.2→0.5
        pool: int = 4,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.emb_dim = emb_dim

        # Depthwise temporal conv per channel
        self.depthwise_temporal = nn.Conv1d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
            groups=in_ch,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(in_ch)

        # Spatial depthwise filter
        self.spatial = nn.Conv1d(
            in_channels=in_ch,
            out_channels=in_ch * depth_multiplier,
            kernel_size=1,
            groups=in_ch,
            bias=False,
        )
        self.bn_spatial = nn.BatchNorm1d(in_ch * depth_multiplier)
        self.spatial_pool = nn.AvgPool1d(kernel_size=pool, stride=pool)

        # Separable conv
        D = depth_multiplier
        self.separable_depthwise = nn.Conv1d(
            in_channels=in_ch * D,
            out_channels=in_ch * D,
            kernel_size=16,
            padding=8,
            groups=in_ch * D,
            bias=False,
        )
        self.separable_pointwise = nn.Conv1d(
            in_channels=in_ch * D,
            out_channels=in_ch,  # back to in_ch
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(in_ch)
        self.pool2 = nn.AvgPool1d(kernel_size=pool, stride=pool)

        # Pointwise to emb_dim per channel (operate on (B*C,1,T))
        self.pointwise = nn.Conv1d(1, emb_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C == self.in_ch, f"Expected {self.in_ch} channels, got {C}"

        # Block 1: Temporal depthwise + spatial filter
        y = self.depthwise_temporal(x)
        y = self.bn1(y)
        y = F.elu(y)

        y = self.spatial(y)
        y = self.bn_spatial(y)
        y = F.elu(y)
        y = self.spatial_pool(y)
        y = self.dropout(y)

        # Block 2: Separable conv
        y = self.separable_depthwise(y)
        y = self.separable_pointwise(y)
        y = self.bn2(y)
        y = F.elu(y)
        y = self.pool2(y)
        y = self.dropout(y)

        # Pointwise embedding per channel
        T_out = y.shape[-1]
        y = y.reshape(B * C, 1, T_out)
        y = self.pointwise(y)
        y = self.bn3(y)
        y = F.elu(y)

        y = y.mean(dim=-1)                # (B*C, F)
        y = y.reshape(B, C, self.emb_dim)  # (B, C, F)
        return y


class TEECNResidual(nn.Module):
    """
    TEECN-style residual correction on graph embedding (order p=2):
      z' = z + alpha1 * proj1(tanh(W1 z)) + alpha2 * proj2(tanh(W2 z)^2)
    """
    def __init__(self, dim: int, hidden: Optional[int] = None, dropout: float = 0.3):
        super().__init__()
        h = hidden or dim
        self.fc1 = nn.Linear(dim, h)
        self.fc2 = nn.Linear(dim, h)
        self.proj1 = nn.Linear(h, dim)
        self.proj2 = nn.Linear(h, dim)

        self.alpha1 = nn.Parameter(torch.ones(dim))
        self.alpha2 = nn.Parameter(torch.ones(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a1 = torch.tanh(self.fc1(z))
        a2 = torch.tanh(self.fc2(z))
        term1 = self.proj1(self.dropout(a1))
        term2 = self.proj2(self.dropout(a2 * a2))
        return z + term1 * self.alpha1 + term2 * self.alpha2


@dataclass
class EEGNetGNNTEECNConfig:
    in_channels: int
    num_classes: int = 2
    node_emb_dim: int = 64          # v5: 32→64
    temporal_kernel: int = 128       # v5: 64→128
    temporal_pool: int = 4
    topk: int = 8
    gat_hidden: int = 128           # v5: 64→128
    gat_heads: int = 8              # v5: 4→8
    gnn_layers: int = 2
    frontend_dropout: float = 0.5   # v5: 0.2→0.5
    gnn_dropout: float = 0.5        # v5: 0.3→0.5
    classifier_dropout: float = 0.5  # v5: 0.4→0.5
    teecn_dropout: float = 0.3      # v5: 0.1→0.3
    graph_self_loops: bool = False


class EEGNetGNNTEECN(nn.Module):
    def __init__(self, cfg: EEGNetGNNTEECNConfig):
        super().__init__()
        self.cfg = cfg

        self.frontend = EEGNetTemporalFrontend(
            in_ch=cfg.in_channels,
            emb_dim=cfg.node_emb_dim,
            temporal_kernel=cfg.temporal_kernel,
            dropout=cfg.frontend_dropout,
            pool=cfg.temporal_pool,
        )

        self.gnn1 = GATv2Conv(
            in_channels=cfg.node_emb_dim,
            out_channels=cfg.gat_hidden,
            heads=cfg.gat_heads,
            concat=True,
            edge_dim=1,
            dropout=cfg.gnn_dropout,
        )
        gnn1_out = cfg.gat_hidden * cfg.gat_heads

        if cfg.gnn_layers == 1:
            self.gnn2 = None
            gnn_out_dim = gnn1_out
        else:
            self.gnn2 = GATv2Conv(
                in_channels=gnn1_out,
                out_channels=cfg.gat_hidden,
                heads=1,
                concat=True,
                edge_dim=1,
                dropout=cfg.gnn_dropout,
            )
            gnn_out_dim = cfg.gat_hidden

        self.teecn = TEECNResidual(dim=gnn_out_dim, hidden=gnn_out_dim, dropout=cfg.teecn_dropout)

        # v5: LayerNorm + deeper classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(gnn_out_dim),
            nn.Linear(gnn_out_dim, gnn_out_dim),
            nn.ReLU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(gnn_out_dim, gnn_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.classifier_dropout * 0.5),
            nn.Linear(gnn_out_dim // 2, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,C,T)"""
        B, C, T = x.shape
        assert C == self.cfg.in_channels, f"Expected {self.cfg.in_channels} channels, got {C}"

        edge_index, edge_attr, batch_vec = pearson_topk_graph_batch(
            x, topk=self.cfg.topk, add_self_loops=self.cfg.graph_self_loops
        )

        node_feats = self.frontend(x).reshape(B * C, -1)  # (B*C,F)

        h = self.gnn1(node_feats, edge_index, edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.cfg.gnn_dropout, training=self.training)

        if self.gnn2 is not None:
            h = self.gnn2(h, edge_index, edge_attr)
            h = F.elu(h)
            h = F.dropout(h, p=self.cfg.gnn_dropout, training=self.training)

        g = global_mean_pool(h, batch_vec)  # (B,D)
        g = self.teecn(g)
        logits = self.classifier(g)
        return logits


# ============================================================
# 5) Mixup
# ============================================================
def mixup_data(x, y, alpha=0.4):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 6) Eval (window-level + subject-level)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """Window-level evaluation."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    y_true, y_pred = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * yb.size(0)
        total_correct += (preds == yb).sum().item()
        total += yb.size(0)

        y_true.append(yb.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
    return avg_loss, acc, bal_acc, macro_f1, y_true, y_pred


def subject_level_voting(
    y_true_windows: np.ndarray,
    y_pred_windows: np.ndarray,
    subject_ids: List[str],
) -> Tuple[float, float, float, Dict[str, int], Dict[str, int]]:
    """
    Majority voting per subject from window-level predictions.

    Returns:
        subject_acc, subject_balacc, subject_f1, subj_true, subj_pred
    """
    subj_preds = defaultdict(list)
    subj_true = {}
    for y_t, y_p, sid in zip(y_true_windows, y_pred_windows, subject_ids):
        subj_preds[sid].append(int(y_p))
        subj_true[sid] = int(y_t)

    subj_final_pred = {}
    for sid, preds in subj_preds.items():
        # Majority vote
        counter = Counter(preds)
        subj_final_pred[sid] = counter.most_common(1)[0][0]

    subjects_sorted = sorted(subj_true.keys())
    y_t_subj = np.array([subj_true[s] for s in subjects_sorted])
    y_p_subj = np.array([subj_final_pred[s] for s in subjects_sorted])

    subj_acc = float(np.mean(y_t_subj == y_p_subj))
    subj_balacc = float(balanced_accuracy_score(y_t_subj, y_p_subj)) if len(y_t_subj) > 0 else 0.0
    subj_f1 = float(f1_score(y_t_subj, y_p_subj, average="macro")) if len(y_t_subj) > 0 else 0.0

    return subj_acc, subj_balacc, subj_f1, subj_true, subj_final_pred


# ============================================================
# 7) LR Warmup Scheduler
# ============================================================
class WarmupReduceLROnPlateau:
    """Linear warmup for warmup_epochs, then ReduceLROnPlateau."""
    def __init__(self, optimizer, warmup_epochs, base_lr, after_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.after_scheduler = after_scheduler
        self.current_epoch = 0

    def step(self, metric=None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            if metric is not None:
                self.after_scheduler.step(metric)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
# 8) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fyp_root", type=str, default=str(Path.home() / "FYP"))
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/ds004504_ad_cn_shards_v4")

    # Split
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Train — v5 defaults
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)    # v5: 1e-2→5e-2
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)             # v5: 12→15
    parser.add_argument("--warmup_epochs", type=int, default=5)         # v5: new
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=4)

    # Label smoothing — v5 new
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Mixup — v5 new
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--no_mixup", action="store_true", help="Disable mixup")

    # Augmentation
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--scale_lo", type=float, default=0.8)
    parser.add_argument("--scale_hi", type=float, default=1.2)
    parser.add_argument("--channel_drop_prob", type=float, default=0.1)
    parser.add_argument("--time_mask_max", type=int, default=100)       # v5: new

    # EEGNet-GNN-TEECN hyperparams — v5 (scaled up)
    parser.add_argument("--node_emb_dim", type=int, default=64)          # v5: 32→64
    parser.add_argument("--temporal_kernel", type=int, default=128)       # v5: 64→128
    parser.add_argument("--temporal_pool", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--gat_hidden", type=int, default=128)           # v5: 64→128
    parser.add_argument("--gat_heads", type=int, default=8)              # v5: 4→8
    parser.add_argument("--gnn_layers", type=int, default=2, choices=[1, 2])
    parser.add_argument("--frontend_dropout", type=float, default=0.5)   # v5: 0.2→0.5
    parser.add_argument("--gnn_dropout", type=float, default=0.5)        # v5: 0.3→0.5
    parser.add_argument("--classifier_dropout", type=float, default=0.5) # v5: 0.4→0.5
    parser.add_argument("--teecn_dropout", type=float, default=0.3)      # v5: 0.1→0.3
    parser.add_argument("--graph_self_loops", action="store_true")

    args = parser.parse_args()

    # Handle negation flags
    if args.no_augment:
        args.augment = False
    use_mixup = not args.no_mixup

    fyp_root = Path(args.fyp_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = (fyp_root / args.out_dir).resolve()

    run_name = "EEGNet_Teecnet_4504_v5"
    logger = setup_logger(fyp_root / "experiments" / "logs", run_name)

    logger.info("========== RUN START (v5) ==========")
    logger.info(f"FYP root     : {fyp_root}")
    logger.info(f"Dataset root : {dataset_root}")
    logger.info(f"Shard dir    : {out_dir}")

    # Log v5 improvements
    logger.info("--- v5 Improvements over v4 ---")
    logger.info(f"  K-Fold CV        : {args.n_folds}-fold stratified subject-level")
    logger.info(f"  LR               : {args.lr} (with {args.warmup_epochs}-epoch warmup)")
    logger.info(f"  Optimizer        : AdamW (weight_decay={args.weight_decay})")
    logger.info(f"  Scheduler        : Warmup + ReduceLROnPlateau")
    logger.info(f"  Grad clipping    : max_norm={args.max_grad_norm}")
    logger.info(f"  Epochs/fold      : {args.epochs}")
    logger.info(f"  Patience         : {args.patience}")
    logger.info(f"  Label smoothing  : {args.label_smoothing}")
    logger.info(f"  Mixup            : {use_mixup} (alpha={args.mixup_alpha})")
    logger.info(f"  Augmentation     : {args.augment} (noise_std={args.noise_std}, scale=[{args.scale_lo},{args.scale_hi}], ch_drop={args.channel_drop_prob}, time_mask={args.time_mask_max})")
    logger.info(f"  Model params     : node_emb={args.node_emb_dim}, gat_hidden={args.gat_hidden}, gat_heads={args.gat_heads}, temporal_kernel={args.temporal_kernel}")
    logger.info(f"  Dropout          : frontend={args.frontend_dropout}, gnn={args.gnn_dropout}, classifier={args.classifier_dropout}, teecn={args.teecn_dropout}")
    logger.info("--- End v5 Improvements ---")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========== Load ALL shards (train + eval from v4) ==========
    logger.info("========== LOADING ALL SHARDS ==========")
    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"

    if not train_dir.exists() or not eval_dir.exists():
        raise RuntimeError(f"Shard directories not found. Expected {train_dir} and {eval_dir}. Run v4 first.")

    # Load without augmentation first (augmentation toggled per fold)
    full_ds = UnifiedShardedDataset(
        shard_dirs=[train_dir, eval_dir],
        cache_size=args.cache_size,
        logger=logger,
        augment=False,  # will toggle per fold
        noise_std=args.noise_std,
        scale_range=(args.scale_lo, args.scale_hi),
        channel_drop_prob=args.channel_drop_prob,
        time_mask_max=args.time_mask_max,
    )

    x0, y0 = full_ds[0]
    in_channels = int(x0.shape[0])
    win_len = int(x0.shape[1])
    logger.info(f"Input shape: (C={in_channels}, T={win_len}) | Example y={int(y0)}")
    logger.info(f"Total windows: {len(full_ds)}")

    # ========== Subject-level K-fold CV ==========
    all_subjects = full_ds.get_unique_subjects()
    subj_label_map = full_ds.get_subject_label_map()
    subj_labels = np.array([subj_label_map[s] for s in all_subjects])
    logger.info(f"Total subjects: {len(all_subjects)} (AD={int(sum(subj_labels))}, CN={int(sum(1-subj_labels))})")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_results = []

    for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(skf.split(all_subjects, subj_labels)):
        fold_num = fold_idx + 1
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"  FOLD {fold_num}/{args.n_folds}")
        logger.info(f"{'='*60}")

        train_subjects = set(all_subjects[i] for i in train_subj_idx)
        val_subjects = set(all_subjects[i] for i in val_subj_idx)

        logger.info(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
        logger.info(f"Val subjects   ({len(val_subjects)}): {sorted(val_subjects)}")

        # Build index masks
        train_indices = [i for i in range(len(full_ds)) if full_ds.get_subject(i) in train_subjects]
        val_indices = [i for i in range(len(full_ds)) if full_ds.get_subject(i) in val_subjects]

        logger.info(f"Train windows: {len(train_indices)} | Val windows: {len(val_indices)}")

        # Create subsets — augmentation only for training
        # We toggle augmentation via a wrapper
        class AugmentedSubset(Dataset):
            def __init__(self, base_ds, indices, augment):
                self.base_ds = base_ds
                self.indices = indices
                self.augment = augment

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                x, y = self.base_ds[real_idx]
                if self.augment:
                    x = self.base_ds._apply_augmentation(x)
                return x, y

        train_subset = AugmentedSubset(full_ds, train_indices, augment=args.augment)
        val_subset = AugmentedSubset(full_ds, val_indices, augment=False)

        # Class weights for this fold
        train_labels = np.array([full_ds.all_labels[i] for i in train_indices], dtype=np.int64)
        classes, counts = np.unique(train_labels, return_counts=True)
        freq = counts / counts.sum()
        weights = 1.0 / (freq + 1e-12)
        weights = weights / weights.sum() * len(classes)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        logger.info(f"Fold {fold_num} class counts: {dict(zip([int(c) for c in classes], [int(n) for n in counts]))}")
        logger.info(f"Fold {fold_num} class weights: {weights.tolist()}")

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )

        # ---- Model init ----
        set_seed(args.seed + fold_idx)  # Different seed per fold for variety

        cfg = EEGNetGNNTEECNConfig(
            in_channels=in_channels,
            num_classes=2,
            node_emb_dim=args.node_emb_dim,
            temporal_kernel=args.temporal_kernel,
            temporal_pool=args.temporal_pool,
            topk=args.topk,
            gat_hidden=args.gat_hidden,
            gat_heads=args.gat_heads,
            gnn_layers=args.gnn_layers,
            frontend_dropout=args.frontend_dropout,
            gnn_dropout=args.gnn_dropout,
            classifier_dropout=args.classifier_dropout,
            teecn_dropout=args.teecn_dropout,
            graph_self_loops=args.graph_self_loops,
        )
        model = EEGNetGNNTEECN(cfg).to(device)

        if fold_idx == 0:
            total_params, trainable_params = count_parameters(model)
            logger.info("========== MODEL PARAMETERS ==========")
            logger.info(f"Total parameters     : {total_params:,}")
            logger.info(f"Trainable parameters : {trainable_params:,}")
            logger.info(f"Non-trainable params : {total_params - trainable_params:,}")
            for name, module in model.named_children():
                mod_params = sum(p.numel() for p in module.parameters())
                mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
                logger.info(f"  {name:20s} : {mod_params:>10,} params ({mod_train:,} trainable)")
            logger.info("======================================")

        # ---- Optimizer + scheduler ----
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
        scheduler = WarmupReduceLROnPlateau(
            optimizer, warmup_epochs=args.warmup_epochs,
            base_lr=args.lr, after_scheduler=plateau_scheduler,
        )

        # ---- Training loop ----
        logger.info(f"========== TRAINING FOLD {fold_num} ==========")
        best_val_balacc = -1.0
        best_state = None
        patience_left = args.patience

        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            run_loss, run_correct, total = 0.0, 0, 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # v5: Mixup
                if use_mixup and args.mixup_alpha > 0:
                    xb_mixed, y_a, y_b, lam = mixup_data(xb, yb, alpha=args.mixup_alpha)
                    logits = model(xb_mixed)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    preds = torch.argmax(logits, dim=1)
                    # For accuracy, count against original labels
                    run_correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    preds = torch.argmax(logits, dim=1)
                    run_correct += (preds == yb).sum().item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                run_loss += loss.item() * yb.size(0)
                total += yb.size(0)

            train_loss = run_loss / max(total, 1)
            train_acc = run_correct / max(total, 1)

            val_loss, val_acc, val_balacc, val_macro_f1, _, _ = evaluate(model, val_loader, device, criterion)

            # Step scheduler with val_balacc
            scheduler.step(metric=val_balacc)

            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Fold {fold_num} Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"val_balacc={val_balacc:.4f} val_macroF1={val_macro_f1:.4f} | lr={lr_now:.2e}"
            )

            if val_balacc > best_val_balacc + 1e-6:
                best_val_balacc = val_balacc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(f"Fold {fold_num}: Early stopping triggered at epoch {epoch}.")
                    break

        train_time = time.time() - start_time
        logger.info(f"Fold {fold_num} training time (sec): {train_time:.2f}")

        # ---- Restore best model + final eval ----
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info(f"Fold {fold_num}: Restored best model (best_val_balacc={best_val_balacc:.4f})")

        logger.info(f"========== EVALUATION FOLD {fold_num} ==========")
        val_loss, val_acc, val_balacc, val_macro_f1, y_true, y_pred = evaluate(
            model, val_loader, device, criterion
        )

        report_text = classification_report(y_true, y_pred, digits=4, target_names=["CN(0)", "AD(1)"])
        cm = confusion_matrix(y_true, y_pred)

        logger.info(f"Fold {fold_num} Window-level classification report:\n" + report_text)
        logger.info(f"Fold {fold_num} Confusion matrix:\n" + str(cm))

        # ---- Subject-level majority voting ----
        val_subject_ids = [full_ds.get_subject(i) for i in val_indices]
        subj_acc, subj_balacc, subj_f1, subj_true, subj_pred = subject_level_voting(
            y_true, y_pred, val_subject_ids
        )

        logger.info(f"Fold {fold_num} Subject-level results:")
        logger.info(f"  Subject accuracy      : {subj_acc:.4f}")
        logger.info(f"  Subject balanced acc  : {subj_balacc:.4f}")
        logger.info(f"  Subject macro F1      : {subj_f1:.4f}")
        logger.info(f"  Subject predictions   : {dict(subj_pred)}")
        logger.info(f"  Subject true labels   : {dict(subj_true)}")

        fold_result = {
            "fold": fold_num,
            "train_time_sec": float(train_time),
            "window_val_acc": float(val_acc),
            "window_val_balacc": float(val_balacc),
            "window_val_macroF1": float(val_macro_f1),
            "subject_acc": float(subj_acc),
            "subject_balacc": float(subj_balacc),
            "subject_macroF1": float(subj_f1),
            "best_val_balacc": float(best_val_balacc),
            "confusion_matrix": cm.tolist(),
        }
        fold_results.append(fold_result)

        # Save fold model
        models_dir = fyp_root / "models"
        ensure_dir(models_dir)
        model_path = models_dir / f"{run_name}_fold{fold_num}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": "EEGNetGNNTEECN_v5",
                "fold": fold_num,
                "in_channels": in_channels,
                "win_len": win_len,
                "num_classes": 2,
                "seed": args.seed,
                "best_val_balanced_acc": float(best_val_balacc),
                "config": vars(args),
            },
            model_path,
        )
        logger.info(f"Fold {fold_num}: Saved model to {model_path}")

    # ========== AGGREGATE RESULTS ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("  AGGREGATE RESULTS ACROSS ALL FOLDS")
    logger.info("=" * 60)

    # Window-level
    win_accs = [r["window_val_acc"] for r in fold_results]
    win_balaccs = [r["window_val_balacc"] for r in fold_results]
    win_f1s = [r["window_val_macroF1"] for r in fold_results]

    logger.info(f"Window-level accuracy   : {np.mean(win_accs):.4f} ± {np.std(win_accs):.4f}")
    logger.info(f"Window-level bal. acc   : {np.mean(win_balaccs):.4f} ± {np.std(win_balaccs):.4f}")
    logger.info(f"Window-level macro F1   : {np.mean(win_f1s):.4f} ± {np.std(win_f1s):.4f}")

    # Subject-level
    subj_accs = [r["subject_acc"] for r in fold_results]
    subj_balaccs = [r["subject_balacc"] for r in fold_results]
    subj_f1s = [r["subject_macroF1"] for r in fold_results]

    logger.info(f"Subject-level accuracy  : {np.mean(subj_accs):.4f} ± {np.std(subj_accs):.4f}")
    logger.info(f"Subject-level bal. acc  : {np.mean(subj_balaccs):.4f} ± {np.std(subj_balaccs):.4f}")
    logger.info(f"Subject-level macro F1  : {np.mean(subj_f1s):.4f} ± {np.std(subj_f1s):.4f}")

    for r in fold_results:
        logger.info(
            f"  Fold {r['fold']}: win_acc={r['window_val_acc']:.4f} win_balacc={r['window_val_balacc']:.4f} "
            f"subj_acc={r['subject_acc']:.4f} subj_balacc={r['subject_balacc']:.4f}"
        )

    # Save aggregate metrics
    exp_dir = fyp_root / "experiments" / run_name
    ensure_dir(exp_dir)

    metrics = {
        "run_name": run_name,
        "version": "v5",
        "n_folds": args.n_folds,
        "total_subjects": len(all_subjects),
        "total_windows": len(full_ds),
        "total_parameters": total_params if fold_idx == 0 or 'total_params' in dir() else count_parameters(model)[0],
        "fold_results": fold_results,
        "aggregate": {
            "window_acc_mean": float(np.mean(win_accs)),
            "window_acc_std": float(np.std(win_accs)),
            "window_balacc_mean": float(np.mean(win_balaccs)),
            "window_balacc_std": float(np.std(win_balaccs)),
            "window_macroF1_mean": float(np.mean(win_f1s)),
            "window_macroF1_std": float(np.std(win_f1s)),
            "subject_acc_mean": float(np.mean(subj_accs)),
            "subject_acc_std": float(np.std(subj_accs)),
            "subject_balacc_mean": float(np.mean(subj_balaccs)),
            "subject_balacc_std": float(np.std(subj_balaccs)),
            "subject_macroF1_mean": float(np.mean(subj_f1s)),
            "subject_macroF1_std": float(np.std(subj_f1s)),
        },
        "config": vars(args),
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved aggregate metrics: {exp_dir / 'metrics.json'}")
    logger.info("========== RUN END (v5) ==========")


if __name__ == "__main__":
    main()
