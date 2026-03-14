#!/usr/bin/env python3
"""
Binary AD vs CN classifier on OpenNeuro ds004504 (v1.0.2) using derivatives EEGLAB .set files.

Pipeline:
- Read derivatives/sub-*/eeg/*.set (preprocessed / denoised)
- Subject-level split (no leakage)
- Windowing into (C,T) samples + z-score
- Save shards (X_*.npy, y_*.npy, meta_*.csv)
- Train Hybrid EEGNet-like Temporal Backbone + TEECNet (PyTorch + PyG) + evaluate

Dependencies:
  pip install numpy pandas torch scikit-learn mne
  pip install torch-geometric
  (plus the correct pyg backends for your CUDA/torch build if needed)
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

import mne

# ---- torch_geometric (PyG) ----
try:
    import torch_geometric.nn as pyg_nn
except Exception as e:
    raise RuntimeError(
        "torch-geometric is not available. Install it first.\n"
        "Example:\n"
        "  pip install torch-geometric\n"
        "If you use CUDA, ensure you install the matching pyg wheels for your torch version.\n"
        f"Original error: {e}"
    )


# ============================================================
# 1) Logging
# ============================================================
def setup_logger(log_dir: Path, script_stem: str) -> logging.Logger:
    """
    Log file name requirement:
      log filename MUST be exactly the same as python file stem.
      e.g., my_script.py -> my_script.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{script_stem}.log"

    logger = logging.getLogger(script_stem)
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
# 5) TEECNet (Improved + Vectorized Edge Attr + Stabilization)
# ============================================================
class LightweightGraphConv(pyg_nn.MessagePassing):
    """
    Lightweight message passing with dynamic edge-conditioned weights.

    Improvements vs your original:
    - LayerNorm before message passing (stabilizes training)
    - Dropout inside the layer
    - Learnable residual scaling (alpha)
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 2, dropout: float = 0.1):
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pre_norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)

        # edge_attr -> dynamic weight matrix (in_channels x out_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, in_channels * out_channels),
            nn.Tanh(),
        )

        self.self_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # Learnable residual scaling (helps max accuracy without exploding)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.self_weight)
        nn.init.zeros_(self.bias)
        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Pre-norm improves stability for deeper stacks
        x_n = self.pre_norm(x)
        out = self.propagate(edge_index, x=x_n, edge_attr=edge_attr)
        # Residual + self connection
        out = out + torch.mm(x_n, self.self_weight) + self.bias
        out = self.dropout(out)
        return x + self.alpha * out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        weight = self.edge_mlp(edge_attr)  # (E, in*out)
        weight = weight.view(-1, self.in_channels, self.out_channels)  # (E, in, out)
        return torch.bmm(x_j.unsqueeze(1), weight).squeeze(1)  # (E, out)


class TEECNetModule(nn.Module):
    """
    TEECNet block operating on (B, C, F):
      B = batch
      C = channels (graph nodes)
      F = feature dim per node

    Improvements vs your original:
    - Vectorized edge attribute computation (MUCH faster)
    - LN + dropout + residual scaling inside convs
    - Optional edge_attr dropout
    """
    def __init__(
        self,
        num_channels: int,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 2,
        dropout: float = 0.15,
        edge_drop: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.edge_drop = float(edge_drop)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.graph_convs = nn.ModuleList(
            [LightweightGraphConv(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )

        self.register_buffer("edge_index", self._build_edge_index())  # (2, E)

    def _build_edge_index(self) -> torch.Tensor:
        edges = [[i, j] for i in range(self.num_channels) for j in range(self.num_channels) if i != j]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _compute_edge_attr_vectorized(self, h_bc_h: torch.Tensor) -> torch.Tensor:
        """
        h_bc_h: (B, C, H)
        returns: edge_attr (B*E, 2)
        """
        B, C, H = h_bc_h.shape
        src, dst = self.edge_index  # (E,), (E,)
        # (B, E, H)
        src_feat = h_bc_h[:, src, :]
        dst_feat = h_bc_h[:, dst, :]

        cos_sim = F.cosine_similarity(src_feat, dst_feat, dim=-1)  # (B, E)
        dist = torch.norm(dst_feat - src_feat, dim=-1)             # (B, E)
        dist = dist / (dist.mean(dim=1, keepdim=True) + 1e-6)      # normalize per sample

        edge_attr = torch.stack([cos_sim, dist], dim=-1)           # (B, E, 2)

        # Optional edge dropout (regularization)
        if self.edge_drop > 0.0 and self.training:
            mask = (torch.rand(edge_attr.shape[:2], device=edge_attr.device) > self.edge_drop).float()
            edge_attr = edge_attr * mask.unsqueeze(-1)

        return edge_attr.reshape(B * edge_attr.shape[1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F)
        returns: (B, C, F)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,F), got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Expected C={self.num_channels}, got {x.shape[1]}")

        B = x.shape[0]

        # (B,C,F) -> (B,C,H)
        h = self.input_proj(x)

        # Build batched edge_index: (2, B*E)
        edge_index_batch = torch.cat(
            [self.edge_index + b * self.num_channels for b in range(B)],
            dim=1
        )

        # (B,C,H) -> (B*E,2)
        edge_attr = self._compute_edge_attr_vectorized(h)

        # Flatten nodes across batch for PyG: (B*C, H)
        h_flat = h.reshape(B * self.num_channels, -1)

        for conv in self.graph_convs:
            h_flat = F.gelu(conv(h_flat, edge_index_batch, edge_attr))

        # (B*C,H) -> (B,C,H) -> (B,C,F)
        h_out = h_flat.reshape(B, self.num_channels, -1)
        corr = self.output_proj(h_out)

        # Residual correction
        return x + corr


# ============================================================
# 6) Hybrid Model: Temporal DW-Conv Backbone + TEECNet
# ============================================================
class TemporalDWBackbone(nn.Module):
    """
    Produces node features per channel.
    Input:  (B, C, T)
    Output: (B, C, F)

    Lightweight and strong:
    - Depthwise temporal conv per channel (groups=C) to extract band-like patterns
    - Pointwise mixing inside each channel-feature group (cheap)
    - Pool over time -> per-channel embedding
    """
    def __init__(
        self,
        in_chans: int,
        feat_dim: int = 32,
        kernel: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.feat_dim = feat_dim

        # Depthwise temporal conv: C -> C*feat_dim (groups=C)
        self.dw_temporal = nn.Conv1d(
            in_channels=in_chans,
            out_channels=in_chans * feat_dim,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=in_chans,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(in_chans * feat_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Cheap pointwise within the expanded space
        self.pw = nn.Conv1d(in_chans * feat_dim, in_chans * feat_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_chans * feat_dim)

        # Global time pooling (adaptive)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        x = self.dw_temporal(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.pool(x).squeeze(-1)  # (B, C*F)
        B = x.shape[0]
        x = x.view(B, self.in_chans, self.feat_dim)  # (B,C,F)
        return x


class HybridEEG_TEECNet(nn.Module):
    """
    Full classifier:
      (B,C,T) -> backbone -> (B,C,F) -> TEECNet -> (B,C,F)
      -> channel pooling -> (B,F) -> head -> (B,2)
    """
    def __init__(
        self,
        in_chans: int,
        feat_dim: int = 32,
        backbone_kernel: int = 64,
        backbone_dropout: float = 0.2,
        teec_hidden: int = 64,
        teec_layers: int = 3,
        teec_dropout: float = 0.15,
        teec_edge_drop: float = 0.0,
        num_classes: int = 2,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = TemporalDWBackbone(
            in_chans=in_chans,
            feat_dim=feat_dim,
            kernel=backbone_kernel,
            dropout=backbone_dropout,
        )

        self.teec = TEECNetModule(
            num_channels=in_chans,
            feature_dim=feat_dim,
            hidden_dim=teec_hidden,
            num_layers=teec_layers,
            dropout=teec_dropout,
            edge_drop=teec_edge_drop,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,T) -> (B,C,F)
        node_feat = self.backbone(x)
        # graph correction
        node_feat = self.teec(node_feat)
        # channel pooling
        feat = node_feat.mean(dim=1)  # (B,F)
        return self.head(feat)


# ============================================================
# 7) Eval
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, criterion):
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


# ============================================================
# 8) Main
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

    # Optional bandpass
    parser.add_argument("--apply_bandpass", action="store_true")
    parser.add_argument("--bandpass_low", type=float, default=0.5)
    parser.add_argument("--bandpass_high", type=float, default=45.0)

    # Split
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Train
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=2)

    # Hybrid hyperparams
    parser.add_argument("--feat_dim", type=int, default=32)
    parser.add_argument("--backbone_kernel", type=int, default=64)
    parser.add_argument("--backbone_dropout", type=float, default=0.2)

    parser.add_argument("--teec_hidden", type=int, default=64)
    parser.add_argument("--teec_layers", type=int, default=3)
    parser.add_argument("--teec_dropout", type=float, default=0.15)
    parser.add_argument("--teec_edge_drop", type=float, default=0.0)

    parser.add_argument("--head_dropout", type=float, default=0.2)

    parser.add_argument("--rebuild_shards", action="store_true")
    parser.add_argument("--tag", type=str, default="ds004504_AD_CN_derivatives")

    args = parser.parse_args()

    # Log file name must equal python file name stem
    script_stem = Path(sys.argv[0]).stem

    fyp_root = Path(args.fyp_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = (fyp_root / args.out_dir).resolve()

    logger = setup_logger(fyp_root / "experiments" / "logs", script_stem)

    logger.info("========== RUN START ==========")
    logger.info(f"Script        : {Path(sys.argv[0]).name}")
    logger.info(f"FYP root      : {fyp_root}")
    logger.info(f"Dataset root  : {dataset_root}")
    logger.info(f"Out dir       : {out_dir}")
    logger.info(f"Tag           : {args.tag}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
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
        stratify=subj_labels
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

    # Class weights
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
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = HybridEEG_TEECNet(
        in_chans=in_channels,
        feat_dim=args.feat_dim,
        backbone_kernel=args.backbone_kernel,
        backbone_dropout=args.backbone_dropout,
        teec_hidden=args.teec_hidden,
        teec_layers=args.teec_layers,
        teec_dropout=args.teec_dropout,
        teec_edge_drop=args.teec_edge_drop,
        head_dropout=args.head_dropout,
        num_classes=2,
    ).to(device)

    n_params = count_parameters(model)
    logger.info(f"Model: HybridEEG_TEECNet | trainable_params={n_params}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    logger.info("========== TRAINING ==========")
    best_val_balacc = -1.0
    best_state = None
    patience_left = args.patience

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)   # (B,C,T)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            run_loss += loss.item() * yb.size(0)
            run_correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = run_loss / max(total, 1)
        train_acc = run_correct / max(total, 1)

        val_loss, val_acc, val_balacc, val_macro_f1, _, _ = evaluate(model, eval_loader, device, criterion)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
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
                logger.info("Early stopping triggered.")
                break

    train_time = time.time() - start_time
    logger.info(f"Training time (sec): {train_time:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model (best_val_balacc={best_val_balacc:.4f})")

    logger.info("========== EVALUATION ==========")
    val_loss, val_acc, val_balacc, val_macro_f1, y_true, y_pred = evaluate(model, eval_loader, device, criterion)

    report_text = classification_report(y_true, y_pred, digits=4, target_names=["CN(0)", "AD(1)"])
    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True, target_names=["CN(0)", "AD(1)"])
    cm = confusion_matrix(y_true, y_pred)

    logger.info("Classification report:\n" + report_text)
    logger.info("Confusion matrix:\n" + str(cm))

    # Save outputs (use timestamped run dir; log file name stays fixed per requirement)
    run_id = f"{script_stem}_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = fyp_root / "experiments" / run_id
    ensure_dir(exp_dir)
    models_dir = fyp_root / "models"
    ensure_dir(models_dir)

    model_path = models_dir / f"{run_id}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "HybridEEG_TEECNet",
            "trainable_params": int(n_params),
            "in_channels": in_channels,
            "win_len": win_len,
            "num_classes": 2,
            "seed": args.seed,
            "dataset_root": str(dataset_root),
            "out_dir": str(out_dir),
            "tag": args.tag,
            "best_val_balanced_acc": float(best_val_balacc),
            "config": vars(args),
        },
        model_path,
    )

    metrics = {
        "run_id": run_id,
        "tag": args.tag,
        "train_time_sec": float(train_time),
        "train_windows": int(len(train_ds)),
        "eval_windows": int(len(eval_ds)),
        "final_val_loss": float(val_loss),
        "final_val_acc": float(val_acc),
        "final_val_balanced_acc": float(val_balacc),
        "final_val_macro_f1": float(val_macro_f1),
        "best_val_balanced_acc": float(best_val_balacc),
        "trainable_params": int(n_params),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(exp_dir / "confusion_matrix.npy", cm)

    logger.info(f"Saved model   : {model_path}")
    logger.info(f"Saved metrics : {exp_dir / 'metrics.json'}")
    logger.info("========== RUN END ==========")


if __name__ == "__main__":
    main()