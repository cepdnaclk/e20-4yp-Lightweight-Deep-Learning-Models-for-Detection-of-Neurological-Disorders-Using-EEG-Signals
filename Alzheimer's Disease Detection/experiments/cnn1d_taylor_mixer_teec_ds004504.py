#!/usr/bin/env python3
"""
Binary AD vs CN classifier on OpenNeuro ds004504 using PRE-BUILT shards (N,C,T).

Pipeline:
- Load existing shards from:
    <fyp_root>/<out_dir>/train  and  <fyp_root>/<out_dir>/eval
  (each containing X_*.npy, y_*.npy)
- Train 1D CNN + Mixer-style Taylor-TEEC (NO GNN) + evaluate

Dependencies:
  pip install numpy torch scikit-learn

============================================================
LOG NAMING CHANGE (what you asked)
============================================================
Log files (and run folder/model file prefix) are now based on the
*python filename* automatically:

  <script_name>_<tag>_<timestamp>.log

Example:
  cnn1d_mixer_taylor_teec_ds004504_ds004504_mixer_taylor_20260226_112548.log
============================================================
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score


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


# ============================================================
# 3) Sharded Dataset (loads existing X_*.npy/y_*.npy)
# ============================================================
def to_nct(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got {X.shape}")
    return X


class ShardedNpyDataset(Dataset):
    """
    Expects split_dir contains:
      X_0.npy, y_0.npy
      X_1.npy, y_1.npy
      ...
    where X_k shape = (N,C,T), y_k shape = (N,)
    """
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
# 4) DropPath (stochastic depth)
# ============================================================
class DropPath(nn.Module):
    """
    Stochastic depth: randomly drop residual branch during training.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B,1,1) for (B,C,F)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = torch.floor(rand)  # 0/1
        return x * mask / keep_prob


# ============================================================
# 5) Mixer-style Taylor TEEC (NO GNN, WITH channel mixing)
# ============================================================
class TaylorFeatureCorrector(nn.Module):
    """
    Taylor-series-inspired correction applied per token (channel):
      delta = ProjOut( sum_k alpha_k ⊙ clamp(act(Wk(h)))^k )
      y = x + tanh(gate_vec) ⊙ delta
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        order: int = 3,
        dropout: float = 0.1,
        act: str = "tanh",
        gate_init: float = 0.05,
        clamp_val: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.order = order
        self.clamp_val = float(clamp_val)

        self.proj_in = nn.Linear(feature_dim, hidden_dim)
        self.in_ln = nn.LayerNorm(hidden_dim)

        self.order_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(order)])
        self.order_lns = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(order)])
        self.drop = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.full((order, hidden_dim), 0.1, dtype=torch.float32))
        self.proj_out = nn.Linear(hidden_dim, feature_dim)

        # per-feature gate (F,)
        self.gate = nn.Parameter(torch.full((feature_dim,), float(gate_init), dtype=torch.float32))

        if act == "tanh":
            self.act_fn = torch.tanh
        elif act == "relu":
            self.act_fn = F.relu
        elif act == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError("act must be tanh/relu/gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Fdim = x.shape
        if Fdim != self.feature_dim:
            raise ValueError(f"Feature mismatch: expected {self.feature_dim}, got {Fdim}")

        xf = x.reshape(B * C, Fdim)

        h0 = self.proj_in(xf)
        h0 = self.in_ln(h0)
        h0 = self.act_fn(h0)

        corr = 0.0
        for k in range(1, self.order + 1):
            z = self.order_linears[k - 1](h0)
            z = self.order_lns[k - 1](z)
            z = self.act_fn(z)

            # stabilize polynomial powers
            z = torch.clamp(z, -self.clamp_val, self.clamp_val)
            z = z.pow(k)

            z = z * self.alpha[k - 1]
            z = self.drop(z)
            corr = corr + z

        delta = self.proj_out(corr).reshape(B, C, Fdim)
        gate = torch.tanh(self.gate).view(1, 1, Fdim)
        return x + gate * delta


class ChannelTokenMixer(nn.Module):
    """
    Token-mixing MLP across channels (MLP-Mixer style).
    Operates on (B,C,F) by mixing C for each feature independently:
      x -> (B,F,C) -> MLP over C -> back -> residual add
    """
    def __init__(self, num_channels: int, token_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_channels = num_channels
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, num_channels),
            nn.Dropout(dropout),
        )
        self.pre_ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Fdim = x.shape
        if C != self.num_channels:
            raise ValueError(f"Channel mismatch: expected {self.num_channels}, got {C}")

        y = x.permute(0, 2, 1)  # (B,F,C)
        y = self.pre_ln(y)
        y = self.mlp(y)
        y = y.permute(0, 2, 1)  # (B,C,F)
        return x + y


class MixerTaylorTEEC(nn.Module):
    """
    Combined TEEC block:
      - Token-mixing across channels
      - Taylor feature correction per channel
      - Iterative refinement steps
      - Optional DropPath on correction branch
    """
    def __init__(
        self,
        num_channels: int,
        feature_dim: int,
        token_hidden: int = 64,
        teec_hidden: int = 128,
        teec_order: int = 3,
        teec_dropout: float = 0.1,
        token_dropout: float = 0.1,
        steps: int = 2,
        droppath: float = 0.0,
        act: str = "tanh",
        clamp_val: float = 1.0,
    ):
        super().__init__()
        self.steps = int(max(1, steps))

        self.token_mixer = ChannelTokenMixer(
            num_channels=num_channels,
            token_hidden=token_hidden,
            dropout=token_dropout,
        )
        self.corrector = TaylorFeatureCorrector(
            feature_dim=feature_dim,
            hidden_dim=teec_hidden,
            order=teec_order,
            dropout=teec_dropout,
            act=act,
            gate_init=0.05,
            clamp_val=clamp_val,
        )
        self.dp = DropPath(droppath)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for _ in range(self.steps):
            out = self.token_mixer(out)
            corr = self.corrector(out) - out
            out = out + self.dp(corr)
        return out


# ============================================================
# 6) 1D CNN + MixerTaylorTEEC Hybrid Model
# ============================================================
class CNN1D_MixerTaylorTEEC(nn.Module):
    """
    Input: (B,C,T)

    - Shared per-channel CNN: (B*C,1,T) -> (B*C,F) -> reshape (B,C,F)
    - MixerTaylorTEEC on (B,C,F)
    - Fuse over channels and classify
    """
    def __init__(
        self,
        num_channels: int,
        num_classes: int = 2,
        feature_dim: int = 128,

        token_hidden: int = 64,
        teec_hidden: int = 128,
        teec_order: int = 3,
        teec_steps: int = 2,
        teec_dropout: float = 0.1,
        token_dropout: float = 0.1,
        teec_act: str = "tanh",
        teec_clamp: float = 1.0,
        teec_droppath: float = 0.0,

        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim

        self.ch_features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.time_pool = nn.AdaptiveAvgPool1d(1)

        self.teec = MixerTaylorTEEC(
            num_channels=num_channels,
            feature_dim=feature_dim,
            token_hidden=token_hidden,
            teec_hidden=teec_hidden,
            teec_order=teec_order,
            teec_dropout=teec_dropout,
            token_dropout=token_dropout,
            steps=teec_steps,
            droppath=teec_droppath,
            act=teec_act,
            clamp_val=teec_clamp,
        )

        self.post_feat = nn.Sequential(
            nn.Conv1d(feature_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,T), got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Channel mismatch: expected C={self.num_channels}, got {x.shape[1]}")

        B, C, T = x.shape

        x_ch = x.reshape(B * C, 1, T)
        h = self.ch_features(x_ch)
        h = self.time_pool(h).squeeze(-1)
        h = h.reshape(B, C, self.feature_dim)

        h = self.teec(h)

        h = h.permute(0, 2, 1)     # (B,F,C)
        h = self.post_feat(h)      # (B,256,C)
        h = self.global_pool(h).squeeze(-1)
        return self.classifier(h)


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
    parser.add_argument("--out_dir", type=str, default="data/ds004504_ad_cn_shards")
    parser.add_argument("--tag", type=str, default="ds004504_AD_CN_shards")

    parser.add_argument("--seed", type=int, default=42)

    # Train
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=2)

    # Model basics
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    # TEEC Mixer knobs
    parser.add_argument("--token_hidden", type=int, default=64)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--teec_hidden", type=int, default=128)
    parser.add_argument("--teec_order", type=int, default=3)
    parser.add_argument("--teec_steps", type=int, default=2)
    parser.add_argument("--teec_dropout", type=float, default=0.1)
    parser.add_argument("--teec_act", type=str, default="tanh", choices=["tanh", "relu", "gelu"])
    parser.add_argument("--teec_clamp", type=float, default=1.0)
    parser.add_argument("--teec_droppath", type=float, default=0.0)

    args = parser.parse_args()

    fyp_root = Path(args.fyp_root).resolve()
    out_dir = (fyp_root / args.out_dir).resolve()

    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"
    if not train_dir.exists() or not eval_dir.exists():
        raise FileNotFoundError(
            f"Shard folders not found.\nExpected:\n  {train_dir}\n  {eval_dir}\n"
            f"Check --fyp_root and --out_dir."
        )

    # ========================================================
    # LOG NAME based on PYTHON FILE NAME (your requested change)
    # ========================================================
    script_stem = Path(__file__).stem  # <--- THIS is the change
    run_name = f"{script_stem}_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger = setup_logger(fyp_root / "experiments" / "logs", run_name)

    logger.info("========== RUN START ==========")
    logger.info(f"FYP root : {fyp_root}")
    logger.info(f"Shards   : {out_dir}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("========== LOADING SHARDS ==========")
    train_ds = ShardedNpyDataset(train_dir, cache_size=args.cache_size, logger=logger)
    eval_ds = ShardedNpyDataset(eval_dir, cache_size=args.cache_size, logger=logger)

    x0, y0 = train_ds[0]
    in_channels = int(x0.shape[0])
    win_len = int(x0.shape[1])
    logger.info(f"Example window: C={in_channels}, T={win_len} | y={int(y0)}")

    # Class weights from training labels
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

    model = CNN1D_MixerTaylorTEEC(
        num_channels=in_channels,
        num_classes=2,
        feature_dim=args.feature_dim,

        token_hidden=args.token_hidden,
        token_dropout=args.token_dropout,
        teec_hidden=args.teec_hidden,
        teec_order=args.teec_order,
        teec_steps=args.teec_steps,
        teec_dropout=args.teec_dropout,
        teec_act=args.teec_act,
        teec_clamp=args.teec_clamp,
        teec_droppath=args.teec_droppath,

        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model: CNN1D_MixerTaylorTEEC | params={n_params:,} | "
        f"feature_dim={args.feature_dim} teec_order={args.teec_order} teec_steps={args.teec_steps} "
        f"token_hidden={args.token_hidden} droppath={args.teec_droppath}"
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
            xb = xb.to(device, non_blocking=True)  # (B,C,T)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
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

    exp_dir = fyp_root / "experiments" / run_name
    ensure_dir(exp_dir)
    models_dir = fyp_root / "models"
    ensure_dir(models_dir)

    model_path = models_dir / f"{run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": 2,
            "seed": args.seed,
            "out_dir": str(out_dir),
            "tag": args.tag,
            "best_val_balanced_acc": float(best_val_balacc),
            "config": vars(args),
            "model_name": "CNN1D_MixerTaylorTEEC",
        },
        model_path,
    )

    metrics = {
        "run_name": run_name,
        "tag": args.tag,
        "train_time_sec": float(train_time),
        "train_windows": int(len(train_ds)),
        "eval_windows": int(len(eval_ds)),
        "final_val_loss": float(val_loss),
        "final_val_acc": float(val_acc),
        "final_val_balanced_acc": float(val_balacc),
        "final_val_macro_f1": float(val_macro_f1),
        "best_val_balanced_acc": float(best_val_balacc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "model_params": int(sum(p.numel() for p in model.parameters())),
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(exp_dir / "confusion_matrix.npy", cm)

    logger.info(f"Saved model   : {model_path}")
    logger.info(f"Saved metrics : {exp_dir / 'metrics.json'}")
    logger.info("========== RUN END ==========")


if __name__ == "__main__":
    main()