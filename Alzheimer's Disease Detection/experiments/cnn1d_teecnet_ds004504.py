#!/usr/bin/env python3
"""
Binary AD vs CN classifier on OpenNeuro ds004504 using PRE-BUILT shards (N,C,T).

Pipeline (this script):
- Load existing shards from:
    <fyp_root>/<out_dir>/train  and  <fyp_root>/<out_dir>/eval
  (each containing X_*.npy, y_*.npy, meta_*.csv optional)
- Train 1D CNN + TEECNet hybrid + evaluate

Dependencies:
  pip install numpy pandas torch scikit-learn
  pip install torch-geometric
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

# PyG
import torch_geometric.nn as pyg_nn


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
      X_0.npy, y_0.npy, (optional meta_0.csv)
      X_1.npy, y_1.npy, ...
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
# 4) TEECNet (provided code)
# ============================================================
class LightweightGraphConv(pyg_nn.MessagePassing):
    """
    A lightweight message-passing layer where edge attributes generate
    dynamic weights for messages.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 2):
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, in_channels * out_channels),
            nn.Tanh(),
        )

        self.self_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.self_weight)
        nn.init.zeros_(self.bias)
        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        weight = self.edge_mlp(edge_attr)  # (E, in*out)
        weight = weight.view(-1, self.in_channels, self.out_channels)  # (E, in, out)
        return torch.bmm(x_j.unsqueeze(1), weight).squeeze(1)  # (E,out)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return aggr_out + torch.mm(x, self.self_weight) + self.bias


class TEECNetModule(nn.Module):
    """
    TEECNet block operating on (B, C, F)
    """
    def __init__(
        self,
        num_channels: int,
        feature_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        edge_dim: int = 2,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.graph_convs = nn.ModuleList(
            [LightweightGraphConv(hidden_dim, hidden_dim, edge_dim=edge_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.register_buffer("edge_index", self._build_edge_index())

    def _build_edge_index(self) -> torch.Tensor:
        edges = [[i, j] for i in range(self.num_channels) for j in range(self.num_channels) if i != j]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _compute_edge_attr(self, h: torch.Tensor) -> torch.Tensor:
        batch_size = h.shape[0] // self.num_channels
        h = h.reshape(batch_size, self.num_channels, -1)  # (B, C, H)

        src, dst = self.edge_index
        edge_attrs = []
        for b in range(batch_size):
            src_feat = h[b, src]  # (E,H)
            dst_feat = h[b, dst]  # (E,H)

            cos_sim = F.cosine_similarity(src_feat, dst_feat, dim=1).unsqueeze(1)  # (E,1)
            dist = torch.norm(dst_feat - src_feat, dim=1, keepdim=True)            # (E,1)
            dist = dist / (dist.mean() + 1e-6)

            edge_attrs.append(torch.cat([cos_sim, dist], dim=1))                   # (E,2)

        return torch.cat(edge_attrs, dim=0)  # (B*E, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,F), got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Expected C={self.num_channels}, got {x.shape[1]}")

        B = x.shape[0]
        x_flat = x.reshape(B * self.num_channels, -1)   # (B*C, F)
        h = F.relu(self.input_proj(x_flat))             # (B*C, H)

        edge_attr = self._compute_edge_attr(h)          # (B*E, 2)

        edge_index_batch = torch.cat(
            [self.edge_index + b * self.num_channels for b in range(B)],
            dim=1
        )  # (2, B*E)

        for conv in self.graph_convs:
            h = F.relu(conv(h, edge_index_batch, edge_attr))  # (B*C, H)

        h = self.output_proj(h).reshape(B, self.num_channels, -1)  # (B,C,F)
        return x + h


# ============================================================
# 5) 1D CNN + TEECNet Hybrid Model
# ============================================================
class CNN1D_TEECNet(nn.Module):
    """
    Input from dataloader: (B, C, T)

    CNN extracts window-level embedding:
      (B,C,T) -> (B,F,T') -> pool -> (B,F)

    TEECNet expects node features per channel:
      We replicate (B,F) -> (B,C,F) and apply TEECNet over channel graph.

    Then we fuse across channels and classify.
    """
    def __init__(
        self,
        num_channels: int,
        num_classes: int = 2,
        feature_dim: int = 128,
        teec_hidden: int = 32,
        teec_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim

        # CNN backbone: (B,C,T) -> (B,feature_dim,T')
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.time_pool = nn.AdaptiveAvgPool1d(1)

        self.teecnet = TEECNetModule(
            num_channels=num_channels,
            feature_dim=feature_dim,
            hidden_dim=teec_hidden,
            num_layers=teec_layers,
        )

        self.post_graph = nn.Sequential(
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
        # x: (B,C,T)
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,T), got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Channel mismatch: expected C={self.num_channels}, got {x.shape[1]}")

        h = self.features(x)                  # (B,F,T')
        h = self.time_pool(h).squeeze(-1)     # (B,F)

        # replicate to per-channel node features (B,C,F)
        h = h.unsqueeze(1).expand(-1, self.num_channels, -1).contiguous()

        h = self.teecnet(h)                   # (B,C,F)

        # fuse over channels
        h = h.permute(0, 2, 1)                # (B,F,C)
        h = self.post_graph(h)                # (B,256,C)
        h = self.global_pool(h).squeeze(-1)   # (B,256)

        return self.classifier(h)


# ============================================================
# 6) Eval
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
# 7) Main (uses existing shards ONLY)
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

    # Model knobs
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--teec_hidden", type=int, default=32)
    parser.add_argument("--teec_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

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

    run_name = f"cnn1d_teecnet_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    model = CNN1D_TEECNet(
        num_channels=in_channels,
        num_classes=2,
        feature_dim=args.feature_dim,
        teec_hidden=args.teec_hidden,
        teec_layers=args.teec_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: CNN1D_TEECNet | params={n_params:,} | feature_dim={args.feature_dim}")

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
            "model_name": "CNN1D_TEECNet",
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