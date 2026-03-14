#!/usr/bin/env python3
"""
train_eegnet_ds004504.py

Train EEGNet on:
  <fyp_root>/data/ds004504_preprocessed_shards/{train,eval}

Outputs:
  logs   -> <fyp_root>/experiments/logs/train_eegnet_ds004504_<timestamp>.log
  model  -> <fyp_root>/models/train_eegnet_ds004504.pt
  metrics-> <fyp_root>/experiments/train_eegnet_ds004504/metrics.json
"""

import sys
import os
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
# Logging
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
# Dataset
# ============================================================
def to_nct(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected (N,C,T), got {X.shape}")
    return X


class ShardedNpyDataset(Dataset):
    def __init__(self, split_dir: Path, cache_size: int, logger: logging.Logger):
        self.split_dir = split_dir
        self.cache_size = cache_size
        self.logger = logger
        self._x_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

        self.x_paths = sorted(split_dir.glob("X_*.npy"))
        if not self.x_paths:
            raise FileNotFoundError(f"No X_*.npy found in {split_dir}")

        self.shards = []
        self.shard_offsets = []
        total = 0

        self.subjects = []
        self.labels = []

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

        x = x_mmap[local_i].astype(np.float32)    # (C,T)
        y = int(y_mmap[local_i])

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), meta_row


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    xb = torch.stack(xs, dim=0)
    yb = torch.stack(ys, dim=0)
    return xb, yb, list(metas)


def build_subject_balanced_sampler(ds: ShardedNpyDataset) -> WeightedRandomSampler:
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
# Model
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


class EEGNet(nn.Module):
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


# ============================================================
# Helpers
# ============================================================
def amp_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    class _Null:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return _Null()


@torch.no_grad()
def evaluate(model, loader, device, criterion, use_amp):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    y_true = []
    y_pred = []

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with amp_autocast(device, use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.item()) * yb.size(0)
        total_correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    avg_loss = total_loss / max(total, 1)
    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") if len(y_true) else 0.0

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "macro_f1": float(macro_f1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fyp_root", type=str, default=str(Path.home() / "FYP"))
    parser.add_argument("--data_dir", type=str, default="data/ds004504_preprocessed_shards")
    parser.add_argument("--target_fs", type=int, default=250)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    run_name = "train_eegnet_ds004504"
    fyp_root = Path(args.fyp_root).resolve()
    data_dir = (fyp_root / args.data_dir).resolve()
    train_dir = data_dir / "train"
    eval_dir = data_dir / "eval"

    logger = setup_logger(fyp_root / "experiments" / "logs", run_name)

    logger.info("========== RUN START ==========")
    logger.info(f"FYP root: {fyp_root}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Train dir exists: {train_dir.exists()} | {train_dir}")
    logger.info(f"Eval dir exists : {eval_dir.exists()} | {eval_dir}")
    logger.info(f"Python: {sys.version}")

    if not train_dir.exists() or not eval_dir.exists():
        raise FileNotFoundError("train/ or eval/ directory not found")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    logger.info(f"Device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = ShardedNpyDataset(train_dir, cache_size=args.cache_size, logger=logger)
    eval_ds = ShardedNpyDataset(eval_dir, cache_size=args.cache_size, logger=logger)

    x0, y0, _ = train_ds[0]
    in_channels = int(x0.shape[0])
    win_len = int(x0.shape[1])

    logger.info(f"X_tr shape: ({len(train_ds)}, {in_channels}, {win_len}) | y_tr shape: ({len(train_ds)},)")
    logger.info(f"X_ev shape: ({len(eval_ds)}, {in_channels}, {win_len}) | y_ev shape: ({len(eval_ds)},)")

    unique_labels = sorted(set(int(y) for y in train_ds.labels))
    class_counts = {int(k): int(v) for k, v in pd.Series(train_ds.labels).value_counts().sort_index().to_dict().items()}
    logger.info(f"Train labels: {unique_labels}")
    logger.info(f"Train counts: {class_counts}")

    counts = np.array([class_counts[c] for c in unique_labels], dtype=np.float32)
    weights = 1.0 / (counts + 1e-12)
    weights = weights / weights.sum() * len(unique_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

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

    model = EEGNet(
        n_channels=in_channels,
        fs=args.target_fs,
        n_classes=2,
        emb_dim=128,
    ).to(device)

    logger.info(str(model))

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    logger.info("========== TRAINING ==========")

    best_metric = -1.0
    best_state = None
    best_epoch = -1
    best_val_loss = None
    patience_left = args.patience

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        run_correct = 0
        total = 0

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

        ev = evaluate(model, eval_loader, device, criterion, use_amp)
        val_metric = ev["bal_acc"]
        scheduler.step(val_metric)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={ev['loss']:.4f} val_acc={ev['acc']:.4f} val_bal_acc={ev['bal_acc']:.4f} "
            f"val_macro_f1={ev['macro_f1']:.4f} | lr={lr_now:.2e}"
        )

        if val_metric > best_metric + 1e-6:
            best_metric = val_metric
            best_epoch = epoch
            best_val_loss = ev["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping triggered.")
                break

    train_time = time.time() - t0
    logger.info(f"Training time (s): {train_time:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(
            f"Restored best model (best_epoch={best_epoch}, "
            f"best_val_bal_acc={best_metric:.4f})."
        )

    logger.info("========== EVALUATION ==========")
    final_eval = evaluate(model, eval_loader, device, criterion, use_amp)

    report_text = classification_report(
        final_eval["y_true"],
        final_eval["y_pred"],
        digits=4,
        target_names=["CN(0)", "AD(1)"],
        zero_division=0,
    )
    report_dict = classification_report(
        final_eval["y_true"],
        final_eval["y_pred"],
        digits=4,
        target_names=["CN(0)", "AD(1)"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(final_eval["y_true"], final_eval["y_pred"])

    logger.info("Classification report:\n" + report_text)
    logger.info("Confusion matrix:\n" + str(cm))
    logger.info(
        f"Final val_loss={final_eval['loss']:.4f} "
        f"val_acc={final_eval['acc']:.4f} "
        f"val_bal_acc={final_eval['bal_acc']:.4f} "
        f"val_macro_f1={final_eval['macro_f1']:.4f}"
    )

    models_dir = fyp_root / "models"
    ensure_dir(models_dir)

    exp_dir = fyp_root / "experiments" / run_name
    ensure_dir(exp_dir)

    model_path = models_dir / f"{run_name}.pt"
    metrics_path = exp_dir / "metrics.json"
    cm_path = exp_dir / "confusion_matrix.npy"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "EEGNet",
            "in_channels": in_channels,
            "win_len": win_len,
            "target_fs": args.target_fs,
            "num_classes": 2,
            "emb_dim": 128,
            "best_epoch": best_epoch,
            "best_val_bal_acc": float(best_metric),
            "config": vars(args),
        },
        model_path,
    )

    metrics = {
        "run_name": run_name,
        "train_windows": int(len(train_ds)),
        "eval_windows": int(len(eval_ds)),
        "train_counts": class_counts,
        "train_time_sec": float(train_time),
        "best_epoch": int(best_epoch),
        "best_val_bal_acc": float(best_metric),
        "final_eval": {
            "loss": float(final_eval["loss"]),
            "acc": float(final_eval["acc"]),
            "bal_acc": float(final_eval["bal_acc"]),
            "macro_f1": float(final_eval["macro_f1"]),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.save(cm_path, cm)

    logger.info(f"Saved model weights: {model_path}")
    logger.info(f"Saved metrics: {metrics_path}")
    logger.info(f"Saved confusion matrix: {cm_path}")
    logger.info("========== RUN END ==========")


if __name__ == "__main__":
    main()