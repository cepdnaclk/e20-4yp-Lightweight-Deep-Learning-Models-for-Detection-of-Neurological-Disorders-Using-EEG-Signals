#!/usr/bin/env python3
"""
Train + evaluate a 1D CNN on TUAB_preprocessed_sample5GB_balanced (sharded X_*.npy + meta_*.csv)

Folder layout expected:
<root>/
    data/TUAB_preprocessed_sample5GB_balanced/
    train/   X_*.npy, meta_*.csv
    eval/    X_*.npy, meta_*.csv

Main ideas:
- Stream X from disk via np.load(mmap_mode="r") to avoid RAM blowups
- Pair each X shard with the correct meta CSV by matching number of rows
- Read labels from meta CSV, auto-detect label column (or use --label_col)
- Z-score normalize each sample
- Train Conv1d model with early stopping on balanced accuracy
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
from sklearn.preprocessing import LabelEncoder


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
# 2) Path helpers (so directories are obvious)
# ============================================================
def resolve_paths(fyp_root: Path, data_dir: str) -> Dict[str, Path]:
    data_root = (fyp_root / data_dir).resolve()
    return {
        "fyp_root": fyp_root.resolve(),
        "data_root": data_root,
        "train_dir": (data_root / "train").resolve(),
        "eval_dir": (data_root / "eval").resolve(),
    }


def scan_split(split_dir: Path, x_glob: str, meta_glob: str) -> Dict[str, List[Path]]:
    return {
        "x_paths": sorted(split_dir.glob(x_glob)),
        "meta_paths": sorted(split_dir.glob(meta_glob)),
    }


# ============================================================
# 3) Data utilities
# ============================================================
def to_time_channel(X: np.ndarray) -> np.ndarray:
    """
    Ensure X is (N, T, C).
    Supports:
      - (N, T) -> (N, T, 1)
      - (N, C, T) -> transpose -> (N, T, C) (heuristic: C < T)
      - (N, T, C) -> keep
    """
    X = np.asarray(X)
    if X.ndim == 2:
        return X[..., np.newaxis]
    if X.ndim == 3:
        _, a, b = X.shape
        if a < b:
            return np.transpose(X, (0, 2, 1))
        return X
    raise ValueError(f"Unsupported X shape: {X.shape}")


def zscore_per_sample(x_tc: np.ndarray) -> np.ndarray:
    """x_tc is (T, C) -> z-scored (T, C)"""
    mean = x_tc.mean(axis=(0, 1), keepdims=True)
    std = x_tc.std(axis=(0, 1), keepdims=True) + 1e-6
    return (x_tc - mean) / std


def detect_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "label", "y", "target", "class", "cls",
        "abnormal", "is_abnormal",
        "pathology", "diagnosis", "condition"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(
        "Could not auto-detect label column.\n"
        f"Columns are: {list(df.columns)}\n"
        "Re-run with: --label_col <your_label_column_name>"
    )


def fast_row_count_csv(csv_path: Path) -> int:
    """Count rows quickly (assumes 1 header row)."""
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def match_meta_by_rowcount(x_path: Path, meta_paths: List[Path], logger: logging.Logger) -> Path:
    """
    Pair X shard with meta CSV by matching sample count (N) to row count.
    """
    x_mmap = np.load(x_path, mmap_mode="r")
    n = int(x_mmap.shape[0])

    matches = []
    for mp in meta_paths:
        try:
            if fast_row_count_csv(mp) == n:
                matches.append(mp)
        except Exception:
            continue

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(f"No meta CSV with {n} rows found for {x_path.name}")

    matches = sorted(matches, key=lambda p: p.name)
    logger.warning(f"Multiple meta matches for {x_path.name} (n={n}). Using {matches[0].name}")
    return matches[0]


# ============================================================
# 4) Dataset (stream X from disk, keep labels in RAM)
# ============================================================
class ShardedNpyDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        x_paths: List[Path],
        meta_paths: List[Path],
        label_col: Optional[str],
        cache_size: int,
        logger: logging.Logger,
    ):
        self.split_dir = split_dir
        self.label_col = label_col
        self.logger = logger

        if not x_paths:
            raise FileNotFoundError(f"No X shards found in {split_dir}")
        if not meta_paths:
            raise FileNotFoundError(f"No meta CSV found in {split_dir}")

        self.cache_size = cache_size
        self._x_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

        self.shards: List[Dict[str, Any]] = []
        self.shard_offsets: List[int] = []
        total = 0

        for xp in x_paths:
            mp = match_meta_by_rowcount(xp, meta_paths, logger)

            df = pd.read_csv(mp)
            col = self.label_col or detect_label_column(df)
            if col not in df.columns:
                raise ValueError(f"label_col='{col}' not found in {mp.name}. Columns: {list(df.columns)}")

            y_raw = df[col].to_numpy()

            x_mmap = to_time_channel(np.load(xp, mmap_mode="r"))
            n = int(x_mmap.shape[0])
            if len(y_raw) != n:
                raise ValueError(f"Row mismatch: {xp.name} has {n} samples, {mp.name} has {len(y_raw)} rows")

            self.shards.append({"x_path": xp, "y_raw": y_raw, "n": n})
            self.shard_offsets.append(total)
            total += n

            logger.info(f"[{split_dir.name}] Paired {xp.name} (n={n}) <-> {mp.name} | label_col='{col}'")

        self.total_len = total

    def __len__(self) -> int:
        return self.total_len

    def _get_mmap(self, x_path: Path) -> np.ndarray:
        key = str(x_path)
        if key in self._x_cache:
            self._x_cache.move_to_end(key)
            return self._x_cache[key]

        x_mmap = to_time_channel(np.load(x_path, mmap_mode="r"))
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

        x_mmap = self._get_mmap(shard["x_path"])      # (N, T, C)
        x_tc = x_mmap[local_i].astype(np.float32)     # (T, C)
        x_tc = zscore_per_sample(x_tc)                # normalize
        x_ct = np.transpose(x_tc, (1, 0))             # (C, T)

        y = shard["y_raw"][local_i]
        return torch.from_numpy(x_ct), y


# ============================================================
# 5) Model
# ============================================================
class CNN1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.head(x)


# ============================================================
# 6) Train / Eval
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


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 7) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fyp_root", type=str, default=str(Path.cwd()))
    parser.add_argument("--data_dir", type=str, default="data/TUAB_preprocessed_sample5GB_balanced")
    parser.add_argument("--tag", type=str, default="TUAB_preprocessed_sample5GB_balanced")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=4)

    parser.add_argument("--x_glob", type=str, default="X_*.npy")
    parser.add_argument("--meta_glob", type=str, default="meta_*.csv")
    parser.add_argument("--label_col", type=str, default=None)

    args = parser.parse_args()

    # ---- Paths + logger
    paths = resolve_paths(Path(args.fyp_root), args.data_dir)
    run_name = f"cnn1d_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(paths["fyp_root"] / "experiments" / "logs", run_name)

    logger.info("========== RUN START ==========")
    logger.info(f"FYP root : {paths['fyp_root']}")
    logger.info(f"Data root: {paths['data_root']}")
    logger.info(f"Train dir: {paths['train_dir']} (exists={paths['train_dir'].exists()})")
    logger.info(f"Eval dir : {paths['eval_dir']} (exists={paths['eval_dir'].exists()})")

    # ---- Seed + device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Scan data (this is what you were missing)
    train_scan = scan_split(paths["train_dir"], args.x_glob, args.meta_glob)
    eval_scan = scan_split(paths["eval_dir"], args.x_glob, args.meta_glob)

    logger.info(f"[train] X shards: {len(train_scan['x_paths'])} | meta CSVs: {len(train_scan['meta_paths'])}")
    logger.info(f"[eval ] X shards: {len(eval_scan['x_paths'])} | meta CSVs: {len(eval_scan['meta_paths'])}")

    # ---- Build datasets
    train_ds = ShardedNpyDataset(
        split_dir=paths["train_dir"],
        x_paths=train_scan["x_paths"],
        meta_paths=train_scan["meta_paths"],
        label_col=args.label_col,
        cache_size=args.cache_size,
        logger=logger,
    )
    eval_ds = ShardedNpyDataset(
        split_dir=paths["eval_dir"],
        x_paths=eval_scan["x_paths"],
        meta_paths=eval_scan["meta_paths"],
        label_col=args.label_col,
        cache_size=args.cache_size,
        logger=logger,
    )

    # ---- Label encoder from train labels
    train_raw = np.concatenate([sh["y_raw"] for sh in train_ds.shards])
    le = LabelEncoder().fit(train_raw)

    def collate_fn(batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=0)
        ys = np.array(ys, dtype=object)
        ys_enc = torch.tensor(le.transform(ys), dtype=torch.long)
        return xs, ys_enc

    # ---- Class weights (inverse frequency)
    train_y_enc = le.transform(train_raw)
    classes, counts = np.unique(train_y_enc, return_counts=True)
    num_classes = int(len(classes))

    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-12)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    logger.info(f"Classes: {le.classes_.tolist()}")
    logger.info(f"Train counts: {dict(zip([int(c) for c in classes], [int(x) for x in counts]))}")

    # ---- DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    # ---- Infer in_channels from a sample
    x0, _ = train_ds[0]
    in_channels = int(x0.shape[0])
    logger.info(f"Input channels: {in_channels}")

    # ---- Model + optim
    model = CNN1D(in_channels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    # ---- Train (early stopping on balanced accuracy)
    best_val_balacc = -1.0
    best_state = None
    patience_left = args.patience

    logger.info("========== TRAINING ==========")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
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

    # ---- Final evaluation
    logger.info("========== EVALUATION ==========")
    val_loss, val_acc, val_balacc, val_macro_f1, y_true, y_pred = evaluate(model, eval_loader, device, criterion)

    report = classification_report(
        y_true, y_pred, digits=4, output_dict=True,
        target_names=[str(x) for x in le.classes_]
    )
    report_text = classification_report(
        y_true, y_pred, digits=4,
        target_names=[str(x) for x in le.classes_]
    )
    cm = confusion_matrix(y_true, y_pred)

    logger.info("Classification report:\n" + report_text)
    logger.info("Confusion matrix:\n" + str(cm))

    # ---- Save outputs
    exp_dir = paths["fyp_root"] / "experiments" / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    models_dir = paths["fyp_root"] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{run_name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "num_classes": num_classes,
        "label_encoder_classes": le.classes_.tolist(),
        "seed": args.seed,
        "data_dir": str(paths["data_root"]),
        "tag": args.tag,
    }, model_path)

    metrics = {
        "run_name": run_name,
        "tag": args.tag,
        "train_time_sec": float(train_time),
        "train_samples": int(len(train_ds)),
        "eval_samples": int(len(eval_ds)),
        "num_classes": int(num_classes),
        "classes": le.classes_.tolist(),
        "final_val_loss": float(val_loss),
        "final_val_acc": float(val_acc),
        "final_val_balanced_acc": float(val_balacc),
        "final_val_macro_f1": float(val_macro_f1),
        "best_val_balanced_acc": float(best_val_balacc),
        "classification_report": report,
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(exp_dir / "confusion_matrix.npy", cm)

    logger.info(f"Saved model  : {model_path}")
    logger.info(f"Saved metrics: {exp_dir / 'metrics.json'}")
    logger.info(f"Saved CM     : {exp_dir / 'confusion_matrix.npy'}")
    logger.info("========== RUN END ==========")


if __name__ == "__main__":
    main()
