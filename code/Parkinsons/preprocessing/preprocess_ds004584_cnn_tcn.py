# =============================================================================
# Preprocessing for CNN-TCN Model — OpenNeuro ds004584 (Parkinson's Disease)
# =============================================================================
# This script creates a CNN-TCN-specific processed directory by:
# 1. Re-using the existing subject-wise preprocessing pipeline
# 2. Saving data in the format expected by the CNN-TCN model
#
# Output directory: ../data/processed/ds004584_cnn_tcn/
# Files: train.npz, val.npz, test.npz, norm_params.npz, channels.json, splits.json

import os
import json
import shutil
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split

mne.set_log_level("WARNING")

# --- Path Configuration (resolved from this script's location) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
RAW_ROOT = os.path.join(BASE_DIR, "data", "raw", "OpenNeuro_ds004584")
OUT_ROOT = os.path.join(BASE_DIR, "data", "processed", "ds004584_cnn_tcn")
os.makedirs(OUT_ROOT, exist_ok=True)

# --- Filtering Parameters ---
LOW_FREQ = 0.5
HIGH_FREQ = 45
NOTCH_FREQ = 50
RESAMPLE_FREQ = 250

# --- Windowing Parameters ---
WINDOW_SEC = 2.0
OVERLAP_SEC = 1.0

# --- Reproducibility ---
RANDOM_STATE = 42

# =============================================================================
# Step 1: Load Participant Metadata
# =============================================================================
participants = pd.read_csv(
    os.path.join(RAW_ROOT, "participants.tsv"),
    sep="\t"
)

label_map = {"PD": 1, "Control": 0}
participants["label"] = participants["GROUP"].map(label_map)

print(f"Total subjects: {len(participants)}")
print(f"PD patients: {(participants['label'] == 1).sum()}")
print(f"Controls: {(participants['label'] == 0).sum()}")

# =============================================================================
# Step 2: Subject-wise Train/Val/Test Split (same seed as original)
# =============================================================================
subjects = participants["participant_id"].values
labels = participants["label"].values

sub_train, sub_temp, y_train_labels, y_temp = train_test_split(
    subjects, labels, test_size=0.30, stratify=labels, random_state=RANDOM_STATE
)

sub_val, sub_test, y_val_labels, y_test_labels = train_test_split(
    sub_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

splits = {"train": list(sub_train), "val": list(sub_val), "test": list(sub_test)}
with open(os.path.join(OUT_ROOT, "splits.json"), "w") as f:
    json.dump(splits, f, indent=2)

print(f"Train subjects: {len(sub_train)}")
print(f"Val subjects:   {len(sub_val)}")
print(f"Test subjects:  {len(sub_test)}")

# =============================================================================
# Step 3: Find Common EEG Channels
# =============================================================================
def get_subject_channels(sub_id):
    eeg_path = os.path.join(RAW_ROOT, sub_id, "eeg", f"{sub_id}_task-Rest_eeg.set")
    raw = mne.io.read_raw_eeglab(eeg_path, preload=False)
    raw.pick_types(eeg=True)
    return set(raw.ch_names)

print("Finding common channels across all subjects...")
all_subjects = participants["participant_id"].values

common_channels = get_subject_channels(all_subjects[0])
for sub_id in all_subjects[1:]:
    sub_channels = get_subject_channels(sub_id)
    common_channels = common_channels.intersection(sub_channels)

COMMON_CHANNELS = sorted(list(common_channels))
print(f"Found {len(COMMON_CHANNELS)} common channels")

with open(os.path.join(OUT_ROOT, "channels.json"), "w") as f:
    json.dump(COMMON_CHANNELS, f, indent=2)

# =============================================================================
# Step 4: Preprocessing Function
# =============================================================================
def preprocess_subject(sub_id):
    eeg_path = os.path.join(RAW_ROOT, sub_id, "eeg", f"{sub_id}_task-Rest_eeg.set")
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    raw.pick_channels(COMMON_CHANNELS, ordered=True)
    raw.filter(LOW_FREQ, HIGH_FREQ)
    raw.notch_filter(NOTCH_FREQ)
    raw.set_eeg_reference("average")

    n_ica_components = min(20, len(COMMON_CHANNELS) - 1)
    ica = mne.preprocessing.ICA(
        n_components=n_ica_components,
        random_state=RANDOM_STATE,
        max_iter="auto"
    )
    ica.fit(raw)
    raw = ica.apply(raw)
    raw.resample(RESAMPLE_FREQ)
    return raw

# =============================================================================
# Step 5: Build Dataset for a Split
# =============================================================================
def build_split_dataset(subject_ids):
    X_all, y_all, sid_all = [], [], []
    for i, sid in enumerate(subject_ids):
        print(f"  Processing {sid} ({i+1}/{len(subject_ids)})...")
        try:
            label = participants.loc[
                participants["participant_id"] == sid, "label"
            ].values[0]

            raw = preprocess_subject(sid)
            epochs = mne.make_fixed_length_epochs(
                raw, duration=WINDOW_SEC, overlap=OVERLAP_SEC, preload=True
            )
            X = epochs.get_data().astype(np.float32)  # (n_epochs, n_channels, n_samples)
            y = np.full(len(X), label, dtype=np.int64)
            s = np.full(len(X), sid)

            X_all.append(X)
            y_all.append(y)
            sid_all.append(s)
            print(f"    -> {len(X)} epochs, shape: {X.shape[1:]}")
        except Exception as e:
            print(f"    -> ERROR: {e}, skipping subject")
            continue

    return np.concatenate(X_all), np.concatenate(y_all), np.concatenate(sid_all)

# =============================================================================
# Step 6: Process All Splits
# =============================================================================
print("=" * 60)
print("Processing TRAINING set...")
print("=" * 60)
X_train, y_train, sid_train = build_split_dataset(sub_train)

print("\n" + "=" * 60)
print("Processing VALIDATION set...")
print("=" * 60)
X_val, y_val, sid_val = build_split_dataset(sub_val)

print("\n" + "=" * 60)
print("Processing TEST set...")
print("=" * 60)
X_test, y_test, sid_test = build_split_dataset(sub_test)

# =============================================================================
# Step 7: Normalize (z-score, stats from train only)
# =============================================================================
mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
std = (X_train.std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# =============================================================================
# Step 8: Save
# =============================================================================
np.savez(os.path.join(OUT_ROOT, "train.npz"), X=X_train, y=y_train, subject_id=sid_train)
np.savez(os.path.join(OUT_ROOT, "val.npz"), X=X_val, y=y_val, subject_id=sid_val)
np.savez(os.path.join(OUT_ROOT, "test.npz"), X=X_test, y=y_test, subject_id=sid_test)
np.savez(os.path.join(OUT_ROOT, "norm_params.npz"), mean=mean, std=std)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE for CNN-TCN!")
print("=" * 60)
print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")
print(f"Files saved to: {OUT_ROOT}")
