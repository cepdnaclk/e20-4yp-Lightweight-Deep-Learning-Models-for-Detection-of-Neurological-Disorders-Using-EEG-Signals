import json
import os

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


mne.set_log_level("WARNING")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")
RAW_ROOT = os.path.join(BASE_DIR, "data", "raw", "OpenNeuro ds002778")
OUT_ROOT = os.path.join(BASE_DIR, "data", "processed", "ds002778")
os.makedirs(OUT_ROOT, exist_ok=True)


LOW_FREQ = 0.5
HIGH_FREQ = 45.0
NOTCH_FREQ = 60.0
RESAMPLE_FREQ = 250

WINDOW_SEC = 2.0
OVERLAP_SEC = 1.0

RANDOM_STATE = 42


def load_participants():
    participants = pd.read_csv(os.path.join(RAW_ROOT, "participants.tsv"), sep="\t")
    participants["label"] = participants["participant_id"].str.contains("-pd").astype(np.int64)
    return participants


def discover_recordings(participants):
    subject_to_recordings = {}

    for participant_id in participants["participant_id"]:
        subject_dir = os.path.join(RAW_ROOT, participant_id)
        recordings = []

        if not os.path.isdir(subject_dir):
            subject_to_recordings[participant_id] = recordings
            continue

        for session_name in sorted(os.listdir(subject_dir)):
            eeg_dir = os.path.join(subject_dir, session_name, "eeg")
            if not os.path.isdir(eeg_dir):
                continue

            bdf_name = f"{participant_id}_{session_name}_task-rest_eeg.bdf"
            bdf_path = os.path.join(eeg_dir, bdf_name)
            if os.path.isfile(bdf_path):
                recordings.append(
                    {
                        "subject_id": participant_id,
                        "session_id": session_name,
                        "eeg_path": bdf_path,
                    }
                )

        subject_to_recordings[participant_id] = recordings

    return subject_to_recordings


def get_common_channels(recordings):
    common_channels = None

    for recording in recordings:
        raw = mne.io.read_raw_bdf(recording["eeg_path"], preload=False)
        raw.pick_types(eeg=True, stim=False, misc=False, exclude=[])
        channel_names = set(raw.ch_names)
        if common_channels is None:
            common_channels = channel_names
        else:
            common_channels &= channel_names

    if not common_channels:
        raise RuntimeError("No common EEG channels found across ds002778 recordings.")

    return sorted(common_channels)


def preprocess_recording(eeg_path, common_channels):
    raw = mne.io.read_raw_bdf(eeg_path, preload=True)
    raw.pick_types(eeg=True, stim=False, misc=False, exclude=[])
    raw.pick_channels(common_channels, ordered=True)
    raw.filter(LOW_FREQ, HIGH_FREQ)
    raw.notch_filter(NOTCH_FREQ)
    raw.set_eeg_reference("average")

    n_ica_components = min(20, len(common_channels) - 1)
    if n_ica_components >= 2:
        ica = mne.preprocessing.ICA(
            n_components=n_ica_components,
            random_state=RANDOM_STATE,
            max_iter="auto",
        )
        ica.fit(raw)
        raw = ica.apply(raw)

    raw.resample(RESAMPLE_FREQ)
    return raw


def build_split_dataset(subject_ids, subject_to_recordings, participants, common_channels):
    X_all = []
    y_all = []
    sid_all = []

    for subject_index, subject_id in enumerate(subject_ids, start=1):
        subject_recordings = subject_to_recordings.get(subject_id, [])
        label = participants.loc[
            participants["participant_id"] == subject_id, "label"
        ].values[0]

        if not subject_recordings:
            print(f"  Skipping {subject_id} ({subject_index}/{len(subject_ids)}): no EEG recordings found")
            continue

        print(f"  Processing {subject_id} ({subject_index}/{len(subject_ids)})...")
        for recording in subject_recordings:
            session_id = recording["session_id"]
            try:
                raw = preprocess_recording(recording["eeg_path"], common_channels)
                epochs = mne.make_fixed_length_epochs(
                    raw,
                    duration=WINDOW_SEC,
                    overlap=OVERLAP_SEC,
                    preload=True,
                )
                X = epochs.get_data().astype(np.float32)
                y = np.full(len(X), label, dtype=np.int64)
                sid = np.full(len(X), subject_id)

                X_all.append(X)
                y_all.append(y)
                sid_all.append(sid)
                print(f"    {session_id}: {len(X)} epochs, shape {X.shape[1:]}")
            except Exception as exc:
                print(f"    {session_id}: ERROR: {exc}")

    if not X_all:
        raise RuntimeError("Split preprocessing produced no epochs.")

    return np.concatenate(X_all), np.concatenate(y_all), np.concatenate(sid_all)


participants = load_participants()
subject_to_recordings = discover_recordings(participants)

available_subjects = participants[
    participants["participant_id"].map(lambda subject_id: len(subject_to_recordings.get(subject_id, [])) > 0)
].copy()

if available_subjects.empty:
    raise RuntimeError("No ds002778 EEG recordings were found to preprocess.")


subjects = available_subjects["participant_id"].to_numpy()
labels = available_subjects["label"].to_numpy()

print(f"Total subjects with EEG: {len(available_subjects)}")
print(f"PD subjects: {(labels == 1).sum()}")
print(f"Control subjects: {(labels == 0).sum()}")
print(f"Total recordings: {sum(len(recordings) for recordings in subject_to_recordings.values())}")


sub_train, sub_temp, y_train_labels, y_temp = train_test_split(
    subjects,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=RANDOM_STATE,
)

sub_val, sub_test, y_val_labels, y_test_labels = train_test_split(
    sub_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)


splits = {"train": list(sub_train), "val": list(sub_val), "test": list(sub_test)}
with open(os.path.join(OUT_ROOT, "splits.json"), "w", encoding="utf-8") as file_obj:
    json.dump(splits, file_obj, indent=2)

print(f"Train subjects: {len(sub_train)}")
print(f"Val subjects:   {len(sub_val)}")
print(f"Test subjects:  {len(sub_test)}")


all_recordings = [
    recording
    for subject_id in available_subjects["participant_id"]
    for recording in subject_to_recordings[subject_id]
]

print("Finding common EEG channels across all recordings...")
common_channels = get_common_channels(all_recordings)
print(f"Found {len(common_channels)} common EEG channels")

with open(os.path.join(OUT_ROOT, "channels.json"), "w", encoding="utf-8") as file_obj:
    json.dump(common_channels, file_obj, indent=2)


print("=" * 60)
print("Processing TRAINING set...")
print("=" * 60)
X_train, y_train, sid_train = build_split_dataset(
    sub_train,
    subject_to_recordings,
    participants,
    common_channels,
)

print("\n" + "=" * 60)
print("Processing VALIDATION set...")
print("=" * 60)
X_val, y_val, sid_val = build_split_dataset(
    sub_val,
    subject_to_recordings,
    participants,
    common_channels,
)

print("\n" + "=" * 60)
print("Processing TEST set...")
print("=" * 60)
X_test, y_test, sid_test = build_split_dataset(
    sub_test,
    subject_to_recordings,
    participants,
    common_channels,
)


mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
std = (X_train.std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


np.savez(os.path.join(OUT_ROOT, "train.npz"), X=X_train, y=y_train, subject_id=sid_train)
np.savez(os.path.join(OUT_ROOT, "val.npz"), X=X_val, y=y_val, subject_id=sid_val)
np.savez(os.path.join(OUT_ROOT, "test.npz"), X=X_test, y=y_test, subject_id=sid_test)
np.savez(os.path.join(OUT_ROOT, "norm_params.npz"), mean=mean, std=std)


print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE for ds002778!")
print("=" * 60)
print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")
print(f"Files saved to: {OUT_ROOT}")