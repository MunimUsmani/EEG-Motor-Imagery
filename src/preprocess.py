"""
Step 2: EEG Preprocessing Pipeline
Raw EEG → Filtered → Epoched → Normalized → Ready for model

Pipeline:
  1. Load raw EDF files from PhysioNet EEG Motor Movement/Imagery Dataset
  2. Rename channels to standard 10-20 system
  3. Bandpass filter: 7–30 Hz
  4. Epoch around events: T=0 to T=4s
  5. Z-score normalize per epoch
  6. Save as numpy arrays: X and y
"""

import os
import pickle

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


# Constants
RUNS = [6, 10, 14]          # Motor imagery runs
TMIN, TMAX = 0.0, 4.0       # Epoch window in seconds
BASELINE = None             # Use None because TMIN starts at 0.0
FMIN, FMAX = 7.0, 30.0      # Mu and beta motor imagery bands
SFREQ_TARGET = 160          # 160 Hz → about 641 samples for 0–4s inclusive

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_subject(subject_id: int) -> mne.io.BaseRaw:
    """Load and concatenate all selected motor imagery runs for one subject."""

    raw_fnames = eegbci.load_data(
        subjects=[subject_id],
        runs=RUNS,
        path=DATA_DIR,
        verbose=False,
    )

    raws = [
        read_raw_edf(fname, preload=True, verbose=False)
        for fname in raw_fnames
    ]

    raw = concatenate_raws(raws)

    # Standardize channel names to match MNE montage names
    eegbci.standardize(raw)

    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, verbose=False)

    return raw


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Apply bandpass filter and resample."""

    raw.filter(
        FMIN,
        FMAX,
        fir_design="firwin",
        verbose=False,
    )

    raw.resample(
        SFREQ_TARGET,
        verbose=False,
    )

    return raw


def extract_epochs(raw: mne.io.BaseRaw):
    """Extract event-locked epochs and return X, y arrays."""

    events, _ = mne.events_from_annotations(raw, verbose=False)

    # PhysioNet annotations:
    # T1 and T2 are task cues.
    # For runs 6, 10, 14:
    #   run 6:  T1 = left fist,  T2 = right fist
    #   run 10: T1 = left fist,  T2 = right fist
    #   run 14: T1 = both fists, T2 = both feet
    #
    # This keeps a simple 2-class setup:
    #   T1 -> class 0
    #   T2 -> class 1
    event_id_map = {
        "T1": 1,
        "T2": 2,
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_map,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()          # shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1] - 1   # convert labels from 1/2 to 0/1

    # Z-score normalize each epoch independently per channel
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-8
    X = (X - mean) / std

    return X.astype(np.float32), y.astype(np.int64)


def preprocess_all_subjects(subjects: list | None = None):
    """Full pipeline: load/download → preprocess → epoch → save."""

    if subjects is None:
        subjects = list(range(1, 10))

    all_X = []
    all_y = []
    all_subj = []

    for subj in subjects:
        print(f"Processing subject {subj:02d}...", end=" ")

        try:
            raw = load_raw_subject(subj)
            raw = preprocess_raw(raw)
            X, y = extract_epochs(raw)

            out_path = os.path.join(
                PROCESSED_DIR,
                f"subject_{subj:02d}.pkl",
            )

            with open(out_path, "wb") as f:
                pickle.dump(
                    {
                        "X": X,
                        "y": y,
                        "subject": subj,
                    },
                    f,
                )

            all_X.append(X)
            all_y.append(y)
            all_subj.extend([subj] * len(y))

            print(
                f"OK — {len(y)} epochs | "
                f"shape {X.shape} | "
                f"labels {np.unique(y)}"
            )

        except Exception as e:
            print(f"FAILED — {e}")

    if not all_X:
        print("\nNo subjects were processed successfully.")
        return None, None, None

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    s_all = np.array(all_subj)

    combined_path = os.path.join(PROCESSED_DIR, "all_subjects.pkl")

    with open(combined_path, "wb") as f:
        pickle.dump(
            {
                "X": X_all,
                "y": y_all,
                "subjects": s_all,
            },
            f,
        )

    label_distribution = {
        int(k): int(v)
        for k, v in zip(*np.unique(y_all, return_counts=True))
    }

    print(f"\nSaved combined dataset: {X_all.shape} → {combined_path}")
    print(f"Label distribution: {label_distribution}")

    return X_all, y_all, s_all


if __name__ == "__main__":
    print("=" * 55)
    print("EEG Preprocessing Pipeline")
    print(f"Bandpass: {FMIN}–{FMAX} Hz | Window: {TMIN}–{TMAX}s")
    print("=" * 55)

    X, y, subjects = preprocess_all_subjects()

    if X is None:
        print("Preprocessing failed. No data was created.")
    else:
        print("Preprocessing completed successfully.")