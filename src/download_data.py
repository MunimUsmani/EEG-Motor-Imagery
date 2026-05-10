"""
Step 1: Download PhysioNet EEG Motor Movement/Imagery Dataset

Uses MNE's built-in downloader.
Dataset: PhysioNet EEG Motor Movement/Imagery Dataset
Subjects: using subjects 1-9
Runs: motor imagery runs 6, 10, 14
"""

import os
import mne
from mne.datasets import eegbci

RUNS_MOTOR_IMAGERY = [6, 10, 14]   # motor imagery runs
SUBJECTS = list(range(1, 10))       # subjects 1-9

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)


def download_subject(subject_id: int):
    """Download PhysioNet motor imagery data for one subject."""
    print(f"  Downloading subject {subject_id:02d}...")

    raw_fnames = eegbci.load_data(
        subjects=[subject_id],
        runs=RUNS_MOTOR_IMAGERY,
        path=DATA_DIR,
        verbose=False,
    )

    return raw_fnames


def main():
    print("=" * 55)
    print("Downloading PhysioNet EEG Motor Imagery Dataset")
    print("Subjects: 1-9 | Runs: left fist, right fist, feet")
    print("=" * 55)

    all_files = {}

    for subj in SUBJECTS:
        files = download_subject(subj)
        all_files[subj] = files
        print(f"  Subject {subj:02d}: {len(files)} runs saved")

    print(f"\nDone. Files saved to: {DATA_DIR}/")
    print(f"Total subjects downloaded: {len(all_files)}")

    return all_files


if __name__ == "__main__":
    main()