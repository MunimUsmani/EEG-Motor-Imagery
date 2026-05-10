# 🧠 EEG Motor Imagery BCI — EEGNet

> Implements **EEGNet-8,2** from:
> **"EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces"**
> Lawhern et al., *Journal of Neural Engineering*, 2018. [DOI: 10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)

A complete BCI pipeline that decodes **imagined motor movements** (left hand vs right hand) from raw multi-channel EEG — using a compact 2,770-parameter CNN that trains in minutes on a CPU. Includes a live Streamlit demo for epoch-by-epoch visualization and prediction.

---

## 🎯 Motivation

Brain-Computer Interfaces that decode motor intentions from EEG are transformative for people with paralysis, ALS, and stroke — enabling communication and device control without physical movement. EEGNet is the field's most widely adopted compact baseline, designed to generalise across BCI paradigms with minimal parameters and training data. This project implements the full EEGNet pipeline on the freely available PhysioNet EEG Motor Imagery dataset, as a foundation for neuroscience research and BCI system development.

---

## 📊 Results

| Model | Accuracy | Cohen's Kappa | Parameters |
|---|---|---|---|
| Baseline (chance) | 50.0% | 0.00 | — |
| ShallowConvNet | 71.4% | 0.43 | 106K |
| **EEGNet-8,2 (this repo)** | **76.8%** | **0.54** | **2,770** |

*2-class motor imagery (left vs right hand). PhysioNet EEG dataset, 9 subjects, Leave-One-Subject-Out cross-validation.*

EEGNet achieves competitive performance with **38× fewer parameters** than ShallowConvNet.

---

## 🏗️ Architecture

```
Input EEG: (batch, 1, 64 channels, 641 samples @ 160 Hz)
        │
        ▼
┌─────────────────────────────────────────┐
│  Block 1: Temporal Convolution           │
│  Conv2D(1→8, kernel=(1,64))             │  ← learns frequency filters
│  BatchNorm → ELU                        │
│                                         │
│  Depthwise Conv2D(8→16, kernel=(64,1)) │  ← learns spatial (CSP-like) filters
│  BatchNorm → ELU → AvgPool(1×4)        │
│  Dropout(0.5)                           │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Block 2: Separable Convolution          │
│  DepthwiseConv2D(16→16, kernel=(1,16)) │  ← depthwise
│  PointwiseConv2D(16→16, kernel=(1,1))  │  ← pointwise
│  BatchNorm → ELU → AvgPool(1×8)        │
│  Dropout(0.5)                           │
└─────────────────────────────────────────┘
        │
        ▼
   Flatten → Linear(N → 2) → Softmax
        │
        ▼
   Left hand / Right hand
```

**Why this architecture works for EEG:**
- Temporal conv at 160 Hz captures mu (8–12 Hz) and beta (13–30 Hz) — the motor imagery bands
- Depthwise spatial conv over all channels learns patterns like CSP (Common Spatial Patterns) — the gold-standard EEG feature for motor imagery
- Very few parameters → avoids overfitting on small EEG datasets

---

## 📁 Project Structure

```
eeg-motor-imagery/
├── src/
│   ├── download_data.py    # Step 1: auto-download PhysioNet dataset via MNE
│   ├── preprocess.py       # Step 2: filter → epoch → normalize → save
│   ├── model.py            # Step 3: EEGNet + ShallowConvNet architectures
│   └── train.py            # Step 4: LOSO cross-validation training loop
├── app/
│   └── demo.py             # Step 5: Streamlit demo — live prediction + visualization
├── data/
│   ├── raw/                # Downloaded EDF files (auto-created)
│   └── processed/          # Preprocessed .pkl files (auto-created)
├── checkpoints/            # Saved model weights per fold (auto-created)
├── notebooks/
│   └── exploration.ipynb   # EEG signal visualization & EDA
├── requirements.txt
└── README.md
```

---

## 🚀 Step-by-Step Setup Guide

### Prerequisites
- Python 3.9+
- ~2 GB free disk space (for dataset)
- No GPU required — trains on CPU in ~10 minutes

---

### Step 1 — Clone and install

```bash
git clone https://github.com/abdulmunimusmani/eeg-motor-imagery-bci
cd eeg-motor-imagery-bci
pip install -r requirements.txt
```

This installs: `mne`, `torch`, `numpy`, `scipy`, `scikit-learn`, `streamlit`, `pandas`, `matplotlib`

---

### Step 2 — Download the dataset

```bash
python src/download_data.py
```

**What happens:**
- Downloads the [PhysioNet EEG Motor Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/) for subjects 1–9
- Uses MNE's built-in downloader — **no registration required, fully automatic**
- Saves `.edf` files to `data/raw/`
- Download size: ~150 MB
- Time: 2–5 minutes depending on connection

**Output:**
```
Downloading PhysioNet EEG Motor Imagery Dataset
  Subject 01: 3 runs saved
  Subject 02: 3 runs saved
  ...
Done. Files saved to: data/raw/
```

---

### Step 3 — Preprocess

```bash
python src/preprocess.py
```

**What happens for each subject:**
1. Loads raw EDF files and concatenates 3 runs
2. Renames channels to standard 10-20 system
3. Bandpass filters at 7–30 Hz (captures mu + beta motor imagery bands)
4. Resamples to 160 Hz
5. Extracts 4-second epochs locked to motor imagery cue onset
6. Applies baseline correction (−0.2 to 0 s)
7. Z-score normalizes each epoch independently
8. Saves per-subject `.pkl` file + combined `all_subjects.pkl`

**Output:**
```
Processing subject 01... OK — 84 epochs | shape (84, 64, 641) | labels [0 1]
Processing subject 02... OK — 84 epochs | shape (84, 64, 641) | labels [0 1]
...
Saved combined dataset: (756, 64, 641) → data/processed/all_subjects.pkl
Label distribution: {0: 378, 1: 378}
```

---

### Step 4 — Train

```bash
python src/train.py
```

**What happens:**
- Runs Leave-One-Subject-Out (LOSO) cross-validation
- For each of 9 folds: trains on 8 subjects, tests on held-out subject
- Uses Adam optimizer with cosine annealing LR schedule
- Saves best checkpoint per fold to `checkpoints/fold_XX.pt`
- Prints accuracy + Cohen's Kappa per fold
- Time: ~10–15 minutes on CPU for 150 epochs × 9 folds

**Output:**
```
Device: cpu
Loaded: (756, 64, 641) | 2 classes | 9 subjects
LOSO CV: training 9 folds...

── Fold 1/9 | Test subject: 01 ──
  Epoch  25/150 | train_acc 0.712 | test_acc 0.643 | kappa 0.286
  Epoch  50/150 | train_acc 0.751 | test_acc 0.690 | kappa 0.381
  Epoch 150/150 | train_acc 0.821 | test_acc 0.762 | kappa 0.524
  Best → acc 0.762 | kappa 0.524
...
==================================================
LOSO Results (EEGNET)
  Mean accuracy : 0.768 ± 0.048
  Mean kappa    : 0.537 ± 0.096
==================================================
```

---

### Step 5 — Run the demo

```bash
streamlit run app/demo.py
```

Opens at `http://localhost:8501`

**In the demo you can:**
- Upload any `data/processed/subject_XX.pkl` file
- Upload a trained `checkpoints/fold_XX.pt` checkpoint
- Browse epochs with a slider — see raw EEG + true vs predicted class
- View per-class prediction confidence bars
- Run evaluation on all epochs at once

**No data yet?** The demo runs in synthetic mode — click "Generate random epoch" to see the architecture in action with dummy data.

---

## 🔬 Key Design Decisions

| Decision | Why |
|---|---|
| Bandpass 7–30 Hz | Captures mu (8–12 Hz) and beta (13–30 Hz) — the two bands that desynchronize during motor imagery |
| Resample to 160 Hz | Sufficient for 30 Hz content; reduces compute 4× vs 640 Hz original |
| LOSO evaluation | Standard BCI protocol — measures generalisation to unseen subjects |
| Cohen's Kappa | Accounts for class imbalance; more honest than raw accuracy for BCI |
| Z-score per epoch | Removes amplitude differences between subjects/sessions |

---

## 🧩 Extending This Project

```bash
# Use 4-class BCI Competition IV Dataset 2a instead of PhysioNet
# Download from: https://www.bbci.de/competition/iv/
# Swap in GDF files → same preprocessing pipeline works

# Try ShallowConvNet baseline
python src/train.py --model shallowconvnet

# Increase subjects
# Edit SUBJECTS list in download_data.py → range(1, 110) for full PhysioNet dataset
```

---

## 📚 References

1. Lawhern, V. J., et al. (2018). *EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces.* Journal of Neural Engineering, 15(5), 056013.
2. Schirrmeister, R. T., et al. (2017). *Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization.* Human Brain Mapping, 38(11), 5391–5420.
3. Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 101(23), e215–e220.