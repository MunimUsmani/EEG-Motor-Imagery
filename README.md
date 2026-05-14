# 🧠 EEG Motor Imagery BCI — EEGNet

A complete EEG-based Brain-Computer Interface pipeline for **motor imagery classification** using a compact EEGNet-style convolutional neural network. The project classifies EEG epochs into two imagined movement classes using the **PhysioNet EEG Motor Movement/Imagery Dataset**.

The project includes:

- Automatic PhysioNet EEG data download with MNE
- EEG preprocessing: filtering, resampling, epoch extraction, and normalization
- EEGNet implementation in PyTorch
- Leave-One-Subject-Out cross-validation
- Streamlit demo for EEG visualization and prediction
- Gradient-based channel saliency for model interpretability

---

## 📸 Preview

### 1. Streamlit Demo Home with synthetic data visualization
<img width="956" height="419" alt="image" src="https://github.com/user-attachments/assets/21d289af-38b9-40ab-b30f-2f06f27a2a79" />
<img width="959" height="418" alt="image" src="https://github.com/user-attachments/assets/3b37eace-7bca-4d8a-adda-ace2f16978f1" />

### 2. Epoch Explorer &  Prediction Confidence

<img width="788" height="412" alt="image" src="https://github.com/user-attachments/assets/91bf4bbd-9a9a-4f6b-83de-4ef0e04415f2" />

### 3. EEG Signal Visualization

<img width="793" height="211" alt="image" src="https://github.com/user-attachments/assets/291d39a6-6d12-46f1-9391-baae44eeb57a" />
<img width="767" height="208" alt="image" src="https://github.com/user-attachments/assets/8d57e925-e327-4fcd-9cc2-82e4ecf3530d" />


### 4. All-Epoch Evaluation

<img width="746" height="138" alt="image" src="https://github.com/user-attachments/assets/0f082b5f-57e4-490e-be03-8e17c1fd691e" />


### 5. Channel Saliency

<img width="827" height="355" alt="image" src="https://github.com/user-attachments/assets/da58c92b-a023-48a3-b5fe-5e41a762920b" />


### 6. Saliency Bar Chart
<img width="796" height="440" alt="image" src="https://github.com/user-attachments/assets/4ef608c8-1200-44a5-bbea-16202fe01353" />


---

## 🎯 Motivation

Brain-Computer Interfaces decode neural activity from EEG signals and can support assistive technology, neurorehabilitation, and human-computer interaction.

Motor imagery BCI is especially important because users imagine movements, such as moving the left or right hand, without physically performing them. These imagined movements produce changes in EEG rhythms over sensorimotor brain regions, especially around central channels such as **C3**, **Cz**, and **C4**.

This project implements an EEGNet-based motor imagery pipeline and adds an interactive Streamlit interface to inspect predictions and model behavior.

The main goals are:

- Reproduce an EEGNet-style EEG classification pipeline
- Train and evaluate using subject-level cross-validation
- Visualize EEG epochs and prediction confidence
- Add interpretability through gradient-based channel saliency
- Provide a clean demo suitable for presentation or portfolio use

---

## 🧪 Dataset

This project uses the **PhysioNet EEG Motor Movement/Imagery Dataset**.

The current experiment uses subjects **1–9** and selected motor imagery runs.

| Item | Value |
|---|---:|
| Dataset | PhysioNet EEG Motor Movement/Imagery |
| Subjects used | 9 |
| EEG channels | 64 |
| Sampling rate after preprocessing | 160 Hz |
| Epoch duration | 4 seconds |
| Samples per epoch | 641 |
| Classes | 2 |
| Total epochs | 606 |

The combined processed file is:

```text
data/processed/all_subjects.pkl
```

Individual subject files are saved as:

```text
data/processed/subject_01.pkl
data/processed/subject_02.pkl
...
data/processed/subject_09.pkl
```

---

## ⚠️ Important Label Note

This project uses PhysioNet annotation labels **T1** and **T2** from selected motor imagery runs.

For simplicity, the Streamlit demo displays the two classes as:

- Left fist
- Right fist

However, in the PhysioNet dataset, the exact meaning of T1/T2 can depend on the run. Some runs involve left/right fist imagery, while others involve both fists or both feet imagery.

Therefore, this project should be understood as a **two-class motor imagery cue classification pipeline using T1/T2 annotations**, not a strict clinical left-hand-vs-right-hand decoder across every possible PhysioNet run.

---

## 📊 Results

The model was trained using **Leave-One-Subject-Out cross-validation** on subjects 1–9.

| Model | Accuracy | Cohen's Kappa | Parameters |
|---|---:|---:|---:|
| Chance baseline | 50.0% | 0.000 | — |
| EEGNet-8,2 | **74.9% ± 6.2%** | **0.330 ± 0.245** | ~2.7K |

Training summary:

```text
Loaded: (606, 64, 641) | 2 classes | 9 subjects
LOSO CV: training 9 folds...

==================================================
LOSO Results (EEGNET)
  Mean accuracy : 0.749 ± 0.062
  Mean kappa    : 0.330 ± 0.245
==================================================
```

### Interpretation

The model performs clearly above chance while using a very small number of parameters. This makes EEGNet suitable for lightweight EEG decoding experiments and educational BCI research.

Cohen's Kappa is lower than accuracy because it adjusts for chance agreement and is more conservative, especially with small subject-level test sets.

---

## 🔬 Key Research Visuals

### 1. Streamlit Demo Overview

**What it shows:** The app interface, file upload controls, and project workflow.

**Why it matters:** It proves the project is not only a training script, but a usable interactive EEG decoding demo.

Recommended screenshot:

<img width="956" height="419" alt="image" src="https://github.com/user-attachments/assets/21d289af-38b9-40ab-b30f-2f06f27a2a79" />
<img width="959" height="418" alt="image" src="https://github.com/user-attachments/assets/3b37eace-7bca-4d8a-adda-ace2f16978f1" />

---

### 2. Epoch Explorer

**What it shows:** One selected 4-second EEG epoch, its true label, predicted label, and correctness.

**Why it matters:** It allows inspection of individual examples instead of only reporting aggregate metrics.

Recommended screenshot:

<img width="788" height="412" alt="image" src="https://github.com/user-attachments/assets/91bf4bbd-9a9a-4f6b-83de-4ef0e04415f2" />

---

### 3. Prediction Confidence

**What it shows:** Probability/confidence for the two motor imagery classes.

**Why it matters:** It helps explain whether the model is confident or uncertain for each epoch.

Recommended screenshot:

<img width="788" height="412" alt="image" src="https://github.com/user-attachments/assets/91bf4bbd-9a9a-4f6b-83de-4ef0e04415f2" />

---

### 4. EEG Signal Visualization

**What it shows:** Multi-channel EEG traces for the selected epoch.

**Why it matters:** It connects the prediction back to the actual neural signal and helps viewers understand the input data.

Recommended screenshot:

<img width="793" height="211" alt="image" src="https://github.com/user-attachments/assets/291d39a6-6d12-46f1-9391-baae44eeb57a" />
<img width="767" height="208" alt="image" src="https://github.com/user-attachments/assets/8d57e925-e327-4fcd-9cc2-82e4ecf3530d" />

---

### 5. Evaluation Across All Epochs

**What it shows:** Accuracy and Cohen's Kappa for all epochs of the uploaded subject file.

**Why it matters:** It demonstrates model behavior on a subject-level test set.

Recommended screenshot:

<img width="746" height="138" alt="image" src="https://github.com/user-attachments/assets/0f082b5f-57e4-490e-be03-8e17c1fd691e" />

---

### 6. Channel Saliency

**What it shows:** Top EEG channels that most influenced the model's predictions based on gradient saliency.

**Why it matters:** Motor imagery should involve central sensorimotor regions. Channels such as **C3**, **Cz**, **C4**, **FC3**, **FC4**, **CP3**, and **CP4** are especially interesting.

Recommended screenshot:

<img width="827" height="355" alt="image" src="https://github.com/user-attachments/assets/da58c92b-a023-48a3-b5fe-5e41a762920b" />

---

### 7. Saliency Bar Chart

**What it shows:** Ranked saliency scores for the top EEG channels.

**Why it matters:** It provides a more interpretable view of model sensitivity and supports discussion of whether the network is using neurophysiologically meaningful channels.

Recommended screenshot:

<img width="796" height="440" alt="image" src="https://github.com/user-attachments/assets/4ef608c8-1200-44a5-bbea-16202fe01353" />

---

## 🏗️ Architecture

This project implements **EEGNet-8,2**, a compact CNN architecture designed for EEG decoding.

Input shape:

```text
(batch, 1, 64 channels, 641 samples)
```

The input represents approximately 4 seconds of EEG sampled at 160 Hz.

```text
Input EEG
Shape: 1 × 64 × 641
        │
        ▼
┌─────────────────────────────────────────┐
│ Block 1: Temporal Convolution            │
│ Conv2D(1 → 8, kernel=(1, 64))            │
│ BatchNorm → ELU                          │
│                                          │
│ Depthwise Conv2D                         │
│ Learns spatial filters across channels   │
│ BatchNorm → ELU → AvgPool → Dropout      │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Block 2: Separable Convolution           │
│ Depthwise Conv2D                         │
│ Pointwise Conv2D                         │
│ BatchNorm → ELU → AvgPool → Dropout      │
└─────────────────────────────────────────┘
        │
        ▼
Flatten
        │
        ▼
Linear classifier
        │
        ▼
T1 / T2 motor imagery class
```

### Why EEGNet Works Well for EEG

| Component | Purpose |
|---|---|
| Temporal convolution | Learns frequency-related filters |
| Depthwise spatial convolution | Learns spatial patterns across EEG channels |
| Separable convolution | Reduces parameters while allowing feature mixing |
| Dropout | Reduces overfitting on small EEG datasets |

---

## 📁 Project Structure

```text
EEG-MOTOR-IMAGERY/
├── app/
│   └── demo.py
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── preprocess.py
│   ├── model.py
│   └── train.py
├── data/
│   ├── raw/
│   │   └── MNE-eegbci-data/
│   │       └── files/
│   │           └── eegmmidb/
│   │               └── 1.0.0/
│   │                   ├── S001/
│   │                   ├── S002/
│   │                   ├── ...
│   │                   └── S009/
│   └── processed/
│       ├── all_subjects.pkl
│       ├── subject_01.pkl
│       ├── subject_02.pkl
│       ├── ...
│       └── subject_09.pkl
├── checkpoints/
│   ├── fold_01.pt
│   ├── fold_02.pt
│   ├── ...
│   └── results_summary.pkl
├── assets/
│   ├── streamlit-demo.png
│   ├── epoch-explorer.png
│   ├── prediction-confidence.png
│   ├── eeg-signal.png
│   ├── evaluation-metrics.png
│   ├── channel-saliency.png
│   └── saliency-bar-chart.png
├── EEG_Motor_Imagery.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/eeg-motor-imagery.git
cd eeg-motor-imagery
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\activate
```

```bash
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

If Streamlit is missing:

```bash
python -m pip install streamlit
```

---

## 📥 Data Download

Download the PhysioNet EEG Motor Movement/Imagery dataset:

```bash
python src/download_data.py
```

This downloads subjects 1–9 using MNE and saves the EDF files inside:

```text
data/raw/
```

---

## ⚙️ Preprocessing

Run:

```bash
python src/preprocess.py
```

The preprocessing pipeline:

1. Loads raw EDF files
2. Concatenates selected motor imagery runs
3. Standardizes EEG channel names
4. Applies 7–30 Hz bandpass filtering
5. Resamples EEG to 160 Hz
6. Extracts 4-second epochs from cue onset
7. Z-score normalizes each epoch
8. Saves per-subject and combined `.pkl` files

Output:

```text
data/processed/all_subjects.pkl
data/processed/subject_01.pkl
...
data/processed/subject_09.pkl
```

---

## 🏋️ Training

Run:

```bash
python src/train.py
```

The training script performs Leave-One-Subject-Out cross-validation:

- One subject is held out for testing
- Remaining subjects are used for training
- Best fold checkpoint is saved
- Accuracy and Cohen's Kappa are reported

Checkpoints are saved to:

```text
checkpoints/
```

---

## 🖥️ Running the Streamlit Demo

Run:

```bash
python -m streamlit run app/demo.py
```

The app opens at:

```text
http://localhost:8501
```

Upload:

| File type | Example |
|---|---|
| Processed subject file | `data/processed/subject_01.pkl` |
| Matching checkpoint | `checkpoints/fold_01.pt` |

The Streamlit demo currently supports:

- Uploading processed `.pkl` subject files
- Uploading trained `.pt` checkpoints
- Epoch-by-epoch prediction
- True label vs predicted label
- Prediction confidence bars
- Multi-channel EEG visualization
- Evaluation across all epochs
- Gradient-based channel saliency
- Top attended channel display
- Saliency bar chart

---

## 🔬 Model Interpretability: Channel Saliency

The Streamlit app includes a gradient-based channel saliency feature.

The idea is simple:

> If a small change in an EEG channel strongly changes the model output, that channel receives a higher saliency score.

The app computes saliency by:

1. Taking a subset of EEG epochs
2. Enabling gradients on the input tensor
3. Running the trained model
4. Backpropagating from the predicted class score
5. Taking the average absolute gradient per EEG channel
6. Normalizing the scores

For motor imagery, the most important channels are expected to include central sensorimotor electrodes such as:

```text
C3, Cz, C4, FC3, FC4, CP3, CP4
```

This should be interpreted carefully. Saliency indicates model sensitivity, not causal brain activity.

---

## 🧩 Key Design Decisions

| Design Choice | Reason |
|---|---|
| 7–30 Hz bandpass | Captures mu and beta rhythms commonly used in motor imagery EEG |
| 160 Hz resampling | Reduces computation while preserving frequencies below 30 Hz |
| 4-second epochs | Captures the motor imagery cue period |
| Z-score normalization per epoch | Reduces amplitude variation across epochs and subjects |
| LOSO cross-validation | Tests generalization to unseen subjects |
| Cohen's Kappa | More informative than accuracy alone for subject-level EEG classification |
| Channel saliency | Adds interpretability beyond raw prediction accuracy |
| EEGNet architecture | Compact model designed specifically for EEG decoding |

---

## 🧪 Google Colab Workflow

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
%cd /content/drive/MyDrive/EEG-MOTOR-IMAGERY
```

```python
!python src/train.py
```

Zip checkpoints after training:

```python
!zip -r checkpoints.zip checkpoints
```

---

## 📌 Current Limitations

- Uses subjects 1–9 to keep the experiment lightweight
- Uses a two-class T1/T2 annotation setup
- The demo works with offline processed `.pkl` files, not real-time EEG streaming
- Saliency is qualitative and should not be interpreted as causal brain activity
- Performance varies across subjects, which is expected in EEG decoding
- The project is for research learning and prototyping, not clinical use

---

## 🔮 Possible Extensions

Future improvements could include:

- Training on all 109 PhysioNet subjects
- Adding a stricter run-specific left-hand-vs-right-hand setup
- Supporting 4-class motor imagery classification
- Adding confusion matrix visualization to the Streamlit demo
- Adding topographic scalp maps for channel saliency
- Comparing EEGNet with ShallowConvNet and DeepConvNet
- Exporting saliency plots for reports
- Running experiments on BCI Competition IV Dataset 2a
- Adding real-time EEG device support with OpenBCI or Lab Streaming Layer

---

## 🧠 Research Summary

This project implements EEGNet for EEG motor imagery classification and extends the basic implementation with a Streamlit-based interpretability interface.

The main contribution is not only reproducing the EEGNet architecture, but also providing a way to inspect model behavior through gradient-based EEG channel saliency. This allows qualitative analysis of whether the trained model attends to motor-area-related channels such as C3, Cz, and C4.

---

## 📚 References

1. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J.  
   **EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces.**  
   Journal of Neural Engineering, 15(5), 056013, 2018.  
   DOI: `10.1088/1741-2552/aace8c`

2. Schirrmeister, R. T., et al.  
   **Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization.**  
   Human Brain Mapping, 38(11), 5391–5420, 2017.

3. Goldberger, A. L., et al.  
   **PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.**  
   Circulation, 101(23), e215–e220, 2000.

---

## License

This project is intended for educational and research purposes.

---

## Author

**Abdul Munim Usmani**
