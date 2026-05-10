# 🧠 EEG Motor Imagery BCI — EEGNet

Implementation of **EEGNet-8,2** from:

> Lawhern, V. J., et al.  
> **EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces**  
> *Journal of Neural Engineering*, 2018.  
> DOI: `10.1088/1741-2552/aace8c`

This project implements a complete EEG-based Brain-Computer Interface pipeline for motor imagery classification using the freely available **PhysioNet EEG Motor Movement/Imagery Dataset**.

The pipeline includes:

- Automatic EEG dataset download using MNE
- EEG preprocessing: filtering, resampling, epoch extraction, and normalization
- EEGNet model implementation in PyTorch
- Leave-One-Subject-Out cross-validation
- Streamlit demo for prediction and EEG visualization
- Gradient-based channel saliency for model interpretability

---

## 🎯 Motivation

Brain-Computer Interfaces can decode neural activity from EEG signals and have potential applications in assistive technology, neurorehabilitation, and human-computer interaction.

Motor imagery BCI is especially important because users imagine movements, such as moving the left or right hand, without physically performing them. EEGNet is a compact convolutional neural network designed specifically for EEG decoding. It is widely used as a strong baseline because it has very few parameters and can generalize across several BCI paradigms.

This project reproduces the EEGNet-style pipeline and extends it with a simple interpretability feature to inspect which EEG channels influence the model's predictions.

---

## 📊 Results

The model was trained using **Leave-One-Subject-Out cross-validation** on subjects 1–9 from the PhysioNet EEG Motor Movement/Imagery dataset.

| Model | Accuracy | Cohen's Kappa | Parameters |
|---|---:|---:|---:|
| Chance baseline | 50.0% | 0.000 | — |
| EEGNet-8,2 | **74.9% ± 6.2%** | **0.330 ± 0.245** | ~2.7K |

Training output:
Loaded: (606, 64, 641) | 2 classes | 9 subjects
LOSO CV: training 9 folds...

==================================================
LOSO Results (EEGNET)
  Mean accuracy : 0.749 ± 0.062
  Mean kappa    : 0.330 ± 0.245
==================================================

The model reaches above-chance performance while using only a small number of parameters, making it suitable for lightweight EEG decoding experiments.

⚠️ Important Label Note

This project uses PhysioNet annotation labels T1 and T2 from selected motor imagery runs.

For simplicity, the Streamlit demo displays the two classes as:

Left fist
Right fist

However, in the PhysioNet dataset, the exact meaning of T1 and T2 can depend on the run. For example, some runs involve left/right fist imagery, while others may involve both fists or both feet imagery.

Therefore, this project should be understood as a two-class motor imagery cue classification pipeline using PhysioNet T1/T2 annotations, rather than a strict clinical left-hand-vs-right-hand decoder across all possible runs.

🏗️ Architecture

Input shape:

(batch, 1, 64 channels, 641 samples)

The input represents 4 seconds of EEG sampled at 160 Hz.

Input EEG
   │
   ▼
┌─────────────────────────────────────────┐
│ Block 1: Temporal Convolution            │
│ Conv2D(1 → 8, kernel=(1, 64))            │
│ BatchNorm → ELU                          │
│                                          │
│ Depthwise Conv2D(8 → 16, kernel=(64, 1)) │
│ BatchNorm → ELU → AvgPool                │
│ Dropout                                  │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│ Block 2: Separable Convolution           │
│ Depthwise Conv2D                         │
│ Pointwise Conv2D                         │
│ BatchNorm → ELU → AvgPool                │
│ Dropout                                  │
└─────────────────────────────────────────┘
   │
   ▼
Flatten → Linear → Class Prediction
Why EEGNet works well for EEG

EEGNet is designed around the structure of EEG data:

Temporal convolution learns frequency-related filters.
Depthwise spatial convolution learns spatial patterns across EEG channels.
Separable convolution reduces parameters while still allowing feature mixing.
The compact architecture helps reduce overfitting on small EEG datasets.
📁 Project Structure
eeg-motor-imagery/
├── app/
│   └── demo.py              # Streamlit demo: prediction, EEG plots, saliency
│
├── src/
│   ├── download_data.py     # Step 1: download PhysioNet EEG data
│   ├── preprocess.py        # Step 2: filter, epoch, normalize, save data
│   ├── model.py             # Step 3: EEGNet and ShallowConvNet architectures
│   └── train.py             # Step 4: LOSO cross-validation training
│
├── data/
│   ├── raw/                 # Raw EDF files, auto-created
│   └── processed/           # Processed .pkl files, auto-created
│
├── checkpoints/             # Trained model weights, auto-created
├── requirements.txt
├── .gitignore
└── README.md
🚀 Setup
Prerequisites
Python 3.9+
Around 2 GB free disk space
GPU optional
CPU training is possible, but Google Colab GPU is recommended for faster experiments
Step 1 — Clone the repository
git clone https://github.com/MunimUsmani/eeg-motor-imagery.git
cd eeg-motor-imagery
Step 2 — Install dependencies
pip install -r requirements.txt

Main dependencies:

mne
numpy
scipy
scikit-learn
torch
pandas
matplotlib
streamlit

If the streamlit command is not available on Windows, run Streamlit using:

python -m streamlit run app/demo.py
Step 3 — Download the dataset
python src/download_data.py

This downloads the PhysioNet EEG Motor Movement/Imagery dataset for subjects 1–9 using MNE.

Output files are saved to:

data/raw/

Example output:

Downloading PhysioNet EEG Motor Imagery Dataset
Subjects: 1-9 | Runs: left fist, right fist, feet

Downloading subject 01...
Subject 01: 3 runs saved
...
Done. Files saved to: data/raw/
Step 4 — Preprocess EEG data
python src/preprocess.py

The preprocessing pipeline:

Loads raw EDF files
Concatenates selected motor imagery runs
Standardizes EEG channel names
Applies 7–30 Hz bandpass filtering
Resamples EEG to 160 Hz
Extracts 4-second epochs from cue onset
Z-score normalizes each epoch independently
Saves per-subject and combined .pkl files

Output files are saved to:

data/processed/

Example processed files:

data/processed/subject_01.pkl
data/processed/subject_02.pkl
...
data/processed/subject_09.pkl
data/processed/all_subjects.pkl

The combined dataset used in training has shape:

(606, 64, 641)

Meaning:

606 epochs
64 EEG channels
641 time samples per epoch
Step 5 — Train EEGNet
python src/train.py

The training script runs Leave-One-Subject-Out cross-validation.

For each fold:

One subject is held out for testing
The model trains on the remaining subjects
The best checkpoint for that fold is saved

Checkpoints are saved to:

checkpoints/

Example checkpoint files:

checkpoints/fold_01.pt
checkpoints/fold_02.pt
...
checkpoints/fold_09.pt
checkpoints/results_summary.pkl

Example training result:

LOSO Results (EEGNET)
  Mean accuracy : 0.749 ± 0.062
  Mean kappa    : 0.330 ± 0.245
Step 6 — Run the Streamlit demo
streamlit run app/demo.py

Or on Windows, if streamlit is not recognized:

python -m streamlit run app/demo.py

The app opens at:

http://localhost:8501

In the demo, upload:

A processed subject file:
data/processed/subject_XX.pkl
The matching trained checkpoint:
checkpoints/fold_XX.pt

For example:

Upload subject file	Upload checkpoint
subject_01.pkl	fold_01.pt
subject_02.pkl	fold_02.pt
subject_03.pkl	fold_03.pt
subject_04.pkl	fold_04.pt
subject_05.pkl	fold_05.pt
subject_06.pkl	fold_06.pt
subject_07.pkl	fold_07.pt
subject_08.pkl	fold_08.pt
subject_09.pkl	fold_09.pt

The demo supports:

Epoch-by-epoch prediction
True label vs predicted label
Prediction confidence bars
EEG signal visualization
Evaluation across all epochs
Gradient-based channel saliency
🔬 Model Interpretability: Channel Saliency

The Streamlit app includes a gradient-based channel saliency feature.

This estimates which EEG channels most influence the model's prediction. In simple terms, if a small change in a channel strongly changes the model's output, that channel receives a higher saliency score.

The app displays:

Top attended EEG channels
Saliency scores
Saliency bar chart
Whether central motor-area channels appear in the top channels

For motor imagery, channels around the sensorimotor cortex are especially interesting, including:

C3, Cz, C4, FC3, FC4, CP3, CP4

If these appear among the most salient channels, it suggests the model may be relying on neurophysiologically relevant regions.

This should be interpreted carefully. Saliency indicates model sensitivity, not causal brain activity.

🧪 Google Colab Training Workflow

For faster training, the project can be trained on Google Colab.

Recommended workflow:

Run preprocessing locally
Upload the project folder to Google Drive
Open Google Colab
Mount Drive
Run training with GPU
Download the checkpoints/ folder

Example Colab commands:

from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/eeg-motor-imagery
!python src/train.py

Zip checkpoints after training:

!zip -r checkpoints.zip checkpoints

Download:

from google.colab import files
files.download("checkpoints.zip")
🧩 Key Design Decisions
Design Choice	Reason
7–30 Hz bandpass	Captures mu and beta frequency ranges commonly used in motor imagery EEG
160 Hz resampling	Reduces compute while preserving frequencies below 30 Hz
4-second epochs	Captures the motor imagery cue period
Z-score normalization per epoch	Reduces amplitude variation across epochs and subjects
LOSO cross-validation	Tests generalization to unseen subjects
Cohen's Kappa	More informative than accuracy alone for classification evaluation
Channel saliency	Adds interpretability beyond raw prediction accuracy
📌 Current Limitations
This project uses only subjects 1–9 from PhysioNet to keep the experiment lightweight.
The current setup uses a 2-class classification task based on T1/T2 annotations.
The Streamlit demo is for offline processed .pkl files, not real-time EEG streaming.
Gradient saliency is a qualitative interpretability method and should not be treated as proof of neural causality.
Model performance varies across subjects, which is expected in EEG due to high inter-subject variability.
The project is intended for research learning and prototyping, not clinical use.
🔮 Possible Extensions

Future improvements could include:

Training on all 109 PhysioNet subjects
Adding a strict run-specific left-vs-right hand classification setup
Supporting 4-class classification
Adding topographic scalp maps for saliency visualization
Comparing EEGNet with ShallowConvNet and DeepConvNet
Adding confusion matrices to the Streamlit demo
Exporting saliency plots for reports
Running experiments on BCI Competition IV Dataset 2a
Adding real-time EEG device support using OpenBCI or Lab Streaming Layer
📚 References
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J.
EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces.
Journal of Neural Engineering, 15(5), 056013, 2018.
DOI: 10.1088/1741-2552/aace8c
Schirrmeister, R. T., et al.
Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization.
Human Brain Mapping, 38(11), 5391–5420, 2017.
Goldberger, A. L., et al.
PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.
Circulation, 101(23), e215–e220, 2000.
🧠 Research Summary

This project implements EEGNet for EEG motor imagery classification and extends the basic implementation with a Streamlit-based interpretability interface.

The main contribution is not only reproducing the EEGNet architecture, but also providing a simple way to inspect model behavior through gradient-based EEG channel saliency. This allows qualitative analysis of whether the trained model attends to motor-area-related channels such as C3, Cz, and C4.

License

This project is intended for educational and research purposes.

Author
Abdul Munim Usmani
