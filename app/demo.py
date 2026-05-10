"""
Step 5: Streamlit Demo App
Upload a preprocessed subject .pkl file → live EEG visualization + prediction

Run with: streamlit run app/demo.py
"""

import streamlit as st
import numpy as np
import torch
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import EEGNet

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Motor Imagery BCI",
    page_icon="🧠",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────
CLASS_NAMES  = {0: "Left fist", 1: "Right fist"}
CLASS_COLORS = {0: "#4A90D9", 1: "#E8593C"}
SFREQ        = 160   # Hz after resampling

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🧠 EEG Motor Imagery Decoder")
st.caption(
    "Implements **EEGNet** (Lawhern et al., J. Neural Engineering 2018) — "
    "decoding imagined hand movements from EEG signals."
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    uploaded_data = st.file_uploader(
        "Upload subject .pkl file",
        type=["pkl"],
        help="Files are in data/processed/subject_XX.pkl",
    )

    uploaded_model = st.file_uploader(
        "Upload model checkpoint (optional)",
        type=["pt"],
        help="Files are in checkpoints/fold_XX.pt",
    )

    st.divider()
    st.subheader("About")
    st.markdown("""
**EEGNet** is a compact CNN with only ~2,700 parameters, designed
specifically for EEG classification. It uses:

- **Temporal conv** → frequency filters
- **Depthwise conv** → spatial (CSP-like) filters
- **Separable conv** → efficient feature mixing

**Dataset**: PhysioNet EEG Motor Imagery
- 64 channels · 160 Hz · 4-second epochs
- Classes: left fist vs right fist imagery
""")

# ── Main area ──────────────────────────────────────────────────────────────
if uploaded_data is None:
    # Show demo mode with synthetic data
    st.info("👆 Upload a processed .pkl file to begin. Showing **synthetic demo** below.")

    st.subheader("How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Record EEG")
        st.markdown("Subject imagines moving left or right hand while 64-channel EEG is recorded.")
    with col2:
        st.markdown("### 2️⃣ Preprocess")
        st.markdown("Bandpass filter (7–30 Hz), epoch around cue, z-score normalize.")
    with col3:
        st.markdown("### 3️⃣ Classify")
        st.markdown("EEGNet predicts the imagined movement class from the 4-second EEG epoch.")

    st.divider()
    st.subheader("🔬 Synthetic Demo — try the model architecture")

    n_ch, n_t = 64, 641
    model = EEGNet(n_classes=2, n_channels=n_ch, n_samples=n_t)
    model.eval()

    if st.button("Generate random epoch and predict"):
        dummy_eeg = torch.randn(1, 1, n_ch, n_t)
        with torch.no_grad():
            logits = model(dummy_eeg)
            probs  = torch.softmax(logits, dim=1).squeeze().numpy()

        pred_class = int(probs.argmax())
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.metric("Predicted class", CLASS_NAMES[pred_class])
            for cls, name in CLASS_NAMES.items():
                st.progress(float(probs[cls]), text=f"{name}: {probs[cls]*100:.1f}%")

        with col_b:
            # Show a few EEG channels
            import pandas as pd
            channels_to_show = ["Fc3", "Fc4", "C3", "Cz", "C4"]
            ch_indices = [0, 1, 4, 5, 6]   # approximate channel indices
            time_axis  = np.linspace(0, 4, n_t)
            eeg_data   = dummy_eeg.squeeze().numpy()

            chart_data = pd.DataFrame(
                {f"Ch {i}": eeg_data[ch_indices[j]] for j, i in enumerate(ch_indices)},
                index=time_axis,
            )
            st.line_chart(chart_data, height=250)
            st.caption("Synthetic EEG signals (random — for architecture demo only)")

else:
    # Real data mode
    data = pickle.load(uploaded_data)
    X, y = data["X"], data["y"]
    subj = data.get("subject", "?")

    st.success(f"Loaded subject {subj} — {len(y)} epochs | shape {X.shape}")

    n_channels, n_samples = X.shape[1], X.shape[2]

    # Load model
    model = EEGNet(n_classes=2, n_channels=n_channels, n_samples=n_samples)
    if uploaded_model is not None:
        state = torch.load(uploaded_model, map_location="cpu")
        model.load_state_dict(state)
        st.success("Model checkpoint loaded")
    else:
        st.warning("No checkpoint loaded — using random weights (predictions are random)")
    model.eval()

    st.divider()
    st.subheader("Epoch Explorer")

    epoch_idx = st.slider("Select epoch", 0, len(y) - 1, 0)
    true_label = int(y[epoch_idx])
    epoch_data = X[epoch_idx]   # (n_ch, n_t)

    # Predict
    x_tensor = torch.from_numpy(epoch_data).unsqueeze(0).unsqueeze(0)  # (1,1,C,T)
    with torch.no_grad():
        logits = model(x_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_class = int(probs.argmax())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("True label",      CLASS_NAMES[true_label])
    with col2:
        st.metric("Predicted",       CLASS_NAMES[pred_class])
    with col3:
        correct = pred_class == true_label
        st.metric("Correct?", "✅ Yes" if correct else "❌ No")

    st.divider()

    # Probability bars
    st.subheader("Prediction confidence")
    for cls, name in CLASS_NAMES.items():
        st.progress(float(probs[cls]), text=f"{name}: {probs[cls]*100:.1f}%")

    # EEG plot
    st.subheader("EEG signal — selected epoch")
    import pandas as pd
    time_axis   = np.linspace(0, 4, n_samples)
    ch_to_show  = min(6, n_channels)
    chart_data  = pd.DataFrame(
        {f"Ch {i}": epoch_data[i] for i in range(ch_to_show)},
        index=time_axis,
    )
    st.line_chart(chart_data, height=280)
    st.caption(f"Showing first {ch_to_show} of {n_channels} channels. Time: 0–4 seconds. "
               f"Filtered: 7–30 Hz. True class: **{CLASS_NAMES[true_label]}**")

    # Run on all epochs
    st.divider()
    st.subheader("Run on all epochs")
    if st.button("Evaluate all epochs"):
        X_tensor = torch.from_numpy(X).unsqueeze(1)
        with torch.no_grad():
            all_logits = model(X_tensor)
            all_preds  = all_logits.argmax(1).numpy()

        from sklearn.metrics import accuracy_score, cohen_kappa_score
        acc   = accuracy_score(y, all_preds)
        kappa = cohen_kappa_score(y, all_preds)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc*100:.1f}%")
        c2.metric("Cohen's Kappa", f"{kappa:.3f}")