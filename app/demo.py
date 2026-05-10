import os
import sys
import pickle
import streamlit as st
import numpy as np
import torch
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import EEGNet

st.set_page_config(
    page_title="EEG Motor Imagery BCI",
    page_icon="🧠",
    layout="wide",
)

CLASS_NAMES = {0: "Left fist", 1: "Right fist"}
CLASS_COLORS = {0: "#4A90D9", 1: "#E8593C"}
SFREQ = 160  

CHANNEL_NAMES_64 = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
    "Fp1", "Fpz", "Fp2",
    "Af7", "Af3", "Afz", "Af4", "Af8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "Ft7", "Ft8",
    "T7", "T8",
    "T9", "T10",
    "Tp7", "Tp8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "Po7", "Po3", "Poz", "Po4", "Po8",
    "O1", "Oz", "O2",
    "Iz",
]



def get_channel_names(n_channels: int):
    """Return channel names if known, otherwise fallback to Ch 0, Ch 1, ..."""
    if n_channels == len(CHANNEL_NAMES_64):
        return CHANNEL_NAMES_64

    return [f"Ch {i}" for i in range(n_channels)]


def compute_channel_saliency(model, X: np.ndarray, max_epochs: int = 50):
    """
    Compute gradient-based saliency for each EEG channel.

    Idea:
        If a small change in a channel strongly changes the model output,
        that channel receives higher saliency.

    Args:
        model: trained EEGNet model
        X: EEG epochs, shape (n_epochs, n_channels, n_times)
        max_epochs: number of epochs to use for faster saliency computation

    Returns:
        normalized channel saliency, shape (n_channels,)
    """

    model.eval()

    X_subset = X[:max_epochs]

    x_tensor = torch.from_numpy(X_subset).float().unsqueeze(1)

    x_tensor.requires_grad_(True)

    model.zero_grad()

    logits = model(x_tensor)

    predicted_classes = logits.argmax(dim=1)

    selected_scores = logits.gather(
        1,
        predicted_classes.view(-1, 1),
    ).sum()

    selected_scores.backward()

    gradients = x_tensor.grad

    channel_saliency = gradients.abs().mean(dim=(0, 1, 3)).detach().cpu().numpy()

    channel_saliency = channel_saliency / (channel_saliency.max() + 1e-8)

    return channel_saliency


def make_saliency_dataframe(channel_saliency: np.ndarray, n_channels: int):
    """Create sorted dataframe for saliency visualization."""
    channel_names = get_channel_names(n_channels)

    saliency_df = pd.DataFrame(
        {
            "Channel": channel_names,
            "Saliency": channel_saliency,
        }
    )

    saliency_df = saliency_df.sort_values(
        by="Saliency",
        ascending=False,
    ).reset_index(drop=True)

    return saliency_df



st.title("🧠 EEG Motor Imagery Decoder")
st.caption(
    "Implements **EEGNet** (Lawhern et al., J. Neural Engineering 2018) — "
    "decoding imagined hand movements from EEG signals."
)
st.divider()

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
    st.markdown(
        """
**EEGNet** is a compact CNN with only ~2,700 parameters, designed
specifically for EEG classification. It uses:

- **Temporal conv** → frequency filters
- **Depthwise conv** → spatial (CSP-like) filters
- **Separable conv** → efficient feature mixing

**Dataset**: PhysioNet EEG Motor Imagery
- 64 channels · 160 Hz · 4-second epochs
- Classes: left fist vs right fist imagery
"""
    )


if uploaded_data is None:
    st.info("👆 Upload a processed .pkl file to begin. Showing **synthetic demo** below.")

    st.subheader("How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Record EEG")
        st.markdown(
            "Subject imagines moving left or right hand while "
            "64-channel EEG is recorded."
        )

    with col2:
        st.markdown("### 2️⃣ Preprocess")
        st.markdown("Bandpass filter (7–30 Hz), epoch around cue, z-score normalize.")
    with col3:
        st.markdown("### 3️⃣ Classify")
        st.markdown(
            "EEGNet predicts the imagined movement class from the "
            "4-second EEG epoch."
        )

    st.divider()
    st.subheader("🔬 Synthetic Demo — try the model architecture")

    n_ch, n_t = 64, 641
    model = EEGNet(n_classes=2, n_channels=n_ch, n_samples=n_t)
    model.eval()

    if st.button("Generate random epoch and predict"):
        dummy_eeg = torch.randn(1, 1, n_ch, n_t)

        with torch.no_grad():
            logits = model(dummy_eeg)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        pred_class = int(probs.argmax())
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.metric("Predicted class", CLASS_NAMES[pred_class])
            for cls, name in CLASS_NAMES.items():
                st.progress(
                    float(probs[cls]),
                    text=f"{name}: {probs[cls] * 100:.1f}%",
                )

        with col_b:
            channels_to_show = ["Fc3", "Fc4", "C3", "Cz", "C4"]
            ch_indices = [0, 1, 4, 5, 6]
            time_axis = np.linspace(0, 4, n_t)
            eeg_data = dummy_eeg.squeeze().numpy()

            chart_data = pd.DataFrame(
                {
                    channels_to_show[j]: eeg_data[ch_indices[j]]
                    for j in range(len(ch_indices))
                },
                index=time_axis,
            )

            st.line_chart(chart_data, height=250)
            st.caption("Synthetic EEG signals: random data for architecture demo only.")

else:
    data = pickle.load(uploaded_data)

    X = data["X"]
    y = data["y"]
    subj = data.get("subject", "?")

    st.success(f"Loaded subject {subj} — {len(y)} epochs | shape {X.shape}")

    n_channels = X.shape[1]
    n_samples = X.shape[2]

    model = EEGNet(
        n_classes=2,
        n_channels=n_channels,
        n_samples=n_samples,
    )

    model_loaded = False

    if uploaded_model is not None:
        state = torch.load(uploaded_model, map_location="cpu")
        model.load_state_dict(state)
        model_loaded = True
        st.success("Model checkpoint loaded")
    else:
        st.warning("No checkpoint loaded — using random weights. Predictions are random.")

    model.eval()

    st.divider()
    st.subheader("Epoch Explorer")

    epoch_idx = st.slider("Select epoch", 0, len(y) - 1, 0)
    true_label = int(y[epoch_idx])
    epoch_data = X[epoch_idx]  

    x_tensor = torch.from_numpy(epoch_data).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_class = int(probs.argmax())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("True label", CLASS_NAMES[true_label])

    with col2:
        st.metric("Predicted", CLASS_NAMES[pred_class])

    with col3:
        correct = pred_class == true_label
        st.metric("Correct?", "✅ Yes" if correct else "❌ No")

    st.divider()

    st.subheader("Prediction confidence")

    for cls, name in CLASS_NAMES.items():
        st.progress(
            float(probs[cls]),
            text=f"{name}: {probs[cls] * 100:.1f}%",
        )

    st.subheader("EEG signal — selected epoch")

    time_axis = np.linspace(0, 4, n_samples)
    ch_to_show = min(6, n_channels)
    channel_names = get_channel_names(n_channels)

    chart_data = pd.DataFrame(
        {
            channel_names[i]: epoch_data[i]
            for i in range(ch_to_show)
        },
        index=time_axis,
    )

    st.line_chart(chart_data, height=280)

    st.caption(
        f"Showing first {ch_to_show} of {n_channels} channels. "
        f"Time: 0–4 seconds. Filtered: 7–30 Hz. "
        f"True class: **{CLASS_NAMES[true_label]}**"
    )

    st.divider()
    st.subheader("Run on all epochs")

    if st.button("Evaluate all epochs"):
        X_tensor = torch.from_numpy(X).float().unsqueeze(1)

        with torch.no_grad():
            all_logits = model(X_tensor)
            all_preds = all_logits.argmax(1).numpy()

        from sklearn.metrics import accuracy_score, cohen_kappa_score

        acc = accuracy_score(y, all_preds)
        kappa = cohen_kappa_score(y, all_preds)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc * 100:.1f}%")
        c2.metric("Cohen's Kappa", f"{kappa:.3f}")

    st.divider()
    st.subheader("Model Interpretability — Channel Saliency")

    st.markdown(
        """
This section computes **gradient-based saliency** to estimate which EEG channels
most influence the model's predictions.

In simple terms: if a small change in a channel strongly affects the model's
output, that channel receives a higher saliency score.
"""
    )

    if not model_loaded:
        st.warning(
            "Upload a trained checkpoint first. Saliency from random weights is not meaningful."
        )

    max_epochs_for_saliency = st.slider(
        "Number of epochs to use for saliency",
        min_value=5,
        max_value=len(y),
        value=min(50, len(y)),
        step=1,
        help="Using fewer epochs is faster. Using more epochs gives a more stable average.",
    )

    if st.button("Compute channel saliency"):
        with st.spinner("Computing gradient saliency..."):
            channel_saliency = compute_channel_saliency(
                model=model,
                X=X,
                max_epochs=max_epochs_for_saliency,
            )

            saliency_df = make_saliency_dataframe(
                channel_saliency=channel_saliency,
                n_channels=n_channels,
            )

        st.success("Channel saliency computed.")

        top_channels = saliency_df.head(5)["Channel"].tolist()
        top_channels_text = ", ".join(top_channels)

        st.markdown("### Top attended channels")
        st.success(f"Top 5 channels: {top_channels_text}")

        st.dataframe(
            saliency_df.head(15),
            use_container_width=True,
        )

        st.markdown("### Saliency bar chart")

        top_20_df = saliency_df.head(20).set_index("Channel")
        st.bar_chart(top_20_df)

        st.markdown(
            """
### Interpretation guide

For left/right hand motor imagery, central sensorimotor channels such as
**C3**, **Cz**, and **C4** are especially interesting.

If these channels appear near the top, it suggests the model may be relying
on neurophysiologically relevant EEG regions for motor imagery decoding.

This is not proof that the model has "learned the motor cortex", but it is
a useful qualitative sanity check.
"""
        )

        motor_channels = {"C3", "Cz", "C4", "Cp3", "Cp4", "Fc3", "Fc4"}
        detected_motor_channels = [
            ch for ch in saliency_df.head(10)["Channel"].tolist()
            if ch in motor_channels
        ]

        if detected_motor_channels:
            st.info(
                "Motor-area-related channels found in top 10: "
                + ", ".join(detected_motor_channels)
            )
        else:
            st.info(
                "No classic central motor-area channels appeared in the top 10. "
                "This can happen because EEG is noisy and saliency methods are approximate."
            )