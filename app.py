"""Streamlit front end: upload a brain MRI slice, get a prediction, a Grad-CAM
heatmap, and an MC-Dropout uncertainty estimate.

Research / educational demo only. Not a diagnostic device.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import (
    CKPT_PATH,
    FULL_NAME,
    MC_PASSES,
    download_checkpoint,
    gradcam,
    load_model,
    max_entropy,
    mc_predict,
    overlay,
    preprocess,
)

st.set_page_config(page_title="Brain Tumour MRI Classifier", layout="wide")


def _model_url() -> str:
    """Read the weights URL from Streamlit secrets or the environment."""
    try:
        if "MODEL_URL" in st.secrets:
            return st.secrets["MODEL_URL"]
    except Exception:
        pass
    return os.environ.get("MODEL_URL", "")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    if not CKPT_PATH.exists():
        url = _model_url()
        if not url:
            st.error(
                "No checkpoint found and MODEL_URL is not set. "
                "Add MODEL_URL to your Streamlit secrets."
            )
            st.stop()
        download_checkpoint(url)
    return load_model()


st.title("Brain Tumour MRI Classification")
st.caption(
    "Transfer-learning CNN with a Monte Carlo Dropout head, explained with Grad-CAM. "
    "Research and educational use only - this is not a diagnostic tool and must "
    "not be used for medical decisions."
)

with st.sidebar:
    st.header("Settings")
    passes = st.slider(
        "MC-Dropout passes (T)",
        min_value=5,
        max_value=40,
        value=MC_PASSES,
        step=5,
        help="More passes give a steadier uncertainty estimate but take longer.",
    )

uploaded = st.file_uploader("Upload an MRI slice", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a JPG or PNG image to see the prediction, heatmap and uncertainty.")
    st.stop()

image = Image.open(uploaded)
rgb01, input_tensor = preprocess(image)

model, classes = get_model()

with st.spinner(f"Running {passes} stochastic forward passes ..."):
    mean_probs, entropy, variance = mc_predict(model, input_tensor, passes=passes)

top_idx = int(np.argmax(mean_probs))
top_code = classes[top_idx]
label = FULL_NAME.get(top_code, top_code)
norm_entropy = entropy / max_entropy(len(classes))

c1, c2, c3 = st.columns(3)
c1.metric("Prediction", label)
c2.metric("Confidence", f"{mean_probs[top_idx]:.1%}")
c3.metric("Uncertainty", f"{norm_entropy:.1%}", help="Normalised predictive entropy")

if norm_entropy > 0.5:
    st.warning(
        "High predictive uncertainty - the model disagrees with itself across "
        "stochastic passes. Treat this prediction as unreliable."
    )

explain_idx = top_idx
with st.expander("Explain a different class instead"):
    explain_idx = int(
        st.selectbox(
            "Class to explain",
            options=list(range(len(classes))),
            index=top_idx,
            format_func=lambda i: (
                f"{FULL_NAME.get(classes[i], classes[i])} ({mean_probs[i]:.1%})"
            ),
        )
    )

with st.spinner("Computing Grad-CAM ..."):
    cam, _ = gradcam(model, input_tensor, class_idx=explain_idx)
    heatmap = overlay(rgb01, cam)

left, right = st.columns(2)
with left:
    st.image(rgb01, caption="Input (224x224)", use_container_width=True)
with right:
    st.image(
        heatmap,
        caption=f"Grad-CAM for {FULL_NAME.get(classes[explain_idx], classes[explain_idx])}",
        use_container_width=True,
    )

st.subheader("Class probabilities")
table = pd.DataFrame(
    {
        "Class": [FULL_NAME.get(c, c) for c in classes],
        "Probability": mean_probs,
    }
).sort_values("Probability", ascending=False)
st.bar_chart(table.set_index("Class"))

st.caption(
    f"Predictive entropy {entropy:.3f} (max {max_entropy(len(classes)):.3f}) - "
    f"summed variance {variance:.4f} over {passes} passes."
)
