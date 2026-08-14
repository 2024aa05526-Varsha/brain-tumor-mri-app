"""Brain tumour MRI reading station.

Layout: a three-up summary strip across the top, then a split canvas with the
imaging on the left and the readout stacked on the right, so a whole study fits
on one screen without scrolling.

Research and educational demo only. Not a diagnostic device.
"""

import os

import numpy as np
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
from viewport import render_viewport

st.set_page_config(
    page_title="MRI Reading Station", layout="wide", initial_sidebar_state="expanded"
)

MONO = 'ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace'

CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1280px; }

  .rs-head { border-bottom: 2px solid #16202B; padding-bottom: 9px; margin-bottom: 22px; }
  .rs-head h1 { font-family: MONOFACE; font-size: 15px; letter-spacing: .2em;
    text-transform: uppercase; color: #16202B; margin: 0; font-weight: 700; }
  .rs-head p { font-size: 12px; color: #62727F; margin: 5px 0 0; }

  /* three-up summary strip */
  .rs-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
    background: #DCE2E8; border: 1px solid #DCE2E8; margin-bottom: 22px; }
  .rs-cell { background: #FFFFFF; padding: 15px 18px 17px; }
  .rs-cap { font-family: MONOFACE; font-size: 9.5px; letter-spacing: .19em;
    text-transform: uppercase; color: #8494A1; margin-bottom: 9px; }
  .rs-big { font-size: 25px; line-height: 1.15; color: #16202B; font-weight: 600;
    letter-spacing: -.01em; }
  .rs-big.accent { color: #2F5DA8; }
  .rs-sub { font-family: MONOFACE; font-size: 11px; color: #62727F; margin-top: 7px; }

  /* uncertainty scale */
  .rs-scale { position: relative; height: 8px; margin: 13px 0 7px; border-radius: 1px;
    background: linear-gradient(90deg,#2F5DA8 0%,#6E93C7 42%,#D9A05B 74%,#B4531F 100%); }
  .rs-mark { position: absolute; top: -5px; width: 2px; height: 18px;
    background: #16202B; box-shadow: 0 0 0 1.5px #FFFFFF; }
  .rs-ticks { display: flex; justify-content: space-between; font-family: MONOFACE;
    font-size: 9px; letter-spacing: .13em; text-transform: uppercase; color: #8494A1; }

  /* differential rows */
  .rs-panel { background: #FFFFFF; border: 1px solid #DCE2E8; padding: 16px 18px 18px; }
  .rs-row { display: grid; grid-template-columns: 1fr 52px; gap: 10px;
    align-items: baseline; margin-top: 13px; }
  .rs-row .n { font-size: 13px; color: #3B4956; }
  .rs-row .v { font-family: MONOFACE; font-size: 12px; color: #16202B; text-align: right; }
  .rs-row.top .n { color: #2F5DA8; font-weight: 600; }
  .rs-row.top .v { color: #2F5DA8; font-weight: 600; }
  .rs-track { grid-column: 1 / -1; height: 5px; background: #EDF0F3; margin-top: 5px; }
  .rs-fill { height: 100%; background: #9DB2CA; }
  .rs-row.top .rs-fill { background: #2F5DA8; }

  .rs-flag { border-left: 3px solid #B4531F; background: #FBF3EE; padding: 11px 14px;
    margin-top: 15px; font-size: 12.5px; color: #7A3A15; }
  .rs-note { font-family: MONOFACE; font-size: 10.5px; color: #8494A1; margin-top: 22px;
    border-top: 1px solid #DCE2E8; padding-top: 11px; }
  .rs-empty { background: #FFFFFF; border: 1px dashed #C9D2DB; padding: 46px 34px;
    text-align: center; }
  .rs-empty p { color: #62727F; font-size: 13px; margin: 0; }
</style>
""".replace("MONOFACE", MONO)
st.markdown(CSS, unsafe_allow_html=True)


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
                "No checkpoint found. Set MODEL_URL in Streamlit secrets, or place "
                "the .pt file next to app.py."
            )
            st.stop()
        download_checkpoint(url)
    return load_model()


def cell(caption: str, value: str, sub: str = "", accent: bool = False) -> str:
    klass = "rs-big accent" if accent else "rs-big"
    tail = f'<div class="rs-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="rs-cell"><div class="rs-cap">{caption}</div>'
        f'<div class="{klass}">{value}</div>{tail}</div>'
    )


st.markdown(
    '<div class="rs-head"><h1>MRI Reading Station</h1>'
    "<p>Tumour classification with Grad-CAM evidence and Monte Carlo Dropout "
    "uncertainty. Research use only, not for clinical decisions.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption("STUDY")
    uploaded = st.file_uploader("Scan file", type=["jpg", "jpeg", "png"])
    st.caption("ACQUISITION")
    passes = st.slider(
        "Stochastic passes (T)", 5, 40, MC_PASSES, step=5,
        help="More passes steady the uncertainty estimate but cost time.",
    )
    alpha = st.slider(
        "Overlay opacity", 0.1, 0.8, 0.4, step=0.05,
        help="How strongly the Grad-CAM heatmap is blended over the scan.",
    )

if uploaded is None:
    st.markdown(
        '<div class="rs-empty"><p>Open a scan from the sidebar to read out a '
        "prediction, its evidence map, and how much the model disagrees with "
        "itself across stochastic passes.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

image = Image.open(uploaded)
rgb01, input_tensor = preprocess(image)
model, classes = get_model()

with st.spinner(f"Running {passes} stochastic passes ..."):
    mean_probs, entropy, variance = mc_predict(model, input_tensor, passes=passes)

top_idx = int(np.argmax(mean_probs))
label = FULL_NAME.get(classes[top_idx], classes[top_idx])
norm_entropy = float(entropy / max_entropy(len(classes)))
order = list(np.argsort(mean_probs)[::-1])
runner = order[1]
agreement = "Decisive" if norm_entropy < 0.3 else "Split" if norm_entropy < 0.6 else "No signal"

with st.sidebar:
    st.caption("EVIDENCE MAP")
    explain_idx = int(
        st.selectbox(
            "Class to explain",
            options=list(range(len(classes))),
            index=top_idx,
            format_func=lambda i: (
                f"{FULL_NAME.get(classes[i], classes[i])}  {mean_probs[i]:.0%}"
            ),
        )
    )

# --- summary strip -----------------------------------------------------
st.markdown(
    '<div class="rs-strip">'
    + cell("Finding", label, f"of {len(classes)} classes", accent=True)
    + cell("Mean confidence", f"{mean_probs[top_idx]:.1%}", f"over {passes} passes")
    + cell("Model agreement", agreement, f"entropy {norm_entropy:.0%} of maximum")
    + "</div>",
    unsafe_allow_html=True,
)

with st.spinner("Computing Grad-CAM ..."):
    cam, _ = gradcam(model, input_tensor, class_idx=explain_idx)
    heat = overlay(rgb01, cam, alpha=alpha)

explained = FULL_NAME.get(classes[explain_idx], classes[explain_idx])
FRAME = (220, 226, 232)

# --- split canvas: imaging left, readout right -------------------------
canvas, readout = st.columns([2, 1], gap="large")

with canvas:
    a, b = st.columns(2, gap="small")
    a.image(
        render_viewport(
            rgb01,
            top_left=f"{uploaded.name[:20]}\nRESAMPLED 224",
            top_right="SOURCE",
            bottom_right=f"T = {passes}",
            border=FRAME,
        ),
        use_container_width=True,
    )
    b.image(
        render_viewport(
            heat,
            top_left=f"GRAD-CAM\n{explained.upper()}",
            top_right=f"P = {mean_probs[explain_idx]:.1%}",
            bottom_left="TARGET: features",
            bottom_right=f"ALPHA {alpha:.2f}",
            border=FRAME,
        ),
        use_container_width=True,
    )
    st.markdown(
        f'<div class="rs-note">Predictive entropy {entropy:.3f} of '
        f"{max_entropy(len(classes)):.3f} max &nbsp;·&nbsp; summed variance "
        f"{variance:.4f} &nbsp;·&nbsp; backbone {model.backbone_name}</div>",
        unsafe_allow_html=True,
    )

with readout:
    rows = []
    for i in order:
        name = FULL_NAME.get(classes[i], classes[i])
        pct = float(mean_probs[i]) * 100
        klass = "rs-row top" if i == top_idx else "rs-row"
        rows.append(
            f'<div class="{klass}"><div class="n">{name}</div>'
            f'<div class="v">{pct:.1f}%</div>'
            f'<div class="rs-track"><div class="rs-fill" style="width:{pct:.1f}%">'
            "</div></div></div>"
        )
    st.markdown(
        '<div class="rs-panel"><div class="rs-cap">Differential</div>'
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )

    mark = min(max(norm_entropy, 0.0), 1.0) * 100
    st.markdown(
        '<div class="rs-panel" style="margin-top:14px">'
        '<div class="rs-cap">Agreement across passes</div>'
        f'<div class="rs-scale"><div class="rs-mark" style="left:{mark:.1f}%"></div></div>'
        '<div class="rs-ticks"><span>Decisive</span><span>Split</span>'
        "<span>No signal</span></div></div>",
        unsafe_allow_html=True,
    )

    if norm_entropy > 0.5:
        st.markdown(
            '<div class="rs-flag">The model disagrees with itself across passes. '
            f"Second candidate is {FULL_NAME.get(classes[runner], classes[runner])} "
            f"at {mean_probs[runner]:.1%}. Read this result as unreliable.</div>",
            unsafe_allow_html=True,
        )
