# Brain Tumour MRI Classifier — Interactive Demo

An interactive front end for the brain tumour classification model, wrapping the
trained network in a browser interface that answers three questions about any MRI
slice you give it.

**This is a research artefact built for a dissertation. It has no regulatory
clearance, has never been validated on prospective clinical data, and must not
inform any decision about a real patient.**

---

## The three outputs

**What is it?** The network sorts the slice into one of four categories — glioma,
meningioma, pituitary tumour, or no tumour — and reports a probability for each.
Rather than showing only the winner, the interface ranks all four, because the gap
between first and second place carries as much information as the top score alone.

**Where did that come from?** A Grad-CAM heatmap is computed against the final
convolutional block and blended over the input. Warm regions contributed most to
the selected class. You can retarget the map at any of the four classes from the
sidebar, which is useful for asking why the network *rejected* an alternative.

**How sure is it?** Dropout layers are left switched on at inference and the slice
is pushed through the network repeatedly, so each pass sees a slightly different
sub-network. If those passes agree, the prediction is stable; if they scatter, the
model is guessing. That spread is summarised as predictive entropy and shown on a
scale from Decisive to No signal.

---

## Pipeline

```
MRI slice (any size, JPG/PNG)
      |
      v  resize 224x224, scale to [0,1], ImageNet normalisation
      |
      +--> T stochastic passes, dropout active ---> mean probabilities
      |                                             predictive entropy
      |                                             summed variance
      |
      +--> single deterministic pass, gradients on
                 |
                 v  hooks on model.features capture activations + gradients
                 v  channel weights = mean gradient, ReLU, bilinear upsample
                 v
           Grad-CAM map --> jet colormap --> alpha blend over slice
```

The classifier is a pretrained backbone with the classifier head replaced by
`Dropout → Linear(feat_dim, 256) → ReLU → Dropout → Linear(256, 4)`. Three
backbones are supported — ResNet-50, VGG-16 and EfficientNet-B0 — and the app picks
the right one automatically, because the training notebook stores the backbone name
alongside the weights.

---

## Running it

### Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

Ten tests should pass. They build the network from scratch with random weights, so
nothing is downloaded and the run finishes in seconds.

Then drop the trained `.pt` file into the project folder and launch:

```bash
streamlit run app.py
```

Keeping the checkpoint local is the simplest option while you're developing — the
app only reaches for the network when it can't find weights on disk.

### On Streamlit Community Cloud

Publish the checkpoint as a GitHub Release asset rather than committing it; a 96 MB
binary in version control slows every clone and sits close to GitHub's per-file
ceiling. Copy the asset's download link, then at
[share.streamlit.io](https://share.streamlit.io) point a new app at this repository
with `app.py` as the entry point and add one secret:

```toml
MODEL_URL = "https://github.com/<user>/<repo>/releases/download/v1.0/best_resnet50.pt"
```

Continuous deployment needs no configuration. Streamlit watches the default branch
and rebuilds whenever it changes, so the GitHub Actions workflow only has to gate
quality — `ruff` for style, `pytest` for behaviour — and Streamlit handles shipping.

---

## Reading the results honestly

The agreement scale is the part most worth understanding. Predictive entropy is
normalised against `ln(4)`, the entropy of a four-way coin flip, so the reading is a
percentage of the worst possible case rather than an absolute figure. Below 30% the
stochastic passes broadly concur. Above 60% they don't, and the interface says so
explicitly and names the runner-up, since "uncertain" is far less actionable than
"torn between glioma and meningioma".

Two caveats that the interface can't enforce. High agreement is not the same as
correctness — a network can be consistently and confidently wrong, particularly on
images unlike anything in training. And Grad-CAM shows where activation was strong,
not where the pathology is; a map centred on the skull or on a scanner artefact is
a warning about the model, not a finding about the patient.

---

## Files

```
app.py                      interface and layout
viewport.py                 renders scan panels with burned-in corner annotations
inference.py                model definition, preprocessing, MC-Dropout, Grad-CAM
tests/test_smoke.py         ten tests, no checkpoint required
.streamlit/config.toml      theme
.github/workflows/ci.yml    lint and test on every push
requirements.txt            runtime dependencies, CPU-only PyTorch
requirements-dev.txt        adds pytest and ruff
ruff.toml                   lint configuration
```

Note the absence of `packages.txt`. Grad-CAM is implemented directly with PyTorch
hooks and the colormap comes from matplotlib, so OpenCV never enters the dependency
tree and the build needs no system packages at all — which removes the most common
cause of failed deployments on Streamlit Cloud.

---

## Timings

Measured on a single CPU core, one 224×224 slice:

| Operation | Elapsed |
|---|---|
| Stochastic passes, T = 10 | 1.3 s |
| Stochastic passes, T = 20 | 2.5 s |
| Stochastic passes, T = 40 | 5.2 s |
| Grad-CAM | 1.5 s |

The sidebar exposes T as a slider so responsiveness can be traded against the
stability of the uncertainty estimate. The dissertation reports results at T = 30;
set the slider there if you want the app's figures to match the write-up.

---

## Known limits

Free Streamlit hosting allocates roughly 1 GB of memory. ResNet-50 and
EfficientNet-B0 both fit comfortably once PyTorch is loaded; VGG-16 carries around
528 MB of weights on its own and is likely to be evicted, so deploy one of the other
two even if VGG-16 scored better during training.

The `--extra-index-url` directive at the top of `requirements.txt` is load-bearing.
Without it, pip resolves the CUDA build of PyTorch — several gigabytes of GPU
libraries that will never be used on a CPU host — and the build exceeds its time
limit before installation finishes.

Idle apps are suspended after roughly twelve hours and take half a minute to wake,
plus a one-off checkpoint download on the first request after a cold start.

---

## Acknowledgements

Built as part of an M.Tech dissertation. Grad-CAM follows Selvaraju et al. (2017);
the Monte Carlo Dropout treatment of uncertainty follows Gal and Ghahramani (2016).
