# Brain Tumour MRI Classification — Demo App

A minimal Streamlit app that serves the transfer-learning classifier from the brain
tumour notebook and returns three things for every MRI slice: the predicted class,
a **Grad-CAM** heatmap showing *where* the evidence is, and an **MC-Dropout**
estimate of *how certain* the model is.

> Research and educational demo only. Not a diagnostic device.

---

## What's here

| File | Purpose |
|---|---|
| `inference.py` | `TumorClassifier`, preprocessing, MC-Dropout, Grad-CAM, overlay |
| `app.py` | Streamlit UI |
| `tests/test_smoke.py` | Eight tests covering the pipeline, no checkpoint needed |
| `.github/workflows/ci.yml` | Lint + test on every push |
| `requirements.txt` | Pinned runtime deps, CPU-only PyTorch |
| `requirements-dev.txt` | Adds `pytest` and `ruff` |
| `ruff.toml` | Lint rules |

There is deliberately **no `packages.txt`**. Grad-CAM here is the notebook's own
hook-based implementation, so there's no OpenCV and no apt dependencies at all —
which is what makes the build reliable on Streamlit Cloud.

---

## Step 1 — Get the checkpoint out of Colab

The notebook writes `best_resnet50.pt` to
`MyDrive/brain_tumor_dissertation/checkpoints/`. Download it. It's about 96 MB.

The file is a dict containing `model`, `best_val`, `classes` and `backbone`, so the
app recovers the architecture and the label order from the checkpoint itself. If you
trained `vgg16` or `efficientnet_b0` instead, the same file just works — no code
changes needed.

## Step 2 — Create the GitHub repo

Create it **empty**: no README, no .gitignore, no licence. Ticking any of those
creates a commit on the remote that your local repo doesn't have, which is what
causes `failed to push some refs`.

```bash
git init
git add .
git commit -m "Streamlit app + CI for brain tumour MRI classifier"
git branch -M main
git remote add origin https://github.com/<you>/brain-tumor-mri-app.git
git push -u origin main
```

Make it **public** so GitHub Actions minutes are unlimited.

## Step 3 — Host the weights (not in git)

`.gitignore` excludes `*.pt` on purpose. On GitHub: **Releases → Draft a new release
→ tag `v1.0` → attach `best_resnet50.pt` → Publish**, then copy the asset URL:

```
https://github.com/<you>/brain-tumor-mri-app/releases/download/v1.0/best_resnet50.pt
```

## Step 4 — Run it locally first

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                        # should print 8 passed
cp ~/Downloads/best_resnet50.pt .
streamlit run app.py
```

Copying the checkpoint into the folder skips the download entirely, which also
avoids the macOS certificate problem that `urlretrieve` runs into.

## Step 5 — Deploy

Go to <https://share.streamlit.io>, connect GitHub, pick this repo, main branch,
`app.py`. Under **Advanced settings → Secrets**:

```toml
MODEL_URL = "https://github.com/<you>/brain-tumor-mri-app/releases/download/v1.0/best_resnet50.pt"
```

Streamlit rebuilds on every push to `main`, so CI checks the code and Streamlit
ships it. No deploy job, no secrets in Actions.

---

## Measured performance (CPU, single image)

| Operation | Time |
|---|---|
| MC-Dropout, T=10 | 1.3 s |
| MC-Dropout, T=20 (default) | 2.5 s |
| MC-Dropout, T=40 | 5.2 s |
| Grad-CAM | 1.5 s |

The sidebar slider lets you trade uncertainty stability against latency. T=20 is a
reasonable default; the notebook uses 30 for the reported results, so use that
setting if you want the app's numbers to line up with the dissertation.

## Known constraints

- **Memory.** Streamlit Community Cloud's free tier gives roughly 1 GB. ResNet-50
  plus PyTorch lands around 500–600 MB with the CPU-only wheel. `vgg16` is much
  heavier (~528 MB of weights alone) and will likely not fit — deploy `resnet50` or
  `efficientnet_b0`.
- **CPU wheels.** The `--extra-index-url` line in `requirements.txt` matters. Without
  it pip pulls the CUDA build (~2.5 GB) and the build times out. If the pinned
  resolution fails, drop the versions on `torch` / `torchvision` and keep the index line.
- **Cold start.** The app sleeps after about 12 hours idle and takes 30–60 s to wake,
  including the one-time checkpoint download.
- **Uncertainty is a guide, not a guarantee.** The app flags predictions above 50%
  normalised entropy as unreliable, but a confident wrong answer is still possible.
  Check the calibration numbers (ECE) from the notebook before trusting confidence
  values quantitatively.
