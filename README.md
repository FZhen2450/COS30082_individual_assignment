# COS30082 Bird Species Classification

Fa Zhen NG · 104400141

Two architectures: **MobileNetV1** and **EfficientNetV2-S**. Teacher-approved
**Streamlit** replaces the original Hugging Face demonstration requirement.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_gpu.ps1
```

Select `.venv\Scripts\python.exe` as the notebook kernel. Run one GPU training
task at a time; global `python` is Python 2.7 on this laptop.

| Workflow | Notebook | Results | Streamlit |
|---|---|---|---|
| Detailed, 13 paired stages | `Bird_Classification_Local_GPU.ipynb` | `artifacts_retrain_v3/` | `app.py` |
| Compact, 3 paired rounds | `Bird_Classification_Compact_GPU.ipynb` | `artifacts_compact_v1/` | `app_compact.py` |

The workflows have separate checkpoints and histories. Keep their default output
directories to avoid mixing results. Both use the same input files: `Train/`,
`Test/`, `train.txt`, `test.txt`. Original PDF and Colab notebook remain unchanged.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
# For the compact study instead:
.\.venv\Scripts\python.exe -m streamlit run app_compact.py
```

Both apps support photo uploads, Top-5 predictions and comparison/epoch tables.
They prefer the exported validation winner when available. A photo's probability
is not dataset accuracy; the model can only choose among its 200 known species.

## Experiments

Detailed: frozen baseline → fine-tuning → crop → horizontal flip → vertical flip
→ rotation → ColorJitter → inversion/tints → dropout → weight decay → label
smoothing → erasing → grayscale. Each step trains both architectures from the
same original ImageNet weights and seed. Settings are cumulative, so changes
measure conditional effects, not each method's isolated benefit.

Detailed defaults: 224px, batch 8, accumulation 4, 15 epochs maximum, 2 warm-up
epochs, patience 5, head LR 3e-4, backbone LR 3e-5, AMP, fixed BatchNorm statistics.

Compact: frozen baseline → moderate crop/flip/color plus fine-tuning → light
regularization and longer training. Defaults: 384px, batch 2, accumulation 16,
15/35/60 maximum epochs, 3 warm-up epochs, backbone LR 1e-5. Crops retain 85–100%
area; no inversion, vertical flips or large rotations. This compares whole recipes
with different budgets. Higher accuracy or a 90% target cannot be guaranteed.

Train/Val Accuracy and Loss appear every epoch. `Train Eval Accuracy` disables
random augmentation/dropout; its difference from validation is the clean gap.
Tables report accuracy in percent and changes in percentage points.

## Detailed study results

All 26 detailed runs and final evaluation are saved locally. Best validation
recipe: **EfficientNetV2-S with fine-tuning, random crop and horizontal flip**.

| Metric | Official Test | Test excluding four Train overlaps |
|---|---:|---:|
| Top-1 accuracy | 76.41% | 76.67% |
| Average accuracy per class | 75.90% | 76.17% |

Fit/validation: 3,863/966 images, 200 classes. Six conflicting duplicate training
rows stay in fit; identical content never crosses fit/validation. Test has 1,204
images. Selection uses validation; finalization locks choices before Test access.
Detailed report uses this study only, not compact or smoke-test results.

## Submission

The assignment requires a **report PDF of at most 4 pages**, name/ID, two main
sections (Methodology; Result and Discussion), both required metrics, one code ZIP,
and a presentation video of at most 10 minutes. Include the YouTube link and the
teacher-approved Streamlit demonstration details. The editable report is
`submission/COS30082_Report.docx`; add the video link after recording and export
to PDF before submission. The deleted guide/progress files were optional notes.

```powershell
.\.venv\Scripts\python.exe tools/package_source.py
# Separate compact source package:
.\.venv\Scripts\python.exe tools/package_compact_source.py
```

ZIPs include source and their own study's available CSV/JSON/PNG evidence, excluding
dataset photos, weights, environments and caches. Rebuild after changes. Deployment
weights are `<study>/deployment/bird_model.pth`; provide separately if requested.

## Project files

- `birdlab.py`, `bird_visuals.py`, `app.py`: detailed training, augmentation previews and app.
- `birdlab_compact.py`, `app_compact.py`: isolated compact recipe and app.
- `tools/`: notebook builders, packaging, verification and report generation.
- `submission/`: report and source ZIPs; submit files as required by Canvas.
- `logs/`, `smoke_artifacts/`: implementation checks, never assignment scores.
- `artifacts/`, `deployment/`: older local artifacts, preserved for reference.

ImageNet caches, datasets and local training outputs stay out of Git. Report
rendering intermediates stay in `tmp/report_qa/`.
