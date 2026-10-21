# Progress

## Implemented

- Read all three assignment pages and documented submission requirements.
- Preserved the original notebook and datasets.
- Created the local GPU notebook and shared, commented PyTorch training module.
- Limited architectures to MobileNetV1 (width 1.0) and EfficientNetV2-S.
- Added eleven cumulative recipes, each paired across BOTH architectures before
  advancing: frozen baseline, fine-tuning, random crop, horizontal flip, rotation,
  color, dropout, weight decay, label smoothing, erasing and grayscale. Total: 22 runs.
- Added original-plus-five examples before each stage and a second preview of
  the full cumulative recipe; runtime augmentation samples are not limited to five.
- Added fixed validation image/class, crop-box coordinates and real Grad-CAM
  evolution panels across groups of five trained checkpoints for both architectures.
- Added Streamlit Stage visuals to browse previews and model attention panels.
- Added changes from the previous recipe and baseline, plus paired CSV/charts.
- Baseline accuracy of 20-40% is an expectation only; no score is forced.
- Added fixed grouped/stratified split, content fingerprints and duplicate reporting.
- Added AMP, gradient accumulation, early stopping, epoch checkpoints and resume.
- Added both required metrics, final-test selection lock, per-class analysis,
  confusion matrices and prediction examples.
- Added Streamlit photo upload, Top-5 inference and experiment comparisons.
- Added local setup, source packaging and Streamlit demonstration instructions.
- Updated the submission workflow following the teacher's approval to use
  Streamlit instead of Hugging Face; removed platform-specific deployment files.
- Mapped the supplied 20-point rubric to report evidence, modeling validation,
  Streamlit demonstrations and a timed video outline in ASSIGNMENT_GUIDE.md.

## Validation status

Verified locally on 2026-10-21:

- Installed project-local PyTorch 2.7.1+cu118 / torchvision 0.22.1+cu118;
  CUDA detects the NVIDIA GeForce RTX 3050 Laptop GPU (4 GB).
- Confirmed GPU calculations and actual AMP training/backprop for both architectures.
- Verified the frozen baseline leaves backbone weights unchanged; fine-tuning
  and the final cumulative recipe update backbone weights for both architectures.
- Verified MobileNetV1 has 13 depthwise layers and 3,411,976 parameters with its
  200-class head. Cached local weights allow offline loading.
- Set a common conservative AMP initial scale (128) for stable first unfrozen
  batches; the saved run configuration records this numerical setting.
- Cached the ImageNet pretrained weights for MobileNetV1 and EfficientNetV2-S.
- Interrupted each tiny training run after an epoch checkpoint, resumed it, and
  verified completion of the next epoch with saved optimizer/scaler/RNG state.
- Loaded each saved model for CPU inference and checked its Top-5 predictions.
- Extended the notebook to 71 cells with explicit epoch metrics, preview, training/comparison and attention
  cells for each paired stage.
- Checked one-setting-at-a-time recipe accumulation, previous-stage ordering,
  paired CSV output and changes from the immediately previous recipe.
- Checked five sampled crop boxes, more than five distinct runtime crops, fixed
  validation example/class, unchanged source image and restored preview RNG states.
- Checked Grad-CAM against a known red-square CNN, including frozen parameters,
  empty positive maps, hook cleanup and restoration of model train/eval modes.
- Fixed stale-kernel ImportError for normalization by explicitly reloading both
  local modules before the notebook binds imported functions; the notebook
  builder includes the same fix.
- Rendered and inspected the crop examples and real-checkpoint attention panel;
  completed stages have real explanations and missing stages remain blank.
- Verified the 3,863 / 966 split retains all 200 classes with no identical content
  across fit/validation; recorded six duplicate training rows.
- Streamlit AppTest: no exceptions in the untrained/empty state.
- Streamlit AppTest with smoke-only checkpoints: model loading, experiment
  tables, charts, learning-curve selection, augmentation gallery and attention
  progress panels render without exceptions.
- Synthetic-fixture integration check: final validation selection lock, both
  200-class metrics, known-overlap exclusion, per-class/confusion/example reports,
  model export and cached rerun pass. The actual assignment Test set was not
  evaluated during these checks. Report figures also save without a GUI backend.

## GitHub progress

Source changes are kept on `codex/local-gpu-training`, based on the supplied
repository's `main` history. Images, environments, cached ImageNet weights,
smoke-test outputs and trained checkpoints are excluded from source control.

Smoke tests use only 17 fit/9 validation images, stored in `smoke_artifacts`,
not `artifacts`. They are pipeline checks, not assignment accuracy results. Their
scores must not be used in the report or deployed as a completed classifier.

## Remaining work for the assignment

- After the current stage finishes, rerun the notebook import cell to load the
  updated metric display helpers.
- Continue the remaining paired cumulative stages; completed runs are skipped.
- Review actual learning curves and validation comparisons.
- Run final evaluation and export the real trained deployment checkpoint.
- Run the trained Streamlit application and demonstrate uploads, predictions and
  experiment comparisons in the video.
- Write the <=4-page report with actual results, name and student ID.
- Record the <=10-minute video and add its YouTube link and Streamlit run
  instructions to the report (plus an app URL if deployed online).
- Rebuild the source ZIP after training to include the final results.

Training histories are now present for both architectures through the rotation
stage; later-stage training was in progress during the 2026-10-22 display update.
The display update keeps the existing training records and checkpoints and adds
explicit Train/Val Accuracy and Loss to logs, tables, plots and Streamlit.
Final Test accuracy is not claimed before the final evaluation is completed.
The report should include the YouTube link and Streamlit run instructions;
an online app link is optional if the application is deployed online.
