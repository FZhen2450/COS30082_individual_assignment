# Progress

## Implemented

- Read all three assignment pages and documented submission requirements.
- Preserved the original notebook and datasets.
- Created the local GPU notebook and shared, commented PyTorch training module.
- Limited architectures to MobileNetV3-Large and EfficientNetV2-S.
- Added eight independent variants per architecture and validation delta charts.
- Added fixed grouped/stratified split, content fingerprints and duplicate reporting.
- Added AMP, gradient accumulation, early stopping, epoch checkpoints and resume.
- Added both required metrics, final-test selection lock, per-class analysis,
  confusion matrices and prediction examples.
- Added Streamlit photo upload, Top-5 inference and experiment comparisons.
- Added local setup, source packaging and Hugging Face Docker Space instructions.

## Validation status

Verified locally on 2026-10-21:

- Installed project-local PyTorch 2.7.1+cu118 / torchvision 0.22.1+cu118;
  CUDA detects the NVIDIA GeForce RTX 3050 Laptop GPU (4 GB).
- Confirmed GPU calculations and actual AMP training/backprop for both architectures.
- Downloaded both official ImageNet pretrained weight files into the local cache.
- Interrupted each tiny training run after an epoch checkpoint, resumed it, and
  verified completion of the next epoch with saved optimizer/scaler/RNG state.
- Loaded each saved model for CPU inference and checked its Top-5 predictions.
- Validated all 24 notebook cells and Python syntax; notebook outputs start clean.
- Verified the 3,863 / 966 split retains all 200 classes with no identical content
  across fit/validation; recorded six duplicate training rows.
- Streamlit AppTest: no exceptions in the untrained/empty state.
- Streamlit AppTest with smoke-only checkpoints: model loading, experiment
  tables, charts and learning-curve selection render without exceptions.
- Synthetic-fixture integration check: final validation selection lock, both
  200-class metrics, known-overlap exclusion, per-class/confusion/example reports,
  model export and cached rerun pass. The actual assignment Test set was not
  evaluated during these checks. Report figures also save without a GUI backend.

## GitHub progress

Source changes are kept on `codex/local-gpu-training`, based on the supplied
repository's `main` history. Images, environments, cached ImageNet weights,
smoke-test outputs and trained checkpoints are excluded from source control.

Smoke tests use only eight fit/four validation images, stored in `smoke_artifacts`,
not `artifacts`. They are pipeline checks, not assignment accuracy results. Their
scores must not be used in the report or deployed as a completed classifier.

## Remaining work for the assignment

- Run the full baseline and ablation matrix in the new notebook.
- Review actual learning curves and validation comparisons.
- Run final evaluation and export the real trained deployment checkpoint.
- Host the trained application on Hugging Face.
- Write the <=4-page report with actual results, name and student ID.
- Record the <=10-minute video and add both links to the report.
- Rebuild the source ZIP after training to include the final results.

No full training accuracy, final trained model or published Hugging Face URL is
claimed at the implementation stage.
