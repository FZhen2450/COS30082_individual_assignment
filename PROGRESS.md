# Progress — 2026-10-22 fresh retraining revision

The current workflow uses MobileNetV1 width 1.0 and EfficientNetV2-S with 13
cumulative recipes (26 runs). Source lives on `codex/local-gpu-training`.

- New notebook outputs reset; default study is `artifacts_retrain_v3`.
- Old `artifacts` checkpoints/results remain intact and are excluded from this study.
- Added distinct-size crop previews, wider random rotations, separate vertical
  flips, stronger ColorJitter, inversion and RGB/yellow overlay experiments.
- Fine-tuning and other non-pixel methods show settings instead of identical images.
- Model comparisons and per-epoch accuracy/loss use tables; Grad-CAM panels removed.
- Streamlit and final export now use only the current study directory.
- Submission packaging includes only new-study result artifacts, not old runs.

Small-sample verification uses **smoke_artifacts/retrain_v3**, never the formal
study: 17 fit / 9 validation images, 96px, two epochs, both real architectures.
These scores are implementation checks and must not appear as assignment results.
See `smoke_retrain_v3.log` and verification JSON files for actual check outcomes.
Synthetic finalization verifies metrics/export without opening assignment Test.
Visual verification checks crop/rotation ranges, actual color transforms and RNG.

Verification passed: both-architecture GPU training/freeze/unfreeze/resume/inference,
notebook validation, percentage-point comparison calculations, distinct crop and
angle previews, inversion/tint pixel values, source/RNG preservation, synthetic
200-class finalization and study-local export. Streamlit AppTest passed empty-study
isolation, checkpoint switching and accuracy/loss tables without comparison charts.
Crop and color preview PNGs were also visually inspected. Formal new-study training
was not started; no test-set scores or completed deployment are claimed.

Full revised experiments have not been run. Stop any old notebook training and
restart the kernel before starting the new study. Run all stages in order, inspect
real gains/declines, freeze selection, then evaluate Test and export. Finally check
Streamlit with trained weights, write the <=4-page report, record the <=10-minute
video, and rebuild the submission ZIP. No final accuracy or grade is promised.
