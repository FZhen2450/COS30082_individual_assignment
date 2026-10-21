# COS30082 - Bird Species Classification

Local GPU training and a Streamlit demonstration for the assignment's 200 bird
species. The original `Data_loading1.ipynb` remains unchanged. Start with
**`Bird_Classification_Local_GPU.ipynb`**.

## Quick start (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_gpu.ps1
.\.venv\Scripts\python.exe -m jupyterlab
```

In VS Code select `.venv\Scripts\python.exe` as the notebook kernel. Run its cells
in order. A normal system `python` command selects Python 2.7 on this laptop.
The setup script deliberately uses Python 3.13 and an isolated project environment.

Required data (already present locally; excluded from Git):

```text
Train/       # 4,829 training JPEG images
Test/        # 1,204 official test JPEG images
train.txt    # filename integer_label, labels 0..199
test.txt
```

Dataset: the assignment-provided CUB-200 subset, not a newly downloaded split.

## Experiments

Two architectures: **MobileNetV1 (width 1.0)** and **EfficientNetV2-S**.
Eleven cumulative recipes: frozen-backbone baseline, then add backbone fine-tuning,
random crop, horizontal flip, rotation, color, dropout, weight decay, label smoothing,
random erasing and grayscale. Crop, flip and rotation are separate steps.
Each notebook stage trains BOTH architectures before comparing them: 22 runs.
Each recipe retains all earlier settings but restarts from the same architecture's
ImageNet weights and seed, rather than receiving extra epochs from an earlier run.
If the teacher's two-model limit means **two training runs total**, confirm that
interpretation before running the recipe sequence: controlled comparisons
require multiple training runs of the same architecture.

All recipes share the split, epoch budget, schedule, BatchNorm running-statistics
policy and validation early stopping. Stage 0 only trains the classifier; later
stages fine-tune after two classifier warm-up epochs. A baseline of 20-40% is an
expectation, not a guarantee or an enforced stopping target. Improvements can be
negative. Report both change from the previous recipe and total change from the
baseline; this cumulative design does not measure each method in isolation.

Before each stage, display the same validation photo plus five examples of the
new augmentation and five draws from the full cumulative recipe. Crop coordinates
appear on the original. Training samples new transforms on every image access,
not just five previews. Non-image methods show settings instead of invented pixel
filters. Preview RNG states are restored so viewing examples cannot change training.
Crop follows the common deterministic resize to avoid fixed-center fallback on
wide photos; preview boxes are mapped back onto the original coordinates.

After each paired training, Grad-CAM explains the same image and true class for
both models across groups of up to five checkpoints. This is class-related model
attention, not thermal sensing or a training method. Colors are normalized per map,
and do not prove an accuracy improvement. Missing stages remain blank. Explanations
only use completed checkpoints and are cached against image/weight signatures.

MobileNetV1 uses `timm==1.0.27`, model `mobilenetv1_100.ra4_e3600_r224_in1k`.
It contains depthwise/pointwise blocks, global average pooling and a linear head.
Public pretrained weights are downloaded once and cached locally. No hosting
account is required. MobileNetV1 mean/std is 0.5; EfficientNetV2-S uses its ImageNet
mean/std, preserved consistently in loaders and deployment. Different ImageNet
pretraining recipes limit claims that performance differences are architectural alone.

Default: 224px, batch 8, gradient accumulation 4, mixed precision, 15 epochs max
(2 head-only warm-up in fine-tuning recipes), backbone/head LR 3e-5/3e-4.
AMP starts at scale 128 for numerical stability and adjusts dynamically.
Windows DataLoader workers = 0.
The fixed low resolution is a documented compromise for 4 GB VRAM.
On CUDA OOM, restart the kernel and set batch 4 / accumulation 8 with a new
`output_dir`. Full experiments can take hours; check the first baseline first.

## Evaluation integrity

Only Train is partitioned into fit/validation. Byte-identical images are grouped
by reserving repeated-content groups for fit before stratifying unique images. Three identical-image pairs inside Train have conflicting
labels; these are preserved and recorded in `split.json`. The split is 3,863 fit / 966 validation images. Each class appears in both;
the six ambiguous duplicate rows are reserved for fit.

Four Test images are byte-identical to Train. The final evaluation reports both
the official complete-set result and the supplementary clean subset result.
Report this issue rather than silently changing the official test data.

Required metrics are overall **Top-1 accuracy** and **Average accuracy per class**
(macro recall across 200 true classes). Validation selects checkpoints and the
deployment model. Final test access locks the study against further training.
Use a new output directory for a new pre-test study; test results already seen
cannot become an unbiased validation source merely by changing the directory.

`artifacts/` contains split/configuration records, epoch histories, best/last
checkpoints, stage accuracy/delta plots, `comparison.csv`, `paired_comparison.csv`, and final test metrics,
per-class CSVs, confusion matrices and prediction examples. `last.pth` supports
continuation from the last finished epoch, including optimizer/scaler/RNG state.
Interrupted partially completed epochs are replayed. Completed experiments skip.
Checkpoints are local project artifacts; load only ones you trust.

Each epoch explicitly logs Train Loss, Train Accuracy (Top-1), Val Loss and
Val Accuracy (Top-1), plus validation average accuracy per class, best validation
accuracy and the clean train-eval generalization gap. Each stage shows an epoch
table and conventional train/validation curves. `history_table(checkpoint_path)`
and `plot_history(checkpoint_path)` read existing CSVs without retraining.
Loss plots use unsmoothed cross entropy; the label-smoothed optimization objective
is stored separately as `train_objective`. Required final metrics use Test data.

## Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

After training: upload photos, inspect the Top-5 species probabilities, browse
validation accuracy changes and training curves. Inference uses CPU so that the
GPU remains available for training. Before training the app explains that no
checkpoint exists; it does not show invented results.

The **Stage visuals** tab browses generated augmentation examples and Grad-CAM
progress panels from `artifacts/visualizations/`.

Final evaluation exports `deployment/bird_model.pth` with label mapping and
preprocessing metadata. The teacher has approved Streamlit in place of the
original Hugging Face requirement. Demonstrate the local application by running
the command above and opening the local URL printed in the terminal. A valid
final model requires completed training. No Hugging Face account is needed.

In the video, demonstrate photo upload, Top-5 predictions and experiment
comparisons. Include the YouTube link and Streamlit run instructions in the
report; an online application link can be added if you deploy it online.

## Submission and progress

See [ASSIGNMENT_GUIDE.md](ASSIGNMENT_GUIDE.md) for the assignment requirements
and report structure; see [PROGRESS.md](PROGRESS.md) for work completed so far.

```powershell
.\.venv\Scripts\python.exe tools/package_source.py
```

This builds `submission/COS30082_source.zip` with source and any available
experiment results, excluding images, environments, pretrained caches and weights.
Rebuild after training. Submit a separate report PDF and video link as required.

## References

- Assignment brief: `Assignment1_v2.pdf`, pages 1-3.
- [Official paired PyTorch / torchvision CUDA wheels](https://pytorch.org/get-started/previous-versions/).
- [MobileNetV1 paper](https://arxiv.org/abs/1704.04861).
- [MobileNetV1 pretrained model card](https://huggingface.co/timm/mobilenetv1_100.ra4_e3600_r224_in1k).
- [EfficientNetV2 documentation](https://docs.pytorch.org/vision/stable/models/efficientnetv2.html).
- [Grad-CAM paper](https://arxiv.org/abs/1610.02391).
- [RandomResizedCrop documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html).
- [NVIDIA CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

The pinned CUDA 11.8 build is chosen for the local RTX 3050 / driver 529.04,
instead of changing the existing global CPU-only PyTorch installation.
