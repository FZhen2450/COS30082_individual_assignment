"""Generate the readable local notebook; source logic lives in birdlab.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md('''
# COS30082 Assignment 1 - Local GPU Bird Classification

**Student name:** TODO &nbsp; **Student ID:** TODO

This replaces the mixed Colab/TensorFlow/PyTorch workflow with one local PyTorch
pipeline. Keep `birdlab.py` beside this notebook. Original files are preserved.

**Two architectures only:** MobileNetV3-Large and EfficientNetV2-S.
Each architecture has independent baseline / ablation runs; these are variants,
not additional architectures. Every run starts with the same pretrained weights
and seed. Never intentionally degrade the baseline or invent improvements.

Read `ASSIGNMENT_GUIDE.md` for the submission checklist and experiment rationale.
''')
md('''
## 1. Select the local GPU environment

Run `powershell -ExecutionPolicy Bypass -File .\\setup_gpu.ps1` in a terminal once.
In VS Code select `.venv\\Scripts\\python.exe` as this notebook's kernel, then
restart the kernel. The system `python` on this machine is Python 2.7: do not use it.

The project uses PyTorch 2.7.1 / torchvision 0.22.1 CUDA 11.8 wheels for the
installed NVIDIA driver. See [official installation commands](https://pytorch.org/get-started/previous-versions/).
No Google Drive, `/content`, TensorFlow, or `timm` is required.
''')
code('''
import os
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
assert (ROOT / "birdlab.py").exists(), "Open the notebook from the assignment project folder."
# Store downloaded ImageNet weights locally; no global cache changes.
os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")

import torch
import torchvision
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Run setup_gpu.ps1 and select the .venv kernel. CPU training is blocked by default."
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print("GPU calculation:", (torch.ones(8, device="cuda") ** 2).sum().item())
''')
md('''
## 2. Settings for the RTX 3050 4 GB

Images are 224 x 224 for both models. EfficientNetV2-S normally has a larger
recommended evaluation resolution; 224 is an explicit memory/accuracy tradeoff.
Batch size 8 x 4 accumulated batches = nominal effective batch 32. AMP is on.
If memory runs out, restart the kernel and use batch 4 / accumulation 8 in a
**new output directory**. Close other GPU programs. `num_workers=0` is intentional
for Windows notebooks and reproducible checkpoint continuation.

Each run uses 2 classifier-only epochs, then full fine-tuning (up to 15 total).
All runs share the same scheduler, BatchNorm policy and validation-based early
stopping; those common settings are not claimed as isolated ablation gains.
Training runs sequentially and can take hours. Start with one baseline cell;
do not run the full matrix unattended before checking its first learning curve.
''')
code('''
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from birdlab import (
    ARCHITECTURES, Config, experiment_plan, prepare_data, make_transform,
    seed_everything, train_experiment, results_table, plot_results,
    plot_history, final_evaluation, load_predictor, predict_image,
)

cfg = Config(
    data_root=str(ROOT), output_dir="artifacts", seed=42,
    image_size=224, batch_size=8, accumulation_steps=4, num_workers=0,
    epochs=15, head_epochs=2, patience=5,
    backbone_lr=3e-5, head_lr=3e-4, require_gpu=True, pretrained=True,
)
plan = experiment_plan()
display(pd.DataFrame([vars(experiment) for experiment in plan]))
''')
md('''
## 3. Audit training data; create one grouped, stratified validation split

The supplied annotations describe 4,829 training images and 200 classes. The
official 1,204-image Test set is not used for selecting weights or augmentation.
Approximately 80% of Train is used for fitting and 20% for validation.

Byte-identical training groups are reserved for fit before stratifying unique images;
some original labels conflict. These original labels are retained and documented,
not silently corrected. The exact file lists and training content fingerprint
are saved in `artifacts/split.json`. Changing the dataset/split requires a new
study output directory. Original images are never modified.
''')
code('''
fit_df, val_df, class_names, manifest = prepare_data(cfg)
print(f"Fit: {len(fit_df)} | Validation: {len(val_df)} | Species: {len(class_names)}")
assert not set(fit_df.sha256) & set(val_df.sha256)
print("Byte-identical train rows:", len(manifest["duplicate_training_rows"]))
display(pd.DataFrame(manifest["duplicate_training_rows"]))
counts = pd.DataFrame({"fit": fit_df.label.value_counts(), "validation": val_df.label.value_counts()}).sort_index()
display(counts.describe())
counts.plot(figsize=(12, 3), title="Examples per class (fit / validation)", xlabel="Class ID", ylabel="Images")
plt.show()
display(pd.DataFrame({"label": range(len(class_names)), "species": class_names}).head(10))
''')
md('''
## 4. Inspect augmentations

- Geometry: random crop/zoom, horizontal flip and mild rotation; no artificial
  change to a bird's body shape, which could change species cues.
- Color: mild ColorJitter. Strong color changes can damage fine-grained bird labels.
- Grayscale: randomly remove color from 20% of training samples, then keep three
  channels. This is **not infrared imaging**; RGB photographs contain no thermal data.
- Random erasing appears only in the combined recipe, so its isolated contribution
  cannot be inferred from this experiment matrix.

Only training images receive augmentation; validation and deployment are deterministic.
''')
code('''
from birdlab import MEAN, STD
seed_everything(cfg.seed)
with Image.open(fit_df.iloc[0].path) as source:
    source = source.convert("RGB")
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for ax, experiment in zip(axes, [plan[0], plan[1], plan[2], plan[3], plan[-1]]):
    image = make_transform(experiment, cfg.image_size)(source)
    image = image * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    ax.imshow(image.clamp(0, 1).permute(1, 2, 0))
    ax.set_title(experiment.name)
    ax.axis("off")
plt.tight_layout()
plt.show()
''')
md('''
## 5. MobileNetV3 baseline

Baseline: ImageNet pretrained model + fresh 200-class head, deterministic resize,
cross entropy, no dropout, no weight decay, no label smoothing, no stochastic depth.
It is a legitimate reference; it may already perform well. Loss of training fit,
or validation accuracy much lower than fit accuracy, must be interpreted from
the curves rather than assumed in advance.

The first run downloads official ImageNet weights into `.cache/torch`.
Running a cell again skips completed runs or resumes the last completed epoch.
''')
code('''
mobile_baseline = train_experiment(
    ARCHITECTURES[0], plan[0], cfg, fit_df, val_df, class_names, manifest)
display(pd.DataFrame([mobile_baseline]))
plot_history(mobile_baseline["checkpoint"])
''')
md('''
## 6. MobileNetV3 independent ablations and combined recipe

Each single-factor experiment resets to the same pretrained initialization.
It does not continue from the preceding experiment. A negative delta is valid
evidence that the tested method/settings did not help. The combined recipe is
predefined; it is not guaranteed to outperform all individual runs.
''')
code('''
for experiment in plan[1:]:
    train_experiment(ARCHITECTURES[0], experiment, cfg, fit_df, val_df, class_names, manifest)
display(results_table(cfg.output_dir))
plot_results(cfg.output_dir)
''')
md('''
## 7. EfficientNetV2-S baseline

Use the same split, resolution, epoch budget, seed and optimization policy.
EfficientNetV2-S has more parameters, so training may be slower. The original
EfficientNetV2-M / ConvNeXt experiments are intentionally not carried over.
''')
code('''
efficient_baseline = train_experiment(
    ARCHITECTURES[1], plan[0], cfg, fit_df, val_df, class_names, manifest)
display(pd.DataFrame([efficient_baseline]))
plot_history(efficient_baseline["checkpoint"])
''')
md('''
## 8. EfficientNetV2-S independent ablations and combined recipe
''')
code('''
for experiment in plan[1:]:
    train_experiment(ARCHITECTURES[1], experiment, cfg, fit_df, val_df, class_names, manifest)
display(results_table(cfg.output_dir))
plot_results(cfg.output_dir)
''')
md('''
## 9. Compare accuracy changes and diagnose overfitting (validation only)

Top-1 = correctly classified images / all images.
Average accuracy per class = mean of the 200 class recalls, each defined as
correct predictions for class c / images whose true label is c.
It is **not** one-vs-rest binary accuracy.

`delta_vs_baseline_pp` uses percentage points: 0.70 -> 0.75 is +5 pp, not +5%.
Fit accuracy and cross entropy are also measured in eval mode on unaugmented
training images, so dropout and augmentation do not make the gap misleading.
The loss plotted for every run is unsmoothed cross entropy, even when the
optimized objective includes label smoothing.

Report one-seed results as one-seed results, not proof of statistical significance.
No method is guaranteed to increase accuracy. Extra seeds can be run in separate
output directories before opening Test, if time permits.
''')
code('''
comparison = results_table(cfg.output_dir)
display(comparison[["architecture", "experiment", "val_top1", "val_average_per_class",
                    "delta_vs_baseline_pp", "fit_top1", "gap", "best_epoch", "training_seconds"]])
plot_results(cfg.output_dir)
for architecture in ARCHITECTURES:
    winner = comparison[comparison.architecture == architecture].sort_values(
        ["val_top1", "experiment"], ascending=[False, True]).iloc[0]
    print("Validation winner:", architecture, winner.experiment, f"{winner.val_top1:.2%}")
    plot_history(winner.checkpoint)
''')
md('''
## 10. Final evaluation on the official Test set

**Run only after all experiments and validation decisions are complete.**
This cell locks selection, then evaluates the baseline and validation winner for
each architecture (up to four checkpoints). The deployment model is the global
validation winner, regardless of test ranking. Do not tune further using Test.

The supplied Test set contains 4 byte-identical overlaps with Train. Report the
official complete-set metrics **and** the supplementary clean-subset metrics.
Test data and all original labels are retained. Reports include confusion matrix,
per-class accuracy, predictions, and correct/incorrect examples. Re-running reads
saved metrics; changing models after test access is blocked for this study.
''')
code('''
test_summary = final_evaluation(cfg, class_names, manifest)
display(test_summary)
print("Exported deployment/bird_model.pth and deployment/model_info.json")
''')
md('''
## 11. Test the exported checkpoint locally

The displayed scores come from real trained weights. Model loading uses the
saved label mapping and the same input preprocessing, with no ImageNet download.
''')
code('''
predictor, checkpoint = load_predictor("deployment/bird_model.pth", "cpu")
# Replace with your own bird photograph to demonstrate generalisation.
example = ROOT / "Test" / "0000.jpg"
with Image.open(example) as image:
    display(image.resize((320, 320)))
    display(predict_image(predictor, checkpoint, image))
del predictor
''')
md('''
## 12. Streamlit and submission

In a terminal run:
```
.venv\\Scripts\\python.exe -m streamlit run app.py
```
Upload photos, inspect Top-5 scores, and compare the experiment results.

For Hugging Face create a **Docker Space** and follow `HF_SPACE_README.md`.
Hosting requires uploading the actual `deployment/bird_model.pth` after training.

Submission checklist:
1. PDF report, at most 4 pages, with your name and ID; Methodology and Result and
   Discussion; both required test metrics, architecture/loss/hyperparameters,
   evidence of overfitting, ablation effects, error analysis and justified winner.
2. Source code ZIP, generated with `.venv\\Scripts\\python.exe tools/package_source.py`.
3. At most 10-minute video: concepts, model design, code, training/results and
   Hugging Face web application demonstration. Include its YouTube link and
   the Hugging Face link in the report.

Write the discussion after actual runs. Do not fill missing accuracy values with
assumed numbers. See `PROGRESS.md` for the current implementation status.
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (.venv GPU)", "language": "python", "name": "bird-local-gpu"},
    "language_info": {"name": "python", "version": "3.13"},
})
nbf.validate(notebook)
nbf.write(notebook, ROOT / "Bird_Classification_Local_GPU.ipynb")
print(f"Created notebook with {len(cells)} clean cells")
