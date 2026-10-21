"""Generate paired cumulative MobileNetV1 / EfficientNetV2 experiments."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md('''
# COS30082 - MobileNetV1 vs EfficientNetV2-S, step by step

**Student name:** TODO &nbsp; **Student ID:** TODO

Two architectures only: width-1.0 **MobileNetV1** and **EfficientNetV2-S**.
MobileNetV1 uses depthwise + pointwise convolutions, global average pooling and
a 200-class linear head.

Run each stage cell in order. It trains BOTH models using the same recipe, then
shows their comparison. Each new recipe keeps ALL methods added previously.
Recipes restart from their own architecture's same ImageNet weights and seed;
they do not inherit the previous checkpoint. This controls for extra epochs.

Baseline accuracy of 20-40% is a hope, not a setting or a guaranteed result.
The baseline is legitimate frozen-backbone transfer learning. Scores may be
above or below that range; every gain or decline is recorded honestly.

Restart the kernel before running this revised workflow. Keep `birdlab.py` beside
the notebook. See `ASSIGNMENT_GUIDE.md` for rubric and submission details.
''')
md('''
## 1. Local GPU environment

Run `powershell -ExecutionPolicy Bypass -File .\\setup_gpu.ps1` once, then select
`.venv\\Scripts\\python.exe` as the kernel. The global `python` is Python 2.7.
This project uses PyTorch CUDA 11.8 and `timm==1.0.27` for actual MobileNetV1.
No Colab / Google Drive / TensorFlow is used. Public pretrained weights are
cached locally; no hosted app or hosting account is required for training.
''')
code('''
import os
import sys
from pathlib import Path
ROOT = Path.cwd().resolve()
assert (ROOT / "birdlab.py").is_file(), "Open this notebook in the project folder."
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import torch
import torchvision
import timm
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__, "timm:", timm.__version__)
assert torch.cuda.is_available(), "Run setup_gpu.ps1 and select the .venv GPU kernel."
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print("GPU check:", torch.ones(8, device="cuda").sum().item())
''')
md('''
## 2. Common settings and cumulative recipe table

Input 224px, batch 8, accumulation 4, AMP, Windows workers 0, max 15 epochs.
Both models start with ImageNet-1k pretrained weights (different pretraining
recipes; this is a comparison of available pretrained systems).
MobileNetV1 expects mean/std 0.5; EfficientNetV2 uses ImageNet mean/std.
The loaders and saved deployment metadata preserve each normalization.

Stage 0 trains only the linear classifier for the whole run, with a frozen
backbone. Stage 1 adds backbone fine-tuning after 2 head-only warm-up epochs.
Stages 2-10 keep fine-tuning and cumulatively add one setting at a time.
The seed, split, resolution, epoch budget, scheduler and BatchNorm policy stay
fixed. Early stopping and checkpoint selection use validation only.

22 runs total: 11 recipes x 2 architectures. Full training may take hours.
On OOM, restart the kernel, use batch 4 / accumulation 8, and use a new study
directory if any run has already been saved with different settings.
''')
code('''
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Image as DisplayImage
from birdlab import (
    ARCHITECTURES, Config, experiment_plan, prepare_data, normalization,
    make_transform, seed_everything, train_stage, results_table,
    paired_comparison, plot_results, plot_history, final_evaluation,
    load_predictor, predict_image,
)
from bird_visuals import fixed_example, preview_stage, plot_attention_progress
cfg = Config(
    data_root=str(ROOT), output_dir="artifacts", seed=42,
    image_size=224, batch_size=8, accumulation_steps=4, num_workers=0,
    epochs=15, head_epochs=2, patience=5,
    backbone_lr=3e-5, head_lr=3e-4, pretrained=True, require_gpu=True,
)
plan = experiment_plan()
display(pd.DataFrame([vars(recipe) for recipe in plan]))
''')
md('''
## 3. Fixed validation split and data-quality audit

4,829 training images, 200 species. Reserve six ambiguous duplicate training
rows for fit, then stratify unique images: 3,863 fit / 966 validation. Identical
content never crosses partitions. Keep original labels and document conflicts.
Save filenames and a content fingerprint in `artifacts/split.json`.
Official Test (1,204 images) is not used for model selection or tuning.
''')
code('''
fit_df, val_df, class_names, manifest = prepare_data(cfg)
assert not set(fit_df.sha256) & set(val_df.sha256)
print("Fit:", len(fit_df), "Validation:", len(val_df), "Species:", len(class_names))
display(pd.DataFrame(manifest["duplicate_training_rows"]))
counts = pd.DataFrame({"fit": fit_df.label.value_counts(), "validation": val_df.label.value_counts()}).sort_index()
display(counts.describe())
counts.plot(figsize=(12, 3), xlabel="Class ID", ylabel="Images", title="Fit / validation class counts")
plt.show()
''')
md('''
## 4. Pin ONE validation photo for every method and attention comparison

Crop, horizontal flip and rotation are separate stages, so the crop effect is
not confounded with two other new methods. Before each training stage, inspect
the new method on the SAME photo: original + 5 examples. Training is NOT limited
to those five: a random transformation is resampled each time an image is loaded.
The original dataset is not duplicated on disk or changed.

The second preview row shows five draws from the FULL cumulative recipe.
Grayscale/erasing effects are forced only in the isolated method illustrations;
full-recipe previews and training keep their real application probabilities.
Non-image methods (fine-tuning, dropout, weight decay, label smoothing) show an
explanation, not a fictional input-image transformation.

After each paired training cell, inspect the same photo's Grad-CAM for both
models across up to five stages. Target the same TRUE species each time, even
if predictions change. These heatmaps explain class-related regions, not physical
temperature. Grad-CAM does not train weights or increase accuracy by itself.
Untrained stages stay blank. Colors are normalized per map; a visually attractive
heatmap is not evidence of better accuracy. See https://arxiv.org/abs/1610.02391.
Validation, Test and Streamlit always classify deterministic, unaugmented inputs.
''')
code('''
# Set a specific validation filename before the first run if desired.
# Once saved, keep this same image/class for all stage comparisons.
EXAMPLE_FILENAME = None
example_image_path, example_class = fixed_example(cfg, val_df, EXAMPLE_FILENAME)
print("Fixed example:", example_image_path.name, "True species:", class_names[example_class])
display(DisplayImage(filename=str(example_image_path), width=380))
''')

descriptions = [
    ("Baseline: frozen backbone", "Only train the 200-class linear head. No random augmentation, dropout, weight decay or label smoothing. Both models use the same training budget; do not stop at a selected accuracy range."),
    ("Add backbone fine-tuning", "Keep the baseline preprocessing/loss. After 2 head-only warm-up epochs, unfreeze the backbone with a smaller learning rate. Compare with the frozen baseline for BOTH models."),
    ("Add random crop", "Keep fine-tuning; add RandomResizedCrop only. Display five sampled parts of the SAME image before training both models. New random crops continue to be generated during every epoch."),
    ("Add horizontal flip", "Keep fine-tuning + random crop; add horizontal flip with probability 0.5."),
    ("Add mild rotation", "Keep all previous methods; add random rotation from -10 to +10 degrees."),
    ("Add color augmentation", "Keep all previous methods; add mild ColorJitter."),
    ("Add dropout", "Keep all previous methods; add classifier dropout=0.3."),
    ("Add weight decay", "Keep all previous methods; add AdamW weight_decay=0.001."),
    ("Add label smoothing", "Keep all previous methods; add label_smoothing=0.1."),
    ("Add random erasing", "Keep all previous methods; add random erasing with probability 0.15."),
    ("Add grayscale experiment", "Keep all previous methods; add grayscale with probability 0.2. This tests reduced color dependence; it is not guaranteed to help."),
]
for index, (title, description) in enumerate(descriptions):
    md(f"## Stage {index}. {title}\n\n{description}\n\nRun this cell after the previous paired stage. Completed runs skip; interrupted runs resume from the last completed epoch. Both models appear before moving to the next recipe.")
    code(f'''
# SAME original + five examples of this method, then the full cumulative recipe.
for preview_path in preview_stage({index}, example_image_path, cfg, n_examples=5):
    display(DisplayImage(filename=str(preview_path)))
''')
    code(f'''
stage_results = train_stage({index}, cfg, fit_df, val_df, class_names, manifest)
display(stage_results[["architecture", "experiment", "val_top1", "val_average_per_class",
                       "delta_vs_previous_pp", "delta_vs_baseline_pp", "gap", "best_epoch"]].style.format({{
    "val_top1": "{{:.2%}}", "val_average_per_class": "{{:.2%}}", "gap": "{{:.2%}}",
    "delta_vs_previous_pp": "{{:+.2f}}", "delta_vs_baseline_pp": "{{:+.2f}}",
}}))
plot_results(cfg.output_dir)
''')
    first = (index // 5) * 5
    last = min(first + 5, len(descriptions))
    code(f'''
# Explanation only: compare real trained checkpoints on the unchanged photo.
# Untrained columns are placeholders; this does not modify or train the models.
attention_path = plot_attention_progress(
    cfg, example_image_path, example_class, stage_indices=range({first}, {last}))
display(DisplayImage(filename=str(attention_path)))
''')
    if index == 0:
        code('''
for checkpoint in stage_results.checkpoint:
    plot_history(checkpoint)
''')

md('''
## 5. Compare all paired stages (validation only)

Top-1 = correct images / all images. Average accuracy per class = mean recall
across the 200 true species. It is not one-vs-rest binary accuracy.

`delta_vs_previous_pp` measures the newly added method in the context of all
earlier methods. `delta_vs_baseline_pp` measures total change since stage 0.
A change from 30% to 40% is +10 percentage points. Declines remain visible.
This cumulative experiment does not measure each method in isolation or prove
statistical significance from one seed. The best recipe may be an EARLIER stage.

Fit-eval curves use unaugmented inputs with dropout off; compare these with
validation to diagnose overfitting. Loss curves use unsmoothed cross entropy
even when the optimized training objective uses label smoothing.
''')
code('''
comparison = results_table(cfg.output_dir)
display(paired_comparison(cfg.output_dir))
for architecture in ARCHITECTURES:
    winner = comparison[comparison.architecture == architecture].sort_values(
        ["val_top1", "experiment"], ascending=[False, True]).iloc[0]
    print("Validation winner:", architecture, winner.experiment, f"{winner.val_top1:.2%}")
    plot_history(winner.checkpoint)
# First five stages on exactly the same image and target class; later stages in
# additional panels. These are real model explanations, not synthetic thermal data.
for start in range(0, len(plan), 5):
    panel = plot_attention_progress(cfg, example_image_path, example_class,
                                    range(start, min(start + 5, len(plan))))
    display(DisplayImage(filename=str(panel)))
''')
md('''
## 6. Final Test evaluation - only after all paired stages

Lock the baseline and validation winner for each architecture before opening
Test. The deployment model is the global validation winner, regardless of test
ranking. No further training/tuning against the viewed Test results.

Report official complete-Test metrics and a supplementary clean subset excluding
four byte-identical Train/Test overlaps. Outputs include both required metrics,
per-class CSVs, confusion matrices and correct/incorrect prediction examples.
''')
code('''
test_summary = final_evaluation(cfg, class_names, manifest)
display(test_summary)
print("Exported deployment/bird_model.pth and deployment/model_info.json")
''')
md('''
## 7. Test the exported model
''')
code('''
predictor, checkpoint = load_predictor("deployment/bird_model.pth", "cpu")
with Image.open(ROOT / "Test/0000.jpg") as image:
    display(image.resize((320, 320)))
    display(predict_image(predictor, checkpoint, image))
del predictor
''')
md('''
## 8. Streamlit and submission

After training, run in a terminal:
```
.venv\\Scripts\\python.exe -m streamlit run app.py
```
Demonstrate image upload, Top-5 predictions, model selection and stage-by-stage
comparison charts. The teacher has approved Streamlit for the web application.

Submit a <=4-page PDF with name/ID, Methodology and Result and Discussion,
real experimental evidence and both required test metrics. Record a <=10-minute
video explaining models/code/results and demonstrating Streamlit; include the
YouTube link and Streamlit run instructions in the report.
Generate the source ZIP after training:
```
.venv\\Scripts\\python.exe tools/package_source.py
```
The ZIP excludes weights, image folders and environments. Do not use smoke-test
scores as assignment results. No final accuracy is claimed before full training.
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (.venv GPU)", "language": "python", "name": "bird-local-gpu"},
    "language_info": {"name": "python", "version": "3.13"},
})
nbf.validate(notebook)
nbf.write(notebook, ROOT / "Bird_Classification_Local_GPU.ipynb")
print(f"Created {len(cells)} cells: 11 paired cumulative stages with fixed-image previews and Grad-CAM")
