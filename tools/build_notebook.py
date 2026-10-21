"""Generate the fresh, table-based paired augmentation study."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))

md('''
# COS30082: MobileNetV1 vs EfficientNetV2-S — fresh paired experiments
**Student name:** TODO **Student ID:** TODO

Run in order. Each stage trains BOTH architectures before the next stage.
13 cumulative recipes × 2 architectures = 26 runs. Each recipe restarts from
the same original ImageNet weights and seed, not the previous bird checkpoint.
Baseline accuracy is measured honestly; 20–40% cannot be guaranteed or forced.

This revision writes only to **artifacts_retrain_v3**. Old `artifacts` runs are
preserved and excluded from new comparisons and Streamlit. Stop old training and
**restart the kernel** before starting. See ASSIGNMENT_GUIDE.md for submission.
''')
md('''
## 1. Local GPU environment
Run `powershell -ExecutionPolicy Bypass -File .\\setup_gpu.ps1` once and select
`.venv\\Scripts\\python.exe` as the kernel. Global Python is 2.7.
Full training takes hours. Use one GPU training process at a time.
''')
code('''
import os, sys, importlib
from pathlib import Path
ROOT = Path.cwd().resolve()
assert (ROOT / "birdlab.py").is_file(), "Open the notebook in the project folder"
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import torch, torchvision, timm
print("Python:", sys.executable)
print("Versions:", torch.__version__, torchvision.__version__, timm.__version__)
assert torch.cuda.is_available(), "Select the .venv GPU kernel"
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
''')
md('''
## 2. Initialization and fair comparison
Both backbones are **already pretrained on ImageNet**. Stage 0 freezes the
backbone and trains a new 200-species linear head. Stage 1 additionally unfreezes
the backbone after two head-only warm-up epochs, at a lower learning rate.
This is fine-tuning even on your first bird-data training; new classes or a
previous bird checkpoint are not required. Fine-tuning changes weights, not
input pixels, so it has no five-image preview.
Reference: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Later recipes retain earlier settings. Split, seed, resolution, 15-epoch maximum,
early-stopping policy, scheduler and fixed BatchNorm statistics are shared.
The two pretrained backbones have different pretraining histories. On OOM use
batch 4 / accumulation 8 in a fresh output folder.
''')
code('''
import pandas as pd
from PIL import Image
from IPython.display import display, Image as DisplayImage
import birdlab
importlib.invalidate_caches()
importlib.reload(birdlab)
import bird_visuals
importlib.reload(bird_visuals)
from birdlab import (
    ARCHITECTURES, DEFAULT_OUTPUT_DIR, Config, experiment_plan, prepare_data,
    train_stage, results_table, paired_comparison, comparison_table,
    stage_configuration, history_table, final_evaluation, load_predictor, predict_image,
)
from bird_visuals import fixed_example, preview_stage
cfg = Config(
    data_root=str(ROOT), output_dir=DEFAULT_OUTPUT_DIR, seed=42,
    image_size=224, batch_size=8, accumulation_steps=4, num_workers=0,
    epochs=15, head_epochs=2, patience=5, backbone_lr=3e-5, head_lr=3e-4,
    pretrained=True, require_gpu=True,
)
plan = experiment_plan()
print("CURRENT STUDY:", Path(cfg.output_dir).resolve())
display(pd.DataFrame([vars(recipe) for recipe in plan]))
''')
md('''
## 3. Fixed split and data audit
4,829 training rows, 200 species. Six ambiguous duplicate rows remain in fit;
stratifying unique images gives 3,863 fit / 966 validation images. Identical
content cannot cross this boundary. Original labels are preserved and audited.
The official 1,204 Test images are held out until all model choices are frozen.
''')
code('''
fit_df, val_df, class_names, manifest = prepare_data(cfg)
assert not set(fit_df.sha256) & set(val_df.sha256)
print("Fit:", len(fit_df), "Validation:", len(val_df), "Species:", len(class_names))
display(pd.DataFrame(manifest["duplicate_training_rows"]))
display(pd.DataFrame({"fit": fit_df.label.value_counts(),
                      "validation": val_df.label.value_counts()}).sort_index())
''')
md('''
## 4. Augmentation examples; model comparisons use tables only
Use ONE fixed validation photo purely to illustrate changes. Data stays intact.
Five displayed crops span different area bands and positions; five rotations
span different angle bands. This deliberate preview selection makes the range
visible. Actual training independently resamples on every access in every epoch,
with far more than five possibilities.
Horizontal = left/right; vertical = up/down. Each flip has two states, so its
isolated preview shows original + flipped, not five duplicates. The second row
shows five draws of the full cumulative recipe. Non-pixel methods show settings.
Grad-CAM panels and model comparison graphs are removed; epoch metrics use tables.

Vertical flips, large rotations, false colors and grayscale can remove useful
bird cues. These are experiments, not guaranteed improvements. Keep declines
in the report; deploy the validation winner, not automatically the last stage.
Inverted/tinted RGB photos are **not infrared or thermal measurements**.
''')
code('''
EXAMPLE_FILENAME = None  # Optional validation filename; keep fixed once selected.
example_image_path, example_class = fixed_example(cfg, val_df, EXAMPLE_FILENAME)
print("Example:", example_image_path.name, "True species:", class_names[example_class])
display(DisplayImage(filename=str(example_image_path), width=380))
''')
descriptions = [
    ("Frozen-backbone baseline", "Train only the new linear classifier; no random augmentation or added regularization."),
    ("Backbone fine-tuning", "Two head-only warm-up epochs, then unfreeze backbone at a smaller learning rate. Input images stay identical."),
    ("Random resized crop", "Area 40–100%, aspect ratio 0.75–1.33. Five examples cover different sizes/positions; runtime crops are independently random."),
    ("Horizontal flip", "Left/right reversal, probability 0.5."),
    ("Vertical flip experiment", "Up/down reversal, probability 0.2; this may be unnatural for upright birds."),
    ("Random rotation", "Uniform random angle −45° to +45°. Five previews cover different angle bands."),
    ("Stronger ColorJitter", "Brightness ±0.4, contrast ±0.4, saturation ±0.5, hue ±0.1; random strengths and order."),
    ("Inversion and color-layer experiment", "Total probability 0.25: inversion or red/green/blue/yellow overlay. Conditional weights 10/25/25/25/15%; tint opacity random 20–45%. Preview forces each effect."),
    ("Classifier dropout", "Add dropout p=0.3; feature regularization, not an image transformation."),
    ("Weight decay", "Add AdamW weight decay 0.001."),
    ("Label smoothing", "Add smoothing 0.1. Displayed Train Loss remains unsmoothed cross-entropy for comparability."),
    ("Random erasing", "Probability 0.25, area 2–15%, fill with model normalization mean."),
    ("Grayscale experiment", "Probability 0.2; reduced color dependence can help or hurt."),
]
for i, (title, description) in enumerate(descriptions):
    md(f"## Stage {i}: {title}\n\n{description}\n\nRetain all earlier settings. Run after the previous paired stage. Completed runs skip; interrupted runs resume from the last saved epoch in THIS study only.")
    code(f'''
display(stage_configuration({i}, cfg))
for preview_path in preview_stage({i}, example_image_path, cfg, n_examples=5):
    display(DisplayImage(filename=str(preview_path)))
''')
    code(f'''
stage_results = train_stage({i}, cfg, fit_df, val_df, class_names, manifest)
display(comparison_table(cfg.output_dir))
for checkpoint in stage_results.checkpoint:
    print(Path(checkpoint).parent.parent.name, Path(checkpoint).parent.name)
    display(history_table(checkpoint).round(4))
''')
md('''
## 5. Validation comparison table
Top-1 = correct images / all images. Average per-class accuracy = macro recall
over the 200 species. Accuracy columns are percentages. Changes are percentage
points: 30% → 40% = +10 pp. Previous-stage delta measures the added method GIVEN
earlier methods; it is not its isolated effect. Baseline delta is cumulative.
These are single-seed observations; keep both gains and declines.

Every epoch logs Train Accuracy, Train Loss, Val Accuracy and Val Loss.
Online Train Accuracy includes training augmentations; Train Eval Accuracy uses
deterministic inputs. Its difference from Val Accuracy is the clean gap.
Best identifies the validation-selected epoch, not another accuracy metric.
''')
code('''
display(comparison_table(cfg.output_dir))
paired_comparison(cfg.output_dir)  # Separate machine-readable fraction-valued CSV.
all_results = results_table(cfg.output_dir)
if not all_results.empty:
    winner = all_results.sort_values("val_top1", ascending=False).iloc[0]
    print("Validation winner:", winner.architecture, winner.experiment, f"{winner.val_top1:.2%}")
print("Readable table:", Path(cfg.output_dir) / "comparison_display.csv")
''')
md('''
## 6. Final Test — only after all stages
Freeze all choices first. Evaluate both baselines and each architecture's best
validation recipe. Four known Train/Test overlaps are reported, alongside clean
metrics excluding them. The selection lock prevents tuning from repeated Test
feedback. Exports stay inside the current study.
''')
code('''
test_results = final_evaluation(cfg, class_names, manifest)
display(test_results)
print("Export:", Path(cfg.output_dir) / "deployment/bird_model.pth")
''')
md('''
## 7. Streamlit and photo testing
Streamlit reads only this study. Before new checkpoints exist it shows a pending
message. Upload a photo for Top-5 predictions and inspect tables/augmentations.
In another terminal: `.\\.venv\\Scripts\\python.exe -m streamlit run app.py`
''')
code('''
PHOTO_PATH = None  # Set your image filename after training.
if PHOTO_PATH:
    model, checkpoint = load_predictor(Path(cfg.output_dir) / "deployment/bird_model.pth", "cpu")
    with Image.open(PHOTO_PATH) as photo:
        display(photo.copy())
        display(predict_image(model, checkpoint, photo))
''')
md('''
## 8. Submission
Follow ASSIGNMENT_GUIDE.md: report PDF (Methodology; Result and Discussion), code
ZIP and presentation video/link. Fill name/ID. Teacher approved Streamlit; keep
that approval. After training run `.\\.venv\\Scripts\\python.exe tools/package_source.py`
to include this study's results. Dataset images and model weights are excluded;
provide trained weights separately if requested.
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (.venv GPU)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
})
nbf.validate(nb)
nbf.write(nb, ROOT / "Bird_Classification_Local_GPU.ipynb")
print(f"Generated {len(cells)} cells; fresh outputs and table-only comparisons.")
