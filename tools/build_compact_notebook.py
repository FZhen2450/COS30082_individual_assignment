"""Create a separate concise notebook without touching the detailed notebook."""
from pathlib import Path
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
md('''
# Bird classification — three paired rounds
**Name:** TODO **Student ID:** TODO

MobileNetV1 + EfficientNetV2-S; three rounds, six runs. Original notebook and
results are preserved. New results: **artifacts_compact_v1**. Stop other GPU
training, restart kernel and select `.venv\\Scripts\\python.exe` before running.

Goal: improve generalization and aim for >90% accuracy, **not a guarantee**.
Actual round scores may fall. The table separately tracks the best validation
checkpoint so far; its nondecreasing score is selection, not a fabricated gain.
Real uploaded photos can differ from this dataset; high confidence is not proof
of correctness, and the classifier only knows these 200 species.
''')
code('''
import os, sys, importlib
from pathlib import Path
ROOT = Path.cwd().resolve()
assert (ROOT / "birdlab_compact.py").exists(), "Open in the assignment folder"
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import torch
assert torch.cuda.is_available(), "Select .venv GPU kernel"
print(sys.executable, torch.cuda.get_device_name(0))
import birdlab_compact as lab
importlib.invalidate_caches()
importlib.reload(lab)
from IPython.display import display
cfg = lab.default_config()
display(lab.settings_table(cfg))
print("Outputs:", Path(cfg.output_dir).resolve())
''')
md('''
## Data and training policy
Same audited stratified split: 3,863 fit / 966 validation, all 200 classes.
Test stays held out. Every round restarts from original ImageNet weights and
seed; baseline trains a new head, later rounds unfreeze after 3 warm-up epochs.
BatchNorm running statistics stay fixed for the small GPU batches.

384px input, batch 2 × accumulation 16; AMP. This increases detail versus the
224px study, with greater runtime. All rounds use the same resolution. The
EfficientNetV2-S pretrained weights use 384px in their official inference recipe:
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html
Our square resize is shared by evaluation and deployment, not the official
aspect-preserving center-crop transform. MobileNet's original pretraining is 224px.

Round budgets are 15 / 35 / 60 epochs, with validation early stopping. This is
a comparison of complete training recipes: fine-tuning, epoch budget and
regularization also change, so gains cannot be attributed to augmentation alone.
No upside-down birds, large rotations, inversion, false-color layers or grayscale.
''')
code('''
fit, val, names, manifest = lab.prepare_data(cfg)
print("Fit:", len(fit), "Validation:", len(val), "Classes:", len(names))
assert not set(fit.sha256) & set(val.sha256)
''')
for i, text in enumerate([
    "Baseline: frozen backbone, new classifier, no random augmentation.",
    "Moderate augmentation + fine-tuning: crop retains 85–100% area, left/right flip, mild ColorJitter. No extreme transformations.",
    "Final optimized recipe: retain moderate augmentation; add dropout 0.2, weight decay 0.0001, smoothing 0.05, lower head LR and longer training.",
]):
    md(f"## Round {i + 1}\n{text}\n\nBoth models complete before the next round. Logs show Train/Val Accuracy and Loss. Negative changes stay visible.")
    code(f'''
round_results = lab.train_round({i}, cfg, fit, val, names, manifest)
display(lab.comparison_table(cfg.output_dir))
for checkpoint in round_results.checkpoint:
    print(Path(checkpoint).parent.parent.name)
    display(lab.history_table(checkpoint).round(4))
''')
md('''
## Freeze choices, evaluate Test and export the validation winner
Run only after all three rounds, then stop tuning from this Test set. Report
Top-1 and mean per-class accuracy, including official and overlap-excluded Test.
Best validation checkpoint wins even if an earlier round beats the final recipe.
Above-90% training accuracy alone does not meet the generalization goal.
''')
code('''
display(lab.comparison_table(cfg.output_dir))
display(lab.final_evaluation(cfg, names, manifest))
print("Deployment:", Path(cfg.output_dir) / "deployment/bird_model.pth")
''')
md('''
## Streamlit and submission
Run `.\\.venv\\Scripts\\python.exe -m streamlit run app_compact.py` in a terminal.
This app uses only the compact study. It prefers the exported validation winner.
Use clear photos of one bird belonging to the known 200 species; inspect Top-5
and compare with ground truth. Neither Top-1 confidence nor one successful photo
measures real-world accuracy. Do not present smoke-test scores as experiment results.

After training: `.\\.venv\\Scripts\\python.exe tools/package_compact_source.py`.
See README.md for settings, limitations and the separate source ZIP.
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (.venv GPU)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"}})
nbf.validate(nb)
nbf.write(nb, ROOT / "Bird_Classification_Compact_GPU.ipynb")
print(f"Generated separate compact notebook: {len(cells)} cells")
