"""Three-round study using a private engine; original workflow stays untouched.

The private module has its own globals so changing this study's plan/transforms
does not monkey-patch an imported birdlab or change an active original notebook.
"""
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from torchvision import transforms as T

_name = "_birdlab_compact_engine"
_spec = importlib.util.spec_from_file_location(_name, Path(__file__).with_name("birdlab.py"))
engine = importlib.util.module_from_spec(_spec)
sys.modules[_name] = engine
_spec.loader.exec_module(engine)

DEFAULT_OUTPUT_DIR = "artifacts_compact_v1"
Config = engine.Config
ARCHITECTURES = engine.ARCHITECTURES

def experiment_plan():
    return [
        engine.Experiment("00_baseline"),
        engine.Experiment("01_random_augmentation", fine_tune=True, crop=True, flip=True, color=True),
        engine.Experiment("02_final_optimized", fine_tune=True, crop=True, flip=True, color=True,
                          dropout=.2, weight_decay=1e-4, label_smoothing=.05),
    ]

def augmentation_settings():
    return {"resize": "square, same for evaluation/deployment", "crop_area": [.85, 1.0],
            "crop_ratio": [.9, 1.1], "horizontal_flip_p": .5,
            "color_jitter": {"brightness": .1, "contrast": .1, "saturation": .1, "hue": .015},
            "no_vertical_flip_rotation_inversion_erasing": True}

def make_transform(exp=None, size=384, mean=engine.MEAN, std=engine.STD):
    steps = [T.Resize((size, size))]
    if exp and exp.crop:
        steps.append(T.RandomResizedCrop(size, scale=(.85, 1.), ratio=(.9, 1.1)))
    if exp and exp.flip:
        steps.append(T.RandomHorizontalFlip(.5))
    if exp and exp.color:
        steps.append(T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.015))
    return T.Compose(steps + [T.ToTensor(), T.Normalize(mean, std)])

engine.PIPELINE_VERSION = "compact_v1_three_rounds"
engine.experiment_plan = experiment_plan
engine.augmentation_settings = augmentation_settings
engine.make_transform = make_transform

def default_config():
    return Config(output_dir=DEFAULT_OUTPUT_DIR, image_size=384, batch_size=2,
                  accumulation_steps=16, epochs=60, head_epochs=3, patience=12,
                  backbone_lr=1e-5, head_lr=3e-4)

def round_config(index, cfg):
    # Whole-system progression, not an isolated augmentation ablation: the
    # fine-tuning/epoch/regularization changes must be disclosed in the report.
    if not 0 <= index < 3:
        raise ValueError("Choose round 0, 1 or 2")
    return replace(cfg, epochs=min(cfg.epochs, (15, 35, 60)[index]),
                   patience=min(cfg.patience, (5, 8, 12)[index]),
                   head_lr=cfg.head_lr if index < 2 else cfg.head_lr / 3)

def settings_table(cfg):
    import pandas as pd
    return pd.DataFrame([{
        "Round": exp.name, "Max epochs": round_config(i, cfg).epochs,
        "Image size": cfg.image_size, "Fine-tune": exp.fine_tune,
        "Crop/flip/mild color": exp.crop, "Dropout": exp.dropout,
        "Weight decay": exp.weight_decay, "Label smoothing": exp.label_smoothing,
        "Backbone LR": cfg.backbone_lr, "Head LR": round_config(i, cfg).head_lr,
    } for i, exp in enumerate(experiment_plan())])

def train_round(index, cfg, fit, val, names, manifest):
    return engine.train_stage(index, round_config(index, cfg), fit, val, names, manifest)

def comparison_table(output_dir=DEFAULT_OUTPUT_DIR):
    table = engine.comparison_table(output_dir)
    if table.empty:
        return table
    # Measured round scores remain visible, including negative changes. Only
    # this explicitly labeled checkpoint-selection score is nondecreasing.
    for architecture in ARCHITECTURES:
        col = f"{architecture}: Val Top-1 (%)"
        if col in table:
            table[f"{architecture}: Best so far (%)"] = table[col].cummax()
    table.to_csv(Path(output_dir) / "comparison_display.csv")
    return table

prepare_data = engine.prepare_data
history_table = engine.history_table
results_table = engine.results_table
final_evaluation = engine.final_evaluation
load_predictor = engine.load_predictor
predict_image = engine.predict_image
