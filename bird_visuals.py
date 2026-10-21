"""Visible augmentation examples only; quantitative model comparisons use tables."""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from birdlab import (ARCHITECTURES, CROP_SCALE, CROP_RATIO, ROTATION_RANGE,
                     COLOR_JITTER, COLOR_EFFECT_NAMES, apply_color_effect,
                     experiment_plan, make_transform, normalization, read_json, write_json)


def fixed_example(cfg, val_frame, filename=None):
    """One validation photo for visual illustration only; never added to fit."""
    target = Path(cfg.output_dir) / "visualizations/example.json"
    saved = read_json(target) if target.exists() else None
    requested = filename or (saved["filename"] if saved else str(val_frame.iloc[0].filename))
    rows = val_frame[val_frame.filename == requested]
    if rows.empty:
        raise ValueError("Choose a validation filename")
    row = rows.iloc[0]
    record = {"filename": str(row.filename), "label": int(row.label),
              "image_sha256": hashlib.sha256(Path(row.path).read_bytes()).hexdigest(), "split": "validation"}
    if saved is not None and saved != record:
        raise ValueError("Fixed example changed; use the saved example or a fresh output directory")
    write_json(target, record)
    return Path(row.path), int(row.label)


def _new_method(stage_index):
    plan = experiment_plan()
    if stage_index == 0:
        return "baseline"
    changes = [key for key in vars(plan[stage_index]) if key != "name" and
               getattr(plan[stage_index], key) != getattr(plan[stage_index - 1], key)]
    if len(changes) != 1:
        raise ValueError("Expected one added method")
    return changes[0]


def _stratified_crop(index, count, size):
    """Illustrate different size bands/positions within the training support.

    Display selection is stratified for legibility; runtime crops remain random.
    """
    edges = np.linspace(CROP_SCALE[0], CROP_SCALE[1], count + 1)
    fraction = float(torch.empty(1).uniform_(float(edges[index]), float(edges[index + 1])).item())
    lower, upper = max(CROP_RATIO[0], fraction), min(CROP_RATIO[1], 1 / fraction)
    ratio = math.exp(float(torch.empty(1).uniform_(math.log(lower), math.log(upper)).item()))
    height = min(size, round(size * math.sqrt(fraction / ratio)))
    width = min(size, round(size * math.sqrt(fraction * ratio)))
    max_top, max_left = size - height, size - width
    # Different quadrants plus center, with genuine random offsets in each zone.
    zones = [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5)]
    row_zone, col_zone = zones[index % len(zones)]

    def offset(maximum, zone):
        if zone == 0:
            low, high = 0, maximum // 2
        elif zone == 1:
            low, high = maximum // 2, maximum
        else:
            low, high = maximum // 4, 3 * maximum // 4
        return int(torch.randint(low, high + 1, (1,)).item())

    return offset(max_top, row_zone), offset(max_left, col_zone), height, width


def preview_stage(stage_index, image_path, cfg, n_examples=5):
    """Five visible size/angle/color examples; no fake views for fine-tuning.

    Flips/grayscale have only two states and show those two, not five duplicates.
    Five cumulative runtime draws are shown for image-augmentation stages.
    All CPU RNG states are restored after previewing.
    """
    if not 1 <= n_examples <= 12:
        raise ValueError("Choose 1..12 examples")
    recipe = experiment_plan()[stage_index]
    method = _new_method(stage_index)
    if method in {"baseline", "fine_tune", "dropout", "weight_decay", "label_smoothing"}:
        return ()  # The notebook displays stage_configuration instead.
    with Image.open(image_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    source = TF.resize(original, [cfg.image_size, cfg.image_size])
    folder = Path(cfg.output_dir) / "visualizations/method_previews"
    folder.mkdir(parents=True, exist_ok=True)
    py_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    records = []
    try:
        random.seed(cfg.seed + stage_index)
        np.random.seed(cfg.seed + stage_index)
        torch.random.default_generator.manual_seed(cfg.seed + stage_index)
        count = 1 if method in {"flip", "vertical_flip", "grayscale"} else n_examples
        figure = Figure(figsize=(3 * (count + 1), 4.0))
        axes = figure.subplots(1, count + 1, squeeze=False)[0]
        axes[0].imshow(original)
        axes[0].set_title("Same original")
        mean, std = normalization(ARCHITECTURES[0])
        for index in range(count):
            params = {"sample": index + 1, "method": method}
            sample = source.copy()
            label = f"Example {index + 1}"
            if method == "crop":
                top, left, height, width = _stratified_crop(index, count, cfg.image_size)
                sample = TF.resized_crop(source, top, left, height, width, [cfg.image_size, cfg.image_size])
                y, x = top * original.height / cfg.image_size, left * original.width / cfg.image_size
                h, w = height * original.height / cfg.image_size, width * original.width / cfg.image_size
                axes[0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=f"C{index % 10}", linewidth=1.5))
                fraction = height * width / cfg.image_size ** 2
                label = f"Crop {index + 1}: area {fraction:.0%}"
                params.update(top=y, left=x, height=h, width=w, area_fraction=fraction,
                              resized_box=[top, left, height, width], preview_sampling="stratified size and position")
            elif method == "flip":
                sample = TF.hflip(sample)
                label = "Left/right flip (p=0.5 in training)"
                params.update(display_forced=True, training_probability=0.5)
            elif method == "vertical_flip":
                sample = TF.vflip(sample)
                label = "Up/down flip (p=0.2 in training)"
                params.update(display_forced=True, training_probability=0.2)
            elif method == "rotation":
                edges = np.linspace(*ROTATION_RANGE, count + 1)
                angle = float(torch.empty(1).uniform_(float(edges[index]), float(edges[index + 1])).item())
                sample = TF.rotate(sample, angle, interpolation=T.InterpolationMode.BILINEAR,
                                   fill=tuple(round(value * 255) for value in mean))
                label = f"Random angle {angle:+.1f} deg"
                params.update(angle_degrees=angle, preview_sampling="stratified angle bands")
            elif method == "color":
                sample = T.ColorJitter(**COLOR_JITTER)(sample)
                label = f"Stronger ColorJitter {index + 1}"
            elif method == "color_effects":
                effect = COLOR_EFFECT_NAMES[index % len(COLOR_EFFECT_NAMES)]
                strength = float(torch.empty(1).uniform_(0.2, 0.45).item())
                sample = apply_color_effect(sample, effect, strength)
                label = f"{effect.title()} (effect forced for display)"
                params.update(effect=effect, strength=strength, display_forced=True,
                              total_training_effect_probability=0.25)
            elif method == "grayscale":
                sample = TF.to_grayscale(sample, num_output_channels=3)
                label = "Grayscale (p=0.2 in training)"
                params.update(display_forced=True, training_probability=0.2)
            elif method == "erasing":
                tensor = T.RandomErasing(p=1, scale=(0.02, 0.15), value=mean)(TF.to_tensor(sample))
                sample = TF.to_pil_image(tensor)
                label = f"Erasing {index + 1} (effect shown)"
                params.update(display_forced=True, training_probability=0.25)
            else:
                raise ValueError(method)
            axes[index + 1].imshow(sample)
            axes[index + 1].set_title(label, fontsize=10, color=f"C{index % 10}" if method == "crop" else "black")
            records.append(params)
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"{recipe.name}: augmentation examples only; preview selection is not the training schedule")
        figure.tight_layout(rect=(0, 0, 1, 0.91))
        method_path = folder / f"{recipe.name}_new_method.png"
        figure.savefig(method_path, dpi=140)
        write_json(folder / f"{recipe.name}_samples.json", {
            "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "preview_seed": cfg.seed + stage_index, "examples": records,
            "training": "New independent random transforms on every training image access",
        })
        cumulative = Figure(figsize=(3 * (n_examples + 1), 4.0))
        axes = cumulative.subplots(1, n_examples + 1, squeeze=False)[0]
        axes[0].imshow(original)
        axes[0].set_title("Same original")
        transform = make_transform(recipe, cfg.image_size, mean, std)
        for index in range(n_examples):
            tensor = transform(original)
            tensor = tensor * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
            axes[index + 1].imshow(tensor.clamp(0, 1).permute(1, 2, 0))
            axes[index + 1].set_title(f"Runtime draw {index + 1}")
        for axis in axes:
            axis.axis("off")
        cumulative.suptitle(f"{recipe.name}: full recipe, MobileNetV1 pixel preview; actual training probabilities")
        cumulative.tight_layout(rect=(0, 0, 1, 0.91))
        cumulative_path = folder / f"{recipe.name}_full_recipe.png"
        cumulative.savefig(cumulative_path, dpi=140)
    finally:
        random.setstate(py_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
    return method_path, cumulative_path
