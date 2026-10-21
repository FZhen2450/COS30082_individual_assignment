"""Fixed-image augmentation examples and Grad-CAM across training recipes.

Previews are illustrative, not extra training images. Grad-CAM is an explanatory
gradient map, not thermal sensing, and never changes training or selection.
Reference: Selvaraju et al., https://arxiv.org/abs/1610.02391
"""
from __future__ import annotations

import gc
import hashlib
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from birdlab import (ARCHITECTURES, experiment_plan, load_predictor,
                     make_transform, normalization, read_json, write_json)


def fixed_example(cfg, val_frame, filename=None):
    """Pin one validation photo across all recipes, architectures and reruns."""
    target = Path(cfg.output_dir) / "visualizations/example.json"
    if target.exists():
        saved = read_json(target)
        if filename is not None and filename != saved["filename"]:
            raise ValueError("Example already fixed for this study; use its saved filename")
        rows = val_frame[val_frame.filename == saved["filename"]]
    else:
        rows = val_frame if filename is None else val_frame[val_frame.filename == filename]
    if rows.empty:
        raise ValueError("Choose an image from the validation split")
    row = rows.iloc[0]
    record = {"filename": str(row.filename), "label": int(row.label),
              "image_sha256": hashlib.sha256(Path(row.path).read_bytes()).hexdigest(),
              "split": "validation"}
    if target.exists() and saved != record:
        raise ValueError("Fixed example content/label changed")
    write_json(target, record)
    return Path(row.path), int(row.label)


def _new_method(stage_index):
    plan = experiment_plan()
    if not 0 <= stage_index < len(plan):
        raise ValueError("Invalid stage index")
    if stage_index == 0:
        return "baseline"
    previous, current = plan[stage_index - 1], plan[stage_index]
    changed = [key for key in vars(current) if key != "name" and
               getattr(current, key) != getattr(previous, key)]
    if len(changed) != 1:
        raise ValueError("A visual comparison requires one added setting per stage")
    return changed[0]


def preview_stage(stage_index, image_path, cfg, n_examples=5):
    """Show the SAME original and five draws, plus cumulative training views.

    Crop coordinates are genuinely sampled with the training crop distribution.
    All CPU RNG states are restored so viewing examples cannot change training.
    """
    if not 1 <= n_examples <= 12:
        raise ValueError("Choose 1..12 displayed examples; training is not limited to this count")
    plan = experiment_plan()
    method = _new_method(stage_index)
    recipe = plan[stage_index]
    with Image.open(image_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    folder = Path(cfg.output_dir) / "visualizations/method_previews"
    folder.mkdir(parents=True, exist_ok=True)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(cfg.seed + stage_index)
        np.random.seed(cfg.seed + stage_index)
        torch.random.default_generator.manual_seed(cfg.seed + stage_index)
        notes = {
            "baseline": "Frozen ImageNet backbone; train the 200-class head.\nNo random image augmentation.",
            "fine_tune": "Unfreeze backbone after head warm-up.\nThis changes trainable weights, not image pixels.",
            "dropout": "Dropout p=0.3 masks classifier activations during training.\nIt is not image erasing; no pixel filter is invented here.",
            "weight_decay": "AdamW weight_decay=0.001 regularizes learned weights.\nThis setting does not transform the input photo.",
            "label_smoothing": "Cross-entropy label_smoothing=0.1 softens training targets.\nFor 200 classes: target class 0.9005, each other class 0.0005.\nThe input image is unchanged.",
        }
        if method in notes:
            figure = Figure(figsize=(11, 4))
            axes = figure.subplots(1, 2)
            axes[0].imshow(original)
            axes[0].set_title("Fixed validation photo (illustration only)")
            axes[1].text(0.02, 0.5, notes[method], va="center", wrap=True, fontsize=11)
            for axis in axes:
                axis.axis("off")
            records = [{"method": method, "pixel_transformation": False}]
        else:
            figure = Figure(figsize=(3 * (n_examples + 1), 3.6))
            axes = figure.subplots(1, n_examples + 1)
            axes[0].imshow(original)
            axes[0].set_title("Same original")
            records = []
            for index in range(n_examples):
                params = {"sample": index + 1, "method": method}
                if method == "crop":
                    crop_input = TF.resize(original, [cfg.image_size, cfg.image_size])
                    top, left, height, width = T.RandomResizedCrop.get_params(
                        crop_input, scale=(0.75, 1.0), ratio=(0.85, 1.15))
                    sample = TF.resized_crop(crop_input, top, left, height, width,
                                             [cfg.image_size, cfg.image_size])
                    original_top = top * original.height / cfg.image_size
                    original_left = left * original.width / cfg.image_size
                    original_height = height * original.height / cfg.image_size
                    original_width = width * original.width / cfg.image_size
                    axes[0].add_patch(Rectangle((original_left, original_top), original_width,
                                               original_height, fill=False,
                                               edgecolor=f"C{index % 10}", linewidth=1.5))
                    params.update(top=original_top, left=original_left,
                                  height=original_height, width=original_width,
                                  coordinate_space="original image (mapped from common resize)",
                                  resized_box=[top, left, height, width])
                else:
                    sample = TF.resize(original, [cfg.image_size, cfg.image_size])
                    if method == "flip":
                        applied = bool(torch.rand(()) < 0.5)
                        sample = TF.hflip(sample) if applied else sample
                        params["applied"] = applied
                    elif method == "rotation":
                        angle = float(torch.empty(1).uniform_(-10, 10).item())
                        sample = TF.rotate(sample, angle)
                        params["angle_degrees"] = angle
                    elif method == "color":
                        sample = T.ColorJitter(0.15, 0.15, 0.15, 0.02)(sample)
                    elif method == "grayscale":
                        # Show its effect in all five panels, while documenting
                        # the training probability instead of pretending p=1.
                        sample = TF.to_grayscale(sample, num_output_channels=3)
                        params["display_forced"] = True
                        params["training_probability"] = 0.2
                    elif method == "erasing":
                        sample = T.ToTensor()(sample)
                        # Training erases normalized tensors with 0, equivalent
                        # to filling RGB pixels with the model-specific mean.
                        mean, _ = normalization(ARCHITECTURES[0])
                        sample = T.RandomErasing(p=1, scale=(0.02, 0.1), value=mean)(sample)
                        sample = TF.to_pil_image(sample)
                        params["display_forced"] = True
                        params["training_probability"] = 0.15
                    else:
                        raise ValueError(method)
                axes[index + 1].imshow(sample)
                label = f"Draw {index + 1}"
                if method == "rotation":
                    label += f" ({params['angle_degrees']:+.1f} deg)"
                if method in ("erasing", "grayscale"):
                    label += " (effect shown)"
                axes[index + 1].set_title(label, color=f"C{index % 10}" if method == "crop" else "black")
                records.append(params)
            for axis in axes:
                axis.axis("off")
        figure.suptitle(f"{recipe.name}: NEW method only; {n_examples} shown examples are not a training limit")
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        output = folder / f"{recipe.name}_new_method.png"
        figure.savefig(output, dpi=140)
        write_json(folder / f"{recipe.name}_samples.json", {
            "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "preview_seed": cfg.seed + stage_index, "examples": records,
            "preview_architecture": ARCHITECTURES[0],
            "training": "On-the-fly stochastic transforms, one new draw per image access",
        })
        # Show exactly what the entire cumulative training recipe would do.
        mean, std = normalization(ARCHITECTURES[0])
        cumulative = Figure(figsize=(3 * (n_examples + 1), 3.6))
        axes = cumulative.subplots(1, n_examples + 1)
        axes[0].imshow(original)
        axes[0].set_title("Same original")
        transform = make_transform(recipe, cfg.image_size, mean, std)
        for index in range(n_examples):
            tensor = transform(original)
            tensor = tensor * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
            axes[index + 1].imshow(tensor.clamp(0, 1).permute(1, 2, 0))
            axes[index + 1].set_title(f"Training draw {index + 1}")
        for axis in axes:
            axis.axis("off")
        cumulative.suptitle(f"{recipe.name}: FULL recipe, MobileNetV1 pixel preview (training probabilities unchanged)")
        cumulative.tight_layout(rect=(0, 0, 1, 0.92))
        cumulative_path = folder / f"{recipe.name}_full_recipe.png"
        cumulative.savefig(cumulative_path, dpi=140)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
    return output, cumulative_path


@torch.enable_grad()
def grad_cam(model, checkpoint, image, target_class):
    """Gradient-weighted last-feature map for ONE FIXED class, not temperature."""
    if not 0 <= target_class < len(checkpoint["class_names"]):
        raise ValueError("Invalid target class")
    original = ImageOps.exif_transpose(image).convert("RGB")
    size = checkpoint["image_size"]
    device = next(model.parameters()).device
    tensor = make_transform(size=size, mean=checkpoint["mean"], std=checkpoint["std"])(original)
    tensor = tensor.unsqueeze(0).to(device).requires_grad_(True)
    captured = {}

    def capture(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Expected a B,C,H,W feature map")
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.update(gradient=gradient))

    modes = [(module, module.training) for module in model.modules()]
    handle = model.features[-1].register_forward_hook(capture)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        logits[0, target_class].backward()
        activation = captured["activation"].detach().float()
        weights = captured["gradient"].detach().float().mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True).relu()
        cam = nn_resize(cam, size)[0, 0].cpu().numpy()
        cam -= cam.min()
        maximum = float(cam.max())
        cam = cam / maximum if maximum > 1e-12 else np.zeros_like(cam)
        rgb = np.asarray(TF.resize(original, [size, size]), dtype=np.float32) / 255
        colors = matplotlib.colormaps["inferno"](cam)[..., :3]
        # No artificial hotspot when the positive Grad-CAM map is empty.
        overlay = (1 - 0.5 * cam[..., None]) * rgb + 0.5 * cam[..., None] * colors
        overlay = Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255))
        probabilities = logits.detach().float().softmax(1)[0].cpu()
        return cam, overlay, {
            "target_class": target_class,
            "target_species": checkpoint["class_names"][target_class],
            "target_probability": float(probabilities[target_class]),
            "predicted_species": checkpoint["class_names"][int(probabilities.argmax())],
            "positive_map_present": maximum > 1e-12,
            "interpretation": "Grad-CAM, normalized per map; not thermal data or an accuracy metric",
        }
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        for module, mode in modes:
            module.training = mode


def nn_resize(tensor, size):
    return torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)


def attention_example(checkpoint_path, image_path, target_class, output_dir):
    """Cache a real checkpoint explanation; invalidate on image/weight changes."""
    checkpoint_path, image_path = Path(checkpoint_path), Path(image_path)
    token = hashlib.sha256(image_path.read_bytes()).hexdigest()
    folder = Path(output_dir) / "visualizations/attention" / token[:12] / checkpoint_path.parent.parent.name
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{checkpoint_path.parent.name}_class{target_class}"
    png, metadata_path = folder / f"{stem}.png", folder / f"{stem}.json"
    stat = checkpoint_path.stat()
    signature = {"image_sha256": token, "weight_mtime_ns": stat.st_mtime_ns,
                 "weight_size": stat.st_size, "target_class": target_class,
                 "method": "gradcam_last_feature_v1"}
    if png.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("signature") == signature:
            return png, metadata
    # Building a model for explanation must not consume the notebook's RNG.
    with torch.random.fork_rng(devices=[]):
        model, checkpoint = load_predictor(checkpoint_path, "cpu")
    try:
        with Image.open(image_path) as image:
            _, overlay, metadata = grad_cam(model, checkpoint, image, target_class)
        overlay.save(png)
        metadata["signature"] = signature
        write_json(metadata_path, metadata)
    finally:
        del model
        gc.collect()
    return png, metadata


def plot_attention_progress(cfg, image_path, target_class, stage_indices=range(5)):
    """Same image/class for BOTH architectures; up to 5 stages per report panel.

    Missing checkpoints stay empty. Never fabricate an evolution before training.
    """
    indices = list(stage_indices)
    plan = experiment_plan()
    if not indices or len(indices) > 5 or any(not 0 <= i < len(plan) for i in indices):
        raise ValueError("Choose 1..5 valid stage indices")
    figure = Figure(figsize=(3 * (len(indices) + 1), 7))
    axes = figure.subplots(2, len(indices) + 1, squeeze=False)
    with Image.open(image_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    for row, architecture in enumerate(ARCHITECTURES):
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{architecture}\nSame validation photo", fontsize=10)
        for column, index in enumerate(indices, start=1):
            checkpoint_path = Path(cfg.output_dir) / architecture / plan[index].name / "best.pth"
            # An interrupted run's best checkpoint is not a completed stage.
            if checkpoint_path.exists() and (checkpoint_path.parent / "result.json").exists():
                png, metadata = attention_example(checkpoint_path, image_path, target_class, cfg.output_dir)
                with Image.open(png) as image:
                    axes[row, column].imshow(image.copy())
                axes[row, column].set_title(
                    f"{plan[index].name}\np(target)={metadata['target_probability']:.1%}", fontsize=9)
            else:
                axes[row, column].text(0.5, 0.5, "Not trained yet", ha="center", va="center")
                axes[row, column].set_title(plan[index].name, fontsize=9)
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle(f"Grad-CAM for fixed class {target_class}: model attention, NOT temperature or accuracy")
    figure.text(0.5, 0.02, "Colors normalized separately per map; localization alone does not prove better classification.",
                ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 0.92))
    target = Path(cfg.output_dir) / "visualizations/attention_progress"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"stages_{indices[0]:02d}_{indices[-1]:02d}_class{target_class}.png"
    figure.savefig(path, dpi=140)
    return path
