"""Check fixed-photo previews, true Grad-CAM gradients and checkpoint panels.

Run after verify_pipeline.py. All figures/scores are smoke-only artifacts.
"""
import hashlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch import nn

import birdlab as b
from bird_visuals import (attention_example, fixed_example, grad_cam,
                          plot_attention_progress, preview_stage)


def main():
    torch.set_num_threads(4)
    # A known convolution/classifier should localize a known red square. Every
    # parameter is frozen: explanations must still work via input gradients.
    class ToyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 1, 1, bias=False), nn.ReLU())
            self.classifier = nn.Linear(1, 2, bias=False)

        def forward(self, x):
            return self.classifier(self.features(x).mean(dim=(2, 3)))

    toy = ToyCNN()
    with torch.no_grad():
        toy.features[0].weight.zero_()
        toy.features[0].weight[0, 0] = 1
        toy.classifier.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    for parameter in toy.parameters():
        parameter.requires_grad_(False)
    toy.features[0].eval()
    modes = [module.training for module in toy.modules()]
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:4, :4, 0] = 255
    example = Image.fromarray(pixels)
    checkpoint = {"image_size": 8, "mean": [0, 0, 0], "std": [1, 1, 1],
                  "class_names": ["red", "negative red"]}
    cam, _, metadata = grad_cam(toy, checkpoint, example, 0)
    assert cam[:4, :4].mean() > 0.99 and cam[4:, 4:].max() == 0
    assert metadata["positive_map_present"]
    assert [module.training for module in toy.modules()] == modes
    assert not toy.features[-1]._forward_hooks
    cam, _, metadata = grad_cam(toy, checkpoint, example, 1)
    assert not cam.any() and not metadata["positive_map_present"]

    cfg = b.Config(data_root=str(ROOT), output_dir=str(ROOT / "smoke_artifacts"))
    manifest = b.read_json(Path(cfg.output_dir) / "split.json")
    train = b.read_annotations(ROOT, "train")
    validation = train[train.filename.isin(manifest["validation"])].reset_index(drop=True)
    path, target_class = fixed_example(cfg, validation)
    assert fixed_example(cfg, validation) == (path, target_class)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for stage_index in range(len(b.experiment_plan())):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        files = preview_stage(stage_index, path, cfg)
        assert all(file.is_file() for file in files)
        assert python_state == random.getstate()
        after_numpy = np.random.get_state()
        assert numpy_state[0] == after_numpy[0] and np.array_equal(numpy_state[1], after_numpy[1])
        assert numpy_state[2:] == after_numpy[2:] and torch.equal(torch_state, torch.get_rng_state())
    samples = b.read_json(Path(cfg.output_dir) / "visualizations/method_previews/02_random_crop_samples.json")["examples"]
    assert len(samples) == 5
    boxes = {(row["top"], row["left"], row["height"], row["width"]) for row in samples}
    assert len(boxes) > 1
    with Image.open(path) as image:
        for top, left, height, width in boxes:
            assert 0 <= top < top + height <= image.height + 1e-6
            assert 0 <= left < left + width <= image.width + 1e-6
        # More than five possible runtime crops from a single original.
        transform = b.make_transform(b.experiment_plan()[2])
        variants = {hashlib.sha256(transform(image.convert("RGB")).numpy().tobytes()).hexdigest()
                    for _ in range(20)}
        assert len(variants) > 5
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash

    for architecture in b.ARCHITECTURES:
        weight = Path(cfg.output_dir) / architecture / "00_baseline/best.pth"
        png, metadata = attention_example(weight, path, target_class, cfg.output_dir)
        stamp = png.stat().st_mtime_ns
        again, cached = attention_example(weight, path, target_class, cfg.output_dir)
        assert again == png and cached == metadata and png.stat().st_mtime_ns == stamp
        assert metadata["target_class"] == target_class
    panel = plot_attention_progress(cfg, path, target_class, range(5))
    assert panel.is_file()
    assert not (Path(cfg.output_dir) / b.ARCHITECTURES[0] / "03_horizontal_flip/result.json").exists()
    b.write_json(Path(cfg.output_dir) / "visual_verification.json", {
        "purpose": "Smoke-only checks, not assignment accuracy",
        "gradcam_known_localization": True, "zero_positive_map_handled": True,
        "frozen_parameters_supported": True, "previews_preserve_rng": True,
        "runtime_crop_variants": len(variants), "target_class_fixed": target_class,
        "cached_explanations": True, "missing_stages_not_fabricated": True,
    })
    print("Visual checks PASS: real gradients, same photo/class, RNG restoration, >5 crops, cached panels.")
    print("Panel:", panel)


if __name__ == "__main__":
    main()
