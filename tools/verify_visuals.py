"""Verify new augmentation ranges, faithful previews and untouched RNG/source."""
import hashlib
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
import birdlab as b
from bird_visuals import fixed_example, preview_stage

def main():
    cfg = b.Config(data_root=str(ROOT), output_dir=str(ROOT / "smoke_artifacts/retrain_v3"))
    manifest = b.read_json(Path(cfg.output_dir) / "split.json")
    train = b.read_annotations(ROOT, "train")
    val = train[train.filename.isin(manifest["validation"])].reset_index(drop=True)
    path, label = fixed_example(cfg, val)
    assert fixed_example(cfg, val) == (path, label)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for index in range(len(b.experiment_plan())):
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        files = preview_stage(index, path, cfg)
        pixels = b.stage_configuration(index, cfg)["Changes pixels"].all()
        assert len(files) == (2 if pixels else 0)
        assert all(p.is_file() for p in files)
        assert py == random.getstate() and torch.equal(cpu, torch.get_rng_state())
        after = np.random.get_state()
        assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]
    folder = Path(cfg.output_dir) / "visualizations/method_previews"
    def examples(name):
        return b.read_json(folder / f"{name}_samples.json")["examples"]
    crops = examples("02_random_crop")
    areas = [r["area_fraction"] for r in crops]
    assert len(areas) == 5 and all(x < y for x, y in zip(areas, areas[1:]))
    assert min(areas) >= .395 and max(areas) <= 1 and max(areas) - min(areas) > .35
    assert len({(r["top"], r["left"]) for r in crops}) == 5
    with Image.open(path) as image:
        for row in crops:
            assert 0 <= row["top"] < row["top"] + row["height"] <= image.height + 1e-6
            assert 0 <= row["left"] < row["left"] + row["width"] <= image.width + 1e-6
        transform = b.make_transform(b.experiment_plan()[2])
        variants = {hashlib.sha256(transform(image.convert("RGB")).numpy().tobytes()).hexdigest() for _ in range(20)}
        assert len(variants) > 5
    angles = [r["angle_degrees"] for r in examples("05_random_rotation")]
    assert all(-45 <= a <= 45 for a in angles) and max(angles) - min(angles) > 50
    assert all(x < y for x, y in zip(angles, angles[1:]))
    effects = examples("07_color_effects")
    assert tuple(r["effect"] for r in effects) == b.COLOR_EFFECT_NAMES
    assert all(.2 <= r["strength"] <= .45 for r in effects)
    for name in ["03_horizontal_flip", "04_vertical_flip", "12_grayscale"]:
        assert len(examples(name)) == 1
    pixels = np.full((8, 8, 3), [40, 80, 120], dtype=np.uint8)
    image = Image.fromarray(pixels)
    assert np.array_equal(np.array(b.apply_color_effect(image, "invert")), 255 - pixels)
    for effect, channel in [("red", 0), ("green", 1), ("blue", 2)]:
        tinted = np.array(b.apply_color_effect(image, effect, .4)).astype(float)
        expected = .6 * pixels.astype(float)
        expected[:, :, channel] += .4 * 255
        assert np.max(np.abs(tinted - expected)) <= 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    b.write_json(Path(cfg.output_dir) / "visual_verification.json", {
        "purpose": "Smoke-only, not model accuracy", "crop_areas": areas,
        "rotation_angles": angles, "color_effects": list(b.COLOR_EFFECT_NAMES),
        "source_and_rng_preserved": True, "non_pixel_methods_have_no_images": True})
    print("PASS: distinct crops/angles, both flip directions, actual RGB effects, source/RNG preserved.")

if __name__ == "__main__":
    main()
