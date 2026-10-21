"""Synthetic-fixture check of test selection, metrics, reports and export.

Requires the checkpoints created by verify_pipeline.py. Never reads assignment
Test images or exports a model to the real deployment directory.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import birdlab as b
from PIL import Image


def main():
    with tempfile.TemporaryDirectory(prefix="finalization_", dir=ROOT / "smoke_artifacts") as temporary:
        tmp = Path(temporary)
        for split in ["Train", "Test"]:
            (tmp / split).mkdir()
            lines = []
            for label in range(200):
                name = f"{label:04d}.png"
                # Exactly one known Train/Test overlap, to check clean metrics.
                color = (label, 0 if split == "Train" or label == 0 else 1, 0)
                Image.new("RGB", (16, 16), color).save(tmp / split / name)
                lines.append(f"{name} {label}")
            (tmp / f"{split.lower()}.txt").write_text("\n".join(lines))
        out = tmp / "artifacts"
        names = b.read_json(ROOT / "smoke_artifacts/split.json")["class_names"]
        for architecture in b.ARCHITECTURES:
            for i, experiment in enumerate(b.experiment_plan()):
                folder = out / architecture / experiment.name
                folder.mkdir(parents=True)
                if i in [0, len(b.experiment_plan()) - 1]:
                    shutil.copy2(ROOT / "smoke_artifacts" / architecture / "00_baseline/best.pth",
                                 folder / "best.pth")
                # Synthetic scores select known paths; not benchmark results.
                b.write_json(folder / "result.json", {
                    "architecture": architecture, "experiment": experiment.name,
                    "val_top1": 0.01 * i, "checkpoint": str(folder / "best.pth"),
                })
        cfg = b.Config(data_root=str(tmp), output_dir=str(out), batch_size=8)
        previous = Path.cwd()
        os.chdir(tmp)
        try:
            result = b.final_evaluation(cfg, names, {"train_sha256": "synthetic-fixture"})
            assert len(result) == 4 and (result.samples == 200).all()
            assert (result.overlap_count == 1).all() and (result.clean_samples == 199).all()
            assert Path("deployment/bird_model.pth").is_file()
            again = b.final_evaluation(cfg, names, {"train_sha256": "synthetic-fixture"})
            assert result.equals(again)
            for run in result.run:
                for name in ["per_class.csv", "confusion_matrix.png", "prediction_examples.png"]:
                    assert (out / run / "test" / name).is_file()
            print("PASS: selection lock, 200-class metrics, overlap exclusion, reports, export and cached rerun.")
        finally:
            os.chdir(previous)
    print("Actual assignment Test was not evaluated; real deployment remains empty.")


if __name__ == "__main__":
    main()
