"""Tiny GPU integration check, NOT an assignment experiment or benchmark.

Uses eight fit/four validation samples and separate smoke_artifacts. Downloads
official weights on first run. Checks AMP backprop, checkpoint resume and inference.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")

import nbformat
import numpy as np
import torch
from PIL import Image
import birdlab


def main():
    assert torch.cuda.is_available(), "Select the GPU .venv"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    nb = nbformat.read(ROOT / "Bird_Classification_Local_GPU.ipynb", as_version=4)
    nbformat.validate(nb)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook cell {index}", "exec")
    scores, matrix, recalls, _ = birdlab.metrics_from_predictions([0, 0, 1, 2], [0, 1, 1, 1], 3)
    assert scores["top1"] == 0.5 and np.isclose(scores["average_per_class"], 0.5)
    assert matrix.shape == (3, 3)
    cfg = birdlab.Config(data_root=str(ROOT), output_dir="smoke_artifacts",
                         epochs=2, head_epochs=1, batch_size=8, accumulation_steps=2)
    fit, val, names, manifest = birdlab.prepare_data(cfg)
    assert len(names) == 200 and not set(fit.sha256) & set(val.sha256)
    print("Split:", len(fit), len(val), "duplicate rows:", len(manifest["duplicate_training_rows"]), flush=True)
    results = []
    for architecture in birdlab.ARCHITECTURES:
        torch.cuda.reset_peak_memory_stats()
        # Simulate interruption AFTER an epoch checkpoint has been written.
        # If a prior run already completed, the normal skip behavior is checked.
        folder = Path(cfg.output_dir) / architecture / "00_baseline"
        if not (folder / "last.pth").exists():
            original = birdlab.save_checkpoint

            def interrupt_once(path, payload):
                original(path, payload)
                if Path(path).name == "last.pth":
                    raise KeyboardInterrupt("Intentional smoke-test interruption")

            birdlab.save_checkpoint = interrupt_once
            try:
                birdlab.train_experiment(architecture, birdlab.experiment_plan()[0], cfg,
                                         fit.head(8), val.head(4), names, manifest)
            except KeyboardInterrupt:
                print("Verified epoch checkpoint before interruption", flush=True)
            finally:
                birdlab.save_checkpoint = original
        result = birdlab.train_experiment(architecture, birdlab.experiment_plan()[0], cfg,
                                          fit.head(8), val.head(4), names, manifest)
        assert result["epochs_run"] == 2
        model, checkpoint = birdlab.load_predictor(result["checkpoint"], "cpu")
        with Image.open(val.iloc[0].path) as image:
            predictions = birdlab.predict_image(model, checkpoint, image)
        assert len(predictions) == 5
        assert predictions.probability.between(0, 1).all()
        assert predictions.probability.is_monotonic_decreasing
        print(architecture, "PASS; peak GPU allocated GiB:",
              round(torch.cuda.max_memory_allocated() / 2**30, 3), flush=True)
        results.append({"architecture": architecture, "pipeline_pass": True,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
        del model
    birdlab.plot_results(cfg.output_dir)
    birdlab.plot_history(str(Path(cfg.output_dir) / birdlab.ARCHITECTURES[0] / "00_baseline/best.pth"))
    birdlab.write_json(Path(cfg.output_dir) / "verification.json", {
        "purpose": "Tiny pipeline smoke test; not benchmark results",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "fit_count": len(fit), "validation_count": len(val), "models": results})
    print("All pipeline checks passed. Full experiments have NOT been run.", flush=True)


if __name__ == "__main__":
    main()
