"""Tiny GPU integration check, NOT an assignment experiment or benchmark.

Uses 17 fit/9 validation samples and separate smoke_artifacts. Checks frozen
baseline, fine-tuning, cumulative augmentation, AMP, resume and inference.
"""
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["TQDM_DISABLE"] = "1"

import nbformat
import numpy as np
import torch
from PIL import Image
import birdlab


def main():
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "Select the GPU .venv"
    print("SMOKE TEST ONLY: 17 fit / 9 validation images; not assignment accuracy.", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    nb = nbformat.read(ROOT / "Bird_Classification_Local_GPU.ipynb", as_version=4)
    nbformat.validate(nb)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook cell {index}", "exec")
    scores, matrix, recalls, _ = birdlab.metrics_from_predictions([0, 0, 1, 2], [0, 1, 1, 1], 3)
    assert scores["top1"] == 0.5 and np.isclose(scores["average_per_class"], 0.5)
    assert matrix.shape == (3, 3)
    cfg = birdlab.Config(data_root=str(ROOT), output_dir="smoke_artifacts/retrain_v3",
                         image_size=96, epochs=2, head_epochs=1, batch_size=2, accumulation_steps=2)
    fit, val, names, manifest = birdlab.prepare_data(cfg)
    assert len(names) == 200 and not set(fit.sha256) & set(val.sha256)
    print("Split:", len(fit), len(val), "duplicate rows:", len(manifest["duplicate_training_rows"]), flush=True)
    small_fit = fit.groupby("label").head(2).head(17)
    small_val = val.groupby("label").head(1).head(9)
    plan = birdlab.experiment_plan()
    assert not plan[0].fine_tune and all(recipe.fine_tune for recipe in plan[1:])
    for previous, current in zip(plan, plan[1:]):
        assert sum(getattr(previous, key) != getattr(current, key)
                   for key in vars(previous) if key != "name") == 1
    try:
        birdlab.train_stage(7, cfg, small_fit, small_val, names, manifest)
    except RuntimeError:
        pass
    else:
        raise AssertionError("A future stage should be blocked before its predecessor")
    results = []
    for architecture in birdlab.ARCHITECTURES:
        torch.cuda.reset_peak_memory_stats()
        initial_model = birdlab.build_model(architecture, 200, pretrained=True)
        feature_key = next(key for key in initial_model.state_dict() if key.startswith("features."))
        initial_feature = initial_model.state_dict()[feature_key].clone()
        del initial_model
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
                                         small_fit, small_val, names, manifest)
            except KeyboardInterrupt:
                print("Verified epoch checkpoint before interruption", flush=True)
            finally:
                birdlab.save_checkpoint = original
        result = birdlab.train_experiment(architecture, birdlab.experiment_plan()[0], cfg,
                                          small_fit, small_val, names, manifest)
        assert result["epochs_run"] == 2
        baseline_state = torch.load(folder / "last.pth", map_location="cpu", weights_only=True)
        assert torch.equal(baseline_state["model"][feature_key], initial_feature)
        del baseline_state
        for recipe in [plan[1], plan[2], plan[-1]]:
            variant = birdlab.train_experiment(architecture, recipe, cfg,
                                                small_fit, small_val, names, manifest)
            state = torch.load(Path(variant["checkpoint"]).parent / "last.pth",
                               map_location="cpu", weights_only=True)
            assert state["history"][-1]["phase"] == "fine_tune"
            assert not torch.equal(state["model"][feature_key], initial_feature)
            del state
        model, checkpoint = birdlab.load_predictor(result["checkpoint"], "cpu")
        assert tuple(checkpoint["mean"]) == birdlab.normalization(architecture)[0]
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
    paired_stage = birdlab.train_stage(1, cfg, small_fit, small_val, names, manifest)
    assert len(paired_stage) == 2 and (paired_stage.experiment == plan[1].name).all()
    crop_stage = birdlab.train_stage(2, cfg, small_fit, small_val, names, manifest)
    assert len(crop_stage) == 2 and plan[2].crop and not plan[2].flip and not plan[2].rotation
    frame = birdlab.results_table(cfg.output_dir)
    for architecture in birdlab.ARCHITECTURES:
        rows = frame[frame.architecture == architecture].set_index("experiment")
        expected = 100 * (rows.loc[plan[1].name, "val_top1"] - rows.loc[plan[0].name, "val_top1"])
        assert np.isclose(rows.loc[plan[1].name, "delta_vs_previous_pp"], expected)
        assert np.isnan(rows.loc[plan[-1].name, "delta_vs_previous_pp"])
    comparison = birdlab.comparison_table(cfg.output_dir)
    assert len(comparison) == 4
    assert any("Val Top-1 (%)" in key for key in comparison.columns)
    birdlab.write_json(Path(cfg.output_dir) / "verification.json", {
        "purpose": "Tiny pipeline smoke test; not benchmark results",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "fit_count": len(fit), "validation_count": len(val), "models": results,
        "smoke_fit_count": len(small_fit), "smoke_validation_count": len(small_val),
        "recipes_checked": [plan[0].name, plan[1].name, plan[2].name, plan[-1].name]})
    print("All pipeline checks passed. Full experiments have NOT been run.", flush=True)


if __name__ == "__main__":
    main()
