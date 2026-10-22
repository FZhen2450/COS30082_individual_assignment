"""Six tiny GPU runs, original-engine isolation and compact deployment checks."""
import os
import sys
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import nbformat
import torch
import numpy as np
from PIL import Image
import birdlab as original
import birdlab_compact as lab

def main():
    torch.set_num_threads(4)
    assert len(original.experiment_plan()) == 13 and len(lab.experiment_plan()) == 3
    assert original.PIPELINE_VERSION != lab.engine.PIPELINE_VERSION
    assert lab.default_config().output_dir == "artifacts_compact_v1"
    assert [lab.round_config(i, lab.default_config()).epochs for i in range(3)] == [15,35,60]
    nb = nbformat.read(ROOT / "Bird_Classification_Compact_GPU.ipynb", 4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == "code": compile(c.source, "compact notebook cell", "exec")
    out = ROOT / "smoke_artifacts/compact_v1"
    cfg = replace(lab.default_config(), output_dir=str(out), image_size=96,
                  batch_size=2, accumulation_steps=2, epochs=2, head_epochs=1, patience=2)
    fit, val, names, manifest = lab.prepare_data(cfg)
    smallfit, smallval = fit.head(17), val.head(9)
    for i in range(3):
        rows = lab.train_round(i, cfg, smallfit, smallval, names, manifest)
        assert len(rows) == 2
    table = lab.comparison_table(out)
    assert len(table) == 3
    for arch in lab.ARCHITECTURES:
        col = f"{arch}: Val Top-1 (%)"
        assert np.allclose(table[f"{arch}: Best so far (%)"], table[col].cummax())
        for stage in lab.experiment_plan():
            ck = out / arch / stage.name / "best.pth"
            model, metadata = lab.load_predictor(ck, "cpu")
            assert metadata["pipeline_version"] == "compact_v1_three_rounds"
            pred = lab.predict_image(model, metadata, Image.open(val.iloc[0].path))
            assert len(pred) == 5 and pred.probability.is_monotonic_decreasing
            del model
    # Exercise test gate and selection/export using synthetic data, never real Test.
    with tempfile.TemporaryDirectory(dir=ROOT / "smoke_artifacts", prefix="compact_final_") as tmp:
        tmp = Path(tmp)
        for split in ["Train", "Test"]:
            (tmp / split).mkdir()
            lines = []
            for label in range(200):
                filename = f"{label:04d}.png"
                Image.new("RGB", (16,16), (label, int(split == "Test" and label > 0),0)).save(tmp / split / filename)
                lines.append(f"{filename} {label}")
            (tmp / f"{split.lower()}.txt").write_text("\n".join(lines))
        target = tmp / "results"
        for arch in lab.ARCHITECTURES:
            for i, stage in enumerate(lab.experiment_plan()):
                folder = target / arch / stage.name
                folder.mkdir(parents=True)
                shutil.copy2(out / arch / stage.name / "best.pth", folder / "best.pth")
                lab.engine.write_json(folder / "result.json", {
                    "architecture":arch, "experiment":stage.name, "val_top1":i/100,
                    "checkpoint":str(folder / "best.pth")})
        testcfg = replace(cfg, output_dir=str(target), data_root=str(tmp), batch_size=8)
        final = lab.final_evaluation(testcfg, names, {"train_sha256":"synthetic"})
        assert len(final) == 4 and (final.clean_samples == 199).all()
        assert (target / "deployment/bird_model.pth").exists()
        assert final.equals(lab.final_evaluation(testcfg,names,{"train_sha256":"synthetic"}))
    # Test the separate app against smoke results without replacing production paths.
    from streamlit.testing.v1 import AppTest
    source = (ROOT / "app_compact.py").read_text(encoding="utf-8")
    source = source.replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({str(ROOT)!r})")
    source = source.replace("STUDY = ROOT / DEFAULT_OUTPUT_DIR", f"STUDY = Path({str(out)!r})")
    app = AppTest.from_string(source).run(timeout=60)
    assert not app.exception and len(app.dataframe) == 2
    app.selectbox[0].select_index(len(app.selectbox[0].options)-1)
    app.run(timeout=60)
    assert not app.exception
    assert len(original.experiment_plan()) == 13
    lab.engine.write_json(out / "verification.json", {
        "purpose":"Tiny integration test, not assignment accuracy", "six_gpu_runs":True,
        "notebook_valid":True, "original_engine_isolated":True,
        "synthetic_finalization_and_app":True})
    print("PASS: compact six-run GPU pipeline, isolation, tables, inference, synthetic export and Streamlit.")
    print("Full compact experiments have NOT been run. Original notebook/results unchanged.")

if __name__ == "__main__": main()
