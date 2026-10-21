"""Check new-study isolation and Streamlit tables using smoke checkpoints."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
import birdlab as b

def main():
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    assert not app.exception
    if not list((ROOT / b.DEFAULT_OUTPUT_DIR).glob("*/*/best.pth")):
        assert any("no trained checkpoint" in item.value for item in app.info)
        assert not app.dataframe  # Must not substitute old artifacts.
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    source = source.replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({str(ROOT)!r})")
    source = source.replace("STUDY = ROOT / DEFAULT_OUTPUT_DIR", 'STUDY = ROOT / "smoke_artifacts/retrain_v3"')
    smoke = AppTest.from_string(source).run(timeout=60)
    assert not smoke.exception and len(smoke.dataframe) == 2
    assert any("Val Top-1 (%)" in col for col in smoke.dataframe[0].value.columns)
    assert "Train Accuracy (%)" in smoke.dataframe[1].value.columns
    checkpoint_selector = next(s for s in smoke.selectbox if s.label == "Model checkpoint")
    checkpoint_selector.select_index(len(checkpoint_selector.options) - 1)
    smoke.run(timeout=60)
    assert not smoke.exception
    assert not smoke.get("arrow_vega_lite_chart") and not smoke.get("pyplot")
    print("PASS: new-study isolation, real checkpoint selection, comparison/epoch tables, no model charts.")

if __name__ == "__main__":
    main()
