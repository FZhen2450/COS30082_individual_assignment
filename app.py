"""Streamlit model demonstration. Run: python -m streamlit run app.py."""
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from birdlab import load_predictor, predict_image

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="CUB-200 Bird Classifier", page_icon="🐦", layout="wide")
st.title("🐦 Bird Species Classification")
st.caption("COS30082 · MobileNetV3 / EfficientNetV2-S · 200 bird species")


@st.cache_resource(max_entries=2)
def cached_model(path, modified_ns):
    # CPU inference leaves the laptop GPU available for training. Building with
    # pretrained=False means deployment never downloads ImageNet weights.
    return load_predictor(path, "cpu")


choices = sorted((ROOT / "deployment").glob("*.pth"))
choices += sorted((ROOT / "artifacts").glob("*/*/best.pth"))
tab_predict, tab_experiments = st.tabs(["Test an image", "Experiment comparison"])
with tab_predict:
    if not choices:
        st.info("Train the notebook first. Its final evaluation exports deployment/bird_model.pth.")
        st.write("Local run: `.venv\\Scripts\\python.exe -m streamlit run app.py`")
    else:
        checkpoint_path = st.selectbox("Model checkpoint", choices,
                                       format_func=lambda p: str(p.relative_to(ROOT)))
        model, checkpoint = cached_model(str(checkpoint_path), checkpoint_path.stat().st_mtime_ns)
        st.write(f"**Architecture:** {checkpoint['architecture']} · "
                 f"**Experiment:** {checkpoint['experiment']['name']} · "
                 f"**Validation accuracy:** {checkpoint['val_metrics']['top1']:.2%}")
        uploaded = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            try:
                image = Image.open(uploaded)
                image.load()
                left, right = st.columns(2)
                left.image(image, caption=uploaded.name, width="stretch")
                predictions = predict_image(model, checkpoint, image)
                right.metric("Predicted species", predictions.iloc[0].species)
                right.dataframe(predictions.assign(
                    probability=predictions.probability.map(lambda p: f"{p:.2%}")),
                    hide_index=True, width="stretch")
                right.bar_chart(predictions.set_index("species"))
                st.caption("Scores are model probabilities, not calibrated certainty. "
                           "This classifier always chooses among its 200 species, even for non-bird images.")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
                st.error(f"Cannot read this image: {error}")

with tab_experiments:
    comparison = ROOT / "artifacts/comparison.csv"
    if comparison.exists():
        frame = pd.read_csv(comparison)
        view = frame[["architecture", "experiment", "val_top1", "val_average_per_class",
                      "delta_vs_baseline_pp", "gap", "best_epoch"]].copy()
        for column in ["val_top1", "val_average_per_class", "gap"]:
            view[column] *= 100
        st.write("Validation results (%) and change from baseline (percentage points)")
        st.dataframe(view.round(3), hide_index=True, width="stretch")
        st.bar_chart(frame.pivot(index="experiment", columns="architecture", values="val_top1") * 100)
        st.bar_chart(frame.pivot(index="experiment", columns="architecture", values="delta_vs_baseline_pp"))
        st.download_button("Download experiment results", comparison.read_bytes(),
                           "comparison.csv", "text/csv")
        run = st.selectbox("Learning curves", frame.checkpoint.tolist())
        history_path = ROOT / Path(run).parent / "history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path).set_index("epoch")
            st.line_chart(history[["fit_eval_top1", "val_top1"]] * 100)
            st.line_chart(history[["fit_eval_loss", "val_loss"]])
    else:
        st.info("Comparison charts will appear after experiments finish. No results are fabricated.")
    summary = ROOT / "artifacts/test_summary.csv"
    if summary.exists():
        st.write("Final test results (fractions from 0 to 1)")
        st.dataframe(pd.read_csv(summary), hide_index=True, width="stretch")
        st.caption("Model selection uses validation only. Clean metrics exclude byte-identical Train/Test overlap.")
