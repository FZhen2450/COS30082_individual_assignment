"""Streamlit demo for the fresh retraining study; result comparisons use tables."""
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from birdlab import DEFAULT_OUTPUT_DIR, comparison_table, history_table, load_predictor, predict_image

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / DEFAULT_OUTPUT_DIR
st.set_page_config(page_title="CUB-200 Bird Classifier", page_icon="🐦", layout="wide")
st.title("🐦 Bird Species Classification")
st.caption("MobileNetV1 / EfficientNetV2-S · fresh retraining study · 200 species")


@st.cache_resource(max_entries=2)
def cached_model(path, modified_ns):
    return load_predictor(path, "cpu")


choices = sorted((STUDY / "deployment").glob("*.pth"))
choices += sorted(STUDY.glob("*/*/best.pth"))
tab_predict, tab_tables, tab_previews = st.tabs(["Test an image", "Comparison tables", "Augmentation examples"])
with tab_predict:
    if not choices:
        st.info("The new study has no trained checkpoint yet. Run Stage 0 in the revised notebook.")
    else:
        path = st.selectbox("Model checkpoint", choices, format_func=lambda p: str(p.relative_to(STUDY)))
        model, metadata = cached_model(str(path), path.stat().st_mtime_ns)
        st.write(f"**Model:** {metadata['architecture']} · **Stage:** {metadata['experiment']['name']} · "
                 f"**Val Top-1 Accuracy:** {metadata['val_metrics']['top1']:.2%}")
        upload = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png", "webp"])
        if upload:
            try:
                image = Image.open(upload)
                image.load()
                left, right = st.columns(2)
                left.image(image, caption=upload.name, width="stretch")
                result = predict_image(model, metadata, image)
                right.metric("Predicted species", result.iloc[0].species)
                right.dataframe(result.assign(probability=result.probability.map(lambda p: f"{p:.2%}")),
                                hide_index=True, width="stretch")
                st.caption("Probabilities for one image are not dataset accuracy. The model always picks among its 200 species.")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
                st.error(f"Cannot read this image: {error}")

with tab_tables:
    table = comparison_table(STUDY)
    if table.empty:
        st.info("Paired comparison tables appear after the new baseline finishes. Old-study results are not mixed in.")
    else:
        st.write("Validation accuracy (%) and improvements (percentage points)")
        st.dataframe(table, width="stretch")
        st.caption("Each stage keeps earlier settings. Negative changes are shown; stronger augmentation is not guaranteed to help.")
        csv = STUDY / "comparison_display.csv"
        st.download_button("Download comparison table", csv.read_bytes(), "comparison_display.csv", "text/csv")
        histories = sorted(STUDY.glob("*/*/history.csv"))
        selected = st.selectbox("Per-epoch training metrics", histories,
                                format_func=lambda p: str(p.parent.relative_to(STUDY)))
        st.dataframe(history_table(selected.parent / "best.pth").round(4), hide_index=True, width="stretch")
        st.caption("Train Accuracy uses actual training batches. Train Eval Accuracy disables random augmentation and dropout.")
    test = STUDY / "test_summary.csv"
    if test.exists():
        metrics = pd.read_csv(test)
        names = {"top1": "Test Top-1 Accuracy (%)", "average_per_class": "Test Avg/Class Accuracy (%)",
                 "clean_top1": "Clean Test Top-1 (%)", "clean_average_per_class": "Clean Test Avg/Class (%)"}
        for column in names:
            metrics[column] *= 100
        st.write("Final Test results (model selection used validation only)")
        st.dataframe(metrics.rename(columns=names).round(3), hide_index=True, width="stretch")

with tab_previews:
    st.write("Image augmentations, not model-comparison pictures")
    st.caption("Crop/rotation/colour examples are selected for visible differences. Runtime training remains random. "
               "Fine-tuning and regularization do not alter input pixels and have no artificial image preview.")
    previews = sorted((STUDY / "visualizations/method_previews").glob("*.png"))
    if previews:
        image_path = st.selectbox("Augmentation method", previews, format_func=lambda p: p.stem)
        st.image(str(image_path), width="stretch")
    else:
        st.info("Run an image-augmentation preview cell in the new notebook to generate examples.")
