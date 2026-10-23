"""Choose one validation-selected model per architecture and version."""
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from predictor import load_predictor, predict_image

ROOT = Path(__file__).resolve().parent
WORKFLOWS = {"Version 1": "version_1", "Version 2": "version_2"}
ARCHITECTURES = {"efficientnet_v2_s": "EfficientNetV2-S (best)", "mobilenet_v1": "MobileNetV1 (best)"}


@st.cache_resource(max_entries=2)
def cached_model(path, modified_ns):
    return load_predictor(path, "cpu")


def model_label(path):
    return ARCHITECTURES[path.stem]


def available_models(workflow):
    """Read the exported selection, never enumerate every training checkpoint."""
    manifest_path = ROOT / "models/manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest["versions"][WORKFLOWS[workflow]]["models"]
    return [ROOT / selected[arch]["file"] for arch in ARCHITECTURES if (ROOT / selected[arch]["file"]).is_file()]


def main(default_workflow="Version 2"):
    st.set_page_config(page_title="Bird Species Classifier", page_icon="🐦", layout="wide")
    st.title("🐦 Bird Species Classifier")
    workflow = st.selectbox("Choose a version", list(WORKFLOWS), index=list(WORKFLOWS).index(default_workflow))
    choices = available_models(workflow)
    if not choices:
        st.info(f"The selected models for {workflow} are unavailable. Please choose another version.")
        return
    selected = st.selectbox("Choose a model", choices, format_func=model_label, key=f"model_{workflow}")
    upload = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png", "webp"])
    if upload is not None:
        try:
            with Image.open(upload) as opened:
                opened.load()
                photo = ImageOps.exif_transpose(opened).convert("RGB")
            with st.spinner("Identifying the bird..."):
                model, metadata = cached_model(str(selected), selected.stat().st_mtime_ns)
                predictions = predict_image(model, metadata, photo, top_k=5)
            left, right = st.columns(2)
            left.image(photo, caption=upload.name, width="stretch")
            right.subheader("Top-5 predictions")
            for rank, row in enumerate(predictions, 1):
                right.markdown(f"**{rank}. {row['species']}** — {row['probability']:.2%}")
                right.progress(row["probability"])
            right.caption("Percentages show the model's probability for this photo.")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            st.error("This image could not be opened. Please upload a JPG, PNG or WebP photo.")


if __name__ == "__main__":
    main()
