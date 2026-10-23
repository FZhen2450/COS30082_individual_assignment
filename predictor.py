"""CPU inference without the training, plotting or dataset dependencies."""
import timm
import torch
from PIL import ImageOps
from torch import nn
from torchvision import models, transforms


class MobileNetV1(nn.Module):
    """Match the exact module names used in the saved MobileNetV1 weights."""
    def __init__(self, num_classes, dropout=0.0):
        super().__init__()
        backbone = timm.create_model("mobilenetv1_100.ra4_e3600_r224_in1k", pretrained=False)
        self.features = nn.Sequential(backbone.conv_stem, backbone.bn1, backbone.blocks, backbone.conv_head, backbone.bn2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.classifier.in_features, num_classes))

    def forward(self, images):
        features = self.pool(self.features(images)).flatten(1)
        return self.classifier(features)


def load_predictor(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    num_classes = len(checkpoint["class_names"])
    dropout = checkpoint["experiment"]["dropout"]
    if checkpoint["architecture"] == "mobilenet_v1":
        model = MobileNetV1(num_classes, dropout)
    elif checkpoint["architecture"] == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None, dropout=dropout, stochastic_depth_prob=0.0)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model architecture in checkpoint")
    # Keep only preprocessing metadata in Streamlit's cache, not a second copy
    # of every tensor alongside the reconstructed model.
    model.load_state_dict(checkpoint.pop("model"), strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_image(model, checkpoint, image, device="cpu", top_k=5):
    image = ImageOps.exif_transpose(image).convert("RGB")
    size = checkpoint["image_size"]
    preprocess = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])
    tensor = preprocess(image).unsqueeze(0).to(device)
    probabilities = model(tensor).softmax(1)[0].cpu()
    values, indices = probabilities.topk(min(top_k, len(probabilities)))
    return [{"species": checkpoint["class_names"][index], "probability": float(value)} for value, index in zip(values.tolist(), indices.tolist())]
