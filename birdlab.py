"""Reproducible CUB-200 experiments shared by the notebook and Streamlit.

Only two architectures are supported. Each ablation starts independently from
the same ImageNet weights and classifier seed; test data never selects weights.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from torchvision.ops import StochasticDepth
from tqdm.auto import tqdm

ARCHITECTURES = ("mobilenet_v3_large", "efficientnet_v2_s")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class Config:
    data_root: str = "."
    output_dir: str = "artifacts"
    seed: int = 42
    val_fraction: float = 0.2
    image_size: int = 224
    batch_size: int = 8
    accumulation_steps: int = 4
    num_workers: int = 0  # Reliable inside Windows Jupyter; no multiprocessing recursion.
    epochs: int = 15
    head_epochs: int = 2
    patience: int = 5
    backbone_lr: float = 3e-5
    head_lr: float = 3e-4
    require_gpu: bool = True
    pretrained: bool = True


@dataclass(frozen=True)
class Experiment:
    name: str
    geometry: bool = False
    color: bool = False
    grayscale: bool = False
    dropout: float = 0.0
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    erasing: bool = False


def experiment_plan():
    # Single-factor experiments are each compared with 00_baseline, not with
    # the preceding row. The final recipe tests the combination, not causality.
    return [
        Experiment("00_baseline"),
        Experiment("01_geometry", geometry=True),
        Experiment("02_color", color=True),
        Experiment("03_grayscale", grayscale=True),
        Experiment("04_dropout", dropout=0.3),
        Experiment("05_weight_decay", weight_decay=1e-3),
        Experiment("06_label_smoothing", label_smoothing=0.1),
        Experiment("07_combined", geometry=True, color=True, dropout=0.3,
                   weight_decay=1e-3, label_smoothing=0.1, erasing=True),
    ]


def read_annotations(root, split):
    root = Path(root).resolve()
    rows = []
    for line in (root / f"{split.lower()}.txt").read_text().splitlines():
        if not line.strip():
            continue
        filename, label = line.rsplit(maxsplit=1)
        path = (root / split.title() / filename).resolve()
        if not path.is_relative_to(root / split.title()):
            raise ValueError(f"Image outside {split} directory: {filename}")
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"filename": filename, "path": str(path), "label": int(label)})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.filename.duplicated().any():
        raise ValueError(f"Empty or duplicate annotation in {split}")
    if not frame.label.between(0, 199).all():
        raise ValueError("Expected CUB labels 0..199")
    return frame


def prepare_data(cfg):
    """Persist one stratified split and fail if an existing run changes data."""
    train = read_annotations(cfg.data_root, "train")
    if sorted(train.label.unique().tolist()) != list(range(200)):
        raise ValueError("Training data must include all 200 classes")
    # File contents, not absolute paths, identify this dataset across machines.
    print("Checking training images and content fingerprint...")
    digest = hashlib.sha256()
    hashes = []
    for row in tqdm(train.itertuples(), total=len(train), desc="Audit Train"):
        content_hash = hashlib.sha256(Path(row.path).read_bytes()).hexdigest()
        with Image.open(row.path) as image:
            image.verify()
        hashes.append(content_hash)
        digest.update(f"{row.filename}:{row.label}:{content_hash}\n".encode())
    train["sha256"] = hashes
    # Reserve the six ambiguous duplicate rows for fit, then stratify the unique
    # images. This keeps all 200 species in validation while preventing identical
    # content crossing partitions. A generic grouped-fold heuristic can omit
    # classes here because the three duplicate groups have conflicting labels.
    repeated = train.sha256.duplicated(keep=False)
    unique_idx = np.flatnonzero(~repeated)
    fit_unique, val_idx = train_test_split(
        unique_idx, test_size=int(np.ceil(len(train) * cfg.val_fraction)),
        stratify=train.loc[unique_idx, "label"], random_state=cfg.seed)
    fit_idx = np.concatenate([fit_unique, np.flatnonzero(repeated)])
    fit = train.iloc[sorted(fit_idx)].reset_index(drop=True)
    val = train.iloc[sorted(val_idx)].reset_index(drop=True)
    if fit.label.nunique() != 200 or val.label.nunique() != 200:
        raise ValueError("Grouped split must retain all 200 classes in each partition")
    duplicate_rows = train[train.sha256.duplicated(keep=False)]
    names = []
    for label in range(200):
        stems = train.loc[train.label == label, "filename"].map(
            lambda name: re.sub(r"_\d+_(?:\d+|xxx)$", "", Path(name).stem))
        if stems.nunique() != 1:
            raise ValueError(f"Cannot infer unique class name for label {label}")
        names.append(stems.iloc[0].replace("_", " "))
    manifest = {
        "seed": cfg.seed, "val_fraction": cfg.val_fraction,
        "train_sha256": digest.hexdigest(), "class_names": names,
        "split_method": "Stratified unique-image holdout; duplicate-content groups reserved for fit",
        "duplicate_training_rows": duplicate_rows[["filename", "label", "sha256"]].to_dict("records"),
        "fit": fit.filename.tolist(), "validation": val.filename.tolist(),
    }
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "split.json"
    if target.exists() and read_json(target) != manifest:
        raise ValueError("Dataset/split changed. Use a new output_dir for a new study.")
    write_json(target, manifest)
    return fit, val, names, manifest


def make_transform(exp=None, size=224):
    # Evaluation and deployment use exactly the same deterministic transform.
    steps = []
    if exp and exp.geometry:
        steps += [T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
                  T.RandomHorizontalFlip(), T.RandomRotation(10)]
    else:
        steps += [T.Resize((size, size))]
    if exp and exp.color:
        steps += [T.ColorJitter(brightness=0.15, contrast=0.15,
                               saturation=0.15, hue=0.02)]
    if exp and exp.grayscale:
        steps += [T.RandomGrayscale(p=0.2)]
    steps += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if exp and exp.erasing:
        steps += [T.RandomErasing(p=0.15, scale=(0.02, 0.1), value=0)]
    return T.Compose(steps)


class BirdDataset(Dataset):
    def __init__(self, frame, transform):
        self.rows = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row.label)


def make_loader(frame, cfg, exp=None, shuffle=False, generator=None):
    return DataLoader(
        BirdDataset(frame, make_transform(exp, cfg.image_size)),
        batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_model(architecture, num_classes, dropout=0.0, pretrained=True):
    if architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights, dropout=dropout)
    elif architecture == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights, dropout=dropout,
                                        stochastic_depth_prob=0.0)
    else:
        raise ValueError(f"Choose one of {ARCHITECTURES}")
    # Native stochastic depth is disabled across all runs for a clean baseline.
    for module in model.modules():
        if isinstance(module, StochasticDepth):
            module.p = 0.0
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def configure_trainable(model, head_only):
    for parameter in model.features.parameters():
        parameter.requires_grad = not head_only
    # MobileNet has a final pooling stage but no trainable parameters outside
    # features/classifier. Both architectures share these two parameter groups.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def device_for(cfg):
    if cfg.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select .venv Python kernel; run setup_gpu.ps1 first.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics_from_predictions(truth, predicted, n_classes):
    matrix = confusion_matrix(truth, predicted, labels=list(range(n_classes)))
    support = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), support,
                       out=np.zeros(n_classes, dtype=float), where=support > 0)
    return {
        "top1": float(np.trace(matrix) / max(1, matrix.sum())),
        "average_per_class": float(recall.mean()),
    }, matrix, recall, support


def run_epoch(model, loader, device, optimizer=None, scaler=None,
              smoothing=0.0, accumulation=1, head_only=False):
    training = optimizer is not None
    model.train(training)
    if training:
        # Tiny batches produce unstable BatchNorm statistics. Use pretrained
        # running statistics in every experiment, while affine weights can train.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        if head_only:
            model.features.eval()
        optimizer.zero_grad(set_to_none=True)
    truth, predicted, loss_sum, optimized_loss_sum = [], [], 0.0, 0.0
    seen = 0
    for batch_index, (images, labels) in enumerate(tqdm(loader, leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                raw_loss = nn.functional.cross_entropy(logits, labels)
                objective = nn.functional.cross_entropy(logits, labels, label_smoothing=smoothing)
            if training:
                # Weight by samples so a short last batch/window is correct.
                window_start = (batch_index // accumulation) * accumulation
                window_end = min(window_start + accumulation, len(loader))
                window_samples = min(window_end * loader.batch_size, len(loader.dataset)) - window_start * loader.batch_size
                scaler.scale(objective * len(labels) / window_samples).backward()
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        size = len(labels)
        seen += size
        loss_sum += raw_loss.item() * size
        optimized_loss_sum += objective.item() * size
        truth.extend(labels.detach().cpu().tolist())
        predicted.extend(logits.detach().argmax(1).cpu().tolist())
    if seen == 0:
        raise ValueError("Empty data loader")
    metrics, _, _, _ = metrics_from_predictions(truth, predicted, model.classifier[-1].out_features)
    return {**metrics, "loss": loss_sum / seen, "objective": optimized_loss_sum / seen}, truth, predicted


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_experiment(architecture, exp, cfg, fit, val, names, manifest):
    if architecture not in ARCHITECTURES:
        raise ValueError(architecture)
    if cfg.epochs <= cfg.head_epochs or cfg.head_epochs < 0 or cfg.batch_size < 1:
        raise ValueError("Need epochs > head_epochs >= 0 and batch_size >= 1")
    if cfg.accumulation_steps < 1 or cfg.num_workers != 0:
        raise ValueError("Use accumulation_steps >= 1 and num_workers=0 for reproducible notebook resume")
    folder = Path(cfg.output_dir) / architecture / exp.name
    folder.mkdir(parents=True, exist_ok=True)
    run_spec = {"config": asdict(cfg), "experiment": asdict(exp),
                "architecture": architecture, "dataset": manifest["train_sha256"]}
    spec_path = folder / "run.json"
    if spec_path.exists() and read_json(spec_path) != run_spec:
        raise ValueError(f"Run settings changed: {folder}. Use a new output_dir.")
    write_json(spec_path, run_spec)
    if (folder / "result.json").exists():
        print(f"Already complete: {architecture}/{exp.name}")
        return read_json(folder / "result.json")
    if (Path(cfg.output_dir) / "test_lock.json").exists():
        raise RuntimeError("Test evaluation has started. Start a new study directory to train further.")
    device = device_for(cfg)
    seed_everything(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = make_loader(fit, cfg, exp, True, generator)
    val_loader = make_loader(val, cfg)
    model = build_model(architecture, len(names), exp.dropout, cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": cfg.backbone_lr},
        {"params": model.classifier.parameters(), "lr": cfg.head_lr},
    ], weight_decay=exp.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs - cfg.head_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history, best, stale, start, elapsed = [], -1.0, 0, 0, 0.0
    last_path = folder / "last.pth"
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best, stale = state["history"], state["best"], state["stale"]
        start, elapsed = state["epoch"] + 1, state["elapsed"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        numpy_state = state["numpy_rng"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                             *numpy_state[2:]))
        generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming {architecture}/{exp.name} at epoch {start + 1}")
    try:
        for epoch in range(start, cfg.epochs):
            if stale >= cfg.patience:
                break
            tick = time.perf_counter()
            head_only = epoch < cfg.head_epochs
            configure_trainable(model, head_only)
            train_metrics, _, _ = run_epoch(
                model, train_loader, device, optimizer, scaler, exp.label_smoothing,
                cfg.accumulation_steps, head_only,
            )
            # Measure train/validation on identical deterministic, unaugmented
            # inputs with dropout off: a meaningful generalisation-gap estimate.
            fit_metrics, _, _ = run_epoch(model, make_loader(fit, cfg), device)
            val_metrics, _, _ = run_epoch(model, val_loader, device)
            if not head_only:
                scheduler.step()
            elapsed += time.perf_counter() - tick
            row = {"epoch": epoch + 1, "phase": "head" if head_only else "fine_tune",
                   **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"fit_eval_{k}": v for k, v in fit_metrics.items()},
                   **{f"val_{k}": v for k, v in val_metrics.items()},
                   "gap": fit_metrics["top1"] - val_metrics["top1"],
                   "backbone_lr": optimizer.param_groups[0]["lr"],
                   "head_lr": optimizer.param_groups[1]["lr"]}
            history.append(row)
            if val_metrics["top1"] > best:
                best = val_metrics["top1"]
                stale = 0
                save_checkpoint(folder / "best.pth", {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "architecture": architecture, "experiment": asdict(exp),
                    "class_names": names, "image_size": cfg.image_size,
                    "mean": list(MEAN), "std": list(STD), "epoch": epoch + 1,
                    "val_metrics": val_metrics, "train_fingerprint": manifest["train_sha256"],
                })
            elif not head_only:
                stale += 1
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            numpy_state = np.random.get_state()
            save_checkpoint(last_path, {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "history": history, "best": best, "stale": stale,
                "elapsed": elapsed, "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                "loader_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            })
            print(f"{architecture}/{exp.name} {epoch + 1}/{cfg.epochs} "
                  f"fit={fit_metrics['top1']:.2%} val={val_metrics['top1']:.2%} "
                  f"gap={row['gap']:.2%} best={best:.2%}", flush=True)
        checkpoint = torch.load(folder / "best.pth", map_location="cpu", weights_only=True)
        chosen = next(row for row in history if row["epoch"] == checkpoint["epoch"])
        result = {"architecture": architecture, "experiment": exp.name,
                  "best_epoch": checkpoint["epoch"], "epochs_run": len(history),
                  "val_top1": checkpoint["val_metrics"]["top1"],
                  "val_average_per_class": checkpoint["val_metrics"]["average_per_class"],
                  "fit_top1": chosen["fit_eval_top1"], "gap": chosen["gap"],
                  "training_seconds": elapsed, "pretrained": cfg.pretrained,
                  "checkpoint": str(folder / "best.pth")}
        write_json(folder / "result.json", result)
        return result
    finally:
        del model, optimizer, scaler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def results_table(output_dir):
    results = [read_json(p) for p in sorted(Path(output_dir).glob("*/*/result.json"))]
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    baseline = frame[frame.experiment == "00_baseline"].set_index("architecture").val_top1
    frame["delta_vs_baseline_pp"] = [
        100 * (row.val_top1 - baseline.get(row.architecture, np.nan))
        for row in frame.itertuples()]
    frame.to_csv(Path(output_dir) / "comparison.csv", index=False)
    return frame


def plot_results(output_dir):
    frame = results_table(output_dir)
    if frame.empty:
        print("No completed experiments yet.")
        return
    charts = Path(output_dir) / "charts"
    charts.mkdir(exist_ok=True)
    pivot = frame.pivot(index="experiment", columns="architecture", values="val_top1") * 100
    ax = pivot.plot.bar(figsize=(12, 5), ylabel="Validation Top-1 (%)", rot=30)
    ax.figure.tight_layout()
    ax.figure.savefig(charts / "ablation_accuracy.png", dpi=160)
    plt.show()
    pivot = frame.pivot(index="experiment", columns="architecture", values="delta_vs_baseline_pp")
    ax = pivot.plot.bar(figsize=(12, 5), ylabel="Change vs baseline (percentage points)", rot=30)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.figure.tight_layout()
    ax.figure.savefig(charts / "ablation_delta.png", dpi=160)
    plt.show()


def plot_history(checkpoint_path):
    history = pd.read_csv(Path(checkpoint_path).parent / "history.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history.epoch, history.fit_eval_loss, label="Fit (eval)")
    axes[0].plot(history.epoch, history.val_loss, label="Validation")
    axes[0].set(title="Unsmoothed cross entropy", xlabel="Epoch")
    axes[1].plot(history.epoch, history.fit_eval_top1 * 100, label="Fit (eval)")
    axes[1].plot(history.epoch, history.val_top1 * 100, label="Validation")
    axes[1].set(title="Accuracy (%)", xlabel="Epoch")
    axes[2].plot(history.epoch, history.gap * 100)
    axes[2].set(title="Fit - validation gap (pp)", xlabel="Epoch")
    for ax in axes[:2]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(checkpoint_path).parent / "learning_curves.png", dpi=160)
    plt.show()


def load_predictor(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["architecture"], len(checkpoint["class_names"]),
                        checkpoint["experiment"]["dropout"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_image(model, checkpoint, image, device="cpu", top_k=5):
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = make_transform(size=checkpoint["image_size"])(image).unsqueeze(0).to(device)
    probs = model(tensor).softmax(1)[0].cpu()
    values, indices = probs.topk(min(top_k, len(probs)))
    return pd.DataFrame({"species": [checkpoint["class_names"][i] for i in indices.tolist()],
                         "probability": values.tolist()})


def final_evaluation(cfg, names, manifest):
    """Lock selection BEFORE reading Test. Evaluate baseline and val winner only.

    Reruns reuse saved results; they cannot silently change the selected models.
    Test comparisons are descriptive and cannot justify selecting a new recipe.
    """
    out = Path(cfg.output_dir)
    frame = results_table(out)
    expected = {(arch, exp.name) for arch in ARCHITECTURES for exp in experiment_plan()}
    actual = set(zip(frame.architecture, frame.experiment)) if not frame.empty else set()
    if not expected.issubset(actual):
        raise RuntimeError("Complete all 8 experiments for BOTH architectures before opening Test.")
    selected = []
    for architecture in ARCHITECTURES:
        rows = frame[frame.architecture == architecture]
        winner = rows.sort_values(["val_top1", "experiment"], ascending=[False, True]).iloc[0]
        baseline = rows[rows.experiment == "00_baseline"].iloc[0]
        for row in [baseline, winner]:
            key = f"{architecture}/{row.experiment}"
            if key not in selected:
                selected.append(key)
    lock = {"selected": selected, "global_winner": str(frame.sort_values(
        ["val_top1", "architecture", "experiment"], ascending=[False, True, True]).iloc[0].checkpoint),
        "train_fingerprint": manifest["train_sha256"]}
    lock_path = out / "test_lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise RuntimeError("Selection changed after test access. Do not tune against Test.")
    write_json(lock_path, lock)
    test = read_annotations(cfg.data_root, "test")
    training_hashes = {hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in read_annotations(cfg.data_root, "train").path}
    digest = hashlib.sha256()
    overlap = []
    for row in test.itertuples():
        hashed = hashlib.sha256(Path(row.path).read_bytes()).hexdigest()
        overlap.append(hashed in training_hashes)
        digest.update(f"{row.filename}:{row.label}:{hashed}\n".encode())
    test_signature = {"test_sha256": digest.hexdigest(),
                      "train_test_identical": test.loc[overlap, "filename"].tolist()}
    if (out / "test_data.json").exists() and read_json(out / "test_data.json") != test_signature:
        raise ValueError("Test data changed after evaluation")
    write_json(out / "test_data.json", test_signature)
    if sorted(test.label.unique().tolist()) != list(range(len(names))):
        raise ValueError("Test must include all 200 classes to compute the required macro metric")
    results = []
    for key in selected:
        destination = out / key / "test"
        destination.mkdir(parents=True, exist_ok=True)
        metric_path = destination / "metrics.json"
        if metric_path.exists():
            results.append(read_json(metric_path))
            continue
        model, checkpoint = load_predictor(out / key / "best.pth", device_for(cfg))
        metrics, truth, predicted = run_epoch(model, make_loader(test, cfg), device_for(cfg))
        _, matrix, recall, support = metrics_from_predictions(truth, predicted, len(names))
        pd.DataFrame({"filename": test.filename, "truth": truth, "prediction": predicted,
                      "overlaps_training": overlap}).to_csv(
            destination / "predictions.csv", index=False)
        pd.DataFrame({"species": names, "support": support, "accuracy": recall}).to_csv(
            destination / "per_class.csv", index=False)
        pd.DataFrame(matrix, index=names, columns=names).to_csv(destination / "confusion_matrix.csv")
        write_json(destination / "classification_report.json", classification_report(
            truth, predicted, labels=list(range(len(names))), target_names=names,
            output_dict=True, zero_division=0))
        metrics.update({"run": key, "samples": len(test)})
        clean = ~np.asarray(overlap)
        clean_metrics, _, _, clean_support = metrics_from_predictions(
            np.asarray(truth)[clean], np.asarray(predicted)[clean], len(names))
        metrics.update({"overlap_count": int(sum(overlap)), "clean_samples": int(clean.sum()),
                        "clean_top1": clean_metrics["top1"],
                        "clean_average_per_class": clean_metrics["average_per_class"],
                        "clean_classes_present": int((clean_support > 0).sum())})
        # Standalone figures also work in headless Python/Docker without Tk.
        fig = Figure(figsize=(10, 9))
        ax = fig.subplots()
        im = ax.imshow(matrix, cmap="Blues")
        ax.set(xlabel="Predicted class ID", ylabel="True class ID", title=key)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(destination / "confusion_matrix.png", dpi=160)
        plt.close(fig)
        # Correct/incorrect image examples support error analysis in the report.
        good = [i for i, (a, b) in enumerate(zip(truth, predicted)) if a == b][:4]
        bad = [i for i, (a, b) in enumerate(zip(truth, predicted)) if a != b][:4]
        fig = Figure(figsize=(14, 7))
        axes = fig.subplots(2, 4)
        for axis in axes.flat:
            axis.axis("off")
        for row_index, indices in enumerate([good, bad]):
            for column, index in enumerate(indices):
                with Image.open(test.iloc[index].path) as image:
                    axes[row_index, column].imshow(ImageOps.exif_transpose(image).convert("RGB"))
                axes[row_index, column].set_title(
                    f"True: {names[truth[index]]}\nPred: {names[predicted[index]]}", fontsize=8)
        fig.tight_layout()
        fig.savefig(destination / "prediction_examples.png", dpi=160)
        plt.close(fig)
        # Mark this evaluation complete only after every report artifact exists.
        write_json(metric_path, metrics)
        results.append(metrics)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = pd.DataFrame(results)
    summary.to_csv(out / "test_summary.csv", index=False)
    # Deployment is selected by VALIDATION, irrespective of test ranking.
    import shutil
    Path("deployment").mkdir(exist_ok=True)
    shutil.copy2(lock["global_winner"], "deployment/bird_model.pth")
    write_json("deployment/model_info.json", {**lock, "test_results": results,
                                              "selection": "validation top1 only"})
    return summary
