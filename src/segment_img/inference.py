from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor

BACKGROUND_LABEL = "__background__"
COCO_WEIGHTS = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class InstancePrediction:
    index: int
    label_id: int
    label: str
    score: float
    box: tuple[int, int, int, int]
    mask: np.ndarray


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    return torch.device(requested)


def collect_image_paths(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_labels(labels_file: Path | None) -> list[str] | None:
    if labels_file is None:
        return None

    labels = labels_file.read_text(encoding="utf-8").splitlines()
    return normalize_labels(labels)


def normalize_labels(raw_labels: Sequence[str] | None) -> list[str] | None:
    if raw_labels is None:
        return None

    labels = [line.strip() for line in raw_labels if line.strip()]
    if not labels:
        raise ValueError("At least one non-empty label is required.")

    if labels[0].lower() not in {BACKGROUND_LABEL, "background"}:
        labels.insert(0, BACKGROUND_LABEL)
    else:
        labels[0] = BACKGROUND_LABEL

    return labels


def load_model(
    device: torch.device,
    checkpoint: Path | None = None,
    labels: Sequence[str] | None = None,
    num_classes: int | None = None,
):
    if checkpoint is None:
        model = maskrcnn_resnet50_fpn(weights=COCO_WEIGHTS)
        resolved_labels = list(COCO_WEIGHTS.meta["categories"])
    else:
        resolved_num_classes = _resolve_num_classes(labels=labels, num_classes=num_classes)
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=resolved_num_classes,
        )
        state_dict = _extract_state_dict(torch.load(checkpoint, map_location="cpu"))
        cleaned_state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        model.load_state_dict(cleaned_state_dict)
        resolved_labels = _build_fallback_labels(labels, resolved_num_classes)

    model.to(device)
    model.eval()
    return model, resolved_labels


def predict_instances(
    model,
    image_path: Path,
    device: torch.device,
    labels: Sequence[str],
    score_threshold: float,
    mask_threshold: float,
    top_k: int | None,
) -> tuple[Image.Image, list[InstancePrediction]]:
    image = Image.open(image_path).convert("RGB")
    tensor = to_tensor(image).to(device)

    with torch.inference_mode():
        output = model([tensor])[0]

    scores = output["scores"].detach().cpu()
    keep = torch.nonzero(scores >= score_threshold).flatten()

    if keep.numel() == 0:
        return image, []

    if top_k is not None and top_k > 0:
        keep = keep[:top_k]

    boxes = output["boxes"][keep].detach().cpu().tolist()
    label_ids = output["labels"][keep].detach().cpu().tolist()
    masks = output["masks"][keep, 0].detach().cpu().numpy() >= mask_threshold
    kept_scores = scores[keep].tolist()

    predictions: list[InstancePrediction] = []
    for index, (box, label_id, score, mask) in enumerate(
        zip(boxes, label_ids, kept_scores, masks),
        start=1,
    ):
        rounded_box = tuple(int(round(value)) for value in box)
        predictions.append(
            InstancePrediction(
                index=index,
                label_id=label_id,
                label=resolve_label_name(label_id, labels),
                score=float(score),
                box=rounded_box,
                mask=np.asarray(mask, dtype=bool),
            )
        )

    return image, predictions


def resolve_label_name(label_id: int, labels: Sequence[str]) -> str:
    if 0 <= label_id < len(labels):
        label = labels[label_id]
        if label not in {BACKGROUND_LABEL, "N/A"}:
            return label
    return f"class_{label_id}"


def _resolve_num_classes(
    labels: Sequence[str] | None,
    num_classes: int | None,
) -> int:
    inferred_classes = len(labels) if labels is not None else None
    if num_classes is None and inferred_classes is None:
        raise ValueError(
            "When using a custom checkpoint, provide --labels-file or --num-classes."
        )
    if num_classes is not None and inferred_classes is not None and num_classes != inferred_classes:
        raise ValueError(
            f"num_classes={num_classes} does not match labels file size={inferred_classes}."
        )
    return num_classes or inferred_classes  # type: ignore[return-value]


def _build_fallback_labels(
    labels: Sequence[str] | None,
    num_classes: int,
) -> list[str]:
    if labels is not None:
        return list(labels)
    return [BACKGROUND_LABEL] + [f"class_{index}" for index in range(1, num_classes)]


def _extract_state_dict(checkpoint_data):
    if isinstance(checkpoint_data, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint_data.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint_data, dict):
        return checkpoint_data
    raise TypeError("Unsupported checkpoint format. Expected a state_dict-like object.")
