from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .inference import InstancePrediction
from .visualization import color_for_index, create_overlay


def save_prediction_artifacts(
    source_image: Path,
    image: Image.Image,
    predictions: list[InstancePrediction],
    destination: Path,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    masks_dir = destination / "masks"
    masks_dir.mkdir(exist_ok=True)

    overlay = create_overlay(image=image, predictions=predictions)
    overlay.save(destination / "overlay.png")

    total_pixels = max(1, image.width * image.height)
    union_mask = np.zeros((image.height, image.width), dtype=bool)
    detections: list[dict] = []

    for prediction in predictions:
        label_slug = _slugify(prediction.label)
        mask_name = f"{prediction.index:03d}_{label_slug}.png"
        mask_path = masks_dir / mask_name
        mask_pixels = int(np.count_nonzero(prediction.mask))

        Image.fromarray(np.uint8(prediction.mask) * 255, mode="L").save(mask_path)
        union_mask |= prediction.mask

        detections.append(
            {
                "instance_id": prediction.index,
                "label_id": prediction.label_id,
                "label": prediction.label,
                "score": round(prediction.score, 4),
                "box": {
                    "xmin": prediction.box[0],
                    "ymin": prediction.box[1],
                    "xmax": prediction.box[2],
                    "ymax": prediction.box[3],
                },
                "mask_path": (Path("masks") / mask_name).as_posix(),
                "color_rgb": list(color_for_index(prediction.index)),
                "mask_pixels": mask_pixels,
                "coverage_ratio": round(mask_pixels / total_pixels, 4),
            }
        )

    summary = {
        "source_image": str(source_image),
        "overlay_image": "overlay.png",
        "image_size": {
            "width": image.width,
            "height": image.height,
        },
        "detection_count": len(predictions),
        "segmented_pixel_ratio": round(float(np.count_nonzero(union_mask)) / total_pixels, 4),
        "detections": detections,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()
    return slug or "object"
