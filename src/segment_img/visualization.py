from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .inference import InstancePrediction

PALETTE: tuple[tuple[int, int, int], ...] = (
    (244, 67, 54),
    (33, 150, 243),
    (76, 175, 80),
    (255, 152, 0),
    (156, 39, 176),
    (0, 188, 212),
    (255, 235, 59),
    (121, 85, 72),
)


def color_for_index(index: int) -> tuple[int, int, int]:
    return PALETTE[(index - 1) % len(PALETTE)]


def create_overlay(
    image: Image.Image,
    predictions: Iterable[InstancePrediction],
    alpha: float = 0.45,
) -> Image.Image:
    predictions = list(predictions)
    base = np.array(image, dtype=np.float32, copy=True)

    for prediction in predictions:
        color = np.asarray(color_for_index(prediction.index), dtype=np.float32)
        mask = prediction.mask
        base[mask] = base[mask] * (1.0 - alpha) + color * alpha

    result = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()

    for prediction in predictions:
        color = color_for_index(prediction.index)
        x1, y1, x2, y2 = prediction.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        _draw_caption(
            draw=draw,
            font=font,
            x=x1,
            y=y1,
            caption=f"{prediction.label} {prediction.score:.2f}",
            color=color,
        )

    return result


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    caption: str,
    color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = draw.textbbox((x, y), caption, font=font)
    text_height = bottom - top
    padding = 3
    text_top = max(0, y - text_height - (padding * 2))
    background = (
        x,
        text_top,
        x + (right - left) + (padding * 2),
        text_top + text_height + (padding * 2),
    )
    draw.rectangle(background, fill=color)
    draw.text((x + padding, text_top + padding), caption, fill="white", font=font)
