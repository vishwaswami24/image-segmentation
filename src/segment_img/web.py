from __future__ import annotations

import argparse
from collections import Counter
import secrets
from datetime import datetime
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from .inference import IMAGE_SUFFIXES, load_model, normalize_labels, predict_instances, resolve_device
from .results import save_prediction_artifacts


def create_app(output_root: Path | None = None) -> FastAPI:
    assets_root = files("segment_img").joinpath("web_assets")
    templates = Jinja2Templates(directory=str(assets_root.joinpath("templates")))
    static_root = Path(str(assets_root.joinpath("static")))
    runs_root = (output_root or Path.cwd() / "outputs" / "web_runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="Segment-Img UI",
        description="Browser UI for Mask R-CNN image segmentation.",
    )
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
    app.mount("/runs", StaticFiles(directory=str(runs_root)), name="runs")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=_context(),
        )

    @app.post("/segment", response_class=HTMLResponse)
    async def segment(
        request: Request,
        image_file: UploadFile | None = File(default=None),
        score_threshold: float = Form(default=0.6),
        mask_threshold: float = Form(default=0.5),
        top_k: int = Form(default=20),
        device: str = Form(default="auto"),
        checkpoint_path: str = Form(default=""),
        labels_text: str = Form(default=""),
        num_classes: int | None = Form(default=None),
    ):
        form_state = {
            "score_threshold": score_threshold,
            "mask_threshold": mask_threshold,
            "top_k": top_k,
            "device": device,
            "checkpoint_path": checkpoint_path,
            "labels_text": labels_text,
            "num_classes": num_classes,
        }

        try:
            if image_file is None or not image_file.filename:
                raise ValueError("Choose an image to segment.")

            resolved_labels = _labels_from_text(labels_text)
            checkpoint = _resolve_checkpoint(checkpoint_path)

            request_id = _request_id()
            run_dir = runs_root / request_id
            run_dir.mkdir(parents=True, exist_ok=True)

            source_name = _source_filename(image_file.filename)
            source_path = run_dir / source_name
            payload = await image_file.read()
            source_path.write_bytes(payload)

            model, model_labels, model_device = _cached_model(
                device_name=device,
                checkpoint_path=str(checkpoint) if checkpoint else None,
                labels=tuple(resolved_labels) if resolved_labels else None,
                num_classes=num_classes,
            )
            source_image, predictions = predict_instances(
                model=model,
                image_path=source_path,
                device=model_device,
                labels=model_labels,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                top_k=top_k,
            )
            summary = save_prediction_artifacts(
                source_image=source_path,
                image=source_image,
                predictions=predictions,
                destination=run_dir,
            )

            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context=_context(
                    form_state=form_state,
                    result=_result_context(
                        request_id=request_id,
                        source_name=source_name,
                        summary=summary,
                        model_name=checkpoint.name if checkpoint else "COCO pretrained Mask R-CNN",
                        device=str(model_device),
                    ),
                ),
            )
        except Exception as exc:  # pragma: no cover - rendered in browser
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context=_context(form_state=form_state, error=str(exc)),
                status_code=400,
            )

    return app


@lru_cache(maxsize=6)
def _cached_model(
    device_name: str,
    checkpoint_path: str | None,
    labels: tuple[str, ...] | None,
    num_classes: int | None,
):
    device = resolve_device(device_name)
    model, resolved_labels = load_model(
        device=device,
        checkpoint=Path(checkpoint_path) if checkpoint_path else None,
        labels=labels,
        num_classes=num_classes,
    )
    return model, tuple(resolved_labels), device


def _context(
    form_state: dict | None = None,
    result: dict | None = None,
    error: str | None = None,
) -> dict:
    return {
        "title": "Segment-Img UI",
        "form": {
            "score_threshold": 0.6,
            "mask_threshold": 0.5,
            "top_k": 20,
            "device": "auto",
            "checkpoint_path": "",
            "labels_text": "",
            "num_classes": None,
            **(form_state or {}),
        },
        "result": result,
        "error": error,
    }


def _result_context(
    request_id: str,
    source_name: str,
    summary: dict,
    model_name: str,
    device: str,
) -> dict:
    detections = []
    label_counts = Counter()
    label_score_totals: dict[str, float] = {}
    label_coverage_totals: dict[str, float] = {}
    score_total = 0.0

    for detection in summary["detections"]:
        label = detection["label"]
        score_pct = round(detection["score"] * 100, 2)
        coverage_pct = round(detection["coverage_ratio"] * 100, 2)

        label_counts[label] += 1
        label_score_totals[label] = label_score_totals.get(label, 0.0) + score_pct
        label_coverage_totals[label] = label_coverage_totals.get(label, 0.0) + coverage_pct
        score_total += score_pct

        detections.append(
            {
                **detection,
                "mask_url": f"/runs/{request_id}/{detection['mask_path']}",
                "coverage_pct": coverage_pct,
                "score_pct": score_pct,
                "color_css": _color_css(detection["color_rgb"]),
            }
        )

    label_breakdown = _label_breakdown(
        label_counts=label_counts,
        label_score_totals=label_score_totals,
        label_coverage_totals=label_coverage_totals,
    )
    dominant_label = label_breakdown[0]["label"] if label_breakdown else "No detections"
    largest_detection = max(detections, key=lambda item: item["coverage_pct"], default=None)
    avg_confidence_pct = round(score_total / len(detections), 2) if detections else 0.0

    return {
        "request_id": request_id,
        "model_name": model_name,
        "device": device,
        "source_url": f"/runs/{request_id}/{source_name}",
        "overlay_url": f"/runs/{request_id}/overlay.png",
        "summary_url": f"/runs/{request_id}/summary.json",
        "image_width": summary["image_size"]["width"],
        "image_height": summary["image_size"]["height"],
        "detection_count": summary["detection_count"],
        "segmented_pixel_pct": round(summary["segmented_pixel_ratio"] * 100, 2),
        "avg_confidence_pct": avg_confidence_pct,
        "unique_label_count": len(label_breakdown),
        "dominant_label": dominant_label,
        "largest_detection": largest_detection,
        "label_breakdown": label_breakdown,
        "detections": detections,
    }


def _label_breakdown(
    label_counts: Counter,
    label_score_totals: dict[str, float],
    label_coverage_totals: dict[str, float],
) -> list[dict]:
    if not label_counts:
        return []

    max_count = max(label_counts.values())
    rows = []
    for label, count in label_counts.items():
        rows.append(
            {
                "label": label,
                "count": count,
                "avg_score_pct": round(label_score_totals[label] / count, 2),
                "coverage_pct": round(label_coverage_totals[label], 2),
                "bar_pct": round((count / max_count) * 100, 2),
            }
        )

    rows.sort(key=lambda item: (-item["count"], -item["avg_score_pct"], item["label"]))
    return rows


def _color_css(color_rgb: list[int]) -> str:
    return f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"


def _labels_from_text(labels_text: str) -> list[str] | None:
    if not labels_text.strip():
        return None
    return normalize_labels(labels_text.splitlines())


def _resolve_checkpoint(checkpoint_path: str) -> Path | None:
    if not checkpoint_path.strip():
        return None
    checkpoint = Path(checkpoint_path.strip()).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _request_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{secrets.token_hex(3)}"


def _source_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".png"
    return f"source{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Segment-Img browser UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the UI server.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "web_runs",
        help="Directory where UI runs and generated artifacts are stored.",
    )
    args = parser.parse_args()

    app = create_app(output_root=args.output_root.resolve())
    uvicorn.run(app, host=args.host, port=args.port)


app = create_app()
