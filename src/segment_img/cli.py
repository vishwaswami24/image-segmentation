from __future__ import annotations

import argparse
from pathlib import Path

from .inference import collect_image_paths, load_labels, load_model, predict_instances, resolve_device
from .results import save_prediction_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run image segmentation with Mask R-CNN and export masks."
    )
    parser.add_argument("--input", type=Path, required=True, help="Image or folder to process.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory where segmentation results will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a custom fine-tuned Mask R-CNN checkpoint.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional text file with one class name per line.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Class count for a custom checkpoint when no labels file is supplied.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.6,
        help="Minimum confidence score for keeping a detection.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert soft masks into binary masks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Maximum number of detections to keep per image.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    labels = load_labels(args.labels_file.resolve() if args.labels_file else None)
    device = resolve_device(args.device)
    model, resolved_labels = load_model(
        device=device,
        checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
        labels=labels,
        num_classes=args.num_classes,
    )

    image_paths = collect_image_paths(input_path)
    if not image_paths:
        raise SystemExit(f"No supported images found in: {input_path}")

    for image_path in image_paths:
        image, predictions = predict_instances(
            model=model,
            image_path=image_path,
            device=device,
            labels=resolved_labels,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            top_k=args.top_k,
        )
        image_output_dir = _output_dir_for(input_path, image_path, output_dir)
        _write_outputs(
            source_image=image_path,
            image=image,
            predictions=predictions,
            destination=image_output_dir,
        )
        print(f"Processed {image_path} -> {image_output_dir}")


def _output_dir_for(input_root: Path, image_path: Path, output_dir: Path) -> Path:
    if input_root.is_file():
        return output_dir / image_path.stem
    relative = image_path.relative_to(input_root)
    return output_dir / relative.parent / relative.stem


def _write_outputs(
    source_image: Path,
    image,
    predictions,
    destination: Path,
) -> None:
    save_prediction_artifacts(
        source_image=source_image,
        image=image,
        predictions=predictions,
        destination=destination,
    )


if __name__ == "__main__":
    main()
