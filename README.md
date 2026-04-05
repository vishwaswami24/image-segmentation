# Segment-Img 
[![Python >=3.10](https://img.shields.io/badge/Python-3.14+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.17+-f26521?logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-005571?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243.svg)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-10.0+-F7B800.svg?logo=pillow&logoColor=white)](https://pillow.readthedocs.io/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.30+-800000.svg)](https://www.uvicorn.org/)

`Segment-Img` is a lightweight starter project for image segmentation with Mask R-CNN. It goes beyond bounding boxes by predicting the exact pixel-level shape of each detected object, which is the core idea behind workflows used in areas like medical imaging and autonomous driving.

## Features

- Runs instance segmentation with a pretrained `torchvision` Mask R-CNN model.
- Supports custom checkpoints for domain-specific datasets.
- Saves an overlay image with masks, bounding boxes, and labels.
- Exports one binary mask per detected object.
- Writes a `summary.json` file for downstream pipelines.
- Includes a browser UI for interactive uploads and mask inspection.
- Adds a richer analysis dashboard with before-and-after comparison, focus mode, and detection filtering.

## Project Layout

```text
Segment-Img/
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── outputs/                 # Generated results
└── src/
    └── segment_img/
        ├── __init__.py
        ├── cli.py
        ├── inference.py
        ├── results.py
        ├── visualization.py
        ├── web.py
        └── web_assets/
            ├── static/
            │   ├── app.js
            │   └── styles.css
            └── templates/
                └── index.html
    └── segment_img.egg-info/  # Generated after pip install -e .
```


## Installation

Create and activate a virtual environment first:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the package:

```powershell
pip install -e .
```

If you want GPU acceleration, install the matching PyTorch build for your machine from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -e .`.

On the first run with the default pretrained model, `torchvision` may download Mask R-CNN weights if they are not already cached locally.

## Quick Start

Run segmentation on a single image:

```powershell
segment-img --input .\images\street.jpg --output .\outputs
```

Run on a folder of images:

```powershell
segment-img --input .\images --output .\outputs --score-threshold 0.65
```

Use a custom fine-tuned checkpoint:

```powershell
segment-img --input .\medical-scans --output .\outputs --checkpoint .\checkpoints\maskrcnn.pt --labels-file .\labels.txt
```

## Browser UI

Launch the local web app:

```powershell
segment-img-ui
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

Use a different host, port, or output folder if needed:

```powershell
segment-img-ui --host 0.0.0.0 --port 8080 --output-root .\outputs\web_runs
```

The UI supports:

- Uploading a single image and previewing the segmented overlay.
- Comparing the raw image against the overlay with a split-view slider.
- Reviewing per-instance masks with confidence, coverage, and spotlighted detail stats.
- Filtering and sorting detections by label, confidence, and coverage.
- Switching between CPU, CUDA, or automatic device selection.
- Pointing to a custom checkpoint and optional label list for domain-specific models.

## Labels File Format

When using a custom checkpoint, provide a text file with one class name per line:

```text
tumor
lesion
organ
```

The CLI automatically inserts a background class if it is missing.

## Output Structure

Each processed image gets its own folder:

```text
outputs/
  street/
    overlay.png
    summary.json
    masks/
      001_person.png
      002_car.png
```

`overlay.png` shows the segmented objects, `masks/` stores per-instance binary masks, and `summary.json` captures scores, boxes, labels, and saved mask paths.

## Notes For Real-World Domains

- Medical imaging usually needs a custom dataset and a fine-tuned checkpoint because COCO-pretrained weights are built for everyday object categories.
- Autonomous driving datasets often benefit from custom labels such as lane markers, vehicles, pedestrians, road signs, or drivable area.
- This starter focuses on inference so you can plug it into a larger labeling, evaluation, or deployment pipeline.

## License

This project is licensed under the [MIT License](LICENSE).

