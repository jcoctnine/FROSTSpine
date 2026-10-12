# FROSTSpine

Reference-guided spine segmentation for a single image using a frozen DINOv3 ViT-L/16 backbone, without training.

## Installation

Use Python 3.10 or newer. An NVIDIA GPU is recommended; CPU inference is supported but slower.

```bash
cd FROSTSpine
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU inference, first install PyTorch and torchvision builds compatible with your CUDA environment. The bundled `dinov3/` directory contains the minimal ViT inference sources. No Python modules from the parent workspace are required.

## Model weights

Obtain `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`, the official ViT-L/16 LVD1689M checkpoint, following the instructions in the [DINOv3 repository](https://github.com/facebookresearch/dinov3).

Weights are approximately 1.2 GB and excluded from Git. Place them in `weights/` or another local directory and pass their location with `--weights`. The current default in `run.py` points to the original workspace checkpoint; use an explicit path for a standalone checkout.

## Run one image

From this project directory, run the bundled example:

```bash
python run.py --weights path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

The default input is `examples/image.png`. To use another image:

```bash
python run.py --image path/to/image.png --weights path/to/model.pth --output outputs/my_image --device cuda
```

The default encoder resolution is 512 to reduce GPU memory usage. Use `--image-size 1024` for the original resolution, which requires more memory and may change the predictions.

The entry point accepts a single image and contains no batch processing or GUI. Outputs are saved to this project's `outputs/` directory by default. Repeated runs overwrite the same filenames:

- `coarse_mask.png`: the coarse mask used for localization and crop generation.
- `full_mask.png`: the intersection of both crop passes, at the original image resolution, with background 0 and foreground 255.
- `full_mask_overlay.png`: the final segmentation overlaid in green on the input image.

## Reference images

`examples/` contains one target image and two required reference image/mask pairs:

- `coarse_reference.png` and `coarse_reference_mask.png` for full-image localization.
- `crop_reference.png` and `crop_reference_mask.png` for crop segmentation.

Nonzero mask pixels indicate foreground. Each reference mask must match its reference image dimensions. Replace the defaults with `--coarse-reference`, `--coarse-reference-mask`, `--crop-reference`, and `--crop-reference-mask`.

The example demonstrates inference and includes no ground-truth annotations or accuracy evaluation.

## Validation

The two-pass example completed at resolution 512 with Python 3.11, PyTorch 2.3.1+cu121, and an RTX 4060 Laptop GPU with 8 GB of memory. The output dimensions were 950 × 1656 with pixel values 0/255. The final mask was verified to equal the exact intersection of the two full-image predictions. This verifies execution, not segmentation accuracy.

## Project structure

```text
FROSTSpine/
  run.py                 # Single-image entry point
  frostspine/            # Segmentation, preprocessing, clustering, and geometry
  dinov3/                # Minimal ViT inference dependencies
  examples/              # Target image and reference pairs
  weights/README.md      # Checkpoint placement instructions
  requirements.txt
  LICENSE
  DINOV3_LICENSE
  NOTICE.md
```

This folder can serve as the root of a standalone GitHub repository. `.gitignore` excludes checkpoints, outputs, and caches. See [NOTICE.md](NOTICE.md) for attribution, modifications, and licensing.

The example images were copied from the original workspace, which does not record their redistribution permission. Confirm permission before publishing them, or replace them with publicly distributable examples.
