"""Command-line entry point for the bundled single-image example."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="FROSTSpine single-image spine segmentation")
    parser.add_argument("--image", type=Path, default=ROOT / "examples/image.png")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image-size", type=int, default=512, help="Square encoder resolution; original project uses 1024")
    parser.add_argument("--box-height-ratio", type=float, default=0.035)
    parser.add_argument("--box-width-ratio", type=float, default=0.17)
    parser.add_argument("--coarse-reference", type=Path, default=ROOT / "examples/coarse_reference.png")
    parser.add_argument("--coarse-reference-mask", type=Path, default=ROOT / "examples/coarse_reference_mask.png")
    parser.add_argument("--crop-reference", type=Path, default=ROOT / "examples/crop_reference.png")
    parser.add_argument("--crop-reference-mask", type=Path, default=ROOT / "examples/crop_reference_mask.png")
    args = parser.parse_args()
    if args.image_size < 32 or args.image_size % 16:
        parser.error("--image-size must be a multiple of 16 and at least 32")
    if not 0 < args.box_height_ratio <= 1 or not 0 < args.box_width_ratio <= 1:
        parser.error("box size ratios must be in (0, 1]")
    for name in ("image", "weights", "coarse_reference", "coarse_reference_mask", "crop_reference", "crop_reference_mask"):
        if not getattr(args, name).is_file():
            parser.error(f"Missing {name.replace('_', ' ')}: {getattr(args, name)}")
    return args


def main():
    args = parse_args()
    from frostspine.pipeline import segment_image
    segment_image(args)


if __name__ == "__main__":
    main()
