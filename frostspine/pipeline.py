"""Intersect two offset crop segmentation passes for one image."""
from pathlib import Path

import numpy as np
import torch
import warnings
from PIL import Image

from .geometry import clean_mask, centerline_from_mask, crop_bounds, generate_boxes
from .model import FROSTSpine


def build_model(weights: Path, device: str, image_size: int) -> FROSTSpine:
    from dinov3.models.vision_transformer import DinoVisionTransformer

    # Same architecture as dinov3_vitl16 (LVD1689M) in the original project.
    encoder = DinoVisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        ffn_ratio=4, pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate", pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32", layerscale_init=1e-5,
        norm_layer="layernormbf16", ffn_layer="mlp", n_storage_tokens=4,
        mask_k_bias=True,
    )
    state = torch.load(weights, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    del state
    encoder.eval().requires_grad_(False)
    return FROSTSpine(
        encoder, image_size=image_size, svd_components=500, tau=0.6,
        merge_threshold=0.2, resize_to_orig_size=True, device=device,
    ).to(device).eval().requires_grad_(False)


@torch.inference_mode()
def predict(model, image, reference: Image.Image, reference_mask: Image.Image):
    # FROSTSpine clears its reference and target after each segment() call.
    model.set_reference(reference, reference_mask)
    model.set_target(image)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*", category=UserWarning)
        return model.segment().cpu().numpy().astype(bool)


def load_reference(image_path: Path, mask_path: Path):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    with Image.open(mask_path) as source:
        mask = source.convert("L")
    if image.size != mask.size:
        raise ValueError(f"Reference image and mask sizes differ: {image_path}, {mask_path}")
    if not np.asarray(mask).any():
        raise ValueError(f"Reference mask is empty: {mask_path}")
    return image, mask


def segment_boxes(model, image, boxes, reference):
    """Union predictions within one pass in original image coordinates."""
    full_mask = np.zeros((image.height, image.width), dtype=bool)
    for box in boxes:
        left, top, right, bottom = crop_bounds(box, image.width, image.height)
        if left >= right or top >= bottom:
            raise RuntimeError("Generated an empty crop.")
        mask = predict(model, image.crop((left, top, right, bottom)), *reference)
        if mask.shape != (bottom - top, right - left):
            raise RuntimeError("Crop mask dimensions do not match its image.")
        full_mask[top:bottom, left:right] |= mask
    return full_mask


def segment_image(args):
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    coarse_ref = load_reference(args.coarse_reference, args.coarse_reference_mask)
    crop_ref = load_reference(args.crop_reference, args.crop_reference_mask)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*", category=UserWarning)
        model = build_model(args.weights, device, args.image_size)
    coarse_mask = clean_mask(predict(model, image, *coarse_ref), 500, 5, 0)
    points = centerline_from_mask(coarse_mask, 8, 7, 3)
    height = max(image.height * args.box_height_ratio, 20.0)
    width = max(image.width * args.box_width_ratio, 20.0)
    boxes = generate_boxes(points, height, width, height)
    offset_boxes = generate_boxes(points, height, width, height, offset=height / 2.0)
    if not boxes or not offset_boxes:
        raise RuntimeError("The coarse mask is empty or too short to generate spine crops.")

    mask_pass1 = segment_boxes(model, image, boxes, crop_ref)
    mask_pass2 = segment_boxes(model, image, offset_boxes, crop_ref)
    full_mask = mask_pass1 & mask_pass2
    if not full_mask.any():
        raise RuntimeError("The intersection is empty; check the references or image size.")

    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(coarse_mask.astype(np.uint8) * 255).save(args.output / "coarse_mask.png")
    Image.fromarray(full_mask.astype(np.uint8) * 255).save(args.output / "full_mask.png")
    overlay = np.asarray(image).astype(np.float32)
    overlay[full_mask] = 0.55 * overlay[full_mask] + 0.45 * np.array([0, 255, 0])
    Image.fromarray(overlay.astype(np.uint8)).save(args.output / "full_mask_overlay.png")
    print((args.output / "full_mask.png").resolve(), flush=True)
