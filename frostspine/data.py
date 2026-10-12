"""Image and binary mask preprocessing for FROSTSpine."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def build_transform(image_size: int) -> transforms.Compose:
    """Build the standard image transform for FROSTSpine inference.

    Args:
        image_size: target spatial resolution.

    Returns:
        A torchvision Compose transform.
    """
    return transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def load_image(
    image: str | Image.Image | torch.Tensor,
    transform: transforms.Compose,
    device: torch.device | str,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an image as a batched tensor with shape (1, C, H, W)."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        orig_size = (image.height, image.width)
        image_tensor = transform(image).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        image_tensor = image
        if image_tensor.ndim not in (3, 4):
            raise ValueError("Tensor image must have shape (C,H,W) or (1,C,H,W)")
        image_tensor = image_tensor.reshape(-1, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1])
        if image_tensor.shape[0] != 1:
            raise ValueError("Tensor image must have shape (C,H,W) or (1,C,H,W)")
        orig_size = (image_tensor.shape[-2], image_tensor.shape[-1])
        image_tensor = F.interpolate(
            image_tensor,
            size=transform.transforms[0].size,
            mode="bilinear",
            align_corners=False,
        )
    else:
        orig_size = (image.height, image.width)
        image_tensor = transform(image).unsqueeze(0)

    return image_tensor.to(device), orig_size

def load_mask(
    mask: str | Image.Image | torch.Tensor,
    image_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Load a binary mask as a tensor with shape (1, H, W)."""
    if isinstance(mask, torch.Tensor):
        # Accept (H,W), (1,H,W), (H,W,C), (C,H,W), (1,H,W,C), (1,C,H,W)
        if mask.ndim == 2:
            mask_tensor = mask.unsqueeze(0)  # (1,H,W)
        elif mask.ndim == 3:
            # Heuristic: if first dim is small (1/3/4), treat as CHW else HWC
            if mask.shape[0] in (1, 3, 4):
                chw = mask
                hw = chw.any(dim=0)
            else:
                hw = mask.any(dim=-1)
            mask_tensor = hw.unsqueeze(0)
        elif mask.ndim == 4:
            if mask.shape[0] != 1:
                raise ValueError("Tensor mask with 4 dims must have batch=1")
            m = mask[0]
            if m.ndim != 3:
                raise ValueError("Unexpected tensor mask shape")
            if m.shape[0] in (1, 3, 4):
                hw = m.any(dim=0)
            else:
                hw = m.any(dim=-1)
            mask_tensor = hw.unsqueeze(0)
        else:
            raise ValueError("Tensor mask must have 2-4 dimensions")

        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
            raise ValueError("Tensor mask must be convertible to shape (1,H,W)")
        mask_tensor = mask_tensor.to(device)
        if mask_tensor.dtype != torch.bool:
            mask_tensor = mask_tensor > 0
    else:
        if isinstance(mask, str):
            mask = Image.open(mask)

        # Ensure single-channel for PIL masks (avoid HxWx3)
        if isinstance(mask, Image.Image) and mask.mode not in ("1", "L", "I", "F"):
            mask = mask.convert("L")

        arr = np.asarray(mask)
        if arr.ndim == 3:
            # If still multi-channel for any reason, reduce to 2D.
            arr = (arr > 0).any(axis=-1)
        else:
            arr = arr > 0

        mask_tensor = torch.tensor(arr, dtype=torch.bool).unsqueeze(0).to(device)

    return F.interpolate(
        mask_tensor.unsqueeze(0).float(),
        size=(image_size, image_size),
        mode="nearest",
    ).squeeze(0) > 0.5

def downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Downsample a (1, 1, H, W) binary mask to feature resolution (h, w)."""
    down = F.interpolate(mask.float(), size=(h, w), mode='bilinear', align_corners=False)[0, 0] > 0.5
    if down.sum() == 0:
        down = F.interpolate(mask.float(), size=(h, w), mode='nearest')[0, 0] > 0.5
        if down.sum() == 0:
            center = torch.argwhere(mask[0, 0] > 0).float().mean(dim=0)
            scale = mask.shape[-1] // w
            cy, cx = (center / scale).int()
            down[cy, cx] = True
    return down
