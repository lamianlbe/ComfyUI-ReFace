"""Sapiens body-part segmentation for head mask extraction.

Uses Meta's Sapiens 0.3B model (native bf16, torch.export) for
high-resolution (1024×768) human body-part segmentation.
From: https://github.com/facebookresearch/sapiens

Goliath 28-class label map:
  0: Background, 1: Apparel, 2: Face_Neck, 3: Hair,
  4: Left_Foot, 5: Left_Hand, 6: Left_Lower_Arm, 7: Left_Lower_Leg,
  8: Left_Shoe, 9: Left_Sock, 10: Left_Upper_Arm, 11: Left_Upper_Leg,
  12: Lower_Clothing, 13: Right_Foot, 14: Right_Hand, 15: Right_Lower_Arm,
  16: Right_Lower_Leg, 17: Right_Shoe, 18: Right_Sock, 19: Right_Upper_Arm,
  20: Right_Upper_Leg, 21: Torso, 22: Upper_Clothing,
  23: Lower_Lip, 24: Upper_Lip, 25: Lower_Teeth, 26: Upper_Teeth, 27: Tongue
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

# Head labels: Face_Neck(2) + Hair(3) + Lower_Lip(23) + Upper_Lip(24) +
#              Lower_Teeth(25) + Upper_Teeth(26) + Tongue(27)
HEAD_LABELS = {2, 3, 23, 24, 25, 26, 27}

# Sapiens input: height=1024, width=768
SAPIENS_INPUT_H = 1024
SAPIENS_INPUT_W = 768

# Normalization (0-255 scale, NOT 0-1)
SAPIENS_MEAN = np.array([123.5, 116.5, 103.5], dtype=np.float32).reshape(3, 1, 1)
SAPIENS_STD = np.array([58.5, 57.0, 57.5], dtype=np.float32).reshape(3, 1, 1)

_parser_model = None
_device = None

SAPIENS_MODEL_DIR = "reface"
SAPIENS_MODEL_NAME = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_bfloat16.pt2"
SAPIENS_MODEL_URL = (
    "https://huggingface.co/facebook/sapiens-seg-0.3b-bfloat16/resolve/main/"
    "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_bfloat16.pt2"
)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ──────────────────────────────────────────────────────────────────────────────


def _get_device():
    global _device
    if _device is None:
        import comfy.model_management
        _device = comfy.model_management.get_torch_device()
    return _device


def _supports_bf16(device):
    """Check if device supports bfloat16."""
    if device.type == 'cuda':
        return torch.cuda.is_bf16_supported()
    return False


def _download_sapiens():
    """Download Sapiens model if not present."""
    model_dir = os.path.join(folder_paths.models_dir, SAPIENS_MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, SAPIENS_MODEL_NAME)

    if os.path.isfile(model_path):
        return model_path

    print(f"[ReFace] Downloading Sapiens 0.3B bf16 segmentation model to {model_path} ...")
    print(f"[ReFace] URL: {SAPIENS_MODEL_URL}")
    print("[ReFace] This is ~650MB, please wait...")

    import urllib.request
    urllib.request.urlretrieve(SAPIENS_MODEL_URL, model_path)
    print("[ReFace] Sapiens model downloaded.")
    return model_path


def get_face_parser():
    """Get or create singleton Sapiens segmentation model."""
    global _parser_model
    if _parser_model is not None:
        return _parser_model

    model_path = _download_sapiens()
    device = _get_device()

    # Load native bf16 torch.export model
    model = torch.export.load(model_path).module()
    model = model.to(device)
    model.eval()

    dtype_str = "bfloat16"

    # Apply torch.compile for fused ops and faster inference
    # First run will be slow (compilation), subsequent runs are fast
    try:
        model = torch.compile(model, mode="max-autotune")
        print(f"[ReFace] Sapiens 0.3B loaded on {device} ({dtype_str}, compiled)")
    except Exception as e:
        print(f"[ReFace] Sapiens 0.3B loaded on {device} ({dtype_str}, torch.compile failed: {e})")

    _parser_model = model
    return _parser_model


def _preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    """Preprocess RGB image for Sapiens.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.

    Returns:
        (1, 3, 1024, 768) float tensor, normalized.
    """
    # Resize to model input size
    img = cv2.resize(image_rgb, (SAPIENS_INPUT_W, SAPIENS_INPUT_H),
                     interpolation=cv2.INTER_LINEAR)

    # HWC → CHW, uint8 → float32
    img = img.transpose(2, 0, 1).astype(np.float32)

    # Normalize (values in 0-255 range)
    img = (img - SAPIENS_MEAN) / SAPIENS_STD

    return torch.from_numpy(img).unsqueeze(0)


def parse_head_mask(
    image_rgb: np.ndarray,
    instance_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Parse image and return binary head mask using Sapiens.

    Args:
        image_rgb: RGB image as numpy array (H, W, 3), uint8.
        instance_mask: Optional binary mask (H, W), bool or uint8.
            If provided, the head parsing result is intersected with this mask
            so that only the head of the specified person instance is returned.

    Returns:
        Binary mask as numpy array (H, W), uint8, 0 or 255.
    """
    model = get_face_parser()
    device = _get_device()
    h, w = image_rgb.shape[:2]

    # Preprocess and cast to bf16 (model is native bf16)
    tensor = _preprocess(image_rgb).to(device=device, dtype=torch.bfloat16)

    # Run inference
    with torch.no_grad():
        output = model(tensor)

    # Model returns a list; take first element
    result = output[0] if isinstance(output, (list, tuple)) else output

    # Ensure 4D: (1, C, H, W)
    if result.dim() == 3:
        result = result.unsqueeze(0)

    # Upsample to original image size
    result = F.interpolate(
        result.float(), size=(h, w), mode='bilinear', align_corners=False,
    ).squeeze(0)  # (C, H, W)

    # Argmax to get class IDs
    seg_map = result.argmax(dim=0).cpu().numpy()  # (H, W)

    # Build head mask from head labels
    head_mask = np.zeros((h, w), dtype=np.uint8)
    for label in HEAD_LABELS:
        head_mask[seg_map == label] = 255

    # Intersect with instance mask if provided
    if instance_mask is not None:
        inst = instance_mask.astype(np.uint8)
        if inst.max() == 1:
            inst = inst * 255
        head_mask[inst == 0] = 0

    return head_mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside the mask using contour-based approach."""
    # Morphological closing to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Fill holes: find contours and fill
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

    return filled
