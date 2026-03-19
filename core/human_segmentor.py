"""Human instance segmentation using YOLO26x-seg.

Uses the extra-large YOLO26 segmentation model to detect person instances
and produce per-instance binary masks.
"""

import os
import numpy as np
import torch

import folder_paths

_model = None
_device_str = None

YOLO_MODEL_NAME = "yolo26x-seg.pt"


def _get_model_dir():
    return os.path.join(folder_paths.models_dir, "reface")


def _get_device_str():
    """Determine the best device string for ultralytics: 'cuda', 'mps', or 'cpu'."""
    global _device_str
    if _device_str is not None:
        return _device_str

    if torch.cuda.is_available():
        _device_str = "cuda"
    elif torch.backends.mps.is_available():
        _device_str = "mps"
    else:
        _device_str = "cpu"

    print(f"[ReFace] YOLO using device: {_device_str}")
    return _device_str


def get_yolo_model():
    """Get or create singleton YOLO26x-seg model."""
    global _model
    if _model is not None:
        return _model

    from ultralytics import YOLO

    model_dir = _get_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, YOLO_MODEL_NAME)

    if not os.path.isfile(model_path):
        # ultralytics auto-downloads to current dir; we load with the name
        # and it will download automatically, then we can move it
        print(f"[ReFace] YOLO26x-seg model not found at {model_path}")
        print(f"[ReFace] Downloading {YOLO_MODEL_NAME} ...")
        _model = YOLO(YOLO_MODEL_NAME)
        # Save to our model directory for future use
        import shutil
        downloaded_path = YOLO_MODEL_NAME
        if os.path.isfile(downloaded_path):
            shutil.move(downloaded_path, model_path)
    else:
        _model = YOLO(model_path)

    return _model


def segment_humans(image_rgb: np.ndarray):
    """Detect all person instances and return their masks.

    Args:
        image_rgb: RGB image as numpy array (H, W, 3), uint8.

    Returns:
        List of dicts, each with:
          - 'mask': binary mask (H, W), bool
          - 'bbox': (x1, y1, x2, y2) floats
          - 'confidence': detection confidence
        Sorted by mask area descending. Empty list if no person detected.
    """
    model = get_yolo_model()
    device = _get_device_str()
    H, W = image_rgb.shape[:2]

    # YOLO expects RGB, which is what we provide
    results = model(image_rgb, device=device, verbose=False)
    result = results[0]

    if result.masks is None or result.boxes is None:
        return []

    humans = []
    for i, cls_id in enumerate(result.boxes.cls):
        # COCO class 0 = person
        if int(cls_id) != 0:
            continue

        mask = result.masks.data[i].cpu().numpy()  # (mask_h, mask_w)
        # Resize mask to original image size if needed
        if mask.shape[0] != H or mask.shape[1] != W:
            import cv2
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        mask_bool = mask > 0.5
        bbox = result.boxes.xyxy[i].cpu().numpy()  # (x1, y1, x2, y2)
        conf = float(result.boxes.conf[i])

        humans.append({
            "mask": mask_bool,
            "bbox": (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            "confidence": conf,
        })

    # Sort by mask area descending
    humans.sort(key=lambda h: h["mask"].sum(), reverse=True)
    return humans


def find_human_containing_point(humans, cx, cy):
    """Find the human instance whose mask contains the given point.

    Args:
        humans: List of human dicts from segment_humans().
        cx, cy: Point coordinates (float).

    Returns:
        The human dict whose mask contains (cx, cy), or None.
        If multiple humans contain the point, returns the one with the
        largest mask area.
    """
    ix, iy = int(round(cx)), int(round(cy))

    candidates = []
    for human in humans:
        mask = human["mask"]
        h, w = mask.shape
        if 0 <= iy < h and 0 <= ix < w and mask[iy, ix]:
            candidates.append(human)

    if not candidates:
        return None

    # Already sorted by area descending, return the largest
    return candidates[0]
