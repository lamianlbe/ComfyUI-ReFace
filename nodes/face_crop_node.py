"""ReFace Face Crop Node for ComfyUI.

Pipeline:
  1. YOLO26x-seg human instance segmentation
  2. InsightFace largest face detection
  3. Find the human instance containing the face center
  4. BiSeNet head parsing constrained to that instance
  5. Fill holes in the head mask
  6. Compute bbox with configurable expansion
  7. Size threshold check
"""

import numpy as np
import torch


class ReFaceCrop:
    """Detect the largest face, segment the head within its person instance, and crop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_expand_pixels": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Number of pixels to expand the bounding box outward on each side.",
                }),
                "min_bbox_ratio": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Minimum bbox area as a fraction of the total image area. "
                               "Below this the detection is considered failed.",
                }),
                "background_mode": (["transparent", "color"],),
                "background_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Hex color for background when mode is 'color' (e.g. #000000).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "REFACE_EMBEDDING")
    RETURN_NAMES = ("face", "detected", "embedding")
    FUNCTION = "execute"
    CATEGORY = "ReFace"

    def execute(self, image, bbox_expand_pixels, min_bbox_ratio, background_mode, background_color):
        # image: [B, H, W, C] float32 0-1, process first frame
        img_tensor = image[0]  # [H, W, C]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)  # RGB uint8
        H, W = img_np.shape[:2]

        # ── Step 1: Human instance segmentation (YOLO26x-seg) ──
        from ..core.human_segmentor import segment_humans

        humans = segment_humans(img_np)

        # ── Step 2: Face detection (InsightFace) ──
        from ..core.face_detector import detect_largest_face

        img_bgr = img_np[:, :, ::-1].copy()
        face_info = detect_largest_face(img_bgr)

        if face_info is None:
            return self._empty_result()

        cx, cy = face_info["center"]
        self._last_embedding = face_info.get("embedding", None)

        # ── Step 3: Find human instance containing the face center ──
        from ..core.human_segmentor import find_human_containing_point

        instance_mask = None
        if humans:
            human = find_human_containing_point(humans, cx, cy)
            if human is not None:
                instance_mask = human["mask"]  # (H, W) bool

        # ── Step 4: Head parsing constrained to instance ──
        from ..core.face_parser import parse_head_mask

        head_mask = parse_head_mask(img_np, instance_mask=instance_mask)  # (H, W) uint8

        # ── Step 5: Fill holes (done inside parse_head_mask via _fill_holes) ──
        from ..core.face_parser import _fill_holes
        head_mask = _fill_holes(head_mask)

        # ── Step 6: Compute bbox with expansion ──
        ys, xs = np.where(head_mask > 0)
        if len(ys) == 0:
            return self._empty_result()

        mask_x1, mask_y1 = int(xs.min()), int(ys.min())
        mask_x2, mask_y2 = int(xs.max()), int(ys.max())

        pad = bbox_expand_pixels
        left = max(0, mask_x1 - pad)
        top = max(0, mask_y1 - pad)
        right = min(W, mask_x2 + 1 + pad)
        bottom = min(H, mask_y2 + 1 + pad)
        width = right - left
        height = bottom - top

        # ── Step 7: Size threshold ──
        bbox_area = width * height
        image_area = W * H
        if bbox_area / image_area < min_bbox_ratio:
            return self._empty_result()

        # ── Crop and apply mask ──
        crop_rgb = img_np[top:bottom, left:right].copy()
        crop_mask = head_mask[top:bottom, left:right]
        mask_bool = crop_mask > 0

        if background_mode == "transparent":
            crop_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            crop_rgba[:, :, :3] = crop_rgb
            crop_rgba[:, :, 3] = crop_mask  # 0 or 255
            out_np = crop_rgba.astype(np.float32) / 255.0
        else:
            bg = self._parse_hex_color(background_color)
            crop_rgb[~mask_bool] = bg
            out_np = crop_rgb.astype(np.float32) / 255.0

        out_tensor = torch.from_numpy(out_np).unsqueeze(0)  # [1, H, W, C]
        embedding = getattr(self, "_last_embedding", None)

        return (out_tensor, True, embedding)

    @staticmethod
    def _empty_result():
        empty = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
        return (empty, False, None)

    @staticmethod
    def _parse_hex_color(hex_str: str):
        hex_str = hex_str.lstrip("#")
        if len(hex_str) != 6:
            return (0, 0, 0)
        try:
            return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
        except ValueError:
            return (0, 0, 0)


class ReFacePackDetection:
    """Pack all ReFace Crop Head outputs into a single REFACE_DETECTION struct.

    Outputs None if detected is False.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face": ("IMAGE",),
                "detected": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {
                "embedding": ("REFACE_EMBEDDING",),
            },
        }

    RETURN_TYPES = ("REFACE_DETECTION",)
    RETURN_NAMES = ("detection",)
    FUNCTION = "execute"
    CATEGORY = "ReFace"

    def execute(self, face, detected, embedding=None):
        if not detected:
            return (None,)

        detection = {
            "face": face,
            "detected": True,
            "embedding": embedding,
        }
        return (detection,)


NODE_CLASS_MAPPINGS = {
    "ReFaceCrop": ReFaceCrop,
    "ReFacePackDetection": ReFacePackDetection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReFaceCrop": "ReFace Crop Head",
    "ReFacePackDetection": "Pack ReFace Detection",
}
