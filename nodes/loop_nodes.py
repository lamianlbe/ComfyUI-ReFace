"""ReFace Loop Nodes for ComfyUI.

ReFaceLoopStart  – preprocesses src/dst faces, matches them, then iterates.
ReFaceLoopEnd    – pastes modified head back, advances the loop index.

Loop mechanism follows the ComfyUI execution-inversion pattern:
  * LoopStart acts as the "open" node (outputs FLOW_CONTROL).
  * LoopEnd acts as the "close" node (graph expansion for next iteration).
  * Loop-carried variable (_loop_ctx) is passed through a hidden input
    that LoopEnd sets when expanding the graph. dst_image is stored
    inside loop_ctx so no separate wire is needed.
"""

import numpy as np
import torch

# ComfyUI execution-inversion imports
from comfy_execution.graph_utils import GraphBuilder

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    class ExecutionBlocker:
        def __init__(self, v):
            self.v = v

try:
    from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
except ImportError:
    ALL_NODE_CLASS_MAPPINGS = {}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_link(v):
    """Check whether *v* is a ComfyUI link reference ``[node_id, output_index]``."""
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


# Sentinel for "any type" – mirrors the pattern used by ComfyUI-Easy-Use.
class _AnyType(str):
    """A string subclass that equals any other string (wildcard type)."""
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __hash__(self):
        return hash("*")

any_type = _AnyType("*")


# ──────────────────────────────────────────────────────────────────────────────
# ReFaceLoopStart
# ──────────────────────────────────────────────────────────────────────────────

MAX_SRC_FACES = 4


class ReFaceLoopStart:
    """Preprocess faces, match src↔dst, then loop over each matched pair.

    **First iteration** (_loop_ctx is None):
      1. Filter None src_face inputs.
      2. YOLO26x-seg → human instance masks on *dst_image*.
      3. InsightFace → detect all faces in *dst_image*, keep top-N by area.
      4. ArcFace matching (skipped when 1 src + 1 person).
      5. Run LIP + Pascal on the full image once (union).
      6. Per match: intersect head mask with instance mask → bbox → threshold.
      7. Build items array and start iteration.

    **Subsequent iterations** (_loop_ctx is provided by LoopEnd):
      Skip all heavy computation; just output the current item's data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "dst_image": ("IMAGE",),
                "bbox_expand_pixels": ("INT", {
                    "default": 50, "min": 0, "max": 500, "step": 1,
                    "tooltip": "Pixels to expand the head bounding box on each side.",
                }),
                "min_bbox_ratio": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Minimum bbox-area / image-area ratio. Smaller → discard.",
                }),
            },
            "optional": {},
            "hidden": {
                # Loop-carried variable (set by ReFaceLoopEnd on iterations > 0)
                "_loop_ctx": (any_type,),
                "unique_id": "UNIQUE_ID",
            },
        }
        for i in range(MAX_SRC_FACES):
            inputs["optional"][f"src_face_{i}"] = ("REFACE_DETECTION",)
        return inputs

    RETURN_TYPES = ("FLOW_CONTROL", "REFACE_LOOP_CTX",
                    "IMAGE", "MASK",
                    "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("flow", "loop_ctx",
                    "dst_head_image", "dst_head_mask",
                    "src_face_image", "has_data")
    FUNCTION = "execute"
    CATEGORY = "ReFace"

    # ── public entry point ────────────────────────────────────────────────

    def execute(self, dst_image, bbox_expand_pixels, min_bbox_ratio, **kwargs):
        loop_ctx = kwargs.get("_loop_ctx", None)

        if loop_ctx is None:
            # ── First iteration: heavy preprocessing ──
            src_faces = self._collect_src_faces(kwargs)
            if not src_faces:
                ctx = {"items": [], "current_index": 0, "dst_image": dst_image}
            else:
                ctx = self._preprocess(dst_image, src_faces,
                                       bbox_expand_pixels, min_bbox_ratio)
        else:
            # ── Subsequent iteration: reuse context ──
            ctx = loop_ctx

        current_dst = ctx["dst_image"]
        idx = ctx["current_index"]
        items = ctx["items"]

        if idx >= len(items):
            # Nothing (left) to process → block processing outputs
            return self._empty_outputs(ctx)

        item = items[idx]
        bbox = item["dst_head_bbox"]
        left, top = bbox["left"], bbox["top"]
        right, bottom = bbox["right"], bbox["bottom"]

        # Crop head image from the *running* dst_image
        img_hw = current_dst[0]  # (H, W, C)
        dst_head_image = img_hw[top:bottom, left:right, :].unsqueeze(0).clone()

        # Crop head mask
        full_mask = item["dst_head_mask"]  # (H, W) uint8 0/255
        crop_mask = full_mask[top:bottom, left:right]
        mask_tensor = torch.from_numpy(
            crop_mask.astype(np.float32) / 255.0
        ).unsqueeze(0)  # (1, h, w)

        # Src face image
        src_face_image = item["src_face"]["face"]  # IMAGE tensor

        return ("stub", ctx,
                dst_head_image, mask_tensor,
                src_face_image, True)

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _collect_src_faces(kwargs):
        """Filter non-None REFACE_DETECTION inputs."""
        faces = []
        for i in range(MAX_SRC_FACES):
            sf = kwargs.get(f"src_face_{i}", None)
            if sf is not None and sf.get("detected", False):
                faces.append(sf)
        return faces

    @staticmethod
    def _empty_outputs(ctx):
        """Return ExecutionBlocker for processing outputs so that nodes
        between LoopStart and LoopEnd are skipped entirely.
        Non-processing outputs (flow, loop_ctx) remain valid
        so that LoopEnd can still execute and return dst_image as-is."""
        blocker = ExecutionBlocker(None)
        return ("stub", ctx,
                blocker, blocker,          # dst_head_image, dst_head_mask
                blocker, False)            # src_face_image, has_data

    # ── heavy preprocessing (runs once) ───────────────────────────────────

    def _preprocess(self, dst_image, src_faces, bbox_expand_pixels, min_bbox_ratio):
        img_tensor = dst_image[0]  # (H, W, C)
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        H, W = img_np.shape[:2]
        img_bgr = img_np[:, :, ::-1].copy()

        # ── Step 2: YOLO human instance segmentation ──
        from ..core.human_segmentor import segment_humans, find_human_containing_point
        humans = segment_humans(img_np)

        # ── Step 3: InsightFace detect all dst faces, sorted by area desc ──
        from ..core.face_detector import detect_all_faces, compute_embedding_from_image
        dst_faces = detect_all_faces(img_bgr, max_count=len(src_faces))
        if not dst_faces:
            return {"items": [], "current_index": 0, "dst_image": dst_image}

        # ── Step 4: Matching ──
        only_one_src = len(src_faces) == 1
        only_one_human = len(humans) <= 1
        skip_arcface = only_one_src and only_one_human

        if skip_arcface:
            matches = [(0, 0)]
        else:
            src_embeddings = []
            for sf in src_faces:
                emb = sf.get("embedding", None)
                if emb is None:
                    face_np = self._detection_to_bgr(sf)
                    emb = compute_embedding_from_image(face_np) if face_np is not None else None
                src_embeddings.append(emb)
            matches = self._match_faces(dst_faces, src_embeddings)

        if not matches:
            return {"items": [], "current_index": 0, "dst_image": dst_image}

        # ── Step 5: Run LIP + Pascal on the FULL image once ──
        from ..core.face_parser import parse_head_mask

        full_head_mask = parse_head_mask(img_np, instance_mask=None)

        # ── Step 6: Per-match: intersect with instance mask → bbox ──
        items = []
        for dst_idx, src_idx in matches:
            dst_face = dst_faces[dst_idx]
            src_face = src_faces[src_idx]
            cx, cy = dst_face["center"]

            # Find human instance containing this face
            if humans:
                human = find_human_containing_point(humans, cx, cy)
                if human is None:
                    continue
                instance_mask = human["mask"]
            else:
                continue

            # Intersect full head mask with this person's instance mask
            inst = instance_mask.astype(np.uint8)
            if inst.max() == 1:
                inst = inst * 255
            head_mask = full_head_mask.copy()
            head_mask[inst == 0] = 0

            # Compute bbox
            ys, xs = np.where(head_mask > 0)
            if len(ys) == 0:
                continue

            mask_x1, mask_y1 = int(xs.min()), int(ys.min())
            mask_x2, mask_y2 = int(xs.max()), int(ys.max())

            pad = bbox_expand_pixels
            left = max(0, mask_x1 - pad)
            top_  = max(0, mask_y1 - pad)
            right = min(W, mask_x2 + 1 + pad)
            bottom = min(H, mask_y2 + 1 + pad)
            width = right - left
            height = bottom - top_

            # Size threshold
            if (width * height) / (W * H) < min_bbox_ratio:
                continue

            items.append({
                "dst_head_bbox": {
                    "left": left, "top": top_, "width": width, "height": height,
                    "right": right, "bottom": bottom,
                },
                "dst_head_mask": head_mask,  # full-image-size (H, W) uint8
                "src_face": src_face,
            })

        return {"items": items, "current_index": 0, "dst_image": dst_image}

    # ── face matching ─────────────────────────────────────────────────────

    @staticmethod
    def _match_faces(dst_faces, src_embeddings):
        """Greedy matching: largest dst_face first picks the most similar src_face.

        Each src_face is used at most once.

        Returns:
            List of (dst_index, src_index) tuples.
        """
        matches = []
        used_src = set()

        for dst_idx, dst_face in enumerate(dst_faces):
            dst_emb = dst_face.get("embedding")
            if dst_emb is None:
                continue

            best_src_idx = None
            best_sim = -2.0  # cosine sim range is [-1, 1]

            for src_idx, src_emb in enumerate(src_embeddings):
                if src_idx in used_src or src_emb is None:
                    continue
                sim = float(np.dot(dst_emb, src_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_src_idx = src_idx

            if best_src_idx is not None:
                used_src.add(best_src_idx)
                matches.append((dst_idx, best_src_idx))

        return matches

    @staticmethod
    def _detection_to_bgr(detection):
        """Extract a BGR numpy array from a REFACE_DETECTION dict."""
        face_tensor = detection.get("face")
        if face_tensor is None:
            return None
        face_np = (face_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        # Handle RGBA → RGB
        if face_np.shape[2] == 4:
            face_np = face_np[:, :, :3]
        return face_np[:, :, ::-1].copy()


# ──────────────────────────────────────────────────────────────────────────────
# ReFaceLoopEnd
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Color correction / blending helpers
# ──────────────────────────────────────────────────────────────────────────────


def _lab_color_transfer(src_rgb, ref_rgb, mask):
    """Transfer color statistics from ref to src in Lab space, within the mask region.

    Args:
        src_rgb: Modified head region (H, W, 3) uint8 RGB.
        ref_rgb: Original dst region (H, W, 3) uint8 RGB.
        mask: Binary mask (H, W) uint8, 0 or 255.

    Returns:
        Color-corrected src (H, W, 3) uint8 RGB.
    """
    import cv2

    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mask_bool = mask > 127

    if mask_bool.sum() < 10:
        return src_rgb

    # Compute stats only within the mask
    for ch in range(3):
        src_ch = src_lab[:, :, ch]
        ref_ch = ref_lab[:, :, ch]
        src_mean = src_ch[mask_bool].mean()
        src_std = src_ch[mask_bool].std() + 1e-6
        ref_mean = ref_ch[mask_bool].mean()
        ref_std = ref_ch[mask_bool].std() + 1e-6

        # Normalize src to ref stats
        src_lab[:, :, ch] = (src_ch - src_mean) * (ref_std / src_std) + ref_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)


def _poisson_blend(src_rgb, dst_rgb, mask):
    """Poisson blending (seamless clone) of src onto dst using the mask.

    Args:
        src_rgb: Modified head region (H, W, 3) uint8 RGB.
        dst_rgb: Original dst region (H, W, 3) uint8 RGB.
        mask: Binary mask (H, W) uint8, 0 or 255.

    Returns:
        Blended result (H, W, 3) uint8 RGB.
    """
    import cv2

    # seamlessClone works in BGR
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2BGR)

    # Find center of the mask for seamlessClone
    ys, xs = np.where(mask > 127)
    if len(ys) < 10:
        return src_rgb

    center = (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2))

    # Ensure mask is proper format (must be single channel, 255 for foreground)
    mask_clone = (mask > 127).astype(np.uint8) * 255

    try:
        result_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask_clone, center,
                                        cv2.NORMAL_CLONE)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error:
        # Fallback to direct paste if seamlessClone fails
        return src_rgb


def _wavelet_color_transfer(src_rgb, ref_rgb, mask):
    """Wavelet-based color transfer: use low-frequency color from ref and
    high-frequency detail from src.

    Decomposes both images into low-freq (Gaussian blur) and high-freq
    (detail = original - low_freq) components. Reconstructs by combining
    ref's low-freq color with src's high-freq detail.

    Args:
        src_rgb: Modified head region (H, W, 3) uint8 RGB.
        ref_rgb: Original dst region (H, W, 3) uint8 RGB.
        mask: Binary mask (H, W) uint8, 0 or 255.

    Returns:
        Color-corrected src (H, W, 3) uint8 RGB.
    """
    import cv2

    mask_bool = mask > 127
    if mask_bool.sum() < 10:
        return src_rgb

    src_f = src_rgb.astype(np.float32)
    ref_f = ref_rgb.astype(np.float32)

    # Gaussian blur as low-frequency decomposition
    # Kernel size proportional to image size for scale-invariance
    ksize = max(3, min(src_f.shape[0], src_f.shape[1]) // 8)
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    src_low = cv2.GaussianBlur(src_f, (ksize, ksize), 0)
    ref_low = cv2.GaussianBlur(ref_f, (ksize, ksize), 0)

    # High-frequency detail from src
    src_high = src_f - src_low

    # Reconstruct: ref's low-freq color + src's high-freq detail
    result = ref_low + src_high

    # Only apply within the mask, keep src outside
    mask_3d = mask_bool[:, :, np.newaxis].astype(np.float32)
    result = result * mask_3d + src_f * (1 - mask_3d)

    return np.clip(result, 0, 255).astype(np.uint8)


# Class types that should stop the upstream trace (other loop boundaries).
_LOOP_END_CLASSES = frozenset([
    "ReFaceLoopEnd",
    "easy forLoopEnd",
    "easy whileLoopEnd",
])


class ReFaceLoopEnd:
    """Paste the modified head back onto *dst_image* and advance the loop.

    When more items remain, the graph between LoopStart↔LoopEnd is cloned
    (execution-inversion pattern) so that the next item is processed.
    When done, the final *dst_image* is returned.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "loop_ctx": ("REFACE_LOOP_CTX",),
                "blend_mode": (["direct", "poisson", "color_match", "wavelet"],
                               {"default": "direct"}),
            },
            "optional": {
                # Optional so that LoopEnd still executes when LoopStart
                # blocks the processing pipeline (items array is empty).
                "modified_head_image": ("IMAGE",),
                "modified_head_mask": ("MASK",),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("dst_image",)
    FUNCTION = "execute"
    CATEGORY = "ReFace"

    # ── public entry point ────────────────────────────────────────────────

    def execute(self, flow, loop_ctx, blend_mode="direct",
                modified_head_image=None, modified_head_mask=None,
                dynprompt=None, unique_id=None):
        items = loop_ctx.get("items", [])
        idx = loop_ctx.get("current_index", 0)
        dst_image = loop_ctx["dst_image"]

        # Nothing to process, or no modified image provided → return as-is
        if not items or idx >= len(items) or modified_head_image is None:
            return (dst_image,)

        # ── Paste modified head back onto dst_image ──
        updated_dst = self._paste_back(
            dst_image, modified_head_image, modified_head_mask,
            items[idx], blend_mode)

        # ── Advance index and update dst_image in context ──
        new_ctx = dict(loop_ctx)
        new_ctx["current_index"] = idx + 1
        new_ctx["dst_image"] = updated_dst

        # ── Done? ──
        if new_ctx["current_index"] >= len(items):
            return (updated_dst,)

        # ── More items → expand graph for next iteration ──
        return self._expand_loop(flow, new_ctx, dynprompt, unique_id)

    # ── paste logic ───────────────────────────────────────────────────────

    @staticmethod
    def _paste_back(dst_image, modified_head_image, modified_head_mask,
                    item, blend_mode):
        """Paste modified head back with optional color correction and mask blending."""
        import cv2

        bbox = item["dst_head_bbox"]
        left, top = bbox["left"], bbox["top"]
        right, bottom = bbox["right"], bbox["bottom"]
        target_h = bottom - top
        target_w = right - left

        updated = dst_image.clone()
        mod = modified_head_image[0]  # (h, w, C)

        # Resize modified image to target bbox size
        # Use 'area' for downscaling (sharper), 'bicubic' for upscaling
        if mod.shape[0] != target_h or mod.shape[1] != target_w:
            is_downscale = (mod.shape[0] > target_h) or (mod.shape[1] > target_w)
            mode = "area" if is_downscale else "bicubic"
            mod = (
                torch.nn.functional.interpolate(
                    mod.permute(2, 0, 1).unsqueeze(0),
                    size=(target_h, target_w),
                    mode=mode,
                    **({} if mode == "area" else {"align_corners": False}),
                )
                .squeeze(0)
                .permute(1, 2, 0)
            )

        # Handle channel mismatch (e.g. user returns RGBA but dst is RGB)
        dst_c = updated.shape[3]
        mod_c = mod.shape[2]
        if mod_c > dst_c:
            mod = mod[:, :, :dst_c]
        elif mod_c < dst_c:
            pad = torch.zeros(mod.shape[0], mod.shape[1], dst_c - mod_c,
                              dtype=mod.dtype, device=mod.device)
            mod = torch.cat([mod, pad], dim=2)

        # Resize mask to target bbox size (nearest interpolation)
        mask_np = None
        if modified_head_mask is not None:
            mask_t = modified_head_mask[0]  # (h, w)
            if mask_t.shape[0] != target_h or mask_t.shape[1] != target_w:
                mask_t = (
                    torch.nn.functional.interpolate(
                        mask_t.unsqueeze(0).unsqueeze(0),
                        size=(target_h, target_w),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            mask_np = (mask_t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Convert to numpy uint8 for color correction / blending
        dst_region = (updated[0, top:bottom, left:right, :].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        mod_np = (mod.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        if blend_mode == "color_match" and mask_np is not None:
            # Lab color transfer: match modified region's color stats to original
            mod_np = _lab_color_transfer(mod_np, dst_region, mask_np)

        if blend_mode == "wavelet" and mask_np is not None:
            # Wavelet: ref low-freq color + src high-freq detail
            mod_np = _wavelet_color_transfer(mod_np, dst_region, mask_np)

        if blend_mode == "poisson" and mask_np is not None:
            # Poisson blending (seamless clone)
            mod_np = _poisson_blend(mod_np, dst_region, mask_np)
            # After poisson blend, result is already composited, paste directly
            result = torch.from_numpy(mod_np.astype(np.float32) / 255.0).to(
                dtype=updated.dtype, device=updated.device)
            updated[0, top:bottom, left:right, :] = result
            return updated

        # Mask-based alpha blending (for direct & color_match modes)
        if mask_np is not None:
            alpha = mask_np.astype(np.float32) / 255.0  # (h, w) in [0, 1]
            alpha_3d = alpha[:, :, np.newaxis]  # (h, w, 1)
            blended = (mod_np.astype(np.float32) * alpha_3d +
                       dst_region.astype(np.float32) * (1 - alpha_3d))
            blended = blended.clip(0, 255).astype(np.uint8)
        else:
            blended = mod_np

        result = torch.from_numpy(blended.astype(np.float32) / 255.0).to(
            dtype=updated.dtype, device=updated.device)
        updated[0, top:bottom, left:right, :] = result
        return updated

    # ── graph expansion (execution-inversion loop) ────────────────────────

    def _expand_loop(self, flow, new_ctx, dynprompt, unique_id):
        """Clone the sub-graph between LoopStart and this LoopEnd for the
        next iteration, wiring updated loop-carried variables."""
        graph = GraphBuilder()
        open_node = flow[0]  # ReFaceLoopStart node ID

        # 1. Trace upstream dependencies
        upstream = {}
        parent_ids = []
        self._explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))

        # 2. Discover output nodes (e.g. PreviewImage) inside the loop body
        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for nid, node in prompts.items():
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            class_def = ALL_NODE_CLASS_MAPPINGS.get(class_type)
            if class_def and getattr(class_def, "OUTPUT_NODE", False):
                for _k, v in node["inputs"].items():
                    if _is_link(v):
                        output_nodes[nid] = v

        self._explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)

        # 3. Collect all nodes contained between open ↔ close
        contained = {}
        self._collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        # 4. Clone nodes
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            label = "Recurse" if node_id == unique_id else node_id
            n = graph.node(original["class_type"], label)
            n.set_override_display_id(node_id)

        # 5. Re-wire inputs
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            n = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original["inputs"].items():
                if _is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    n.set_input(k, parent.out(v[1]))
                else:
                    n.set_input(k, v)

        # 6. Override loop-carried variable on the cloned LoopStart
        new_open = graph.lookup_node(open_node)
        new_open.set_input("_loop_ctx", new_ctx)

        # 7. Return from the cloned LoopEnd
        my_clone = graph.lookup_node("Recurse")
        return {
            "result": (my_clone.out(0),),
            "expand": graph.finalize(),
        }

    # ── upstream tracing helpers (adapted from ComfyUI-Easy-Use) ──────────

    def _explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for _k, v in node_info["inputs"].items():
            if not _is_link(v):
                continue
            parent_id = v[0]
            display_id = dynprompt.get_display_node_id(parent_id)
            display_node = dynprompt.get_node(display_id)
            class_type = display_node["class_type"]
            if class_type not in _LOOP_END_CLASSES:
                parent_ids.append(display_id)
            if parent_id not in upstream:
                upstream[parent_id] = []
                self._explore_dependencies(parent_id, dynprompt, upstream, parent_ids)
            upstream[parent_id].append(node_id)

    @staticmethod
    def _explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id, link in output_nodes.items():
                nid = link[0]
                if nid in parent_ids and display_id == nid and output_id not in upstream[parent_id]:
                    if "." in parent_id:
                        arr = parent_id.split(".")
                        arr[-1] = output_id
                        upstream[parent_id].append(".".join(arr))
                    else:
                        upstream[parent_id].append(output_id)

    def _collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self._collect_contained(child_id, upstream, contained)


# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ReFaceLoopStart": ReFaceLoopStart,
    "ReFaceLoopEnd": ReFaceLoopEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReFaceLoopStart": "ReFace Loop Start",
    "ReFaceLoopEnd": "ReFace Loop End",
}
