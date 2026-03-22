"""Face detection using YuNet (OpenCV) + ArcFace (ONNX, lazy-loaded).

YuNet: lightweight face detector (~88KB), loads instantly.
ArcFace: face recognition/embedding model, loaded only when multi-face
matching is needed.

YuNet model: face_detection_yunet_2023mar.onnx
  from https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

ArcFace model: w600k_r50.onnx
  from https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
"""

import os
import numpy as np
import cv2

import folder_paths

_yunet_detector = None
_arcface_session = None

YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

ARCFACE_MODEL_NAME = "w600k_r50.onnx"


def _get_model_dir():
    return os.path.join(folder_paths.models_dir, "reface")


def _download_yunet():
    """Download YuNet model if not present."""
    model_dir = _get_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, YUNET_MODEL_NAME)

    if os.path.isfile(model_path):
        return model_path

    print(f"[ReFace] Downloading YuNet face detection model to {model_path} ...")
    import urllib.request
    urllib.request.urlretrieve(YUNET_MODEL_URL, model_path)
    print("[ReFace] YuNet model downloaded.")
    return model_path


# ──────────────────────────────────────────────────────────────────────────────
# YuNet face detection
# ──────────────────────────────────────────────────────────────────────────────


def _get_yunet(width, height):
    """Get or create YuNet detector, resized to input dimensions."""
    global _yunet_detector
    model_path = _download_yunet()

    if _yunet_detector is None:
        _yunet_detector = cv2.FaceDetectorYN.create(
            model_path, "", (width, height),
            score_threshold=0.5, nms_threshold=0.3, top_k=5000)
        print(f"[ReFace] YuNet face detector loaded ({os.path.getsize(model_path) / 1024:.0f}KB)")
    else:
        _yunet_detector.setInputSize((width, height))

    return _yunet_detector


def _yunet_detect(image_bgr):
    """Run YuNet detection on a BGR image.

    Returns:
        List of raw detection arrays. Each has 15 values:
        [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        Landmarks: right_eye, left_eye, nose_tip, right_corner_mouth, left_corner_mouth
    """
    h, w = image_bgr.shape[:2]
    detector = _get_yunet(w, h)
    _, faces = detector.detect(image_bgr)
    if faces is None:
        return []
    return faces


def _yunet_to_dict(det):
    """Convert YuNet detection array to a dict matching our API."""
    x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
    x2, y2 = x + w, y + h
    area = float(w * h)
    cx = x + w / 2.0
    cy = y + h / 2.0
    score = float(det[14])

    # YuNet landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
    right_eye = (float(det[4]), float(det[5]))
    left_eye = (float(det[6]), float(det[7]))
    nose = (float(det[8]), float(det[9]))
    right_mouth = (float(det[10]), float(det[11]))
    left_mouth = (float(det[12]), float(det[13]))

    eye_dist = float(np.sqrt(
        (left_eye[0] - right_eye[0]) ** 2 +
        (left_eye[1] - right_eye[1]) ** 2))

    return {
        "bbox": (float(x), float(y), float(x2), float(y2)),
        "center": (float(cx), float(cy)),
        "area": area,
        "score": score,
        "eye_dist": eye_dist,
        "landmarks": {
            "right_eye": right_eye,
            "left_eye": left_eye,
            "nose": nose,
            "right_mouth": right_mouth,
            "left_mouth": left_mouth,
        },
        "embedding": None,  # Computed lazily via ArcFace when needed
    }


def detect_largest_face(image_bgr: np.ndarray):
    """Detect faces and return the largest one.

    Args:
        image_bgr: BGR image (H, W, 3), uint8.

    Returns:
        A dict with keys 'bbox', 'center', 'area', 'eye_dist', 'landmarks',
        'embedding' (None), or None if no face detected.
    """
    faces = _yunet_detect(image_bgr)
    if len(faces) == 0:
        return None

    best = max(faces, key=lambda f: f[2] * f[3])  # max by w*h
    return _yunet_to_dict(best)


def detect_all_faces(image_bgr: np.ndarray, max_count: int | None = None):
    """Detect all faces and return them sorted by size (largest first).

    Args:
        image_bgr: BGR image (H, W, 3), uint8.
        max_count: Maximum number of faces to return.

    Returns:
        List of dicts sorted by area descending. Empty if none found.
    """
    faces = _yunet_detect(image_bgr)
    if len(faces) == 0:
        return []

    results = [_yunet_to_dict(f) for f in faces]
    results.sort(key=lambda r: r["area"], reverse=True)

    if max_count is not None:
        results = results[:max_count]

    return results


# ──────────────────────────────────────────────────────────────────────────────
# ArcFace embedding (lazy-loaded, only for multi-face matching)
# ──────────────────────────────────────────────────────────────────────────────

# ArcFace alignment template (112x112) — standard 5-point landmarks
# Order: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _get_onnx_providers():
    """Build ONNX Runtime provider list: CUDA > CoreML > CPU."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _get_arcface():
    """Get or create singleton ArcFace ONNX session."""
    global _arcface_session
    if _arcface_session is not None:
        return _arcface_session

    import onnxruntime as ort

    model_path = os.path.join(_get_model_dir(), ARCFACE_MODEL_NAME)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[ReFace] ArcFace model not found at {model_path}.\n"
            f"Please download w600k_r50.onnx and place it there.\n"
            f"See: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch")

    providers = _get_onnx_providers()
    print(f"[ReFace] Loading ArcFace model with providers: {providers}")

    _arcface_session = ort.InferenceSession(model_path, providers=providers)
    print("[ReFace] ArcFace model loaded.")
    return _arcface_session


def _align_face_for_arcface(image_bgr, landmarks_dict):
    """Align face using 5-point landmarks to 112x112 for ArcFace.

    Args:
        image_bgr: BGR image (H, W, 3), uint8.
        landmarks_dict: dict with 'left_eye', 'right_eye', 'nose',
                        'left_mouth', 'right_mouth'.

    Returns:
        Aligned face (112, 112, 3) BGR uint8.
    """
    src_pts = np.array([
        landmarks_dict["left_eye"],
        landmarks_dict["right_eye"],
        landmarks_dict["nose"],
        landmarks_dict["left_mouth"],
        landmarks_dict["right_mouth"],
    ], dtype=np.float32)

    # Estimate similarity transform
    tform = cv2.estimateAffinePartial2D(src_pts, ARCFACE_DST, method=cv2.LMEDS)[0]
    if tform is None:
        # Fallback: full affine
        tform = cv2.getAffineTransform(src_pts[:3], ARCFACE_DST[:3])

    aligned = cv2.warpAffine(image_bgr, tform, (112, 112), borderValue=0)
    return aligned


def compute_embedding(image_bgr: np.ndarray, face_dict: dict):
    """Compute ArcFace embedding for a detected face.

    Args:
        image_bgr: Full BGR image.
        face_dict: Detection dict with 'landmarks'.

    Returns:
        Normalized embedding (512,) numpy array, or None on failure.
    """
    landmarks = face_dict.get("landmarks")
    if landmarks is None:
        return None

    aligned = _align_face_for_arcface(image_bgr, landmarks)

    # Preprocess for ArcFace: BGR→RGB, HWC→CHW, normalize to [-1, 1]
    img = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]
    img = img.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 112, 112)

    session = _get_arcface()
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})[0]

    # L2 normalize
    emb = output[0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def compute_embedding_from_image(image_bgr: np.ndarray):
    """Detect largest face in image and compute its ArcFace embedding.

    Convenience function for computing embeddings from cropped face images.

    Args:
        image_bgr: BGR image (H, W, 3), uint8.

    Returns:
        Normalized embedding (512,) numpy array, or None if no face found.
    """
    face = detect_largest_face(image_bgr)
    if face is None:
        return None
    return compute_embedding(image_bgr, face)
