"""Face detection using RetinaFace from insightface."""

import os
import numpy as np

import folder_paths

_detector = None


def _get_model_dir():
    return os.path.join(folder_paths.models_dir, "reface")


def _get_onnx_providers():
    """Build ONNX Runtime provider list: CUDA > CoreML (macOS) > CPU."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def get_face_detector():
    """Get or create singleton face detector (insightface RetinaFace)."""
    global _detector
    if _detector is not None:
        return _detector

    from insightface.app import FaceAnalysis

    model_dir = _get_model_dir()
    os.makedirs(model_dir, exist_ok=True)

    providers = _get_onnx_providers()
    print(f"[ReFace] InsightFace using ONNX providers: {providers}")

    _detector = FaceAnalysis(
        name="buffalo_l",
        root=model_dir,
        providers=providers,
    )
    _detector.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
    return _detector


def _face_to_dict(face):
    """Convert an insightface Face object to a serializable dict."""
    x1, y1, x2, y2 = face.bbox[:4]
    area = (x2 - x1) * (y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)

    # Compute inter-eye distance from landmarks
    # InsightFace kps: [left_eye, right_eye, nose, left_mouth, right_mouth]
    eye_dist = None
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 2:
        le, re = kps[0], kps[1]
        eye_dist = float(np.sqrt((le[0] - re[0]) ** 2 + (le[1] - re[1]) ** 2))

    return {
        "bbox": (float(x1), float(y1), float(x2), float(y2)),
        "center": (float(cx), float(cy)),
        "area": float(area),
        "embedding": embedding,
        "eye_dist": eye_dist,
    }


def detect_largest_face(image_bgr: np.ndarray):
    """Detect faces and return the largest one.

    Args:
        image_bgr: BGR image as numpy array (H, W, 3), uint8.

    Returns:
        A dict with keys 'bbox', 'center', 'area', 'embedding',
        or None if no face detected.
    """
    detector = get_face_detector()
    faces = detector.get(image_bgr)

    if not faces:
        return None

    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return _face_to_dict(best)


def detect_all_faces(image_bgr: np.ndarray, max_count: int | None = None):
    """Detect all faces and return them sorted by size (largest first).

    Args:
        image_bgr: BGR image as numpy array (H, W, 3), uint8.
        max_count: Maximum number of faces to return. None for all.

    Returns:
        List of dicts, each with 'bbox', 'center', 'area', 'embedding'.
        Sorted by bounding-box area descending. Empty list if none found.
    """
    detector = get_face_detector()
    faces = detector.get(image_bgr)

    if not faces:
        return []

    results = [_face_to_dict(f) for f in faces]
    results.sort(key=lambda r: r["area"], reverse=True)

    if max_count is not None:
        results = results[:max_count]

    return results


def compute_embedding_from_image(image_bgr: np.ndarray):
    """Run InsightFace on an image and return the largest face's embedding.

    Useful for computing ArcFace features from a cropped face image.

    Args:
        image_bgr: BGR image (H, W, 3), uint8.

    Returns:
        Normalized embedding (512,) numpy array, or None if no face found.
    """
    detector = get_face_detector()
    faces = detector.get(image_bgr)
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return getattr(best, "normed_embedding", getattr(best, "embedding", None))
