"""Bystander face blur using OpenCV's built-in YuNet face detector. Zero extra dependencies.

Usage:
    from app.faceblur import blur_bystanders
    blurred = blur_bystanders(img, target_box)

The target's face (overlapping the red box) is left untouched; all others are Gaussian-blurred.
Auto-downloads the YuNet ONNX model (~400 KB) on first use. Never blocks or raises.
"""
import logging
import os
import urllib.request

import cv2
import numpy as np

log = logging.getLogger("suraksha.faceblur")

_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_MODEL_DIR = os.getenv("FACEBLUR_MODEL_DIR", os.path.join(os.path.dirname(__file__), ".models"))
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_detection_yunet_2023mar.onnx")
_SCORE_THR = float(os.getenv("FACEBLUR_SCORE", "0.7"))
_NMS_THR = 0.3
_detector: cv2.FaceDetectorYN | None = None
_last_size: tuple[int, int] = (0, 0)


def _ensure_model() -> str:
    if os.path.isfile(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_MODEL_DIR, exist_ok=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        log.info("downloaded YuNet model to %s", _MODEL_PATH)
    except Exception as e:
        log.warning("YuNet download failed: %s", e)
    return _MODEL_PATH


def _get_detector(w: int, h: int) -> cv2.FaceDetectorYN | None:
    global _detector, _last_size
    if _detector is not None and _last_size == (w, h):
        return _detector
    path = _ensure_model()
    if not os.path.isfile(path):
        return None
    try:
        _detector = cv2.FaceDetectorYN.create(path, "", (w, h), _SCORE_THR, _NMS_THR)
        _last_size = (w, h)
        return _detector
    except Exception as e:
        log.warning("YuNet init failed: %s", e)
        return None


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def blur_bystanders(img: np.ndarray, target_box: tuple, ksize: int = 31) -> np.ndarray:
    """Blur all detected faces EXCEPT the one overlapping target_box (x1, y1, x2, y2).

    Returns the image (mutated in place for speed). On any error returns the original unchanged.
    """
    try:
        h, w = img.shape[:2]
        det = _get_detector(w, h)
        if det is None:
            return img
        _, faces = det.detect(img)
        if faces is None or len(faces) == 0:
            return img
        for face in faces:
            fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            face_box = (fx, fy, fx + fw, fy + fh)
            if _iou(face_box, target_box) > 0.1:
                continue  # this is the target's face — leave it
            # clamp to image bounds
            x1c = max(0, fx)
            y1c = max(0, fy)
            x2c = min(w, fx + fw)
            y2c = min(h, fy + fh)
            if x2c - x1c < 4 or y2c - y1c < 4:
                continue
            ks = max(ksize, int(max(fw, fh) * 0.4)) | 1  # scale kernel to face size, keep odd
            img[y1c:y2c, x1c:x2c] = cv2.GaussianBlur(img[y1c:y2c, x1c:x2c], (ks, ks), 0)
    except Exception as e:
        log.debug("blur_bystanders error: %s", e)
    return img