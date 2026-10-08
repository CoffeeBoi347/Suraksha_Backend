"""app/framequality.py - blur / low-light / jitter handling for the Suraksha vision HUD.

Cheap (~1 ms on a 320px gray copy). Nothing here changes what the app *sends* to the client.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


@dataclass
class FrameQuality:
    sharp: float = 0.0        # variance of Laplacian (raw)
    ratio: float = 1.0        # sharp / recent-best sharp for this connection (1.0 = as sharp as it gets)
    brightness: float = 128.0
    contrast: float = 50.0

    @property
    def bad(self) -> bool:    # camera shake / heavy blur: keypoints are garbage, don't trust motion cues
        return self.ratio < 0.30

    @property
    def soft(self) -> bool:   # mildly blurry: sharpen for the detector, lower conf
        return self.ratio < 0.60

    @property
    def flat(self) -> bool:   # dark or washed out: CLAHE helps
        return self.brightness < 70 or self.contrast < 32

    @property
    def degraded(self) -> bool:
        return self.bad or self.soft or self.flat


def sharpness(img: np.ndarray, width: int = 320) -> float:
    """Variance of Laplacian on a downscaled gray copy. Works on BGR or gray."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_32F).var())


class QualityTracker:
    """Per-connection. Sharpness is scene dependent, so judge each frame against the recent best, not a constant."""

    def __init__(self, decay: float = 0.985):
        self.base = 0.0
        self.decay = decay

    def assess(self, frame: np.ndarray) -> FrameQuality:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, max(1, int(h * 320 / w))), interpolation=cv2.INTER_AREA) if w > 320 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        s = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        self.base = max(s, self.base * self.decay)
        ratio = s / self.base if self.base > 1e-6 else 1.0
        return FrameQuality(sharp=s, ratio=ratio, brightness=float(gray.mean()), contrast=float(gray.std()))


def enhance(frame: np.ndarray, q: FrameQuality) -> np.ndarray:
    """Inference-only copy. Never feed this to Gemini or snapshots (keep evidence real)."""
    if not (q.soft or q.flat):
        return frame
    out = frame
    if q.flat:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if q.soft and not q.bad:                      # unsharp on heavy blur only amplifies noise
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    return out


class KeypointSmoother:
    """Confidence-gated EMA per track. alpha=1 -> passthrough. Big jumps (> jump*bh) are real motion: not smoothed."""

    def __init__(self):
        self.prev: np.ndarray | None = None

    def __call__(self, kxy, kcf, bh: float, alpha: float, min_conf: float = 0.3, jump: float = 0.35):
        if kxy is None:
            self.prev = None
            return kxy
        cur = np.asarray(kxy, dtype=np.float32)
        if self.prev is None or alpha >= 1.0 or self.prev.shape != cur.shape:
            self.prev = cur.copy()
            return kxy
        ok = np.ones(len(cur), bool) if kcf is None else (np.asarray(kcf) >= min_conf)
        near = np.linalg.norm(cur - self.prev, axis=1) < jump * max(bh, 1.0)
        m = ok & near
        out = cur.copy()
        out[m] = alpha * cur[m] + (1.0 - alpha) * self.prev[m]
        self.prev = out.copy()
        return out


def best_crops(crops: list, n: int, min_keep: int = 3, rel_floor: float = 0.4) -> tuple[list, float]:
    """From time-ordered crops pick up to n sharpest, drop ones far blurrier than the best, keep time order.
    -> (crops, blur_ratio of the set vs best). Always keeps >= min_keep if available."""
    if len(crops) <= 1:
        return crops, 1.0
    sc = [sharpness(c, 192) for c in crops]
    top = max(sc) or 1.0
    idx = sorted(range(len(crops)), key=lambda i: sc[i], reverse=True)
    keep = [i for i in idx if sc[i] >= rel_floor * top][:n]
    for i in idx:                                  # backfill so Gemini still gets enough frames
        if len(keep) >= min(min_keep, len(crops)):
            break
        if i not in keep:
            keep.append(i)
    keep.sort()
    avg = sum(sc[i] for i in keep) / len(keep)
    return [crops[i] for i in keep], float(min(1.0, avg / top))