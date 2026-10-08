"""Person re-identification using torchreid OSNet x0_25 embeddings.

Drop-in replacement for the old colour-histogram reid.py. Same API:
    from app.reid import PersonGallery, embed_person

Falls back to colour-histogram if torchreid is not installed, so the
existing code keeps working without any new dependency during development.

Install:  pip install torchreid  (MIT licence)
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque

import cv2
import numpy as np

log = logging.getLogger("suraksha.reid")

# ---- try torchreid, fall back to histogram ----
_USE_OSNET = False
_osnet_extractor = None

try:
    import torchreid  # type: ignore
    _USE_OSNET = True
except ImportError:
    log.info("torchreid not installed — falling back to colour-histogram re-ID")

EMBED_DIM = 512 if _USE_OSNET else 256        # OSNet x0_25 = 512-d; histogram = 256 bins
MATCH_THRESH = float(os.getenv("REID_MATCH_THRESH", "0.55" if _USE_OSNET else "0.65"))
GALLERY_TTL_S = float(os.getenv("REID_GALLERY_TTL", "120"))     # forget identities after 2 min
GALLERY_MAX = int(os.getenv("REID_GALLERY_MAX", "64"))
RETURN_WINDOW_S = 60.0        # someone who re-appears within this window counts as "returned"
PERSIST_MIN_SIGHTINGS = 3     # seen at least this many times across sessions
PERSIST_MIN_SPAN_S = 20.0     # first-to-last sighting spans this long

_INPUT_SIZE = (128, 256)      # (w, h) — OSNet default


def _init_osnet():
    global _osnet_extractor
    if _osnet_extractor is not None:
        return
    try:
        _osnet_extractor = torchreid.utils.FeatureExtractor(
            model_name="osnet_x0_25",
            model_path="",          # auto-downloads ~2 MB
            device="cuda" if os.getenv("DEVICE", "cpu") == "cuda" else "cpu",
        )
        log.info("OSNet x0_25 re-ID model loaded")
    except Exception as e:
        log.warning("OSNet init failed, falling back to histogram: %s", e)
        globals()["_USE_OSNET"] = False


def _osnet_embed(crop_bgr: np.ndarray) -> np.ndarray | None:
    """256-d L2-normalised OSNet embedding from a person crop."""
    _init_osnet()
    if _osnet_extractor is None:
        return None
    try:
        img = cv2.resize(crop_bgr, _INPUT_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feat = _osnet_extractor(img_rgb)      # returns tensor [1, 512]
        v = feat.cpu().numpy().flatten().astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-6)
        return v
    except Exception as e:
        log.debug("osnet embed error: %s", e)
        return None


def _hist_embed(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Colour-histogram embedding (fallback). 256-d, L2-normalised."""
    try:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [128], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()
        vec = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        vec /= max(np.linalg.norm(vec), 1e-6)
        return vec
    except Exception:
        return None


def embed_person(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    """Extract an embedding vector from a person crop inside `frame`."""
    h, w = frame.shape[:2]
    cx1 = int(max(0, x1))
    cy1 = int(max(0, y1))
    cx2 = int(min(w, x2))
    cy2 = int(min(h, y2))
    if cx2 - cx1 < 16 or cy2 - cy1 < 24:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    if _USE_OSNET:
        v = _osnet_embed(crop)
        if v is not None:
            return v
    return _hist_embed(crop)


class _Identity:
    """One tracked identity in the gallery."""
    __slots__ = ("emb", "sightings", "first_ts", "last_ts", "last_dist", "track_keys", "gone_since")

    def __init__(self, emb: np.ndarray, ts: float, dist: float, track_key: str):
        self.emb = emb.copy()
        self.sightings = 1
        self.first_ts = ts
        self.last_ts = ts
        self.last_dist = dist
        self.track_keys: set[str] = {track_key}
        self.gone_since: float | None = None     # when the track disappeared (for return detection)

    def update(self, emb: np.ndarray, ts: float, dist: float, track_key: str):
        # EMA update of the embedding for drift resistance
        alpha = 0.3
        self.emb = alpha * emb + (1 - alpha) * self.emb
        self.emb /= max(np.linalg.norm(self.emb), 1e-6)
        self.sightings += 1
        self.last_ts = ts
        self.last_dist = dist
        self.track_keys.add(track_key)
        self.gone_since = None


class PersonGallery:
    """Assigns track-level embeddings to gallery identities. Thread-unsafe — call from the frame loop."""

    def __init__(self):
        self._ids: dict[int, _Identity] = {}     # identity id -> Identity
        self._track_map: dict[str, int] = {}     # current track key -> identity id
        self._next_id = 0

    def assign(self, track_key: str, ts: float, emb: np.ndarray, dist: float,
               live_keys: set[str]) -> int | None:
        """Match embedding to gallery, return identity id (or None if embedding is bad)."""
        if emb is None:
            return self._track_map.get(track_key)

        # already mapped?
        if track_key in self._track_map:
            iid = self._track_map[track_key]
            if iid in self._ids:
                self._ids[iid].update(emb, ts, dist, track_key)
                return iid

        # find best match among existing identities
        best_id, best_sim = None, -1.0
        for iid, ident in self._ids.items():
            sim = float(np.dot(emb, ident.emb))
            if sim > best_sim:
                best_sim, best_id = sim, iid

        if best_sim >= MATCH_THRESH and best_id is not None:
            ident = self._ids[best_id]
            # mark as returned if was gone
            if ident.gone_since is not None and ts - ident.gone_since <= RETURN_WINDOW_S:
                ident.gone_since = None  # returned!
            ident.update(emb, ts, dist, track_key)
            self._track_map[track_key] = best_id
            return best_id

        # new identity
        iid = self._next_id
        self._next_id += 1
        self._ids[iid] = _Identity(emb, ts, dist, track_key)
        self._track_map[track_key] = iid
        # cap gallery size
        if len(self._ids) > GALLERY_MAX:
            oldest = min(self._ids, key=lambda k: self._ids[k].last_ts)
            self._remove_id(oldest)
        return iid

    def persistence(self, iid: int, ts: float, user_speed: float) -> tuple[int, float, str]:
        """How persistent is this identity? -> (rank 0|1|2, score, reason).
        Used for follow detection: someone who keeps re-appearing is following."""
        ident = self._ids.get(iid)
        if ident is None:
            return 0, 0.0, ""
        span = ts - ident.first_ts
        n = ident.sightings
        n_tracks = len(ident.track_keys)
        # track churn = ByteTrack lost and re-acquired them (they returned)
        returned = n_tracks >= 2

        rank, score, why = 0, 0.0, ""
        if returned and span >= PERSIST_MIN_SPAN_S and user_speed >= 0.4:
            rank, score, why = 2, 0.65, "same person keeps reappearing"
        elif n >= PERSIST_MIN_SIGHTINGS and span >= PERSIST_MIN_SPAN_S:
            rank, score, why = 1, 0.40, "person seen repeatedly"
        return rank, score, why

    def _remove_id(self, iid: int):
        self._ids.pop(iid, None)
        for k in [k for k, v in self._track_map.items() if v == iid]:
            del self._track_map[k]

    def prune(self, ts: float):
        """Drop old identities and stale track mappings."""
        for iid in [iid for iid, ident in self._ids.items() if ts - ident.last_ts > GALLERY_TTL_S]:
            self._remove_id(iid)
        # mark gone for identities whose tracks all disappeared
        for iid, ident in self._ids.items():
            if ident.gone_since is None:
                # check if any of its track keys are still live (we don't have live_keys here,
                # but staleness is handled by the TTL above)
                pass