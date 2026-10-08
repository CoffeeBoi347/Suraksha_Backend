"""Server-side audio event classification for Suraksha.

Detects: scream, gunshot, glass breaking, explosion, siren — events the client
microphone picks up but can't classify on-device with confidence.

Usage:
    from app.audio_classify import AudioClassifier
    clf = AudioClassifier()                   # loads model once
    events = clf.classify(pcm_int16, sr=16000)
    # events = [("Scream", 0.87), ("Gunshot", 0.12)]

Dependencies:
    pip install torch torchaudio    # already present for YOLO
    The model downloads on first use (~80 MB for CNN14, ~15 MB for MobileNetV1).

Licence: PANNs weights are MIT. torchaudio is BSD-2-Clause.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

log = logging.getLogger("suraksha.audio_classify")

# Events we care about and their AudioSet class indices (ontology ID -> integer index in the 527-class model)
# Full AudioSet ontology: https://research.google.com/audioset/ontology.html
EVENTS_OF_INTEREST = {
    # screams / distress
    "Screaming": 0.70,
    "Shout": 0.50,
    "Crying, sobbing": 0.50,
    # weapons
    "Gunshot, gunfire": 0.60,
    "Explosion": 0.55,
    # breaking
    "Glass": 0.50,
    "Shatter": 0.55,
    "Breaking": 0.50,
    # vehicle threat
    "Tire squeal": 0.50,
    "Skidding": 0.50,
    # emergency
    "Siren": 0.45,
    "Emergency vehicle": 0.45,
    "Car alarm": 0.50,
}

# Map AudioSet label -> our canonical label for the engine
LABEL_MAP = {
    "Screaming": "scream",
    "Shout": "scream",
    "Crying, sobbing": "distress",
    "Gunshot, gunfire": "gunshot",
    "Explosion": "explosion",
    "Glass": "glass_break",
    "Shatter": "glass_break",
    "Breaking": "glass_break",
    "Tire squeal": "vehicle_threat",
    "Skidding": "vehicle_threat",
    "Siren": "siren",
    "Emergency vehicle": "siren",
    "Car alarm": "vehicle_threat",
}

_MODEL_NAME = os.getenv("AUDIO_MODEL", "Cnn14_mAP=0.431.pth")
_SAMPLE_RATE = 32000   # PANNs default


class AudioClassifier:
    """Wraps a PANNs (Pretrained Audio Neural Networks) model for environmental sound classification.

    Uses torchaudio's built-in mel-spectrogram pipeline so no extra feature libs are needed.
    Falls back gracefully if the model can't load.
    """

    def __init__(self):
        self._model = None
        self._labels: list[str] = []
        self._interest_idx: dict[int, tuple[str, float]] = {}  # model-index -> (our_label, threshold)
        self._ready = False
        self._init_attempted = False

    def _init(self):
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            import torch
            import torchaudio  # type: ignore

            # Use torchaudio's pipeline for PANNs if available, else manual
            # Try loading via torch hub (panns_inference package)
            try:
                from panns_inference import AudioTagging  # type: ignore
                self._model = AudioTagging(checkpoint_path=None, device="cpu")
                self._labels = self._model.labels
            except ImportError:
                log.info("panns_inference not installed; using manual PANNs loader")
                self._load_manual(torch)
                return

            # build interest index
            for i, label in enumerate(self._labels):
                if label in EVENTS_OF_INTEREST:
                    self._interest_idx[i] = (LABEL_MAP[label], EVENTS_OF_INTEREST[label])

            self._ready = True
            log.info("AudioClassifier ready (%d labels, %d events tracked)", len(self._labels), len(self._interest_idx))

        except Exception as e:
            log.warning("AudioClassifier init failed: %s", e)

    def _load_manual(self, torch):
        """Manual loading without panns_inference package — just uses the checkpoint directly."""
        # For now, mark as not ready and use the lightweight fallback
        log.info("AudioClassifier: manual PANNs loading not implemented; use pip install panns_inference")
        self._ready = False

    def classify(self, audio: np.ndarray, sr: int = 16000, top_k: int = 5) -> list[tuple[str, float]]:
        """Classify an audio chunk.

        Args:
            audio: PCM int16 or float32 mono audio, any sample rate
            sr: sample rate of the input
            top_k: max events to return

        Returns:
            List of (canonical_label, confidence) for events of interest above threshold,
            sorted by confidence descending.
        """
        self._init()
        if not self._ready:
            return self._fallback_classify(audio, sr)

        try:
            import torch
            import torchaudio

            # convert to float32 if needed
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            # resample if needed
            if sr != _SAMPLE_RATE:
                waveform = torch.from_numpy(audio).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=_SAMPLE_RATE)
                waveform = resampler(waveform)
                audio = waveform.squeeze(0).numpy()

            # run inference
            clipwise_output, _ = self._model.inference(audio[np.newaxis, :])
            probs = clipwise_output[0]  # shape: (527,)

            # filter to events of interest
            results = []
            for idx, (label, threshold) in self._interest_idx.items():
                conf = float(probs[idx])
                if conf >= threshold:
                    results.append((label, round(conf, 3)))

            # deduplicate by canonical label (keep highest confidence)
            best: dict[str, float] = {}
            for label, conf in results:
                best[label] = max(best.get(label, 0.0), conf)

            return sorted(best.items(), key=lambda x: -x[1])[:top_k]

        except Exception as e:
            log.debug("classify error: %s", e)
            return []

    def _fallback_classify(self, audio: np.ndarray, sr: int) -> list[tuple[str, float]]:
        """Simple energy + ZCR based scream detection when PANNs isn't available.
        Not as accurate, but zero-dependency and better than nothing."""
        try:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            if len(audio) < sr * 0.1:  # too short
                return []

            # RMS energy
            rms = float(np.sqrt(np.mean(audio ** 2)))

            # Zero crossing rate (high = scream-like)
            signs = np.sign(audio)
            zcr = float(np.mean(np.abs(np.diff(signs)) > 0))

            # Spectral centroid approximation (high-pitched = scream-like)
            fft = np.abs(np.fft.rfft(audio[:min(len(audio), sr)]))
            freqs = np.fft.rfftfreq(min(len(audio), sr), 1.0 / sr)
            centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-10))

            # Heuristic scream score
            scream_score = 0.0
            if rms > 0.05:          # reasonably loud
                if zcr > 0.15:      # lots of zero crossings
                    scream_score += 0.3
                if centroid > 1500:  # high-pitched
                    scream_score += 0.3
                if rms > 0.15:      # very loud
                    scream_score += 0.2
                if centroid > 3000:  # very high-pitched
                    scream_score += 0.1

            results = []
            if scream_score >= 0.5:
                results.append(("scream", round(min(scream_score, 0.95), 3)))

            # Very loud transient = possible gunshot/explosion
            if rms > 0.3 and zcr < 0.1:
                results.append(("gunshot", round(min(rms * 0.8, 0.7), 3)))

            return results

        except Exception:
            return []


# Singleton for the vision loop
_classifier: AudioClassifier | None = None


def get_classifier() -> AudioClassifier:
    global _classifier
    if _classifier is None:
        _classifier = AudioClassifier()
    return _classifier