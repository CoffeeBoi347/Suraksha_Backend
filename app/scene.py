"""Scene context: modifies how suspicious we are. NEVER triggers an alert by itself.

  dark  - frame brightness (night / dark alley / poor light = less guardianship, worse detection)
  safe  - user is inside a saved safe zone (home/office) or flipped "safe mode" manually
Privacy: zones live in memory for the connection only; nothing is stored server-side.
"""
import math

import cv2
import numpy as np

DARK_ON = 55.0       # mean gray (0-255) below this -> dark
DARK_OFF = 70.0      # hysteresis: must go above this to leave dark
EMA = 0.2
EVERY_N = 3          # analyse every Nth frame (it's cheap, but no need for all)


def _hav_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class SceneContext:
    def __init__(self):
        self.brightness: float | None = None
        self.dark = False
        self.zones: list[tuple[float, float, float]] = []   # (lat, lon, radius_m)
        self.manual_safe = False
        self._n = 0

    def set_zones(self, zones) -> None:
        out = []
        for z in (zones or [])[:10]:
            try:
                lat, lon, rad = float(z["lat"]), float(z["lon"]), float(z.get("radius_m", 100))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    out.append((lat, lon, max(20.0, min(rad, 1000.0))))
            except (KeyError, TypeError, ValueError):
                continue
        self.zones = out

    def update(self, frame: np.ndarray, gps: dict) -> dict:
        self._n += 1
        if self._n % EVERY_N == 1 or self.brightness is None:
            small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
            b = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
            self.brightness = b if self.brightness is None else EMA * b + (1 - EMA) * self.brightness
            if self.dark and self.brightness > DARK_OFF:
                self.dark = False
            elif not self.dark and self.brightness < DARK_ON:
                self.dark = True
        return {"dark": self.dark, "safe": self.is_safe(gps), "brightness": round(self.brightness or 0.0, 1)}

    def is_safe(self, gps: dict) -> bool:
        if self.manual_safe:
            return True
        lat, lon = gps.get("latitude"), gps.get("longitude")
        if lat is None or lon is None:
            return False
        acc = float(gps.get("accuracy_m", 0.0) or 0.0)
        return any(_hav_m(lat, lon, zl, zo) <= max(zr - min(acc, zr * 0.5), 10.0) for zl, zo, zr in self.zones)