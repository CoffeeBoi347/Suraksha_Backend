import time
from collections import deque

import numpy as np

# COCO keypoints
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12

HISTORY_S = 6.0
FOLLOW_MIN_S = 90.0      
FOLLOW_MAX_DIST = 12.0
FOLLOW_MIN_FRAMES = 10     
APPROACH_FAST = 1.5      
APPROACH_BRISK = 0.7     
NEAR_M = 2.0
CLOSE_M = 3.5
LEVELS = ("NORMAL", "MODERATE", "CRITICAL")


class TrackState:
    __slots__ = ("hist", "first_seen", "last_seen", "prev_wrists", "follow_flagged")

    def __init__(self, now: float):
        self.hist: deque = deque()      
        self.first_seen = now
        self.last_seen = now
        self.prev_wrists: dict = {}
        self.follow_flagged = False


class ThreatEngine:
    """One per WebSocket connection. Not thread-safe — call from the frame loop only."""

    def __init__(self):
        self.tracks: dict[str, TrackState] = {}
        self.audio_scream = 0.0       
        self.audio_ts = 0.0
        self.user_speed_mps = 0.0     

    def set_audio(self, scream_prob: float):
        self.audio_scream = max(0.0, min(1.0, scream_prob))
        self.audio_ts = time.time()

    def set_user_speed(self, mps: float):
        self.user_speed_mps = max(0.0, mps)

    def drop_missing(self, live_keys: set[str], now: float, ttl: float = 8.0):
        for k in list(self.tracks):
            if k not in live_keys and now - self.tracks[k].last_seen > ttl:
                del self.tracks[k]

    def update(self, key: str, now: float, dist_m: float, cx_norm: float,
               box_w_px: float, box_h_px: float, kxy, kconf, kp_conf_floor: float,
               weapon: bool = False) -> dict:
        st = self.tracks.get(key)
        if st is None:
            st = self.tracks[key] = TrackState(now)
        st.last_seen = now
        st.hist.append((now, dist_m, cx_norm, box_h_px))
        while st.hist and now - st.hist[0][0] > HISTORY_S:
            st.hist.popleft()

        signals: dict[str, float] = {}
        reasons: list[str] = []

        approach = 0.0
        if len(st.hist) >= 3:
            t0, d0 = st.hist[0][0], st.hist[0][1]
            dt = now - t0
            if dt > 0.4:
                approach = (d0 - dist_m) / dt
        if approach >= APPROACH_FAST and dist_m < 6.0:
            signals["approach"] = 0.55
            reasons.append(f"closing fast ({approach:.1f} m/s)")
        elif approach >= APPROACH_BRISK and dist_m < CLOSE_M:
            signals["approach"] = 0.30
            reasons.append("walking straight at you")

        if dist_m <= 1.2:
            signals["proximity"] = 0.40
            reasons.append("within arm's reach")
        elif dist_m <= NEAR_M:
            signals["proximity"] = 0.25
            reasons.append("very close")
        elif dist_m <= CLOSE_M:
            signals["proximity"] = 0.12

        if dist_m <= NEAR_M and abs(cx_norm - 0.5) < 0.18:
            signals["blocking"] = 0.20
            reasons.append("directly in your path")

        dwell = now - st.first_seen
        if (dwell >= FOLLOW_MIN_S and dist_m <= FOLLOW_MAX_DIST
                and len(st.hist) >= FOLLOW_MIN_FRAMES and self.user_speed_mps > 0.4):
            signals["following"] = 0.45
            if not st.follow_flagged:
                st.follow_flagged = True
            reasons.append(f"following you for {int(dwell // 60)}m")

        pose, st.prev_wrists = _pose_signal(kxy, kconf, kp_conf_floor, box_w_px, box_h_px, st.prev_wrists)
        if pose > 0:
            signals["pose"] = pose
            reasons.append("aggressive arm motion")

        if weapon:
            signals["weapon"] = 0.70
            reasons.append("possible weapon")

        if self.audio_scream >= 0.6 and now - self.audio_ts < 4.0 and dist_m <= 8.0:
            signals["audio"] = 0.35 * self.audio_scream
            reasons.append("scream detected")

        score = float(np.clip(sum(signals.values()), 0.0, 1.0))

        critical_now = weapon or (signals.get("approach", 0) >= 0.55 and dist_m <= CLOSE_M)
        if critical_now or score >= 0.70:
            level = "CRITICAL"
        elif score >= 0.35:
            level = "MODERATE"
        else:
            level = "NORMAL"

        return {"score": score, "level": level, "signals": signals,
                "reason": ", ".join(reasons[:3]) or "no threat indicators",
                "approach_mps": round(approach, 2), "dwell_s": int(dwell)}


def _pose_signal(kxy, kconf, floor: float, box_w_px: float, box_h_px: float,
                 prev_wrists: dict) -> tuple[float, dict]:
    def kp(i):
        if kxy is None or i >= len(kxy):
            return None
        if kconf is not None and kconf[i] < floor:
            return None
        p = kxy[i]
        return None if (p[0] <= 0 and p[1] <= 0) else p

    score, now_wrists = 0.0, {}
    ls, rs = kp(L_SH), kp(R_SH)
    sh_w = float(np.linalg.norm(ls - rs)) if ls is not None and rs is not None else max(box_w_px * 0.4, 1.0)

    for sh_i, el_i, wr_i in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)):
        sh, el, wr = kp(sh_i), kp(el_i), kp(wr_i)
        if wr is None:
            continue
        now_wrists[wr_i] = wr
        if sh is not None and el is not None and wr[1] < sh[1]:
            if float(np.linalg.norm(wr - sh)) / sh_w > 1.3:   
                score += 0.22
        if wr_i in prev_wrists:
            if float(np.linalg.norm(wr - prev_wrists[wr_i])) / sh_w > 0.8: 
                score += 0.28

    if box_w_px > box_h_px * 1.2 and kp(L_HIP) is not None:
        score += 0.12   # lunging / wide stance

    return min(score, 0.6), now_wrists
