"""Counter-reasoning gate: is this person's ATTENTION and PATH actually aimed at her, or is he busy?

Why (criminal psychology, behaviour only):
 * Attack cycle - target selection -> surveillance / "interview" -> positioning -> attack. Through every
   stage the offender's attention is FIXED on the target and his path is built around her. A bystander's
   attention and path are built around his own goal (phone, friend, shop, bus). That difference is what we read.
 * Left of Bang (Van Horne & Riley, USMC Combat Hunter): baseline first, react only to ANOMALIES, and only to a
   CLUSTER of indicators. "Facing her", "walking her way", "arms moving" is the baseline of any Indian street.
 * Hazelwood & Burgess approach styles (con / blitz / surprise): all need orientation toward the victim. Blitz
   has almost no lead-up, but it shows as a physical act at arm's range, which is handled separately.
 * Grayson & Stein (1981) and pre-incident-indicator work (de Becker): offenders watch before they act;
   repeated glances at her, then away ("target glancing"), and looking around for witnesses ("scanning").
Counter-evidence (he is OCCUPIED): head bowed at a phone, phone at the ear, talking to someone beside him,
back turned, moving away, or a path that will pass her by more than ~1.3 m.
Never uses appearance, clothing, age, gender, religion, caste or skin colour.
"""
import math
from collections import deque

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6
L_WR, R_WR = 9, 10

WINDOW_S = 3.0              # attention / busy window
MIN_VALID = 4               # readable-head frames needed before attention means anything
MIN_BOX_H = 110             # px: below this the head keypoints are noise -> attention "unknown"
FRONTAL_YAW = 0.22          # |nose - ear midpoint| / ear span below this = looking at the camera
HEAD_DOWN_K = 0.18          # nose less than this * body scale above the shoulder line = head bowed
PHONE_K = 0.45              # wrist within this * body scale of an ear / nose = phone at ear
ATTN_ON = 0.6               # share of the window looking at her = attention on her
BUSY_ON = 0.5
BACK_ON = 0.6
PATH_WINDOW_S = 2.0
PATH_MIN_SAMPLES = 5
AIM_MISS_M = 0.9            # predicted closest pass within this = path aimed at her
PASS_MISS_M = 1.3           # predicted closest pass wider than this = walking past
AIM_HORIZON_S = 6.0
MIN_REL_SPEED = 0.25        # m/s, below = standing
RECEDE_MPS = 0.4            # range opening faster than this = leaving
ENGAGE_DIST_M = 2.5         # standing still, looking at her, this close = con / interview position

STATES = ("unknown", "away", "busy", "passing", "neutral", "watching", "targeted")


def locate(cx_norm: float, dist: float, hfov_deg: float) -> tuple[float, float]:
    """Camera-frame ground position: (lateral m, +right; forward m)."""
    th = (cx_norm - 0.5) * math.radians(hfov_deg)
    return dist * math.sin(th), dist * math.cos(th)


def head_read(kxy, kcf, floor: float, box_h: float) -> dict:
    """One frame of head / hands. frontal = face pointed at the camera; down = bowed (phone / reading)."""
    out = {"valid": False, "frontal": False, "yaw": 0.0, "down": False, "back": False, "phone": False}
    if kxy is None or kcf is None or len(kcf) <= R_SH or box_h < MIN_BOX_H:
        return out

    def ok(i):
        return len(kcf) > i and float(kcf[i]) >= floor

    sh = ok(L_SH) and ok(R_SH)
    eyes = [i for i in (L_EYE, R_EYE) if ok(i)]
    ears = [i for i in (L_EAR, R_EAR) if ok(i)]
    if not ok(NOSE):
        if sh and not eyes:
            out.update(valid=True, back=True)          # shoulders, no face: back to her
        return out
    nx, ny = float(kxy[NOSE][0]), float(kxy[NOSE][1])

    if len(ears) == 2:
        mid = (float(kxy[L_EAR][0]) + float(kxy[R_EAR][0])) / 2.0
        span = abs(float(kxy[L_EAR][0]) - float(kxy[R_EAR][0]))
        yaw = (nx - mid) / span if span >= 2 else None
    elif len(eyes) == 2 and not ears:
        mid = (float(kxy[L_EYE][0]) + float(kxy[R_EYE][0])) / 2.0
        span = abs(float(kxy[L_EYE][0]) - float(kxy[R_EYE][0])) * 2.2   # eye span ~0.45 of ear span
        yaw = (nx - mid) / span if span >= 2 else None
    else:
        ref = ears[0] if ears else (eyes[0] if eyes else None)
        if ref is None:
            return out
        yaw = 1.0 if nx >= float(kxy[ref][0]) else -1.0                # one ear only: profile
    if yaw is None:
        return out

    sw = abs(float(kxy[L_SH][0]) - float(kxy[R_SH][0])) if sh else 0.0
    scale = max(sw, 0.2 * box_h)
    down = False
    if sh:
        sh_y = (float(kxy[L_SH][1]) + float(kxy[R_SH][1])) / 2.0
        down = (sh_y - ny) < HEAD_DOWN_K * scale
        heads = [kxy[i] for i in ears] or [kxy[NOSE]]
        for w in (L_WR, R_WR):
            if ok(w) and float(kxy[w][1]) < sh_y + 0.1 * scale:
                wx, wy = float(kxy[w][0]), float(kxy[w][1])
                if any(math.hypot(wx - float(h[0]), wy - float(h[1])) < PHONE_K * scale for h in heads):
                    out["phone"] = True
    out.update(valid=True, yaw=float(yaw), frontal=abs(yaw) < FRONTAL_YAW, down=down)
    return out


def _fit(samples, t_now: float):
    ts = np.fromiter((s[0] for s in samples), float)
    xs = np.fromiter((s[1] for s in samples), float)
    zs = np.fromiter((s[2] for s in samples), float)
    tm = ts - ts.mean()
    den = float((tm * tm).sum())
    if den < 1e-6:
        return None
    vx = float((tm * (xs - xs.mean())).sum()) / den
    vz = float((tm * (zs - zs.mean())).sum()) / den
    dt = t_now - float(ts.mean())
    return xs.mean() + vx * dt, zs.mean() + vz * dt, vx, vz


class IntentTracker:
    """One per track. update() -> {"state", "attn", "busy", "aimed", "passing", "miss_m", "why"}."""

    def __init__(self, hfov_deg: float = 65.0):
        self.hfov = hfov_deg
        self.head: deque = deque()   # (t, valid, looking, busy, back)
        self.pos: deque = deque()    # (t, lat, fwd)
        self.last: dict = {"state": "unknown", "attn": None, "busy": 0.0, "aimed": False, "passing": False,
                           "miss_m": None, "why": ""}

    def update(self, t: float, kxy, kcf, floor: float, box_h: float, cx_norm: float, dist: float,
               others=(), follow_lvl: int = 0, reapproach: int = 0) -> dict:
        lat, fwd = locate(cx_norm, dist, self.hfov)
        h = head_read(kxy, kcf, floor, box_h)
        talk = False
        if h["valid"] and not h["back"] and abs(h["yaw"]) >= FRONTAL_YAW:
            s = 1.0 if h["yaw"] > 0 else -1.0       # head turned toward image right (+) / left (-)
            talk = any(abs(of - fwd) < 1.2 and 0.3 < (ol - lat) * s < 1.6 for ol, of in others)
        looking = h["valid"] and h["frontal"] and not h["down"]
        self.head.append((t, h["valid"], looking, h["down"] or h["phone"] or talk, h["back"]))
        while self.head and t - self.head[0][0] > WINDOW_S:
            self.head.popleft()
        self.pos.append((t, lat, fwd))
        while self.pos and t - self.pos[0][0] > PATH_WINDOW_S:
            self.pos.popleft()

        valid = [x for x in self.head if x[1]]
        n = len(valid)
        attn = sum(1 for x in valid if x[2]) / n if n >= MIN_VALID else None
        busy = sum(1 for x in valid if x[3]) / n if n >= MIN_VALID else 0.0
        back = sum(1 for x in valid if x[4]) / n if n >= MIN_VALID else 0.0

        aimed = passing = standing = False
        miss = None
        range_rate = 0.0
        if len(self.pos) >= PATH_MIN_SAMPLES and self.pos[-1][0] - self.pos[0][0] >= 1.0:
            f = _fit(self.pos, t)
            if f is not None:
                px, pz, vx, vz = f
                r = math.hypot(px, pz)
                speed = math.hypot(vx, vz)
                range_rate = (px * vx + pz * vz) / r if r > 1e-3 else 0.0
                if speed < MIN_REL_SPEED:
                    standing, miss = True, r
                else:
                    t_cpa = -(px * vx + pz * vz) / (speed * speed)
                    if t_cpa <= 0:
                        passing, miss = True, r                              # already past / opening
                    else:
                        miss = math.hypot(px + vx * t_cpa, pz + vz * t_cpa)
                        aimed = t_cpa <= AIM_HORIZON_S and miss <= AIM_MISS_M
                        passing = miss >= PASS_MISS_M

        why = []
        if follow_lvl >= 2 or reapproach >= 2:
            state = "targeted"
            why.append("keeps returning to her")
        elif back >= BACK_ON or range_rate > RECEDE_MPS:
            state = "away"
        elif busy >= BUSY_ON and (attn is None or attn < 0.4):
            state = "busy"
        elif attn is None:
            state = "passing" if passing else "unknown"
        elif attn >= ATTN_ON and (aimed or (standing and dist <= ENGAGE_DIST_M) or follow_lvl >= 1):
            state = "targeted"
            why.append("eyes on her + path aimed at her" if aimed else "eyes on her, holding position close")
        elif attn >= ATTN_ON:
            state = "watching"
        elif passing:
            state = "passing"
        else:
            state = "neutral"
        self.last = {"state": state, "attn": None if attn is None else round(attn, 2), "busy": round(busy, 2),
                     "aimed": aimed, "passing": passing, "miss_m": None if miss is None else round(miss, 2),
                     "why": ", ".join(why)}
        return self.last