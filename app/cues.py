"""Behaviour cues for the Suraksha vision HUD. Pure functions/classes: no models, no network, unit-testable.

Research behind the cues:
- Offender approach styles (Hazelwood & Burgess): BLITZ, SURPRISE, CON.
- Pre-attack indicators judged against the local baseline ("Left of Bang", Van Horne & Riley 2014):
  target glancing, scanning for witnesses, closing distance. What everyone around does is not a cue.
- Counter-surveillance TEDD (time, environment, distance, demeanour): a follower keeps showing up across
  her stops and turns; one coincidence is not following.
- Routine activity theory (Cohen & Felson 1979): risk rises with no capable guardian around (isolated, dark).
- Indian patterns: chain/bag snatching by two-wheeler pillion riders, harassment by circling bikes, groping in
  crowds; NCRB: cruelty by husband/relatives is the largest category of crimes against women (so home != safe).
"""
from __future__ import annotations

import math
import os
import re
import statistics
import unicodedata
from collections import deque

import cv2
import numpy as np

# ---------------- COCO keypoints ----------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12

KP_CONF = 0.4
FACE_KP_CONF = 0.5
STRIKE_KP_CONF = float(os.getenv("STRIKE_KP_CONF", "0.45"))
STRIKE_SPEED_MID = float(os.getenv("STRIKE_SPEED_MID", "4.0"))   # shoulder-widths / s
STRIKE_SPEED_HI = float(os.getenv("STRIKE_SPEED_HI", "7.0"))
SWING_PATH = float(os.getenv("SWING_PATH", "4.0"))             # shoulder-widths of wrist travel in the window
POSE_WINDOW_S = 0.8
REACH_ELBOW_DEG = 155.0
FACE_MIN_BOX_H = 110                                           # px; smaller faces give noise keypoints

SHOULDER_WIDTH_M = 0.40      # between COCO shoulder keypoints
TORSO_M = 0.48               # shoulder midpoint -> hip midpoint
PERSON_HEIGHT_M = 1.65
SH_PER_TORSO = SHOULDER_WIDTH_M / TORSO_M


def kp(kxy, kconf, i: int, thr: float = KP_CONF):
    if kxy is None or i >= len(kxy):
        return None
    if kconf is not None and (i >= len(kconf) or float(kconf[i]) < thr):
        return None
    p = np.asarray(kxy[i], dtype=float)
    return None if (p[0] <= 0 and p[1] <= 0) else p


def conf_ok(kconf, i: int, thr: float) -> bool:
    return kconf is None or (i < len(kconf) and float(kconf[i]) >= thr)


def _mid(a, b):
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def body_metrics(kxy, kconf) -> tuple[float, float]:
    """(shoulder width px, torso length px); 0 when not measurable."""
    ls, rs = kp(kxy, kconf, L_SH), kp(kxy, kconf, R_SH)
    sh = float(np.linalg.norm(ls - rs)) if ls is not None and rs is not None else 0.0
    smid, hmid = _mid(ls, rs), _mid(kp(kxy, kconf, L_HIP), kp(kxy, kconf, R_HIP))
    torso = float(np.linalg.norm(smid - hmid)) if smid is not None and hmid is not None else 0.0
    return sh, torso


def body_scale(kxy, kconf, bh: float) -> float:
    """Shoulder-width-equivalent px that does not collapse when the person turns side-on.
    Foreshortening only ever shrinks a body segment on screen, so the largest equivalent is the safest."""
    sh, torso = body_metrics(kxy, kconf)
    return max(sh, torso * SH_PER_TORSO, 0.2 * bh, 1.0)


# ---------------- distance ----------------

def estimate_distance(kxy, kconf, box_h_px: float, w_img: int, h_img: int, hfov_deg: float,
                      box_w_px: float | None = None) -> float:
    """Metres to a person. Shoulder width, torso length and full height each read too FAR when the person is
    side-on, leaning, sitting or cut off by the frame edge - never too near - so the smallest sane one wins.
    (Old rule used shoulders whenever visible: a side-on man 3 m away read as 15-20 m.)"""
    f = w_img / (2 * math.tan(math.radians(hfov_deg) / 2))     # square pixels: fy == fx
    sh, torso = body_metrics(kxy, kconf)
    est = [f * PERSON_HEIGHT_M / max(box_h_px, 1.0)]
    if 14.0 < torso <= 0.85 * box_h_px:
        est.append(f * TORSO_M / torso)
    if sh > 10.0 and (box_w_px is None or sh <= 0.95 * box_w_px):
        est.append(f * SHOULDER_WIDTH_M / sh)
    return float(np.clip(min(est), 0.4, 25.0))


class Median3:
    """3-sample median: kills single-frame keypoint glitches that fake a fast approach."""

    def __init__(self):
        self.v: deque = deque(maxlen=3)

    def __call__(self, x: float) -> float:
        self.v.append(x)
        return float(statistics.median(self.v))


# ---------------- pose ----------------

def _elbow_angle(sh, el, wr) -> float | None:
    if sh is None or el is None or wr is None:
        return None
    a, b = sh - el, wr - el
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1 or nb < 1:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def pose_signals(hist: deque, kxy, kconf, now: float, bh: float) -> tuple[float, str]:
    """Per-frame body-language score 0-1 from wrist motion and arm posture.
    `now` must be the frame's capture/arrival time, not when inference finished."""
    if kxy is None:
        return 0.0, ""
    ls, rs, nose = kp(kxy, kconf, L_SH), kp(kxy, kconf, R_SH), kp(kxy, kconf, NOSE)
    els = (kp(kxy, kconf, L_EL), kp(kxy, kconf, R_EL))
    wrists = (kp(kxy, kconf, L_WR), kp(kxy, kconf, R_WR))
    hip_mid = _mid(kp(kxy, kconf, L_HIP), kp(kxy, kconf, R_HIP))
    scale = body_scale(kxy, kconf, bh)

    # wrists relative to the shoulder midpoint: her head turning / his walking don't look like a punch
    rel = (None, None)
    if ls is not None and rs is not None:
        mid = (ls + rs) / 2.0
        rel = tuple((w - mid) if (w is not None and conf_ok(kconf, i, STRIKE_KP_CONF)) else None
                    for w, i in zip(wrists, (L_WR, R_WR)))
    hist.append((now, rel, scale))
    while hist and now - hist[0][0] > POSE_WINDOW_S:
        hist.popleft()

    score, why = 0.0, ""

    def bump(s: float, w: str):
        nonlocal score, why
        if s > score:
            score, why = s, w

    # 1) fast wrist step (punch / slap / grab lunge): newest step only
    for side in (0, 1):
        pts = [(t, w[side], s) for t, w, s in hist if w[side] is not None]
        if len(pts) >= 2 and pts[-1][0] == now:
            (t0, p0, s0), (t1, p1, s1) = pts[-2], pts[-1]
            dt = t1 - t0
            if 0.03 <= dt <= 0.5:
                speed = float(np.linalg.norm(p1 - p0)) / dt / max((s0 + s1) / 2, 1.0)
                if speed >= STRIKE_SPEED_HI:
                    bump(0.85, "Strike-like arm motion")
                elif speed >= STRIKE_SPEED_MID:
                    bump(0.6, "Fast arm motion")

    # 2) hand(s) above head height (weak alone: waving, stretching)
    raised = [w is not None and nose is not None and w[1] < nose[1] - 0.1 * bh for w in wrists]
    if any(raised):
        if score >= 0.6:
            score = min(1.0, score + 0.1)       # raised + fast = overhead strike / swing
        bump(0.3, "Arm raised")
    if all(raised):
        bump(0.5, "Both arms raised")

    # 3) straight arm held out at chest height = reaching / grabbing.
    #    (Old rule: wrist far from shoulder / shoulder width -> fired on anyone side-on with arms hanging.)
    for sh, el, w in ((ls, els[0], wrists[0]), (rs, els[1], wrists[1])):
        ang = _elbow_angle(sh, el, w)
        if ang is None or ang < REACH_ELBOW_DEG:
            continue
        not_hanging = w[1] < (hip_mid[1] - 0.35 * scale if hip_mid is not None else sh[1] + 0.6 * scale)
        if not_hanging and float(np.linalg.norm(w - sh)) >= 1.1 * scale:
            bump(0.35, "Arm reaching out")
            break

    # 4) repeated large wrist travel = swinging an object / flailing
    for side in (0, 1):
        seq = [(t, w[side], s) for t, w, s in hist if w[side] is not None]
        if len(seq) >= 4:
            path = sum(float(np.linalg.norm(b - a)) for (ta, a, _), (tb, b, _) in zip(seq, seq[1:]) if tb - ta <= 0.3)
            if path / max(float(np.mean([s for _, _, s in seq])), 1.0) >= SWING_PATH:
                bump(0.6, "Swinging arm motion")
    return round(score, 3), why


# ---------------- objects & vehicles (COCO ids) ----------------
OBJECT_CLASSES = {34: "bat", 43: "knife", 76: "scissors", 56: "chair", 39: "bottle", 40: "glass", 75: "vase",
                  38: "racket", 36: "skateboard", 25: "umbrella",
                  1: "bicycle", 3: "motorcycle", 2: "car", 5: "bus", 7: "truck"}
# fork / book dropped: carried constantly, almost never used in attacks -> pure noise
OBJECT_MIN_CONF = {"knife": 0.30, "scissors": 0.45, "bat": 0.35, "chair": 0.25, "bottle": 0.30, "glass": 0.40,
                   "vase": 0.35, "racket": 0.35, "skateboard": 0.35, "umbrella": 0.35,
                   "bicycle": 0.35, "motorcycle": 0.35, "car": 0.35, "bus": 0.40, "truck": 0.40}
# Labels beyond COCO (gun, axe, hammer, stick, brick) only appear with custom-trained weights. Add a new weapon type by
# adding it here + in OBJECT_MIN_CONF (+ ALIASES if the weights name it differently). No other code changes.
EXTRA_HAND = {"gun", "axe", "hammer", "stick", "brick"}
OBJECT_MIN_CONF.update({"gun": 0.30, "axe": 0.30, "hammer": 0.35, "stick": 0.35, "brick": 0.35})
SHARP = {"knife", "scissors", "gun", "axe"}     # dangerous when merely in hand (no need to be raised)
ALIASES = {"baseball bat": "bat", "tennis racket": "racket", "wine glass": "glass", "cup": "glass",
           "pistol": "gun", "handgun": "gun", "firearm": "gun", "revolver": "gun", "rifle": "gun",
           "shotgun": "gun", "weapon": "knife", "machete": "knife", "dagger": "knife", "blade": "knife",
           "rod": "stick", "pipe": "stick", "club": "stick", "baton": "stick"}
TWO_WHEELERS = {"bicycle", "motorcycle"}
VEHICLES = {"car", "bus", "truck"}
HAND_OBJECTS = (set(OBJECT_CLASSES.values()) | EXTRA_HAND) - TWO_WHEELERS - VEHICLES
DRINKWARE = {"bottle", "glass"}


def _area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _inter(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def canon(name: str) -> str | None:
    """Model class name -> our label, or None if we don't use it. Works for COCO and custom weights alike."""
    n = str(name).lower().strip().replace("_", " ")
    n = ALIASES.get(n, n)
    return n if n in OBJECT_MIN_CONF else None


def class_ids(names: dict, labels: set | None = None) -> list[int]:
    """Class ids of a loaded model (model.names) that map to our labels (optionally only those in `labels`)."""
    return [i for i, n in names.items() if (c := canon(n)) and (labels is None or c in labels)]


def parse_objects(result, dx: float = 0.0, dy: float = 0.0, conf_scale: float = 1.0) -> list:
    """Ultralytics result -> [(label, conf, (x1, y1, x2, y2))] in frame coordinates."""
    out = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return out
    names = getattr(result, "names", None)
    for b in boxes:
        cid = int(b.cls[0])
        label = canon(names[cid]) if isinstance(names, dict) and cid in names else OBJECT_CLASSES.get(cid)
        conf = float(b.conf[0])
        if label and conf >= OBJECT_MIN_CONF[label] * conf_scale:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append((label, conf, (x1 + dx, y1 + dy, x2 + dx, y2 + dy)))
    return out


def merge_objects(objs: list) -> list:
    out: list = []
    for o in sorted(objs, key=lambda o: -o[1]):
        if all(not (o[0] == k[0] and iou(o[2], k[2]) > 0.5) for k in out):
            out.append(o)
    return out


def _pt_rect_dist(p, r) -> float:
    return math.hypot(max(r[0] - p[0], 0.0, p[0] - r[2]), max(r[1] - p[1], 0.0, p[1] - r[3]))


def match_object(objs: list, box, kxy, kconf) -> dict | None:
    """Best hand-held object for this person -> {label, conf, held, raised} or None.
    held   = a wrist is on the object (not merely near the person)
    raised = held above the shoulder line or with the holding hand cocked above it; drinking from a bottle
             and an open umbrella overhead don't count."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    scale = body_scale(kxy, kconf, bh)
    sh_pts = [p for p in (kp(kxy, kconf, L_SH), kp(kxy, kconf, R_SH)) if p is not None]
    sh_y = float(np.mean([p[1] for p in sh_pts])) if sh_pts else y1 + 0.2 * bh
    nose = kp(kxy, kconf, NOSE)
    wrists = [w for w in (kp(kxy, kconf, L_WR), kp(kxy, kconf, R_WR)) if w is not None]
    best = None
    for label, conf, r in objs:
        if label not in HAND_OBJECTS:
            continue
        cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
        if label == "chair" and cy > y1 + 0.5 * bh:            # chair low in the box = sitting on it
            continue
        if not (x1 - 0.35 * bw <= cx <= x2 + 0.35 * bw and y1 - 0.3 * bh <= cy <= y2 + 0.1 * bh):
            continue
        held, raised = False, False
        if wrists:
            hand = min(wrists, key=lambda w: _pt_rect_dist(w, r))
            held = _pt_rect_dist(hand, r) <= 0.35 * scale
            if held:
                above = cy < sh_y - 0.05 * bh or hand[1] < sh_y - 0.05 * bh
                drinking = label in DRINKWARE and nose is not None and _pt_rect_dist(nose, r) <= 0.5 * scale
                open_umbrella = label == "umbrella" and (r[2] - r[0]) > 1.3 * (r[3] - r[1])
                raised = above and not (drinking or open_umbrella)
        cand = (held, raised, conf, label)
        if best is None or cand[:3] > best[:3]:
            best = cand
    if best is None:
        return None
    return {"label": best[3], "conf": best[2], "held": best[0], "raised": best[1], "uncertain": not wrists}


def ride_flags(person_boxes: list, objs: list) -> list[tuple[str, bool]]:
    """Per person: (vehicle, pillion). 'two_wheeler' = sitting on a bike (box overlaps it and is centred over it);
    'car'/'bus'/'truck' = at or in one. pillion = 2+ people on the same bike."""
    flags = [("", False)] * len(person_boxes)
    for label, _c, r in objs:
        if label in TWO_WHEELERS:
            on = [i for i, p in enumerate(person_boxes)
                  if _inter(p, r) >= 0.2 * _area(p) and r[0] <= (p[0] + p[2]) / 2 <= r[2]]
            for i in on:
                flags[i] = ("two_wheeler", len(on) >= 2 or flags[i][1])
    for label, _c, r in objs:
        if label in VEHICLES:
            for i, p in enumerate(person_boxes):
                if not flags[i][0] and _inter(p, r) >= 0.3 * _area(p):
                    flags[i] = (label, False)
    return flags


# ---------------- gaze: target glancing & scanning for witnesses ----------------

def head_state(kxy, kconf, bh: float) -> str | None:
    """'front' (head towards the camera, within ~40 deg), 'left'/'right' (turned towards that side of the IMAGE),
    'back', or None when unknown."""
    if kxy is None or bh < FACE_MIN_BOX_H:
        return None
    nose = kp(kxy, kconf, NOSE, FACE_KP_CONF)
    le, re_ = kp(kxy, kconf, L_EYE, FACE_KP_CONF), kp(kxy, kconf, R_EYE, FACE_KP_CONF)
    if nose is None and le is None and re_ is None:
        return "back" if (kp(kxy, kconf, L_SH) is not None and kp(kxy, kconf, R_SH) is not None) else None
    if nose is None:
        return None
    if le is not None and re_ is not None and abs(le[0] - re_[0]) >= 3:
        r = (nose[0] - (le[0] + re_[0]) / 2) / abs(le[0] - re_[0])
        if abs(r) <= 0.3:
            return "front"
        return "right" if r > 0 else "left"
    ref = kp(kxy, kconf, L_EAR)
    ref = ref if ref is not None else kp(kxy, kconf, R_EAR)
    ref = ref if ref is not None else (le if le is not None else re_)
    if ref is None:
        return None
    return "right" if nose[0] > ref[0] else "left"


class GazeTracker:
    """Debounced head-orientation history of one track."""

    def __init__(self):
        self.last_raw: str | None = None
        self.state: str | None = None
        self.changes: deque = deque()          # (t, state)

    def update(self, t: float, s: str | None) -> None:
        if s is not None and s == self.last_raw and s != self.state:   # 2 consecutive frames to commit
            self.state = s
            self.changes.append((t, s))
        self.last_raw = s
        while self.changes and t - self.changes[0][0] > 30.0:
            self.changes.popleft()

    def glances(self, t: float, window: float = 20.0) -> int:
        """Separate looks AT her (non-front -> front) within the window."""
        n, prev = 0, None
        for ts, s in self.changes:
            if t - ts <= window and s == "front" and prev not in (None, "front"):
                n += 1
            prev = s
        return n

    def scans(self, t: float, window: float = 8.0) -> int:
        """Left/right head swings within the window (looking around, e.g. for witnesses)."""
        sides = [s for ts, s in self.changes if t - ts <= window and s in ("left", "right")]
        return sum(1 for a, b in zip(sides, sides[1:]) if a != b)


class ApproachEpisodes:
    """Separate entries into her personal space (<= near m) after backing off (>= far m): probing / CON approach."""

    def __init__(self, near: float = 2.0, far: float = 3.5, window: float = 120.0):
        self.near, self.far, self.window = near, far, window
        self.inside = False
        self.entries: deque = deque()

    def update(self, t: float, dist: float) -> int:
        if not self.inside and dist <= self.near:
            self.inside = True
            self.entries.append(t)
        elif self.inside and dist >= self.far:
            self.inside = False
        while self.entries and t - self.entries[0] > self.window:
            self.entries.popleft()
        return len(self.entries)


def _angdiff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


class FollowTest:
    """TEDD check: did this person keep pace with her, then stop when she stopped or turn when she turned?
    One match is weak (a red light stops everyone), two is strong. An event matched by 3+ people at once
    (signal, crossing, bus stop) is ignored. Works for people she can see (ahead / beside); people behind
    her are covered by re-ID 'returned' persistence."""
    WALK_MPS, STILL_MPS = 0.7, 0.25

    def __init__(self):
        self.events: deque = deque()            # (t_event, kind)
        self.walk_since: float | None = None
        self.still_since: float | None = None
        self.walked = False
        self.heading_ref: float | None = None
        self.turn_since: float | None = None
        self.hist: dict = {}                    # track -> deque[(t, dist, |approach|)]
        self.matched: dict = {}                 # track -> set[t_event]
        self.by_event: dict = {}                # t_event -> set[track]

    def _event(self, te: float, kind: str) -> None:
        self.events.append((te, kind))
        self.by_event.setdefault(te, set())

    def user(self, t: float, speed: float, heading: float | None = None) -> None:
        if speed >= self.WALK_MPS:
            self.still_since = None
            self.walk_since = self.walk_since if self.walk_since is not None else t
            if t - self.walk_since >= 8.0:
                self.walked = True
        elif speed <= self.STILL_MPS:
            self.walk_since = None
            self.still_since = self.still_since if self.still_since is not None else t
            if self.walked and t - self.still_since >= 4.0:
                self._event(self.still_since, "stop")
                self.walked = False
        if heading is not None and speed >= self.WALK_MPS:
            if self.heading_ref is None:
                self.heading_ref = heading
            d = _angdiff(heading, self.heading_ref)
            if abs(d) >= 60.0:
                self.turn_since = self.turn_since if self.turn_since is not None else t
                if t - self.turn_since >= 4.0:
                    self._event(self.turn_since, "turn")
                    self.heading_ref, self.turn_since = heading, None
            else:
                self.turn_since = None
                self.heading_ref = (self.heading_ref + 0.1 * d) % 360.0      # follow slow drift
        while self.events and t - self.events[0][0] > 90.0:
            self.events.popleft()

    def track(self, key, t: float, dist: float, approach: float) -> None:
        h = self.hist.setdefault(key, deque())
        h.append((t, dist, abs(approach)))
        while h and t - h[0][0] > 60.0:
            h.popleft()
        for te, kind in self.events:
            if t < te + 5.0 or te in self.matched.get(key, ()):
                continue
            before = [x for x in h if te - 15.0 <= x[0] <= te - 1.0]
            after = [x for x in h if te + 2.0 <= x[0] <= te + 30.0]
            if len(before) < 3 or len(after) < 3:
                continue
            if before[-1][0] - before[0][0] < 5.0 or after[-1][0] - after[0][0] < 3.0:
                continue
            d0 = statistics.median(x[1] for x in before)
            if d0 > 10.0 or statistics.median(x[2] for x in before) > 0.35:
                continue                                     # wasn't keeping pace with her
            if max(x[1] for x in after) > d0 + 2.5:
                continue                                     # walked on / left: normal
            if kind == "turn" and statistics.median(x[2] for x in after) > 0.45:
                continue
            self.matched.setdefault(key, set()).add(te)
            self.by_event[te].add(key)

    def level(self, key) -> int:
        return sum(1 for te in self.matched.get(key, ()) if len(self.by_event.get(te, ())) <= 2)

    def prune(self, t: float) -> None:
        for k in [k for k, h in self.hist.items() if not h or t - h[-1][0] > 600.0]:
            self.hist.pop(k, None)
            self.matched.pop(k, None)


class AudioBaseline:
    """Scream-probability baseline. In a noisy scene (crowd, festival, match, protest) the classifier stays high for
    everyone's shouting; a real scream is a JUMP above that background. update() -> (value to act on, noisy)."""

    def __init__(self, window: float = 20.0, min_samples: int = 6, noisy_at: float = 0.4, jump: float = 0.3):
        self.window, self.min_samples, self.noisy_at, self.jump = window, min_samples, noisy_at, jump
        self.h: deque = deque()

    def update(self, t: float, v: float) -> tuple[float, bool]:
        while self.h and t - self.h[0][0] > self.window:
            self.h.popleft()
        bg = [x for ts, x in self.h if t - ts >= 1.0]            # the last second is the event, not the background
        self.h.append((t, v))
        if len(bg) < self.min_samples:
            return v, False
        base = float(statistics.median(bg))
        if base >= self.noisy_at:
            return (v if v - base >= self.jump else 0.0), True
        return v, False


class GestureHold:
    """Nominates a person for the offensive-gesture check (middle finger, crude signs). Pose keypoints stop at the
    wrist, so this cannot see fingers; it only decides WHEN to ask Gemini to look. Two paths:
      A) normal range (<= max_dist): forearm raised, hand between chest and head, not at the face (phone call,
         hair, drinking), held steady (a wave is moving), face turned to her.
      B) close range (<= close_dist, selfie / arm's-reach distance): shoulders, elbows and wrists are cut off by the
         frame, so pose cannot place the hand at all. A person this close, facing her for close_hold_s, is checked
         once instead; the hand is big in the picture, so Gemini reads it directly.
    One nomination per raised hand / per close encounter (again after 30 s if it continues)."""

    def __init__(self, hold_s: float = 0.5, max_dist: float = 4.0, refractory: float = 15.0, rearm_s: float = 1.0,
                 close_dist: float = 1.6, close_hold_s: float = 1.0, again_s: float = 30.0):
        self.hold_s, self.max_dist, self.refractory, self.rearm_s = hold_s, max_dist, refractory, rearm_s
        self.close_dist, self.close_hold_s, self.again_s = close_dist, close_hold_s, again_s
        self.pts: deque = deque()
        self.since: float | None = None
        self.close_since: float | None = None
        self.down_since: float | None = None
        self.armed = True
        self.last_fire = -1e9

    @staticmethod
    def _raised(kxy, kconf, bh: float):
        ls, rs, nose = kp(kxy, kconf, L_SH), kp(kxy, kconf, R_SH), kp(kxy, kconf, NOSE)
        scale = body_scale(kxy, kconf, bh)
        best = None
        for sh, el, wr, idx in ((ls, kp(kxy, kconf, L_EL), kp(kxy, kconf, L_WR), L_WR),
                                (rs, kp(kxy, kconf, R_EL), kp(kxy, kconf, R_WR), R_WR)):
            if sh is None or el is None or wr is None or not conf_ok(kconf, idx, STRIKE_KP_CONF):
                continue
            if wr[1] > el[1] - 0.4 * scale:                                   # forearm not raised
                continue
            if wr[1] < (nose[1] - 0.3 * scale if nose is not None else sh[1] - 1.2 * scale):
                continue                                                      # above the head: cheering, stretching
            if nose is not None and float(np.hypot(*(wr - nose))) < 0.8 * scale:
                continue                                                      # at the face: phone, hair, drinking
            if best is None or wr[1] < best[0][1]:
                best = (wr, wr - sh, scale)
        return best

    @staticmethod
    def _any_wrist(kxy, kconf):
        """Best guess at where a hand is even when keypoint confidence is poor (None if there is no clue)."""
        c = [w for w in (kp(kxy, kconf, L_WR, 0.15), kp(kxy, kconf, R_WR, 0.15)) if w is not None]
        return min(c, key=lambda w: w[1]) if c else None

    def update(self, t: float, kxy, kconf, bh: float, dist: float, facing: bool):
        """-> (candidate, wrist_xy or None). Stays True while the situation holds and no check has been made."""
        near = bool(facing and dist <= self.close_dist)
        hit = self._raised(kxy, kconf, bh) if (kxy is not None and facing and dist <= self.max_dist) else None
        if hit is None and not near:
            self.pts.clear()
            self.since = self.close_since = None
            self.down_since = self.down_since if self.down_since is not None else t
            if t - self.down_since >= self.rearm_s:
                self.armed = True
            return False, None
        self.down_since = None
        ready = (self.armed and t - self.last_fire >= self.refractory) or t - self.last_fire >= self.again_s

        held = False
        if hit is not None:                                                   # path A: raised hand, steady
            wr, rel, scale = hit
            self.pts.append((t, rel, scale))
            while self.pts and t - self.pts[0][0] > 2 * self.hold_s:
                self.pts.popleft()
            self.since = self.since if self.since is not None else t
            win = [p for p in self.pts if t - p[0] <= self.hold_s]
            if t - self.since >= self.hold_s and len(win) >= 3:
                extent = np.ptp(np.array([p[1] for p in win]), axis=0)        # full range of hand movement
                held = float(np.linalg.norm(extent)) / max(float(np.mean([p[2] for p in win])), 1.0) <= 0.6
        else:
            self.pts.clear()
            self.since = None
        if held and ready:
            return True, hit[0]

        if near:                                                              # path B: selfie / arm's-reach range
            self.close_since = self.close_since if self.close_since is not None else t
            if t - self.close_since >= self.close_hold_s and ready:
                return True, self._any_wrist(kxy, kconf)
        else:
            self.close_since = None
        return False, None

    def consume(self, t: float) -> None:
        """Call when the check was actually sent."""
        self.armed, self.last_fire = False, t


def hand_crop(frame: np.ndarray, wrist, bh: float, out: int = 224):
    """Upscaled close-up around the hand (it extends beyond the wrist keypoint) so fingers are legible."""
    if wrist is None:
        return None
    h, w = frame.shape[:2]
    side = int(max(48.0, 0.30 * bh))
    cx, cy = float(wrist[0]), float(wrist[1]) - 0.08 * bh
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], (out, out), interpolation=cv2.INTER_CUBIC)


class WeaponEvidence:
    """Is this person holding that object? Detector confidence summed over a short window, one sample per detection
    pass. A weak frame alone never counts; several weak ones or one strong one do; motion or a raised object
    (corroboration) lowers the bar. Replaces 'N hits in a row', which blur and flicker kept resetting."""

    def __init__(self, window: float = 2.5, strong: float = 0.45, corroborated: float = 0.18):
        self.window, self.strong, self.corroborated = window, strong, corroborated
        self.h: dict = {}

    def add(self, t: float, label: str, conf: float) -> None:
        d = self.h.setdefault(label, deque())
        if d and d[-1][0] == t:
            d[-1] = (t, max(d[-1][1], conf))
        else:
            d.append((t, conf))

    def score(self, t: float, label: str) -> float:
        d = self.h.get(label)
        if d is None:
            return 0.0
        while d and t - d[0][0] > self.window:
            d.popleft()
        return float(sum(c for _, c in d))

    def trusted(self, t: float, label: str, corroborated: bool = False) -> bool:
        return self.score(t, label) >= (self.corroborated if corroborated else self.strong)


def wrist_boxes(items: list, w_img: int, h_img: int, frac: float = 0.40, lo: int = 64, hi: int = 224) -> list:
    """Square boxes around each visible wrist of the given people (dicts with kxy, kcf, bh). An object in a hand is
    small in the full frame but large in this crop, which is what makes a far knife detectable."""
    out = []
    for it in items:
        side = float(min(hi, max(lo, frac * it["bh"])))
        for idx in (L_WR, R_WR):
            wr = kp(it["kxy"], it["kcf"], idx, 0.3)
            if wr is None:
                continue
            x1, y1 = max(0.0, wr[0] - side / 2), max(0.0, wr[1] - side / 2)
            x2, y2 = min(float(w_img), wr[0] + side / 2), min(float(h_img), wr[1] + side / 2)
            if x2 - x1 >= 24 and y2 - y1 >= 24:
                out.append((x1, y1, x2, y2))
    return out


def active_wrist(kxy, kconf):
    """The raised-most visible wrist (the hand most likely doing something), or None."""
    cands = [w for w, i in ((kp(kxy, kconf, L_WR, STRIKE_KP_CONF), L_WR), (kp(kxy, kconf, R_WR, STRIKE_KP_CONF), R_WR))
             if w is not None]
    return min(cands, key=lambda w: w[1]) if cands else None


class CrowdBaseline:
    """Baseline: if most people in view show the same arm motion (dance, procession, cricket, boarding scrum),
    arm motion is normal here and not a threat cue by itself."""

    def __init__(self, window: float = 4.0):
        self.window = window
        self.hist: deque = deque()

    def update(self, t: float, pose_scores: list) -> bool:
        self.hist.append((t, len(pose_scores), sum(1 for p in pose_scores if p >= 0.3)))
        while self.hist and t - self.hist[0][0] > self.window:
            self.hist.popleft()
        if len(self.hist) < 4:
            return False
        people = sum(n for _, n, _ in self.hist)
        active = sum(a for _, _, a in self.hist)
        return active / len(self.hist) >= 4 and active >= 0.6 * people


def pre_attack_cues(*, dist: float, approach: float, isolated: bool, dark: bool, glances: int, scans: int,
                    reapproach: int, two_wheeler: bool, pillion: bool, near_s: float,
                    company: bool = False) -> list[tuple[int, str, str]]:
    """Pre-attack indicators fuse() does not model -> [(rank, tag, reason)]; rank 0 = worth a Gemini look only."""
    out: list[tuple[int, str, str]] = []
    if not company:
        if scans >= 3 and dist <= 6.0 and (glances >= 2 or approach >= 0.4):
            out.append((1, "witness_check", "Looking around, then at you"))
        if glances >= 3 and dist <= 8.0:
            out.append((1 if (isolated or dark) else 0, "target_glance", "Keeps watching you"))
        if reapproach >= 2 and dist <= 3.5:
            out.append((1, "re_approach", "Came close to you again"))
    if two_wheeler and pillion and dist <= 3.5 and near_s >= 2.0:
        out.append((1 if (isolated or dark or approach >= 0.3) else 0, "two_wheeler",
                    "Two-wheeler pair close beside you"))
    return out


# ---------------- speech (OTHER people's speech; text is never stored) ----------------
_W = r"[\w\u0900-\u0DFF]"     # word chars incl. Indic vowel signs; plain \w missed them ("माल" matched "मालूम")
_YOU = r"(?:tujhe|tujhko|tereko|tumhe|tumhein|tumko|usko|isko|use|ise)"
_YOU_HI = "(?:तुझे|तुझको|तुम्हें|तुमको|उसे|इसे|उसको|इसको)"
_YOU_BN = r"(?:toke|tomake|tor)"

# level 2: threats + coercion scripts ("come quietly", "don't scream"). Bare "dekh lunga"/"utha lunga" are
# everyday ("I'll handle it" / "I'll pick it up"), so they need a 'you' object.
_THREAT = (
    rf"{_YOU} (?:utha|uthwa) (?:lunga|lenge|lega|dunga|denge|le jaunga|ke le jaunga)",
    rf"{_YOU} dekh (?:lunga|lenge)", rf"{_YOU} (?:maar|mar) (?:dunga|denge|dalunga|daalunga|dalenge|daalenge)",
    rf"{_YOU} (?:nahi )?(?:chhod|chod)(?:u|oo)nga", r"jaan se (?:maar|mar)\w*", r"kahin ka nahi (?:chhod|chod)\w*",
    r"(?:acid|tezaab|tejab) (?:daal|dal|phek|fek|phenk)\w*", r"(?:chaku|chaaku|knife) (?:maar|ghusa|ghus)\w*",
    r"kidnap\w*", r"(?:gaadi|gadi) (?:me|mein|main) baith\w*", r"baith ja (?:gaadi|gadi)",
    r"chup ?chaap chal\w*", r"(?:aawaaz|awaaz|awaz|aawaz) (?:mat|nahi) nikal\w*", r"chilla(?:na|yi|ya)? mat",
    r"i(?:'ll| will) kill", r"kill you", r"get in the (?:car|van|auto|cab)", r"don'?t scream", r"come quietly",
    r"shut up and (?:come|walk|move)", r"give me your (?:phone|bag|chain|purse)",
    rf"{_YOU_HI} (?:उठा|उठवा) (?:लूंगा|लूँगा|लेंगे)", rf"{_YOU_HI} देख (?:लूंगा|लूँगा|लेंगे)",
    rf"{_YOU_HI} मार (?:दूंगा|दूँगा|डालूंगा|डालूँगा|देंगे)", "जान से मार", "(?:तेजाब|तेज़ाब|एसिड) (?:डाल|फेंक)",
    "चुपचाप चल", "आवाज़? मत", "चिल्लाना मत", "गाड़ी में बैठ",
    rf"{_YOU_BN} (?:tule nebo|tule niye jabo|mere debo|mere felbo)", r"mere felbo", r"chechabi na",
    r"chup kore chol", r"gari ?te oth", "(?:তোকে|তোমাকে) (?:তুলে নেব|মেরে দেব|মেরে ফেলব)", "মেরে ফেলব",
    "চেঁচাবি না", "চুপ করে চল",
    # ---- additional Hindi threat patterns ----
    rf"{_YOU} (?:rakh|rakhun|rakh lu)(?:unga|enge|lenge)\b",           # "I'll keep you" (abduction)
    r"(?:zinda|jeeta) nahi (?:chhod|chod)\w*",                         # "won't let you live"
    r"(?:kapde|kapda) phad\w*",                                        # "tear your clothes"
    rf"{_YOU} (?:khatam|khatm) kar (?:dunga|denge)",                   # "I'll finish you"
    r"(?:nanga|nangi) kar (?:dunga|denge|dalenge|daalunga)",            # strip threat
    r"(?:bahar|baahar) (?:mat|nahi) niklna",                           # "don't step outside" (control)
    r"police ko (?:mat|nahi) bata(?:na)?",                             # "don't tell the police"
    r"kisi ko (?:mat|nahi) bata(?:na)?",                               # "don't tell anyone"
    r"zor se chilla(?:na|yi|ya)? (?:mat|nahi)",                        # "don't shout loudly"
    # ---- Devanagari equivalents ----
    "(?:ज़िंदा|जीता) नहीं (?:छोड|छोड़)", "कपड़े? फाड़", "नंगी कर",
    "बाहर (?:मत|नहीं) निकलना", "पुलिस को (?:मत|नहीं) बताना", "किसी को (?:मत|नहीं) बताना",
    # ---- additional Bengali threat patterns (romanised + script) ----
    r"nangto kore debo", r"kapor chire debo", r"baire berona",
    r"police ke bolbi na", r"kake bolbi na",
    "নাঙ্গটো করে দেব", "কাপড় ছিঁড়ে দেব", "বাইরে বেরোনা",
    "পুলিশকে বলবি না", "কাকে বলবি না",
    # ---- Tamil (romanised, for STT that outputs roman) ----
    r"un uyir eduppaen", r"un udambu kedukkaadhu",
    r"sathama iruda", r"yaarukkum sollaadhae",
    r"car la earu", r"van la earu",
)
# level 1: sexualised remarks / slurs aimed at women. Market words ("maal", "kitne ka", "patakha") need context;
# generic profanity is dropped (men swear among themselves constantly - it drowned the signal).
_ABUSE = (
    r"chikni", r"(?:kya|mast|badhiya|zabardast|kadak|tagda) maal", r"(?:kya|mast) pat(?:a|aa)kh?a",
    r"chhammak ?chhallo", r"(?:apna|apni) number (?:de|do|dena)", r"ghar (?:chhod|chod) (?:du|doon|dunga|de)",
    r"sexy", r"hot lag\w*", r"(?:aye|oye) (?:jaanu|baby|chhamiya|sweety)", r"bazaru", r"randi", r"raand",
    r"(?:chhinaal|chinaal|chinal)", r"kutiya", r"(?:bhaag|bhag)(?:na)? mat",
    r"(?:phone|mobile|chain|purse|paise) nikal\w*", r"slut", r"bitch", r"whore", r"hey (?:sexy|baby|beautiful)",
    "चिकनी", "(?:क्या|मस्त) माल", "(?:क्या|मस्त) पटाखा", "रंडी", "छिनाल", "कुतिया", "सेक्सी", "भाग(?:ना)? मत",
    r"khanki", r"magi", r"beshya", r"(?:ki|darun) maal", "খানকি", "মাগি", "মাগী", "বেশ্যা", "কি মাল",
    # --- additional Hindi / Hindustani harassment ---
    r"item\s*(?:hai|ho|lag\w*)", r"(?:kya|mast)\s*figure", r"(?:teri|tumhari)\s*(?:aukaat|aukat)",
    r"(?:ek|do)\s*(?:raat|ghanta)", r"kitne\s*(?:ka|ki|mein)", r"thumka\s*laga",
    r"(?:jism|badan)\s*(?:mast|kadak|zabardast)", r"(?:aaja|aa\s*ja)\s*mere\s*(?:saath|paas)",
    r"(?:bol|bata)\s*kitne\s*(?:ka|ki|mein)", r"(?:oye|aye|ae)\s*(?:chhori|chori|ladki|madam)",
    r"(?:dupatta|chunni|palla|kapda)\s*(?:hata|sarak|utha|sar(?:a|ak))\w*",
    # Devanagari equivalents
    r"आइटम", r"(?:क्या|मस्त)\s*फिगर", r"(?:तेरी|तुम्हारी)\s*(?:औकात|औक़ात)",
    r"(?:एक|दो)\s*(?:रात|घंटा)", r"कितने\s*(?:का|की|में)", r"ठुमका\s*लगा",
    r"(?:आजा|आ\s*जा)\s*मेरे\s*(?:साथ|पास)", r"(?:ओये|ऐ)\s*(?:छोरी|लड़की)",
    r"(?:दुपट्टा|चुन्नी|पल्ला|कपड़ा)\s*(?:हटा|सरक|उठा)\w*",
    # --- additional Bengali harassment ---
    r"(?:ki|darun|joss|fatafati)\s*figur(?:e)?", r"(?:aye|ore|ei)\s*(?:magi|meye|bou|didi)",
    r"(?:kapor|jama)\s*(?:khol|chhir|sar)\w*", r"(?:ek|du)(?:ta|ti)\s*(?:raat|din)\s*(?:dey|de|dao)",
    r"(?:কি|দারুণ)\s*ফিগার", r"(?:এই|ওরে)\s*(?:মেয়ে|বৌদি)", r"(?:কাপড়|জামা)\s*(?:খোল|ছিড়|সড়)\w*",
    # --- Tamil romanised harassment ---
    r"(?:enna|oru)\s*(?:figure|item)", r"(?:adi|dei|da)\s*(?:thevidiya|thevdiya|devadiya)",
    r"(?:un|unga)\s*(?:udambu|body)\s*(?:super|semma)", r"oru\s*(?:raathiri|naal)\s*(?:vaada|vaa)",
    r"thundu", r"loosu\s*(?:ponnu|paiyan)", r"(?:ava|ival)\s*oru\s*(?:item|maal)",
)


def _compile(pats) -> list:
    return [re.compile(rf"(?<!{_W}){unicodedata.normalize('NFC', p)}(?!{_W})") for p in pats]


_THREAT_RE, _ABUSE_RE = _compile(_THREAT), _compile(_ABUSE)


def classify_speech(text: str) -> tuple[int, list[str]]:
    """-> (level 0|1|2, matched phrases). Weak on purpose: a cue for the engine, never an alert by itself.
    Music near her (Bollywood item songs) will match level 1; filter music on the client."""
    low = unicodedata.normalize("NFC", text.lower().replace("\u2019", "'"))
    low = re.sub(r"[^\w\u0900-\u0DFF']+", " ", low).strip()
    thr = [m.group(0) for p in _THREAT_RE if (m := p.search(low))]
    if thr:
        return 2, thr
    ab = [m.group(0) for p in _ABUSE_RE if (m := p.search(low))]
    return (1, ab) if ab else (0, [])


# ---------------- timing ----------------

class FrameClock:
    """Server-aligned capture time per frame. Optional client stamp; else arrival time. Never post-inference time:
    slow object-detection frames used to shrink the next frame's dt and fake a 2-3x faster punch."""

    def __init__(self):
        self.offset: float | None = None
        self.last = 0.0

    def stamp(self, recv_ts: float, client_ts: float | None = None) -> float:
        t = recv_ts
        if client_ts is not None:
            o = recv_ts - client_ts
            if self.offset is None or o < self.offset or o - self.offset > 5.0:
                self.offset = o              # min latency ~ clock offset; reset on client clock jumps
            t = client_ts + self.offset
        t = max(t, self.last + 1e-3)
        self.last = t
        return t