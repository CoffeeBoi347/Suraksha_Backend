"""Threat engine v3 — base-rate aware, research-backed attack-pattern cues.

Design rules (routine-activity theory + offender approach research: blitz / surprise / con):
  * Presence / proximity alone NEVER scores. Almost everyone near you is harmless.
  * An approach is a SEQUENCE (shadow -> converge -> contact), not a snapshot.
  * One indicator = MODERATE (and is sent to Gemini for verification). CRITICAL needs >=2
    independent indicators, a held sharp weapon, or a swung object + anything.
  * Context matters: crowded = guardians present (damp), isolated/dark = no capable guardian (boost).
  * Nothing here looks at appearance, clothing, age, gender or ethnicity. Behaviour only.

Cues mapped to research:
  converge / converge_fast / contact / strike / object_swing  -> BLITZ (sudden violence)
  ambush                                                      -> SURPRISE (appears close from the side, closes)
  engage / block                                              -> CON (stops in talking range, bars the path)
  flank                                                       -> gangs / pairs closing together
  follow / follow_strong                                      -> STALK / PROBE
  stare / linger / verbal_abuse / verbal_threat / rejected_follow -> INDIA STREET HARASSMENT (stare, pacing,
        remarks, retaliation after refusal). Valid in crowds: most incidents happen in daylight crowds.
"""
import os
import time
from collections import deque

import numpy as np

from app.intent import head_read

NOSE, L_EYE, R_EYE, L_SH, R_SH = 0, 1, 2, 5, 6

HISTORY_S = 6.0
FOLLOW_MIN_S = 25.0           # same tracked person near you this long while you walk
FOLLOW_MAX_DIST = 12.0
FOLLOW_MIN_FRAMES = 8
FOLLOW_LATCH_S = 30.0         # once flagged, stays flagged (track flicker must not reset it)

CBDR_WINDOW_S = 1.5           # constant bearing, decreasing range window
CBDR_MAX_BEARING_STD = 0.09   # std of normalised x-position over the window
CBDR_MIN_APPROACH = 0.5       # m/s, ego-motion compensated
CBDR_MAX_DIST = 7.0
FAST_APPROACH = 2.0           # m/s. Normal walking is 1.2-1.5: 1.2 flagged every oncoming pedestrian
NET_CLOSE_MIN = 1.2           # metres actually gained for "rushing"
FAST_MAX_DIST = 6.0
FACING_WINDOW_S = 2.0
FACING_MIN_FRAC = 0.5         # facing = head yaw pointed at the camera, not just "nose visible"

CONTACT_DIST = 1.2
BLOCK_DIST = 2.5
BLOCK_MIN_S = 1.0
POSE_WINDOW_S = 1.5
POSE_MIN_SAMPLES = 2          # frames with pose score >= POSE_MIN within the window
POSE_MIN = 0.4
STRIKE_MIN = 0.85
STRIKE_MIN_SAMPLES = 2
STRIKE_MAX_DIST = 3.0

# research-backed patterns
AMBUSH_FIRST_DIST = 4.5       # first seen already this close ...
AMBUSH_EDGE = 0.3             # ... and off to the side (|cx-0.5| >= this)
AMBUSH_MIN_APPROACH = 0.6
FLANK_MAX_DIST = 6.0
ENGAGE_DIST = 2.2
ENGAGE_MIN_S = 2.0

# India street-harassment research: stare -> pace-matching -> verbal -> disguised contact -> retaliation on refusal
STARE_FACE_FRAC = 0.6         # facing the wearer for most of the window ...
STARE_MIN_S = 4.0             # ... for this long, while close
STARE_MAX_DIST = 5.0
LINGER_DIST = 1.6             # stays within arm-ish range of a WALKING wearer (pacing / shadowing, works in crowds)
LINGER_MIN_S = 8.0
REJECT_WINDOW_S = 12.0        # engaged / blocked her, she walked on, they are still right there
REJECT_MAX_DIST = 3.5
VERBAL_FRESH_S = 6.0          # heard abuse / threat this recently
VERBAL_MAX_DIST = 5.0         # attributed to the NEAREST person only
# One microphone cannot tell WHO is speaking. A voice (words or a scream) is blamed on a track only when that
# person is the plausible source; in a crowd the source is unknowable and nobody is blamed.
VOICE_MARGIN_M = 0.8          # speaker must be this much nearer than anyone else
VOICE_CROWD_MAX_DIST = 1.5    # inside a crowd only someone within reach can be blamed
VOICE_TOO_MANY = 8            # this many people within 8 m = a crowd of voices: blame nobody
VOICE_AUDIO_FEW = 3           # scream may still be tied to a track when <= this many people are within 8 m

ISOLATION_RADIUS_M = 15.0
ISOLATION_SEEN_S = 2.0
ISOLATION_MEMORY_S = 20.0     # a 65 deg camera sees ~20% of her surroundings: "alone" = nobody seen for 20 s
CROWD_RADIUS_M = 3.5
CROWD_PEOPLE = 3              # this person + 2 others within radius
DIST_EMA = 0.5

# ---- encirclement (boxed-in) ----
ENCIRCLE_DIST_M = 4.0          # within this radius
ENCIRCLE_MIN_PEOPLE = 3        # at least 3 people (not counting the wearer)
ENCIRCLE_SPREAD_DEG = 150      # angular spread of surrounding people >= this -> encircled
ENCIRCLE_MIN_S = 2.0           # sustained for at least 2 s
ENCIRCLE_FACING_FRAC = 0.5
ENCIRCLE_ENABLED = os.getenv("ENCIRCLE_ENABLED", "0") == "1"   # a forward camera cannot see behind her

WEIGHTS = {
    "converge": 0.30, "converge_fast": 0.50, "contact": 0.40, "block": 0.25,
    "follow": 0.40, "follow_strong": 0.50, "pose": 0.30, "strike": 0.60,
    "audio": 0.35, "weapon_held": 0.80, "weapon_near": 0.30,
    "engage": 0.20, "ambush": 0.35, "flank": 0.35, "encircle": 0.55,
    "object_held": 0.30, "object_swing": 0.60,
    "stare": 0.15, "linger": 0.25, "rejected_follow": 0.45,
    "verbal_abuse": 0.25, "verbal_threat": 0.45,
}
LABELS = {
    "converge": "walking straight at you", "converge_fast": "rushing at you",
    "contact": "aggressive motion within arm's reach", "block": "standing in your path",
    "follow": "following you", "follow_strong": "same person keeps staying near you",
    "pose": "aggressive arm motion", "strike": "strike-like arm motion",
    "audio": "scream detected", "weapon_held": "possible weapon in hand",
    "weapon_near": "object near hands",
    "engage": "stopped close in front of you", "ambush": "came at you from the side",
    "flank": "two people closing on you", "encircle": "people surrounding you",
    "object_held": "holding an object that could be a weapon",
    "object_swing": "raising or swinging an object",
    "stare": "staring at you", "linger": "keeping pace right next to you",
    "rejected_follow": "kept after you once you walked away",
    "verbal_abuse": "abusive words heard nearby", "verbal_threat": "threat heard nearby",
}
# rank-2 pairs: two of these together are CRITICAL on their own
HARD = {"contact", "audio", "strike", "converge_fast", "object_swing"}
SOLO_RANK1 = {"strike", "follow_strong", "converge_fast", "contact", "ambush", "flank", "encircle",
              "object_swing", "linger", "rejected_follow", "verbal_abuse", "verbal_threat",
              "object_held"}   # vision.py only sets object_held for an object held RAISED / cocked, not just carried
CRITICAL_MIN_WEIGHT = 0.90    # 3+ cues are CRITICAL only with a HARD cue and real combined weight
SOLO_CTX = {"converge", "block", "engage", "pose", "weapon_near"}   # solo rank 1 only if alone or dark
PRE_ATTACK = {"converge", "converge_fast", "ambush", "flank", "encircle", "engage", "block", "follow",
              "follow_strong", "linger", "rejected_follow", "verbal_abuse", "verbal_threat"}
LEVELS = ("NORMAL", "MODERATE", "CRITICAL")


def fuse(cats: dict, ctx: dict) -> tuple[int, float, str]:
    """cats: {name: strength 0..1 (unused beyond presence) } -> (rank 0|1|2, score, reason).
    ctx: {"isolated": bool, "crowd": bool, "dark": bool, "safe": bool}"""
    c = dict(cats)
    if ctx.get("safe"):                  # home / office: walking up to you, standing near = normal life
        for k in ("converge", "block", "follow", "follow_strong", "engage", "ambush", "flank",
                  "stare", "linger", "rejected_follow"):
            c.pop(k, None)
    if ctx.get("crowd"):                 # people walk toward / past you in crowds all day
        for k in ("converge", "block", "engage", "ambush", "flank"):
            c.pop(k, None)
    # contact / strike / pose all come from the SAME arm-motion evidence -> count once (strongest)
    if "contact" in c:
        c.pop("pose", None); c.pop("strike", None)
    elif "strike" in c:
        c.pop("pose", None)
    if "object_swing" in c:
        c.pop("object_held", None); c.pop("pose", None)   # same evidence, count once
    if "block" in c:
        c.pop("engage", None)            # same evidence, count once
    if "verbal_threat" in c:
        c.pop("verbal_abuse", None)
    if "linger" in c and ("follow" in c or "follow_strong" in c):
        c.pop("linger", None)            # both say "keeps pace with you" -> count once
    c = {k: v for k, v in c.items() if k in WEIGHTS and v > 0}
    n = len(c)
    isolated = bool(ctx.get("isolated"))
    dark = bool(ctx.get("dark"))
    keys = set(c)

    rank = 0
    if "weapon_held" in keys:
        rank = 2 if n >= 2 else 1
    elif ("weapon_near" in keys and n >= 2) or ("object_swing" in keys and n >= 2):
        rank = 2
    elif n >= 3:
        # three weak harassment cues (stare + linger + abusive words) are serious but not violence: MODERATE
        rank = 2 if (keys & HARD and sum(WEIGHTS[k] for k in keys) >= CRITICAL_MIN_WEIGHT) else 1
    elif n == 2:
        follow = keys & {"follow", "follow_strong"}
        if keys <= HARD:
            rank = 2
        elif follow and "converge_fast" in keys:
            rank = 2 if (isolated or dark) else 1
        elif follow and "converge" in keys and isolated and dark:
            rank = 2                     # dark + alone + followed + walked up to: classic setup
        elif "verbal_threat" in keys and len(keys & PRE_ATTACK) >= 2:
            rank = 2                     # threatened her AND closing / following / blocking
        elif isolated and dark and len(keys & PRE_ATTACK) == 2:
            rank = 2                     # two approach cues, no witnesses, dark
        else:
            rank = 1
    elif n == 1:
        k = next(iter(keys))
        if k in SOLO_RANK1 or (k in SOLO_CTX and (isolated or dark)) or (k == "follow" and (isolated or dark)):
            rank = 1

    score = float(np.clip(sum(WEIGHTS[k] for k in keys), 0.0, 1.0))
    if isolated and dark and rank >= 1:
        score = min(1.0, score + 0.05)
    score = {0: min(score, 0.30), 1: float(np.clip(score, 0.35, 0.69)), 2: max(score, 0.70)}[rank]

    top = sorted(keys, key=lambda k: WEIGHTS[k], reverse=True)[:3]
    reason = ", ".join(LABELS[k] for k in top) if rank else "no threat indicators"
    if rank and isolated:
        reason += " (dark, no one else around)" if dark else " (no one else around)"
    return rank, score, reason


class TrackState:
    __slots__ = ("hist", "first_seen", "last_seen", "dist_ema", "last_dist",
                 "face", "pose", "path_since", "follow_until",
                 "first_dist", "first_cx", "last_approach", "engage_since",
                 "stare_since", "engaged_at", "face_frac")

    def __init__(self, now: float):
        self.hist: deque = deque()        # (t, dist, cx_norm, box_h)
        self.first_seen = now
        self.last_seen = now
        self.dist_ema: float | None = None
        self.last_dist = 99.0
        self.face: deque = deque()        # (t, facing_bool)
        self.pose: deque = deque()        # (t, pose_score)
        self.path_since: float | None = None
        self.follow_until = 0.0
        self.first_dist: float | None = None
        self.first_cx = 0.5
        self.last_approach = 0.0
        self.engage_since: float | None = None
        self.stare_since: float | None = None
        self.engaged_at: float | None = None
        self.face_frac = 0.0


class ThreatEngine:
    """One per WebSocket connection. Not thread-safe — call from the frame loop only."""

    def __init__(self, hfov: float = 65.0):
        self.hfov = hfov
        self.tracks: dict[str, TrackState] = {}
        self.sightings: dict[str, tuple[float, float]] = {}   # key -> (last seen, dist); outlives tracks
        self.audio_scream = 0.0
        self.audio_ts = 0.0
        self.user_speed_mps = 0.0
        self.verbal_level = 0          # 0 none, 1 abuse, 2 threat
        self.verbal_ts = 0.0
        self.verbal_words: list[str] = []
        self._encircle_since: float | None = None

    def set_verbal(self, level: int, words: list[str]):
        self.verbal_level = max(0, min(2, level))
        self.verbal_ts = time.time()
        self.verbal_words = words[:3]

    def _is_nearest(self, key: str, now: float) -> bool:
        d = self.tracks[key].last_dist
        return all(s.last_dist >= d for k, s in self.tracks.items()
                   if k != key and now - s.last_seen < 1.0)

    def _count_near(self, now: float, radius: float) -> int:
        return sum(1 for s in self.tracks.values() if now - s.last_seen < 1.0 and s.last_dist <= radius)

    def _likely_source(self, key: str, now: float, crowd: bool) -> bool:
        """Could this person be the one talking? Clearly the nearest, not in a crowd of voices."""
        me = self.tracks[key].last_dist
        if self._count_near(now, 8.0) >= VOICE_TOO_MANY:
            return False
        if any(s.last_dist < me + VOICE_MARGIN_M for k, s in self.tracks.items()
               if k != key and now - s.last_seen < 1.0):
            return False
        return (not crowd) or me <= VOICE_CROWD_MAX_DIST

    def set_audio(self, scream_prob: float):
        self.audio_scream = max(0.0, min(1.0, scream_prob))
        self.audio_ts = time.time()

    def set_user_speed(self, mps: float):
        self.user_speed_mps = max(0.0, mps)

    def global_audio_alert(self, now: float, thresh: float = 0.8) -> bool:
        return self.audio_scream >= thresh and now - self.audio_ts < 4.0

    def drop_missing(self, live_keys: set[str], now: float, ttl: float = 8.0):
        for k in list(self.tracks):
            if k not in live_keys and now - self.tracks[k].last_seen > ttl:
                del self.tracks[k]
        for k in [k for k, (ts, _) in self.sightings.items() if now - ts > ISOLATION_MEMORY_S]:
            del self.sightings[k]

    def context(self, key: str, now: float) -> dict:
        others = [s for k, s in self.tracks.items() if k != key and now - s.last_seen < ISOLATION_SEEN_S]
        isolated = not any(k != key and now - ts <= ISOLATION_MEMORY_S and dd <= ISOLATION_RADIUS_M
                           for k, (ts, dd) in self.sightings.items())
        n_close = 1 + sum(1 for s in others if s.last_dist <= CROWD_RADIUS_M)
        return {"isolated": isolated, "crowd": n_close >= CROWD_PEOPLE}

    def encirclement(self, now: float) -> bool:
        """Are people surrounding the wearer? True if >= ENCIRCLE_MIN_PEOPLE close tracks
        are spread around her by >= ENCIRCLE_SPREAD_DEG and at least some face her."""
        close = [(s.last_dist, s.hist[-1][2] if s.hist else 0.5,
                  sum(1 for _, f in s.face if f) / max(len(s.face), 1))
                 for s in self.tracks.values()
                 if now - s.last_seen < 1.5 and s.last_dist <= ENCIRCLE_DIST_M]
        if not ENCIRCLE_ENABLED or len(close) < ENCIRCLE_MIN_PEOPLE:
            self._encircle_since = None
            return False
        # real bearings (old code mapped the frame to +-90 deg and its wrap-around gap called
        # left edge + centre + right edge "surrounded"). Only meaningful with a wide / 360 camera.
        angles = sorted((cx - 0.5) * self.hfov for _, cx, _ in close)
        angular_coverage = angles[-1] - angles[0]
        # check: enough of them face her?
        facing_frac = sum(1 for _, _, ff in close if ff >= 0.3) / len(close)
        if angular_coverage < ENCIRCLE_SPREAD_DEG or facing_frac < ENCIRCLE_FACING_FRAC:
            self._encircle_since = None
            return False
        self._encircle_since = self._encircle_since or now
        return now - self._encircle_since >= ENCIRCLE_MIN_S

    def update(self, key: str, now: float, dist_m: float, cx_norm: float,
               box_w_px: float, box_h_px: float, kxy, kconf, kp_conf_floor: float,
               ext_pose: tuple | None = None) -> dict:
        st = self.tracks.get(key)
        if st is None:
            st = self.tracks[key] = TrackState(now)
        st.last_seen = now
        if st.first_dist is None:
            st.first_dist, st.first_cx = dist_m, cx_norm

        st.dist_ema = dist_m if st.dist_ema is None else DIST_EMA * dist_m + (1 - DIST_EMA) * st.dist_ema
        d = st.dist_ema
        st.last_dist = d
        self.sightings[key] = (now, d)
        st.hist.append((now, d, cx_norm, box_h_px))
        while st.hist and now - st.hist[0][0] > HISTORY_S:
            st.hist.popleft()

        # facing: nose + at least one eye visible => face is toward the camera (rough, honest)
        facing = _facing(kxy, kconf, kp_conf_floor, box_h_px)
        st.face.append((now, facing))
        while st.face and now - st.face[0][0] > FACING_WINDOW_S:
            st.face.popleft()
        face_frac = sum(1 for _, f in st.face if f) / max(len(st.face), 1)
        st.face_frac = face_frac

        # sustained pose (one frame of "arm up" is a wave / jitter)
        p_score, p_why = ext_pose if ext_pose else (0.0, "")
        st.pose.append((now, float(p_score)))
        while st.pose and now - st.pose[0][0] > POSE_WINDOW_S:
            st.pose.popleft()
        n_pose = sum(1 for _, s in st.pose if s >= POSE_MIN)
        n_strike = sum(1 for _, s in st.pose if s >= STRIKE_MIN)

        ctx = self.context(key, now)

        # ego-compensated closing speed
        approach = _approach_slope(st.hist)
        in_front = abs(cx_norm - 0.5) < 0.35
        if in_front:
            approach -= self.user_speed_mps
        st.last_approach = approach

        cats: dict[str, float] = {}

        # --- converge: constant bearing + decreasing range + facing you ---
        recent = [h for h in st.hist if now - h[0] <= CBDR_WINDOW_S]
        if (len(recent) >= 5 and recent[-1][0] - recent[0][0] >= CBDR_WINDOW_S * 0.7
                and approach >= CBDR_MIN_APPROACH and d <= CBDR_MAX_DIST and face_frac >= FACING_MIN_FRAC
                and float(np.std([h[2] for h in recent])) <= CBDR_MAX_BEARING_STD):
            cats["converge"] = 1.0
        net_close = _net_closing(st.hist)
        if (approach >= FAST_APPROACH and net_close >= NET_CLOSE_MIN and d <= FAST_MAX_DIST
                and face_frac >= FACING_MIN_FRAC and len(st.hist) >= 5):
            cats["converge_fast"] = 1.0
            cats.pop("converge", None)

        # --- contact: aggressive motion sustained AND within reach ---
        if d <= CONTACT_DIST and n_pose >= POSE_MIN_SAMPLES:
            cats["contact"] = 1.0
        elif n_pose >= POSE_MIN_SAMPLES:
            cats["pose"] = 1.0
        if n_strike >= STRIKE_MIN_SAMPLES and d <= STRIKE_MAX_DIST:
            cats["strike"] = 1.0

        # --- block: sits in your walking path while YOU move and THEY don't ---
        if (d <= BLOCK_DIST and abs(cx_norm - 0.5) < 0.18 and self.user_speed_mps > 0.4
                and abs(approach) < 0.4):
            st.path_since = st.path_since or now
            if now - st.path_since >= BLOCK_MIN_S:
                cats["block"] = 1.0
        else:
            st.path_since = None

        # --- follow: same track near you long enough while you walk; latched ---
        dwell = now - st.first_seen
        if (dwell >= FOLLOW_MIN_S and d <= FOLLOW_MAX_DIST and len(st.hist) >= FOLLOW_MIN_FRAMES
                and self.user_speed_mps > 0.4):
            st.follow_until = now + FOLLOW_LATCH_S
        if now < st.follow_until:
            cats["follow"] = 1.0

        # --- ambush (surprise approach): appears already close, off to the side, then closes ---
        if (now - st.first_seen <= 4.0 and st.first_dist is not None and st.first_dist <= AMBUSH_FIRST_DIST
                and abs(st.first_cx - 0.5) >= AMBUSH_EDGE and approach >= AMBUSH_MIN_APPROACH and d <= 5.0):
            cats["ambush"] = 1.0

        # --- flank (pairs / gangs): 2+ people closing at once, BOTH facing her (two pedestrians != a pair) ---
        if approach >= 0.8 and d <= FLANK_MAX_DIST and face_frac >= FACING_MIN_FRAC:
            if any(k2 != key and now - s2.last_seen < 1.0 and s2.last_approach >= 0.8
                   and s2.face_frac >= FACING_MIN_FRAC
                   and s2.last_dist <= FLANK_MAX_DIST for k2, s2 in self.tracks.items()):
                cats["flank"] = 1.0

        # --- encircle: multiple people surrounding her, sustained ---
        if d <= ENCIRCLE_DIST_M and self.encirclement(now):
            cats["encircle"] = 1.0

        # --- engage (con approach): stops in talking range, faces her, holds ---
        if d <= ENGAGE_DIST and face_frac >= FACING_MIN_FRAC and in_front and abs(approach) < 0.3:
            st.engage_since = st.engage_since or now
            if now - st.engage_since >= ENGAGE_MIN_S:
                cats["engage"] = 1.0
        else:
            st.engage_since = None

        # --- stare (probing): faces the wearer, close, sustained. Weak alone; only counts with another cue ---
        if d <= STARE_MAX_DIST and face_frac >= STARE_FACE_FRAC:
            st.stare_since = st.stare_since or now
            if now - st.stare_since >= STARE_MIN_S:
                cats["stare"] = 1.0
        else:
            st.stare_since = None

        # --- linger: stays within ~1.6 m of a WALKING wearer for 8 s+ (pacing / shadowing; valid in crowds too) ---
        if (dwell >= LINGER_MIN_S and d <= LINGER_DIST and len(st.hist) >= FOLLOW_MIN_FRAMES
                and self.user_speed_mps > 0.4):
            cats["linger"] = 1.0

        # --- rejected_follow (retaliation on refusal): engaged/blocked her, she walked on, still right there ---
        if "engage" in cats or "block" in cats:
            st.engaged_at = now
        if (st.engaged_at is not None and 0.5 <= now - st.engaged_at <= REJECT_WINDOW_S
                and self.user_speed_mps > 0.6 and d <= REJECT_MAX_DIST and approach >= -0.2):
            cats["rejected_follow"] = 1.0

        # --- verbal: abuse / threat heard, attributed to the nearest person only ---
        heard: list[str] = []
        if (self.verbal_level > 0 and now - self.verbal_ts < VERBAL_FRESH_S and d <= VERBAL_MAX_DIST
                and self._likely_source(key, now, ctx["crowd"])):
            cats["verbal_threat" if self.verbal_level >= 2 else "verbal_abuse"] = 1.0
            heard = self.verbal_words

        # --- audio: scream while this person is near ---
        if (self.audio_scream >= 0.6 and now - self.audio_ts < 4.0 and d <= 8.0
                and (self._likely_source(key, now, ctx["crowd"])
                     or (not ctx["crowd"] and self._count_near(now, 8.0) <= VOICE_AUDIO_FEW))):
            cats["audio"] = self.audio_scream

        rank, score, reason = fuse(cats, ctx)
        return {"score": score, "rank": rank, "level": LEVELS[rank], "signals": dict(cats),
                "cats": cats, "ctx": ctx, "reason": reason,
                "approach_mps": round(approach, 2), "dwell_s": int(dwell), "heard": heard}


def _facing(kxy, kconf, floor: float, box_h: float) -> bool:
    """Head yaw pointed at the camera and not bowed. Old rule ("nose + one eye visible") was true for
    anyone not showing their back, so every oncoming pedestrian counted as 'facing her'."""
    h = head_read(kxy, kconf, floor, box_h)
    return bool(h["valid"] and h["frontal"] and not h["down"])


def _approach_slope(hist: deque) -> float:
    """m/s the person is closing (positive = getting closer), least-squares over the window."""
    if len(hist) < 4:
        return 0.0
    ts = np.fromiter((h[0] for h in hist), dtype=float)
    ds = np.fromiter((h[1] for h in hist), dtype=float)
    if ts[-1] - ts[0] < 0.8:
        return 0.0
    tm = ts - ts.mean()
    denom = float((tm * tm).sum())
    if denom < 1e-6:
        return 0.0
    return -float((tm * (ds - ds.mean())).sum()) / denom


def _net_closing(hist: deque) -> float:
    """metres actually gained over the window using medians of the first/last samples (jitter-proof)."""
    if len(hist) < 6:
        return 0.0
    ds = [h[1] for h in hist]
    return float(np.median(ds[:3]) - np.median(ds[-3:]))