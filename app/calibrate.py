"""Tune Suraksha from user feedback:  python -m app.calibrate

Reads threat_events labelled real_threat / false_alarm (needs the `features` column) and prints:
  1. per-cue false-alarm rate        -> which cues to weaken / gate in threat.fuse()
  2. logistic fit real_threat ~ cues -> relative weights to copy into fuse()
  3. Gemini confidence vs precision  -> GEMINI_MIN_CONF / GEMINI_SCAN_MIN_CONF
Feedback only measures false alarms, not misses: for misses, replay labelled clips.
"""
from collections import defaultdict

import numpy as np

from app.db import supabase_admin

NUMERIC = ("isolated", "dark", "crowd_activity", "held", "raised", "pillion", "gesture", "audio_noisy",
           "company", "safe", "frame_degraded")


def fetch() -> list[dict]:
    rows, start = [], 0
    while True:
        res = (supabase_admin.table("threat_events").select("level,user_label,verified_by_gemini,features")
               .in_("user_label", ["real_threat", "false_alarm"]).range(start, start + 999).execute())
        batch = res.data or []
        rows += [r for r in batch if r.get("features")]
        if len(batch) < 1000:
            return rows
        start += 1000


def cues(f: dict) -> set[str]:
    """Everything that was true when the alert fired. Rows logged by older versions simply lack the newer keys."""
    out = set(f.get("cats", [])) | {f"local:{x}" for x in f.get("local", [])} | {k for k in NUMERIC if f.get(k)}
    if f.get("object"):
        out.add(f"object:{f['object']}")                      # which weapon / object type the alert was about
    if f.get("rider"):
        out.add(f"rider:{f['rider']}")
    g = f.get("gemini") or {}
    if g.get("pattern") and g["pattern"] != "none":
        out.add(f"gemini:{g['pattern']}")                     # what Gemini thought it was (weapon, harass, ...)
    return out


def logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 4000, lr: float = 0.2):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) + l2 * w) / len(y)
        b -= lr * float(np.mean(p - y))
    return w, b


def main():
    rows = fetch()
    print(f"{len(rows)} labelled events with features")
    if len(rows) < 50:
        print("Too few labels to trust anything below; keep collecting.")
    if not rows:
        return
    y = np.array([r["user_label"] == "real_threat" for r in rows], float)
    sets = [cues(r["features"]) for r in rows]

    stats = defaultdict(lambda: [0, 0])
    for s, lab in zip(sets, y):
        for c in s:
            stats[c][0] += 1
            stats[c][1] += int(lab)
    print("\ncue                      n   false-alarm rate")
    for c, (n, real) in sorted(stats.items(), key=lambda kv: -(1 - kv[1][1] / kv[1][0])):
        print(f"{c:22s} {n:5d}   {1 - real / n:6.0%}")

    names = sorted(stats)
    X = np.array([[c in s for c in names] for s in sets], float)
    w, b = logistic(X, y)
    print(f"\nlogistic weights (bias {b:+.2f}); positive = predicts real threats")
    for c, wi in sorted(zip(names, w), key=lambda kv: -kv[1]):
        print(f"{c:22s} {wi:+.2f}")

    g_all = [(r["features"]["gemini"], lab) for r, lab in zip(rows, y)
             if (r["features"].get("gemini") or {}).get("sev") == "threat"]
    # 'scan' checks had no local cue behind them (GEMINI_SCAN_MIN_CONF); the rest use GEMINI_MIN_CONF
    for title, pick in (("checks backed by a local cue  -> GEMINI_MIN_CONF", lambda k: k != "scan"),
                        ("scan checks (no local cue)     -> GEMINI_SCAN_MIN_CONF", lambda k: k == "scan")):
        g = [(v, lab) for v, lab in g_all if pick(v.get("kind", ""))]
        if g:
            print(f"\nGemini 'threat' verdicts, {title}:  conf >= t  ->  precision (n)")
            for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
                sel = [lab for v, lab in g if v["conf"] >= t]
                if sel:
                    print(f"  {t:.2f}  {np.mean(sel):.0%} ({len(sel)})")


if __name__ == "__main__":
    main()