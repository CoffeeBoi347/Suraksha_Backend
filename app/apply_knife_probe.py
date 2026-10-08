"""Is it the camera or the detector?   python tools/knife_probe.py --source 0 [--target knife|gun|...]

Runs the target class (default knife) on every frame at several input sizes and tells you, per situation, how often it fires
at the production confidence (0.30), the knife's size in pixels, and how blurry the frames were.

Hold the (blunt / prop) knife and press a key to say what you are doing, then walk to 2 m, 4 m, 6 m:
    s = knife held still      w = knife swinging      n = no knife in view (false-alarm check)      q = quit
Read the result:
    still detects, swing doesn't, blur ratio low while swinging  -> motion blur (camera exposure)
    even still doesn't detect, knife px < ~25                    -> too few pixels: raise --sizes / camera resolution
    detects only at large --sizes                                -> run production with that WEAPON_IMGSZ
    nothing detects even big and close                            -> the COCO knife class is the limit: fine-tune
No-GUI run on a recorded clip:  --source clip.mp4 --label swing
"""
import argparse
import collections

import cv2
import numpy as np
from ultralytics import YOLO

PROD_CONF = 0.30      # production floor (cues.OBJECT_MIN_CONF)


def _sharp(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (320, max(1, int(h * 320 / w)))) if w > 320 else frame
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())


def _best_knife(res, ids):
    best = None
    boxes = getattr(res, "boxes", None)
    for b in (boxes if boxes is not None else []):
        if int(b.cls[0]) not in ids:
            continue
        conf = float(b.conf[0])
        if best is None or conf > best[0]:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            best = (conf, (x1, y1, x2, y2))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--target", default="knife", help="class name in the model, e.g. knife, gun, 'baseball bat'")
    ap.add_argument("--sizes", type=int, nargs="+", default=[480, 640, 960, 1280])
    ap.add_argument("--label", choices=["still", "swing", "none"], help="no-GUI mode: label every frame")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    model = YOLO(a.weights)
    names = getattr(model, "names", None) or {43: "knife"}
    ids = [i for i, n in names.items() if str(n).lower().replace("_", " ") == a.target.lower()]
    if not ids:
        raise SystemExit(f"'{a.target}' is not a class of {a.weights}. Classes: {sorted(set(map(str, names.values())))[:40]}")
    keys = {ord("s"): "still", ord("w"): "swing", ord("n"): "none"}
    label = a.label or "none"
    stats = collections.defaultdict(lambda: {"n": 0, "hit": 0, "conf": 0.0, "px": 0.0})
    blur = collections.defaultdict(list)
    base, n = 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        s = _sharp(frame)
        base = max(s, base * 0.985)
        blur[label].append(s / base if base > 1e-6 else 1.0)
        view = frame.copy()
        for i, sz in enumerate(a.sizes):
            res = model.predict(source=frame, classes=ids, imgsz=sz, conf=0.05, verbose=False)
            best = _best_knife(res[0], ids) if res else None
            st = stats[(label, sz)]
            st["n"] += 1
            if best and best[0] >= PROD_CONF:
                st["hit"] += 1
                st["conf"] += best[0]
                st["px"] += max(best[1][2] - best[1][0], best[1][3] - best[1][1])
            if a.label is None:
                txt = f"{sz}: " + (f"{best[0]:.2f} {max(best[1][2]-best[1][0], best[1][3]-best[1][1]):.0f}px" if best else "-")
                cv2.putText(view, txt, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if best and best[0] >= PROD_CONF else (0, 165, 255), 1)
                if best:
                    x1, y1, x2, y2 = (int(v) for v in best[1])
                    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0) if best[0] >= PROD_CONF else (0, 165, 255), 1)
        if a.label is None:
            cv2.putText(view, f"mode: {label}  (s still / w swing / n none / q quit)  blur {blur[label][-1]:.2f}",
                        (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("knife probe", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            label = keys.get(k, label)
        if a.max_frames and n >= a.max_frames:
            break
    cap.release()
    print(f"\n{'situation':9s} {'imgsz':>5s} {'frames':>6s} {'detected@0.30':>14s} {'mean conf':>9s} {'knife px':>8s}   blur ratio (1.0 = sharpest)")
    for (lab, sz), st in sorted(stats.items()):
        if st["n"] == 0:
            continue
        h = st["hit"]
        print(f"{lab:9s} {sz:>5d} {st['n']:>6d} {100 * h / st['n']:>13.0f}% "
              f"{(st['conf'] / h if h else 0):>9.2f} {(st['px'] / h if h else 0):>8.0f}   "
              f"{float(np.mean(blur[lab])) if blur[lab] else 0:.2f}")


if __name__ == "__main__":
    main()