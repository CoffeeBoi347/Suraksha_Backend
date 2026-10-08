"""Finger reading for payload.py.   python app/apply_finger_patch.py app/payload.py
Run AFTER the gesture + weapon patches. Needs:  pip install mediapipe   (model auto-downloads to models/ on first run)
  1. any person within 6 m with a hand above the hips gets a 21-point hand read (MediaPipe), facing ANY way:
     front, side, or back turned (middle finger flipped over the shoulder while walking away)
  2. 2 positive reads in 1.5 s = local middle-finger hit -> nominated to Gemini with the hand close-up (bypasses
     cooldown, like the gesture check). Still at most MODERATE ('suspicious', harass): never CRITICAL, never SOS
  3. prompt: over-the-shoulder / from-behind middle finger counts
Checks every edit point first; if any is missing it changes NOTHING and says which. Safe to run twice."""
import sys

EDITS = [
    ("import", "WeaponEvidence, AimHold, HandDetector, active_wrist, wrist_boxes,\n)",
     "WeaponEvidence, AimHold, HandDetector, active_wrist, wrist_boxes,\n"
     "    FingerReader, FingerVote, hand_up, finger_crop,\n)"),
    ("constants", 'BENIGN_S = float(os.getenv("BENIGN_S", "90"))',
     'FINGERS = os.getenv("FINGERS", "1") == "1"                       # local 21-point hand read (mediapipe)\n'
     'FINGER_MAX_DIST = float(os.getenv("FINGER_MAX_DIST", "6.0"))     # beyond this a hand is too few pixels\n'
     'FINGER_EVERY_S = float(os.getenv("FINGER_EVERY_S", "0.2"))       # per person; ~10 ms CPU per read\n'
     '_fingers = FingerReader()\n'
     'BENIGN_S = float(os.getenv("BENIGN_S", "90"))'),
    ("track", '"aim": AimHold()', '"aim": AimHold(), "fv": FingerVote()'),
    ("read", '                aim = st["aim"].update(ft, kxy, kcf, bh, dist, head_state(kxy, kcf, bh) in ("front", None))\n',
     '                aim = st["aim"].update(ft, kxy, kcf, bh, dist, head_state(kxy, kcf, bh) in ("front", None))\n'
     '                # 0b. finger read: 21 hand landmarks in metric 3-D read the same from palm or back of the hand, so a\n'
     '                # middle finger flipped over the shoulder while he walks away counts. Local; Gemini confirms.\n'
     '                if FINGERS and not company and not gesture and dist <= FINGER_MAX_DIST:\n'
     '                    up = hand_up(kxy, kcf, bh)\n'
     '                    if up is not None and ft - st["fv"].last_read >= FINGER_EVERY_S:\n'
     '                        st["fv"].add(ft, _fingers.read(finger_crop(frame, up, bh)))\n'
     '                    if up is not None and st["fv"].confirmed(ft):\n'
     '                        hand_img = finger_crop(frame, up, bh)\n'
     '                        gesture, gst = hand_img is not None, st["fv"]\n'),
    ("prompt", '"pointing at something else, a phone call are NOT.\\n"',
     '"pointing at something else, a phone call are NOT. A middle finger raised over the shoulder or from "\n'
     '    "behind toward her (e.g. while walking away) counts too.\\n"'),
]


def main(path: str) -> int:
    src = open(path, encoding="utf-8").read()
    if "FingerVote" in src:
        print("already patched")
        return 0
    if "AimHold" not in src:
        print("NOT patched. Run apply_gesture_patch.py and apply_weapon_patch.py first.")
        return 1
    missing = [n for n, old, _ in EDITS if src.count(old) != 1]
    if missing:
        print("NOT patched. These edit points were not found exactly once:", ", ".join(missing))
        return 1
    for _, old, new in EDITS:
        src = src.replace(old, new, 1)
    compile(src, path, "exec")
    open(path, "w", encoding="utf-8").write(src)
    print(f"patched {len(EDITS)} places in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "app/payload.py"))