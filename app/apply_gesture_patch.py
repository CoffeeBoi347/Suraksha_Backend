"""Offensive-gesture check for vision.py.   python tools/apply_gesture_patch.py app/vision.py
A raised, steady, camera-facing hand (or anyone facing her at arm's length) nominates the person; Gemini then sees an upscaled hand close-up and judges
the fingers (middle finger, crude signs). Result is at most MODERATE ('suspicious', pattern harass): never CRITICAL,
never SOS. Checks every edit point first; if any is missing it changes NOTHING and says which. Safe to run twice."""
import sys

EDITS = [
    ("import", "    pose_signals, pre_attack_cues, ride_flags,\n)",
               "    pose_signals, pre_attack_cues, ride_flags, GestureHold, hand_crop,\n)"),
    ("prompt", '"on her after she walked away, with or without words heard.\\n"',
               '"on her after she walked away, with or without words heard; OR an offensive hand gesture aimed at her "\n'
               '    "(middle finger, crude sexual sign, kissing or groping motions), judged from the hand close-up when "\n'
               '    "given. Such a gesture alone is \'suspicious\', never \'threat\'. Friends joking, waving, thumbs-up, "\n'
               '    "pointing at something else, a phone call are NOT.\\n"'),
    ("track", '"last_dbg": 0.0', '"last_dbg": 0.0, "gest": GestureHold()'),
    ("candidate", "                # 0. objects: held != threat. Sharp in hand, raised, or swung is.\n",
                  "                # 0a. offensive-gesture candidate: a hand held up toward her, steady, face turned to her. Pose keypoints\n"
                  "                # stop at the wrist, so this only nominates; Gemini judges the fingers from an upscaled hand close-up.\n"
                  "                crowded = engine.context(key, ft)[\"crowd\"]     # packed scene: no close-range checks (Gemini budget)\n"
                  "                gesture, g_wrist = (False, None) if company else st[\"gest\"].update(\n"
                  "                    ft, kxy, kcf, bh, max(dist, 1.7) if crowded else dist, head_state(kxy, kcf, bh) == \"front\")\n"
                  "                # very close (< 1.6 m) the frame cuts the body off and pose cannot place the hand: send the person view\n"
                  "                hand_img = ((hand_crop(frame, g_wrist, bh) if g_wrist is not None\n"
                  "                             else _person_crop(frame, d[\"box\"], False)) if gesture else None)\n"
                  "                gesture = gesture and hand_img is not None\n"
                  "                gst = st[\"gest\"] if gesture else None\n\n"
                  "                # 0. objects: held != threat. Sharp in hand, raised, or swung is.\n"),
    ("scan", "and (obj_alert or p_score >= 0.3 or any(c[0] == 0 for c in local)",
             "and (obj_alert or gesture or p_score >= 0.3 or any(c[0] == 0 for c in local)"),
    ("append", 'candidates.append((prio, key, kind, ready, st["last_crop"], d["box"], held, dist, hint, sensor))',
               'candidates.append((prio, key, kind, ready, st["last_crop"], d["box"], held, dist, hint, sensor,\n'
               '                                       hand_img, gst))'),
    ("unpack", "for _prio, key, kind, ready, last_ts, box, wide, dist, hint, sensor in sorted(",
               "for _prio, key, kind, ready, last_ts, box, wide, dist, hint, sensor, hand, gst in sorted("),
    ("dispatch", "                pending[key] = (asyncio.create_task(\n"
                 "                    verify_with_gemini(crops, dist, hint, sensor, _context_image(frame, box))), kind)\n"
                 "                cooldown[key] = ft\n",
                 "                pending[key] = (asyncio.create_task(\n"
                 "                    verify_with_gemini(crops, dist, hint, sensor, _context_image(frame, box), hand)), kind)\n"
                 "                cooldown[key] = ft\n"
                 "                if gst is not None:\n"
                 "                    gst.consume(ft)                  # one gesture check per raised hand\n"),
    ("signature", "scene: np.ndarray | None = None) -> tuple[str, float, str, str, str] | None:",
                  "scene: np.ndarray | None = None,\n"
                  "                             hand: np.ndarray | None = None) -> tuple[str, float, str, str, str] | None:"),
    ("hand", "        for index, crop in enumerate(crops, start=1):\n",
             "        if hand is not None:\n"
             "            ok, buf = cv2.imencode(\".jpg\", hand, [cv2.IMWRITE_JPEG_QUALITY, 92])\n"
             "            if ok:\n"
             "                contents += [\"Hand close-up: the raised hand of the person in the red box. Judge whether it makes an \"\n"
             "                             \"offensive or obscene gesture aimed at the wearer (middle finger, crude sexual sign).\",\n"
             "                             types.Part.from_bytes(data=buf.tobytes(), mime_type=\"image/jpeg\")]\n"
             "        for index, crop in enumerate(crops, start=1):\n"),
    ("cooldown", "and ft - cooldown.get(key, 0.0) > cd and ready:",
                 "and (ft - cooldown.get(key, 0.0) > cd or gesture) and ready:"),   # once per raised hand anyway
    ("features", '"follow": follow_lvl, "user_speed": round(user_speed, 2),',
                 '"follow": follow_lvl, "user_speed": round(user_speed, 2), "gesture": gesture,'),
]


def main(path: str) -> int:
    src = open(path, encoding="utf-8").read()
    if "GestureHold" in src:
        print("already patched")
        return 0
    missing = [name for name, old, _ in EDITS if src.count(old) != 1]
    if missing:
        print("NOT patched. These edit points were not found exactly once:", ", ".join(missing))
        return 1
    for _, old, new in EDITS:
        src = src.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(src)
    print(f"patched {len(EDITS)} places in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "app/payload.py"))