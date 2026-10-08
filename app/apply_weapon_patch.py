"""Weapon / sharp-object hardening for vision.py.   python tools/apply_weapon_patch.py app/vision.py
Run AFTER apply_gesture_patch.py. Checks every edit point first; if any is missing it changes NOTHING and says which.
  1. detector also runs on close-ups around each wrist (a far knife is tiny in the frame, big in the hand crop)
  2. evidence builds up across frames (weak frames add up) instead of 'N hits in a row'
  3. only an object IN a hand counts; cutlery on a table does not (blade with unseen wrists still counts)
  4. Gemini also gets the hand close-up when the arm is moving fast / an object is raised
  5. a person Gemini saw with a weapon stays flagged ~20 s after the blade is hidden
  6. ordinary tool use Gemini clears (cooking, cutting fruit) is not re-flagged for ~90 s
  7. prompt: acid, hidden weapons, screens/posters, holstered or sheathed items, tools
Safe to run twice."""
import re
import sys

OLD_BLOCK = '                if obj is not None:\n                    hits = st["whits"].setdefault(obj["label"], [])\n                    if not hits or hits[-1] != obj_cache[0]:\n                        hits.append(obj_cache[0])            # one hit per detection pass, per label\n                for lbl in list(st["whits"]):\n                    st["whits"][lbl] = [h for h in st["whits"][lbl] if ft - h <= WEAPON_HIT_WINDOW_S]\n                    if not st["whits"][lbl]:\n                        del st["whits"][lbl]\n                obj_label, held, raised = "", False, False\n                extra: dict[str, float] = {}\n                if obj is not None and (obj["held"] or obj["label"] in SHARP):\n                    need = 1 if (p_score >= 0.6 or obj["raised"]) else WEAPON_MIN_HITS   # motion corroborates\n                    if len(st["whits"].get(obj["label"], [])) >= need:\n                        obj_label, held, raised = obj["label"], obj["held"], obj["raised"]\n                        if held and obj_label in SHARP:\n                            extra["weapon_held"] = 1.0\n                        elif raised:\n                            extra["object_held"] = 1.0       # held up / cocked, not just carried\n                obj_alert = obj_label in SHARP or raised\n'

NEW_BLOCK = '                if obj is not None:\n                    st["wev"].add(obj_cache[0], obj["label"], obj["conf"])      # evidence builds up across passes\n                obj_label, held, raised = "", False, False\n                extra: dict[str, float] = {}\n                # only an object IN a hand counts (a knife on the table beside her does not). If the wrists are not\n                # visible we cannot tell, so a blade is still taken seriously.\n                if obj is not None and (obj["held"] or (obj["uncertain"] and obj["label"] in SHARP)):\n                    corroborated = p_score >= 0.6 or obj["raised"]              # motion / raised object lowers the bar\n                    cooking = st["benign_until"] > ft and not (obj["raised"] or p_score >= 0.5)\n                    if st["wev"].trusted(ft, obj["label"], corroborated) and not cooking:\n                        obj_label, held, raised = obj["label"], obj["held"], obj["raised"]\n                        if held and obj_label in SHARP:\n                            extra["weapon_held"] = 1.0\n                        elif raised:\n                            extra["object_held"] = 1.0       # held up / cocked, not just carried\n                obj_alert = obj_label in SHARP or raised\n'

# (name, old, new, is_regex)
EDITS = [
    ("import", "GestureHold, hand_crop,\n)", "GestureHold, hand_crop,\n    WeaponEvidence, active_wrist, wrist_boxes,\n)", False),
    ("constants", "WEAPON_HIT_WINDOW_S = 2.5\n",
     "WEAPON_HIT_WINDOW_S = 2.5\n"
     "OBJ_HAND_PASS = os.getenv(\"OBJ_HAND_PASS\", \"1\") == \"1\"       # detect objects inside close-ups around each wrist\n"
     "OBJ_HAND_MAX_DIST_M = float(os.getenv(\"OBJ_HAND_MAX_DIST_M\", \"7.0\"))\n"
     "OBJ_HAND_MAX_PEOPLE = int(os.getenv(\"OBJ_HAND_MAX_PEOPLE\", \"2\"))\n"
     "OBJ_HAND_IMGSZ = int(os.getenv(\"OBJ_HAND_IMGSZ\", \"256\"))\n"
     "OBJ_HAND_CONF_SCALE = 0.6        # per-label floors x this inside a hand close-up (strong context)\n"
     "ARMED_MEMORY_S = float(os.getenv(\"ARMED_MEMORY_S\", \"20\"))   # stays MODERATE this long after Gemini saw a weapon\n"
     "BENIGN_S = float(os.getenv(\"BENIGN_S\", \"90\"))               # tool use Gemini cleared (cooking...): don't re-flag\n", False),
    ("track", '"gest": GestureHold()', '"gest": GestureHold(), "wev": WeaponEvidence(), "armed_until": 0.0, "benign_until": 0.0', False),
    ("signature", "def detect_objects(frame: np.ndarray, near_boxes: list) -> list:",
     "def detect_objects(frame: np.ndarray, near_boxes: list, hand_boxes: list = ()) -> list:", False),
    ("handpass", "    return merge_objects(found)",
     "    if hand_boxes:      # close-ups around the wrists: a knife that is 12 px in the frame is 40+ px here\n"
     "        with torch.inference_mode():\n"
     "            h, w = frame.shape[:2]\n"
     "            crops, offs = [], []\n"
     "            for x1, y1, x2, y2 in hand_boxes:\n"
     "                x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))\n"
     "                if x2 - x1 >= 24 and y2 - y1 >= 24:\n"
     "                    crops.append(np.ascontiguousarray(frame[y1:y2, x1:x2]))\n"
     "                    offs.append((x1, y1))\n"
     "            if crops:\n"
     "                rs = _obj_model.predict(source=crops, classes=_HAND_OBJ_IDS, imgsz=OBJ_HAND_IMGSZ, conf=0.12,\n"
     "                                        device=DEVICE, verbose=False)\n"
     "                for r, (dx, dy) in zip(rs, offs):\n"
     "                    found += parse_objects(r, dx, dy, OBJ_HAND_CONF_SCALE)\n"
     "    return merge_objects(found)", False),
    ("handboxes", 'near = [d["box"] for d in sorted(dets, key=lambda d: d["dist"]) if d["dist"] <= OBJ_CROP_MAX_DIST_M]',
     'near = [d["box"] for d in sorted(dets, key=lambda d: d["dist"]) if d["dist"] <= OBJ_CROP_MAX_DIST_M]\n'
     '                hand_boxes = (wrist_boxes([d for d in sorted(dets, key=lambda d: d["dist"])\n'
     '                                           if d["dist"] <= OBJ_HAND_MAX_DIST_M][:OBJ_HAND_MAX_PEOPLE], w_img, h_img)\n'
     '                              if OBJ_HAND_PASS else [])', False),
    ("callsite", r"detect_objects, (inf_frame|frame), near\)", r"detect_objects, \1, near, hand_boxes)", True),
    ("evidence", OLD_BLOCK, NEW_BLOCK, False),
    ("handimg", '                    prio = ({"critical": 2, "rank1": 1, "scan": 0}[kind], int(obj_alert), -dist)',
     '                    if hand_img is None and (obj_alert or p_score >= 0.5 or "object_swing" in cats):\n'
     '                        hand_img = hand_crop(frame, active_wrist(kxy, kcf), bh)   # let Gemini see what is in the hand\n'
     '                    prio = ({"critical": 2, "rank1": 1, "scan": 0}[kind], int(obj_alert), -dist)', False),
    ("handtext", '"Hand close-up: the raised hand of the person in the red box. Judge whether it makes an "\n'
                 '                             "offensive or obscene gesture aimed at the wearer (middle finger, crude sexual sign).",',
     '"Hand close-up of the person in the red box. Check what is held in it (blade, gun, heavy object, bottle or "\n'
     '                             "container of liquid) and whether it makes an offensive or obscene gesture aimed at the wearer.",', False),
    ("company", "                v = verdicts.get(key)\n",
     "                v = verdicts.get(key)\n"
     "                if company and v and v[5] == \"harass\" and v[1] == \"suspicious\":\n"
     "                    v = None          # with friends / family: teasing and gestures are not harassment\n", False),
    ("memory", 'rank, score, reason, cleared = 0, min(score, 0.2), "", True   # e.g. chair carried on the head',
     'rank, score, reason, cleared = 0, min(score, 0.2), "", True   # e.g. chair carried on the head\n'
     '                # Gemini saw ordinary tool use (cooking, cutting fruit, shaving): stop re-flagging this person for a while\n'
     '                if (v and v[1] == "none" and v[2] >= 0.85 and set(cats) <= {"weapon_held"} and not raised\n'
     '                        and p_score < 0.3):\n'
     '                    st["benign_until"] = ft + BENIGN_S\n'
     '                    if rank == 1:\n'
     '                        rank, score, reason, cleared = 0, min(score, 0.2), "", True\n'
     '                # Gemini saw a weapon: this person stays flagged after the blade is hidden or out of view\n'
     '                if v and v[5] == "weapon" and v[1] in ("suspicious", "threat") and v[2] >= 0.6:\n'
     '                    st["armed_until"] = ft + ARMED_MEMORY_S\n'
     '                if ft < st["armed_until"] and rank < 1 and not cleared:\n'
     '                    rank, score, reason = 1, max(score, 0.5), "Seen with a weapon moments ago"', False),
    ("prompt_a", '"NOT threats: waving, stretching, dancing in a procession or wedding, selfies, friends hugging, a tap on the "',
     '"ACID AND WEAPONS OUT OF SIGHT: a bottle, cup, jug or bag of liquid held ready, raised toward her face or thrown "\n'
     '    "is an acid-attack cue (\'threat\'). A blade, gun or heavy object in a hand (even small, partly hidden or only "\n'
     '    "visible in the hand close-up) is \'threat\' when raised, pointed, swung or presented at someone, otherwise "\n'
     '    "\'suspicious\' if the hand is hidden or the grip looks ready.\\n\\n"\n'
     '    "NOT threats: waving, stretching, dancing in a procession or wedding, selfies, friends hugging, a tap on the "', False),
    ("prompt_b", '"cross a road, someone asking directions who leaves when she walks on, a tense face alone, ordinary walking.\\n\\n"',
     '"cross a road, someone asking directions who leaves when she walks on, a tense face alone, ordinary walking; "\n'
     '    "people, weapons or violence shown on screens, posters, hoardings, mirrors, statues or mannequins; holstered, "\n'
     '    "sheathed or slung items that are not drawn or pointed; tools or cutlery used for their purpose (cooking, "\n'
     '    "cutting fruit, shaving, sewing) or carried low or on the shoulder; toy or prop weapons in obvious play.\\n\\n"', False),
]


def _count(src, old, rx):
    return len(re.findall(old, src)) if rx else src.count(old)


def main(path: str) -> int:
    src = open(path, encoding="utf-8").read()
    if "WeaponEvidence" in src:
        print("already patched")
        return 0
    if "GestureHold" not in src:
        print("NOT patched. Run tools/apply_gesture_patch.py first.")
        return 1
    missing = [n for n, old, _, rx in EDITS if _count(src, old, rx) != 1]
    if missing:
        print("NOT patched. These edit points were not found exactly once:", ", ".join(missing))
        return 1
    for _, old, new, rx in EDITS:
        src = re.sub(old, new, src, count=1) if rx else src.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(src)
    print(f"patched {len(EDITS)} places in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "app/payload.py"))