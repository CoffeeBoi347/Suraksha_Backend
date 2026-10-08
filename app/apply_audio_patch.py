"""Crowd-noise fix for vision.py.   python tools/apply_audio_patch.py app/vision.py
Checks every edit point first; if any is missing it changes NOTHING and says which. Safe to run twice."""
import sys

EDITS = [
    ("import", "    ApproachEpisodes, CrowdBaseline, FollowTest,",
               "    ApproachEpisodes, AudioBaseline, CrowdBaseline, FollowTest,"),
    ("constant", "AUDIO_ALERT_REPEAT_S = 10.0\n",
                 "AUDIO_ALERT_REPEAT_S = 10.0\n"
                 "AUDIO_SOS_MAX_PEOPLE = int(os.getenv(\"AUDIO_SOS_MAX_PEOPLE\", \"3\"))  # audio-only SOS countdown never starts in a crowd\n"),
    ("state", "    crowd = CrowdBaseline()\n",
              "    crowd = CrowdBaseline()\n    audio_bg = AudioBaseline()\n    audio_noisy = False                 # loud scene (crowd / festival): screams need a jump above it\n"),
    ("nonlocal", "scream_val, scream_ts, company\n", "scream_val, scream_ts, company, audio_noisy\n"),
    ("audio", '''                scream_val = max(0.0, min(1.0, float(msg["scream"])))
                scream_ts = time.time()
                engine.set_audio(scream_val)
''', '''                scream_ts = time.time()
                scream_val, audio_noisy = audio_bg.update(scream_ts, max(0.0, min(1.0, float(msg["scream"]))))
                engine.set_audio(scream_val)
'''),
    ("speech", '''                if lvl:
                    engine.set_verbal(lvl, words)''', '''                if lvl and not (audio_noisy and lvl < 2):   # loud scene: transcripts are garbage, abuse words unreliable
                    engine.set_verbal(lvl, words)'''),
    ("features", '"follow": follow_lvl, "user_speed": round(user_speed, 2),',
                 '"follow": follow_lvl, "user_speed": round(user_speed, 2), "audio_noisy": audio_noisy,'),
    ("alert", "if len(audio_hits) >= AUDIO_ALERT_HITS and now_w - audio_alert_sent >= AUDIO_ALERT_REPEAT_S:",
              "if (len(audio_hits) >= AUDIO_ALERT_HITS and not audio_noisy\n"
              "                    and now_w - audio_alert_sent >= AUDIO_ALERT_REPEAT_S):"),
    ("sos", "if AUTO_SOS_ENABLED and countdown_task is None and firing is None and now_w >= snooze_until:\n"
            "                    armed_frame, armed_reason = frame.copy(), \"Sustained scream detected\"",
            "if (AUTO_SOS_ENABLED and countdown_task is None and firing is None and now_w >= snooze_until\n"
            "                        and len(dets) <= AUDIO_SOS_MAX_PEOPLE):     # a crowd cheering is not an emergency\n"
            "                    armed_frame, armed_reason = frame.copy(), \"Sustained scream detected\""),
]


def main(path: str) -> int:
    src = open(path, encoding="utf-8").read()
    if "AudioBaseline" in src:
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