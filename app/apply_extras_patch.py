#!/usr/bin/env python3
"""Wire face-blur, server-side audio classification, and weapon model into vision.py.

Run from the project root:
    python tools/apply_extras_patch.py            # dry-run (prints the diff)
    python tools/apply_extras_patch.py --apply    # write the patched file

What it does (all are additive; existing behaviour is untouched):

1. FACE BLUR  — Imports app.faceblur and calls blur_bystanders inside
   _context_image so every scene snapshot sent to Gemini has bystander
   faces blurred (privacy). The target person's face is NOT blurred.

2. SERVER-SIDE AUDIO CLASSIFICATION — Imports app.audio_classify and adds
   a new control-message handler for "audio_raw" that decodes a base64 PCM
   chunk, runs PANNs (or the energy-based fallback), and injects the
   detected events into the engine exactly like the existing client-side
   "audio" handler.  The existing "audio" handler is left untouched.

3. WEAPON MODEL NOTE — Already wired via WEAPON_WEIGHTS env var in
   vision.py line 110. The patch adds a startup log line so the operator
   can confirm which weights are loaded.

All patches are idempotent: running twice does nothing the second time.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

VISION = Path("app/vision.py")

# ──────────────────────────────── patches ────────────────────────────────

def _already(src: str, marker: str) -> bool:
    return marker in src


def patch_faceblur_import(src: str) -> str:
    """Add faceblur import after the reid import."""
    MARKER = "from app.faceblur import"
    if _already(src, MARKER):
        return src
    anchor = "from app.reid import PersonGallery, embed_person"
    if anchor not in src:
        print("  WARN: reid import anchor not found; skipping face-blur import patch")
        return src
    return src.replace(anchor, anchor + "\n" + MARKER + " blur_bystanders")


def patch_context_image(src: str) -> str:
    """Call blur_bystanders inside _context_image so Gemini sees blurred bystanders."""
    MARKER = "blur_bystanders"
    # Check if the function already has blur
    # Find _context_image function body
    fn_pattern = r'(def _context_image\(frame: np\.ndarray, box\) -> np\.ndarray:.*?\n)'
    m = re.search(fn_pattern, src)
    if not m:
        print("  WARN: _context_image signature not found; skipping face-blur body patch")
        return src

    # The body after the docstring — we insert blur after the copy
    old = '    img = _shrink(frame, CONTEXT_MAX_SIDE).copy()'
    if old not in src:
        print("  WARN: _context_image body anchor not found; skipping")
        return src
    if _already(src, "blur_bystanders(img"):
        return src

    new = old + """
    # privacy: blur bystander faces (target stays unblurred)
    s = img.shape[1] / frame.shape[1]
    _bx = tuple(int(v * s) for v in box)
    img = blur_bystanders(img, _bx)"""
    return src.replace(old, new, 1)


def patch_audio_classify_import(src: str) -> str:
    """Add audio_classify import."""
    MARKER = "from app.audio_classify import"
    if _already(src, MARKER):
        return src
    anchor = "from app.faceblur import blur_bystanders"
    if anchor not in src:
        # try without faceblur (in case patches are applied individually)
        anchor = "from app.reid import PersonGallery, embed_person"
    if anchor not in src:
        print("  WARN: import anchor not found; skipping audio_classify import")
        return src
    return src.replace(anchor, anchor + "\n" + MARKER + " get_classifier as get_audio_classifier")


def patch_audio_raw_handler(src: str) -> str:
    """Add an 'audio_raw' handler in handle_control for server-side audio classification."""
    MARKER = 'kind == "audio_raw"'
    if _already(src, MARKER):
        return src
    # Insert right after the existing "audio" handler block.
    # Anchor: the speech handler line
    anchor = '        elif kind == "speech":  # {"type":"speech","text":"..."}  OTHER people\'s speech; never stored'
    if anchor not in src:
        print("  WARN: speech handler anchor not found; skipping audio_raw patch")
        return src

    new_block = '''        elif kind == "audio_raw":  # {"type":"audio_raw","pcm_b64":"...","sr":16000}
            # Server-side audio classification: client sends raw PCM, we run PANNs
            import base64
            pcm_b64 = msg.get("pcm_b64")
            if isinstance(pcm_b64, str):
                try:
                    pcm_bytes = base64.b64decode(pcm_b64)
                    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
                    audio_sr = int(msg.get("sr", 16000))
                    clf = get_audio_classifier()
                    events = clf.classify(audio_np, sr=audio_sr)
                    for evt_label, evt_conf in events:
                        if evt_label == "scream" and evt_conf >= 0.5:
                            scream_val = max(scream_val, evt_conf)
                            scream_ts = time.time()
                            engine.set_audio(scream_val)
                            if scream_val >= AUDIO_ALERT_MIN:
                                audio_hits.append(scream_ts)
                                audio_hits[:] = [s for s in audio_hits if scream_ts - s <= AUDIO_ALERT_WINDOW_S]
                        elif evt_label == "gunshot" and evt_conf >= 0.55:
                            engine.set_audio(evt_conf)  # treat gunshot like a high-urgency audio event
                except Exception as _ae:
                    log.debug("audio_raw decode/classify error: %s", _ae)
'''
    return src.replace(anchor, new_block + anchor)


def patch_weapon_log(src: str) -> str:
    """Add a startup log line showing which weapon weights are loaded."""
    MARKER = "Weapon model:"
    if _already(src, MARKER):
        return src
    anchor = '_obj_model = YOLO(OBJ_WEIGHTS) if OBJ_DETECT else None'
    if anchor not in src:
        print("  WARN: _obj_model init not found; skipping weapon log patch")
        return src
    return src.replace(
        anchor,
        anchor + '\nif _obj_model is not None:\n    log.info("Weapon model: %s (classes: %s)", OBJ_WEIGHTS, getattr(_obj_model, "names", "?"))')


def patch_person_crop_blur(src: str) -> str:
    """Also blur bystanders in person crops sent to Gemini (the trigger-moment crop)."""
    MARKER = "blur_bystanders(trig"
    if _already(src, MARKER):
        return src
    # Find the trigger crop section in the Gemini dispatch
    anchor = "                if trig is not None:\n                    crops = crops[-max(1, TEMPORAL_FRAME_COUNT - 1):] + [trig]"
    if anchor not in src:
        print("  WARN: trigger-crop anchor not found; skipping person-crop blur patch")
        return src
    new = anchor.replace(
        "                if trig is not None:",
        "                if trig is not None:\n                    trig = blur_bystanders(trig, (0, 0, trig.shape[1], trig.shape[0]))  # full-crop: target IS the crop"
    )
    # Actually we should NOT blur the target crop — the target IS the crop,
    # so blur_bystanders with target_box == full crop will skip all faces
    # inside the crop (IoU > 0.1 with the full image). This is correct:
    # nobody in the person crop is a "bystander".
    # So actually this is a no-op for the person crop. Skip this patch.
    return src


# ──────────────────────────────── main ────────────────────────────────

PATCHES = [
    ("Face-blur import", patch_faceblur_import),
    ("Audio-classify import", patch_audio_classify_import),
    ("_context_image face blur", patch_context_image),
    ("audio_raw handler", patch_audio_raw_handler),
    ("Weapon model log", patch_weapon_log),
]


def main():
    ap = argparse.ArgumentParser(description="Wire extras into vision.py")
    ap.add_argument("--apply", action="store_true", help="Write the patched file (default: dry-run)")
    args = ap.parse_args()

    if not VISION.exists():
        print(f"ERROR: {VISION} not found. Run from the project root.")
        sys.exit(1)

    orig = VISION.read_text()
    src = orig

    for name, fn in PATCHES:
        before = src
        src = fn(src)
        status = "applied" if src != before else "already present / skipped"
        print(f"  [{status}] {name}")

    if src == orig:
        print("\nNothing to do — all patches already applied.")
        return

    if args.apply:
        bak = VISION.with_suffix(".py.bak_extras")
        shutil.copy2(VISION, bak)
        VISION.write_text(src)
        print(f"\nPatched {VISION}  (backup: {bak})")
    else:
        # Show a simple diff summary
        orig_lines = orig.splitlines(keepends=True)
        new_lines = src.splitlines(keepends=True)
        added = len(new_lines) - len(orig_lines)
        print(f"\nDry run: {added:+d} lines.  Re-run with --apply to write.")


if __name__ == "__main__":
    main()