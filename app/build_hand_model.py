"""One-time: bake an open-vocabulary hand-object detector.   python tools/build_hand_model.py [base weights]
pip install -U ultralytics   (this script pulls CLIP once; the runtime does not need it)
Writes models/hand-world.pt; the hand pass picks it up automatically (HAND_MODEL env to override)."""
import os
import sys

from ultralytics import YOLO

WEAPONS = ["knife", "kitchen knife", "machete", "sickle", "meat cleaver", "dagger", "sword", "box cutter",
           "scissors", "pistol", "revolver", "rifle", "axe", "hammer", "metal rod", "wooden stick", "baseball bat",
           "brick", "glass bottle", "plastic bottle", "cup"]
# Decoys: what hands usually hold. With class-agnostic NMS they beat 'knife' on a phone / pen / torch (the commonest
# false alarm). canon() maps them to None, so they are dropped and never alert.
DECOYS = ["mobile phone", "wallet", "pen", "comb", "spoon", "keys", "remote control", "cigarette", "toothbrush",
          "flashlight", "umbrella handle", "bag strap"]

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "yolov8s-worldv2.pt"   # m-worldv2 if the GPU has room
    m = YOLO(base)
    m.set_classes(WEAPONS + DECOYS)
    os.makedirs("models", exist_ok=True)
    m.save("models/hand-world.pt")
    print("saved models/hand-world.pt with", len(WEAPONS), "weapon +", len(DECOYS), "decoy classes")