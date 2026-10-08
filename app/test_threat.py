from app.threat import fuse

ISO = {"isolated": True}


def test_raised_object_alone_is_moderate():
    assert fuse({"object_held": 1.0}, {})[0] == 1


def test_weak_harassment_trio_is_moderate_not_critical():
    assert fuse({"stare": 1, "linger": 1, "verbal_abuse": 1}, {})[0] == 1


def test_three_real_cues_still_critical():
    assert fuse({"converge": 1, "follow": 1, "engage": 1}, ISO)[0] == 2
    assert fuse({"converge_fast": 1, "follow_strong": 1, "ambush": 1}, {})[0] == 2


def test_sharp_weapon_and_swing_unchanged():
    assert fuse({"weapon_held": 1, "pose": 1}, {})[0] == 2
    assert fuse({"object_swing": 1, "strike": 1}, {})[0] == 2
    assert fuse({"weapon_held": 1}, {})[0] == 1


def test_stare_alone_still_nothing():
    assert fuse({"stare": 1}, ISO)[0] == 0