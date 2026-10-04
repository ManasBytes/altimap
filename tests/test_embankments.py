import numpy as np

from viewer import embankments as emb


def test_levee_crest_survives_the_bare_earth_opening():
    # 1 m pixels: a 30 m-wide levee 8 m tall along column 100, which the bare-earth opening erased.
    dem = np.full((50, 200), 5.0, np.float32)
    dem[:, 85:115] = 13.0
    opened = np.full_like(dem, 5.0)
    w = emb.corridor([np.array([[100.0, 0.0], [100.0, 49.0]])], dem.shape, 1.0)
    assert w[25, 100] == 1.0 and w[25, 100 + 25] == 1.0  # full weight on the crest and slopes
    assert w[25, 100 + 45] == 0.0  # beyond half-width + feather: untouched
    ground = emb.keep_in_ground(opened, dem, w)
    assert ground[25, 100] == 13.0  # the crest is back
    assert (ground >= opened).all()  # never lowered anywhere
    assert np.array_equal(ground[:, :50], opened[:, :50])  # the rest of the scene unchanged


def test_embankment_kinds_follow_their_tags():
    assert emb.kind({"man_made": "dyke"}) == "levee"
    assert emb.kind({"embankment": "yes", "highway": "primary"}) == "road embankment"
    assert emb.kind({"embankment": "yes", "railway": "rail"}) == "rail embankment"
    assert emb.kind({"man_made": "embankment"}) == "embankment"
    assert emb.kind({"waterway": "dam"}) == "dam"
