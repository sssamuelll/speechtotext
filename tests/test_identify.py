import numpy as np

from speechtotext.speakers.identify import cosine, assign_names


def test_cosine_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == 1.0


def test_assign_names_matches_closest_above_threshold():
    enrolled = {"Alice": np.array([1.0, 0.0]), "Bob": np.array([0.0, 1.0])}
    clusters = {"SPEAKER_00": np.array([0.9, 0.1]), "SPEAKER_01": np.array([0.1, 0.9])}
    got = assign_names(clusters, enrolled, threshold=0.5)
    assert got == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


def test_assign_names_below_threshold_unmatched():
    enrolled = {"Alice": np.array([1.0, 0.0])}
    clusters = {"SPEAKER_00": np.array([0.0, 1.0])}  # orthogonal -> cosine 0
    assert assign_names(clusters, enrolled, threshold=0.5) == {}


def test_assign_names_no_collision_same_name():
    # Two clusters similar to Alice: only the best one gets the name.
    enrolled = {"Alice": np.array([1.0, 0.0])}
    clusters = {"SPEAKER_00": np.array([1.0, 0.0]), "SPEAKER_01": np.array([0.8, 0.2])}
    got = assign_names(clusters, enrolled, threshold=0.5)
    assert got == {"SPEAKER_00": "Alice"}
