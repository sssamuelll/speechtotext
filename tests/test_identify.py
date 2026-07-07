import numpy as np

from speechtotext.speakers.identify import cosine, assign_names


def test_cosine_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == 1.0


def test_assign_names_matches_closest_above_threshold():
    enrolled = {"Samuel": np.array([1.0, 0.0]), "Ale": np.array([0.0, 1.0])}
    clusters = {"SPEAKER_00": np.array([0.9, 0.1]), "SPEAKER_01": np.array([0.1, 0.9])}
    got = assign_names(clusters, enrolled, threshold=0.5)
    assert got == {"SPEAKER_00": "Samuel", "SPEAKER_01": "Ale"}


def test_assign_names_below_threshold_unmatched():
    enrolled = {"Samuel": np.array([1.0, 0.0])}
    clusters = {"SPEAKER_00": np.array([0.0, 1.0])}  # ortogonal -> coseno 0
    assert assign_names(clusters, enrolled, threshold=0.5) == {}


def test_assign_names_no_collision_same_name():
    # dos clusters parecidos a Samuel: solo el mejor se lleva el nombre
    enrolled = {"Samuel": np.array([1.0, 0.0])}
    clusters = {"SPEAKER_00": np.array([1.0, 0.0]), "SPEAKER_01": np.array([0.8, 0.2])}
    got = assign_names(clusters, enrolled, threshold=0.5)
    assert got == {"SPEAKER_00": "Samuel"}
