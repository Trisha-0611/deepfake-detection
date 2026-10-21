"""Unit tests for the properties in Table 5.3 that must hold exactly (TC4-TC7).
Run:  python test_explain.py   (or pytest)"""
import numpy as np

from explain import conflict_index, shapley_values
from face_pipeline import interpolate, smooth, track


def test_tc4_shapley_efficiency():
    g = lambda x: np.tanh(x["spatial"]) * 2 + x["frequency"] * x["temporal"] - 0.3 * x["spatial"] * x["temporal"]
    z = {"spatial": 1.7, "frequency": -0.4, "temporal": 2.2}
    base = {"spatial": -1.0, "frequency": -0.5, "temporal": -0.8}
    phi = shapley_values(g, z, base)
    assert abs(sum(phi.values()) - (g(z) - g(base))) < 1e-6


def test_tc5_tc6_conflict_bounds():
    assert abs(conflict_index([0, 0, 1]) - 1) < 1e-9
    assert abs(conflict_index([0, 1]) - 1) < 1e-9
    assert conflict_index([0.3, 0.3, 0.3]) == 0


def test_tc7_longest_track_selected():
    box = lambda x: (np.array([x, 0, x + 100, 100], np.float32), np.full((5, 2), x, np.float32))
    dets = [[box(0), box(500)], [box(2)], [box(4), box(500)], [box(6)]]
    assert sorted(track(dets)) == [0, 1, 2, 3]


def test_smoothing_stays_within_clips():
    pts = {0: np.zeros(2), 1: np.ones(2), 100: np.full(2, 9.0)}
    s = smooth(pts)
    assert np.allclose(s[0], 0.5) and np.allclose(s[100], 9.0)


def test_landmark_interpolation():
    filled = interpolate({0: np.zeros(2), 4: np.full(2, 4.0), 20: np.ones(2)}, range(21), max_gap=4)
    assert np.allclose(filled[1], 1.0) and np.allclose(filled[3], 3.0)
    assert 10 not in filled                  # gap 4 -> 20 is too long to bridge


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
