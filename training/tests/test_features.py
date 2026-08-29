import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_dataset import build_features  # noqa: E402


def landmark(x, y):
    return SimpleNamespace(x=x, y=y)


def test_build_features_returns_42_values_for_21_landmarks():
    hand = [landmark(0.1 * i, 0.2 * i) for i in range(21)]

    features = build_features(hand)

    assert features is not None
    assert len(features) == 42


def test_build_features_normalizes_by_subtracting_min():
    hand = [landmark(0.5, 0.9), landmark(0.2, 0.4), landmark(0.8, 0.6)] + [
        landmark(0.5, 0.5) for _ in range(18)
    ]

    features = build_features(hand)

    # x0 - min(x) = 0.5 - 0.2 = 0.3 ; y0 - min(y) = 0.9 - 0.4 = 0.5
    assert features[0] == pytest.approx(0.3)
    assert features[1] == pytest.approx(0.5)
    # o menor x e o menor y do conjunto viram sempre 0
    assert min(features[0::2]) == pytest.approx(0.0)
    assert min(features[1::2]) == pytest.approx(0.0)


def test_build_features_rejects_wrong_landmark_count():
    hand = [landmark(0.1, 0.1) for _ in range(5)]

    assert build_features(hand) is None
