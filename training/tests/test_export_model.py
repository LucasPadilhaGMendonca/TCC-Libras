import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_model import validate_feature_count  # noqa: E402


def test_validate_feature_count_accepts_expected_count():
    validate_feature_count(42)  # não deve levantar


def test_validate_feature_count_rejects_wrong_count():
    with pytest.raises(SystemExit):
        validate_feature_count(10)
