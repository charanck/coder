from __future__ import annotations

import pytest

from tests.support.helpers import build_sample_workspace


@pytest.fixture()
def sample_workspace(tmp_path):
    build_sample_workspace(tmp_path)
    return tmp_path