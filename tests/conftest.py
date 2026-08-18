"""Shared pytest fixtures.

Tests that need real dataset files are skipped unless the MYOKIND_DATASET_ROOT
environment variable points at a local ``Dataset/`` folder (the one directly
containing ``Sub001/`` ... ``Sub021/``). This repository never ships the
dataset itself, so these tests cannot run out of the box - only the
synthetic-fixture tests do.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def dataset_root() -> Path:
    root = os.environ.get("MYOKIND_DATASET_ROOT")
    if not root or not Path(root).is_dir():
        pytest.skip("Set MYOKIND_DATASET_ROOT to a local Dataset/ folder to run tests against real data")
    return Path(root)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
