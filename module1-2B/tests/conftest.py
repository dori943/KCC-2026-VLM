from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.utils import load_json


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_raw_outputs(repo_root: Path) -> dict[str, Any]:
    base = repo_root / "fixtures" / "module1_raw_outputs"
    return {
        "rubber_ball_like_object": load_json(base / "rubber_ball_like_object.json"),
        "wooden_block_like_object": load_json(base / "wooden_block_like_object.json"),
        "mug_or_container_like_object": load_json(base / "mug_or_container_like_object.json"),
        "Gemini_Generated_Image_gvc8a5gvc8a5gvc8": load_json(
            base / "Gemini_Generated_Image_gvc8a5gvc8a5gvc8.json"
        ),
    }
