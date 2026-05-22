"""Mock provider that loads fixture cases without external API calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.providers.base import ProviderResult
from app.utils import load_json, project_root


class MockProvider:
    """Fixture-backed provider for local and deterministic runs."""

    def __init__(self, fixtures_root: Path | None = None) -> None:
        self.fixtures_root = fixtures_root or project_root() / "fixtures"
        self.cases_root = self.fixtures_root / "cases"

    def _resolve_case_id(self, image_path: Path | None, case_id: str | None) -> str:
        if case_id:
            return case_id
        if image_path:
            return image_path.stem
        raise ValueError("MockProvider requires either case_id or image_path.")

    def _load_case_meta(self, case_id: str) -> dict[str, Any]:
        path = self.cases_root / f"{case_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown fixture case_id: {case_id}")
        return load_json(path)

    def get_module1_output(
        self,
        image_path: Path | None = None,
        case_id: str | None = None,
        module1_output_path: Path | None = None,
    ) -> ProviderResult:
        """Load fixture raw output using case metadata."""
        resolved_case = self._resolve_case_id(image_path=image_path, case_id=case_id)
        case_meta = self._load_case_meta(resolved_case)
        raw_path = project_root() / case_meta["module1_output_path"]
        payload = load_json(raw_path)
        return ProviderResult(
            raw_output=payload,
            metadata={
                "provider": "mock",
                "case_id": resolved_case,
                "image_path": case_meta["image_path"],
                "module1_output_path": case_meta["module1_output_path"],
            },
        )
