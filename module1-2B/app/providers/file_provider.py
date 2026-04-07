"""File-based provider for precomputed Module 1 raw outputs."""

from __future__ import annotations

from pathlib import Path

from app.providers.base import ProviderResult
from app.utils import load_json


class FileProvider:
    """Load Module 1 raw output from a file path."""

    def get_module1_output(
        self,
        image_path: Path | None = None,
        case_id: str | None = None,
        module1_output_path: Path | None = None,
    ) -> ProviderResult:
        """Return JSON payload loaded from module1_output_path."""
        if module1_output_path is None:
            raise ValueError("FileProvider requires --module1-output path.")
        payload = load_json(module1_output_path)
        return ProviderResult(
            raw_output=payload,
            metadata={
                "provider": "file",
                "image_path": str(image_path) if image_path else None,
                "case_id": case_id,
                "module1_output_path": str(module1_output_path),
            },
        )
