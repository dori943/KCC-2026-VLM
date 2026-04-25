"""Provider interfaces for obtaining Module 1 raw outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any


@dataclass
class ProviderResult:
    """Result returned by provider implementations."""

    raw_output: dict[str, Any]
    metadata: dict[str, Any]


class Module1Provider(Protocol):
    """Contract for Module 1 data providers."""

    def get_module1_output(
        self,
        image_path: Path | None = None,
        case_id: str | None = None,
        module1_output_path: Path | None = None,
    ) -> ProviderResult:
        """Return Module 1 raw output payload and metadata."""
