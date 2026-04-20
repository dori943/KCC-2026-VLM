"""Provider interfaces and implementations for Module 2-B input bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.utils import load_json, project_root


@dataclass(slots=True)
class Module2BBundleResult:
    """Resolved Module 2-B input bundle and metadata."""

    bundle: dict[str, Any]
    metadata: dict[str, Any]


class Module2BBundleProvider(Protocol):
    """Contract for providers that return module2_common_input + module2a_output bundle."""

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        module2_common_path: Path | None = None,
        module2a_output_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2BBundleResult:
        """Return a resolved bundle dictionary."""


class FileBundleProvider:
    """Load Module 2-B bundle from file(s)."""

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        module2_common_path: Path | None = None,
        module2a_output_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2BBundleResult:
        """Load bundle from one file or two split files."""
        if bundle_path is not None:
            payload = load_json(bundle_path)
            return Module2BBundleResult(
                bundle=payload,
                metadata={
                    "provider": "file",
                    "mode": "bundle",
                    "bundle_path": str(bundle_path),
                    "case_id": case_id,
                },
            )

        if module2_common_path is None or module2a_output_path is None:
            raise ValueError(
                "FileBundleProvider requires --bundle or both --module2-common and --module2a-output."
            )

        module2_common_input = load_json(module2_common_path)
        module2a_output = load_json(module2a_output_path)
        return Module2BBundleResult(
            bundle={
                "module2_common_input": module2_common_input,
                "module2a_output": module2a_output,
            },
            metadata={
                "provider": "file",
                "mode": "split",
                "module2_common_path": str(module2_common_path),
                "module2a_output_path": str(module2a_output_path),
                "case_id": case_id,
            },
        )


class MockBundleProvider:
    """Fixture-backed Module 2-B provider with deterministic case resolution."""

    def __init__(self, fixtures_root: Path | None = None) -> None:
        self.fixtures_root = fixtures_root or (project_root() / "fixtures")
        self.cases_root = self.fixtures_root / "module2b_cases"

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        module2_common_path: Path | None = None,
        module2a_output_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2BBundleResult:
        """Load bundle from fixture case id."""
        if case_id is None:
            raise ValueError("MockBundleProvider requires --case-id.")

        case_dir = self.cases_root / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Unknown Module 2-B fixture case_id: {case_id}")

        bundle = load_json(case_dir / "bundle.json")
        return Module2BBundleResult(
            bundle=bundle,
            metadata={
                "provider": "mock",
                "mode": "fixture_bundle",
                "case_id": case_id,
                "bundle_path": str(case_dir / "bundle.json"),
            },
        )


def list_module2b_case_ids(fixtures_root: Path | None = None) -> list[str]:
    """Return declared Module 2-B fixture case IDs."""
    root = fixtures_root or (project_root() / "fixtures")
    index = load_json(root / "module2b_cases" / "index.json")
    return list(index.get("cases", []))
