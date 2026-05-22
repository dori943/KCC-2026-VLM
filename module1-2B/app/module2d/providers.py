"""Provider interfaces for Module 2-D input bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.utils import load_json, project_root


@dataclass(slots=True)
class Module2DInputResult:
    bundle: dict[str, Any]
    metadata: dict[str, Any]


class Module2DInputProvider(Protocol):
    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2DInputResult: ...


class FileInputProvider:
    """bundle.json을 직접 읽는 provider (테스트용)."""

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2DInputResult:
        if bundle_path is None:
            raise ValueError("FileInputProvider requires --bundle path.")
        payload = load_json(bundle_path)
        return Module2DInputResult(
            bundle=payload,
            metadata={"provider": "file", "bundle_path": str(bundle_path), "case_id": case_id},
        )


class MockInputProvider:
    """fixtures에서 읽는 provider (테스트용)."""

    def __init__(self, fixtures_root: Path | None = None) -> None:
        self.cases_root = (fixtures_root or project_root() / "fixtures") / "module2d_cases"

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2DInputResult:
        if case_id is None:
            raise ValueError("MockInputProvider requires --case-id.")
        case_dir = self.cases_root / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Unknown Module 2-D fixture case_id: {case_id}")
        bundle = load_json(case_dir / "bundle.json")
        return Module2DInputResult(
            bundle=bundle,
            metadata={"provider": "mock", "case_id": case_id,
                      "bundle_path": str(case_dir / "bundle.json")},
        )


class Module2COutputProvider:
    """Module 2-C run_dir을 직접 읽어 Module 2-D input으로 변환."""

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2DInputResult:
        if bundle_path is None:
            raise ValueError("Module2COutputProvider requires --bundle (module2c run_dir).")

        p = Path(bundle_path)
        if p.is_dir():
            module2c_output = load_json(p / "module2c_output.json")
            raw_input_bundle = load_json(p / "raw_input_bundle.json")
        else:
            module2c_output = load_json(p)
            raw_input_bundle = load_json(p.parent / "raw_input_bundle.json")

        bundle = _convert_module2c_to_2d_input(module2c_output, raw_input_bundle)

        return Module2DInputResult(
            bundle=bundle,
            metadata={
                "provider": "module2c_output",
                "bundle_path": str(bundle_path),
                "case_id": case_id,
            },
        )


# ─── 변환 함수 ────────────────────────────────────────────────

def _convert_module2c_to_2d_input(
    module2c_output: dict[str, Any],
    raw_input_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Module 2-C output + raw_input_bundle → Module 2-D input 변환.

    raw_input_bundle은 Module 2-C의 raw_input_bundle.json으로,
    task / tool_constraints / scene_objects / object_physical_properties가 그대로 담겨있음.
    여기에 Module 2-C가 생성한 candidate_tools만 추가하면 2-D input 완성.
    """
    return {
        "task": raw_input_bundle.get("task", ""),
        "tool_constraints": raw_input_bundle.get("tool_constraints", {}),
        "candidate_tools": module2c_output.get("candidate_tools", []),
        "scene_objects": raw_input_bundle.get("scene_objects", []),
        "object_physical_properties": raw_input_bundle.get("object_physical_properties", []),
    }
