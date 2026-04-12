"""Provider interfaces for Module 3 input bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.utils import load_json, project_root


@dataclass(slots=True)
class Module3InputResult:
    bundle: dict[str, Any]
    metadata: dict[str, Any]


class Module3InputProvider(Protocol):
    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module3InputResult: ...


class FileInputProvider:
    """bundle.json을 직접 읽는 provider (테스트용)."""

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module3InputResult:
        if bundle_path is None:
            raise ValueError("FileInputProvider requires --bundle path.")
        payload = load_json(bundle_path)
        return Module3InputResult(
            bundle=payload,
            metadata={"provider": "file", "bundle_path": str(bundle_path), "case_id": case_id},
        )


class MockInputProvider:
    """fixtures에서 읽는 provider (테스트용)."""

    def __init__(self, fixtures_root: Path | None = None) -> None:
        self.cases_root = (fixtures_root or project_root() / "fixtures") / "module3_cases"

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module3InputResult:
        if case_id is None:
            raise ValueError("MockInputProvider requires --case-id.")
        case_dir = self.cases_root / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Unknown Module 3 fixture case_id: {case_id}")
        bundle = load_json(case_dir / "bundle.json")
        return Module3InputResult(
            bundle=bundle,
            metadata={"provider": "mock", "case_id": case_id,
                      "bundle_path": str(case_dir / "bundle.json")},
        )


class Module2DOutputProvider:
    """Module 2-D run_dir을 직접 읽어 Module 3 input으로 변환."""

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module3InputResult:
        if bundle_path is None:
            raise ValueError("Module2DOutputProvider requires --bundle (module2d run_dir).")

        p = Path(bundle_path)
        if p.is_dir():
            module2d_output = load_json(p / "module2d_output.json")
            raw_input_bundle = load_json(p / "raw_input_bundle.json")
        else:
            module2d_output = load_json(p)
            raw_input_bundle = load_json(p.parent / "raw_input_bundle.json")

        bundle = _convert_module2d_to_3_input(module2d_output, raw_input_bundle)

        return Module3InputResult(
            bundle=bundle,
            metadata={
                "provider": "module2d_output",
                "bundle_path": str(bundle_path),
                "case_id": case_id,
            },
        )


# ─── 변환 함수 ────────────────────────────────────────────────

def _convert_module2d_to_3_input(
    module2d_output: dict[str, Any],
    raw_input_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Module 2-D output + raw_input_bundle → Module 3 input 변환."""

    # selected_candidate_id로 후보 찾기
    selected_id = module2d_output.get("selected_candidate_id")
    evaluated = module2d_output.get("evaluated_candidates", [])

    selected_candidate = None
    filter_result = None

    # candidate_tools에서 selected_candidate 찾기
    for c in raw_input_bundle.get("candidate_tools", []):
        if c.get("candidate_id") == selected_id:
            selected_candidate = c
            break

    # evaluated_candidates에서 filter_result 찾기
    for e in evaluated:
        if e.get("candidate_id") == selected_id:
            filter_result = e
            break

    # pass=true인 후보가 없으면 최고점 후보 사용
    if selected_candidate is None and raw_input_bundle.get("candidate_tools"):
        selected_candidate = raw_input_bundle["candidate_tools"][0]
    if filter_result is None and evaluated:
        filter_result = evaluated[0]

    # scene_context 구성
    scene_objects = raw_input_bundle.get("scene_objects", [])
    scene_context = {
        "camera_prior": {
            "optical_center": [0.55, -0.35, 0.8],
            "tilt_down_deg": 40,
        },
        "scene_objects": scene_objects,
    }

    return {
        "task": raw_input_bundle.get("task", ""),
        "scene_context": scene_context,
        "object_physical_properties": raw_input_bundle.get("object_physical_properties", []),
        "tool_constraints": raw_input_bundle.get("tool_constraints", {}),
        "selected_candidate": selected_candidate or {},
        "filter_result": filter_result or {},
    }
