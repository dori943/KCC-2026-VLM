"""Utility helpers shared across the experiment harness."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml as _yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _yaml = None


def project_root() -> Path:
    """Return repository root assuming this file is in app/."""
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_id() -> str:
    """Build a filesystem-friendly timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_unique_run_dir(output_root: Path, stem: str) -> Path:
    """Create a unique run directory under output_root based on stem."""
    candidate = output_root / stem
    if not candidate.exists():
        return ensure_dir(candidate)
    index = 1
    while True:
        fallback = output_root / f"{stem}_{index:02d}"
        if not fallback.exists():
            return ensure_dir(fallback)
        index += 1


def derive_task_name(
    task_name: str | None = None,
    bundle_path: Path | None = None,
    case_id: str | None = None,
) -> str:
    """Resolve a stable task name for per-task output routing."""
    if task_name and task_name.strip():
        return _slugify_task_name(task_name)
    if case_id and case_id.strip():
        return _slugify_task_name(case_id)
    if bundle_path is not None:
        path = Path(bundle_path)
        if path.is_dir():
            run_name = path.name
            if re.match(r"^module[0-9a-z]+_[0-9]{8}_[0-9]{6}_.+$", run_name):
                return _slugify_task_name(path.parent.name)
            return _slugify_task_name(run_name)
        if path.parent.name:
            return _slugify_task_name(path.parent.name)
        if path.stem:
            return _slugify_task_name(path.stem)
    return "ad_hoc"


def build_task_output_root(output_root: Path, task_name: str) -> Path:
    """Return outputs/<task_name> directory and ensure it exists."""
    return ensure_dir(output_root / _slugify_task_name(task_name))


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file as dictionary."""
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    """Write JSON file with stable formatting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML as dictionary."""
    text = path.read_text(encoding="utf-8")
    if _yaml is not None:
        loaded = _yaml.safe_load(text)
    else:
        loaded = _simple_yaml_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be object: {path}")
    return loaded


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to CSV; headers from union of row keys."""
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: list[str] = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clamp(value: float, low: float, high: float) -> tuple[float, bool]:
    """Clamp value and return (clamped_value, was_clamped)."""
    clamped = max(low, min(high, value))
    return clamped, clamped != value


def get_path(data: Any, dotted_path: str, default: Any = None) -> Any:
    """Get nested value by dotted path from dictionaries."""
    current: Any = data
    for token in dotted_path.split("."):
        if isinstance(current, dict) and token in current:
            current = current[token]
        else:
            return default
    return current


def to_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _slugify_task_name(value: str) -> str:
    """Normalize user/task labels into filesystem-safe folder names."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "ad_hoc"


def _simple_yaml_load(text: str) -> Any:
    """Minimal YAML parser fallback for this project's config subset.

    Supports:
    - nested dict/list by indentation (2 spaces)
    - inline lists: [a, b]
    - inline dicts: { key: value, ... }
    - quoted and unquoted scalar values
    """

    lines = [
        line.rstrip("\n")
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    index = 0

    def parse_block(indent: int, force_list: bool | None = None) -> Any:
        nonlocal index
        container: Any = [] if force_list else {}
        while index < len(lines):
            line = lines[index]
            current_indent = len(line) - len(line.lstrip(" "))
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Invalid indentation near line: {line}")
            stripped = line.strip()

            if stripped.startswith("- "):
                if not isinstance(container, list):
                    if container == {}:
                        container = []
                    else:
                        raise ValueError(f"Unexpected list item at line: {line}")
                item_text = stripped[2:].strip()
                index += 1
                if item_text == "":
                    child = parse_block(indent + 2, _peek_is_list(indent + 2))
                    container.append(child)
                    continue
                if _looks_like_key_value(item_text) and not (
                    item_text.startswith("{") and item_text.endswith("}")
                ):
                    item_dict: dict[str, Any] = {}
                    key, value_text = _split_key_value(item_text)
                    if value_text == "":
                        item_dict[key] = parse_block(indent + 4, _peek_is_list(indent + 4))
                    else:
                        item_dict[key] = _parse_scalar(value_text)
                    while index < len(lines):
                        next_line = lines[index]
                        next_indent = len(next_line) - len(next_line.lstrip(" "))
                        next_stripped = next_line.strip()
                        if next_indent < indent + 2:
                            break
                        if next_indent == indent + 2 and next_stripped.startswith("- "):
                            break
                        if next_indent != indent + 2:
                            raise ValueError(f"Unsupported nested indentation near line: {next_line}")
                        key2, value_text2 = _split_key_value(next_stripped)
                        index += 1
                        if value_text2 == "":
                            item_dict[key2] = parse_block(
                                indent + 4, _peek_is_list(indent + 4)
                            )
                        else:
                            item_dict[key2] = _parse_scalar(value_text2)
                    container.append(item_dict)
                else:
                    container.append(_parse_scalar(item_text))
                continue

            if isinstance(container, list):
                raise ValueError(f"Unexpected dict entry in list at line: {line}")
            key, value_text = _split_key_value(stripped)
            index += 1
            if value_text == "":
                container[key] = parse_block(indent + 2, _peek_is_list(indent + 2))
            else:
                container[key] = _parse_scalar(value_text)
        return container

    def _peek_is_list(indent: int) -> bool:
        nonlocal index
        if index >= len(lines):
            return False
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        return current_indent == indent and line.strip().startswith("- ")

    return parse_block(0, None)


def _looks_like_key_value(text: str) -> bool:
    return ":" in text


def _split_key_value(text: str) -> tuple[str, str]:
    parts = text.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid key/value line: {text}")
    key = parts[0].strip()
    value = parts[1].strip()
    return key, value


def _parse_scalar(text: str) -> Any:
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "null":
        return None
    if text.startswith("[") and text.endswith("]"):
        return _parse_inline_list(text[1:-1].strip())
    if text.startswith("{") and text.endswith("}"):
        return _parse_inline_dict(text[1:-1].strip())
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_inline_list(inner: str) -> list[Any]:
    if inner == "":
        return []
    items = _split_top_level(inner, ",")
    return [_parse_scalar(item.strip()) for item in items]


def _parse_inline_dict(inner: str) -> dict[str, Any]:
    if inner == "":
        return {}
    result: dict[str, Any] = {}
    pairs = _split_top_level(inner, ",")
    for pair in pairs:
        key, value = _split_key_value(pair.strip())
        result[key.strip('"').strip("'")] = _parse_scalar(value.strip())
    return result


def _split_top_level(text: str, delimiter: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth_brace = 0
    depth_bracket = 0
    in_quote = False
    quote_char = ""
    for idx, ch in enumerate(text):
        if in_quote:
            if ch == quote_char:
                in_quote = False
            continue
        if ch in {"'", '"'}:
            in_quote = True
            quote_char = ch
            continue
        if ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        elif ch == delimiter and depth_brace == 0 and depth_bracket == 0:
            items.append(text[start:idx])
            start = idx + 1
    items.append(text[start:])
    return items


# ─────────────────────────────────────────────────────────────
# Vision input helpers — task_scene_image 보조 입력용 (2A/2B/2C)
# ─────────────────────────────────────────────────────────────

import base64 as _base64


def _encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """Read image file and return (base64_str, mime_type)."""
    suffix = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    b64 = _base64.b64encode(image_path.read_bytes()).decode("ascii")
    return b64, mime


def build_user_content_with_image(
    text: str,
    task_scene_image_path: "Path | None",
    detail: str = "auto",
):
    """Build multimodal user content for chat.completions.

    - task_scene_image_path 가 유효 경로이면 [{type:text}, {type:image_url}] 리스트 반환
    - None 이거나 파일 없으면 text str 그대로 반환 (backwards compat)
    """
    if task_scene_image_path is None:
        return text
    p = Path(task_scene_image_path)
    if not p.exists():
        return text
    b64, mime = _encode_image_to_base64(p)
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
                "detail": detail,
            },
        },
    ]


# 모듈별 system prompt 에 추가할 가드 안내문 (재사용)

GUARD_TEXT_2A = """\

[참고 이미지 — task_scene_image 보조 자료]
첨부된 이미지는 task가 발생하는 현실적 정황을 시각으로 보여주는 보조 자료다.
- 정성적(qualitative) 정황 이해에만 사용. (예: 좁은 공간/깊이/접근성 직관)
- subgoal 의 개수, id, 순서, required_atoms 를 이미지 보고 변경하지 말 것.
- task description 이 source of truth. 이미지는 description 의 이해를 돕는 보조용일 뿐.
"""

GUARD_TEXT_2B = """\

[참고 이미지 — task_scene_image 보조 자료]
첨부된 이미지는 task 환경의 정성적(qualitative) 정황을 보여주는 보조 자료다.
⚠ 매우 중요 — 정량 제약 산출 시 주의:
- 이미지에서 본 수치 (예: \"틈이 2cm 정도 보임\") 를 numeric_estimates 또는
  derived_constraints 의 max/min/lower_bound/upper_bound 에 직접 매핑하지 마라.
- 이미지는 환경의 일반적 정황 이해 (좁은 공간, 깊은 구멍, 매달림 등) 에만 사용.
- 정량 제약은 보수적(느슨하게) 산출하라. 너무 빡빡한 thickness/reach 제약은 후보 전멸을 유발한다.
- task description 의 표현을 우선 신뢰하고 이미지로 구체적 수치를 강화하지 마라.
"""

GUARD_TEXT_2C = """\

[참고 이미지 — task_scene_image 보조 자료]
첨부된 이미지는 task 환경의 정성적(qualitative) 정황을 보여주는 보조 자료다.
- 환경 직관 / 도구 결합 발상 / 창발적 후보 생성에 활용하라.
- derived_constraints 와 subgoal_constraints 는 텍스트 입력이 source of truth.
  이미지를 핑계로 제약을 무시하지 말 것.
- 이미지에 보이는 객체 (소파, 하수구 등) 는 scene_objects 와 별개의 정황 자료다.
  used_objects 는 반드시 scene_objects.name 만 사용하라.
"""

