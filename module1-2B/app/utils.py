"""Utility helpers shared across the experiment harness."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────
# Task-based output routing
# ──────────────────────────────────────────────
# 파이프라인 chain에서 task 식별자는 `module2b_<task>` 형태로 가장 안쪽에 보존됨.
# task name이 underscore 포함 가능하도록 끝(또는 다음 `/`)까지 매칭.
_TASK_FROM_PATH_RE = re.compile(r"module2b_([A-Za-z0-9][A-Za-z0-9\-_]*?)(?:/|$)")


def _sanitize_task_name(raw: str) -> str:
    """파일시스템 안전한 이름."""
    cleaned = re.sub(r"[^A-Za-z0-9\-_]+", "_", raw).strip("_")
    return cleaned or "default"


def derive_task_name(
    task_name: str | None = None,
    bundle_path: Path | str | None = None,
    image_path: Path | str | None = None,
    case_id: str | None = None,
    default: str = "default",
) -> str:
    """Task 폴더 이름 결정.

    우선순위:
    1. 명시된 task_name
    2. bundle_path에서 `module2b_<task>` 자동 추출
    3. image_path.stem (module1 진입점)
    4. case_id
    5. default ('default')
    """
    if task_name:
        return _sanitize_task_name(task_name)
    if bundle_path:
        m = _TASK_FROM_PATH_RE.search(str(bundle_path))
        if m:
            return _sanitize_task_name(m.group(1))
    if image_path:
        stem = Path(str(image_path)).stem
        if stem:
            return _sanitize_task_name(stem)
    if case_id:
        return _sanitize_task_name(case_id)
    return default


def build_task_output_root(output_root: Path, task_name: str) -> Path:
    """outputs/<task_name>/ 경로 생성 및 반환."""
    return ensure_dir(output_root / task_name)


def ensure_unique_run_dir(parent: Path, stem: str) -> Path:
    """같은 이름의 run_dir이 있으면 _01, _02 suffix 붙여 유니크하게 생성."""
    candidate = parent / stem
    if not candidate.exists():
        return ensure_dir(candidate)
    index = 1
    while True:
        fallback = parent / f"{stem}_{index:02d}"
        if not fallback.exists():
            return ensure_dir(fallback)
        index += 1

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


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file as dictionary."""
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    """Write JSON file with stable formatting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
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
