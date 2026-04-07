"""Shared helper functions for Module 2-B deterministic reasoning."""

from __future__ import annotations

import re
from typing import Iterable


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    """Return unique string items preserving first occurrence order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def normalize_text(text: str | None) -> str:
    """Normalize text for deterministic token matching."""
    if text is None:
        return ""
    return " ".join(str(text).lower().strip().split())


def tokenize(text: str | None) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return re.findall(r"[a-z0-9_]+", normalized)


def clamp01(value: float) -> float:
    """Clamp a float into [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def mean_or_default(values: Iterable[float], default: float) -> float:
    """Return arithmetic mean or default when empty."""
    vals = list(values)
    if not vals:
        return default
    return sum(vals) / float(len(vals))


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    """Return True if text contains any keyword substring."""
    lowered = normalize_text(text)
    return any(keyword.lower() in lowered for keyword in keywords)


def stable_round(value: float, ndigits: int = 4) -> float:
    """Round float in one place for reproducible artifact values."""
    return round(float(value), ndigits)
