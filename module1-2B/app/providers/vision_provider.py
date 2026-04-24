"""Image-driven Module 1 provider using OpenAI vision inference."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

from app.providers.base import ProviderResult
from app.utils import load_json, project_root

_CONTACT_PROFILES = ("cavity_rim", "tip", "broad_flat_face", "curved_side", "edge")
_CONTACT_PROFILE_TIE_PRIORITY = ("tip", "edge", "cavity_rim", "broad_flat_face", "curved_side")
_CONTACT_PROFILE_WITH_UNKNOWN = _CONTACT_PROFILES + ("unknown",)
_CONTACT_PROFILE_LLM_CACHE: dict[str, str] = {}
_API_USAGE_TRACKER: dict[str, int] = {
    "api_call_count": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}

# Stage-1 deterministic lexical matcher (expanded dictionary).
_CONTACT_PROFILE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cavity_rim": (
        "ring",
        "loop",
        "bowl",
        "bucket",
        "tube",
        "container",
        "recess",
        "cup",
        "mug",
        "jar",
        "bottle",
        "can",
        "socket",
        "holder",
        "sleeve",
        "cavity",
        "opening",
        "mouth",
        "nozzle",
        "funnel",
        "hole",
        "slot",
        "groove",
    ),
    "tip": (
        "tip",
        "needle",
        "pin",
        "hook",
        "clip",
        "paperclip",
        "tweezer",
        "tweezers",
        "forceps",
        "pick",
        "awl",
        "point",
        "pointed",
        "screwdriver",
        "driver",
        "blade",
        "knife",
        "scalpel",
        "chisel",
        "dart",
        "skewer",
        "toothpick",
        "probe",
    ),
    "broad_flat_face": (
        "sheet",
        "mesh",
        "paper",
        "cloth",
        "tape",
        "block",
        "cube",
        "magnet",
        "card",
        "business card",
        "sticky note",
        "sticky notes",
        "post it",
        "post-it",
        "note",
        "notebook",
        "label",
        "envelope",
        "board",
        "panel",
        "plate",
        "tile",
        "pad",
        "folder",
    ),
    "curved_side": (
        "motor",
        "cylinder",
        "cylindrical",
        "roll",
        "roller",
        "wheel",
        "sphere",
        "ball",
        "pipe",
        "hose",
        "spool",
        "reel",
    ),
    "edge": (
        "stick",
        "rod",
        "wire",
        "string",
        "rope",
        "spring",
        "shaft",
        "pen",
        "pencil",
        "ruler",
        "spatula",
        "scraper",
        "strip",
        "bar",
        "rail",
        "chopstick",
        "edge",
        "swab",
        "cotton swab",
        "qtip",
        "q-tip",
        "applicator",
        "stem",
    ),
}

_ROLE_TO_PROFILE = {
    "container_cavity": "cavity_rim",
    "rigid_tip": "tip",
    "thin_edge": "edge",
    "flat_face": "broad_flat_face",
    "grip_body": "curved_side",
}


class VisionProvider:
    """Generate Module 1 raw output directly from an input image."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
        api_base: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = (
            model
            or os.getenv("MODULE1_VISION_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4.1-mini"
        )
        base = api_base or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        self.api_url = base.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds

    def get_module1_output(
        self,
        image_path: Path | None = None,
        case_id: str | None = None,
        module1_output_path: Path | None = None,
    ) -> ProviderResult:
        """Infer Module 1 raw JSON from image content."""
        if image_path is None:
            raise ValueError("VisionProvider requires --image path.")
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for provider=vision. "
                "Set the environment variable and retry."
            )
        if module1_output_path is not None:
            raise ValueError(
                "provider=vision does not use --module1-output. "
                "Remove --module1-output to run direct image recognition."
            )

        _reset_api_usage_tracker()
        raw_output = self._infer_from_image(image_path=image_path)
        return ProviderResult(
            raw_output=raw_output,
            metadata={
                "provider": "vision",
                "model": self.model,
                "image_path": str(image_path),
                "case_id": case_id,
                "api_url": self.api_url,
                "api_usage": _snapshot_api_usage_tracker(),
            },
        )

    def _infer_from_image(self, image_path: Path) -> dict[str, Any]:
        schema = _sanitize_response_schema(
            load_json(project_root() / "schemas" / "module1_raw_output_lite.schema.json")
        )
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        mime_type = _mime_type_for_image(path=image_path)

        system_prompt = (
            "You are a vision parser for robotics planning. "
            "Return only strict JSON that follows the provided JSON schema."
        )
        user_prompt = (
            "Analyze the image and produce module1_raw_output_lite.\n"
            "- Include all visible object instances.\n"
            "- Keep ordering left_to_right_then_front_to_back.\n"
            "- Use conservative physical inferences when uncertain.\n"
            "- Return JSON only."
        )

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "module1_raw_output_lite",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0,
        }

        try:
            payload = _post_json(
                url=self.api_url,
                api_key=self.api_key,
                body=body,
                timeout_seconds=self.timeout_seconds,
            )
        except ValueError:
            fallback_body = {
                "model": self.model,
                "messages": body["messages"],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            }
            payload = _post_json(
                url=self.api_url,
                api_key=self.api_key,
                body=fallback_body,
                timeout_seconds=self.timeout_seconds,
            )

        parsed = _extract_json_content(response_payload=payload)
        return _normalize_module1_payload(parsed=parsed)


def _post_json(
    url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    _increment_api_call_count()
    req = url_request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with url_request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except url_error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"OpenAI API error ({exc.code}): {error_text[:1000]}"
        ) from exc
    except url_error.URLError as exc:
        raise ValueError(f"OpenAI API network error: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise ValueError("OpenAI API returned non-object JSON payload.")
    _record_api_usage(payload)
    return payload


def _reset_api_usage_tracker() -> None:
    _API_USAGE_TRACKER["api_call_count"] = 0
    _API_USAGE_TRACKER["prompt_tokens"] = 0
    _API_USAGE_TRACKER["completion_tokens"] = 0
    _API_USAGE_TRACKER["total_tokens"] = 0


def _snapshot_api_usage_tracker() -> dict[str, int]:
    return {
        "api_call_count": int(_API_USAGE_TRACKER.get("api_call_count", 0)),
        "prompt_tokens": int(_API_USAGE_TRACKER.get("prompt_tokens", 0)),
        "completion_tokens": int(_API_USAGE_TRACKER.get("completion_tokens", 0)),
        "total_tokens": int(_API_USAGE_TRACKER.get("total_tokens", 0)),
    }


def _increment_api_call_count() -> None:
    _API_USAGE_TRACKER["api_call_count"] = int(_API_USAGE_TRACKER.get("api_call_count", 0)) + 1


def _record_api_usage(payload: dict[str, Any]) -> None:
    usage_raw = payload.get("usage")
    if not isinstance(usage_raw, dict):
        return
    prompt_tokens = _safe_int(usage_raw.get("prompt_tokens"))
    completion_tokens = _safe_int(usage_raw.get("completion_tokens"))
    total_tokens = _safe_int(usage_raw.get("total_tokens"))
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    _API_USAGE_TRACKER["prompt_tokens"] = int(_API_USAGE_TRACKER.get("prompt_tokens", 0)) + prompt_tokens
    _API_USAGE_TRACKER["completion_tokens"] = int(_API_USAGE_TRACKER.get("completion_tokens", 0)) + completion_tokens
    _API_USAGE_TRACKER["total_tokens"] = int(_API_USAGE_TRACKER.get("total_tokens", 0)) + total_tokens


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_json_content(response_payload: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "OpenAI response did not include a valid message content payload."
        ) from exc
    if not isinstance(content, str):
        raise ValueError("OpenAI response content is not a JSON string.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenAI response content is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI response JSON root must be object.")
    return parsed


def _normalize_module1_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    source = parsed.get("module1_raw_output_lite", parsed)
    if not isinstance(source, dict):
        source = parsed
    entries = _extract_object_entries(parsed=parsed, source=source)

    notes = "vision-derived all-visible object inventory"
    caveats: list[str] = [
        "single-image inference only",
        "hidden-side geometry and mass are uncertain",
    ]
    scene_summary = source.get("scene_summary")
    if isinstance(scene_summary, dict):
        note_value = scene_summary.get("notes")
        if isinstance(note_value, str) and note_value.strip():
            notes = note_value.strip()
        caveat_value = scene_summary.get("coverage_caveats")
        if isinstance(caveat_value, list):
            caveats = [str(item) for item in caveat_value]

    objects = [_build_object_entry(raw_entry=item, index=i) for i, item in enumerate(entries)]

    return {
        "schema_name": "module1_raw_output_lite",
        "schema_version": "0.4",
        "scene_summary": {
            "selection_policy": "all visible object instances",
            "ordering_rule": "left_to_right_then_front_to_back",
            "notes": notes,
            "coverage_caveats": caveats,
        },
        "objects": objects,
    }


def _extract_object_entries(parsed: dict[str, Any], source: dict[str, Any]) -> list[Any]:
    if isinstance(source.get("objects"), list):
        return source["objects"]
    lite = parsed.get("module1_raw_output_lite")
    if isinstance(lite, list):
        return lite
    for key in ("visible_objects", "object_instances", "items", "detections", "instances"):
        if isinstance(source.get(key), list):
            return source[key]
        if isinstance(parsed.get(key), list):
            return parsed[key]
    return []


def _build_object_entry(raw_entry: Any, index: int) -> dict[str, Any]:
    item = raw_entry if isinstance(raw_entry, dict) else {}
    object_name = _pick_text(
        [item.get("object_name"), item.get("object"), item.get("name"), raw_entry if isinstance(raw_entry, str) else None],
        f"unknown_object_{index + 1}",
    )
    coarse_location = _pick_text(
        [item.get("coarse_location_hint"), item.get("position"), item.get("location"), item.get("place")],
        "unknown_scene_location",
    )
    quantity = _to_positive_int(item.get("quantity_estimate") or item.get("count"), 1)
    visibility = _normalize_visibility(item.get("visibility"))
    accessibility = _infer_accessibility(item.get("accessibility"), coarse_location)
    support_context = _infer_support_context(coarse_location)
    pose_class = _infer_pose_class(object_name, coarse_location)

    profile = _infer_contact_profile(
        name=object_name,
        raw_entry=item,
        coarse_location=coarse_location,
    )
    role_canonical, part_name = _infer_role_and_part(profile)
    material = _infer_material(object_name)
    uncertainty = _build_uncertainty(visibility)

    relation = "inside" if support_context == "in_container" else "on_surface"
    relation_note = "inside recess/container context" if relation == "inside" else "on support surface"

    return {
        "object_id": f"obj_{index + 1:02d}",
        "object_name": object_name,
        "object_type_canonical": _to_snake_case(object_name),
        "grouped": quantity > 1,
        "quantity_estimate": quantity,
        "coarse_location_hint": coarse_location,
        "visibility": visibility,
        "accessibility": accessibility,
        "state": {
            "pose_class": pose_class,
            "orientation_note": _pick_text([item.get("orientation_note"), item.get("orientation")], "single-view orientation estimate"),
            "support_context": support_context,
        },
        "scene_relations": [
            {
                "relation": relation,
                "object_ref": None,
                "relation_note": relation_note,
            }
        ],
        "observed_vs_inferred": {
            "observed_cues": [f"visible instance: {object_name}", f"coarse location: {coarse_location}"],
            "inferred_aspects": [f"primary_contact_profile: {profile}", f"material_prior: {material}"],
            "assumed_aspects": ["single-view image; depth and mass are approximated"],
        },
        "geometry_cues": _infer_geometry(object_name, profile),
        "scale_anchor_status": "weak_prior",
        "material_hypotheses": _material_hypotheses(material),
        "functional_parts": [
            {
                "part_name": part_name,
                "role": f"primary {profile} region",
                "role_canonical": role_canonical,
                "contact_profile": profile,
                "local_material": material,
                "local_property_note": "inferred from appearance",
                "local_property_tags": ["inferred", "single_view"],
            }
        ],
        "affordance_card": {
            "object_name": object_name,
            "observed_visual_features": [_to_snake_case(object_name), "single_view_rgb_image"],
            "inferred_physical_properties": [f"material_prior:{material}", f"contact_profile:{profile}"],
            "usable_parts": [
                {
                    "part_name": part_name,
                    "affordance_scores": _scores(profile),
                    "interaction_primitives": _primitives(profile),
                    "target_mode_numeric": _target_numeric(profile),
                }
            ],
            "connection_modes": _connection_modes(profile),
            "weaknesses_or_risks": _weaknesses(profile),
            "uncertain_points": [
                "exact metric dimensions are uncalibrated",
                "material composition is estimated",
            ],
            "confidence": 0.71,
        },
        "physical_properties": _physical_properties(material),
        "uncertainty": uncertainty,
    }


def _normalize_visibility(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"full", "partial", "heavily_occluded"}:
        return text
    if text in {"occluded", "hidden"}:
        return "heavily_occluded"
    return "full"


def _infer_accessibility(explicit: Any, location: str) -> str:
    text = str(explicit or "").strip().lower()
    if text in {"clear", "partial", "occluded", "entangled", "nested"}:
        return text
    loc = location.lower()
    if any(token in loc for token in ("inside", "hole", "well", "bottom")):
        return "nested"
    if any(token in loc for token in ("occluded", "blocked")):
        return "occluded"
    return "clear"


def _infer_support_context(location: str) -> str:
    loc = location.lower()
    if any(token in loc for token in ("inside", "hole", "well", "bottom")):
        return "in_container"
    return "on_surface"


def _infer_pose_class(name: str, location: str) -> str:
    text = f"{name} {location}".lower()
    if any(token in text for token in ("upright", "standing", "vertical")):
        return "upright"
    if "lean" in text:
        return "leaning"
    if any(token in text for token in ("inside", "hole", "well", "bottom")):
        return "inside_container"
    return "lying"


def _infer_contact_profile(
    name: str,
    raw_entry: Any | None = None,
    coarse_location: str | None = None,
) -> str:
    """Infer contact profile via multi-signal deterministic scoring.

    Stage-1:
      - expanded lexical matching
      - geometry / part-role / target-numeric hints when available
    Stage-2:
      - fallback classifier for long-tail object names
    """
    text = _build_profile_text(name=name, raw_entry=raw_entry, coarse_location=coarse_location)
    scores = {profile: 0.0 for profile in _CONTACT_PROFILES}

    _accumulate_keyword_profile_scores(scores=scores, normalized_text=text)
    _accumulate_structural_profile_scores(
        scores=scores,
        raw_entry=raw_entry,
        coarse_location=coarse_location or "",
    )

    profile, score = _pick_best_profile(scores=scores)
    if score >= 0.6:
        return profile

    fallback_profile = _infer_contact_profile_fallback(normalized_text=text)
    if fallback_profile != "unknown":
        return fallback_profile

    llm_profile = _infer_contact_profile_via_llm(
        name=name,
        normalized_text=text,
        raw_entry=raw_entry,
        coarse_location=coarse_location or "",
    )
    if llm_profile in _CONTACT_PROFILES:
        return llm_profile

    # Keep weak signal if available; otherwise mark unknown.
    return profile if score >= 0.45 else "unknown"


def _build_profile_text(name: str, raw_entry: Any | None, coarse_location: str | None) -> str:
    values: list[str] = [name]
    if coarse_location:
        values.append(coarse_location)
    if isinstance(raw_entry, dict):
        for key in (
            "object_type_canonical",
            "object_type",
            "type",
            "category",
            "label",
            "description",
        ):
            value = raw_entry.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value)
    return _normalize_profile_text(" ".join(values))


def _normalize_profile_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _text_has_token(normalized_text: str, token: str) -> bool:
    normalized_token = _normalize_profile_text(token)
    if not normalized_token:
        return False
    haystack = f" {normalized_text} "
    needle = f" {normalized_token} "
    return needle in haystack


def _accumulate_keyword_profile_scores(
    scores: dict[str, float],
    normalized_text: str,
) -> None:
    for profile, tokens in _CONTACT_PROFILE_KEYWORDS.items():
        for token in tokens:
            if _text_has_token(normalized_text=normalized_text, token=token):
                # Multi-word tokens carry slightly stronger evidence.
                scores[profile] += 0.8 if " " not in token else 0.95


def _accumulate_structural_profile_scores(
    scores: dict[str, float],
    raw_entry: Any | None,
    coarse_location: str,
) -> None:
    if any(token in coarse_location.lower() for token in ("inside", "hole", "slot", "recess", "drawer")):
        scores["cavity_rim"] += 0.15

    if not isinstance(raw_entry, dict):
        return

    geometry = raw_entry.get("geometry_cues", {})
    if isinstance(geometry, dict):
        primary = str(geometry.get("primary_contact_profile", "")).lower()
        if "tip" in primary or "point" in primary:
            scores["tip"] += 0.8
        if "edge" in primary:
            scores["edge"] += 0.8
        if "rim" in primary or "cavity" in primary:
            scores["cavity_rim"] += 0.8
        if "flat" in primary or "face" in primary:
            scores["broad_flat_face"] += 0.8
        if "curved" in primary or "cylinder" in primary or "round" in primary:
            scores["curved_side"] += 0.8

        if bool(geometry.get("has_open_cavity", False)):
            scores["cavity_rim"] += 0.6
        if bool(geometry.get("has_flat_contact_face", False)):
            scores["broad_flat_face"] += 0.45
        if bool(geometry.get("has_pointed_or_thin_end", False)):
            scores["tip"] += 0.55
            scores["edge"] += 0.25

        aspect_ratio = str(geometry.get("aspect_ratio_hint", "")).lower()
        if aspect_ratio == "elongated":
            scores["edge"] += 0.35
        if aspect_ratio == "sheet_like":
            scores["broad_flat_face"] += 0.5
        if aspect_ratio in {"compact", "round"}:
            scores["curved_side"] += 0.2

    for part in raw_entry.get("functional_parts", []) or []:
        if not isinstance(part, dict):
            continue
        role = str(part.get("role_canonical", "")).lower()
        mapped = _ROLE_TO_PROFILE.get(role)
        if mapped:
            scores[mapped] += 0.95

        part_profile = str(part.get("contact_profile", "")).lower()
        if part_profile in _CONTACT_PROFILES:
            scores[part_profile] += 1.2

    usable_parts = ((raw_entry.get("affordance_card") or {}).get("usable_parts") or [])
    if usable_parts and isinstance(usable_parts[0], dict):
        target_mode = usable_parts[0].get("target_mode_numeric", {}) or {}
        point_score = _to_non_negative_float(target_mode.get("point_score"))
        edge_score = _to_non_negative_float(target_mode.get("edge_score"))
        face_score = _to_non_negative_float(target_mode.get("face_score"))
        rim_score = _to_non_negative_float(target_mode.get("rim_score"))
        cavity_score = _to_non_negative_float(target_mode.get("cavity_score"))
        axis_score = _to_non_negative_float(target_mode.get("axis_score"))

        if point_score >= 0.6:
            scores["tip"] += 0.5
        if edge_score >= 0.55:
            scores["edge"] += 0.5
        if face_score >= 0.55:
            scores["broad_flat_face"] += 0.5
        if max(rim_score, cavity_score) >= 0.5:
            scores["cavity_rim"] += 0.5
        if axis_score >= 0.55:
            scores["edge"] += 0.25
            scores["curved_side"] += 0.2


def _pick_best_profile(scores: dict[str, float]) -> tuple[str, float]:
    priority_index = {profile: idx for idx, profile in enumerate(_CONTACT_PROFILE_TIE_PRIORITY)}
    best_profile = "unknown"
    best_score = -1.0
    best_priority = 999
    for profile in _CONTACT_PROFILES:
        score = float(scores.get(profile, 0.0))
        idx = priority_index.get(profile, 999)
        if score > best_score or (score == best_score and idx < best_priority):
            best_profile = profile
            best_score = score
            best_priority = idx
    return best_profile, best_score


def _infer_contact_profile_fallback(normalized_text: str) -> str:
    """Stage-2 fallback classifier for long-tail or unseen object names."""
    if _text_has_any(normalized_text, ("tweezers", "tweezer", "forceps", "screwdriver", "needle", "pin", "awl", "pick", "probe", "toothpick", "blade", "knife", "scalpel")):
        return "tip"
    if _text_has_any(normalized_text, ("ruler", "rod", "stick", "shaft", "strip", "bar", "rail", "chopstick", "pen", "pencil", "wire", "string", "rope", "swab", "cotton swab", "qtip", "q tip")):
        return "edge"
    if _text_has_any(normalized_text, ("business card", "card", "paper", "note", "sticky note", "post it", "sheet", "panel", "board", "label", "envelope", "plate", "pad")):
        return "broad_flat_face"
    if _text_has_any(normalized_text, ("cup", "mug", "bottle", "jar", "can", "container", "bowl", "bucket", "hole", "slot", "recess", "socket", "opening")):
        return "cavity_rim"
    if _text_has_any(normalized_text, ("cylinder", "cylindrical", "roller", "wheel", "ball", "sphere", "pipe", "hose", "spool")):
        return "curved_side"
    return "unknown"


def _text_has_any(normalized_text: str, tokens: tuple[str, ...]) -> bool:
    return any(_text_has_token(normalized_text=normalized_text, token=token) for token in tokens)


def _infer_contact_profile_via_llm(
    name: str,
    normalized_text: str,
    raw_entry: Any | None,
    coarse_location: str,
) -> str:
    if not _is_llm_profile_fallback_enabled():
        return "unknown"

    cache_key = _contact_profile_cache_key(name=name, normalized_text=normalized_text, coarse_location=coarse_location)
    cached = _CONTACT_PROFILE_LLM_CACHE.get(cache_key)
    if cached:
        return cached

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "unknown"

    model = (
        os.getenv("MODULE1_CONTACT_PROFILE_MODEL")
        or os.getenv("MODULE1_VISION_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1-mini"
    )
    base = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    api_url = base.rstrip("/") + "/chat/completions"

    role_hint = ""
    if isinstance(raw_entry, dict):
        role_hint = str(raw_entry.get("object_type") or raw_entry.get("object_type_canonical") or raw_entry.get("category") or "")

    system_prompt = (
        "You classify contact profile for manipulation planning. "
        "Return one label only from the allowed set."
    )
    user_prompt = (
        "Choose the best contact_profile for this object from:\n"
        "- cavity_rim\n- tip\n- broad_flat_face\n- curved_side\n- edge\n- unknown\n\n"
        f"object_name: {name}\n"
        f"normalized_text: {normalized_text}\n"
        f"coarse_location_hint: {coarse_location}\n"
        f"type_hint: {role_hint}\n"
        "Output strict JSON with keys: contact_profile, confidence."
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "contact_profile": {"type": "string", "enum": list(_CONTACT_PROFILE_WITH_UNKNOWN)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["contact_profile", "confidence"],
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "module1_contact_profile",
                "strict": True,
                "schema": schema,
            },
        },
        "temperature": 0,
    }

    try:
        payload = _post_json(
            url=api_url,
            api_key=api_key,
            body=body,
            timeout_seconds=25.0,
        )
    except ValueError:
        fallback_body = {
            "model": model,
            "messages": body["messages"],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        try:
            payload = _post_json(
                url=api_url,
                api_key=api_key,
                body=fallback_body,
                timeout_seconds=25.0,
            )
        except ValueError:
            return "unknown"

    try:
        parsed = _extract_json_content(response_payload=payload)
    except ValueError:
        return "unknown"

    source = parsed.get("module1_contact_profile", parsed)
    if not isinstance(source, dict):
        return "unknown"

    candidate = _normalize_profile_label(str(source.get("contact_profile", "")).strip())
    if candidate in _CONTACT_PROFILES:
        _CONTACT_PROFILE_LLM_CACHE[cache_key] = candidate
        return candidate
    return "unknown"


def _is_llm_profile_fallback_enabled() -> bool:
    raw = str(os.getenv("MODULE1_ENABLE_CONTACT_PROFILE_LLM_FALLBACK", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _contact_profile_cache_key(name: str, normalized_text: str, coarse_location: str) -> str:
    return _normalize_profile_text(f"{name}::{normalized_text}::{coarse_location}")


def _normalize_profile_label(label: str) -> str:
    normalized = _normalize_profile_text(label).replace(" ", "_")
    alias = {
        "broad_flatface": "broad_flat_face",
        "broad_face": "broad_flat_face",
        "flat_face": "broad_flat_face",
        "flat": "broad_flat_face",
        "cavity": "cavity_rim",
        "rim": "cavity_rim",
    }
    normalized = alias.get(normalized, normalized)
    if normalized in _CONTACT_PROFILE_WITH_UNKNOWN:
        return normalized
    return "unknown"


def _infer_role_and_part(profile: str) -> tuple[str, str]:
    if profile == "cavity_rim":
        return ("container_cavity", "rim_or_cavity")
    if profile == "tip":
        return ("rigid_tip", "tip_end")
    if profile == "edge":
        return ("thin_edge", "edge_segment")
    if profile == "broad_flat_face":
        return ("flat_face", "flat_face")
    if profile == "curved_side":
        return ("grip_body", "curved_body")
    return ("unknown", "primary_region")


def _infer_material(name: str) -> str:
    text = name.lower()
    if any(token in text for token in ("wood", "stick", "ruler")):
        return "wood"
    if any(token in text for token in ("string", "rope", "cloth", "mesh", "cotton", "swab", "qtip", "q-tip")):
        return "fiber"
    if any(
        token in text
        for token in (
            "motor",
            "spring",
            "paperclip",
            "wire",
            "magnet",
            "screwdriver",
            "tweezer",
            "tweezers",
            "forceps",
            "blade",
            "knife",
            "clip",
            "pin",
            "needle",
        )
    ):
        return "metal"
    if any(token in text for token in ("tape", "ring", "plastic", "card", "paper", "sticky note", "post-it")):
        return "polymer"
    return "mixed"


def _infer_geometry(name: str, profile: str) -> dict[str, Any]:
    text = name.lower()
    aspect = "unknown"
    thickness = "medium"
    if any(
        token in text
        for token in (
            "stick",
            "rod",
            "wire",
            "string",
            "rope",
            "spring",
            "shaft",
            "pen",
            "pencil",
            "ruler",
            "screwdriver",
            "tweezer",
            "tweezers",
            "forceps",
            "needle",
            "pin",
            "swab",
            "qtip",
            "q-tip",
            "awl",
            "pick",
            "skewer",
            "chopstick",
            "blade",
        )
    ):
        aspect, thickness = "elongated", "thin"
    if any(
        token in text
        for token in (
            "sheet",
            "mesh",
            "paper",
            "cloth",
            "film",
            "card",
            "business card",
            "sticky note",
            "post-it",
            "note",
            "label",
            "envelope",
            "board",
            "panel",
        )
    ):
        aspect, thickness = "sheet_like", "thin"
    if any(token in text for token in ("block", "cube", "motor", "magnet", "battery", "box", "clip", "clamp")):
        aspect, thickness = "blocky", "thick"
    if any(
        token in text
        for token in ("ring", "loop", "bowl", "bucket", "tube", "container", "cup", "mug", "jar", "bottle", "can")
    ) and aspect == "unknown":
        aspect = "compact"

    has_open_cavity = profile == "cavity_rim"
    has_flat = profile == "broad_flat_face"
    has_pointed = profile in {"tip", "edge"}
    roll_risk = "round_cross_section" if profile == "curved_side" else "none"

    return {
        "shape_class": _to_snake_case(name),
        "aspect_ratio_hint": aspect,
        "size_relative": "unknown",
        "thickness_class": thickness,
        "primary_contact_profile": profile,
        "has_pointed_or_thin_end": has_pointed,
        "has_flat_contact_face": has_flat,
        "has_open_cavity": has_open_cavity,
        "roll_risk_source": roll_risk,
    }


def _material_hypotheses(material: str) -> list[dict[str, Any]]:
    secondary = "composite" if material != "mixed" else "unknown"
    return [{"material": material, "probability": 0.7}, {"material": secondary, "probability": 0.3}]


def _scores(profile: str) -> dict[str, float]:
    table = {
        "tip": {"poke": 0.72, "insert": 0.63, "press": 0.34},
        "edge": {"align": 0.62, "scrape": 0.48, "pry": 0.28},
        "broad_flat_face": {"support": 0.74, "press": 0.46, "stabilize": 0.39},
        "cavity_rim": {"hook": 0.58, "capture": 0.52, "support": 0.31},
        "curved_side": {"grasp": 0.55, "drag": 0.43, "push": 0.37},
        "unknown": {"push": 0.35, "support": 0.34, "align": 0.22},
    }
    return table.get(profile, table["unknown"])


def _primitives(profile: str) -> dict[str, float]:
    table = {
        "tip": {"poke": 0.74, "insert": 0.66, "tap": 0.3},
        "edge": {"align": 0.64, "scrape": 0.5, "guide": 0.31},
        "broad_flat_face": {"support": 0.76, "press": 0.48, "brace": 0.42},
        "cavity_rim": {"hook": 0.61, "capture": 0.55, "pull": 0.36},
        "curved_side": {"grasp": 0.57, "drag": 0.45, "roll": 0.25},
        "unknown": {"push": 0.37, "support": 0.35, "align": 0.24},
    }
    return table.get(profile, table["unknown"])


def _target_numeric(profile: str) -> dict[str, Any]:
    base = {
        "point_score": 0.12,
        "edge_score": 0.24,
        "face_score": 0.28,
        "rim_score": 0.0,
        "cavity_score": 0.0,
        "axis_score": 0.21,
        "hook_gap_score": 0.0,
        "exposure_ratio": 0.73,
        "clearance_ratio": 0.58,
        "usable_span_m": 0.09,
        "local_thickness_m": 0.01,
        "tip_radius_m": 0.003,
        "flat_patch_m2": 0.0,
        "approach_directions_count": 2,
    }
    if profile == "tip":
        base.update({"point_score": 0.86, "edge_score": 0.29, "face_score": 0.0, "tip_radius_m": 0.001, "local_thickness_m": 0.002, "usable_span_m": 0.12})
    elif profile == "edge":
        base.update({"point_score": 0.18, "edge_score": 0.81, "face_score": 0.24, "tip_radius_m": 0.0015, "local_thickness_m": 0.003, "usable_span_m": 0.14})
    elif profile == "broad_flat_face":
        base.update({"point_score": 0.0, "edge_score": 0.23, "face_score": 0.85, "flat_patch_m2": 0.002})
    elif profile == "cavity_rim":
        base.update({"point_score": 0.0, "edge_score": 0.26, "face_score": 0.2, "rim_score": 0.79, "cavity_score": 0.67, "hook_gap_score": 0.33})
    elif profile == "curved_side":
        base.update({"point_score": 0.08, "edge_score": 0.32, "face_score": 0.18, "axis_score": 0.51})
    return base


def _connection_modes(profile: str) -> list[dict[str, Any]]:
    if profile == "cavity_rim":
        return [{"mode": "wrap_around", "score": 0.56, "note": "rim/cavity mediated connection"}, {"mode": "rest_on_rim", "score": 0.34, "note": "rim support contact"}]
    if profile in {"tip", "edge"}:
        return [{"mode": "insert_into_gap", "score": 0.51, "note": "slender-contact insertion"}, {"mode": "edge_contact", "score": 0.36, "note": "line or point contact"}]
    if profile == "broad_flat_face":
        return [{"mode": "mate_flat_face", "score": 0.62, "note": "planar face contact"}, {"mode": "brace_between_surfaces", "score": 0.38, "note": "surface bracing"}]
    return [{"mode": "surface_contact", "score": 0.44, "note": "generic contact mode"}, {"mode": "stabilize_contact", "score": 0.31, "note": "stabilization contact"}]


def _weaknesses(profile: str) -> list[str]:
    out: list[str] = []
    if profile in {"tip", "edge"}:
        out.append("thin tip/edge may snag or scratch")
    if profile == "curved_side":
        out.append("roll drift risk on curved contact")
    if not out:
        out.append("exact contact friction uncertain")
    return out


def _physical_properties(material: str) -> dict[str, Any]:
    if material == "metal":
        labels = ("high", "low", "medium", "low", "medium", "high", "low", "low", "negligible_deformation", "high", "high", "bend")
    elif material in {"fiber", "polymer"}:
        labels = ("low", "high", "high", "medium", "very_light", "low", "low", "medium", "compressible", "low", "low", "buckle")
    elif material == "wood":
        labels = ("medium", "low", "medium", "medium", "light", "medium", "low", "medium", "elastic_deformation", "medium", "medium", "crack")
    else:
        labels = ("medium", "medium", "medium", "medium", "light", "medium", "low", "medium", "elastic_deformation", "medium", "medium", "slip")

    names = (
        "stiffness",
        "deformability",
        "surface_friction",
        "slip_tendency",
        "mass_category",
        "density_category",
        "restitution",
        "fragility",
        "press_response_type",
        "tip_force_transmission",
        "load_bearing",
        "failure_mode",
    )
    out: dict[str, Any] = {}
    for name, label in zip(names, labels):
        out[name] = {"label": label, "confidence": 0.64, "evidence": "single-view material/shape prior"}
    return out


def _build_uncertainty(visibility: str) -> dict[str, float]:
    occlusion = 0.08 if visibility == "full" else 0.24 if visibility == "partial" else 0.38
    components = {"occlusion": occlusion, "scale": 0.22, "material": 0.27, "mass": 0.25, "dynamics": 0.33, "part_structure": 0.28}
    overall = round(sum(components.values()) / 6.0, 2)
    return {"overall": overall, **components}


def _pick_text(candidates: list[Any], default: str) -> str:
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, str):
            text = str(value).strip()
            if text:
                return text
    return default


def _to_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _to_non_negative_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed >= 0 else 0.0


def _to_snake_case(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not lowered:
        return "unknown_object"
    if not lowered[0].isalpha():
        lowered = f"object_{lowered}"
    return lowered


def _mime_type_for_image(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _sanitize_response_schema(schema: Any) -> Any:
    unsupported = {
        "minProperties",
        "maxProperties",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "default",
        "examples",
        "$id",
        "$schema",
    }
    if isinstance(schema, dict):
        return {k: _sanitize_response_schema(v) for k, v in schema.items() if k not in unsupported}
    if isinstance(schema, list):
        return [_sanitize_response_schema(item) for item in schema]
    return schema
