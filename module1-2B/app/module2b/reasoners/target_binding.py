"""Deterministic target binding engine for Module 2-B env-only baseline."""

from __future__ import annotations

from typing import Any

from app.module2b.models import NormalizedContext
from app.module2b.utils import clamp01, contains_any, dedupe_keep_order, normalize_text, stable_round, tokenize


def run_target_binding(
    context: NormalizedContext,
    rules_cfg: dict[str, Any],
    alias_registry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score inventory candidates and resolve deterministic target binding mode/status."""
    weights = rules_cfg["weights"]
    visibility_map = rules_cfg["visibility_score_map"]
    accessibility_map = rules_cfg["accessibility_score_map"]
    thresholds = rules_cfg["thresholds"]

    task_text = " ".join(context.task_text_corpus)
    text_tokens = set(tokenize(task_text))

    plural_tokens = {
        token.lower() for token in alias_registry.get("plural_tokens", [])
    }
    plural_hint = any(token in text_tokens for token in plural_tokens)

    tool_like_terms = [term.lower() for term in alias_registry.get("tool_like_terms", [])]

    candidates: list[dict[str, Any]] = []
    for obj in context.inventory:
        semantic_details = _semantic_match_components(
            obj=obj,
            task_text=task_text,
            text_tokens=text_tokens,
            alias_registry=alias_registry,
            semantic_cfg=rules_cfg["semantic"],
        )
        semantic_score = semantic_details["score"]

        target_state_change_alignment = _target_state_alignment(
            object_name=obj.object_name,
            object_type=obj.object_type_canonical,
            tool_like_terms=tool_like_terms,
        )

        visibility_score = float(visibility_map.get(obj.visibility, 0.6))
        accessibility_score = float(accessibility_map.get(obj.accessibility, 0.6))
        relation_evidence = _relation_evidence_score(
            obj=obj,
            task_text=task_text,
            relation_cfg=rules_cfg["relation_evidence"],
        )

        weighted = (
            (weights["semantic_match_score"] * semantic_score)
            + (weights["target_state_change_alignment"] * target_state_change_alignment)
            + (weights["visibility_score"] * visibility_score)
            + (weights["accessibility_score"] * accessibility_score)
            + (weights["relation_evidence_score"] * relation_evidence["score"])
        )

        uncertainty_penalty = 0.25 * obj.uncertainty_overall
        final_score = clamp01(weighted * (1.0 - uncertainty_penalty))

        candidates.append(
            {
                "object_id": obj.object_id,
                "object_name": obj.object_name,
                "inventory_index": obj.index,
                "semantic_match_score": stable_round(semantic_score),
                "target_state_change_alignment": stable_round(target_state_change_alignment),
                "visibility_score": stable_round(visibility_score),
                "accessibility_score": stable_round(accessibility_score),
                "relation_evidence_score": stable_round(relation_evidence["score"]),
                "relation_evidence_hits": relation_evidence["hits"],
                "semantic_details": semantic_details,
                "uncertainty_overall": stable_round(obj.uncertainty_overall),
                "weighted_score_pre_uncertainty": stable_round(weighted),
                "uncertainty_penalty": stable_round(uncertainty_penalty),
                "final_score": stable_round(final_score),
            }
        )

    ranked = sorted(candidates, key=lambda item: (-item["final_score"], item["inventory_index"]))

    decision = _resolve_mode_and_status(
        ranked=ranked,
        thresholds=thresholds,
        plural_hint=plural_hint,
        task_text=task_text,
        alias_registry=alias_registry,
        rules_cfg=rules_cfg,
    )

    selected_targets = _select_primary_targets(
        ranked=ranked,
        target_mode=decision["target_mode"],
        thresholds=thresholds,
    )

    primary_targets = [
        {
            "object_id": candidate["object_id"],
            "object_name": candidate["object_name"],
            "confidence": stable_round(_candidate_confidence(candidate, decision["binding_status"]), 4),
            "evidence_keys": dedupe_keep_order(
                [
                    "semantic_match",
                    "target_state_change_alignment",
                    "visibility",
                    "accessibility",
                ]
                + (["relation_evidence"] if candidate["relation_evidence_hits"] else [])
            ),
            "context_refs": [],
        }
        for candidate in selected_targets
    ]

    target_confidence = _target_binding_confidence(
        selected_targets=selected_targets,
        binding_status=decision["binding_status"],
        target_mode=decision["target_mode"],
        top_candidate=ranked[0] if ranked else None,
    )

    target_binding = {
        "binding_id": "tb_01",
        "target_mode": decision["target_mode"],
        "binding_status": decision["binding_status"],
        "primary_targets": primary_targets,
        "candidate_ids_ranked": [item["object_id"] for item in ranked],
        "context_refs": [],
        "deferred_reasons": dedupe_keep_order(decision["deferred_reasons"]),
        "confidence": stable_round(target_confidence, 4),
    }

    trace = {
        "text_corpus": context.task_text_corpus,
        "plural_hint": plural_hint,
        "candidate_scoring": ranked,
        "resolution": {
            "target_mode": decision["target_mode"],
            "binding_status": decision["binding_status"],
            "top_score": ranked[0]["final_score"] if ranked else 0.0,
            "second_score": ranked[1]["final_score"] if len(ranked) > 1 else 0.0,
            "deferred_reasons": decision["deferred_reasons"],
            "thresholds": thresholds,
        },
        "confidence_components": {
            "selected_target_scores": [item["final_score"] for item in selected_targets],
            "mode_penalty_applied": decision.get("mode_penalty", 0.0),
            "status_penalty_applied": decision.get("status_penalty", 0.0),
            "target_binding_confidence": stable_round(target_confidence, 4),
        },
        "formula": {
            "weighted_sum": weights,
            "confidence": "mean(selected_final_scores) - mode_penalty - status_penalty (clamped 0..1)",
        },
    }

    return target_binding, trace


def _semantic_match_components(
    obj: Any,
    task_text: str,
    text_tokens: set[str],
    alias_registry: dict[str, Any],
    semantic_cfg: dict[str, Any],
) -> dict[str, Any]:
    object_name = normalize_text(obj.object_name)
    object_type = normalize_text(obj.object_type_canonical)
    location_hint = normalize_text(obj.coarse_location_hint)

    score = 0.0
    hits: list[str] = []

    if object_name and object_name in task_text:
        score += float(semantic_cfg["exact_match_bonus"])
        hits.append("object_name_exact")

    if object_type and object_type in task_text:
        score += float(semantic_cfg["canonical_type_bonus"])
        hits.append("object_type_exact")

    for canonical, aliases in alias_registry.get("aliases", {}).items():
        tokens = [canonical] + [str(item) for item in aliases]
        object_mentions_alias = any(token in object_name or token in object_type for token in tokens)
        text_mentions_alias = any(token in task_text for token in tokens)
        if object_mentions_alias and text_mentions_alias:
            score += float(semantic_cfg["alias_match_bonus"])
            hits.append(f"alias:{canonical}")

    obj_tokens = set(tokenize(object_name) + tokenize(object_type))
    token_overlap = len(obj_tokens & text_tokens)
    if token_overlap > 0:
        score += min(float(semantic_cfg["substring_bonus"]) * token_overlap, 0.35)
        hits.append("token_overlap")

    loc_tokens = tokenize(location_hint)
    if loc_tokens and any(token in text_tokens for token in loc_tokens):
        score += float(semantic_cfg["note_keyword_bonus"])
        hits.append("location_hint_keyword")

    score = min(score, float(semantic_cfg["max_semantic_raw"]))
    return {
        "score": clamp01(score),
        "hits": dedupe_keep_order(hits),
        "object_name": object_name,
        "object_type": object_type,
    }


def _target_state_alignment(object_name: str, object_type: str, tool_like_terms: list[str]) -> float:
    object_text = f"{normalize_text(object_name)} {normalize_text(object_type)}"
    if any(term in object_text for term in tool_like_terms):
        return 0.20
    return 1.0


def _relation_evidence_score(obj: Any, task_text: str, relation_cfg: dict[str, Any]) -> dict[str, Any]:
    supported_relations = set(str(item) for item in relation_cfg.get("supported_relations", []))
    relation_task_tokens = relation_cfg.get("relation_task_tokens", {})

    hits: list[str] = []
    score = 0.0
    for relation in obj.scene_relations:
        rel_name = relation.relation
        if rel_name not in supported_relations:
            continue

        score += 0.25
        tokens = relation_task_tokens.get(rel_name, [])
        if contains_any(task_text, tokens):
            score += 0.25
            hits.append(f"relation_task_match:{rel_name}")
        else:
            hits.append(f"relation_present:{rel_name}")

    return {
        "score": clamp01(score),
        "hits": dedupe_keep_order(hits),
    }


def _resolve_mode_and_status(
    ranked: list[dict[str, Any]],
    thresholds: dict[str, Any],
    plural_hint: bool,
    task_text: str,
    alias_registry: dict[str, Any],
    rules_cfg: dict[str, Any],
) -> dict[str, Any]:
    if not ranked:
        return {
            "target_mode": "none",
            "binding_status": "deferred",
            "deferred_reasons": ["empty_inventory", "missing_upstream_target_carrier"],
            "mode_penalty": 0.25,
            "status_penalty": 0.20,
        }

    top1 = ranked[0]["final_score"]
    top2 = ranked[1]["final_score"] if len(ranked) > 1 else 0.0
    margin = top1 - top2

    resolved_min = float(thresholds["resolved_score_min"])
    partial_min = float(thresholds["partial_score_min"])
    ambiguous_margin = float(thresholds["ambiguous_margin_max"])
    strong_margin = float(thresholds["strong_margin_min"])
    multiple_min = float(thresholds["multiple_score_min"])

    target_like_terms = [term.lower() for term in alias_registry.get("target_like_terms", [])]
    target_text_exists = contains_any(task_text, target_like_terms)

    deferred_reasons: list[str] = []
    mode_penalty = 0.0
    status_penalty = 0.0

    if top1 < partial_min:
        if target_text_exists or rules_cfg["status_policy"].get("implicit_on_missing_target_carrier", True):
            deferred_reasons.append(rules_cfg["status_policy"]["deferred_reason_missing_target_carrier"])
            mode = "implicit"
        else:
            mode = "none"
            deferred_reasons.append("no_target_signal_in_task_text")
        status = "deferred"
        mode_penalty = 0.15
        status_penalty = 0.20
        return {
            "target_mode": mode,
            "binding_status": status,
            "deferred_reasons": deferred_reasons,
            "mode_penalty": mode_penalty,
            "status_penalty": status_penalty,
        }

    if plural_hint:
        multiple_count = sum(1 for item in ranked if item["final_score"] >= multiple_min)
        if multiple_count >= 2:
            mode = "multiple"
            status = "resolved" if top1 >= resolved_min else "partially_resolved"
            mode_penalty = 0.05
            if status != "resolved":
                status_penalty = 0.10
            return {
                "target_mode": mode,
                "binding_status": status,
                "deferred_reasons": deferred_reasons,
                "mode_penalty": mode_penalty,
                "status_penalty": status_penalty,
            }

    if len(ranked) > 1 and top2 >= partial_min and margin <= ambiguous_margin:
        mode = "ambiguous"
        status = "ambiguous"
        deferred_reasons.append("top_candidates_within_margin")
        mode_penalty = 0.15
        status_penalty = 0.15
        return {
            "target_mode": mode,
            "binding_status": status,
            "deferred_reasons": deferred_reasons,
            "mode_penalty": mode_penalty,
            "status_penalty": status_penalty,
        }

    if top1 >= resolved_min and margin >= strong_margin:
        return {
            "target_mode": "single",
            "binding_status": "resolved",
            "deferred_reasons": deferred_reasons,
            "mode_penalty": 0.0,
            "status_penalty": 0.0,
        }

    deferred_reasons.append(rules_cfg["status_policy"]["deferred_reason_low_evidence"])
    return {
        "target_mode": "single",
        "binding_status": "partially_resolved",
        "deferred_reasons": deferred_reasons,
        "mode_penalty": 0.0,
        "status_penalty": 0.10,
    }


def _select_primary_targets(
    ranked: list[dict[str, Any]],
    target_mode: str,
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    if target_mode in {"implicit", "none"}:
        return []
    if target_mode == "multiple":
        floor = float(thresholds["multiple_score_min"])
        selected = [item for item in ranked if item["final_score"] >= floor]
        return selected if selected else ranked[:1]
    if target_mode == "ambiguous":
        return ranked[:2]
    return ranked[:1]


def _candidate_confidence(candidate: dict[str, Any], binding_status: str) -> float:
    score = float(candidate["final_score"])
    if binding_status == "partially_resolved":
        score -= 0.08
    if binding_status == "ambiguous":
        score -= 0.12
    if binding_status == "deferred":
        score -= 0.20
    return clamp01(score)


def _target_binding_confidence(
    selected_targets: list[dict[str, Any]],
    binding_status: str,
    target_mode: str,
    top_candidate: dict[str, Any] | None,
) -> float:
    if selected_targets:
        base = sum(float(item["final_score"]) for item in selected_targets) / float(len(selected_targets))
    elif top_candidate is not None:
        base = float(top_candidate["final_score"]) * 0.55
    else:
        base = 0.10

    mode_penalty = 0.0
    if target_mode == "multiple":
        mode_penalty = 0.05
    elif target_mode == "ambiguous":
        mode_penalty = 0.12
    elif target_mode in {"implicit", "none"}:
        mode_penalty = 0.18

    status_penalty = 0.0
    if binding_status == "partially_resolved":
        status_penalty = 0.10
    elif binding_status == "ambiguous":
        status_penalty = 0.15
    elif binding_status == "deferred":
        status_penalty = 0.20

    return clamp01(base - mode_penalty - status_penalty)
