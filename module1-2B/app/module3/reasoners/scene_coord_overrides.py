"""팀원 manipulation 시뮬레이터 실측 좌표 override.

PyBullet settle 결과(우리 scene_info)와 팀원 sim 환경의 실제 좌표가 약간 다르다는
보고에 따라, task 별로 특정 물체의 center_world (와 옵션으로 size) 를 override 한다.

값 형식 (list):
  - 3-tuple [x, y, z]               : center 만 override. AABB 는 기존 extents 유지하고 평행이동.
  - 6-tuple [x, y, z, sx, sy, sz]   : center + size 둘 다 override.
                                       aabb_min = center - size/2, aabb_max = center + size/2.

확장 방법:
1) 새 task 가 추가되면 ``TASK_KEYWORD_TO_ID`` 에 description 일부 키워드를 매핑.
2) ``OBJECT_CENTER_OVERRIDES_BY_TASK`` 에 task_id → {object_name: [...]} 추가.
"""

from __future__ import annotations
from typing import Any


# task description 의 일부 텍스트로 task_id 식별. 가장 구체적인 키워드 우선.
TASK_KEYWORD_TO_ID: list[tuple[str, str]] = [
    ("명함",           "card_from_gap"),       # T1
    ("깊은 구멍",       "deep_hole_reach"),     # T2
    ("매달린",          "suspended_target"),    # T3
    ("막힌",            "blocked_door_handle"), # T4
    ("유리",            "glass_shard_extract"), # T5
]

# task_id → object_name → [x, y, z] 또는 [x, y, z, sx, sy, sz]
# 팀원 시뮬 환경에서 실측한 좌표 (PyBullet aabb_center + aabb_size).
OBJECT_CENTER_OVERRIDES_BY_TASK: dict[str, dict[str, list[float]]] = {
    "card_from_gap": {
        # 2026-04-27 팀원 manipulation sim 안정화 후 실측. format: [cx, cy, cz, sx, sy, sz]
        "cracker_box":          [1.2022,  0.3067, 0.7327, 0.0808, 0.1722, 0.2219],
        "sugar_box":            [1.2001,  0.1007, 0.7140, 0.0579, 0.1029, 0.1845],
        "pudding_box":          [1.1982, -0.0986, 0.6454, 0.1462, 0.1376, 0.0489],
        "gelatin_box":          [1.1998, -0.3003, 0.6409, 0.0976, 0.1093, 0.0389],
        "bowl":                 [0.9996,  0.2023, 0.6535, 0.1696, 0.1692, 0.0633],
        "mug":                  [1.0089,  0.0002, 0.6670, 0.1256, 0.1018, 0.0899],
        "plate":                [0.9983, -0.2014, 0.6393, 0.2682, 0.2691, 0.0350],
        "fork":                 [0.6661,  0.2979, 0.6335, 0.2057, 0.0389, 0.0303],
        "spoon":                [0.6908,  0.1050, 0.6362, 0.1942, 0.0913, 0.0301],
        "knife":                [0.6727, -0.0990, 0.6338, 0.2240, 0.0315, 0.0428],
        "spatula":              [0.7201, -0.3149, 0.6433, 0.3157, 0.1400, 0.0501],
        "adjustable_wrench":    [0.4317,  0.2372, 0.6320, 0.1339, 0.1897, 0.0344],
        "large_marker":         [0.4499,  0.0027, 0.6354, 0.0295, 0.1290, 0.0274],
        "phillips_screwdriver": [0.3982, -0.2142, 0.6456, 0.2115, 0.1333, 0.0738],
        "flat_screwdriver":     [0.4206, -0.3360, 0.6434, 0.1916, 0.1875, 0.0521],
    },
}


def identify_task_id(task_description: str) -> str | None:
    """task description 에서 task_id 추출. 매칭 없으면 None."""
    if not task_description:
        return None
    for keyword, tid in TASK_KEYWORD_TO_ID:
        if keyword in task_description:
            return tid
    return None


def apply_scene_overrides(
    scene_objects: list[dict[str, Any]],
    task_description: str,
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    """task 매칭되는 물체에 대해 center_world override 적용.

    AABB 는 extents 유지 + 평행이동. ``_overridden`` 플래그를 dict 에 추가하여
    이후 ``_init_positions`` 의 lift 로직이 이 물체를 건너뛰도록 한다.

    반환:
      - 새 scene_objects (deep copy)
      - override 적용된 물체 이름 리스트
      - 식별된 task_id (없으면 None)
    """
    task_id = identify_task_id(task_description)
    if task_id is None:
        return scene_objects, [], None

    overrides = OBJECT_CENTER_OVERRIDES_BY_TASK.get(task_id, {})
    if not overrides:
        return scene_objects, [], task_id

    new_objs: list[dict[str, Any]] = []
    applied: list[str] = []
    for obj in scene_objects:
        name = obj.get("name", "")
        if name not in overrides:
            new_objs.append(obj)
            continue

        ov = overrides[name]
        new_obj = {**obj}
        new_center = [float(c) for c in ov[:3]]
        new_obj["center_world"] = new_center

        if len(ov) >= 6:
            # center + size override → AABB 새로 계산
            size = [float(s) for s in ov[3:6]]
            new_obj["aabb_min"] = [round(new_center[i] - size[i] / 2.0, 6) for i in range(3)]
            new_obj["aabb_max"] = [round(new_center[i] + size[i] / 2.0, 6) for i in range(3)]
        else:
            # center 만 override → 기존 AABB 평행이동 (extents 유지)
            old_center = list(obj.get("center_world", new_center))
            delta = [new_center[i] - old_center[i] for i in range(3)]
            if "aabb_min" in obj and len(obj["aabb_min"]) >= 3:
                new_obj["aabb_min"] = [round(obj["aabb_min"][i] + delta[i], 6) for i in range(3)]
            if "aabb_max" in obj and len(obj["aabb_max"]) >= 3:
                new_obj["aabb_max"] = [round(obj["aabb_max"][i] + delta[i], 6) for i in range(3)]

        new_obj["_overridden"] = True  # lift 스킵 플래그
        new_objs.append(new_obj)
        applied.append(name)

    return new_objs, applied, task_id
