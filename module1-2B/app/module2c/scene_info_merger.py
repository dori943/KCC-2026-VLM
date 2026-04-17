"""scene_info.json을 읽어서 scene_objects의 AABB/center_world를 업데이트하는 함수."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from app.utils import load_json


def merge_scene_info(
    scene_objects: list[dict[str, Any]],
    scene_info_path: Path,
) -> list[dict[str, Any]]:
    """
    파이불렛에서 추출한 scene_info.json으로
    scene_objects의 aabb_min/aabb_max/center_world를 업데이트한다.
    """
    if not scene_info_path.exists():
        print(f"[scene_info] {scene_info_path} 없음 → AABB 업데이트 건너뜀")
        return scene_objects

    scene_info = load_json(scene_info_path)
    pybullet_objects: dict[str, dict[str, Any]] = {}

    for obj in scene_info.get("objects", []):
        label = obj.get("label", "")
        if label:
            pybullet_objects[label] = obj

    updated = 0
    for scene_obj in scene_objects:
        name = scene_obj.get("name", "")
        pb_obj = pybullet_objects.get(name)
        if pb_obj:
            if pb_obj.get("aabb_min"):
                scene_obj["aabb_min"] = pb_obj["aabb_min"]
            if pb_obj.get("aabb_max"):
                scene_obj["aabb_max"] = pb_obj["aabb_max"]
            if pb_obj.get("center_world"):
                scene_obj["center_world"] = pb_obj["center_world"]
            updated += 1

    print(f"[scene_info] {updated}/{len(scene_objects)}개 물체 AABB 업데이트 완료")
    return scene_objects
