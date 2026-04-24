"""KCC2026 Case 1~5 scene_info 일괄 생성기.

사용법:
    python render_scene_cases.py              # 전체 5개 case 생성
    python render_scene_cases.py --case 3     # 특정 case만

출력:
    scene_info_case{N}.json         # Module 2-C providers.py가 읽을 좌표 파일
    scene_capture_case{N}.png       # PyBullet 렌더링 미리보기

Module 2-C provider 쪽 연결:
    Module2BOutputProvider(..., scene_info_path=Path("scene_info_case3.json"))
    또는 CLI --scene-info scene_info_case3.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
ycb_dir = os.path.join(script_dir, "data/object2urdf/examples/ycb")
sys.path.insert(0, script_dir)

from object_info_collector import collect_scene_info


# ───────────────────────────────────────────────────────────────
# Case 별 물체 스펙:  (label, urdf_filename, [x, y, z])
# label은 provider `_SCENE_NAME_MAP` 매칭 키가 됨 (숫자 프리픽스 X)
# ───────────────────────────────────────────────────────────────
CASES: dict[int, list[tuple[str, str, list[float]]]] = {
    1: [
        ("cracker_box",           "003_cracker_box.urdf",           [1.2,  0.3,  0.82]),
        ("sugar_box",             "004_sugar_box.urdf",             [1.2,  0.10, 0.82]),
        ("pudding_box",           "008_pudding_box.urdf",           [1.2, -0.10, 0.82]),
        ("gelatin_box",           "009_gelatin_box.urdf",           [1.2, -0.3,  0.82]),
        ("bowl",                  "024_bowl.urdf",                  [1.0,  0.2,  0.82]),
        ("mug",                   "025_mug.urdf",                   [1.0,  0.0,  0.82]),
        ("plate",                 "029_plate.urdf",                 [1.0, -0.2,  0.82]),
        ("fork",                  "030_fork.urdf",                  [0.7,  0.3,  0.82]),
        ("spoon",                 "031_spoon.urdf",                 [0.7,  0.10, 0.82]),
        ("knife",                 "032_knife.urdf",                 [0.7, -0.1,  0.82]),
        ("spatula",               "033_spatula.urdf",               [0.7, -0.3,  0.82]),
        ("adjustable_wrench",     "042_adjustable_wrench.urdf",     [0.45, 0.2,  0.82]),
        ("large_marker",          "040_large_marker.urdf",          [0.45, 0.0,  0.82]),
        ("phillips_screwdriver",  "043_phillips_screwdriver.urdf",  [0.45,-0.2,  0.82]),
        ("flat_screwdriver",      "044_flat_screwdriver.urdf",      [0.45,-0.35, 0.82]),
    ],
    2: [
        ("pitcher_base",          "019_pitcher_base.urdf",          [1.2,  0.3,  0.82]),
        ("power_drill",           "035_power_drill.urdf",           [1.2,  0.10, 0.82]),
        ("wood_block",            "036_wood_block.urdf",            [1.2, -0.10, 0.82]),
        ("large_marker",          "040_large_marker.urdf",          [1.2, -0.3,  0.82]),
        ("mug",                   "025_mug.urdf",                   [1.0,  0.2,  0.82]),
        ("adjustable_wrench",     "042_adjustable_wrench.urdf",     [1.0,  0.0,  0.82]),
        ("phillips_screwdriver",  "043_phillips_screwdriver.urdf",  [1.0, -0.2,  0.82]),
        ("hammer",                "048_hammer.urdf",                [0.7,  0.3,  0.82]),
        ("small_clamp",           "049_small_clamp.urdf",           [0.7,  0.10, 0.82]),
        ("medium_clamp",          "050_medium_clamp.urdf",          [0.7, -0.1,  0.82]),
        ("large_clamp",           "051_large_clamp.urdf",           [0.7, -0.3,  0.82]),
        ("extra_large_clamp",     "052_extra_large_clamp.urdf",     [0.45, 0.2,  0.82]),
        ("chain",                 "059_chain.urdf",                 [0.45, 0.0,  0.82]),
        ("foam_brick",            "061_foam_brick.urdf",            [0.45,-0.2,  0.82]),
        ("flat_screwdriver",      "044_flat_screwdriver.urdf",      [0.45,-0.35, 0.82]),
    ],
    3: [
        ("large_marker",          "040_large_marker.urdf",          [1.2,  0.3,  0.82]),
        ("cracker_box",           "003_cracker_box.urdf",           [1.2,  0.10, 0.82]),
        ("pudding_box",           "008_pudding_box.urdf",           [1.2, -0.10, 0.82]),
        ("gelatin_box",           "009_gelatin_box.urdf",           [1.2, -0.3,  0.82]),
        ("bowl",                  "024_bowl.urdf",                  [1.0,  0.2,  0.82]),
        ("mug",                   "025_mug.urdf",                   [1.0,  0.0,  0.82]),
        ("sponge",                "026_sponge.urdf",                [1.0, -0.2,  0.82]),
        ("plate",                 "029_plate.urdf",                 [0.7,  0.3,  0.82]),
        ("spatula",               "033_spatula.urdf",               [0.7,  0.10, 0.82]),
        ("foam_brick",            "061_foam_brick.urdf",            [0.7, -0.1,  0.82]),
        ("dice",                  "062_dice.urdf",                  [0.7, -0.3,  0.82]),
        ("marbles",               "063-a_marbles.urdf",             [0.45, 0.2,  0.82]),
        ("cups",                  "065-a_cups.urdf",                [0.45, 0.0,  0.82]),
        ("timer",                 "076_timer.urdf",                 [0.45,-0.2,  0.82]),
        ("wood_block",            "036_wood_block.urdf",            [0.45,-0.35, 0.82]),
    ],
    4: [
        ("hammer",                "048_hammer.urdf",                [1.2,  0.3,  0.82]),
        ("peg_test",              "071_nine_hole_peg_test.urdf",    [1.2,  0.10, 0.82]),
        ("extra_large_clamp",     "052_extra_large_clamp.urdf",     [1.2, -0.10, 0.82]),
        ("large_clamp",           "051_large_clamp.urdf",           [1.2, -0.3,  0.82]),
        ("medium_clamp",          "050_medium_clamp.urdf",          [1.0,  0.2,  0.82]),
        ("wood_block",            "036_wood_block.urdf",            [1.0,  0.0,  0.82]),
        ("phillips_screwdriver",  "043_phillips_screwdriver.urdf",  [1.0, -0.2,  0.82]),
        ("flat_screwdriver",      "044_flat_screwdriver.urdf",      [0.7,  0.3,  0.82]),
        ("large_marker",          "040_large_marker.urdf",          [0.7,  0.1,  0.82]),
        ("small_marker",          "041_small_marker.urdf",          [0.7, -0.1,  0.82]),
        ("bowl",                  "024_bowl.urdf",                  [0.7, -0.3,  0.82]),
        ("adjustable_wrench",     "042_adjustable_wrench.urdf",     [0.45, 0.2,  0.82]),
        ("chain",                 "059_chain.urdf",                 [0.45, 0.0,  0.82]),
        ("foam_brick",            "061_foam_brick.urdf",            [0.45,-0.2,  0.82]),
        ("wood_blocks",           "070-a_colored_wood_blocks.urdf", [0.45,-0.35, 0.82]),
    ],
    5: [
        ("cracker_box",           "003_cracker_box.urdf",           [1.2,  0.3,  0.82]),
        ("pudding_box",           "008_pudding_box.urdf",           [1.2,  0.10, 0.82]),
        ("gelatin_box",           "009_gelatin_box.urdf",           [1.2, -0.10, 0.82]),
        ("plate",                 "029_plate.urdf",                 [1.2, -0.3,  0.82]),
        ("bowl",                  "024_bowl.urdf",                  [1.0,  0.2,  0.82]),
        ("mug",                   "025_mug.urdf",                   [1.0,  0.0,  0.82]),
        ("sponge",                "026_sponge.urdf",                [1.0, -0.2,  0.82]),
        ("spatula",               "033_spatula.urdf",               [0.7,  0.3,  0.82]),
        ("wood_block",            "036_wood_block.urdf",            [0.7,  0.10, 0.82]),
        ("foam_brick",            "061_foam_brick.urdf",            [0.7, -0.1,  0.82]),
        ("large_marker",          "040_large_marker.urdf",          [0.7, -0.3,  0.82]),
        ("flat_screwdriver",      "044_flat_screwdriver.urdf",      [0.45, 0.2,  0.82]),
        ("cups",                  "065-a_cups.urdf",                [0.45, 0.0,  0.82]),
        ("phillips_screwdriver",  "043_phillips_screwdriver.urdf",  [0.45,-0.2,  0.82]),
        ("rubiks_cube",           "077_rubiks_cube.urdf",           [0.45,-0.35, 0.82]),
    ],
}


def render_case(case_id: int) -> None:
    specs = CASES[case_id]
    print(f"\n{'='*60}\n[Case {case_id}] 물체 {len(specs)}개 로드\n{'='*60}")

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(1 / 240.0)
    p.setGravity(0, 0, -9.8)
    flags = p.URDF_USE_INERTIA_FROM_FILE

    p.loadURDF("plane.urdf")
    p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, basePosition=[0, 0, 0.65])
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])

    # 테이블/로봇 먼저 안정화
    for _ in range(50):
        p.stepSimulation()

    object_registry: dict[str, int] = {}
    print(f"[1] 물체 {len(specs)}개 순차 로드 + 개별 안정화")
    for label, urdf_file, pos in specs:
        urdf_path = os.path.join(ycb_dir, urdf_file)
        if not os.path.exists(urdf_path):
            print(f"  [경고] URDF 없음 → 건너뜀: {urdf_file}")
            continue
        try:
            # 콜리전 폭발 방지: 각 물체를 조금 더 높은 z에서 드롭
            drop_pos = [pos[0], pos[1], max(pos[2], 0.85)]
            # YCB URDF들은 내부에 scale="10 10 10" 박혀있어서 globalScaling=0.1 필수
            body_id = p.loadURDF(
                urdf_path,
                basePosition=drop_pos,
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                globalScaling=0.1,
                flags=flags,
            )
            object_registry[label] = body_id
            # 다음 물체 로드 전 현재 물체 낙하/안정화
            for _ in range(120):
                p.stepSimulation()
        except p.error as e:
            print(f"  [경고] {label} 로드 실패: {e}")

    print(f"[1b] 전체 최종 안정화 (300 step)")
    for _ in range(300):
        p.stepSimulation()

    print("[2] 물체 정보 수집 중...")
    scene_info = collect_scene_info(object_registry)
    for obj in scene_info["objects"]:
        body_id = object_registry.get(obj["label"])
        if body_id is None:
            continue
        aabb_min, aabb_max = p.getAABB(body_id)
        obj["aabb_min"] = [round(v, 4) for v in aabb_min]
        obj["aabb_max"] = [round(v, 4) for v in aabb_max]
        obj["center_world"] = [
            round((aabb_min[0] + aabb_max[0]) / 2, 4),
            round((aabb_min[1] + aabb_max[1]) / 2, 4),
            round((aabb_min[2] + aabb_max[2]) / 2, 4),
        ]

    out_json = os.path.join(script_dir, f"scene_info_case{case_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scene_info, f, ensure_ascii=False, indent=2)
    print(f"[저장] {out_json}")

    print("[3] 카메라 렌더링 중...")
    width, height = 640, 480
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.8, 0.0, 0.8],
        distance=1.6,
        yaw=45,
        pitch=-40,
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=100
    )
    _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
    rgb_array = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
    out_png = os.path.join(script_dir, f"scene_capture_case{case_id}.png")
    Image.fromarray(rgb_array).save(out_png)
    print(f"[저장] {out_png}")

    p.disconnect()

    print(f"\n[Case {case_id} 완료] 물체 좌표 요약:")
    for obj in scene_info["objects"]:
        print(f"  {obj['label']:25s} center={obj.get('center_world')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        choices=list(CASES.keys()),
        help="특정 case만 생성 (생략 시 1~5 전체)",
    )
    args = parser.parse_args()

    targets = [args.case] if args.case else sorted(CASES.keys())
    for cid in targets:
        render_case(cid)

    print("\n" + "=" * 60)
    print("[모든 case 생성 완료]")
    print("Module 2-C provider 실행 시:")
    print("  --scene-info scene_info_case{N}.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
