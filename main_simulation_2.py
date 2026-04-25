"""
main_simulation.py
────────────────────────────────────────────────────────────────────────────────
GPT Vision + VL-Grasp 기반 tabletop 물체 파지 시뮬레이션

파이프라인:
  1. PyBullet 씬 초기화 및 물체 로드
  2. 물체 정보 수집 → GPT 씬 분석
  3. 가상 카메라 RGB-D 렌더링
  4. GPT Vision → 타겟 물체 선택
  5. PyBullet AABB → 정확한 픽셀 bbox 계산
  6. VL-Grasp (segmentation + FGC-GraspNet) → 6-DoF grasp pose 추론
  7. Franka Panda IK (seed 적용) → 파지 실행
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import json
import numpy as np
import pybullet as p
import pybullet_data

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ycb_dir = r"/workspace/KCC-2026-VLM/data/object2urdf/examples/ycb"
sys.path.append(ycb_dir)

from object_info_collector import collect_scene_info
from gpt_scene_query import query_gpt_about_scene, query_gpt_pick_target
from robot_controller_2 import (
    render_camera,
    get_grasp_pose_from_vl_grasp,
    get_bbox_from_pybullet,
    load_vl_grasp_models,
    move_end_effector_to,
    open_gripper,
    close_gripper,
    reset_to_home,
    verify_pointcloud_accuracy,
)

# ── PyBullet 초기화 ───────────────────────────────────────────────────────────
p.connect(p.GUI)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
    cameraTargetPosition=[0.55, -0.35, 0.8]
)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(1 / 240.)
p.setGravity(0, 0, -9.8)

# ── 환경 로드 ─────────────────────────────────────────────────────────────────
flags   = p.URDF_USE_INERTIA_FROM_FILE
planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
pandaId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, basePosition=[0, 0, 0.65])
tableId = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])

# ── 물체 로드 ─────────────────────────────────────────────────────────────────
cube_id = p.loadURDF("cube_small.urdf", [0.3, -0.3, 0.7])

chips_can_id = p.loadURDF(
    os.path.join(ycb_dir, "001_chips_can.urdf"),
    basePosition=[1.0, 0.0, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)
apple_id = p.loadURDF(
    os.path.join(ycb_dir, "013_apple.urdf"),
    basePosition=[0.6, -0.1, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)
fork_id = p.loadURDF(
    os.path.join(ycb_dir, "030_fork.urdf"),
    basePosition=[0.5, 0.3, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)
cracker_box_id = p.loadURDF(
    os.path.join(ycb_dir, "003_cracker_box.urdf"),
    basePosition=[0.3, 0.2, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)
mug_id = p.loadURDF(
    os.path.join(ycb_dir, "025_mug.urdf"),
    basePosition=[0.8, -0.2, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)
mustard_bottle_id = p.loadURDF(
    os.path.join(ycb_dir, "006_mustard_bottle.urdf"),
    basePosition=[0., -0.3, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)
large_clamp_id = p.loadURDF(
    os.path.join(ycb_dir, "051_large_clamp.urdf"),
    basePosition=[0.5, -0.3, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)

object_registry = {
    "cube_small":     cube_id,
    "chips_can":      chips_can_id,
    "apple":          apple_id,
    "fork":           fork_id,
    "cracker_box":    cracker_box_id,
    "mug":            mug_id,
    "mustard_bottle": mustard_bottle_id,
    "large_clamp":    large_clamp_id,
}

# ── [1] 물리 안정화 ───────────────────────────────────────────────────────────
print("[1] 물체 안정화 중...")
for _ in range(500):
    p.stepSimulation()
    time.sleep(1. / 240)

# ── [2] 물체 정보 수집 + GPT 분석 ────────────────────────────────────────────
print("[2] 물체 상태 수집 중...")
scene_info = collect_scene_info(object_registry)
print(json.dumps(scene_info, ensure_ascii=False, indent=2))

print("\n[2-1] GPT 씬 분석 중...")
try:
    result = query_gpt_about_scene(scene_info, model="gpt-4o")
    if result["parsed"]:
        print(json.dumps(result["parsed"], ensure_ascii=False, indent=2))
    with open("gpt_scene_result.json", "w", encoding="utf-8") as f:
        json.dump({"scene_info": scene_info,
                   "gpt_response": result["parsed"] or result["raw_response"]},
                  f, ensure_ascii=False, indent=2)
    print("[저장] gpt_scene_result.json")
except Exception as e:
    print(f"[오류] GPT 씬 분석 실패: {e}")

# ── [3] 가상 카메라 렌더링 ────────────────────────────────────────────────────
CAM_CONFIG = {
    "cam_target":   [0.55, -0.35, 0.8],
    "cam_distance": 1.2,
    "cam_yaw":      0,
    "cam_pitch":    -45,
}

print("\n[3] 가상 카메라 렌더링 중...")
rgb, depth, proj_matrix, view_matrix = render_camera(**CAM_CONFIG)

try:
    from PIL import Image as PILImage
    PILImage.fromarray(rgb).save("scene_capture.png")
    print("[3] scene_capture.png 저장 완료")
except Exception:
    pass

# ── [3-1] 포인트클라우드 정확도 검증 ─────────────────────────────────────────
known = {label: list(p.getBasePositionAndOrientation(bid)[0])
         for label, bid in object_registry.items()}
verify_pointcloud_accuracy(depth, proj_matrix, view_matrix, known)

# ── [4] GPT Vision 타겟 선택 ──────────────────────────────────────────────────
TASK = "테이블 위에서 로봇팔이 집기 가장 적합한 물체를 하나 골라주세요."
print(f"\n[4] GPT Vision 타겟 선택: '{TASK}'")

target_label = None
gpt_result   = {}
try:
    gpt_result   = query_gpt_pick_target(
        scene_info=scene_info, task=TASK,
        model="gpt-4o", rgb_image=rgb,
    )
    target_label = gpt_result.get("target_label")
    print(f"[4] GPT 선택: {target_label} / 이유: {gpt_result.get('reason')}")
except Exception as e:
    print(f"[오류] GPT Vision 실패: {e}")

# ── [5] VL-Grasp 모델 로드 ────────────────────────────────────────────────────
print("\n[5] VL-Grasp 모델 로딩 중...")
use_vl_grasp = False
models       = None
try:
    models = load_vl_grasp_models(
        vg_checkpoint="./VL-Grasp/logs/checkpoint_best_r50.pth",
        gn_checkpoint="./VL-Grasp/logs/checkpoint_fgc.tar",
        device="cuda",
    )
    use_vl_grasp = True
except Exception as e:
    print(f"[경고] VL-Grasp 로드 실패: {e}")

# ── [6] Grasp Pose 결정 ───────────────────────────────────────────────────────
grasp_pos   = None
grasp_orn   = None
vl_language = TASK

if target_label and target_label in object_registry:
    body_id     = object_registry[target_label]
    vl_language = f"pick up the {target_label.replace('_', ' ')}"

    if use_vl_grasp:
        print(f"\n[6] VL-Grasp 추론: '{vl_language}'")

        # ✅ PyBullet AABB 기반 정확한 bbox 계산
        pb_bbox = get_bbox_from_pybullet(
            body_id=body_id,
            proj_matrix=proj_matrix,
            view_matrix=view_matrix,
            cam_width=640,
            cam_height=480,
            padding=1.3
        )
        print(f"[PyBullet bbox] {target_label}: {pb_bbox}")

        try:
            grasp_result = get_grasp_pose_from_vl_grasp(
                rgb=rgb, depth=depth,
                proj_matrix=proj_matrix,
                view_matrix=view_matrix,
                language=vl_language,
                models=models,
                scene_info=scene_info,
                target_label=target_label,
                pybullet_bbox=pb_bbox,      # ✅ PyBullet bbox 전달
            )
            grasp_pos = grasp_result["position"]
            grasp_orn = grasp_result["orientation"]
            print(f"[6] VL-Grasp pos: {[round(v,4) for v in grasp_pos]}")
            print(f"[6] VL-Grasp orn: {[round(v,4) for v in grasp_orn]}")

        except Exception as e:
            print(f"[경고] VL-Grasp 실패, fallback: {e}")
            import traceback; traceback.print_exc()
            use_vl_grasp = False

    if not use_vl_grasp:
        actual_pos, _ = p.getBasePositionAndOrientation(body_id)
        obj_info      = next(o for o in scene_info["objects"]
                             if o["label"] == target_label)
        obj_height    = obj_info["size_aabb_m"]["height"]
        grasp_pos     = [actual_pos[0], actual_pos[1],
                         actual_pos[2] + obj_height / 2 + 0.03]
        grasp_orn     = None
        print(f"[6] Fallback grasp → pos: {grasp_pos}")
else:
    print(f"[오류] '{target_label}'을(를) 레지스트리에서 찾을 수 없습니다.")

# ── [7~10] 로봇팔 동작 실행 ───────────────────────────────────────────────────
if grasp_pos:
    down_orn  = p.getQuaternionFromEuler([np.pi, 0, 0])
    final_orn = grasp_orn if grasp_orn else list(down_orn)

    # 홈 포지션 복귀 (IK seed 기준점)
    print("\n[홈] 홈 포지션 복귀 중...")
    reset_to_home(pandaId, steps=800)

    print("[7] 그리퍼 열기")
    open_gripper(pandaId)

    # 접근: grasp 위 20cm
    approach_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20]
    print(f"[8-1] 접근: {[round(v,4) for v in approach_pos]}")
    move_end_effector_to(pandaId, approach_pos,
                         orientation=down_orn, steps=1000)

    # 하강: grasp 위치
    print(f"[8-2] 하강: {[round(v,4) for v in grasp_pos]}")
    move_end_effector_to(pandaId, grasp_pos,
                         orientation=down_orn, steps=800)

    print("[9] 그리퍼 닫기")
    close_gripper(pandaId)

    # 들어올리기
    lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.30]
    print(f"[10] 들어올리기: {[round(v,4) for v in lift_pos]}")
    move_end_effector_to(pandaId, lift_pos,
                         orientation=down_orn, steps=800)

    print("\n[완료] GPT Vision + VL-Grasp 기반 파지 완료!")

    result_log = {
        "task":         TASK,
        "target_label": target_label,
        "gpt_reason":   gpt_result.get("reason", ""),
        "vl_language":  vl_language,
        "grasp_pos":    [float(v) for v in grasp_pos],
        "grasp_orn":    [float(v) for v in final_orn],
        "method":       "vl_grasp" if use_vl_grasp else "fallback",
    }
    with open("grasp_result.json", "w", encoding="utf-8") as f:
        json.dump(result_log, f, ensure_ascii=False, indent=2)
    print("[저장] grasp_result.json / scene_capture.png")

# ── [11] GUI 유지 ─────────────────────────────────────────────────────────────
print("\n[시뮬레이션] GUI 유지 중... (Ctrl+C로 종료)")
while True:
    p.stepSimulation()
    time.sleep(1. / 240)