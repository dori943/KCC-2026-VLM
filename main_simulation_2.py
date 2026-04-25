<<<<<<< HEAD
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

=======
>>>>>>> origin/subin/module2c-3-pipeline
import os
import sys
import time
import json
<<<<<<< HEAD
import numpy as np
import pybullet as p
import pybullet_data

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ycb_dir = r"/workspace/KCC-2026-VLM/data/object2urdf/examples/ycb"
sys.path.append(ycb_dir)

=======
import torch
import pybullet as p
import pybullet_data

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ycb_dir = r"/root/KCC2026_VLM/data/object2urdf/examples/ycb"
sys.path.append(ycb_dir)

# ── 직접 작성한 모듈 ────────────────────────────────────────────────────────────
>>>>>>> origin/subin/module2c-3-pipeline
from object_info_collector import collect_scene_info
from gpt_scene_query import query_gpt_about_scene, query_gpt_pick_target
from robot_controller_2 import (
    render_camera,
    get_grasp_pose_from_vl_grasp,
<<<<<<< HEAD
    get_bbox_from_pybullet,
=======
>>>>>>> origin/subin/module2c-3-pipeline
    load_vl_grasp_models,
    move_end_effector_to,
    open_gripper,
    close_gripper,
    reset_to_home,
    verify_pointcloud_accuracy,
)

<<<<<<< HEAD
# ── PyBullet 초기화 ───────────────────────────────────────────────────────────
p.connect(p.GUI)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
=======
# ── PyBullet 초기화 ─────────────────────────────────────────────────────────────
p.connect(p.GUI)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5, cameraYaw=45, cameraPitch=-45,
>>>>>>> origin/subin/module2c-3-pipeline
    cameraTargetPosition=[0.55, -0.35, 0.8]
)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(1 / 240.)
p.setGravity(0, 0, -9.8)

<<<<<<< HEAD
# ── 환경 로드 ─────────────────────────────────────────────────────────────────
flags   = p.URDF_USE_INERTIA_FROM_FILE
planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
pandaId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, basePosition=[0, 0, 0.65])
tableId = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])

# ── 물체 로드 ─────────────────────────────────────────────────────────────────
=======
# ── 환경 로드 ────────────────────────────────────────────────────────────────────
flags = p.URDF_USE_INERTIA_FROM_FILE

planeId  = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
pandaId  = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, basePosition=[0, 0, 0.65])
tableId  = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])

# ── 물체 로드 ────────────────────────────────────────────────────────────────────
>>>>>>> origin/subin/module2c-3-pipeline
cube_id = p.loadURDF("cube_small.urdf", [0.3, -0.3, 0.7])

chips_can_id = p.loadURDF(
    os.path.join(ycb_dir, "001_chips_can.urdf"),
    basePosition=[1.0, 0.0, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)
<<<<<<< HEAD
=======

>>>>>>> origin/subin/module2c-3-pipeline
apple_id = p.loadURDF(
    os.path.join(ycb_dir, "013_apple.urdf"),
    basePosition=[0.6, -0.1, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)
<<<<<<< HEAD
=======

>>>>>>> origin/subin/module2c-3-pipeline
fork_id = p.loadURDF(
    os.path.join(ycb_dir, "030_fork.urdf"),
    basePosition=[0.5, 0.3, 0.8],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    globalScaling=0.1, flags=flags
)
<<<<<<< HEAD
=======

>>>>>>> origin/subin/module2c-3-pipeline
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

<<<<<<< HEAD
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
=======
# ── 물체 레지스트리: { "사람이 읽기 쉬운 이름": body_id } ────────────────────────
object_registry = {
    "cube_small":       cube_id,
    "chips_can":        chips_can_id,
    "apple":            apple_id,
    "fork":             fork_id,
    "cracker_box":      cracker_box_id,
    "mug":              mug_id,
    "mustard_bottle":   mustard_bottle_id,
    "large_clamp":      large_clamp_id,
}

import pybullet as p

# 현재 카메라 상태 가져오기
camera_info = p.getDebugVisualizerCamera()

# 결과 해석
target_pos = camera_info[11]  # [x, y, z] 형태의 cameraTargetPosition
print(f"현재 카메라 렌즈 정중앙이 가리키는 월드 좌표: {target_pos}")

# ── 물리 시뮬레이션을 충분히 돌려서 물체가 안정화되도록 대기 ─────────────────────
STABILIZE_STEPS = 500   # 약 2초 (500 * 1/240)
print(f"[시뮬레이션] {STABILIZE_STEPS} 스텝 동안 물체 안정화 중...")
for _ in range(STABILIZE_STEPS):
    p.stepSimulation()
    time.sleep(1. / 240)

# ── 물체 정보 수집 ────────────────────────────────────────────────────────────────
print("[정보 수집] 물체 상태 수집 중...")
scene_info = collect_scene_info(object_registry)
print(json.dumps(scene_info, ensure_ascii=False, indent=2))

# ── ChatGPT API 쿼리 ──────────────────────────────────────────────────────────────
print("\n[GPT] ChatGPT API에 씬 정보 전송 중...")
try:
    result = query_gpt_about_scene(scene_info, model="gpt-4o")

    print("\n===== GPT 응답 (파싱된 JSON) =====")
    if result["parsed"]:
        print(json.dumps(result["parsed"], ensure_ascii=False, indent=2))
    else:
        print("[주의] JSON 파싱 실패, 원문 응답:")
        print(result["raw_response"])

    # 결과를 파일로 저장
    with open("gpt_scene_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {"scene_info": scene_info, "gpt_response": result["parsed"] or result["raw_response"]},
            f, ensure_ascii=False, indent=2
        )
    print("\n[저장] 결과가 'gpt_scene_result.json'에 저장되었습니다.")

except Exception as e:
    print(f"[오류] GPT API 호출 실패: {e}")

# ── 가상 카메라 렌더링 (GPT Vision + VL-Grasp 공용) ──────────────────────────
#
#  카메라 파라미터 가이드:
#    cam_target   : 씬 중심 — 테이블 중앙 기준으로 조정
#    cam_distance : 너무 가까우면 물체 잘림, 너무 멀면 작게 보임 (1.0~1.5 권장)
#    cam_yaw      : 0° = 정면, 90° = 우측면
#    cam_pitch    : -30 ~ -60° 사이가 tabletop 인식에 좋음
#
CAM_CONFIG = {
    "cam_target":   [0.55, -0.35, 0.8],
    "cam_distance": 1.2,
    "cam_yaw":      45,
    "cam_pitch":    -45,
}

print("[3] 가상 카메라 렌더링 중...")
rgb, depth, proj_matrix, view_matrix = render_camera(**CAM_CONFIG)
 
# 렌더링 결과를 파일로 저장 (디버깅용)
try:
    from PIL import Image
    Image.fromarray(rgb).save("scene_capture.png")
    print("[3] 씬 이미지 저장 완료: scene_capture.png")
except ImportError:
    import cv2
    import numpy as np
    cv2.imwrite("scene_capture.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("[3] 씬 이미지 저장 완료: scene_capture.png")

>>>>>>> origin/subin/module2c-3-pipeline

# ── [3-1] 포인트클라우드 정확도 검증 ─────────────────────────────────────────
known = {label: list(p.getBasePositionAndOrientation(bid)[0])
         for label, bid in object_registry.items()}
verify_pointcloud_accuracy(depth, proj_matrix, view_matrix, known)

<<<<<<< HEAD
# ── [4] GPT Vision 타겟 선택 ──────────────────────────────────────────────────
TASK = "테이블 위에서 로봇팔이 집기 가장 적합한 물체를 하나 골라주세요."
=======
# ── [4] GPT Vision으로 타겟 물체 선택 ────────────────────────────────────────────

TASK = "테이블 위에서 과일을 집어주세요."
 
>>>>>>> origin/subin/module2c-3-pipeline
print(f"\n[4] GPT Vision 타겟 선택: '{TASK}'")

target_label = None
gpt_result   = {}
try:
<<<<<<< HEAD
    gpt_result   = query_gpt_pick_target(
        scene_info=scene_info, task=TASK,
        model="gpt-4o", rgb_image=rgb,
=======
    gpt_result = query_gpt_pick_target(
        scene_info=scene_info,
        task=TASK,
        model="gpt-4o",
        rgb_image=rgb,      # ← 가상 카메라 이미지 전달 (None으로 바꾸면 텍스트 전용)
>>>>>>> origin/subin/module2c-3-pipeline
    )
    target_label = gpt_result.get("target_label")
    print(f"[4] GPT 선택: {target_label} / 이유: {gpt_result.get('reason')}")
except Exception as e:
<<<<<<< HEAD
    print(f"[오류] GPT Vision 실패: {e}")

# ── [5] VL-Grasp 모델 로드 ────────────────────────────────────────────────────
print("\n[5] VL-Grasp 모델 로딩 중...")
use_vl_grasp = False
models       = None
=======
    print(f"[오류] GPT Vision 호출 실패: {e}")
    target_label = None


# VL-Grasp 모델 로딩
print(" VL-Grasp 모델 로딩 중...")
>>>>>>> origin/subin/module2c-3-pipeline
try:
    models = load_vl_grasp_models(
        vg_checkpoint="./VL-Grasp/logs/checkpoint_best_r50.pth",
        gn_checkpoint="./VL-Grasp/logs/checkpoint_fgc.tar",
<<<<<<< HEAD
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
=======
        device="cuda",   # GPU 없으면 "cpu"
    )
    use_vl_grasp = True
    
except Exception as e:
    print(f"[경고] VL-Grasp 로드 실패, 위치 기반 fallback 사용: {e}")
    use_vl_grasp = False

# ── [6] Grasp Pose 결정 ───────────────────────────────────────────────────────────
grasp_pos = None
grasp_orn = None
vl_language = TASK

if target_label and target_label in object_registry:
    body_id = object_registry[target_label]
 
    if use_vl_grasp:
        # ── VL-Grasp: RGB + Depth + 언어명령 → 6-DoF grasp pose ──────────────
        # GPT가 고른 물체명을 그대로 VL-Grasp 언어 명령으로 활용
        vl_language = f"pick up the {target_label.replace('_', ' ')}"
        print(f"[6] VL-Grasp 추론: '{vl_language}'")
        try:
            grasp_result = get_grasp_pose_from_vl_grasp(
                rgb=rgb,
                depth=depth,
>>>>>>> origin/subin/module2c-3-pipeline
                proj_matrix=proj_matrix,
                view_matrix=view_matrix,
                language=vl_language,
                models=models,
<<<<<<< HEAD
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
=======
            )
            grasp_pos = grasp_result["position"]
            grasp_orn = grasp_result["orientation"]
            print(f"[6] VL-Grasp 결과 → pos: {[round(v,4) for v in grasp_pos]}")
            print(f"[6] VL-Grasp 결과 → orn: {[round(v,4) for v in grasp_orn]}")    
        except Exception as e:
            print(f"[경고] VL-Grasp 추론 실패, fallback으로 전환: {e}")
            import traceback
            traceback.print_exc()  # ← 이걸로 교체
            use_vl_grasp = False  # fallback으로 넘어감
 
    if not use_vl_grasp:
        # ── Fallback: PyBullet 위치 기반 단순 grasp ──────────────────────────
        actual_pos, _ = p.getBasePositionAndOrientation(body_id)
        obj_info = next(o for o in scene_info["objects"] if o["label"] == target_label)
        obj_height = obj_info["size_aabb_m"]["height"]
        grasp_pos = [
            actual_pos[0],
            actual_pos[1],
            actual_pos[2] + obj_height / 2 + 0.05,   # 물체 중심 위 5cm
        ]
        grasp_orn = None   # move_end_effector_to 내부 기본값 사용
        print(f"[6] Fallback 위치 기반 grasp → pos: {grasp_pos}")
else:
    print(f"[오류] GPT가 선택한 '{target_label}'을(를) 레지스트리에서 찾을 수 없습니다.")


# ── 로봇팔 동작 실행 ──────────────────────────────────────────────────────────
import numpy as np

if grasp_pos:
    down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    final_orn = grasp_orn if grasp_orn else list(down_orn)

    print("\n[홈] 홈 포지션 복귀 중...")
    reset_to_home(pandaId, steps=800)

    print("\n[7] 그리퍼 열기")
>>>>>>> origin/subin/module2c-3-pipeline
    open_gripper(pandaId)

    # 접근: grasp 위 20cm
    approach_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20]
    print(f"[8-1] 접근: {[round(v,4) for v in approach_pos]}")
<<<<<<< HEAD
    move_end_effector_to(pandaId, approach_pos,
                         orientation=down_orn, steps=1000)

    # 하강: grasp 위치
    print(f"[8-2] 하강: {[round(v,4) for v in grasp_pos]}")
    move_end_effector_to(pandaId, grasp_pos,
                         orientation=down_orn, steps=800)
=======
    move_end_effector_to(pandaId, approach_pos, orientation=down_orn, steps=1000)

    # 하강: grasp 위치
    print(f"[8-2] 하강: {[round(v,4) for v in grasp_pos]}")
    move_end_effector_to(pandaId, grasp_pos, orientation=down_orn, steps=800)
>>>>>>> origin/subin/module2c-3-pipeline

    print("[9] 그리퍼 닫기")
    close_gripper(pandaId)

    # 들어올리기
    lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.30]
    print(f"[10] 들어올리기: {[round(v,4) for v in lift_pos]}")
<<<<<<< HEAD
    move_end_effector_to(pandaId, lift_pos,
                         orientation=down_orn, steps=800)
=======
    move_end_effector_to(pandaId, lift_pos, orientation=down_orn, steps=800)
>>>>>>> origin/subin/module2c-3-pipeline

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
<<<<<<< HEAD
    print("[저장] grasp_result.json / scene_capture.png")

# ── [11] GUI 유지 ─────────────────────────────────────────────────────────────
=======
    print("[저장] 결과: grasp_result.json / 이미지: scene_capture.png")

# ── [11] 시뮬레이션 GUI 유지 ──────────────────────────────────────────────────────
>>>>>>> origin/subin/module2c-3-pipeline
print("\n[시뮬레이션] GUI 유지 중... (Ctrl+C로 종료)")
while True:
    p.stepSimulation()
    time.sleep(1. / 240)