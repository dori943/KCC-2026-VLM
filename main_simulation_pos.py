import pybullet as p
import pybullet_data
import time
import os

# ─── 경로 설정 ───────────────────────────────────────────────
URDF_PATH = "/workspace/KCC-2026-VLM/outputs/combined_spatula_gelatin_box.urdf"

# 테이블 높이에 맞게 오브젝트 Z 위치 조정 (table.urdf 기준 ~0.625m)
TABLE_BASE_POSITION = [0.5, 0.0, 0.0]
TABLE_HEIGHT = 0.625          # table.urdf 의 실제 상판 높이 (필요시 조정)
OBJECT_Z_OFFSET = 0.05        # 상판 위 약간 띄우기

# ─── PyBullet 초기화 ─────────────────────────────────────────
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())   # plane.urdf / table 검색 경로
p.setGravity(0, 0, -9.81)

# 카메라 초기 시점
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0.5, 0.0, 0.4],
)

# ─── 1) 바닥 평면 ────────────────────────────────────────────
plane_id = p.loadURDF("plane.urdf")
print(f"[✓] plane loaded  (id={plane_id})")

# ─── 2) 테이블 ───────────────────────────────────────────────
table_id = p.loadURDF(
    "table/table.urdf",
    basePosition=TABLE_BASE_POSITION,
    useFixedBase=True,
)
print(f"[✓] table loaded  (id={table_id})")

# 테이블 AABB 로 실제 상판 높이를 자동 측정
t_aabb_min, t_aabb_max = p.getAABB(table_id)
table_top_z = t_aabb_max[2]
print(f"    table top Z = {table_top_z:.4f} m")

# ─── 3) combined object ──────────────────────────────────────
# 오브젝트 URDF를 일단 원점에 로드해 바운딩박스 하단 높이를 측정
temp_id = p.loadURDF(
    URDF_PATH,
    basePosition=[0, 0, 0],
    useFixedBase=True,
)
obj_aabb_min, obj_aabb_max = p.getAABB(temp_id)
obj_bottom_offset = -obj_aabb_min[2]   # 기준점에서 바닥까지의 거리
p.removeBody(temp_id)

# 테이블 위 정중앙에 배치
obj_x = TABLE_BASE_POSITION[0]
obj_y = TABLE_BASE_POSITION[1]
obj_z = table_top_z + obj_bottom_offset + OBJECT_Z_OFFSET

object_id = p.loadURDF(
    URDF_PATH,
    basePosition=[obj_x, obj_y, obj_z],
    useFixedBase=False,
)
print(f"[✓] object loaded (id={object_id})  at z={obj_z:.4f} m")

# ─── 시뮬레이션 루프 ─────────────────────────────────────────
print("\n[GUI 실행 중] 창을 닫거나 Ctrl+C 로 종료하세요.\n")
try:
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
except KeyboardInterrupt:
    print("종료합니다.")
finally:
    p.disconnect()