"""
test_urdf_generation.py
URDF 생성 함수 테스트
"""

import os
import pybullet as p
import pybullet_data
from main_simulation_balloon import (
    load_ycb_objects,
    save_combined_tool_to_urdf,
    YCB_DIR,
)

# PyBullet 초기화
client_id = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 평면 추가
plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
print(f"[Test] Plane loaded: {plane_id}")

# YCB 물체 로드
try:
    ycb_object_ids = load_ycb_objects(ycb_dir=YCB_DIR, table_body_id=plane_id)
    print(f"[Test] Loaded {len(ycb_object_ids)} YCB objects")
    
    # spatula와 gelatin_box 확인
    spatula_id = ycb_object_ids.get("spatula")
    gelatin_id = ycb_object_ids.get("gelatin_box")
    
    if spatula_id is not None and gelatin_id is not None:
        print(f"[Test] Spatula ID: {spatula_id}, Gelatin box ID: {gelatin_id}")
        
        # URDF 생성 테스트
        offset = [-0.16, 0.0, 0.0]  # 테스트 offset
        success, result = save_combined_tool_to_urdf(
            base_body_id=spatula_id,
            attach_body_id=gelatin_id,
            base_label="spatula",
            attach_label="gelatin_box",
            contact_offset=offset,
            ycb_dir=YCB_DIR,
        )
        
        if success:
            print(f"\n[Test] ✅ URDF 생성 성공!")
            print(f"[Test] 저장 경로: {result}")
            
            # 생성된 URDF 내용 확인
            with open(result, 'r') as f:
                content = f.read()
                print(f"\n[Test] URDF 파일 크기: {len(content)} bytes")
                print(f"\n[Test] URDF 처음 500자:")
                print(content[:500])
        else:
            print(f"\n[Test] ❌ URDF 생성 실패: {result}")
    else:
        print(f"[Test] ❌ spatula 또는 gelatin_box를 찾을 수 없습니다")
        
except Exception as e:
    print(f"[Test] ❌ 오류: {e}")
    import traceback
    traceback.print_exc()

p.disconnect()
