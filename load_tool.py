"""
load_tool.py
생성된 URDF 파일을 PyBullet 시뮬레이션에서 로드하고 시각화합니다.

사용법:
    python load_tool.py              # GUI 모드 (기본값)
    python load_tool.py --headless   # 헤드리스 모드 (이미지 저장)
"""

import os
import sys
import time
import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image


def render_camera_view(client_id, tool_id, view_name, camera_distance, camera_yaw, camera_pitch):
    """
    지정된 각도에서 카메라 렌더링을 수행하고 이미지를 저장합니다.
    """
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=[0.5, 0.0, 0.5]
    )
    
    # 카메라 내부 파라미터
    width = 960
    height = 720
    
    # 카메라 행렬 설정
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.5, 0.0, 0.5],
        distance=camera_distance,
        yaw=camera_yaw,
        pitch=camera_pitch,
        roll=0,
        upAxisIndex=2
    )
    
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.01,
        farVal=10
    )
    
    # 이미지 렌더링
    width, height, rgb, depth, mask = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # RGB 이미지 저장
    rgb_array = np.array(rgb, dtype=np.uint8)
    rgb_array = rgb_array.reshape((height, width, 4))[:, :, :3]  # RGBA -> RGB
    
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"tool_view_{view_name}.png")
    img = Image.fromarray(rgb_array)
    img.save(output_path)
    
    return output_path


def main():
    headless = "--headless" in sys.argv
    
    # PyBullet 연결
    if headless:
        client_id = p.connect(p.DIRECT)
        print("[Load] 헤드리스 모드로 실행 중...")
    else:
        client_id = p.connect(p.GUI)
        print("[Load] GUI 모드로 실행 중...")
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 물리 시뮬레이션 설정
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0, numSubSteps=1)
    
    # 평면(테이블) 추가
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
    print(f"[Load] 평면 로드됨: body_id={plane_id}")
    
    # 생성된 URDF 파일 경로
    urdf_path = os.path.join(os.path.dirname(__file__), "outputs", "combined_spatula_gelatin_box.urdf")
    
    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF 파일이 없습니다: {urdf_path}")
        print(f"[INFO] 먼저 main_simulation_balloon.py를 실행하여 URDF 파일을 생성하세요.")
        p.disconnect()
        return
    
    # URDF 파일 로드
    try:
        tool_id = p.loadURDF(urdf_path, [0.5, 0.0, 0.5], globalScaling=1.0)
        print(f"[Load] 조합 도구 로드됨: body_id={tool_id}")
        print(f"[Load] URDF 경로: {urdf_path}")
        
        # 로드된 물체 정보 출력
        pos, orn = p.getBasePositionAndOrientation(tool_id)
        print(f"[Load] 초기 위치: {pos}")
        print(f"[Load] 초기 방향: {orn}")
        
        # AABB (Axis-Aligned Bounding Box) 정보
        aabb_min, aabb_max = p.getAABB(tool_id)
        print(f"[Load] AABB 최소점: {aabb_min}")
        print(f"[Load] AABB 최대점: {aabb_max}")
        
        # 동역학 정보
        dyn_info = p.getDynamicsInfo(tool_id, -1)
        print(f"[Load] 질량: {dyn_info[0]:.3f} kg")
        print(f"[Load] 마찰 계수: {dyn_info[1]:.3f}")
        
        # 링크 정보 출력
        num_links = p.getNumJoints(tool_id)
        print(f"[Load] 링크 개수: {num_links}")
        for i in range(num_links):
            link_info = p.getJointInfo(tool_id, i)
            print(f"  - Joint {i}: {link_info[1].decode()} (type={link_info[2]})")
        
    except Exception as e:
        print(f"[ERROR] URDF 로드 실패: {e}")
        p.disconnect()
        return
    
    # 헤드리스 모드: 이미지 렌더링
    if headless:
        print("\n[Render] 다양한 각도에서 도구를 렌더링 중...")
        
        # 여러 각도에서 렌더링
        views = [
            ("front", 1.0, 0, -30),        # 정면, 아래쪽 각도
            ("side", 1.0, 90, -30),        # 옆면
            ("top", 1.2, 0, -80),          # 위에서 본 전망
            ("isometric", 1.2, 45, -45),   # 등각 투영
            ("rear", 1.0, 180, -30),       # 뒷면
        ]
        
        for view_name, distance, yaw, pitch in views:
            try:
                output_path = render_camera_view(
                    client_id, tool_id, view_name, distance, yaw, pitch
                )
                print(f"[Render] {view_name} 뷰 저장됨: {output_path}")
            except Exception as e:
                print(f"[Render][WARN] {view_name} 뷰 렌더링 실패: {e}")
        
        print("[Render] 렌더링 완료!")
        
    else:
        # GUI 모드: 대화형 시뮬레이션
        # 카메라 설정 (좋은 뷰포인트)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, 0.0, 0.5]
        )
        
        print("\n[Simulation] 시뮬레이션이 시작되었습니다.")
        print("[Simulation] PyBullet GUI 창에서 도구를 확인하세요.")
        print("[Simulation] 종료하려면 Ctrl+C를 누르세요.\n")
        
        # 시뮬레이션 루프
        try:
            while True:
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
        except KeyboardInterrupt:
            print("\n[Simulation] 시뮬레이션 종료...")
    
    p.disconnect()
    print("[Simulation] PyBullet 연결 해제")


if __name__ == "__main__":
    main()
