import os
import math
import pybullet as p
import open3d as o3d
import numpy as np


def get_object_info(obj_id, label=None):
    """
    PyBullet에서 로드된 물체 하나의 위치, 방향, 크기(AABB) 정보를 딕셔너리로 반환.
    
    Args:
        obj_id (int): pybullet loadURDF로 반환된 body ID
        label (str): 사람이 읽기 쉬운 물체 이름 (예: "mug", "cracker_box")
    
    Returns:
        dict: position, orientation(euler), size(AABB 기반) 정보
    """
    pos, orn_quat = p.getBasePositionAndOrientation(obj_id)
    orn_euler = p.getEulerFromQuaternion(orn_quat)  # (roll, pitch, yaw) in radians
    
    # AABB(Axis-Aligned Bounding Box)로 물체의 대략적 크기 추정
    aabb_min, aabb_max = p.getAABB(obj_id)
    size = (
        round(aabb_max[0] - aabb_min[0], 4),
        round(aabb_max[1] - aabb_min[1], 4),
        round(aabb_max[2] - aabb_min[2], 4),
    )
    
    return {
        "id": obj_id,
        "label": label if label else f"object_{obj_id}",
        "position": {
            "x": round(pos[0], 4),
            "y": round(pos[1], 4),
            "z": round(pos[2], 4),
        },
        "orientation_euler_deg": {
            "roll":  round(math.degrees(orn_euler[0]), 2),
            "pitch": round(math.degrees(orn_euler[1]), 2),
            "yaw":   round(math.degrees(orn_euler[2]), 2),
        },
        "size_aabb_m": {
            "width":  size[0],
            "depth":  size[1],
            "height": size[2],
        },
    }


def collect_scene_info(object_registry: dict) -> dict:
    """
    여러 물체의 정보를 한꺼번에 수집.

    Args:
        object_registry (dict): { "label": body_id } 형태의 딕셔너리
            예) {"cube": obj1, "mug": mug_id, "cracker_box": cracker_box_id}

    Returns:
        dict: { "objects": [ {...}, {...}, ... ] }
    """
    objects = []
    for label, body_id in object_registry.items():
        try:
            info = get_object_info(body_id, label=label)
            objects.append(info)
        except Exception as e:
            print(f"[Warning] '{label}' (id={body_id}) 정보 수집 실패: {e}")
    
    return {"objects": objects}


def get_point_cloud_from_ply(ycb_dir: str, object_name: str, body_id: int, num_points: int = 1024) -> np.ndarray:
    """
    merged_cloud.ply에서 포인트클라우드 로드 후
    PyBullet 물체의 실제 위치/방향으로 변환
    """

    ply_path = os.path.join(ycb_dir, object_name, "clouds", "merged_cloud.ply")
    
    # 포인트클라우드 로드
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)  # (N, 3)

    # 포인트 수 맞추기 (랜덤 샘플링)
    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]

    # PyBullet 물체의 실제 위치/방향 가져오기
    pos, orn_quat = p.getBasePositionAndOrientation(body_id)
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn_quat)).reshape(3, 3)

    # 포인트클라우드를 실제 월드 좌표로 변환
    points = (rot_matrix @ points.T).T + np.array(pos)

    return points  # shape: (1024, 3)