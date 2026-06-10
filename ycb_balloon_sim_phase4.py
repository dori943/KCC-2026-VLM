"""
ycb_balloon_sim.py
===================
YCB 도구 (뒤집개 + 푸딩박스) 로 나뭇가지 풍선 꺼내기 시뮬레이션

핵심 물리 메커니즘:
  1. 뒤집개 헤드를 풍선 줄 아래로 진입
  2. 헤드를 들어올려 줄을 가지 끝 방향으로 밀어냄
  3. 풍선이 가지에서 이탈
  4. 푸딩박스 무게가 도구 전체를 아래로 당겨 풍선이 너무 위로 튀지 않게 억제

성공 조건:
  - 풍선이 가지에서 분리됨 (constraint 파괴 OR 풍선이 가지 끝 바깥으로 밀려남)
  - 풍선이 터지지 않음 (과도한 충격력 없음)
  - 풍선이 지면으로 천천히 내려옴

실행:
  python ycb_balloon_sim.py             # 헤드리스, 1 에피소드
  python ycb_balloon_sim.py --gui       # PyBullet GUI
  python ycb_balloon_sim.py --trials 30 # 반복 평가
"""

import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import argparse
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


# ─────────────────────────────────────────────
#  물리 상수 / 태스크 설정
# ─────────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

@dataclass
class SceneConfig:
    # 나무
    tree_pos:          List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    trunk_height:      float = 2.0
    branch_height:     float = 1.55   # 풍선이 걸린 가지 높이 (짧은 도구에 맞게 조정)
    branch_length:     float = 0.35   # 가지 길이
    branch_angle_deg:  float = 40.0   # 가지 각도

    # 풍선
    balloon_radius:    float = 0.07
    balloon_mass:      float = 0.00165 # 헬륨+외피 총 질량. 공기 치환 질량보다 살짝 작아 천천히 뜸
    air_density:       float = 1.225   # kg/m^3, 해수면 근처 공기 밀도
    buoyancy_scale:    float = 1.0     # 1.0이면 아르키메데스 부력 그대로 적용
    balloon_linear_damping:  float = 0.55
    balloon_angular_damping: float = 0.80
    string_length:     float = 0.25
    string_max_force:  float = 0.08   # snag(줄)이 버티는 최대 힘 (N). 도구가 이보다 세게 누르면 풀림
    detach_disp:       float = 0.07   # 매달린 정지 위치에서 이만큼(m) 밀려나면 분리로 판정
    capture_radius:    float = 0.11   # 스패출러 헤드가 풍선 중심에 이만큼(m) 접근하면 "걸려서 포획"

    # 도구 시작 위치 (로봇 손 위치 가정)
    tool_start_pos:    List[float] = field(default_factory=lambda: [0.55, 0.0, 0.85])
    tool_start_rpy:    List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # 평가
    max_steps:         int  = 2400    # 10초 @ 240Hz
    control_steps:     int  = 1800    # 4 phase 스크립트는 7.5초 안에 완료
    min_phase4_steps:  int  = 240     # Phase 4 하강을 최소 1초는 수행해야 성공 인정
    burst_force_limit: float = 20.0   # 이 힘(N) 이상 충돌 → 풍선 터짐 판정
    success_z_max:     float = 0.6    # 성공시 풍선 최종 높이 상한 (너무 위로 날면 실패)


@dataclass
class EpisodeResult:
    success:            bool
    burst:              bool          # 풍선 터짐
    steps:              int
    balloon_final_pos:  List[float]
    balloon_final_z:    float
    min_dist_tool_balloon: float
    max_contact_force:  float
    detached:           bool          # 가지에서 분리됐는지
    failure_reason:     str = ""
    phase_reached:      int  = 0      # 몇 번째 phase까지 도달했는지


# ─────────────────────────────────────────────
#  URDF 인라인 생성 (assets 파일 없을 때 fallback)
# ─────────────────────────────────────────────
def _tmp_urdf(content: str, name: str) -> str:
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "w") as f:
        f.write(content)
    return path


def tree_urdf(cfg: SceneConfig) -> str:
    br = math.radians(cfg.branch_angle_deg)
    bx = cfg.branch_length * math.cos(br)
    bz = cfg.branch_length * math.sin(br)
    return f"""<?xml version="1.0"?>
<robot name="tree">
  <link name="trunk">
    <inertial><mass value="100"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 {cfg.trunk_height/2}"/>
      <geometry><cylinder radius="0.055" length="{cfg.trunk_height}"/></geometry>
      <material name="wood"><color rgba="0.50 0.25 0.05 1"/></material></visual>
    <collision><origin xyz="0 0 {cfg.trunk_height/2}"/>
      <geometry><cylinder radius="0.055" length="{cfg.trunk_height}"/></geometry></collision>
  </link>
  <link name="branch">
    <inertial><mass value="5"/><inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="{bx/2} 0 {bz/2}" rpy="0 {-(math.pi/2 - (math.pi/2 - br))} 0"/>
      <geometry><cylinder radius="0.022" length="{cfg.branch_length}"/></geometry>
      <material name="wood"><color rgba="0.50 0.25 0.05 1"/></material></visual>
    <collision><origin xyz="{bx/2} 0 {bz/2}" rpy="0 {-(math.pi/2 - (math.pi/2 - br))} 0"/>
      <geometry><cylinder radius="0.022" length="{cfg.branch_length}"/></geometry></collision>
  </link>
  <joint name="tj" type="fixed">
    <parent link="trunk"/><child link="branch"/>
    <origin xyz="0 0 {cfg.branch_height}"/>
  </joint>
</robot>"""


def balloon_urdf(cfg: SceneConfig) -> str:
    r = cfg.balloon_radius
    return f"""<?xml version="1.0"?>
<robot name="balloon">
  <link name="balloon_body">
    <inertial><mass value="{cfg.balloon_mass}"/>
      <inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><sphere radius="{r}"/></geometry>
      <material name="red"><color rgba="1.0 0.15 0.15 0.9"/></material></visual>
    <collision>
      <geometry><sphere radius="{r}"/></geometry>
      <contact_coefficients mu="0.5" kp="2000" kd="5"/>
    </collision>
  </link>
</robot>"""


# main_simulation_balloon.py 의 save_combined_tool_to_urdf() 가 저장하는 파일 이름.
#   output_path = outputs/combined_{base_label}_{attach_label}.urdf
# 즉 spatula + gelatin_box 조합은 아래 이름으로 저장된다.
COMBINED_TOOL_BASENAME = "combined_spatula_gelatin_box.urdf"


def find_combined_tool_urdf() -> Optional[str]:
    """
    main_simulation_balloon.py 에서 조합/저장한 실제 도구(spatula + gelatin_box)
    URDF 파일 경로를 찾는다. 찾지 못하면 None 을 반환한다.

    탐색 순서:
      1) 환경변수 COMBINED_TOOL_URDF (명시 경로)
      2) <repo>/outputs/combined_spatula_gelatin_box.urdf  ← 기본 저장 위치
      3) <repo>/assets/combined_spatula_gelatin_box.urdf   ← assets 에 복사한 경우
      4) 과거 이름들 (호환용)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates: List[str] = []

    env_path = os.environ.get("COMBINED_TOOL_URDF")
    if env_path:
        candidates.append(env_path)

    # main_simulation_balloon.py 의 실제 저장 위치
    candidates.append(os.path.join(here, "outputs", COMBINED_TOOL_BASENAME))
    # assets 폴더에 복사해 둔 경우
    candidates.append(os.path.join(ASSETS_DIR, COMBINED_TOOL_BASENAME))
    # 과거/대체 이름 호환
    candidates.append(os.path.join(ASSETS_DIR, "spatula_gelatin_box_combined.urdf"))
    candidates.append(os.path.join(ASSETS_DIR, "spatula_pudding_tool.urdf"))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def combined_tool_urdf() -> str:
    """뒤집개 + 젤라틴박스 조합 도구. 실제 조합 URDF 가 없을 때 쓰는 인라인 fallback."""
    real = find_combined_tool_urdf()
    if real is not None:
        return real
    return inline_fallback_tool_urdf()


def inline_fallback_tool_urdf() -> str:
    """조합 도구 파일이 전혀 없을 때 쓰는 임시 인라인 도구 (스패출러 + 박스)."""
    # assets 없으면 간단 버전으로
    return _tmp_urdf("""<?xml version="1.0"?>
<robot name="spatula_gelatin_tool">
  <link name="base">
    <inertial><mass value="0.15"/><origin xyz="0 0 0.085"/>
      <inertia ixx="3.6e-4" iyy="3.6e-4" izz="2.8e-5" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.085"/>
      <geometry><cylinder radius="0.011" length="0.17"/></geometry>
      <material name="dg"><color rgba="0.25 0.25 0.25 1"/></material></visual>
    <collision><origin xyz="0 0 0.085"/>
      <geometry><cylinder radius="0.011" length="0.17"/></geometry></collision>
  </link>
  <link name="spatula_head">
    <inertial><mass value="0.05"/><origin xyz="0 0 0.04"/>
      <inertia ixx="1.5e-5" iyy="2e-5" izz="3e-5" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.04"/>
      <geometry><box size="0.06 0.08 0.003"/></geometry>
      <material name="sv"><color rgba="0.85 0.85 0.85 1"/></material></visual>
    <collision><origin xyz="0 0 0.04"/>
      <geometry><box size="0.06 0.08 0.003"/></geometry></collision>
  </link>
  <joint name="bh" type="fixed"><parent link="base"/><child link="spatula_head"/>
    <origin xyz="0 0 0.2"/></joint>
  <link name="pudding_box">
    <inertial><mass value="0.187"/><origin xyz="0 0 0"/>
      <inertia ixx="2.1e-4" iyy="5.2e-4" izz="6.0e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0"/>
      <geometry><box size="0.173 0.093 0.039"/></geometry>
      <material name="pb"><color rgba="0.55 0.27 0.07 1"/></material></visual>
    <collision><origin xyz="0 0 0"/>
      <geometry><box size="0.173 0.093 0.039"/></geometry></collision>
  </link>
  <joint name="bp" type="fixed"><parent link="base"/><child link="pudding_box"/>
    <origin xyz="0.0 0.06 0.085"/></joint>
</robot>""", "spatula_pudding_tool_fallback.urdf")


# ─────────────────────────────────────────────
#  스크립트 제어: 4 Phase 동작
# ─────────────────────────────────────────────
class ScriptController:
    """
    Phase 1 (0~25%):  스패출러 헤드를 풍선(가지 아래에 매달림)까지 아래에서 접근
    Phase 2 (25~50%): 헤드를 풍선에 밀착 → 줄에서 떼어내며 도구에 "걸려" 포획
    Phase 3 (50~75%): 포획한 풍선을 아래로 끌어내리기 시작 (젤라틴 무게가 도와줌)
    Phase 4 (75~100%): 계속 하강시켜 손 높이까지 내림
    """

    def __init__(self, cfg: SceneConfig, head_offset: float = 0.23):
        self.cfg = cfg
        # 도구 base 원점 대비 스패출러 헤드(위쪽 끝) 높이. 시뮬레이터가 로드된
        # 실제 도구 AABB 로 갱신해 주면 임의/실제 도구 모두에 대응한다.
        self.head_offset = head_offset
        self._compute_waypoints()

    def _compute_waypoints(self):
        cfg = self.cfg
        br  = math.radians(cfg.branch_angle_deg)
        # 가지 끝 위치
        self.branch_tip = [
            cfg.tree_pos[0] + cfg.branch_length * math.cos(br),
            cfg.tree_pos[1],
            cfg.branch_height + cfg.branch_length * math.sin(br)
        ]
        # 풍선이 매달린 위치 (가지 끝 *아래* string_length 만큼)
        self.balloon_hang_pos = [
            self.branch_tip[0],
            self.branch_tip[1],
            self.branch_tip[2] - cfg.string_length
        ]

    def get_velocity(self, step: int, tool_pos: list, balloon_pos: list) -> list:
        """
        풍선은 가지 *아래* 에 매달려 있다(부력으로 떠올랐다가 줄이 가지에 걸린
        상태, 손이 닿지 않음). 로봇 손(=도구)은 풍선보다 낮은 곳에서 시작하므로,
        스패출러 헤드를 *아래에서* 풍선까지 올려 걸어 포획한 뒤, 젤라틴 무게로
        도구 전체를 아래로 내려 풍선을 손 높이까지 가져온다.

        헤드는 base 보다 self.head_offset 만큼 위에 있으므로, '헤드를 z 에
        두려면' base 목표 z = z - head_offset 으로 명령한다.
        """
        cfg   = self.cfg
        frac  = min(step / cfg.control_steps, 0.999)
        ho    = self.head_offset
        speed = 0.40  # m/s
        bx, by, bz = balloon_pos

        if frac < 0.25:
            # Phase 1: 헤드를 풍선 중심까지 아래에서 접근
            target = [bx, by, bz - ho]
            return self._vel_toward(tool_pos, target, speed)

        elif frac < 0.50:
            # Phase 2: 풍선에 밀착해 확실히 걸기(포획) — 살짝 더 밀어 올림
            target = [bx, by, bz - ho + 0.01]
            return self._vel_toward(tool_pos, target, speed * 0.5)

        else:
            # Phase 3~4: 포획한 풍선을 아래로 끌어내림.
            # xy 는 풍선(=헤드 근처)에 맞춰 흔들림을 줄이고, z 는 바닥 근처까지 하강.
            v = self._vel_toward([tool_pos[0], tool_pos[1], 0.0],
                                 [bx, by, 0.0], speed * 0.3)
            v[2] = -speed * 0.6 if tool_pos[2] > 0.30 else 0.0
            return v

    @staticmethod
    def _vel_toward(current: list, target: list, speed: float) -> list:
        diff = np.array(target) - np.array(current)
        dist = np.linalg.norm(diff)
        if dist < 1e-4:
            return [0.0, 0.0, 0.0]
        return (diff / dist * min(speed, dist)).tolist()

    @staticmethod
    def phase_of(frac: float) -> int:
        if frac < 0.25:   return 1
        elif frac < 0.50: return 2
        elif frac < 0.75: return 3
        else:             return 4


# ─────────────────────────────────────────────
#  메인 시뮬레이터
# ─────────────────────────────────────────────
class YCBBalloonSimulator:

    def __init__(self, cfg: SceneConfig, gui: bool = False):
        self.cfg = cfg
        self.gui = gui
        self.client = None
        self.tree_id = self.balloon_id = self.tool_id = None
        self.string_constraint = None
        self.capture_constraint = None
        self._tmp_files: List[str] = []
        self._balloon_detached = False

    # ── 연결 ──
    def connect(self):
        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        if self.gui:
            p.resetDebugVisualizerCamera(3.0, 30, -25, [0, 0, 1.0], physicsClientId=self.client)

    # ── 환경 리셋 ──
    def reset(self, seed: Optional[int] = None) -> dict:
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self._balloon_detached = False
        self.capture_constraint = None

        cfg = self.cfg
        if seed is not None:
            rng = np.random.default_rng(seed)
            cfg.branch_angle_deg = float(rng.uniform(35, 50))
            cfg.branch_height    = float(rng.uniform(1.4, 1.7))

        # 바닥
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        # 나무
        tree_path = _tmp_urdf(tree_urdf(cfg), "sim_tree.urdf")
        self._tmp_files.append(tree_path)
        self.tree_id = p.loadURDF(tree_path, basePosition=cfg.tree_pos,
                                  useFixedBase=True, physicsClientId=self.client)

        # 풍선 위치 계산
        br = math.radians(cfg.branch_angle_deg)
        self._branch_tip = [
            cfg.tree_pos[0] + cfg.branch_length * math.cos(br),
            cfg.tree_pos[1],
            cfg.branch_height + cfg.branch_length * math.sin(br)
        ]
        # 풍선은 가지 끝에 줄이 걸려, 부력으로 떠오르다 가지 *아래*에 매달려 있다.
        # (위로 올라가던 풍선이 가지에 걸려 손이 닿지 않는 상태)
        balloon_pos = [
            self._branch_tip[0],
            self._branch_tip[1],
            self._branch_tip[2] - cfg.string_length
        ]

        # 풍선
        ball_path = _tmp_urdf(balloon_urdf(cfg), "sim_balloon.urdf")
        self._tmp_files.append(ball_path)
        self.balloon_id = p.loadURDF(ball_path, basePosition=balloon_pos,
                                     physicsClientId=self.client)
        p.changeDynamics(
            self.balloon_id, -1,
            linearDamping=cfg.balloon_linear_damping,
            angularDamping=cfg.balloon_angular_damping,
            lateralFriction=0.8,
            spinningFriction=0.02,
            rollingFriction=0.02,
            physicsClientId=self.client
        )

        # 줄 constraint (point2point)
        # 풍선이 가지 아래 string_length 만큼 떨어진 "매달린 점"에 걸려 있다.
        # 풍선 중심을 그 점에 직접 고정한다(레버 없음) → 가벼운 구체가
        # 긴 지렛대로 인해 수치적으로 튕겨 나가는 불안정을 방지한다.
        # 시각적 줄은 가지 끝 ~ 풍선 사이 디버그 라인으로 표현한다.
        hang_point_local = [
            cfg.branch_length * math.cos(br),
            0,
            cfg.branch_height + cfg.branch_length * math.sin(br) - cfg.string_length
        ]
        self.string_constraint = p.createConstraint(
            parentBodyUniqueId=self.tree_id,
            parentLinkIndex=-1,           # tree base link
            childBodyUniqueId=self.balloon_id,
            childLinkIndex=-1,            # balloon base link
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=hang_point_local,
            childFramePosition=[0, 0, 0],  # 풍선 중심을 매달린 점에 고정
            physicsClientId=self.client
        )
        # 약한 snag: 중력(≈0.016N)은 충분히 버티지만, 도구가 누르면(>maxForce)
        # 줄이 풀려 풍선이 밀려난다. 강체 핀처럼 큰 반력을 만들지 않아
        # 접촉력 폭주(→풍선 터짐)를 막는다. 분리는 변위로 감지한다.
        p.changeConstraint(self.string_constraint,
                           maxForce=self.cfg.string_max_force,
                           physicsClientId=self.client)

        # 도구 로드: spatula + gelatin_box 조합 (main_simulation_balloon.py 저장본 우선)
        real_tool_path = find_combined_tool_urdf()
        start_orn = p.getQuaternionFromEuler(cfg.tool_start_rpy)
        self.tool_id = None
        self._tool_is_fallback = True

        if real_tool_path is not None:
            try:
                self.tool_id = p.loadURDF(
                    real_tool_path,
                    basePosition=cfg.tool_start_pos,
                    baseOrientation=start_orn,
                    useFixedBase=False,
                    physicsClientId=self.client,
                )
                self._tool_is_fallback = False
                if not getattr(self, "_tool_load_logged", False):
                    print(f"[TOOL] 조합 도구 로드 성공: {real_tool_path}")
                    self._tool_load_logged = True
            except Exception as exc:
                print(f"[TOOL][WARN] 조합 URDF 로드 실패 → 인라인 도구로 대체: {exc}")
                print(f"[TOOL][WARN]   경로: {real_tool_path}")
                self.tool_id = None

        if self.tool_id is None:
            if real_tool_path is None and not getattr(self, "_tool_load_logged", False):
                print("[TOOL][WARN] 조합 도구 URDF 를 찾지 못함 → 인라인 임시 도구 사용.")
                print("[TOOL][WARN]   main_simulation_balloon.py 를 먼저 실행해 "
                      "outputs/combined_spatula_gelatin_box.urdf 를 생성하세요.")
                self._tool_load_logged = True
            fallback_path = inline_fallback_tool_urdf()
            self.tool_id = p.loadURDF(
                fallback_path,
                basePosition=cfg.tool_start_pos,
                baseOrientation=start_orn,
                useFixedBase=False,
                physicsClientId=self.client,
            )
            self._tool_is_fallback = True

        self._tool_path = real_tool_path if not self._tool_is_fallback else "(inline fallback)"
        self.gelatin_id = None

        # Phase 4 하강 제어용: 도구 base 원점 대비 *전체* 위쪽 끝(스패출러 헤드)
        # 높이. getAABB(body)는 base 링크만 보므로, 모든 링크의 AABB 를 합쳐
        # 최상단을 구한다. (인라인 도구 ~0.24m, 실제 조합 메시는 자동 대응)
        try:
            base_pos, _ = p.getBasePositionAndOrientation(self.tool_id, physicsClientId=self.client)
            top_z = -1e9
            n_links = p.getNumJoints(self.tool_id, physicsClientId=self.client)
            for li in range(-1, n_links):
                amin, amax = p.getAABB(self.tool_id, li, physicsClientId=self.client)
                top_z = max(top_z, amax[2])
            self._tool_head_offset = max(0.05, float(top_z - base_pos[2]))
        except Exception:
            self._tool_head_offset = 0.23

        # GUI 보조선
        if self.gui:
            p.addUserDebugLine(self._branch_tip, balloon_pos,
                               [0.9, 0.9, 0.9], 2, physicsClientId=self.client)
            p.addUserDebugText("🎈 BALLOON", [balloon_pos[0], balloon_pos[1], balloon_pos[2]+0.12],
                               [1,0.8,0], 1.2, physicsClientId=self.client)
            tool_label = "spatula + gelatin_box" if not self._tool_is_fallback else "inline fallback tool"
            p.addUserDebugText(f"TOOL: {tool_label}",
                               [cfg.tool_start_pos[0], cfg.tool_start_pos[1], cfg.tool_start_pos[2]+0.3],
                               [0.5,0.8,1], 1.0, physicsClientId=self.client)

        # 안정화 (20스텝): 부력 없이 중력만 → 풍선이 가지 아래에 안정적으로 매달림.
        for _ in range(20):
            self._apply_balloon_buoyancy()
            p.stepSimulation(physicsClientId=self.client)

        # 분리 판정 기준이 되는 "매달린 정지 위치"를 저장한다.
        bp0, _ = p.getBasePositionAndOrientation(self.balloon_id, physicsClientId=self.client)
        self._balloon_rest_pos = list(bp0)

        return self._obs()

    # ── 풍선 부력 ──
    def _apply_balloon_buoyancy(self):
        # 가지에 걸려 있는 동안은 줄(snag)에 고정돼 있으므로 부력을 적용하지
        # 않는다 (중력만 받아 가지 아래에 안정적으로 매달림). 도구가 줄을
        # 떼어낸 *뒤* 부터 부력이 작용해 풍선이 떠오르려 하고, 이를 도구가
        # Phase 4 에서 눌러 내려야 한다. → 도구의 역할이 물리적으로 필수가 된다.
        if not self._balloon_detached:
            return
        volume = 4.0 / 3.0 * math.pi * self.cfg.balloon_radius ** 3
        buoyant_force = self.cfg.air_density * volume * 9.81 * self.cfg.buoyancy_scale
        p.applyExternalForce(
            self.balloon_id, -1,
            forceObj=[0.0, 0.0, buoyant_force],
            posObj=[0.0, 0.0, 0.0],
            flags=p.LINK_FRAME,
            physicsClientId=self.client
        )

    # ── 관측 ──
    def _obs(self) -> dict:
        bp, _ = p.getBasePositionAndOrientation(self.balloon_id, physicsClientId=self.client)
        tp, _ = p.getBasePositionAndOrientation(self.tool_id,    physicsClientId=self.client)
        dist  = float(np.linalg.norm(np.array(bp) - np.array(tp)))
        return {"balloon_pos": list(bp), "tool_pos": list(tp), "dist": dist}

    # ── 접촉력 체크 ──
    def _max_contact_force(self) -> float:
        contacts = p.getContactPoints(
            bodyA=self.tool_id, bodyB=self.balloon_id,
            physicsClientId=self.client
        )
        if not contacts:
            return 0.0
        return max(abs(c[9]) for c in contacts)   # normalForce

    # ── 줄 분리 여부 ──
    def _check_detach(self, balloon_pos: list) -> bool:
        """도구가 풍선을 매달린 정지 위치에서 충분히 밀어냈으면 분리로 판단.

        3D 변위 기준이라 가지 아래 매달린 미세 진동에는 반응하지 않고,
        도구가 실제로 풍선을 움직였을 때만 분리된다(=도구가 분리를 일으킴).
        """
        if self._balloon_detached:
            return True
        rest = getattr(self, "_balloon_rest_pos", None)
        if rest is None:
            return False
        disp = math.sqrt(sum((balloon_pos[i] - rest[i]) ** 2 for i in range(3)))
        if disp > self.cfg.detach_disp:
            self._balloon_detached = True
            # snag 완전 해제
            try:
                if self.string_constraint is not None:
                    p.removeConstraint(self.string_constraint, physicsClientId=self.client)
            except Exception:
                pass
            self.string_constraint = None
            return True
        return False

    # ── 풍선 포획 (도구 헤드에 걸기) ──
    def _try_capture(self, tool_pos: list, balloon_pos: list, verbose: bool) -> bool:
        """스패출러 헤드가 풍선에 충분히 접근하면, 가지 snag 을 풀고 풍선을
        도구 헤드에 고정(포획)한다. 이후 도구를 내리면 풍선이 따라 내려온다."""
        if self.capture_constraint is not None:
            return True
        ho = getattr(self, "_tool_head_offset", 0.23)
        head_world = [tool_pos[0], tool_pos[1], tool_pos[2] + ho]
        d = math.dist(head_world, balloon_pos)
        if d > self.cfg.capture_radius:
            return False

        # 1) 가지 snag 해제
        try:
            if self.string_constraint is not None:
                p.removeConstraint(self.string_constraint, physicsClientId=self.client)
        except Exception:
            pass
        self.string_constraint = None

        # 2) 풍선을 헤드 *바로 위* 에 고정 (헤드에 얹혀 걸린 상태).
        #    풍선 중심을 헤드보다 (balloon_radius+여유) 위에 두어 메시 관통(→터짐)을 막는다.
        self.capture_constraint = p.createConstraint(
            parentBodyUniqueId=self.tool_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.balloon_id,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, ho + self.cfg.balloon_radius + 0.02],
            childFramePosition=[0, 0, 0],          # 풍선 중심
            physicsClientId=self.client,
        )
        p.changeConstraint(self.capture_constraint, maxForce=50.0,
                           physicsClientId=self.client)

        # 3) 포획 후에는 도구-풍선 충돌을 끈다 (이미 도구에 걸려 있으므로 접촉력
        #    스파이크로 풍선이 "터지는" 비현실적 현상을 방지).
        try:
            n_links = p.getNumJoints(self.tool_id, physicsClientId=self.client)
            for li in range(-1, n_links):
                p.setCollisionFilterPair(self.tool_id, self.balloon_id, li, -1, 0,
                                         physicsClientId=self.client)
        except Exception:
            pass

        self._balloon_detached = True   # 이후 부력 작용(떠오르려 함) → 무게로 눌러 내림
        if verbose:
            print(f"  ✅ 풍선 포획! 헤드-풍선 거리 {d:.3f}m → 도구에 걸림 (가지에서 분리)")
        return True

    # ── 단일 에피소드 실행 ──
    def run_episode(self, seed: Optional[int] = None, verbose: bool = False) -> EpisodeResult:
        self.reset(seed=seed)
        ctrl = ScriptController(self.cfg, head_offset=getattr(self, "_tool_head_offset", 0.23))

        min_dist      = float("inf")
        max_force     = 0.0
        burst         = False
        detached      = False
        phase_reached = 1
        phase4_start_step = int(self.cfg.control_steps * 0.75)

        for step in range(self.cfg.max_steps):
            obs   = self._obs()
            bpos  = obs["balloon_pos"]
            tpos  = obs["tool_pos"]
            frac  = min(step / self.cfg.control_steps, 0.999)
            phase = ScriptController.phase_of(frac)
            phase_reached = max(phase_reached, phase)

            # 거리 추적
            if obs["dist"] < min_dist:
                min_dist = obs["dist"]

            # 접촉력 체크 (풍선 터짐)
            cf = self._max_contact_force()
            if cf > max_force:
                max_force = cf
            if cf > self.cfg.burst_force_limit:
                burst = True
                if verbose:
                    print(f"  💥 Step {step}: 풍선 터짐! 충격력 {cf:.3f}N")
                break

            # 포획 시도 (헤드가 풍선에 닿으면 가지에서 떼어 도구에 건다)
            if not detached:
                detached = self._try_capture(tpos, bpos, verbose)

            # 속도 명령 적용
            vel = ctrl.get_velocity(step, tpos, bpos)
            p.resetBaseVelocity(self.tool_id, linearVelocity=vel,
                                angularVelocity=[0, 0, 0],
                                physicsClientId=self.client)

            self._apply_balloon_buoyancy()
            p.stepSimulation(physicsClientId=self.client)
            if self.gui:
                time.sleep(1.0 / 240.0)

            phase4_completed = step >= phase4_start_step + self.cfg.min_phase4_steps

            # 성공 조건: 분리 + 풍선이 너무 높이 날지 않음 + Phase 4 하강까지 완료
            # Phase 4 진입 직후가 아니라, 하강 구간을 실제로 수행한 뒤에만 성공 처리한다.
            if detached and bpos[2] < self.cfg.success_z_max and phase4_completed:
                if verbose:
                    print(f"  🎉 Step {step}: 태스크 성공! 풍선 z={bpos[2]:.3f} (Phase 4 하강 완료)")
                break

        final_obs = self._obs()
        bfpos = final_obs["balloon_pos"]

        # 성공 판정
        success = (
            detached
            and not burst
            and bfpos[2] < self.cfg.success_z_max
            and phase_reached >= 4
            and step >= phase4_start_step + self.cfg.min_phase4_steps
        )
        failure_reason = ""
        if burst:            failure_reason = "balloon_burst"
        elif not detached:   failure_reason = "not_detached"
        elif bfpos[2] >= self.cfg.success_z_max:
            failure_reason = "balloon_flew_too_high"
        elif phase_reached < 4 or step < phase4_start_step + self.cfg.min_phase4_steps:
            failure_reason = "phase4_not_completed"

        return EpisodeResult(
            success=success, burst=burst,
            steps=step+1,
            balloon_final_pos=bfpos,
            balloon_final_z=bfpos[2],
            min_dist_tool_balloon=min_dist,
            max_contact_force=max_force,
            detached=detached,
            failure_reason=failure_reason,
            phase_reached=phase_reached
        )

    def disconnect(self):
        try:
            p.disconnect(self.client)
        except Exception:
            pass
        for f in self._tmp_files:
            try: os.remove(f)
            except: pass


# ─────────────────────────────────────────────
#  통계 출력
# ─────────────────────────────────────────────
def print_summary(results: List[EpisodeResult]):
    n = len(results)
    if n == 0:
        return

    successes = [r for r in results if r.success]
    detached  = [r for r in results if r.detached]
    bursts    = [r for r in results if r.burst]

    print(f"\n{'═'*55}")
    print(f"  📊 평가 결과 ({n} 에피소드)")
    print(f"{'═'*55}")

    pct = len(successes) / n * 100
    bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
    grade = "🟢 우수" if pct >= 70 else "🟡 보통" if pct >= 40 else "🔴 미흡"
    print(f"\n  태스크 성공률 : {pct:5.1f}%  [{bar}]  {grade}")
    print(f"  가지 분리율   : {len(detached)/n*100:5.1f}%  ({len(detached)}/{n})")
    print(f"  풍선 터짐률   : {len(bursts)/n*100:5.1f}%  ({len(bursts)}/{n})")

    if successes:
        avg_steps = np.mean([r.steps for r in successes])
        avg_z     = np.mean([r.balloon_final_z for r in successes])
        avg_force = np.mean([r.max_contact_force for r in results])
        print(f"\n  ─── 성공 에피소드 ───")
        print(f"  평균 소요 스텝          : {avg_steps:.0f}")
        print(f"  평균 최종 풍선 높이(z)  : {avg_z:.3f}m")
        print(f"  평균 최대 접촉력        : {avg_force:.4f}N")

    # Phase 도달 분포
    phase_counts = {1:0, 2:0, 3:0, 4:0}
    for r in results:
        phase_counts[r.phase_reached] = phase_counts.get(r.phase_reached, 0) + 1
    print(f"\n  ─── Phase 도달 분포 ───")
    labels = {1:"접근", 2:"접촉", 3:"밀어냄", 4:"하강"}
    for ph, cnt in phase_counts.items():
        bar_p = "█" * cnt + "░" * (n - cnt)
        print(f"  Phase {ph} {labels[ph]:<5}: {cnt:3d}회  [{bar_p}]")

    # 실패 원인
    fails = [r.failure_reason for r in results if not r.success and r.failure_reason]
    if fails:
        from collections import Counter
        print(f"\n  ─── 실패 원인 ───")
        for reason, cnt in Counter(fails).items():
            print(f"  {reason:<30}: {cnt}회")

    print(f"{'═'*55}\n")


# ─────────────────────────────────────────────
#  실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YCB 뒤집개+푸딩박스 풍선 태스크")
    ap.add_argument("--gui",     action="store_true", help="PyBullet GUI 표시")
    ap.add_argument("--trials",  type=int, default=1, help="에피소드 수 (기본 1)")
    ap.add_argument("--seed",    type=int, default=None, help="랜덤 시드")
    ap.add_argument("--verbose", action="store_true", help="스텝별 출력")
    args = ap.parse_args()

    cfg = SceneConfig()
    sim = YCBBalloonSimulator(cfg, gui=args.gui)
    sim.connect()

    print(f"\n🎈 YCB 도구 시뮬레이션 시작")
    print(f"   도구: 뒤집개(spatula) + 젤라틴박스(gelatin_box)")
    print(f"   에피소드: {args.trials}회")
    print(f"   도구 전체 길이: ~28cm (YCB spatula 기준)")

    results = []
    for i in range(args.trials):
        seed = (args.seed + i) if args.seed is not None else i
        if args.trials > 1:
            print(f"\r  에피소드 {i+1:3d}/{args.trials}", end="", flush=True)
        r = sim.run_episode(seed=seed, verbose=args.verbose or args.trials == 1)
        results.append(r)
        if args.trials == 1:
            print(f"\n  결과: {'✅ 성공' if r.success else '❌ 실패'}")
            print(f"  분리: {'Yes' if r.detached else 'No'} | "
                  f"터짐: {'Yes' if r.burst else 'No'} | "
                  f"최종z: {r.balloon_final_z:.3f}m | "
                  f"Phase: {r.phase_reached}")

    if args.trials > 1:
        print()
    print_summary(results)
    sim.disconnect()