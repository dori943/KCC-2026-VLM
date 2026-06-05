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
    balloon_mass:      float = 0.002
    string_length:     float = 0.25
    string_max_force:  float = 0.08   # 줄이 버티는 최대 장력 (N) → 초과시 분리

    # 도구 시작 위치 (로봇 손 위치 가정)
    tool_start_pos:    List[float] = field(default_factory=lambda: [0.55, 0.0, 0.85])
    tool_start_rpy:    List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # 평가
    max_steps:         int  = 1800    # 7.5초 @ 240Hz
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


def combined_tool_urdf() -> str:
    """뒤집개 + 푸딩박스 조합 도구 (assets 폴더 없을 때 fallback)"""
    asset = os.path.join(ASSETS_DIR, "spatula_pudding_tool.urdf")
    if os.path.exists(asset):
        return asset
    # 인라인 fallback
    content = open(asset).read() if os.path.exists(asset) else None
    if content:
        return asset
    # assets 없으면 간단 버전으로
    return _tmp_urdf("""<?xml version="1.0"?>
<robot name="spatula_pudding_tool">
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
    Phase 1 (0~25%):  도구를 가지 바로 아래로 이동 (풍선 줄 아래 진입 준비)
    Phase 2 (25~50%): 도구를 앞으로 밀어 헤드를 풍선 줄에 접촉
    Phase 3 (50~75%): 헤드를 위+앞으로 → 줄을 가지 끝 너머로 밀어냄
    Phase 4 (75~100%): 도구를 천천히 아래로 → 풍선이 따라 내려옴
    """

    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
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
        # 풍선 걸린 위치 (줄 위 가지 끝)
        self.balloon_hang_pos = [
            self.branch_tip[0],
            self.branch_tip[1],
            self.branch_tip[2] + cfg.string_length
        ]

    def get_velocity(self, step: int, tool_pos: list, balloon_pos: list) -> list:
        cfg   = self.cfg
        total = cfg.max_steps
        frac  = step / total

        tip   = self.branch_tip
        # PyBullet expects linear velocity in m/s.
        # We want a gentle but effective tool motion around 240Hz.
        speed = 0.002 * 240.0  # m/s

        if frac < 0.25:
            # Phase 1: 가지 바로 밑, 줄 아래로 접근
            # 목표: 가지 끝 바로 아래, 헤드가 줄 높이보다 낮게
            target = [tip[0] - 0.05, tip[1], tip[2] + cfg.string_length * 0.3]
            return self._vel_toward(tool_pos, target, speed)

        elif frac < 0.50:
            # Phase 2: 가지 방향(x+)으로 전진, 헤드가 줄에 접촉
            target = [tip[0] + 0.02, tip[1], tip[2] + cfg.string_length * 0.5]
            return self._vel_toward(tool_pos, target, speed * 0.7)

        elif frac < 0.75:
            # Phase 3: 위+앞으로 → 줄을 가지 끝 너머로 밀어냄
            target = [tip[0] + 0.12, tip[1], tip[2] + cfg.string_length + 0.08]
            return self._vel_toward(tool_pos, target, speed * 0.8)

        else:
            # Phase 4: 아래로 천천히 → 풍선도 따라 내려옴
            return [0.0, 0.0, -speed * 0.5]

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
        balloon_pos = [
            self._branch_tip[0],
            self._branch_tip[1],
            self._branch_tip[2] + cfg.string_length
        ]

        # 풍선
        ball_path = _tmp_urdf(balloon_urdf(cfg), "sim_balloon.urdf")
        self._tmp_files.append(ball_path)
        self.balloon_id = p.loadURDF(ball_path, basePosition=balloon_pos,
                                     physicsClientId=self.client)

        # 줄 constraint (point2point)
        # 가지 끝 위치 (tree base link의 로컬 좌표)
        branch_tip_local = [
            cfg.branch_length * math.cos(br),
            0,
            cfg.branch_height + cfg.branch_length * math.sin(br)
        ]
        self.string_constraint = p.createConstraint(
            parentBodyUniqueId=self.tree_id,
            parentLinkIndex=-1,           # tree base link
            childBodyUniqueId=self.balloon_id,
            childLinkIndex=-1,            # balloon base link
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=branch_tip_local,
            childFramePosition=[0, 0, -cfg.string_length],  # balloon center offset for string
            physicsClientId=self.client
        )
        # Use a strong constraint and break it manually when the balloon is pulled hard enough.
        p.changeConstraint(self.string_constraint,
                           maxForce=1000.0,
                           physicsClientId=self.client)

        # 도구 로드: spatula + pudding box 조합
        tool_urdf_path = combined_tool_urdf()
        self.tool_id = p.loadURDF(
            tool_urdf_path,
            basePosition=cfg.tool_start_pos,
            baseOrientation=p.getQuaternionFromEuler(cfg.tool_start_rpy),
            useFixedBase=False,
            physicsClientId=self.client
        )
        self.gelatin_id = None

        # GUI 보조선
        if self.gui:
            p.addUserDebugLine(self._branch_tip, balloon_pos,
                               [0.9, 0.9, 0.9], 2, physicsClientId=self.client)
            p.addUserDebugText("🎈 BALLOON", [balloon_pos[0], balloon_pos[1], balloon_pos[2]+0.12],
                               [1,0.8,0], 1.2, physicsClientId=self.client)
            p.addUserDebugText("TOOL: spatula + pudding_box",
                               [cfg.tool_start_pos[0], cfg.tool_start_pos[1], cfg.tool_start_pos[2]+0.3],
                               [0.5,0.8,1], 1.0, physicsClientId=self.client)

        # 안정화 (20스텝)
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client)

        return self._obs()

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
        """가지 끝 xy 거리로 분리 판단 (constraint가 끊기거나 밀려났을 때)"""
        if self._balloon_detached:
            return True
        tip = self._branch_tip
        xy_dist = math.sqrt((balloon_pos[0]-tip[0])**2 + (balloon_pos[1]-tip[1])**2)
        if xy_dist > self.cfg.balloon_radius * 2.5:
            self._balloon_detached = True
            # constraint 제거 (분리됐으니 해제)
            try:
                p.removeConstraint(self.string_constraint, physicsClientId=self.client)
            except Exception:
                pass
            return True
        return False

    # ── 단일 에피소드 실행 ──
    def run_episode(self, seed: Optional[int] = None, verbose: bool = False) -> EpisodeResult:
        self.reset(seed=seed)
        ctrl = ScriptController(self.cfg)

        min_dist      = float("inf")
        max_force     = 0.0
        burst         = False
        detached      = False
        phase_reached = 1

        for step in range(self.cfg.max_steps):
            obs   = self._obs()
            bpos  = obs["balloon_pos"]
            tpos  = obs["tool_pos"]
            frac  = step / self.cfg.max_steps
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

            # 분리 확인
            if not detached:
                if cf > self.cfg.string_max_force * 5:
                    detached = True
                    if verbose:
                        print(f"  ✅ Step {step}: 줄이 끊어짐! 접촉력 {cf:.3f}N")
                    try:
                        if self.string_constraint is not None:
                            p.removeConstraint(self.string_constraint, physicsClientId=self.client)
                    except Exception:
                        pass
                    self.string_constraint = None
                else:
                    detached = self._check_detach(bpos)
                if detached and verbose:
                    print(f"  ✅ Step {step}: 가지에서 분리! 풍선 위치 {[f'{v:.3f}' for v in bpos]}")

            # 속도 명령 적용
            vel = ctrl.get_velocity(step, tpos, bpos)
            p.resetBaseVelocity(self.tool_id, linearVelocity=vel,
                                angularVelocity=[0, 0, 0],
                                physicsClientId=self.client)

            p.stepSimulation(physicsClientId=self.client)
            if self.gui:
                time.sleep(1.0 / 240.0)

            # 성공 조건: 분리 + 풍선이 너무 높이 날지 않음 + 아래로 내려옴
            if detached and bpos[2] < self.cfg.success_z_max and step > self.cfg.max_steps * 0.6:
                if verbose:
                    print(f"  🎉 Step {step}: 태스크 성공! 풍선 z={bpos[2]:.3f}")
                break

        final_obs = self._obs()
        bfpos = final_obs["balloon_pos"]

        # 성공 판정
        success = (
            detached
            and not burst
            and bfpos[2] < self.cfg.success_z_max
        )
        failure_reason = ""
        if burst:            failure_reason = "balloon_burst"
        elif not detached:   failure_reason = "not_detached"
        elif bfpos[2] >= self.cfg.success_z_max:
            failure_reason = "balloon_flew_too_high"

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
    print(f"   도구: 뒤집개(spatula) + 푸딩박스(pudding_box)")
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