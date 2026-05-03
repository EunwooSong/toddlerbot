from typing import Dict, Tuple  # Tuple은 freq range 타입 힌트에 사용

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


# 모터별 self-collision-free 안전 진폭 (rad)
# (시프트된) 진동 중심 기준 ± 이 값까지만 움직임
# left_/right_ prefix를 떼어낸 suffix로 매칭
SAFE_AMPLITUDE_BY_SUFFIX: Dict[str, float] = {
    # neck (충돌 위험 적음)
    "neck_yaw_drive": 0.6,
    "neck_pitch_act": 0.5,
    # waist: 진폭 3배 ↑ (상체 전체 inertia 활용)
    "waist_act_1": 0.45,
    "waist_act_2": 0.45,
    # legs (큰 진폭으로 관성 부하 증가; 좌우 동기화로 충돌 방지됨)
    "hip_pitch": 0.4,         # 다리 전체 swing → hip에 큰 관성 토크
    "hip_roll": 0.25,         # one-sided이라 default → -25° 외전 swing
    "hip_yaw_drive": 0.26,    # one-sided +1, ≈ 15° 외측 회전
    "knee_act": 0.6,          # 하부 다리 swing, knee가 시프트되어 ±34° 가능
    # ankle: 말단이라 충돌 위험 적음 → 진폭/주파수 ↑↑ (발열 ↑↑)
    "ank_roll": 0.7,    # ±40°, 한계 ±90° 내 여유
    "ank_pitch": 0.65,  # ±37°, 한계의 70% 이내
    # arms
    "sho_pitch": 0.3,
    "sho_roll": 0.25,
    "sho_yaw_drive": 0.3,
    "elbow_roll": 0.5,
    "elbow_yaw_drive": 0.4,
    "wrist_pitch_drive": 0.5,
    "wrist_roll": 0.6,
}

# 모터별 주파수 대역 (Hz). 미정의 모터는 DEFAULT_FREQ_RANGE 사용
# Hip 주파수: 관성 토크 확보 위해 약간 증가, peak는 진폭 축소로 균형
FREQ_RANGE_BY_SUFFIX: Dict[str, Tuple[float, float]] = {
    "hip_pitch": (0.3, 1.5),
    "hip_roll": (0.3, 1.5),
    "hip_yaw_drive": (0.3, 1.8),
    "knee_act": (0.5, 2.5),  # 하부 다리 inertia 활용 + 고주파 가속도
    # ankle: 말단 모터, 관성 작음 → 고주파 가능 → 발열 ↑
    "ank_roll": (0.5, 2.8),
    "ank_pitch": (0.5, 2.8),
}
DEFAULT_FREQ_RANGE: Tuple[float, float] = (0.3, 2.0)

# 모터별 진동 중심을 default에서 "여유가 더 큰 한계 방향"으로 시프트하는 양 (rad)
# 비대칭 한계로 인해 default 기준 대칭 진동이 어려운 관절 보정용
# 좌우 모터는 자동으로 반대 부호로 적용됨 (각자의 더 먼 한계 쪽으로)
CENTER_OFFSET_TOWARD_LONGER_SIDE: Dict[str, float] = {
    # knee: default(-30.6° / +30.6°)에서 굽힘 방향으로 시프트
    # 시프트 후 대칭 진동 ±0.4 rad 확보
    "knee_act": 0.5,
}

# One-sided oscillation: default에서 지정된 방향으로만 진동
# 양다리 hip_roll이 안쪽으로 모이는 자기충돌 방지
# 부호: +1 = default 기준 +방향으로만, -1 = default 기준 -방향으로만
# (URDF 부호 규칙에 따라 반대로 동작하면 부호 flip 필요)
ONE_SIDED_DIRECTION_BY_NAME: Dict[str, int] = {
    "left_hip_roll": -1,        # 왼쪽 다리: - 방향이 바깥쪽
    "right_hip_roll": -1,       # 오른쪽 다리: - 방향이 바깥쪽
    "left_hip_yaw_drive": -1,   # 양쪽 hip_yaw 모두 + 방향, ≈ 15° 외측 회전
    "right_hip_yaw_drive": +1,
}


def _strip_lr(motor_name: str) -> str:
    """left_/right_ prefix 제거하여 suffix만 반환."""
    for prefix in ("left_", "right_"):
        if motor_name.startswith(prefix):
            return motor_name[len(prefix):]
    return motor_name


def _get_safe_amplitude(motor_name: str) -> float:
    """suffix로 안전 진폭 조회. 미정의 시 0.2 rad."""
    return SAFE_AMPLITUDE_BY_SUFFIX.get(_strip_lr(motor_name), 0.2)


def _get_center_offset_magnitude(motor_name: str) -> float:
    """suffix로 중심 시프트량 조회. 미정의 시 0.0."""
    return CENTER_OFFSET_TOWARD_LONGER_SIDE.get(_strip_lr(motor_name), 0.0)


def _get_freq_range(motor_name: str) -> Tuple[float, float]:
    """suffix로 주파수 대역 조회. 미정의 시 DEFAULT_FREQ_RANGE."""
    return FREQ_RANGE_BY_SUFFIX.get(_strip_lr(motor_name), DEFAULT_FREQ_RANGE)


class RandomPolicy(BasePolicy, policy_name="random"):
    """LPTN 학습용 랜덤 모션 정책.

    Self-collision 방지:
        - 모터별 차등 진폭 (옵션 B): 부위별 충돌 위험에 따라 진폭 차등 설정
        - 좌우 동기 위상 (옵션 C): left/right 쌍은 동일 위상으로 강제하여
          양팔/양다리가 동시에 같은 방향으로 움직이도록 함

    Peak 전력 최소화:
        - default_motor_pos 기준 offset 방식
        - 보수적 진폭, 저주파 대역, 작은 per-step delta
        - Ramp-up 3초로 시작 transient 차단
    """

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        seed: int = 42,
    ):
        super().__init__(name, robot, init_motor_pos)

        self.num_motors = len(robot.motor_ordering)
        self.rng = np.random.default_rng(seed)

        lo = self.motor_limits[:, 0]
        hi = self.motor_limits[:, 1]

        # default 기준 한계 거리
        dist_to_hi = hi - self.default_motor_pos
        dist_to_lo = self.default_motor_pos - lo

        # 진동 중심 시프트: 비대칭 한계 보정 (knee 등)
        # 시프트 부호는 더 먼 한계 방향
        shifted_center = self.default_motor_pos.copy().astype(np.float32)
        for i, name in enumerate(robot.motor_ordering):
            offset_mag = _get_center_offset_magnitude(name)
            if offset_mag > 0.0:
                # dist_to_hi가 더 크면 +방향으로, 그렇지 않으면 -방향으로 시프트
                if dist_to_hi[i] > dist_to_lo[i]:
                    shifted_center[i] = self.default_motor_pos[i] + offset_mag
                else:
                    shifted_center[i] = self.default_motor_pos[i] - offset_mag

        # 시프트된 중심 기준 한계 거리 재계산
        dist_to_hi_shift = hi - shifted_center
        dist_to_lo_shift = shifted_center - lo
        limit_safe_half = np.minimum(dist_to_hi_shift, dist_to_lo_shift) * 0.7

        # 모터별 self-collision-free 진폭
        collision_safe_half = np.array(
            [_get_safe_amplitude(name) for name in robot.motor_ordering],
            dtype=np.float32,
        )

        # 두 제약 중 작은 쪽 채택
        self.range_half = np.minimum(
            limit_safe_half, collision_safe_half
        ).astype(np.float32)

        # One-sided oscillation 처리: default 기준 한쪽 방향으로만 진동
        # 진동 영역 = [default, default + sign * 2*A] (sign=+1) 또는
        #            [default - 2*A, default] (sign=-1)
        # 중심을 default + sign*A 로 시프트하고 진폭은 A로 유지
        for i, name in enumerate(robot.motor_ordering):
            sign = ONE_SIDED_DIRECTION_BY_NAME.get(name, 0)
            if sign == 0:
                continue
            # default 기준 바깥쪽 안전 거리
            outward_dist = dist_to_hi[i] if sign > 0 else dist_to_lo[i]
            # 한쪽으로 2*A를 사용하므로 outward_dist*0.7/2 가 한계
            limit_amp = float(outward_dist) * 0.7 / 2.0
            amp = min(limit_amp, _get_safe_amplitude(name))
            self.range_half[i] = np.float32(amp)
            shifted_center[i] = self.default_motor_pos[i] + sign * np.float32(amp)

        self.range_center = shifted_center
        self.range_lo = (self.range_center - self.range_half).astype(np.float32)
        self.range_hi = (self.range_center + self.range_half).astype(np.float32)

        # 다중 사인파 합성: 각 모터별 K개 컴포넌트
        self.K = 5
        # 모터별 주파수 대역 적용 (hip은 저주파, 나머지는 default 0.3~2.0 Hz)
        freqs_per_motor = np.zeros((self.num_motors, self.K), dtype=np.float32)
        for i, name in enumerate(robot.motor_ordering):
            f_lo, f_hi = _get_freq_range(name)
            freqs_per_motor[i] = self.rng.uniform(f_lo, f_hi, size=self.K)
        self.freqs = freqs_per_motor

        # 좌우 동기화: left/right 쌍은 동일 phase + 동일 freq
        # → 좌우가 동시에 같은 방향으로 움직여 양팔/양다리 충돌 방지
        phases = self.rng.uniform(
            0, 2 * np.pi, size=(self.num_motors, self.K)
        ).astype(np.float32)
        self._sync_left_right(robot, phases, self.freqs)
        self.phases = phases

        weights = self.rng.uniform(0.5, 1.0, size=(self.num_motors, self.K)).astype(
            np.float32
        )
        weights /= weights.sum(axis=1, keepdims=True)
        self.amplitudes = (weights * self.range_half[:, None]).astype(np.float32)

        # 스텝당 최대 변위 0.05 rad/step → 50Hz 시 2.5 rad/s
        self.motor_max_delta = 0.05

        self.last_motor_target = self.default_motor_pos.copy().astype(np.float32)
        self.is_prepared = False

    def _sync_left_right(
        self,
        robot: Robot,
        phases: npt.NDArray[np.float32],
        freqs: npt.NDArray[np.float32],
    ):
        """left_X 모터의 phase/freq를 right_X 값으로 덮어써서 동기화."""
        name_to_idx = {n: i for i, n in enumerate(robot.motor_ordering)}
        for name, idx in name_to_idx.items():
            if name.startswith("left_"):
                right_name = "right_" + name[len("left_"):]
                if right_name in name_to_idx:
                    r_idx = name_to_idx[right_name]
                    phases[idx] = phases[r_idx]
                    freqs[idx] = freqs[r_idx]

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 7.0 if is_real else 2.0
            # 시프트된 중심까지 prep에서 부드럽게 이동 (knee 시프트 반영)
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                self.init_motor_pos,
                self.range_center,
                self.prep_duration,
                end_time=2.0 if is_real else 0.0,
            )
            self.last_motor_target = self.range_center.copy().astype(np.float32)

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            self.last_motor_target = action.astype(np.float32)
            return {}, action

        t = obs.time - self.prep_duration

        # Ramp-up 3초: 진폭을 0→1로 점진 증가
        ramp = min(1.0, t / 3.0)

        components = self.amplitudes * np.sin(
            2.0 * np.pi * self.freqs * t + self.phases
        )
        offset = ramp * components.sum(axis=1)

        target = self.range_center + offset

        # 1) safe envelope clip
        target = np.clip(target, self.range_lo, self.range_hi)

        # 2) per-step delta clip
        delta = target - self.last_motor_target
        delta = np.clip(delta, -self.motor_max_delta, self.motor_max_delta)
        target = self.last_motor_target + delta

        # 3) 한계 재clip
        target = np.clip(target, self.range_lo, self.range_hi).astype(np.float32)

        self.last_motor_target = target.copy()
        return {}, target
