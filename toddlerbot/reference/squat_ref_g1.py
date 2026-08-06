"""Squat reference for Unitree G1.

Periodic double-support squat: CoM height follows a smooth cosine dip of
`squat_depth` over `cycle_time` (down and back up within one cycle), legs from
the G1-signed com_ik, upper body frozen at default (arms forward-90).

Depth ceiling: ankle dorsiflexion limit (-0.87 rad) binds at ~0.18 m for the
flat-foot, upright-torso squat; default 0.15 m keeps ~0.05 rad margin.
Legs are clipped to joint limits as a safety net regardless.
"""
from toddlerbot.reference.walk_zmp_ref_g1 import WalkZMPReferenceG1
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class SquatG1Reference(WalkZMPReferenceG1):
    def __init__(
        self,
        robot: Robot,
        dt: float,
        cycle_time: float = 4.0,
        squat_depth: float = 0.15,
        **kwargs,
    ):
        self.squat_depth = squat_depth
        super().__init__(robot, dt, cycle_time, waist_roll_max=0.0)
        self.leg_joint_limits = np.array(
            [
                self.robot.joint_limits[n]
                for n in self.robot.joint_ordering
                if self.robot.joint_groups[n] == "leg"
            ],
            dtype=np.float32,
        )

    def _setup_zmp(self):
        pass  # no footstep planning; the squat never uses the walk lookup

    def get_vel(self, command: ArrayType):
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def get_phase_signal(self, time_curr):
        phase = 2 * np.pi * (time_curr / self.cycle_time)
        return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

    def get_state_ref(
        self, state_curr: ArrayType, time_curr, command: ArrayType
    ) -> ArrayType:
        path_state = self.integrate_path_state(state_curr, command)

        joint_pos = self.default_joint_pos.copy()
        motor_pos = self.default_motor_pos.copy()

        phase = (time_curr / self.cycle_time) % 1.0
        # command[2] in [0,1] scales squat depth (per-episode randomized in
        # SquatG1Env — stateless depth curriculum; eval/render set it to 1.0)
        depth = (
            self.squat_depth
            * command[2]
            * 0.5
            * (1.0 - np.cos(2 * np.pi * phase))
        )
        leg_joint_pos = self.com_ik(-depth)
        leg_joint_pos = np.clip(
            leg_joint_pos, self.leg_joint_limits[:, 0], self.leg_joint_limits[:, 1]
        )

        joint_pos = inplace_update(joint_pos, self.leg_joint_indices, leg_joint_pos)
        motor_pos = inplace_update(motor_pos, self.leg_motor_indices, leg_joint_pos)

        stance_mask = np.ones(2, dtype=np.float32)
        return np.concatenate((path_state, motor_pos, joint_pos, stance_mask))
