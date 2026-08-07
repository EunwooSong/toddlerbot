"""Fixed-period phase-clock reference for Unitree G1 (unitree_rl_gym recipe).

No ZMP, no lookup tables: the reference is just the default pose plus a fixed
gait clock (period = cycle_time, stance duty 0.55, legs pi apart) that drives
the stance mask, the phase observation, and the contact-phase reward.
"""
from toddlerbot.reference.walk_zmp_ref_g1 import WalkZMPReferenceG1
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np

STANCE_DUTY = 0.55  # unitree_rl_gym g1_env.py: is_stance = leg_phase < 0.55


class PhaseClockReferenceG1(WalkZMPReferenceG1):
    def _setup_zmp(self):
        # no ZMP machinery; only what the base env still touches
        self.single_support_ratio = 1.0 - STANCE_DUTY
        self.double_support_ratio = STANCE_DUTY

    def get_phase_signal(self, time_curr) -> ArrayType:
        phase = np.mod(time_curr, self.cycle_time) / self.cycle_time
        return np.array(
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            dtype=np.float32,
        )

    def get_state_ref(self, state_curr, time_curr, command) -> ArrayType:
        path_state = self.integrate_path_state(state_curr, command)
        phase = np.mod(time_curr, self.cycle_time) / self.cycle_time
        stance_mask = np.array(
            [phase < STANCE_DUTY, np.mod(phase + 0.5, 1.0) < STANCE_DUTY],
            dtype=np.float32,
        )
        return np.concatenate(
            (path_state, self.default_motor_pos, self.default_joint_pos, stance_mask)
        )
