from typing import Any, Optional

import jax
import jax.numpy as jnp
from brax import base

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.locomotion.walk_env import WalkEnv
from toddlerbot.reference.squat_ref_g1 import SquatG1Reference
from toddlerbot.sim.robot import Robot


class SquatG1Env(WalkEnv, env_name="squat_g1"):
    """Periodic squat on Unitree G1: both feet planted, CoM dips 0.15 m on a
    4 s cosine cycle, upper body frozen (L-shape arms), legs-only action.

    Inherits WalkEnv so the shared reward catalog (torso_roll/pitch,
    feet_slip, feet_distance, ...) stays available; walking-only rewards are
    zeroed in squat_g1.gin. Reward design rationale: see squat reward research
    (2026-08-06) — joint-space tracking (leg_motor_pos) as the main signal
    plus an explicit phase-referenced root-height term (torso_height), the
    literature-standard split (e.g. GMT, Multi-Gait SAMP ~10:1 joint:height).
    """

    SQUAT_DEPTH = 0.15  # m; ankle-dorsiflexion-bounded (see squat_ref_g1)

    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        ref_motion_type: str = "squat",
        fixed_base: bool = False,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        motion_ref = SquatG1Reference(
            robot,
            cfg.sim.timestep * cfg.action.n_frames,
            cycle_time=cfg.action.cycle_time,
            squat_depth=self.SQUAT_DEPTH,
        )

        self.cycle_time = jnp.array(cfg.action.cycle_time)
        self.torso_roll_range = cfg.rewards.torso_roll_range
        self.torso_pitch_range = cfg.rewards.torso_pitch_range
        self.max_feet_air_time = self.cycle_time / 2.0
        self.min_feet_y_dist = cfg.rewards.min_feet_y_dist
        self.max_feet_y_dist = cfg.rewards.max_feet_y_dist

        MJXEnv.__init__(
            self,
            name,
            robot,
            cfg,
            motion_ref,
            fixed_base=fixed_base,
            add_noise=add_noise,
            add_domain_rand=add_domain_rand,
            **kwargs,
        )

    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        return jnp.zeros(8)

    def _reward_torso_height(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Track the phase-referenced pelvis height: z_ref(phase) =
        z_top - depth * (1 - cos(2*pi*phase)) / 2, using the cos component of
        the phase signal.

        Kernel width matters: with k=800 (sigma 3.5 cm) the reward is
        exp(-18)~=0 at the stand-vs-bottom error (0.15 m) — zero gradient, so
        the policy converged to standing still (verified, run 104723). k=50
        (sigma ~14 cm) keeps a usable gradient over the whole squat range;
        literature kernels are even wider (GMT uses k~=1)."""
        cos_ph = info["phase_signal"][1]
        z_ref = self.motion_ref.torso_pos_init[2] - self.SQUAT_DEPTH * 0.5 * (
            1.0 - cos_ph
        )
        z = pipeline_state.x.pos[0][2]
        return jnp.exp(-50.0 * (z - z_ref) ** 2)
