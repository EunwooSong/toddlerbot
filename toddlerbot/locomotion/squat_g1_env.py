"""Unitree G1 periodic squat: both feet planted, CoM dips on a cosine cycle,
upper body frozen (L-shape arms), legs-only action.

Recipe design (2026-08-06 reward research + 7 diagnosed runs): joint-space
reference tracking as the main signal, an explicit phase-referenced
root-height term, and residual-on-reference actions. Details on each choice
live on the method that implements it.
"""
from typing import Any, Optional

import jax
import jax.numpy as jnp
from brax import base

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.locomotion.walk_env import WalkEnv
from toddlerbot.reference.squat_ref_g1 import SquatG1Reference
from toddlerbot.sim.robot import Robot


class SquatG1Recipe:
    """Squat recipe (depth curriculum, residual actions, height reward).

    Engine-agnostic mixin so the thermal env (`_T_Squat_G1`) and the plain env
    (`squat_g1`) share one definition — the recipe must not drift between the
    thermal ablation and its baseline. Mirrors WalkG1Recipe.
    """

    SQUAT_DEPTH = 0.15  # m; ankle-dorsiflexion-bounded (see squat_ref_g1)

    def _squat_setup(self, robot, cfg):
        """Recipe state that must be built after the base env __init__."""
        self.cycle_time = jnp.array(cfg.action.cycle_time)
        self.torso_roll_range = cfg.rewards.torso_roll_range
        self.torso_pitch_range = cfg.rewards.torso_pitch_range
        self.max_feet_air_time = self.cycle_time / 2.0
        self.min_feet_y_dist = cfg.rewards.min_feet_y_dist
        self.max_feet_y_dist = cfg.rewards.max_feet_y_dist

    def make_motion_ref(self, robot, cfg):
        return SquatG1Reference(
            robot,
            cfg.sim.timestep * cfg.action.n_frames,
            cycle_time=cfg.action.cycle_time,
            squat_depth=self.SQUAT_DEPTH,
        )

    def step(self, state, action: jax.Array):
        """Residual-on-reference action: rebase `default_action` to the leg
        slice of the CURRENT reference each step, so the policy outputs a
        +-action_scale correction around the squat trajectory instead of
        around the static default pose.

        Why: tanh-bounded actions with scale 0.25 cannot reach the squat
        (runs 1-4 capped at ~3 cm dip), and raising the scale to 1.3 made
        early exploration violent enough to NaN the physics (run 5). The
        reference carries the large excursion; the policy only balances.
        get_state_ref is pure (path state in, path state out), so the extra
        call here does not double-integrate."""
        time_curr = state.info["step"] * self.dt
        state_ref = self.motion_ref.get_state_ref(
            state.info["state_ref"], time_curr, state.info["command"]
        )
        leg_ref = jnp.asarray(state_ref)[self.ref_start_idx + self.leg_ref_indices]
        state.info["default_action"] = leg_ref
        return super().step(state, action)

    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        """Zero locomotion commands; command[2] = squat-depth fraction,
        randomized per episode (U[0.2, 1.0]) as a stateless curriculum:
        shallow squats are easy balance problems that bootstrap deep ones
        (reward diagnosis showed open-loop full-depth tracking falls in 1 s).
        The fraction is observable (command_obs_indices includes 2)."""
        depth_frac = jax.random.uniform(rng, (), minval=0.2, maxval=1.0)
        return jnp.zeros(8).at[2].set(depth_frac)

    def _reward_torso_height(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Track the phase- and depth-referenced pelvis height: z_ref =
        z_top - depth_frac * DEPTH * (1 - cos(2*pi*phase)) / 2.

        Kernel width matters: with k=800 (sigma 3.5 cm) the reward is
        exp(-18)~=0 at the stand-vs-bottom error (0.15 m) — zero gradient, so
        the policy converged to standing still (verified, run 104723). k=50
        (sigma ~14 cm) keeps a usable gradient over the whole squat range;
        literature kernels are even wider (GMT uses k~=1)."""
        cos_ph = info["phase_signal"][1]
        depth = self.SQUAT_DEPTH * info["command"][2]
        z_ref = self.motion_ref.torso_pos_init[2] - depth * 0.5 * (1.0 - cos_ph)
        z = pipeline_state.x.pos[0][2]
        return jnp.exp(-50.0 * (z - z_ref) ** 2)


class SquatG1Env(SquatG1Recipe, WalkEnv, env_name="squat_g1"):
    """Squat on the plain (no-thermal) stack.

    Inherits WalkEnv for the shared reward catalog (torso_roll/pitch,
    feet_slip, feet_distance, ...); walking-only rewards are zeroed in
    squat_g1.gin.
    """

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
        # WalkEnv.__init__ is bypassed (it builds a ZMP walk reference)
        MJXEnv.__init__(
            self,
            name,
            robot,
            cfg,
            self.make_motion_ref(robot, cfg),
            fixed_base=fixed_base,
            add_noise=add_noise,
            add_domain_rand=add_domain_rand,
            **kwargs,
        )
        self._squat_setup(robot, cfg)
