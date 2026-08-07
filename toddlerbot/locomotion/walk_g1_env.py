"""Unitree G1 walk env — faithful port of the unitree_rl_gym G1 recipe.

Real-robot-proven recipe (oracle: their pretrained motion.pt walks in our
local MuJoCo, 2026-08-06): legs-only 12 actions, fixed 0.8 s gait clock,
legged_gym reward set with G1-tuned weights, 47-dim single-frame obs content
(here stacked by our standard frame-stacking pipeline), PD-in-code + torque
actuators (matches their deploy_mujoco architecture exactly).

Sources: unitree_rl_gym legged_gym/envs/g1/{g1_env.py,g1_config.py} and
deploy/deploy_mujoco/. Upper body (17 joints) stays frozen at default pose
(L-shape arms) via the reference/action-mask machinery.
"""
from typing import Any

import jax
import jax.numpy as jnp
from brax import base

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.locomotion.walk_env import WalkEnv
from toddlerbot.reference.phase_clock_ref_g1 import STANCE_DUTY, PhaseClockReferenceG1
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import quat_inv, rotate_vec

BASE_HEIGHT_TARGET = 0.78     # g1_config.py rewards.base_height_target
SWING_HEIGHT_TARGET = 0.08    # g1_env.py _reward_feet_swing_height
SOFT_DOF_LIMIT = 0.9          # g1_config.py rewards.soft_dof_pos_limit
# obs scales, deploy g1.yaml: ang_vel 0.25, dof_pos 1.0, dof_vel 0.05, cmd [2,2,0.25]
OBS_ANG_VEL_SCALE = 0.25
OBS_DOF_VEL_SCALE = 0.05
OBS_CMD_SCALE = jnp.array([2.0, 2.0, 0.25])
OBS_LIN_VEL_SCALE = 2.0       # privileged critic obs


class WalkG1Recipe:
    """G1 walking recipe (gait clock, 47-dim obs, legged_gym rewards).

    Engine-agnostic mixin so the thermal env (`_T_Walk_G1`) and the plain env
    (`walk_g1`) share one definition — the recipe must not drift between them
    or the thermal ablation stops being a controlled comparison.
    """
    def _g1_setup(self, robot, cfg):
        """Recipe state that must be built after the base env __init__."""
        self.cycle_time = jnp.array(cfg.action.cycle_time)
        self.torso_roll_range = cfg.rewards.torso_roll_range
        self.torso_pitch_range = cfg.rewards.torso_pitch_range
        self.max_feet_air_time = self.cycle_time / 2.0
        self.min_feet_y_dist = cfg.rewards.min_feet_y_dist
        self.max_feet_y_dist = cfg.rewards.max_feet_y_dist

        # noise vector for the 47-dim obs layout (legged_gym magnitudes:
        # ang_vel 0.2*0.25, gravity 0.05, dof_pos 0.01, dof_vel 1.5*0.05)
        self.obs_noise_scale = jnp.concatenate([
            jnp.full(3, 0.2 * OBS_ANG_VEL_SCALE),
            jnp.full(3, 0.05),
            jnp.zeros(3),
            jnp.full(12, 0.01),
            jnp.full(12, 1.5 * OBS_DOF_VEL_SCALE),
            jnp.zeros(12),
            jnp.zeros(2),
        ])

        leg_limits = self.motor_limits[
            jnp.array([i for i, n in enumerate(robot.motor_ordering)
                       if robot.joint_groups[n] == "leg"])
        ]
        mid = (leg_limits[:, 0] + leg_limits[:, 1]) / 2
        rng = leg_limits[:, 1] - leg_limits[:, 0]
        self.soft_leg_limit_lo = mid - 0.5 * rng * SOFT_DOF_LIMIT
        self.soft_leg_limit_hi = mid + 0.5 * rng * SOFT_DOF_LIMIT
        # leg vector order [hp, hr, hy, knee, ar, ap] x2 -> hip roll/yaw columns
        self.hip_ry_cols = jnp.array([1, 2, 7, 8])
        self.default_leg_qpos = self.default_qpos[
            self.q_start_idx + self.leg_motor_indices]

    # ------------------------------------------------------------------ phase
    def _leg_phase(self, info: dict[str, Any]) -> jax.Array:
        t = info["step"] * self.dt
        phase = jnp.mod(t, self.cycle_time) / self.cycle_time
        return jnp.array([phase, jnp.mod(phase + 0.5, 1.0)])

    def _feet_contact(self, info: dict[str, Any]) -> jax.Array:
        cf = info["contact_forces"]
        left = jnp.any(
            cf[0, self.left_foot_collider_indices, 2] > self.contact_force_threshold)
        right = jnp.any(
            cf[0, self.right_foot_collider_indices, 2] > self.contact_force_threshold)
        return jnp.array([left, right])

    # ------------------------------------------------------------------- obs
    def _get_obs(self, pipeline_state, info, obs_history, privileged_obs_history):
        rng, obs_rng = jax.random.split(info["rng"], 2)

        torso_quat = pipeline_state.x.rot[0]
        omega = rotate_vec(pipeline_state.xd.ang[0], quat_inv(torso_quat))
        gravity = rotate_vec(jnp.array([0.0, 0.0, -1.0]), quat_inv(torso_quat))
        lin_vel = rotate_vec(pipeline_state.xd.vel[0], quat_inv(torso_quat))

        qj = (pipeline_state.q[self.q_start_idx + self.leg_motor_indices]
              - self.default_leg_qpos)
        dqj = pipeline_state.qd[self.qd_start_idx + self.leg_motor_indices]

        obs = jnp.concatenate([
            omega * OBS_ANG_VEL_SCALE,
            gravity,
            info["command"][jnp.array([5, 6, 7])] * OBS_CMD_SCALE,
            qj,
            dqj * OBS_DOF_VEL_SCALE,
            info["last_act"],
            info["phase_signal"],
        ])
        privileged_obs = jnp.concatenate([lin_vel * OBS_LIN_VEL_SCALE, obs])

        if self.add_noise:
            obs += self.obs_noise_scale * jax.random.normal(obs_rng, obs.shape)

        obs = jnp.roll(obs_history, obs.size).at[: obs.size].set(obs)
        privileged_obs = (
            jnp.roll(privileged_obs_history, privileged_obs.size)
            .at[: privileged_obs.size]
            .set(privileged_obs)
        )
        return obs, privileged_obs

    # -------------------------------------------------- legged_gym rewards
    # sign convention: functions return POSITIVE costs; gin weights are
    # negative for penalties, exactly like legged_gym scales.

    def _reward_lin_vel_z(self, pipeline_state: base.State, info, action):
        v = rotate_vec(pipeline_state.xd.vel[0], quat_inv(pipeline_state.x.rot[0]))
        return jnp.square(v[2])

    def _reward_ang_vel_xy(self, pipeline_state: base.State, info, action):
        w = rotate_vec(pipeline_state.xd.ang[0], quat_inv(pipeline_state.x.rot[0]))
        return jnp.sum(jnp.square(w[:2]))

    def _reward_orientation(self, pipeline_state: base.State, info, action):
        g = rotate_vec(jnp.array([0.0, 0.0, -1.0]),
                       quat_inv(pipeline_state.x.rot[0]))
        return jnp.sum(jnp.square(g[:2]))

    def _reward_base_height(self, pipeline_state: base.State, info, action):
        return jnp.square(pipeline_state.qpos[2] - BASE_HEIGHT_TARGET)

    def _reward_dof_acc(self, pipeline_state: base.State, info, action):
        qacc = pipeline_state.qacc[self.qd_start_idx + self.leg_motor_indices]
        return jnp.sum(jnp.square(qacc))

    def _reward_dof_vel(self, pipeline_state: base.State, info, action):
        dq = pipeline_state.qd[self.qd_start_idx + self.leg_motor_indices]
        return jnp.sum(jnp.square(dq))

    def _reward_dof_pos_limits(self, pipeline_state: base.State, info, action):
        q = pipeline_state.q[self.q_start_idx + self.leg_motor_indices]
        low = jnp.clip(q - self.soft_leg_limit_lo, max=0.0)
        high = jnp.clip(q - self.soft_leg_limit_hi, min=0.0)
        return jnp.sum(-low + high)

    def _reward_hip_pos(self, pipeline_state: base.State, info, action):
        q = pipeline_state.q[self.q_start_idx + self.leg_motor_indices]
        return jnp.sum(jnp.square(q[self.hip_ry_cols]))

    def _reward_contact_no_vel(self, pipeline_state: base.State, info, action):
        contact = self._feet_contact(info)
        feet_vel = pipeline_state.xd.vel[self.feet_link_ids]
        return jnp.sum(jnp.square(feet_vel * contact[:, None]))

    def _reward_feet_swing_height(self, pipeline_state: base.State, info, action):
        contact = self._feet_contact(info)
        feet_z = pipeline_state.x.pos[self.feet_link_ids, 2]
        return jnp.sum(jnp.square(feet_z - SWING_HEIGHT_TARGET) * (1.0 - contact))

    def _reward_survival(self, pipeline_state: base.State, info, action):
        # legged_gym "alive": constant +1 per step (base class returns a
        # death penalty instead, which never pays standing/walking)
        return jnp.float32(1.0)

    def _reward_feet_contact(self, pipeline_state: base.State, info, action):
        # +1 per foot whose contact state matches the gait-clock stance phase
        is_stance = self._leg_phase(info) < STANCE_DUTY
        contact = self._feet_contact(info)
        return jnp.sum((contact == is_stance).astype(jnp.float32))


class WalkG1Env(WalkG1Recipe, WalkEnv, env_name="walk_g1"):
    """G1 walk, no thermal model (Phase 1)."""

    def __init__(self, name, robot, cfg, ref_motion_type="phase", fixed_base=False,
                 add_noise=True, add_domain_rand=True, **kwargs):
        MJXEnv.__init__(
            self, name, robot, cfg,
            PhaseClockReferenceG1(robot, cfg.sim.timestep * cfg.action.n_frames,
                                  cfg.action.cycle_time, cfg.action.waist_roll_max),
            fixed_base=fixed_base, add_noise=add_noise,
            add_domain_rand=add_domain_rand, **kwargs)
        self._g1_setup(robot, cfg)
