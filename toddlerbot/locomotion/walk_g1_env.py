from typing import Any

import jax.numpy as jnp

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.locomotion.walk_env import WalkEnv
from toddlerbot.reference.walk_zmp_ref_g1 import WalkZMPReferenceG1
from toddlerbot.sim.robot import Robot


class WalkG1Env(WalkEnv, env_name="walk_g1"):
    """Walk environment for Unitree G1: legs-only action, static upper body
    (arms forward-90), G1 ZMP reference. Rewards inherited from WalkEnv."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        ref_motion_type: str = "zmp",
        fixed_base: bool = False,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        motion_ref = WalkZMPReferenceG1(
            robot,
            cfg.sim.timestep * cfg.action.n_frames,
            cfg.action.cycle_time,
            cfg.action.waist_roll_max,
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
