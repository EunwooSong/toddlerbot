from typing import Any, Optional

import jax
import jax.numpy as jnp

from toddlerbot.sim.robot import Robot
from toddlerbot.reference.pushup_ref import PushupReference
from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv

class PushUpnv(MJXEnv, env_name="push_up"):
    """Environment for training crawling locomotion with ToddlerBot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        fixed_base: bool = False,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        """Initialize the crawl environment with motion reference setup."""
        motion_ref = PushupReference(
            robot, cfg.sim.timestep * cfg.action.n_frames
        )
        super().__init__(
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
        """Sample zero command for crawl motion (simplified command structure)."""
        # For crawling, we'll use simpler commands focused on forward movement
        """
        rng, rng_1, rng_2 = jax.random.split(rng, 3)

        # Debug prints for shapes
        # print("Command range shape:", self.command_range.shape)
        # print("Deadzone shape:", self.deadzone.shape)

        # Sample forward velocity command
        forward_vel = jax.random.uniform(
            rng_1,
            (1,),
            minval=self.deadzone[0],  # Use first deadzone value
            maxval=self.command_range[0][1],
        )
        # print("Forward vel shape:", forward_vel.shape)

        # Sample turning command (less frequent than walking)
        turn_vel = jnp.where(
            jax.random.uniform(rng_2, (1,)) < self.turn_chance * 0.5,  # Reduced turn chance
            jax.random.uniform(
                rng_2,
                (1,),
                minval=self.deadzone[1],  # Use second deadzone value
                maxval=self.command_range[1][1],
            ),
            jnp.zeros(1),
        )
        # print("Turn vel shape:", turn_vel.shape)

        # Add zero for the third dimension (yaw)
        yaw_vel = jnp.zeros(1)
        # print("Yaw vel shape:", yaw_vel.shape)

        # Combine commands into a single command array
        command = jnp.concatenate([forward_vel, turn_vel, yaw_vel])
        # print("Final command shape:", command.shape)
        return command
        """
        # Set to zero for now
        return jnp.zeros(3)