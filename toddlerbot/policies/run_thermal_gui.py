"""Entry point: MuJoCo + Thermal model + Tk GUI.

Mirrors `run_policy.main` but uses `MuJoCoThermalSim` and brings up the
`ThermalGUI`. Tk runs on the main thread; the simulation loop runs on a
daemon thread so users can toggle modes / reset state interactively.

Examples
--------
# Baseline walk policy (no MTJX thermal observation in obs):
python toddlerbot/policies/run_thermal_gui.py --policy walk

# MTJX thermal_walk policy with an ablation gin file:
python toddlerbot/policies/run_thermal_gui.py \
    --policy thermal_walk --gin-file ablation/abf2_comp_obs_on_derate_on_safety_off
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Dict, List

import gin
import numpy as np

from toddlerbot.policies import (
    BasePolicy,
    get_policy_class,
    get_policy_names,
)
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.policies.mtjx_policy import MTJXPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.mujoco_thermal_sim import MuJoCoThermalSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import log, snake2camel
from toddlerbot.visualization.thermal_gui import ThermalGUI

from heat2torque import ThermalMode

# Import the policies we actually need explicitly. Avoid pulling
# `run_policy.dynamic_import_policies` since that drags in `dp_policy`,
# which requires torch (and a matching cudnn) — unrelated to thermal work.
from toddlerbot.policies import walk as _walk  # noqa: F401
from toddlerbot.policies import thermal_walk as _thermal_walk  # noqa: F401
from toddlerbot.policies import stand as _stand  # noqa: F401


def _build_policy(args, robot: Robot, init_motor_pos) -> BasePolicy:
    PolicyClass = get_policy_class(args.policy.replace("_fixed", ""))

    fixed_command = None
    if args.command:
        fixed_command = np.array(args.command.split(" "), dtype=np.float32)

    if issubclass(PolicyClass, MTJXPolicy):
        return PolicyClass(
            args.policy, robot, init_motor_pos, args.ckpt, fixed_command=fixed_command
        )
    if issubclass(PolicyClass, MJXPolicy):
        return PolicyClass(
            args.policy, robot, init_motor_pos, args.ckpt, fixed_command=fixed_command
        )
    return PolicyClass(args.policy, robot, init_motor_pos)


def _restart_policy(policy: BasePolicy, sim: MuJoCoThermalSim) -> None:
    """Put `policy` back into its post-construction state so the next `step`
    rebuilds the warm-up trajectory from the current motor positions.

    Mirrors `MJXPolicy.__init__`/`reset` enough that:
      - `is_prepared` flips back to False, forcing `step` to recompute
        `prep_time`/`prep_action` on its first call.
      - `init_motor_pos` is refreshed so the prep interpolation starts from
        wherever the robot is right now (the post-pose-reset default pose).
      - obs history, action buffer, phase, step_curr all clear via `reset()`.
    """
    try:
        init_pos = sim.get_observation().motor_pos
        if hasattr(policy, "init_motor_pos") and init_pos is not None:
            policy.init_motor_pos = np.asarray(init_pos, dtype=np.float32)
    except Exception:
        pass

    if hasattr(policy, "is_prepared"):
        policy.is_prepared = False
    if hasattr(policy, "reset"):
        try:
            policy.reset()
        except Exception:
            pass


def _sim_loop(sim: MuJoCoThermalSim, policy: BasePolicy, stop_event: threading.Event):
    """Simulation loop. Runs on a daemon thread so the GUI mainloop owns the main thread."""

    header = snake2camel(sim.name)
    start = time.time()
    step_idx = 0
    time_until_next = 0.0

    try:
        while not stop_event.is_set():
            # Drain GUI-requested policy restarts (Reset Pose). We do this
            # *before* `get_observation` so the rebuilt prep trajectory
            # starts from the freshly reset motor positions.
            if sim.consume_policy_restart():
                _restart_policy(policy, sim)
                start = time.time()
                step_idx = 0
                time_until_next = 0.0
                log("policy restarted (Reset Pose)", header=header)

            step_start = time.time()

            obs: Obs = sim.get_observation()
            obs.time -= start
            obs.time += time_until_next

            _, motor_target = policy.step(obs, is_real=False)

            motor_angles: Dict[str, float] = {
                name: angle
                for name, angle in zip(sim.robot.motor_ordering, motor_target)
            }
            sim.set_motor_target(motor_angles)
            sim.step()

            step_idx += 1
            step_end = time.time()
            time_until_next = start + policy.control_dt * step_idx - step_end
            if time_until_next > 0:
                if stop_event.wait(time_until_next):
                    break
    except Exception as e:
        log(f"sim loop error: {e}", header=header)
    finally:
        try:
            sim.close()
        except Exception:
            pass


def main(args=None):
    parser = argparse.ArgumentParser(description="MuJoCo + Thermal GUI sandbox.")
    parser.add_argument("--robot", type=str, default="toddlerbot")
    parser.add_argument(
        "--policy",
        type=str,
        default="thermal_walk",
        choices=get_policy_names(),
    )
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--command", type=str, default="")
    parser.add_argument(
        "--vis", type=str, default="view", choices=["view", "render", "none"]
    )
    parser.add_argument("--gin-file", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="thermal",
        choices=["basic", "thermal", "derate"],
        help="Initial thermal mode.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="PRNG seed for thermal init."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record qpos+thermal trajectory; dumped to a npz on exit "
        "(render/replay separately).",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="Output dir for the recording (default: results/mtj_rec_<ts>).",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="real walk session dir (log_data.pkl). Set ⇒ compare-mode: "
        "cold-start aligned to real, log_data.pkl recorded. Unset ⇒ "
        "unchanged random sim.",
    )
    parser.add_argument(
        "--tamb-csv",
        type=str,
        default=None,
        help="ds18b20_<unix>.csv for time-varying ambient (unix-aligned via "
        "real start_time_unix). Optional; absent ⇒ a_t held at real "
        "motor_temp[0].",
    )
    args = parser.parse_args(args)

    if args.gin_file is not None:
        gin_file = args.gin_file
        if not gin_file.endswith(".gin"):
            gin_file = gin_file + ".gin"
        # gin files are stored under toddlerbot/locomotion/
        candidates = [
            gin_file,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "locomotion",
                gin_file,
            ),
        ]
        gin_path = next((p for p in candidates if os.path.exists(p)), None)
        if gin_path is None:
            raise FileNotFoundError(
                f"gin file not found in any of: {candidates}"
            )
        gin.parse_config_file(gin_path)
        print(f"[run_thermal_gui] gin parsed: {gin_path}")

    # Thermal/Derate = 결합 2차 LPTN(논문 최종 모델 A3-H) → USE_COUPLING 포함.
    initial_mode = {
        "basic": ThermalMode.DISABLE,
        "thermal": (
            ThermalMode.USE_THERMAL
            | ThermalMode.MODEL_ORDER_2
            | ThermalMode.USE_COUPLING
        ),
        "derate": (
            ThermalMode.USE_DERATE
            | ThermalMode.USE_THERMAL
            | ThermalMode.MODEL_ORDER_2
            | ThermalMode.USE_COUPLING
        ),
    }[args.mode]

    robot = Robot(args.robot)

    sim = MuJoCoThermalSim(
        robot,
        thermal_mode=initial_mode,
        thermal_seed=args.seed,
        vis_type=args.vis,
        fixed_base="fixed" in args.policy,
        record=args.record,
        record_dir=args.record_dir,
        ref_dir=args.ref,
        tamb_csv=args.tamb_csv,
    )
    if args.record:
        print("[run_thermal_gui] recording ON — trajectory dumps on exit.")
    init_motor_pos = sim.get_observation().motor_pos

    policy = _build_policy(args, robot, init_motor_pos)
    if hasattr(policy, "warmup_late"):
        policy.warmup_late()

    stop_event = threading.Event()
    sim_thread = threading.Thread(
        target=_sim_loop, args=(sim, policy, stop_event), daemon=True
    )
    sim_thread.start()

    gui = ThermalGUI(
        sim,
        motor_ordering=sim.robot.motor_ordering,
        motor_groups=sim.robot.joint_groups,
        on_close=lambda: stop_event.set(),
    )
    try:
        gui.mainloop()
    finally:
        stop_event.set()
        sim_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
