"""Visualize active_log_data.pkl produced by toddlerbot/policies/run_policy.py.

The pickle file contains:
    - obs_list:            List[Obs]            (time, motor_pos/vel/tor/temp, lin/ang_vel, pos, euler)
    - control_inputs_list: List[Dict[str, float]]
    - motor_angles_list:   List[Dict[str, float]]  (target motor angles per step)
    - loop_time_list:      List[List[float]]       (6 timestamps per step)

Usage:
    python visualize_active_log.py <path/to/active_log_data.pkl> [--out-dir DIR]

If --out-dir is omitted, figures are saved next to the pickle file under a `plots/` subfolder.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_log(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def extract_arrays(data: Dict[str, Any]):
    obs_list = data["obs_list"]
    motor_angles_list = data.get("motor_angles_list", [])
    control_inputs_list = data.get("control_inputs_list", [])
    loop_time_list = data.get("loop_time_list", [])

    if len(obs_list) == 0:
        raise ValueError("obs_list is empty — nothing to visualize.")

    # Motor ordering from the first target-angle dict (keys preserve insertion order).
    if len(motor_angles_list) > 0:
        motor_names = list(motor_angles_list[0].keys())
    else:
        n_motors = len(obs_list[0].motor_pos)
        motor_names = [f"motor_{i}" for i in range(n_motors)]

    t = np.array([o.time for o in obs_list], dtype=np.float64)

    def stack(attr):
        vals = [getattr(o, attr) for o in obs_list]
        if any(v is None for v in vals):
            return None
        return np.asarray(vals, dtype=np.float32)

    motor_pos = stack("motor_pos")
    motor_vel = stack("motor_vel")
    motor_tor = stack("motor_tor")
    motor_temp = stack("motor_temp")
    ang_vel = stack("ang_vel")
    lin_vel = stack("lin_vel")
    pos = stack("pos")
    euler = stack("euler")

    # Target action array aligned with motor_names.
    if len(motor_angles_list) > 0:
        action = np.array(
            [[d.get(n, np.nan) for n in motor_names] for d in motor_angles_list],
            dtype=np.float32,
        )
    else:
        action = None

    return {
        "t": t,
        "motor_names": motor_names,
        "motor_pos": motor_pos,
        "motor_vel": motor_vel,
        "motor_tor": motor_tor,
        "motor_temp": motor_temp,
        "ang_vel": ang_vel,
        "lin_vel": lin_vel,
        "pos": pos,
        "euler": euler,
        "action": action,
        "control_inputs_list": control_inputs_list,
        "loop_time_list": loop_time_list,
    }


def _grid(n: int):
    ncols = 4 if n >= 8 else min(n, 4)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _save(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_per_motor(t, data_arr, motor_names, out_dir, file_name, y_label,
                   ref_arr=None, title_prefix=""):
    if data_arr is None:
        return
    n = len(motor_names)
    nrows, ncols = _grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(motor_names):
        ax = axes[i]
        ax.plot(t, data_arr[:, i], lw=0.9, label="obs")
        if ref_arr is not None:
            ax.plot(t, ref_arr[:, i], lw=0.9, ls="--", label="ref")
        ax.set_title(f"{title_prefix}{name}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel(y_label, fontsize=8)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("time (s)", fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    if ref_arr is not None:
        axes[0].legend(fontsize=7, loc="best")
    _save(fig, out_dir, file_name)


def plot_xyz(t, arr, out_dir, file_name, title, y_label,
             labels=("x", "y", "z")):
    if arr is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, lab in enumerate(labels):
        ax.plot(t, arr[:, i], lw=0.9, label=lab)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out_dir, file_name)


def plot_total_torque(t, motor_tor, out_dir):
    if motor_tor is None:
        return
    total = motor_tor.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, total, lw=0.9)
    ax.set_title("Total motor torque / current over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sum(motor_tor)")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "total_torque")


def plot_loop_time(loop_time_list, out_dir):
    if len(loop_time_list) == 0:
        return
    arr = np.asarray(loop_time_list, dtype=np.float64)
    if arr.shape[1] != 6:
        return
    step_start = arr[:, 0]
    segs = {
        "obs": (arr[:, 1] - arr[:, 0]) * 1000,
        "inference": (arr[:, 2] - arr[:, 1]) * 1000,
        "set_action": (arr[:, 3] - arr[:, 2]) * 1000,
        "sim_step": (arr[:, 4] - arr[:, 3]) * 1000,
        "log": (arr[:, 5] - arr[:, 4]) * 1000,
    }
    t_rel = step_start - step_start[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, vals in segs.items():
        ax.plot(t_rel, vals, lw=0.7, label=name)
    ax.set_title("Per-step loop time breakdown")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("duration (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_dir, "loop_time")


def plot_control_inputs(control_inputs_list, t, out_dir):
    if len(control_inputs_list) == 0:
        return
    keys = list(control_inputs_list[0].keys())
    if len(keys) == 0:
        return
    n = min(len(control_inputs_list), len(t))
    arr = np.array(
        [[d.get(k, np.nan) for k in keys] for d in control_inputs_list[:n]],
        dtype=np.float32,
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, k in enumerate(keys):
        ax.plot(t[:n], arr[:, i], lw=0.9, label=k)
    ax.set_title("Control inputs over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir, "control_inputs")


def plot_summary(ex, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    t = ex["t"]

    if ex["motor_tor"] is not None:
        axes[0, 0].plot(t, ex["motor_tor"].sum(axis=1), lw=0.9)
    axes[0, 0].set_title("Total motor torque / current")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].grid(True, alpha=0.3)

    if ex["motor_temp"] is not None:
        mean_temp = ex["motor_temp"].mean(axis=1)
        max_temp = ex["motor_temp"].max(axis=1)
        axes[0, 1].plot(t, mean_temp, lw=0.9, label="mean")
        axes[0, 1].plot(t, max_temp, lw=0.9, label="max")
        axes[0, 1].legend()
    axes[0, 1].set_title("Motor temperature (°C)")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].grid(True, alpha=0.3)

    if ex["euler"] is not None:
        for i, lab in enumerate(("roll", "pitch", "yaw")):
            axes[1, 0].plot(t, ex["euler"][:, i], lw=0.9, label=lab)
        axes[1, 0].legend()
    axes[1, 0].set_title("Base euler (rad)")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    if ex["ang_vel"] is not None:
        for i, lab in enumerate(("wx", "wy", "wz")):
            axes[1, 1].plot(t, ex["ang_vel"][:, i], lw=0.9, label=lab)
        axes[1, 1].legend()
    axes[1, 1].set_title("Base angular velocity (rad/s)")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].grid(True, alpha=0.3)

    _save(fig, out_dir, "summary")


def visualize(pkl_path: str, out_dir: str | None = None):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(pkl_path)), "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {pkl_path} ...")
    data = load_log(pkl_path)
    print(f"  obs_list length: {len(data.get('obs_list', []))}")
    ex = extract_arrays(data)
    print(f"  motors: {len(ex['motor_names'])}  duration: "
          f"{ex['t'][-1] - ex['t'][0]:.1f}s")

    plt.switch_backend("Agg")

    plot_summary(ex, out_dir)
    plot_total_torque(ex["t"], ex["motor_tor"], out_dir)
    plot_xyz(ex["t"], ex["ang_vel"], out_dir, "ang_vel",
             "Base angular velocity", "rad/s", ("roll(x)", "pitch(y)", "yaw(z)"))
    plot_xyz(ex["t"], ex["lin_vel"], out_dir, "lin_vel",
             "Base linear velocity", "m/s")
    plot_xyz(ex["t"], ex["euler"], out_dir, "euler",
             "Base euler angles", "rad", ("roll(x)", "pitch(y)", "yaw(z)"))
    plot_xyz(ex["t"], ex["pos"], out_dir, "base_pos",
             "Base position", "m")

    plot_per_motor(ex["t"], ex["motor_pos"], ex["motor_names"], out_dir,
                   "motor_pos_tracking", "pos (rad)", ref_arr=ex["action"])
    plot_per_motor(ex["t"], ex["motor_vel"], ex["motor_names"], out_dir,
                   "motor_vel", "vel (rad/s)")
    plot_per_motor(ex["t"], ex["motor_tor"], ex["motor_names"], out_dir,
                   "motor_tor", "torque / current")
    plot_per_motor(ex["t"], ex["motor_temp"], ex["motor_names"], out_dir,
                   "motor_temp", "temp (°C)")

    plot_loop_time(ex["loop_time_list"], out_dir)
    plot_control_inputs(ex["control_inputs_list"], ex["t"], out_dir)

    print(f"Done. Figures written to: {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pkl", help="Path to active_log_data.pkl")
    p.add_argument("--out-dir", default=None,
                   help="Directory to save figures (default: <pkl_dir>/plots)")
    args = p.parse_args(argv)
    visualize(args.pkl, args.out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
