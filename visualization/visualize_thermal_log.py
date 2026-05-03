"""Visualize walking + cool-down motor temperatures from log_data.pkl.

Walking motor_temp comes from obs_list (in log_data.pkl if populated, otherwise
from active_log_data.pkl in the same directory). Cool-down motor_temp comes
from cool_down_list entries of the form (time, motor_tor, motor_temp) written
by toddlerbot/policies/run_policy.py. Walking and cool-down are plotted as two
distinct traces on a shared time axis — cool-down time is offset by the final
walking timestamp so the two phases sit back-to-back.

Usage:
    python visualize_thermal_log.py <path/to/log_data.pkl> [--out-dir DIR]

If --out-dir is omitted, figures are saved next to the pickle file under a
`plots/` subfolder.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_log(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _motor_names_from_obs(obs_list, motor_angles_list) -> List[str]:
    if len(motor_angles_list) > 0:
        return list(motor_angles_list[0].keys())
    n_motors = len(obs_list[0].motor_temp)
    return [f"motor_{i}" for i in range(n_motors)]


def _walk_arrays(data: Dict[str, Any]):
    obs_list = data.get("obs_list", [])
    if len(obs_list) == 0:
        return None, None, []
    t = np.array([o.time for o in obs_list], dtype=np.float64)
    temp = np.asarray([o.motor_temp for o in obs_list], dtype=np.float32)
    names = _motor_names_from_obs(obs_list, data.get("motor_angles_list", []))
    return t, temp, names


def _cooldown_arrays(cool_down_list):
    if len(cool_down_list) == 0:
        return None, None
    t = np.array([row[0] for row in cool_down_list], dtype=np.float64)
    # cool_down_list entries are (time, motor_tor, motor_temp) — index 2 is temp.
    temp = np.asarray([row[2] for row in cool_down_list], dtype=np.float32)
    return t, temp


def extract_arrays(pkl_path: str):
    data = load_log(pkl_path)

    walk_t, walk_temp, motor_names = _walk_arrays(data)

    # Fallback: run_policy.py flushes walking to active_log_data.pkl before cool-down.
    if walk_t is None:
        active_path = os.path.join(
            os.path.dirname(os.path.abspath(pkl_path)), "active_log_data.pkl"
        )
        if os.path.exists(active_path):
            active = load_log(active_path)
            walk_t, walk_temp, motor_names = _walk_arrays(active)

    cool_t, cool_temp = _cooldown_arrays(data.get("cool_down_list", []))

    if walk_t is None and cool_t is None:
        raise ValueError(f"No motor temperature data found in {pkl_path}")

    # Cool-down timestamps restart from 0 in run_policy.py, so offset them.
    walk_end = float(walk_t[-1]) if walk_t is not None and len(walk_t) > 0 else 0.0
    if cool_t is not None:
        cool_t = cool_t + walk_end

    n_motors = (walk_temp.shape[1] if walk_temp is not None
                else cool_temp.shape[1])
    if motor_names is None or len(motor_names) == 0:
        motor_names = [f"motor_{i}" for i in range(n_motors)]

    return {
        "walk_t": walk_t,
        "walk_temp": walk_temp,
        "cool_t": cool_t,
        "cool_temp": cool_temp,
        "motor_names": motor_names,
        "walk_end": walk_end if walk_t is not None else None,
    }


def _grid(n: int) -> Tuple[int, int]:
    ncols = 4 if n >= 8 else min(n, 4)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _save(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_per_motor_temp(ex, out_dir, file_name="motor_temp"):
    motor_names = ex["motor_names"]
    walk_t, walk_temp = ex["walk_t"], ex["walk_temp"]
    cool_t, cool_temp = ex["cool_t"], ex["cool_temp"]
    walk_end = ex["walk_end"]

    n = len(motor_names)
    nrows, ncols = _grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for i, name in enumerate(motor_names):
        ax = axes[i]
        if walk_temp is not None:
            ax.plot(walk_t, walk_temp[:, i], lw=0.9,
                    color="tab:blue", label="walk")
        if cool_temp is not None:
            ax.plot(cool_t, cool_temp[:, i], lw=0.9,
                    color="tab:orange", label="cool-down")
        if walk_end is not None and cool_t is not None:
            ax.axvline(walk_end, color="k", ls=":", lw=0.7, alpha=0.6)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("temp (°C)", fontsize=8)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("time (s)", fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7, loc="best")
    _save(fig, out_dir, file_name)


def visualize(pkl_path: str, out_dir: str | None = None):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(pkl_path)), "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {pkl_path} ...")
    ex = extract_arrays(pkl_path)

    walk_n = 0 if ex["walk_t"] is None else len(ex["walk_t"])
    cool_n = 0 if ex["cool_t"] is None else len(ex["cool_t"])
    print(f"  motors: {len(ex['motor_names'])}  "
          f"walk samples: {walk_n}  cool-down samples: {cool_n}")

    plt.switch_backend("Agg")
    plot_per_motor_temp(ex, out_dir)

    print(f"Done. Figures written to: {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pkl", help="Path to log_data.pkl")
    p.add_argument("--out-dir", default=None,
                   help="Directory to save figures (default: <pkl_dir>/plots)")
    args = p.parse_args(argv)
    visualize(args.pkl, args.out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
