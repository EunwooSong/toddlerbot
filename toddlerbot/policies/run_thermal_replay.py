"""replay_gui — interactive replay of a recorded thermal trajectory.

`run_thermal_gui.py --record` 가 남긴 `trajectory.npz`(+meta) 를 실제
MuJoCo passive viewer(마우스 카메라 상호작용)로 재생한다. Tk 컨트롤바로
재생/일시정지 · 배속(0.1~4x) · 프레임 스크럽(임의 프레임 이동) · ±1 ·
처음/끝 이동을 지원한다. (mtj_gui 와 동일 구조: Tk 메인 + 데몬 스레드.)

Usage:
    python toddlerbot/policies/run_thermal_replay.py results/mtj_rec_XXXX
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

# passive viewer 는 실제 GL 컨텍스트 필요 → EGL 강제 해제.
os.environ.pop("MUJOCO_GL", None)

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from toddlerbot.utils.file_utils import find_robot_file_path  # noqa: E402


def _resolve(path: str) -> str:
    d = path if os.path.isdir(path) else (os.path.dirname(path) or ".")
    if not os.path.isfile(os.path.join(d, "trajectory.npz")):
        raise FileNotFoundError(f"trajectory.npz not found in {d}")
    return d


class ReplayGUI:
    def __init__(self, rec_dir: str):
        with open(os.path.join(rec_dir, "meta.json")) as f:
            self.meta = json.load(f)
        npz = np.load(os.path.join(rec_dir, "trajectory.npz"))
        self.qpos = np.asarray(npz["qpos"], dtype=np.float64)      # (T, nq)
        self.h_t = np.asarray(npz["h_t"], dtype=np.float32)        # (T, 30)
        self.w_t = np.asarray(npz["w_t"], dtype=np.float32)
        self.T = int(self.qpos.shape[0])
        self.control_dt = float(self.meta.get("control_dt", 0.02))

        suffix = "_fixed_scene.xml" if self.meta.get("fixed_base") else "_scene.xml"
        xml_path = find_robot_file_path(self.meta["robot"], suffix=suffix)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = float(self.meta.get("dt", self.model.opt.timestep))
        self.data = mujoco.MjData(self.model)

        # ── shared transport state (lock 보호) ──
        self._lock = threading.Lock()
        self._idx = 0
        self._play = False
        self._speed = 1.0
        self._loopback = True          # 끝나면 처음으로
        self._stop = threading.Event()
        self._suppress_slider = False  # 프로그램이 슬라이더 set 할 때 콜백 무시

        self._build_ui(rec_dir)
        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()

    # --------------------------------------------------------------- UI
    def _build_ui(self, rec_dir: str):
        self.root = tk.Tk()
        self.root.title(f"replay_gui — {os.path.basename(rec_dir)} "
                        f"({self.meta['robot']}, {self.T} steps)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")

        self.play_var = tk.StringVar(value="▶ Play")
        ttk.Button(bar, textvariable=self.play_var, width=9,
                   command=self._toggle_play).pack(side="left")
        ttk.Button(bar, text="⏮", width=3, command=lambda: self._seek(0)).pack(
            side="left", padx=(6, 0))
        ttk.Button(bar, text="◀ -1", width=5,
                   command=lambda: self._nudge(-1)).pack(side="left", padx=(4, 0))
        ttk.Button(bar, text="+1 ▶", width=5,
                   command=lambda: self._nudge(1)).pack(side="left", padx=(4, 0))
        ttk.Button(bar, text="⏭", width=3,
                   command=lambda: self._seek(self.T - 1)).pack(
            side="left", padx=(4, 8))

        ttk.Label(bar, text="Speed").pack(side="left")
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(bar, from_=0.1, to=4.0, orient="horizontal", length=160,
                  variable=self.speed_var,
                  command=self._on_speed).pack(side="left", padx=(4, 2))
        self.speed_lbl = tk.StringVar(value="1.00x")
        ttk.Label(bar, textvariable=self.speed_lbl, width=6).pack(side="left")

        self.loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="loop", variable=self.loop_var,
                        command=self._on_loop).pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="paused")
        ttk.Label(bar, textvariable=self.status_var,
                  foreground="#666").pack(side="right")

        scrub = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        scrub.pack(fill="x")
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(
            scrub, from_=0, to=max(self.T - 1, 1), orient="horizontal",
            variable=self.frame_var, command=self._on_scrub)
        self.frame_scale.pack(fill="x")

        self.root.after(80, self._refresh_status)

    # ------------------------------------------------------- callbacks
    def _toggle_play(self):
        with self._lock:
            self._play = not self._play
        self.play_var.set("⏸ Pause" if self._play else "▶ Play")

    def _on_speed(self, _=None):
        s = float(self.speed_var.get())
        with self._lock:
            self._speed = max(0.1, s)
        self.speed_lbl.set(f"{s:.2f}x")

    def _on_loop(self):
        with self._lock:
            self._loopback = bool(self.loop_var.get())

    def _on_scrub(self, _=None):
        if self._suppress_slider:
            return
        # 사용자가 슬라이더를 움직임 → 일시정지 + 해당 프레임으로
        idx = int(float(self.frame_var.get()))
        with self._lock:
            self._idx = max(0, min(idx, self.T - 1))
            self._play = False
        self.play_var.set("▶ Play")

    def _nudge(self, d: int):
        with self._lock:
            self._play = False
            self._idx = max(0, min(self._idx + d, self.T - 1))
        self.play_var.set("▶ Play")

    def _seek(self, idx: int):
        with self._lock:
            self._play = False
            self._idx = max(0, min(idx, self.T - 1))
        self.play_var.set("▶ Play")

    def _refresh_status(self):
        with self._lock:
            i, playing, spd = self._idx, self._play, self._speed
        # 슬라이더를 재생 위치에 동기화 (콜백 억제)
        self._suppress_slider = True
        try:
            self.frame_var.set(i)
        finally:
            self._suppress_slider = False
        self.status_var.set(
            f"{'playing' if playing else 'paused'} | "
            f"frame {i + 1}/{self.T} | t={i * self.control_dt:6.2f}s | "
            f"{spd:.2f}x | maxT_h={self.h_t[i].max():.1f} "
            f"maxT_w={self.w_t[i].max():.1f} °C"
        )
        if not self._stop.is_set():
            self.root.after(80, self._refresh_status)

    def _on_close(self):
        self._stop.set()
        try:
            self.root.destroy()
        except Exception:
            pass

    # --------------------------------------------------- viewer thread
    def _render(self, viewer, idx: int):
        self.data.qpos[:] = self.qpos[idx]
        mujoco.mj_forward(self.model, self.data)
        viewer.sync()

    def _viewer_loop(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            last = -1
            while not self._stop.is_set() and viewer.is_running():
                t0 = time.time()
                with self._lock:
                    idx, playing, spd, loopb = (
                        self._idx, self._play, self._speed, self._loopback
                    )
                if idx != last:
                    self._render(viewer, idx)
                    last = idx
                if playing:
                    nxt = idx + 1
                    if nxt >= self.T:
                        nxt = 0 if loopb else self.T - 1
                        if not loopb:
                            with self._lock:
                                self._play = False
                            self.root.after(0, lambda: self.play_var.set("▶ Play"))
                    with self._lock:
                        # 사용자가 그 사이 스크럽했으면 덮어쓰지 않음
                        if self._play and self._idx == idx:
                            self._idx = nxt
                    period = self.control_dt / max(spd, 1e-3)
                else:
                    period = 0.03  # paused: 가벼운 idle
                sleep = period - (time.time() - t0)
                if sleep > 0:
                    self._stop.wait(sleep)
        self._stop.set()

    def mainloop(self):
        try:
            self.root.mainloop()
        finally:
            self._stop.set()
            self._viewer_thread.join(timeout=2.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interactive trajectory replay GUI.")
    ap.add_argument("path", help="recording dir or trajectory.npz")
    args = ap.parse_args(argv)
    ReplayGUI(_resolve(args.path)).mainloop()


if __name__ == "__main__":
    main()
