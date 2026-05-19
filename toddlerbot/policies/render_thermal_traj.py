"""Offline fast renderer: recorded trajectory → mp4 (robot only).

`run_thermal_gui.py --record` 가 남긴 `trajectory.npz` + `meta.json` 을
헤드리스(EGL)로 재생하며 단일 카메라 프레임을 모아 mp4 로 저장한다.
실시간 페이싱·Tk 없음 → GPU 한계까지 최대한 빠르게. (온도 HUD 없음:
로봇만; 온도는 npz 에 보존되어 필요 시 후처리 가능.)

Examples
--------
    python toddlerbot/policies/render_thermal_traj.py results/mtj_rec_XXXX
    python toddlerbot/policies/render_thermal_traj.py rec/ --speed 3 --render-every 2
    python toddlerbot/policies/render_thermal_traj.py rec/trajectory.npz \
        --camera side --width 1280 --height 720
"""
from __future__ import annotations

import argparse
import json
import os
import time

# EGL: 헤드리스 GPU 오프스크린 (mujoco import 전에 설정해야 함)
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from toddlerbot.utils.file_utils import find_robot_file_path  # noqa: E402


def _write_mp4(path: str, frames, fps: float) -> str:
    """mp4 write — imageio(번들 ffmpeg) 우선, mediapy(시스템 ffmpeg) 폴백.

    이 환경엔 시스템 ffmpeg 가 없으나 imageio_ffmpeg 가 정적 ffmpeg 를
    번들하므로 imageio 경로가 의존성 없이 동작한다.
    """
    try:
        import imageio.v2 as imageio  # imageio_ffmpeg 플러그인 사용
        imageio.mimwrite(
            path, list(frames), fps=fps, codec="libx264",
            quality=8, macro_block_size=None,
        )
        return "imageio"
    except Exception as e_io:
        try:
            import mediapy as media
            media.write_video(path, frames, fps=fps)
            return "mediapy"
        except Exception as e_mp:
            raise RuntimeError(
                f"mp4 write failed (imageio: {e_io}; mediapy: {e_mp}). "
                f"`pip install imageio-ffmpeg` 또는 `apt install ffmpeg`."
            )


def _resolve(path: str) -> tuple[str, str, str]:
    """<dir> | <dir>/trajectory.npz → (npz, meta, base_dir)."""
    if os.path.isdir(path):
        d = path
    else:
        d = os.path.dirname(path) or "."
    npz = os.path.join(d, "trajectory.npz")
    meta = os.path.join(d, "meta.json")
    if not os.path.isfile(npz):
        raise FileNotFoundError(f"trajectory.npz not found in {d}")
    if not os.path.isfile(meta):
        raise FileNotFoundError(f"meta.json not found in {d}")
    return npz, meta, d


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Recorded trajectory → mp4 (headless).")
    ap.add_argument("path", help="recording dir or trajectory.npz")
    ap.add_argument("--out", default="", help="output mp4 (default <dir>/render.mp4)")
    ap.add_argument("--camera", default="perspective",
                    help="mujoco camera name (perspective/side/top/front)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--render-every", type=int, default=1,
                    help="render every Nth control frame (>1 = faster/smaller)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier of the mp4 (offline; unbounded)")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="override fps (default = realtime: 1/control_dt/every*speed)")
    args = ap.parse_args(argv)

    npz_path, meta_path, base_dir = _resolve(args.path)
    with open(meta_path) as f:
        meta = json.load(f)
    data_npz = np.load(npz_path)
    qpos = np.asarray(data_npz["qpos"], dtype=np.float64)  # (T, nq)
    T = qpos.shape[0]

    suffix = "_fixed_scene.xml" if meta.get("fixed_base") else "_scene.xml"
    xml_path = find_robot_file_path(meta["robot"], suffix=suffix)
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = float(meta.get("dt", model.opt.timestep))
    data = mujoco.MjData(model)
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"qpos nq={qpos.shape[1]} ≠ model.nq={model.nq} "
            f"(robot/meta mismatch?)"
        )

    every = max(1, int(args.render_every))
    control_dt = float(meta.get("control_dt", 0.02))
    fps = args.fps if args.fps > 0 else (1.0 / control_dt / every * args.speed)
    out = args.out or os.path.join(base_dir, "render.mp4")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    idxs = range(0, T, every)
    n = len(idxs)
    print(
        f"[render] {meta['robot']} | frames={n}/{T} (every={every}) | "
        f"{args.width}x{args.height} cam={args.camera} | fps={fps:.2f} | EGL"
    )
    frames = []
    t0 = time.time()
    for k, i in enumerate(idxs):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=args.camera)
        frames.append(renderer.render())
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{n}  ({(k + 1) / (time.time() - t0):.0f} fps render)")
    render_s = time.time() - t0
    renderer.close()  # EGL 컨텍스트 먼저 해제 → write 실패해도 teardown noise 없음
    backend = _write_mp4(out, frames, fps)
    dt = time.time() - t0
    print(
        f"[render] DONE → {out}  ({n} frames; render {render_s:.1f}s = "
        f"{n / max(render_s, 1e-6):.0f} fps offline; mp4 via {backend}; "
        f"video {n / fps:.1f}s @ {fps:.1f}fps)"
    )


if __name__ == "__main__":
    main()
