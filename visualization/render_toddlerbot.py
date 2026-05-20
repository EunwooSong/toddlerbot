"""Render Toddlerbot in MuJoCo on a pure-white studio background at up to 8K.

The robot description lives in
``toddlerbot/descriptions/toddlerbot`` (``toddlerbot.xml`` is the full
floating-base model, ``toddlerbot_vis.xml`` is a fixed-base visual-only
variant). Both reference a ground geom named ``floor`` through contact
``<pair>`` elements, so a scene that includes either file *must* define that
geom. This script generates a throwaway scene XML next to the description
files (so the relative mesh/``<include>`` paths keep resolving), swaps the
skybox and ground for flat white, adds soft studio lighting, then renders with
an auto-framed free camera so the whole robot stays centered regardless of
pose.

Offscreen rendering uses EGL by default (GPU, headless-safe). MuJoCo caps the
offscreen buffer at ``visual/global@offwidth``/``offheight``; those are written
into the generated scene to match the requested resolution. 8K is
7680x4320 (the default).

Usage:
    conda activate toddlerbot
    python visualization/render_toddlerbot.py
    python visualization/render_toddlerbot.py --width 3840 --height 2160 \
        --azimuth 120 --elevation -15 --out renders/hero.png
    python visualization/render_toddlerbot.py --views          # 4 angles
    python visualization/render_toddlerbot.py --turntable 60    # spin -> mp4

Output defaults to ``visualization/renders/`` next to this file.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator, List, Tuple

import numpy as np

# MUJOCO_GL must be chosen before mujoco is imported. EGL gives GPU-accelerated
# headless offscreen rendering; fall back only if the user overrode it.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DESC_DIR = os.path.join(
    THIS_DIR, "..", "toddlerbot", "descriptions", "toddlerbot"
)
DESC_DIR = os.path.normpath(DESC_DIR)

# The generated scene <include>s one of these; mesh paths inside them are
# resolved relative to the *main* model file's directory, which is why the
# temp scene has to be written into DESC_DIR.
MODEL_CHOICES = {
    "full": "toddlerbot.xml",  # floating base, stands on the floor
    "vis": "toddlerbot_vis.xml",  # fixed base, visual meshes only
}

SCENE_TEMPLATE = """<mujoco model="toddlerbot_white">
  <include file="{include}"/>
  <statistic center="0 0 0.18" extent="0.6"/>
  <visual>
    <global offwidth="{w}" offheight="{h}" ellipsoidinertia="false"/>
    <quality shadowsize="{shadowsize}" offsamples="{offsamples}"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <map shadowclip="1.2" znear="0.01" zfar="50"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="{bg}" rgb2="{bg}"
             width="8" height="8"/>
    <texture type="2d" name="white_ground" builtin="flat"
             rgb1="{bg}" rgb2="{bg}" width="8" height="8"/>
    <material name="white_ground" texture="white_ground" texuniform="true"
              reflectance="0.0" specular="0.0" shininess="0.0"/>
  </asset>
  <worldbody>
    <light pos="1.2 -1.0 2.6" dir="-0.45 0.38 -1" diffuse="0.75 0.75 0.75"
           specular="0.25 0.25 0.25"/>
    <light pos="-1.4 -1.2 2.0" dir="0.5 0.45 -1" directional="true"
           diffuse="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="white_ground"
          pos="0 0 0" group="{floor_group}"/>
  </worldbody>
</mujoco>
"""


@contextmanager
def temp_scene(
    model_xml: str,
    width: int,
    height: int,
    shadowsize: int,
    offsamples: int,
    bg: str,
    show_floor: bool,
) -> Iterator[str]:
    """Write the white-background scene next to the descriptions, then clean up.

    A hidden geom group keeps the floor available for the contact ``<pair>``
    references while keeping it out of the render when ``--no-floor`` is set.
    """
    xml = SCENE_TEMPLATE.format(
        include=model_xml,
        w=width,
        h=height,
        shadowsize=shadowsize,
        offsamples=offsamples,
        bg=bg,
        floor_group="0" if show_floor else "3",
    )
    path = os.path.join(DESC_DIR, f"__white_scene_{os.getpid()}.xml")
    with open(path, "w") as f:
        f.write(xml)
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def robot_bbox(model: mujoco.MjModel, data: mujoco.MjData) -> Tuple[np.ndarray, float]:
    """Center and diagonal of the robot's geoms (the floor is excluded)."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    pts = np.array(
        [data.geom_xpos[i] for i in range(model.ngeom) if i != floor_id]
    )
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo))


def motor_order() -> List[str]:
    """Canonical motor ordering = key order of ``config_motors.json``.

    This matches ``Robot.motor_ordering`` (``toddlerbot/sim/robot.py``), so a
    plain integer means the same motor here as everywhere else in the project.
    Indexing is 0-based: motor 0 is ``neck_yaw_drive``, motor 7 is
    ``left_knee_act``.
    """
    cfg = os.path.join(DESC_DIR, "config_motors.json")
    with open(cfg) as f:
        return list(json.load(f).keys())


def highlight_motor(
    model: mujoco.MjModel, motor: str, rgba: List[float]
) -> Tuple[str, List[str]]:
    """Recolor the mesh(es) on the body driven by ``motor``'s actuator.

    Each motor's actuator joint lives on its own body that carries that
    motor's distinct visual mesh (e.g. ``left_knee_act`` -> body
    ``xm430_plate`` -> ``left_xm430_plate_visual``). The bulky neighbouring
    housings are separate meshes on other bodies, so per-motor recoloring is
    limited to that motor's own body geoms — which is exactly the mesh
    "connected to" the motor. Returns the body name and the meshes recolored.
    """
    order = motor_order()
    if motor.lstrip("-").isdigit():
        idx = int(motor)
        if not -len(order) <= idx < len(order):
            raise ValueError(
                f"motor index {idx} out of range 0..{len(order) - 1}"
            )
        motor = order[idx]

    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor)
    if aid < 0:
        raise ValueError(
            f"motor '{motor}' not found. Known motors:\n  "
            + ", ".join(order)
        )
    jid = int(model.actuator_trnid[aid, 0])
    bid = int(model.jnt_bodyid[jid])
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)

    meshes: List[str] = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid:
            continue
        model.geom_rgba[g] = rgba
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            meshes.append(
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]
                )
            )
    return bname, meshes


def make_camera(
    center: np.ndarray, diag: float, azimuth: float, elevation: float, zoom: float
) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = diag * zoom
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_one(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam: mujoco.MjvCamera,
) -> np.ndarray:
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def save(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.imwrite(path, img)
    kb = os.path.getsize(path) // 1024
    print(f"  saved {path}  ({img.shape[1]}x{img.shape[0]}, {kb} KB)")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render Toddlerbot on a white background at high resolution."
    )
    p.add_argument("--width", type=int, default=7680, help="pixels (default 8K)")
    p.add_argument("--height", type=int, default=4320, help="pixels (default 8K)")
    p.add_argument("--azimuth", type=float, default=140.0)
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument(
        "--zoom",
        type=float,
        default=1.25,
        help="camera distance = robot-diagonal * zoom (larger = farther)",
    )
    p.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES),
        default="full",
        help="'full' floating-base or 'vis' fixed-base visual-only",
    )
    p.add_argument(
        "--keyframe",
        default="home",
        help="keyframe name to pose the robot (use '' to skip)",
    )
    p.add_argument(
        "--highlight",
        default=None,
        metavar="MOTOR",
        help="recolor a motor's mesh; motor name or 0-based index "
        "(e.g. 7 == left_knee_act)",
    )
    p.add_argument(
        "--highlight-color",
        default="1 0 0 1",
        help='RGBA 0-1 for --highlight (default red "1 0 0 1")',
    )
    p.add_argument("--background", default="1 1 1", help='RGB 0-1, e.g. "1 1 1"')
    p.add_argument("--shadowsize", type=int, default=8192)
    p.add_argument(
        "--samples", type=int, default=8, help="MSAA offsamples (anti-aliasing)"
    )
    p.add_argument(
        "--no-floor",
        action="store_true",
        help="hide the ground plane and its drop shadow",
    )
    p.add_argument(
        "--views",
        action="store_true",
        help="render 4 preset angles (front, 3/4, side, back)",
    )
    p.add_argument(
        "--turntable",
        type=int,
        metavar="N",
        default=0,
        help="render N frames spinning 360 deg and write an mp4",
    )
    p.add_argument(
        "--fps", type=int, default=30, help="turntable frames per second"
    )
    p.add_argument(
        "--out",
        default=os.path.join(THIS_DIR, "renders", "toddlerbot_8k.png"),
        help="output image path (or mp4 with --turntable)",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    model_xml = MODEL_CHOICES[args.model]

    if not os.path.isfile(os.path.join(DESC_DIR, model_xml)):
        print(f"error: {model_xml} not found in {DESC_DIR}", file=sys.stderr)
        return 1

    print(
        f"MUJOCO_GL={os.environ['MUJOCO_GL']}  model={model_xml}  "
        f"resolution={args.width}x{args.height}"
    )

    renderer = None
    with temp_scene(
        model_xml,
        args.width,
        args.height,
        args.shadowsize,
        args.samples,
        args.background,
        show_floor=not args.no_floor,
    ) as scene_path:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)

        if args.keyframe:
            kf = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe
            )
            if kf >= 0:
                mujoco.mj_resetDataKeyframe(model, data, kf)
            else:
                print(f"  warning: keyframe '{args.keyframe}' not found")
        mujoco.mj_forward(model, data)

        if args.highlight is not None:
            rgba = [float(v) for v in args.highlight_color.split()]
            if len(rgba) != 4:
                print("error: --highlight-color needs 4 values", file=sys.stderr)
                return 1
            try:
                bname, meshes = highlight_motor(model, args.highlight, rgba)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(
                f"  highlight: motor '{args.highlight}' -> body '{bname}' "
                f"-> mesh {meshes} colored {rgba}"
            )

        center, diag = robot_bbox(model, data)
        try:
            renderer = mujoco.Renderer(model, args.height, args.width)
        except Exception as exc:  # noqa: BLE001
            print(
                f"error: could not create a {args.width}x{args.height} "
                f"renderer ({exc}). Try a smaller --width/--height.",
                file=sys.stderr,
            )
            return 1

        t0 = time.time()
        if args.turntable > 0:
            out = args.out
            if out.lower().endswith(".png"):
                out = os.path.splitext(out)[0] + ".mp4"
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            print(f"rendering {args.turntable} turntable frames -> {out}")
            with imageio.get_writer(
                out, fps=args.fps, macro_block_size=None
            ) as writer:
                for i in range(args.turntable):
                    az = args.azimuth + 360.0 * i / args.turntable
                    cam = make_camera(
                        center, diag, az, args.elevation, args.zoom
                    )
                    writer.append_data(
                        render_one(renderer, model, data, cam)
                    )
                    print(f"  frame {i + 1}/{args.turntable}", end="\r")
            kb = os.path.getsize(out) // 1024
            print(f"\n  saved {out}  ({kb} KB)")
        elif args.views:
            presets = {
                "front": (90.0, args.elevation),
                "three_quarter": (140.0, args.elevation),
                "side": (180.0, args.elevation),
                "back": (270.0, args.elevation),
            }
            base, ext = os.path.splitext(args.out)
            if ext.lower() != ".png":
                ext = ".png"
            for name, (az, el) in presets.items():
                cam = make_camera(center, diag, az, el, args.zoom)
                save(
                    render_one(renderer, model, data, cam),
                    f"{base}_{name}{ext}",
                )
        else:
            cam = make_camera(
                center, diag, args.azimuth, args.elevation, args.zoom
            )
            save(render_one(renderer, model, data, cam), args.out)

        print(f"done in {time.time() - t0:.1f}s")

    # Close the GL context while it is still valid; otherwise PyOpenGL prints
    # noisy "Exception ignored in __del__" EGL errors at interpreter shutdown.
    if renderer is not None:
        renderer.close()
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
