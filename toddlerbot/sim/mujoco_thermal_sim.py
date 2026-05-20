"""MuJoCoSim variant that runs the heat2torque thermal model alongside MuJoCo.

The simulator owns a `ThermalState` and a JIT-compiled `thermal_step` from
`heat2torque.envs.base.ThermalEnv`. On every physics sub-step it advances the
thermal state with the controller torque (`tau_cmd`) and, when derate is
enabled, overrides `data.ctrl` with the derated torque. The runtime mode is a
public attribute so the GUI can toggle it without locking.
"""

import os
import threading
import time
from functools import partial
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import numpy.typing as npt

from heat2torque import ThermalConfig, ThermalMode, ThermalState
from heat2torque.core.base import step_thermal
from heat2torque.core import compute_xi
from heat2torque.envs.base import ThermalEnv
from heat2torque.utils import (
    load_1st_order_param,
    load_thermal_json,
    matching_actuator_config,
)

from toddlerbot.sim import Obs
from toddlerbot.sim.mujoco_control import MotorController
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot

# Modes that the GUI is allowed to switch to. Keep this small so we only
# pay the JIT compile cost once per mode.
# Thermal/Derate 모드는 결합 2차 LPTN(=논문 최종 모델 A3-H)으로 배치 →
# USE_COUPLING 포함. (per-motor R_ch/R_ck 는 load_thermal_json 으로 적재됨)
_GUI_MODES: tuple = (
    ThermalMode.DISABLE,
    ThermalMode.USE_THERMAL | ThermalMode.MODEL_ORDER_2 | ThermalMode.USE_COUPLING,
    ThermalMode.USE_DERATE
    | ThermalMode.USE_THERMAL
    | ThermalMode.MODEL_ORDER_2
    | ThermalMode.USE_COUPLING,
)


def _resolve_q_dot_max(robot: Robot, n: int) -> npt.NDArray[np.float32]:
    vals = robot.get_joint_attrs("type", "dynamixel", "q_dot_max")
    if vals is None or len(vals) != n:
        return np.full((n,), 10.0, dtype=np.float32)
    return np.asarray(vals, dtype=np.float32)


class MuJoCoThermalSim(MuJoCoSim):
    """MuJoCoSim that drives a `ThermalEnv` alongside the physics step."""

    def __init__(
        self,
        robot: Robot,
        thermal_cfg: Optional[ThermalConfig] = None,
        thermal_mode: ThermalMode = ThermalMode.USE_THERMAL | ThermalMode.MODEL_ORDER_2,
        thermal_seed: int = 0,
        load_1st_order: bool = True,
        record: bool = False,
        record_dir: Optional[str] = None,
        ref_dir: Optional[str] = None,
        tamb_csv: Optional[str] = None,
        **mjsim_kwargs,
    ):
        super().__init__(robot, **mjsim_kwargs)

        if thermal_cfg is None:
            thermal_cfg = ThermalConfig()
        self.thermal_cfg = thermal_cfg

        actuators = matching_actuator_config(robot=robot)
        # spec-대표값 → 모터별 독립 결합 2차 LPTN 파라미터(=A3-H).
        # data/thermal_params.json 부재 시 spec 폴백(경고 후 그대로).
        try:
            actuators = load_thermal_json(actuators) or actuators
        except Exception as e:
            print(f"[MuJoCoThermalSim] load_thermal_json skipped: {e}")
        if load_1st_order:
            try:
                loaded = load_1st_order_param(actuators)
                if loaded is not None:
                    actuators = loaded
            except Exception as e:
                print(f"[MuJoCoThermalSim] load_1st_order_param skipped: {e}")

        # Thermal model is integrated once per control step (control_dt),
        # not per physics substep. Use control_dt as the integration step.
        self.thermal_env = ThermalEnv(actuators, thermal_cfg, dt=self.control_dt)
        self._thermal_dt = float(self.control_dt)
        self._hard_const_ratio = float(self.thermal_env.torque_scale_ratio)

        # Pre-compile one jitted thermal step per mode the GUI exposes.
        # `mode` and `hard_const_ratio` are baked in via partial so JAX sees
        # only `(state, tau_cmd, q_dot, q_dot_max)` as traced inputs.
        self._jit_thermal_steps: Dict[int, callable] = {}
        for m in _GUI_MODES:
            self._jit_thermal_steps[int(m)] = jax.jit(
                partial(
                    step_thermal,
                    dt=self._thermal_dt,
                    mode=int(m),
                    hard_const_ratio=self._hard_const_ratio,
                )
            )

        self.q_dot_max = _resolve_q_dot_max(robot, self.thermal_env.num_actuators)
        self._q_dot_max_jnp = jnp.asarray(self.q_dot_max)

        # 전압모델 효율계수 ξ (per-motor 상수): 25°C stall → sysID τmax
        # (robot attr "tau_max") 앵커. nominal 전기 파라미터로 산출
        # (GUI DR: V[1,1]·K_t[.9,1.1]·R_e[.9,1.1] — ξ는 고정 캘리브).
        self._xi_jnp = jnp.asarray(
            compute_xi(
                jnp.asarray(self.controller.tau_max, dtype=jnp.float32),
                self.thermal_env.nom_K_t,
                self.thermal_env.nom_R_e,
                self.thermal_env.nom_V,
            ),
            dtype=jnp.float32,
        )
        # raw PD 토크 추출용 ∞ (controller Eq.9 magnitude/속도 clip 무력화).
        # q_dot_max ≠ q_dot_tau_max 로 두어 Eq.9의 0-division(dead branch) 회피.
        self._big_np = np.full(
            (self.thermal_env.num_actuators,), 1.0e6, dtype=np.float32
        )
        self._big_qdm_np = self._big_np * 10.0

        self._thermal_lock = threading.Lock()
        self._mode = ThermalMode(int(thermal_mode))
        self._thermal_seed = int(thermal_seed)
        self._reset_pose_pending = False
        self._reset_temps_pending: Optional[int] = None
        # External flag consumed by the sim/run loop so it can re-run the
        # policy warm-up sequence (prep_duration interpolation) after a
        # pose reset. Independent of `_reset_pose_pending` because the sim
        # itself does not own the policy.
        self._policy_restart_pending = False

        # ── Trajectory recording (--record). 라이브 렌더링 안 함: qpos +
        #    thermal 만 버퍼링 → 종료 시 npz 덤프(분리 렌더용). 가볍다. ──
        self._record_enabled = bool(record)
        self._record_dir = record_dir
        self._rec_qpos: List = []
        self._rec_h: List = []
        self._rec_w: List = []
        self._rec_tau: List = []        # tau_der (적용된 derate 토크)
        self._rec_tau_cmd: List = []    # tau_cmd (컨트롤러 명령 토크, clip 전)
        self._rec_tau_max: List = []    # tau_max (현재 온도의 derate 한계)
        self._rec_step = 0
        self._record_saved = False

        # ── sim-vs-real compare-mode (opt-in: ref_dir 제공 시에만) ──
        #   ref 폴더의 real log_data.pkl 로 cold-start 정렬, log_data.pkl
        #   스키마로 기록. tamb_csv 가 있으면 unix 정렬 시변 ambient,
        #   없으면 real per-motor motor_temp[0] 를 ambient 로 유지.
        #   ref_dir 미지정 → 기존 랜덤 reset·npz 동작 완전 불변.
        self._compare = False
        self._tamb = None
        self._cold_T0 = None
        self._sim_start_unix = None
        self._rec_obs: List = []
        self._rec_ctrl_in: List = []
        self._rec_motor_ang: List = []
        if ref_dir:
            from heat2torque.eval import load_real_ref, make_tamb_provider
            ref = load_real_ref(ref_dir)
            self._ref = ref
            self._sim_start_unix = ref.start_time_unix
            self._tamb = make_tamb_provider(ref, tamb_csv)
            if self._tamb is not None:
                T0 = float(self._tamb(0.0))           # csv ambient @ run start
                self._cold_T0 = np.full(
                    self.thermal_env.num_actuators, T0, dtype=np.float32
                )
                src = f"ds18b20 csv (T0={T0:.2f}°C, time-varying a_t)"
            else:
                self._cold_T0 = np.asarray(ref.T0, dtype=np.float32)  # (30,)
                src = (f"real motor_temp[0] (per-motor "
                       f"{self._cold_T0.min():.0f}~{self._cold_T0.max():.0f}°C, "
                       f"a_t held)")
            self._compare = True
            self._record_enabled = True               # compare ⇒ pkl 기록
            print(f"[MuJoCoThermalSim] compare-mode: ref={ref_dir}\n"
                  f"  cold-start ← {src}; real {len(ref.t)} steps "
                  f"({ref.duration:.1f}s), start_unix={ref.start_time_unix:.1f}")

        rng = jax.random.PRNGKey(self._thermal_seed)
        if self._compare:
            self.thermal_state: ThermalState = self.thermal_env.reset_cold(
                self._cold_T0
            )
        else:
            # GUI(mtj_gui): 파라미터 DR 제거 — 적합된 nominal 열모델 그대로.
            self.thermal_state: ThermalState = self.thermal_env.reset_nominal(rng)

        self._warmup_thermal()

    # ------------------------------------------------------------------ utils
    def _warmup_thermal(self):
        """Compile every mode-specific jitted step so toggling is instant."""
        n = self.thermal_env.num_actuators
        zeros = jnp.zeros((n,), dtype=jnp.float32)
        for mode_int, fn in self._jit_thermal_steps.items():
            if mode_int == int(ThermalMode.DISABLE):
                continue
            try:
                ts = fn(self.thermal_state, zeros, zeros,
                        self._q_dot_max_jnp, self._xi_jnp)
                jax.block_until_ready(ts.h_t)
            except Exception as e:
                print(
                    f"[MuJoCoThermalSim] warmup failed for mode={mode_int}: {e}"
                )

    # ----------------------------------------------------------------- public
    @property
    def mode(self) -> ThermalMode:
        return self._mode

    def set_mode(self, mode: ThermalMode):
        with self._thermal_lock:
            self._mode = ThermalMode(int(mode))

    def request_reset_pose(self):
        with self._thermal_lock:
            self._reset_pose_pending = True
            # Pose reset implies the policy must restart its warm-up
            # sequence so the robot smoothly settles back into default pose
            # before the policy resumes normal stepping.
            self._policy_restart_pending = True

    def request_reset_temps(self, seed: Optional[int] = None):
        with self._thermal_lock:
            self._reset_temps_pending = (
                int(seed) if seed is not None else int(time.time_ns() & 0x7FFFFFFF)
            )

    def consume_policy_restart(self) -> bool:
        """Atomically read+clear the policy-restart flag for the run loop."""
        with self._thermal_lock:
            pending = self._policy_restart_pending
            self._policy_restart_pending = False
        return pending

    def get_thermal_snapshot(self) -> Dict[str, Dict[str, float]]:
        ts = self.thermal_state
        h_t = np.asarray(ts.h_t)
        w_t = np.asarray(ts.w_t)
        a_t = np.asarray(ts.a_t)
        tau_cmd = np.asarray(ts.tau_cmd)
        tau_der = np.asarray(ts.tau_der)
        tau_max = np.asarray(ts.tau_max)
        spec_max = np.asarray(self.thermal_env.spec_t_max)
        overheat = np.asarray(ts.overheat)

        # Current (mA): I = tau / K_t (per-motor, temperature-dependent K_t).
        # Use the demag-corrected K_t at the current winding temperature.
        K_t_nom = np.asarray(ts.K_t)
        b_demag = np.asarray(ts.b_demag)
        a_resist = np.asarray(ts.a_resist)
        R_e_nom = np.asarray(ts.R_e)
        V_arr = np.asarray(ts.V)

        delta_t = np.maximum(w_t - 25.0, 0.0)
        K_t_dyn = np.clip(K_t_nom * (1.0 - b_demag * delta_t), 1e-3, None)
        current_mA = (tau_der / K_t_dyn) * 1000.0

        # Reference stall torque from the voltage model (q̇=0):
        #   τ_ref(T) = K_t(T)·(ξ·PWM_DUTY_MAX·V) / R_e(T)
        # 25°C → = sysID τmax (ξ 앵커 정의), 100°C → R_e(T)↑·K_t(T)↓ 로 감소.
        from heat2torque.core.derate import PWM_DUTY_MAX
        xi = np.asarray(self._xi_jnp, dtype=np.float64)
        duty = float(PWM_DUTY_MAX)
        tau_max_ref = K_t_nom * (xi * duty * V_arr) / R_e_nom  # @25 °C
        dT_100 = 100.0 - 25.0
        R_e_100 = R_e_nom * (1.0 + a_resist * dT_100)
        K_t_100 = np.clip(K_t_nom * (1.0 - b_demag * dT_100), 1e-3, None)
        tau_max_ref_hot = K_t_100 * (xi * duty * V_arr) / R_e_100

        # "Clipped" detection: tau_der differs from tau_cmd within numeric eps.
        clipped = np.abs(tau_der - tau_cmd) > 1e-4

        snap: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self.robot.motor_ordering):
            snap[name] = {
                "h_t": float(h_t[i]),
                "w_t": float(w_t[i]),
                "a_t": float(a_t[i]),
                "tau_cmd": float(tau_cmd[i]),
                "tau_der": float(tau_der[i]),
                "tau_max": float(tau_max[i]),
                "tau_max_ref": float(tau_max_ref[i]),
                "tau_max_ref_hot": float(tau_max_ref_hot[i]),
                "spec_t_max": float(spec_max[i]),
                "overheat": bool(overheat[i]),
                "clipped": bool(clipped[i]),
                "current_mA": float(current_mA[i]),
            }
        return snap

    # ------------------------------------------------------------------ inner
    def _apply_pending_resets(self):
        """Drain GUI-requested resets between control steps."""
        if self._reset_pose_pending:
            try:
                self.data.qpos[:] = self.default_qpos.copy()
            except Exception:
                pass
            self.data.qvel[:] = 0.0
            self.data.act[:] = 0.0
            self.data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._reset_pose_pending = False

        if self._reset_temps_pending is not None:
            seed = int(self._reset_temps_pending)
            # GUI Reset-Temps 도 nominal(파라미터 DR 제거) 유지.
            self.thermal_state = self.thermal_env.reset_nominal(
                jax.random.PRNGKey(seed))
            self._reset_temps_pending = None

    # ------------------------------------------------------------------ step
    def step(self):
        with self._thermal_lock:
            self._apply_pending_resets()
            mode = self._mode

        mode_int = int(mode)
        thermal_active = mode_int != int(ThermalMode.DISABLE)
        derate_active = bool(mode_int & int(ThermalMode.USE_DERATE))

        # --- thermal: one integration per control step ------------------
        # Use the latest joint velocity / control torque as the
        # representative input over the upcoming control_dt window.
        if thermal_active:
            jit_fn = self._jit_thermal_steps.get(mode_int)
            if jit_fn is None:
                # Unknown mode → compile lazily and cache.
                jit_fn = jax.jit(
                    partial(
                        step_thermal,
                        dt=self._thermal_dt,
                        mode=mode_int,
                        hard_const_ratio=self._hard_const_ratio,
                    )
                )
                self._jit_thermal_steps[mode_int] = jit_fn

            q = self.data.qpos[self.q_start_idx + self.motor_indices]
            qd = self.data.qvel[self.qd_start_idx + self.motor_indices]
            # derate(전압모델) 모드: 컨트롤러는 raw PD 토크만(Eq.9 무력화),
            # 전압포화/R_e(T)/latch 는 thermal core 가 담당. 비-derate 모드는
            # 기존 Eq.9 거동 유지(baseline 불변).
            if derate_active:
                ctrl_np = np.asarray(
                    self.controller.step(
                        q, qd, self.target_motor_angles,
                        tau_max=self._big_np,
                        q_dot_tau_max=self._big_np,
                        q_dot_max=self._big_qdm_np,
                    ),
                    dtype=np.float32,
                )
            else:
                ctrl_np = np.asarray(
                    self.controller.step(q, qd, self.target_motor_angles),
                    dtype=np.float32,
                )
            qd_np = np.asarray(qd, dtype=np.float32)

            # compare-mode + tamb csv: 시변 ambient (unix 정렬, clamp).
            # csv 없으면 reset_cold 의 per-motor T0 가 그대로 유지됨.
            if self._compare and self._tamb is not None:
                ta = float(self._tamb(self._rec_step * self.control_dt))
                self.thermal_state = self.thermal_state.replace(
                    a_t=jnp.full(self.thermal_env.num_actuators, ta,
                                 dtype=jnp.float32)
                )

            # GUI/sim-vs-real: 결정적 latch 유지 — 노이즈 없는 진실 h_t 를
            # h_sensed 로 동기화(학습 경로는 get_thermal_obs 가 노이즈 갱신).
            # 이 한 줄 없으면 reset 시점 h_sensed 가 stuck → latch 영영 안 됨.
            self.thermal_state = self.thermal_state.replace(
                h_sensed=self.thermal_state.h_t
            )
            self.thermal_state = jit_fn(
                self.thermal_state,
                jnp.asarray(ctrl_np),
                jnp.asarray(qd_np),
                self._q_dot_max_jnp,
                self._xi_jnp,
            )

            if derate_active:
                # Pull tau_der to host once per control step.
                tau_der_np = np.asarray(self.thermal_state.tau_der, dtype=np.float32)

        # --- physics substeps ------------------------------------------
        for _ in range(self.n_frames):
            q = self.data.qpos[self.q_start_idx + self.motor_indices]
            qd = self.data.qvel[self.qd_start_idx + self.motor_indices]

            if thermal_active and derate_active:
                # raw PD(Eq.9 무력화)를 전압모델 전달토크 envelope(|tau_der|,
                # = ξ·K_t(T)·I_act, 포화·온도·latch 반영)로 substep clip.
                ctrl = self.controller.step(
                    q, qd, self.target_motor_angles,
                    tau_max=self._big_np,
                    q_dot_tau_max=self._big_np,
                    q_dot_max=self._big_qdm_np,
                )
                self.data.ctrl[:] = np.clip(
                    ctrl,
                    -np.abs(tau_der_np),
                    np.abs(tau_der_np),
                )
            else:
                ctrl = self.controller.step(q, qd, self.target_motor_angles)
                self.data.ctrl[:] = ctrl

            mujoco.mj_step(self.model, self.data)

        if self.visualizer is not None:
            self.visualizer.visualize(self.data)

        # --- trajectory recording (control-step rate, post-physics) ----
        if self._record_enabled:
            self._rec_qpos.append(self.data.qpos.copy())
            self._rec_h.append(np.asarray(self.thermal_state.h_t, np.float32))
            self._rec_w.append(np.asarray(self.thermal_state.w_t, np.float32))
            self._rec_tau.append(
                np.asarray(self.thermal_state.tau_der, np.float32)
            )
            # clip 정량화용: 명령 토크 + 온도 derate 한계. clip = |tau_cmd|>tau_max
            # (또는 |tau_cmd−tau_der|>eps). thermal OFF 모드에선 stale(=0) 가능.
            self._rec_tau_cmd.append(
                np.asarray(self.thermal_state.tau_cmd, np.float32)
            )
            self._rec_tau_max.append(
                np.asarray(self.thermal_state.tau_max, np.float32)
            )

            # compare-mode: real log_data.pkl 스키마용 Obs 도 수집.
            #   motor_temp = raw h_t (사용자 결정), motor_tor = current mA
            #   = τ_der / K_t(T) · 1000  (real "tor"(=mA) 관례 일치).
            if self._compare:
                ts = self.thermal_state
                h_t = np.asarray(ts.h_t, np.float64)
                w_t = np.asarray(ts.w_t, np.float64)
                K_t = np.asarray(ts.K_t, np.float64)
                b = np.asarray(ts.b_demag, np.float64)
                dT = np.maximum(w_t - 25.0, 0.0)
                K_tT = np.clip(K_t * (1.0 - b * dT), 1e-3, None)
                I_mA = (np.asarray(ts.tau_der, np.float64) / K_tT) * 1000.0
                bo = self.get_observation()  # motor_temp = raw h_t (override됨)
                elapsed = self._rec_step * float(self.control_dt)

                def _cp(a):
                    return None if a is None else np.asarray(a).copy()

                self._rec_obs.append(Obs(
                    time=float(elapsed),
                    motor_pos=_cp(bo.motor_pos),
                    motor_vel=_cp(bo.motor_vel),
                    motor_tor=I_mA.astype(np.float32),
                    lin_vel=_cp(bo.lin_vel), ang_vel=_cp(bo.ang_vel),
                    pos=_cp(bo.pos), euler=_cp(bo.euler),
                    joint_pos=_cp(bo.joint_pos), joint_vel=_cp(bo.joint_vel),
                    motor_temp=np.asarray(h_t, np.float32),
                ))
                self._rec_ctrl_in.append(
                    dict(self.target_motor_angles)
                    if isinstance(self.target_motor_angles, dict) else {}
                )
                self._rec_motor_ang.append(dict(zip(
                    self.robot.motor_ordering,
                    np.asarray(bo.motor_pos, np.float32).tolist(),
                )))

            self._rec_step += 1

    # --------------------------------------------------- recording dump
    def dump_recording(self) -> Optional[str]:
        """Write the recorded trajectory to ``<dir>/trajectory.npz`` + meta.

        분리 렌더(render_thermal_traj.py) / replay_gui 가 소비. mp4 는 만들지
        않는다(사용자 명세: trajectory 만). 빈 버퍼면 skip.
        """
        import json
        from datetime import datetime

        if self._record_saved or not self._rec_qpos:
            return None

        rec_dir = self._record_dir or os.path.join(
            "results", f"mtj_rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(rec_dir, exist_ok=True)
        traj_path = os.path.join(rec_dir, "trajectory.npz")
        np.savez_compressed(
            traj_path,
            qpos=np.asarray(self._rec_qpos, np.float32),
            h_t=np.asarray(self._rec_h, np.float32),
            w_t=np.asarray(self._rec_w, np.float32),
            tau_der=np.asarray(self._rec_tau, np.float32),
            tau_cmd=np.asarray(self._rec_tau_cmd, np.float32),
            tau_max=np.asarray(self._rec_tau_max, np.float32),
        )
        meta = {
            "robot": self.robot.name,
            "fixed_base": bool(self.fixed_base),
            "dt": float(self.dt),
            "n_frames": int(self.n_frames),
            "control_dt": float(self.control_dt),
            "nq": int(self.model.nq),
            "n_steps": int(self._rec_step),
            "motor_ordering": list(self.robot.motor_ordering),
        }
        with open(os.path.join(rec_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # compare-mode: real 과 동일 스키마 단일 log_data.pkl 추가 기록.
        # start_time_unix = real 레퍼런스값 복사(동일 T_amb/시간축 공유).
        pkl_path = None
        if self._compare and self._rec_obs:
            import pickle
            import time as _time
            pkl_path = os.path.join(rec_dir, "log_data.pkl")
            log = {
                "obs_list": self._rec_obs,
                "control_inputs_list": self._rec_ctrl_in,
                "motor_angles_list": self._rec_motor_ang,
                "cool_down_list": [],
                "ckpt_dict": {},
                "start_time_unix": float(
                    self._sim_start_unix
                    if self._sim_start_unix is not None else _time.time()
                ),
                "cool_down_start_unix": None,
                # sim-only (real 미관측 은닉상태 + derate 진단)
                "sim_w_t": np.asarray(self._rec_w, np.float32),
                "sim_tau_der": np.asarray(self._rec_tau, np.float32),
                "sim_tau_cmd": np.asarray(self._rec_tau_cmd, np.float32),
                "ref_dir": getattr(self, "_ref", None)
                and self._ref.folder,
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(log, f)

        self._record_saved = True
        print(
            f"[MuJoCoThermalSim] recorded {self._rec_step} steps → {traj_path}\n"
            + (f"  log_data.pkl : {pkl_path}\n" if pkl_path else "")
            + f"  render : python toddlerbot/policies/render_thermal_traj.py {rec_dir}\n"
            f"  replay : python toddlerbot/policies/run_thermal_replay.py {rec_dir}"
        )
        return rec_dir

    def close(self):
        """Dump the recording (if any) before releasing the visualizer."""
        try:
            if self._record_enabled:
                self.dump_recording()
        finally:
            super().close()

    # --------------------------------------------------------- get_observation
    def get_observation(self) -> Obs:
        obs = super().get_observation()
        h_t = np.asarray(self.thermal_state.h_t, dtype=np.float32)
        # The base class allocated a `motor_temp` of size 30; copy as much as fits.
        if obs.motor_temp is None or obs.motor_temp.shape[0] != h_t.shape[0]:
            obs.motor_temp = h_t.copy()
        else:
            obs.motor_temp[:] = h_t
        return obs
