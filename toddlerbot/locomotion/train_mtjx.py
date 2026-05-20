import os

os.environ["USE_JAX"] = "true"
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# ── JAX/XLA GPU 메모리: 분할 단편화 방지(MEM_FRACTION) ─────────────────
#  과거: PREALLOCATE=false 로 *증분 할당* → 단편화 누적 → cuSolver 가
#  큰 contiguous 행렬 필요할 때 "INTERNAL: cuSolver internal error" 로
#  죽음(brax PPO 학습 epoch 2+ 에서 재현됨, 2026-05-20 ).
#  해결: PREALLOCATE=true(JAX 기본) + MEM_FRACTION 으로 **시작 시 1회
#  큰 덩어리** 선점 → 단편화 0, cuSolver INTERNAL 사라짐.
#  값 산정: 16GB·0.75 ≈ 12GB(사용자 관측 작업셋과 일치). VRAM 다른
#  머신은 셸에서 XLA_PYTHON_CLIENT_MEM_FRACTION 으로 override.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import argparse
import functools
import importlib
import json
import pkgutil
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import gin
import jax
import jax.numpy as jnp
import mediapy as media
import mujoco
import numpy as np
import numpy.typing as npt
import optax
from brax import base
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from moviepy.editor import VideoFileClip, clips_array
from orbax import checkpoint as ocp
from tqdm import tqdm

import wandb
from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv, get_env_class
from toddlerbot.locomotion.ppo_config import PPOConfig
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.misc_utils import dataclass2dict, parse_value

# CUSTOM ENV
from heat2torque.config import ThermalConfig
from heat2torque.agent import TMJXConfig

# CUSTOM ENV Wrapper
from heat2torque.envs.wrapper import ThermalCurriculumWrapper

# Add matplot
import matplotlib.pyplot as plt # 1. Matplotlib 임포트

jax.config.update("jax_default_matmul_precision", 'high')

OUTPUT_DIR = "results"


def dynamic_import_envs(env_package: str):
    """Imports all modules from a specified package.

    This function dynamically imports all modules within a given package, allowing their contents to be accessed programmatically. It is useful for loading environment configurations or plugins from a specified package directory.

    Args:
        env_package (str): The name of the package from which to import all modules.
    """
    package = importlib.import_module(env_package)
    package_path = package.__path__

    # Iterate over all modules in the given package directory
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"{env_package}.{module_name}"
        importlib.import_module(full_module_name)


# Call this to import all policies dynamically
dynamic_import_envs("toddlerbot.locomotion")

# import heat2torque envs
dynamic_import_envs("heat2torque.agent")

def render_video(
    env: MJXEnv,
    rollout: List[Any],
    run_name: str,
    render_every: int = 2,
    height: int = 360,
    width: int = 640,
):
    """Renders and saves a video of the environment from multiple camera angles.

    Args:
        env (MJXEnv): The environment to render.
        rollout (List[Any]): A list of environment states or actions to render.
        run_name (str): The name of the run, used to organize output files.
        render_every (int, optional): Interval at which frames are rendered from the rollout. Defaults to 2.
        height (int, optional): The height of the rendered video frames. Defaults to 360.
        width (int, optional): The width of the rendered video frames. Defaults to 640.

    Creates:
        A video file for each camera angle ('perspective', 'side', 'top', 'front') and a final concatenated video in a 2x2 grid layout, saved in the 'results' directory under the specified run name.
    """
    # Define paths for each camera's video
    video_paths: List[str] = []

    # Render and save videos for each camera
    for camera in ["perspective", "side", "top", "front"]:
        video_path = os.path.join(OUTPUT_DIR, run_name, f"{camera}.mp4")
        media.write_video(
            video_path,
            env.render(
                rollout[::render_every],
                height=height,
                width=width,
                camera=camera,
                eval=True,
            ),
            fps=1.0 / env.dt / render_every,
        )
        video_paths.append(video_path)

    # Load the video clips using moviepy
    clips = [VideoFileClip(path) for path in video_paths]
    # Arrange the clips in a 2x2 grid
    final_video = clips_array([[clips[0], clips[1]], [clips[2], clips[3]]])
    # Save the final concatenated video
    final_video.write_videofile(os.path.join(OUTPUT_DIR, run_name, "eval.mp4"))


def log_metrics(
    metrics: Dict[str, Any],
    time_elapsed: float,
    num_steps: int = -1,
    num_total_steps: int = -1,
    width: int = 80,
    pad: int = 35,
):
    log_data: Dict[str, Any] = {"time_elapsed": time_elapsed}
    log_string = f"""{"#" * width}\n"""
    
    if num_steps >= 0 and num_total_steps > 0:
        log_data["num_steps"] = num_steps
        title = f" \033[1m Learning steps {num_steps}/{num_total_steps} \033[0m "
        log_string += f"""{title.center(width, " ")}\n"""

    # --- 1. 일반 메트릭 처리 (상단 리스트) ---
    for key, value in metrics.items():
        if "std" in key or "temp_" in key:  # 온도 지표는 상단 루프에서 제외
            continue

        words = key.split("/")
        if words[0].startswith("eval"):
            if words[1].startswith("episode") and "reward" not in words[1]:
                metric_name = "rew_" + words[1].replace("episode_", "")
            else:
                metric_name = words[1]
        else:
            metric_name = "_".join(words)

        log_data[metric_name] = value
        if "episode_reward" not in metric_name and "avg_episode_length" not in metric_name:
            log_string += f"""{f"{metric_name}:":>{pad}} {value:.4f}\n"""

    log_string += f"""{"-" * width}\n"""
    log_string += f"""{"Time elapsed:":>{pad}} {time_elapsed:.1f}\n"""
    
    if "eval/episode_reward" in metrics:
        log_string += f"""{"Mean reward:":>{pad}} {metrics["eval/episode_reward"]:.3f}\n"""
    
    # --- 2. 온도 지표 복원 및 요약 출력 (하단 섹션) ---
    # eval_metrics (jnp.where(done, T, 0.0)) 패턴은 eval unroll(250 step)에서
    # eval_done(reset_steps=5000)이 발동하지 않아 넘어지지 않은 env는 0.0이 되어
    # 평균값이 희석됨. 대신 매 step 기록되는 heat_metrics 키(housing_avg 등)를 사용:
    #   eval/episode_housing_avg = sum(T over episode) → / avg_len = 에피소드 평균 온도
    avg_len = max(metrics.get("eval/avg_episode_length", 1.0), 1.0)

    def get_avg_temp(key):
        """heat_metrics 키로 eval/episode 누적값을 에피소드 평균 온도로 변환."""
        full_key = f"eval/episode_{key}"
        val = metrics.get(full_key)
        if val is None:
            return None
        return val / avg_len  # sum → 에피소드 평균

    # 출력할 지표 정의 (이름, heat_metrics 키, 포맷)
    temp_metrics = [
        ("Core Temp (Avg/Max)", "core_avg", "core_max", ".2f"),
        ("Housing Temp (Avg/Max)", "housing_avg", "housing_max", ".2f"),
        ("Torque Derate(MSE)", "torque_derate(MSE)", None, ".4f"),
        ("Overheat Penalty", "overheat_penalty", None, ".4f"),
    ]

    log_string += f"""{"[ Thermal Status (Ep. Average) ]".center(width)}\n"""

    for label, avg_k, max_k, fmt in temp_metrics:
        v_avg = get_avg_temp(avg_k)
        if v_avg is not None:
            if max_k:
                v_max = get_avg_temp(max_k)
                log_string += f"""{f"{label}:":>{pad}} {v_avg:{fmt}} / {v_max:{fmt}}\n"""
            else:
                log_string += f"""{f"{label}:":>{pad}} {v_avg:{fmt}}\n"""

    # --- 2b. desync-강건 온도 로그 (temp_th_*) → 깔끔한 thermal/* wandb 키 ---
    #  brax: eval/episode_<k> = Σ(ep). ÷avg_len → per-step 평균 = 비율/CDF/
    #  시간평균(비동기 persist env 에서도 의미 보존; batch-mean 스미어 해소).
    #  w_gt_* = 권선온도 분포 CDF, derate_*/overheat_fall = 행동률,
    #  block_peak_w = reseed 블록 peak, hot/* = 커리큘럼 hot 구간 조건부.
    _th = {
        "wmax": "temp_th_wmax", "hmax": "temp_th_hmax",
        "amb": "temp_th_amb", "block_peak_w": "temp_th_block_peak_w",
        "derate_frac": "temp_th_derate_frac",
        "derate_sev": "temp_th_derate_sev",
        "overheat_fall": "temp_th_overheat_fall",
        "w_gt_60": "temp_th_w_gt_60", "w_gt_75": "temp_th_w_gt_75",
        "w_gt_90": "temp_th_w_gt_90", "w_gt_105": "temp_th_w_gt_105",
        "w_gt_120": "temp_th_w_gt_120", "h_ge_spec": "temp_th_h_ge_spec",
        "progress": "temp_th_progress",
    }
    th_vals = {}
    for short, k in _th.items():
        v = get_avg_temp(k)
        if v is not None:
            th_vals[short] = float(v)
            log_data[f"thermal/{short}"] = float(v)
    # E: hot 구간 조건부 비율 = E[hot·1[w>90]] / E[hot]
    _hot = get_avg_temp("temp_th_hotmask")
    _hotw = get_avg_temp("temp_th_hot_w_gt_90")
    if _hot is not None and _hotw is not None and float(_hot) > 1e-6:
        log_data["thermal/hot_w_gt_90"] = float(_hotw) / float(_hot)

    # --- split_eval 버킷 진단 (eval 전용 — train 에선 cold_*만 0/0=skip) ---
    #  reset 직후 절반 cold(ambient) + 절반 hot([40,60]+offset). 각 메트릭은
    #  state.metrics 에 x·mask 로 푸시되고 brax 가 episode-sum → /avg_len 후
    #  Σ(x·mask)/Σ(mask) = 조건부 평균.
    #    eval/hot_reward, eval/cold_reward     : 정책 보상 (총합)
    #    eval/hot_lin_vel, eval/cold_lin_vel   : 보행 속도 보상 (freeze 진단 핵심)
    #    eval/hot_fall,   eval/cold_fall       : 낙상률 (per-step)
    #    eval/hot_wmax,   eval/cold_wmax       : 권선 peak (열 노출도)
    #    eval/hot_derate_sev, eval/cold_derate_sev : 토크 derate 정도
    #    eval/hot_frac                         : 실측 hot 버킷 비율 (균형 sanity)
    def _cond_mean(num_key: str, den_key: str, wandb_key: str):
        n = get_avg_temp(num_key); d = get_avg_temp(den_key)
        if n is not None and d is not None and float(d) > 1e-6:
            log_data[wandb_key] = float(n) / float(d)

    for stat in ('reward', 'lin_vel', 'fall', 'wmax', 'derate_sev'):
        _cond_mean(f'temp_th_b_hot_{stat}',  'temp_th_b_hot_mask',
                   f'eval/hot_{stat}')
        _cond_mean(f'temp_th_b_cold_{stat}', 'temp_th_b_cold_mask',
                   f'eval/cold_{stat}')
    _hf = get_avg_temp('temp_th_b_hot_mask')
    if _hf is not None:
        log_data['eval/hot_frac'] = float(_hf)

    if th_vals:
        log_string += f"""{"[ Thermal (desync-robust) ]".center(width)}\n"""
        if "wmax" in th_vals:
            log_string += (
                f"""{f"Winding peak (mean/block):":>{pad}} """
                f"""{th_vals.get('wmax', float('nan')):.1f} / """
                f"""{th_vals.get('block_peak_w', float('nan')):.1f} °C\n""")
        wcdf = " ".join(
            f"{t}:{th_vals.get('w_gt_'+t, 0.0):.2f}"
            for t in ("60", "75", "90", "105", "120") if 'w_gt_'+t in th_vals
        )
        if wcdf:
            log_string += f"""{f"Frac w_t> (CDF):":>{pad}} {wcdf}\n"""
        if "derate_frac" in th_vals:
            log_string += (
                f"""{f"Derate frac/sev:":>{pad}} """
                f"""{th_vals['derate_frac']:.3f} / """
                f"""{th_vals.get('derate_sev', float('nan')):.3f}\n""")
        if "overheat_fall" in th_vals:
            log_string += (
                f"""{f"Overheat-fall rate:":>{pad}} """
                f"""{th_vals['overheat_fall']:.4f}\n""")

    # --- 3. 성능 지표 마무리 ---
    if num_steps > 0 and num_total_steps > 0:
        log_string += f"""{"-" * width}\n"""
        log_string += f"""{"Computation:":>{pad}} {(num_steps / time_elapsed):.1f} steps/s\n"""
        log_string += f"""{"ETA:":>{pad}} {(time_elapsed / num_steps) * (num_total_steps - num_steps):.1f}s\n"""

    print(log_string)
    return log_data


def get_body_mass_attr_range(
    robot: Robot,
    body_mass_range: List[float],
    ee_mass_range: List[float],
    other_mass_range: List[float],
    num_envs: int,
):
    """Generates a range of body mass attributes for a robot across multiple environments.

    This function modifies the body mass and inertia of a robot model based on specified
    ranges for different body parts (torso, end-effector, and others) and returns a dictionary
    containing the updated attributes for each environment.

    Args:
        robot (Robot): The robot object containing configuration and name.
        body_mass_range (List[float]): The range of mass deltas for the torso.
        ee_mass_range (List[float]): The range of mass deltas for the end-effector.
        other_mass_range (List[float]): The range of mass deltas for other body parts.
        num_envs (int): The number of environments to generate.

    Returns:
        Dict[str, jax.Array | npt.NDArray[np.float32]]: A dictionary with keys representing
        different body mass attributes and values as JAX arrays or NumPy arrays containing
        the attribute values across all environments.
    """
    xml_path: str = find_robot_file_path(robot.name, suffix="_scene.xml")
    torso_name = "torso"
    ee_name = robot.config["general"]["ee_name"]

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    body_mass = model.body_mass.copy()
    body_inertia = model.body_inertia.copy()

    body_mass_delta_list = np.linspace(body_mass_range[0], body_mass_range[1], num_envs)
    ee_mass_delta_list = np.linspace(ee_mass_range[0], ee_mass_range[1], num_envs)
    other_mass_delta_list = np.linspace(
        other_mass_range[0], other_mass_range[1], num_envs
    )
    # Randomize the order of the body mass deltas
    body_mass_delta_list = np.random.permutation(body_mass_delta_list)
    ee_mass_delta_list = np.random.permutation(ee_mass_delta_list)
    other_mass_delta_list = np.random.permutation(other_mass_delta_list)

    # Create lists to store attributes for all environments
    body_mass_list = []
    body_inertia_list = []
    actuator_acc0_list = []
    body_invweight0_list = []
    body_subtreemass_list = []
    dof_M0_list = []
    dof_invweight0_list = []
    tendon_invweight0_list = []
    for body_mass_delta, ee_mass_delta, other_mass_delta in zip(
        body_mass_delta_list, ee_mass_delta_list, other_mass_delta_list
    ):
        # Update body mass and inertia in the model
        for i in range(model.nbody):
            body_name = model.body(i).name

            if body_mass[i] < 1e-6 or body_mass[i] < other_mass_range[1]:
                continue

            if torso_name in body_name:
                mass_delta = body_mass_delta
            elif ee_name in body_name:
                mass_delta = ee_mass_delta
            else:
                mass_delta = other_mass_delta

            model.body(body_name).mass = body_mass[i] + mass_delta
            model.body(body_name).inertia = (
                (body_mass[i] + mass_delta) / body_mass[i] * body_inertia[i]
            )

        mujoco.mj_setConst(model, data)

        # Append the values to corresponding lists
        body_mass_list.append(jnp.array(model.body_mass))
        body_inertia_list.append(jnp.array(model.body_inertia))
        actuator_acc0_list.append(np.array(model.actuator_acc0))
        body_invweight0_list.append(jnp.array(model.body_invweight0))
        body_subtreemass_list.append(jnp.array(model.body_subtreemass))
        dof_M0_list.append(jnp.array(model.dof_M0))
        dof_invweight0_list.append(jnp.array(model.dof_invweight0))
        tendon_invweight0_list.append(jnp.array(model.tendon_invweight0))

    # Return a dictionary where each key has a JAX array of all values across environments
    # NOTE(2026-05-21): actuator_acc0 reverted to jnp.stack (was np.stack — bug).
    # numpy 였을 때 domain_randomize() 의 body_mass_attr 루프
    # (`isinstance(v, jnp.ndarray)`) 가 이 키를 스킵 → in_axes_dict 등록 안 됨
    # → vmap 시 in_axes=None 으로 (num_envs, nu) 가 안 벗겨져 mjx 가 scan
    # 입력 길이 = num_envs(1024) vs 기대 nu(30) mismatch 로 IndexError 발생
    # (mjx≥3.2.x strict check). jnp 로 통일하면 다른 키들과 동일 경로 → fix.
    body_mass_attr_range: Dict[str, jax.Array | npt.NDArray[np.float32]] = {
        "body_mass": jnp.stack(body_mass_list),
        "body_inertia": jnp.stack(body_inertia_list),
        "actuator_acc0": jnp.stack(actuator_acc0_list),
        "body_invweight0": jnp.stack(body_invweight0_list),
        "body_subtreemass": jnp.stack(body_subtreemass_list),
        "dof_M0": jnp.stack(dof_M0_list),
        "dof_invweight0": jnp.stack(dof_invweight0_list),
        "tendon_invweight0": jnp.stack(tendon_invweight0_list),
    }

    return body_mass_attr_range


def domain_randomize(
    sys: base.System,
    rng: jax.Array,
    friction_range: List[float],
    damping_range: List[float],
    armature_range: List[float],
    frictionloss_range: List[float],
    body_mass_attr_range: Optional[Dict[str, jax.Array | npt.NDArray[np.float32]]],
) -> Tuple[base.System, base.System]:
    """Randomizes the physical parameters of a system within specified ranges.

    Args:
        sys (base.System): The system whose parameters are to be randomized.
        rng (jax.Array): Random number generator state.
        friction_range (List[float]): Range for randomizing friction values.
        damping_range (List[float]): Range for randomizing damping values.
        armature_range (List[float]): Range for randomizing armature values.
        frictionloss_range (List[float]): Range for randomizing friction loss values.
        body_mass_attr_range (Optional[Dict[str, jax.Array | npt.NDArray[np.float32]]]): Optional dictionary specifying ranges for body mass attributes.

    Returns:
        Tuple[base.System, base.System]: A tuple containing the randomized system and the in_axes configuration for JAX transformations.
    """

    @jax.vmap
    def rand(rng: jax.Array):
        _, rng_friction, rng_damping, rng_armature, rng_frictionloss = jax.random.split(
            rng, 5
        )

        friction = jax.random.uniform(
            rng_friction, (1,), minval=friction_range[0], maxval=friction_range[1]
        )
        friction = sys.geom_friction.at[:, 0].set(friction)

        damping = (
            jax.random.uniform(
                rng_damping, (sys.nv,), minval=damping_range[0], maxval=damping_range[1]
            )
            * sys.dof_damping
        )

        armature = (
            jax.random.uniform(
                rng_armature,
                (sys.nv,),
                minval=armature_range[0],
                maxval=armature_range[1],
            )
            * sys.dof_armature
        )

        frictionloss = (
            jax.random.uniform(
                rng_frictionloss,
                (sys.nv,),
                minval=frictionloss_range[0],
                maxval=frictionloss_range[1],
            )
            * sys.dof_frictionloss
        )
        return friction, damping, armature, frictionloss

    friction, damping, armature, frictionloss = rand(rng)

    body_mass_attr = {}
    if body_mass_attr_range is not None:
        for k, v in body_mass_attr_range.items():
            if isinstance(v, jnp.ndarray):
                body_mass_attr[k] = v[: rng.shape[0]]

    in_axes_dict = {
        "geom_friction": 0,
        "dof_damping": 0,
        "dof_armature": 0,
        "dof_frictionloss": 0,
        **{key: 0 for key in body_mass_attr.keys()},
    }

    sys_dict = {
        "geom_friction": friction,
        "dof_damping": damping,
        "dof_armature": armature,
        "dof_frictionloss": frictionloss,
        **body_mass_attr,
    }

    if body_mass_attr_range is not None:
        sys = sys.replace(
            actuator_acc0=body_mass_attr_range["actuator_acc0"][: rng.shape[0]]
        )

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(in_axes_dict)
    sys = sys.tree_replace(sys_dict)

    return sys, in_axes


def train(
    env: MJXEnv,
    eval_env: MJXEnv,
    make_networks_factory: Any,
    train_cfg: PPOConfig,
    run_name: str,
    restore_path: str,
):
    """Trains a reinforcement learning agent using the Proximal Policy Optimization (PPO) algorithm.

    This function sets up the training environment, initializes configurations, and manages the training process, including saving configurations, logging metrics, and handling checkpoints.

    Args:
        env (MJXEnv): The training environment.
        eval_env (MJXEnv): The evaluation environment.
        make_networks_factory (Any): Factory function to create neural network models.
        train_cfg (PPOConfig): Configuration settings for the PPO training process.
        run_name (str): Name of the training run, used for organizing results.
        restore_path (str): Path to restore a previous checkpoint, if any.
    """
    # wrapper 감싸기
    #env = ThermalCurriculumWrapper(env, train_cfg.num_timesteps, train_cfg.num_envs, TMJXConfig())

    # 사용자 정의 cfg. 이 환경은 모든 정책 학습시 동일하게 적용됨!!
    # e = ThermalConfig.EvalConfig()
    # eval_cfg = TMJXConfig()
    # eval_cfg.thermal_cfg.curriculum.threshold_ratio = 1.0
    # eval_cfg.thermal_cfg.curriculum.init_hot = e.temp_range
    # eval_cfg.thermal_cfg.curriculum.init_cold = e.temp_range
    # eval_cfg.thermal_cfg.curriculum.use_ep_sampling = e.use_ep_sampling
    # eval_cfg.thermal_cfg.curriculum.offset = e.offset
    
    # eval_cfg.thermal_cfg.domain_rand.temp_range = e.temp_range
    # eval_cfg.thermal_cfg.env.mode = e.mode
    # eval_cfg.thermal_cfg.env.use_w_offset = e.use_w_offset
    # eval_cfg.thermal_cfg.env.use_rand_w = e.use_rand_w
    # eval_cfg.thermal_cfg.env.offset = e.offset
    # eval_cfg.thermal_cfg.reward.safety_penalty = e.safety_penalty

    # eval_env = ThermalCurriculumWrapper(eval_env, train_cfg.num_timesteps, train_cfg.num_envs, eval_cfg)

    exp_folder_path = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(exp_folder_path, exist_ok=True)

    restore_checkpoint_path = (
        os.path.abspath(restore_path) if len(restore_path) > 0 else None
    )

    # Save train config to a file and print it
    train_config_dict = dataclass2dict(train_cfg)  # Convert dataclass to dictionary
    with open(os.path.join(exp_folder_path, "train_config.json"), "w") as f:
        json.dump(train_config_dict, f, indent=4)

    # Print the train config
    print("Train Config:")
    print(json.dumps(train_config_dict, indent=4))  # Pretty-print the config

    # Save env config to a file and print it
    env_config_dict = dataclass2dict(env.cfg)  # Convert dataclass to dictionary
    # thermal_cfg는 dataclass field가 아닌 __init__ 속성이므로 asdict()에 포함되지 않음
    if hasattr(env.cfg, 'thermal_cfg'):
        env_config_dict['thermal_cfg'] = dataclass2dict(env.cfg.thermal_cfg)
    with open(os.path.join(exp_folder_path, "env_config.json"), "w") as f:
        json.dump(env_config_dict, f, indent=4)

    # Print the env config
    print("Env Config:")
    print(json.dumps(env_config_dict, indent=4))  # Pretty-print the config

    # Copy the Python scripts
    shutil.copytree(
        os.path.join("toddlerbot", "locomotion"),
        os.path.join(exp_folder_path, "locomotion"),
    )

    wandb.init(
        project="ToddlerBot",
        sync_tensorboard=True,
        name=run_name,
        config=dataclass2dict(train_cfg),
    )

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    def policy_params_fn(current_step: int, make_policy: Any, params: Any):
        # save checkpoints
        save_args = orbax_utils.save_args_from_target(params)
        path = os.path.abspath(os.path.join(exp_folder_path, f"{current_step}"))
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        policy_path = os.path.join(path, "policy")
        model.save_params(policy_path, params)
        # thermal 진단은 brax 변환과 충돌(tracer leak) → in-train 제거.
        # 저장된 체크포인트로 *오프라인* 검증: heat2torque/eval/thermal_diag.py
        # (run/thermal_diag.sh). 학습엔 무영향.

    learning_rate_schedule_fn = optax.cosine_decay_schedule(
        train_cfg.learning_rate,
        train_cfg.decay_steps,
        train_cfg.alpha,
    )

    domain_randomize_fn = None
    if env.add_domain_rand:
        body_mass_attr_range = None
        if not env.fixed_base:
            body_mass_attr_range = get_body_mass_attr_range(
                env.robot,
                env.cfg.domain_rand.body_mass_range,
                env.cfg.domain_rand.ee_mass_range,
                env.cfg.domain_rand.other_mass_range,
                train_cfg.num_envs,
            )

        domain_randomize_fn = functools.partial(
            domain_randomize,
            friction_range=env.cfg.domain_rand.friction_range,
            damping_range=env.cfg.domain_rand.damping_range,
            armature_range=env.cfg.domain_rand.armature_range,
            frictionloss_range=env.cfg.domain_rand.frictionloss_range,
            body_mass_attr_range=body_mass_attr_range,
        )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=train_cfg.num_timesteps,
        num_evals=train_cfg.num_evals,
        # eval 환경 수 — split_eval 분할 정밀도 향상 (brax 기본 128→512).
        # 1σ ≈ 0.5/√512 ≈ ±2.2% (epoch 당) → 다중 epoch 평균 →50% 수렴.
        # eval 시간 ~4× (학습 전체 시간 ~5-10% 증가).
        num_eval_envs=512,
        episode_length=train_cfg.episode_length,
        unroll_length=train_cfg.unroll_length,
        num_minibatches=train_cfg.num_minibatches,
        num_updates_per_batch=train_cfg.num_updates_per_batch,
        discounting=train_cfg.discounting,
        learning_rate=train_cfg.learning_rate,
        learning_rate_schedule_fn=learning_rate_schedule_fn,
        entropy_cost=train_cfg.entropy_cost,
        clipping_epsilon=train_cfg.clipping_epsilon,
        num_envs=train_cfg.num_envs,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed,
        network_factory=make_networks_factory,
        randomization_fn=domain_randomize_fn,
        render_interval=train_cfg.render_interval,
        policy_params_fn=policy_params_fn,
        restore_checkpoint_path=restore_checkpoint_path,
        
        run_name=run_name,
    )

    times = [time.time()]

    last_ckpt_step = 0
    best_ckpt_step = 0
    best_episode_reward = -float("inf")

    def progress(num_steps: int, metrics: Dict[str, Any]):
        nonlocal best_episode_reward, best_ckpt_step, last_ckpt_step

        times.append(time.time())

        if last_ckpt_step > 0:
            shutil.copy2(
                os.path.join(exp_folder_path, str(last_ckpt_step), "policy"),
                os.path.join(exp_folder_path, "policy"),
            )

        last_ckpt_step = num_steps

        episode_reward = float(metrics.get("eval/episode_reward", 0.0))
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_ckpt_step = num_steps

        log_data = log_metrics(
            metrics, times[-1] - times[0], num_steps, train_cfg.num_timesteps
        )

        # Log metrics to wandb
        wandb.log(log_data)

    try:
        _, params, _ = train_fn(
            environment=env, eval_env=eval_env, progress_fn=progress
        )
    except KeyboardInterrupt:
        pass

    shutil.copy2(
        os.path.join(exp_folder_path, str(best_ckpt_step), "policy"),
        os.path.join(exp_folder_path, "best_policy"),
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"best checkpoint step: {best_ckpt_step}")
    print(f"best episode reward: {best_episode_reward}")


def evaluate(
    env: MJXEnv,
    make_networks_factory: Any,
    run_name: str,
    num_steps: int = 1000,
    log_every: int = 100,
):
    """Evaluates a policy in a given environment using a specified network factory and logs the results.

    Args:
        env (MJXEnv): The environment in which the policy is evaluated.
        make_networks_factory (Any): A factory function to create network architectures for the policy.
        run_name (str): The name of the run, used for saving and loading policy parameters.
        num_steps (int, optional): The number of steps to evaluate the policy. Defaults to 1000.
        log_every (int, optional): The frequency (in steps) at which metrics are logged. Defaults to 100.
    """
    ppo_network = make_networks_factory(
        env.obs_size, env.privileged_obs_size, env.action_size
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    policy_path = os.path.join(OUTPUT_DIR, run_name, "best_policy")
    if not os.path.exists(policy_path):
        policy_path = os.path.join(OUTPUT_DIR, run_name, "policy")

    params = model.load_params(policy_path)
    inference_fn = make_policy(params, deterministic=True)

    # initialize the state
    jit_reset = jax.jit(env.reset)
    # jit_reset = env.reset
    jit_step = jax.jit(env.step)
    # jit_step = env.step
    jit_inference_fn = jax.jit(inference_fn)
    # jit_inference_fn = inference_fn

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    # 초기 온도 설정
    # 우리가 정한 에피소드
    # 

    times = [time.time()]
    rollout: List[Any] = [state.pipeline_state]
    for i in tqdm(range(num_steps), desc="Evaluating"):
        ctrl, _ = jit_inference_fn(state.obs, rng)
        state = jit_step(state, ctrl)
        times.append(time.time())
        rollout.append(state.pipeline_state)
        if i % log_every == 0:
            log_metrics(state.metrics, times[-1] - times[0])

    try:
        render_video(env, rollout, run_name)
        wandb.log(
            {
                "video": wandb.Video(
                    os.path.join(OUTPUT_DIR, run_name, "eval.mp4"), format="mp4"
                )
            }
        )
    except Exception:
        print("Failed to render the video. Skipped.")


def main(args=None):
    """Trains or evaluates a policy for a specified robot and environment using PPO.

    This function sets up the training or evaluation of a policy for a robot in a specified environment. It parses command-line arguments to configure the robot, environment, evaluation settings, and other parameters. It then loads configuration files, binds any overridden parameters, and initializes the environment and robot. Depending on the arguments, it either trains a new policy or evaluates an existing one.

    Args:
        args (list, optional): List of command-line arguments. If None, arguments are parsed from sys.argv.

    Raises:
        FileNotFoundError: If a specified gin configuration file or evaluation run is not found.
    """
    parser = argparse.ArgumentParser(description="Train the mjx policy.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="walk",
        help="The name of the env.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="",
        help="Provide the time string of the run to evaluate.",
    )
    parser.add_argument(
        "--restore",
        type=str,
        default="",
        help="Path to the checkpoint folder.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="",
        help="Path to the checkpoint folder.",
    )
    parser.add_argument(
        "--gin-files",
        type=str,
        default="",
        help="List of gin config files",
    )
    parser.add_argument(
        "--config-override",
        type=str,
        default="",
        help="Override config parameters (e.g., SimConfig.timestep=0.01 ObsConfig.frame_stack=10)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="-1",
        help="Select the GPU index. This is equivalent to setting the CUDA_VISIBLE_DEVICES environment variable. A value of -1 (default) makes all GPUs visible."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Directory to save results.",
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output

    # CUDA 설정
    if args.gpu != "-1":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f"Attempting to use CUDA device: {args.gpu}")
    else:
        print(f"Using all available CUDA devices.")

    # 실제 JAX Cuda 확인
    try:
        devices = jax.local_devices()
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device}, Platform: {device.platform}")

        if not any(device.platform == 'gpu' for device in devices):
             print("\nWarning: No GPU detected by JAX. It might be using CPU or TPU.")
             return
        
    except RuntimeError as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure that JAX is installed with CUDA support and the specified GPU is available.")

    gin_file_list = [args.env] + args.gin_files.split(" ")
    for gin_file in gin_file_list:
        if len(gin_file) == 0:
            continue

        gin_file_path = os.path.join(
            os.path.dirname(__file__),
            gin_file + ".gin" if not gin_file.endswith(".gin") else gin_file,
        )
        if not os.path.exists(gin_file_path):
            raise FileNotFoundError(f"File {gin_file_path} not found.")

        gin.parse_config_file(gin_file_path)

    # Bind parameters from --config_override
    if len(args.config_override) > 0:
        for override in args.config_override.split(","):
            key, value = override.split("=", 1)  # Split into key-value pair
            gin.bind_parameter(key, parse_value(value))

    robot = Robot(args.robot)

    try:
        EnvClass = get_env_class(args.env)
    except:
        EnvClass = None
    env_cfg = MJXConfig()
    train_cfg = PPOConfig()

    kwargs = {}
    if len(args.ref) > 0:
        kwargs = {"ref_motion_type": args.ref}

    if "fixed" in args.env:
        train_cfg.num_timesteps = 20_000_000
        train_cfg.num_evals = 200

        env_cfg.rewards.healthy_z_range = [-0.2, 0.2]
        env_cfg.rewards.scales.reset()

        if "walk" in args.env:
            env_cfg.rewards.scales.feet_distance = 0.5

        env_cfg.rewards.scales.leg_motor_pos = 5.0
        env_cfg.rewards.scales.waist_motor_pos = 5.0
        env_cfg.rewards.scales.motor_torque = 5e-2
        env_cfg.rewards.scales.leg_action_rate = 1e-2
        env_cfg.rewards.scales.leg_action_acc = 1e-2
        env_cfg.rewards.scales.waist_action_rate = 1e-2
        env_cfg.rewards.scales.waist_action_acc = 1e-2

    if args.env.startswith("_T_"):
        env_cfg = TMJXConfig(env_cfg)
        env = EnvClass(
            args.env,
            robot,
            env_cfg,  # type: ignore
            fixed_base="fixed" in args.env,
            add_noise=env_cfg.noise.add_noise,
            add_domain_rand=env_cfg.domain_rand.add_domain_rand,
            **kwargs,  # type: ignore
        )

        e = ThermalConfig.EvalConfig()
        eval_cfg = TMJXConfig(env_cfg)
        # split_eval 모드: wrapper.reset 직후 50:50 (bernoulli) cold/hot 버킷
        # 즉시 배정 → cl_progress 의존성 제거. cold = ambient(h=w=a_t),
        # hot = U[40,60] + offset[0,15] (학습 분포의 명확한 hot 절반).
        eval_cfg.thermal_cfg.curriculum.seed_mode          = e.seed_mode
        eval_cfg.thermal_cfg.curriculum.hot_seed_fraction  = e.hot_seed_fraction
        eval_cfg.thermal_cfg.curriculum.hot_seed_h_range   = e.hot_seed_h_range
        eval_cfg.thermal_cfg.curriculum.hot_seed_offset    = e.hot_seed_offset

        # 레거시 필드 (split_eval 경로에선 무시되나 호환 유지)
        eval_cfg.thermal_cfg.curriculum.threshold_ratio = 1.0
        eval_cfg.thermal_cfg.curriculum.init_hot = e.temp_range
        eval_cfg.thermal_cfg.curriculum.init_cold = e.temp_range
        eval_cfg.thermal_cfg.curriculum.use_ep_sampling = e.use_ep_sampling
        eval_cfg.thermal_cfg.curriculum.offset = e.offset

        eval_cfg.thermal_cfg.domain_rand.temp_range = e.temp_range
        eval_cfg.thermal_cfg.env.mode = e.mode
        eval_cfg.thermal_cfg.env.use_w_offset = e.use_w_offset
        eval_cfg.thermal_cfg.env.use_rand_w = e.use_rand_w
        eval_cfg.thermal_cfg.env.offset = e.offset
        eval_cfg.thermal_cfg.reward.safety_penalty = e.safety_penalty

        # 하드 코딩으로 thermal 상태 정의
        eval_cfg.thermal_cfg.env.use_derate = True
        eval_cfg.thermal_cfg.env.use_thermal = True
        eval_cfg.thermal_cfg.env.model_order_2 = True


        eval_env = EnvClass(
            args.env,
            robot,
            eval_cfg,  # type: ignore
            fixed_base="fixed" in args.env,
            add_noise=env_cfg.noise.add_noise,
            add_domain_rand=env_cfg.domain_rand.add_domain_rand,
            **kwargs,  # type: ignore
        )
        test_env = EnvClass(
            args.env,
            robot,
            eval_cfg,  # type: ignore
            fixed_base="fixed" in args.env,
            add_noise=False,
            add_domain_rand=False,
            **kwargs,
        )
        # Curriculum Wrapper 적용 (episode_length → 장주기 reseed 블록 산출)
        env = ThermalCurriculumWrapper(env, train_cfg.num_timesteps, train_cfg.num_envs, env_cfg, episode_length=train_cfg.episode_length)
        eval_env = ThermalCurriculumWrapper(eval_env, train_cfg.num_timesteps, train_cfg.num_envs, eval_cfg, episode_length=train_cfg.episode_length)
        test_env = ThermalCurriculumWrapper(test_env, train_cfg.num_timesteps, train_cfg.num_envs, eval_cfg, episode_length=train_cfg.episode_length)


    else:
        env = EnvClass(
            args.env,
            robot,
            env_cfg,  # type: ignore
            fixed_base="fixed" in args.env,
            add_noise=env_cfg.noise.add_noise,
            add_domain_rand=env_cfg.domain_rand.add_domain_rand,
            **kwargs,  # type: ignore
        )
        eval_env = EnvClass(
            args.env,
            robot,
            env_cfg,  # type: ignore
            fixed_base="fixed" in args.env,
            add_noise=env_cfg.noise.add_noise,
            add_domain_rand=env_cfg.domain_rand.add_domain_rand,
            **kwargs,  # type: ignore
        )
        test_env = EnvClass(
            args.env,
            robot,
            env_cfg,  # type: ignore
            fixed_base="fixed" in args.env,
            add_noise=False,
            add_domain_rand=False,
            **kwargs,
        )

    # 내가 추가한거 확인하기, Debug 용 
    # print(f"init dt: {env.dt}")
    # print(f"init obs_size : {env.obs_size}")
    # print(f'use derate: {env.cfg.tjx_cfg.use_derate}')
    # result = input("Exit?")
    # if result.lower() in ['y', 'yes']:
    #     print("Exiting program.")
    #     return

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
    )

    if len(args.eval) > 0:
        time_str = args.eval
    else:
        time_str = time.strftime("%Y%m%d_%H%M%S")

    config_override_str: str = (
        "" if len(args.config_override) == 0 else f"_{args.config_override}"
    )
    run_name = f"{robot.name}_{args.env}_ppo{config_override_str}_{gin_file_list[1]}_{time_str}"

    if len(args.eval) > 0:
        if os.path.exists(os.path.join(OUTPUT_DIR, run_name)):
            evaluate(test_env, make_networks_factory, run_name)
        else:
            raise FileNotFoundError(f"Run {args.eval} not found.")
    else:
        train(env, eval_env, make_networks_factory, train_cfg, run_name, args.restore)
        evaluate(test_env, make_networks_factory, run_name)


if __name__ == "__main__":
    # jax cache 추가
    cache_path = os.path.join(os.getcwd(), "jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache_path)
    print(f"[JAX] Compilation cache enabled at: {cache_path}")

    main()
