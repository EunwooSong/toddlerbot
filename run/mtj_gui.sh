#!/bin/bash

# Launch MuJoCo + Thermal model + Tk GUI sandbox for ANY trained policy.
#
# Picks, for a given model id, the matching trained checkpoint AND the same
# gin file used at training time (obs config + 2nd-order thermal init), so the
# observation layout the network expects always matches the checkpoint.
#
# Usage:
#   bash run/mtj_gui.sh --list                    # list available models
#   bash run/mtj_gui.sh model_d_l_o_u             # GUI for that trained model
#   bash run/mtj_gui.sh model_e_m derate          # + initial thermal mode
#   bash run/mtj_gui.sh walk                      # baseline walk (no thermal obs)
#   bash run/mtj_gui.sh model_d_l_o_u --record    # + record trajectory (npz)
#   bash run/mtj_gui.sh model_c_h --ref <real_dir> [--tamb-csv <ds18b20.csv>]
#                                                 # sim-vs-real compare-mode:
#                                                 # cold-start aligned to real,
#                                                 # writes log_data.pkl
#                                                 # (no --ref ⇒ unchanged random)
#   bash run/mtj_gui.sh thermal_walk <gin>        # legacy raw passthrough
#   bash run/mtj_gui.sh --render <rec_dir> [opts] # offline mp4 (headless EGL)
#   bash run/mtj_gui.sh --replay <rec_dir>        # interactive replay viewer
#
# Model id accepts any of: model_d_l_o_u | thermal_walk_model_d_l_o_u_policy
# (the thermal_walk_ prefix / _policy suffix are stripped automatically).
# 2nd positional (model/walk only) = thermal mode: basic | thermal | derate.
#
# Resolution per model id <M>:
#   checkpoint : toddlerbot/policies/checkpoints/thermal_walk_<M>_policy
#   gin file   : first that exists, in order —
#                  toddlerbot/locomotion/ablation/<M>.gin        (training)
#                  toddlerbot/policies/ablation/<M>.gin           (deploy copy)
#                  toddlerbot/locomotion/ablation/old/<M>.gin     (legacy)
#   (force a specific gin with env GIN=<abs path | model id>.)
#
# Live GUI modes (env MODE=basic|thermal|derate, default thermal); thermal /
# derate use the coupled 2nd-order LPTN deployed from A3-J1 (per-motor params
# in heat2torque/data/thermal_params.json: R_e = datasheet R_e+R_board, beta=0).
# derate = sysID tau_max (Grandia RSS'24 / Toddlerbot Eq.9, the value passed to
# the controller) scaled by g(dT_w) = (1-b*dT)/(1+a*dT) — winding-temp torque
# loss is emergent R_e(T)/K_t(T) physics, NOT firmware foldback — plus the
# 80C hard-cutoff latch (Dynamixel Overheating shutdown). MODE/SEED via env.

export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
# Optional: pin to one GPU.
# export CUDA_VISIBLE_DEVICES=0

# Run from the toddlerbot package root (one dir above this script's parent).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}" || exit 1

CKPT_DIR="${ROOT}/toddlerbot/policies/checkpoints"
GIN_DIRS=(
  "${ROOT}/toddlerbot/locomotion/ablation"
  "${ROOT}/toddlerbot/policies/ablation"
  "${ROOT}/toddlerbot/locomotion/ablation/old"
)

# strip the conventional thermal_walk_ prefix / _policy suffix.
normalize_model() {
  local m="$1"
  m="${m#thermal_walk_}"
  m="${m%_policy}"
  echo "$m"
}

# echo absolute gin path for a model id, or empty if none found.
find_gin() {
  local m="$1" d
  for d in "${GIN_DIRS[@]}"; do
    if [[ -f "${d}/${m}.gin" ]]; then
      echo "${d}/${m}.gin"
      return 0
    fi
  done
  return 1
}

list_models() {
  echo "Available models (checkpoint  ->  resolved gin):"
  local f base m gin
  shopt -s nullglob
  for f in "${CKPT_DIR}"/thermal_walk_*_policy; do
    base="$(basename "$f")"
    m="$(normalize_model "$base")"
    gin="$(find_gin "$m" || true)"
    printf "  %-22s -> %s\n" "$m" "${gin:-<NO GIN FOUND>}"
  done
  [[ -f "${CKPT_DIR}/walk_policy" ]] && \
    printf "  %-22s -> %s\n" "walk" "(baseline, no gin)"
  shopt -u nullglob
}

# ── subcommand dispatch ────────────────────────────────────────────────
case "${1:-}" in
  --render)
    shift
    exec python toddlerbot/policies/render_thermal_traj.py "$@"
    ;;
  --replay)
    shift
    unset MUJOCO_GL    # passive viewer needs a real GL context, not EGL.
    exec python toddlerbot/policies/run_thermal_replay.py "$@"
    ;;
  --list|list|-l)
    list_models
    exit 0
    ;;
esac

# ── live GUI (optionally recording) ────────────────────────────────────
# MuJoCo passive viewer requires a real GL context, not EGL.
unset MUJOCO_GL

MODE="${MODE:-thermal}"   # basic | thermal | derate
SEED="${SEED:-0}"

RECORD=0
RECORD_DIR=""
REF=""
TAMB_CSV=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --record)      RECORD=1; shift ;;
    --record-dir)  RECORD_DIR="$2"; shift 2 ;;
    --ref)         REF="$2"; shift 2 ;;
    --tamb-csv)    TAMB_CSV="$2"; shift 2 ;;
    *)             POSITIONAL+=("$1"); shift ;;
  esac
done

SELECTOR="${POSITIONAL[0]:-}"
ARG2="${POSITIONAL[1]:-}"

if [[ -z "${SELECTOR}" ]]; then
  echo "Usage: bash run/mtj_gui.sh <model|walk> [mode] [--record] [--record-dir DIR]"
  echo
  list_models
  echo
  echo "(mode = basic|thermal|derate, default '${MODE}'; or set env MODE/SEED)"
  exit 0
fi

POLICY="thermal_walk"
CKPT=""
GIN_FILE=""

if [[ "${SELECTOR}" == "walk" ]]; then
  # Baseline walk: MJXPolicy loads checkpoints/walk_policy itself; no gin.
  POLICY="walk"
  [[ -n "${ARG2}" ]] && MODE="${ARG2}"

elif [[ "${SELECTOR}" == "thermal_walk" ]]; then
  # Legacy raw passthrough: 2nd positional is the gin (unchanged behavior).
  GIN_FILE="${ARG2}"

else
  # Model-driven: resolve checkpoint + training gin from the model id.
  # 2nd positional, if present, is the thermal mode.
  [[ -n "${ARG2}" ]] && MODE="${ARG2}"

  MODEL="$(normalize_model "${SELECTOR}")"
  CKPT="${CKPT_DIR}/thermal_walk_${MODEL}_policy"
  if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    echo >&2
    list_models >&2
    exit 1
  fi

  if [[ -n "${GIN:-}" ]]; then
    # Explicit gin override (absolute path or model id under GIN_DIRS).
    if [[ -f "${GIN}" ]]; then
      GIN_FILE="${GIN}"
    else
      GIN_FILE="$(find_gin "$(normalize_model "${GIN}")" || true)"
    fi
  else
    GIN_FILE="$(find_gin "${MODEL}" || true)"
  fi
  if [[ -z "${GIN_FILE}" || ! -f "${GIN_FILE}" ]]; then
    echo "ERROR: no gin file found for model '${MODEL}'." >&2
    echo "       searched: ${GIN_DIRS[*]/%//${MODEL}.gin}" >&2
    echo "       (or set env GIN=<abs path|model id>)" >&2
    exit 1
  fi
fi

# Validate the thermal mode early (run_thermal_gui.py also enforces choices).
case "${MODE}" in
  basic|thermal|derate) ;;
  *)
    echo "ERROR: invalid mode '${MODE}' (expected basic|thermal|derate)." >&2
    exit 1
    ;;
esac

ARGS=(
  --robot toddlerbot
  --policy "${POLICY}"
  --vis view
  --mode "${MODE}"
  --seed "${SEED}"
)
[[ -n "${CKPT}" ]]      && ARGS+=(--ckpt "${CKPT}")
[[ -n "${GIN_FILE}" ]]  && ARGS+=(--gin-file "${GIN_FILE}")
[[ "${RECORD}" -eq 1 ]] && ARGS+=(--record)
[[ -n "${RECORD_DIR}" ]] && ARGS+=(--record-dir "${RECORD_DIR}")
[[ -n "${REF}" ]]        && ARGS+=(--ref "${REF}")
[[ -n "${TAMB_CSV}" ]]   && ARGS+=(--tamb-csv "${TAMB_CSV}")

echo "[mtj_gui] policy=${POLICY} ckpt=${CKPT:-<default>} gin=${GIN_FILE:-<none>} mode=${MODE} seed=${SEED}"
exec python toddlerbot/policies/run_thermal_gui.py "${ARGS[@]}"
