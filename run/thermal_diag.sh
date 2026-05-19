#!/bin/bash
# Offline post-training thermal validation of saved checkpoint(s).
#
# 학습 중 brax 변환과 충돌하던 in-train 진단을 대체 — 저장된 체크포인트를
# 독립 프로세스(brax-free)로 로드해 전압-derate(적합 J1, 파라미터 DR off)
# + 학습 정책 persistent 롤아웃 → 정책이 발열을 gait 로 관리하는지 정량화.
#
# Usage:
#   bash run/thermal_diag.sh <ckpt_or_run_dir> [gin] [steps] [hot_start]
#     <ckpt_or_run_dir> :
#        - 단일 체크포인트(.../<step>/policy, .../best_policy, 또는 그 dir)
#        - run/exp 디렉터리(results/.../model_ours_YYYYMMDD_.../)
#          → 그 안의 숫자 step 하위폴더 + best_policy 전부 sweep
#     gin       : 기본 ablation/model_ours
#     steps     : persistent 롤아웃 control-step (기본 12000 ≈ 240s)
#     hot_start : >0 이면 권선 초기온도[°C] 강제(검증용; 기본 0=cold)
#
#   결과: <run_dir>/thermal_diag.csv 에 체크포인트당 1줄 append + 요약표.
#   학습 step 따라 w_p99 / overheat_rate / derate_sev 가 *하강*하면
#   = 정책이 발열을 gait 로 회피/분산 = 기여 발현.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}" || exit 1                    # Robot config 경로는 repo root 기준

PY="/home/eunwoo/anaconda3/envs/toddlerbot/bin/python"
export MUJOCO_GL="${MUJOCO_GL:-egl}"      # headless

TARGET="${1:-}"
GIN="${2:-ablation/model_ours}"
STEPS="${3:-12000}"
HOT="${4:-0}"

if [[ -z "${TARGET}" || ! -e "${TARGET}" ]]; then
  echo "Usage: bash run/thermal_diag.sh <ckpt_or_run_dir> [gin] [steps] [hot_start]" >&2
  echo "  (경로가 없거나 미지정)" >&2
  exit 1
fi

run_one() {  # $1=ckpt  $2=csv
  "${PY}" -m heat2torque.eval.thermal_diag \
    --ckpt "$1" --gin "${GIN}" --steps "${STEPS}" \
    --hot-start "${HOT}" --out "$2"
}

# 단일 체크포인트인지(파일이거나 policy 파일을 가진 dir) sweep 대상인지 판별
is_ckpt_dir() { [[ -f "$1/policy" || -f "$1/best_policy" ]]; }

if [[ -f "${TARGET}" ]] || is_ckpt_dir "${TARGET}"; then
  # 단일
  OUT_DIR="$(dirname "$([[ -f "${TARGET}" ]] && echo "${TARGET}" || echo "${TARGET}/x")")"
  CSV="${OUT_DIR}/thermal_diag.csv"
  echo "[thermal_diag.sh] single: ${TARGET} → ${CSV}"
  run_one "${TARGET}" "${CSV}"
else
  # run/exp 디렉터리 sweep: 숫자 step 하위폴더(자연순) + best_policy
  CSV="${TARGET%/}/thermal_diag.csv"
  rm -f "${CSV}"
  echo "[thermal_diag.sh] sweep: ${TARGET} → ${CSV}"
  shopt -s nullglob
  mapfile -t STEP_DIRS < <(
    for d in "${TARGET%/}"/*/; do
      b="$(basename "$d")"
      [[ "$b" =~ ^[0-9]+$ ]] && echo "$b $d"
    done | sort -n | awk '{print $2}'
  )
  for d in "${STEP_DIRS[@]}"; do
    if [[ -f "${d}policy" ]]; then
      echo "  → step $(basename "$d")"
      run_one "${d%/}" "${CSV}" || echo "    (skip: $d)"
    fi
  done
  if [[ -f "${TARGET%/}/best_policy" ]]; then
    echo "  → best_policy"
    run_one "${TARGET%/}/best_policy" "${CSV}" || echo "    (skip best)"
  fi
  shopt -u nullglob
fi

# 요약표 (step 오름차순)
if [[ -f "${CSV}" ]]; then
  echo
  echo "================ thermal_diag 요약 (${CSV}) ================"
  "${PY}" - "${CSV}" <<'PYEOF'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1])))
def k(r):
    s=r.get("step","")
    return (0,int(s)) if s.isdigit() else (1,s)
rows.sort(key=k)
cols=["step","w_p50","w_p99","w_max","w_final","h_max","derate_sev","overheat_rate","fall_s"]
print(" | ".join(f"{c:>12}" for c in cols))
for r in rows:
    print(" | ".join(f"{r.get(c,''):>12}" for c in cols))
print("\n해석: step↑ 에 따라 w_p99/overheat_rate/derate_sev ↓ + fall_s↑(또는 NONE/-1)"
      " → 정책이 발열을 gait 로 회피/분산 = 기여 발현.")
PYEOF
fi
