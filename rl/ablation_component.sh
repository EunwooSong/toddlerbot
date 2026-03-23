#!/bin/bash

# Component & Energy Penalty Ablation Study
# 1) Thermal Obs ON/OFF × Derating ON/OFF (2×2)
# 2) Safety Reward OFF
# 3) Energy Penalty 강화 Baseline (1x, 2x, 5x, 10x) — 열 모델 없음
# 참고: toddlerbot/locomotion/ablation/cl_modified.gin

export MUJOCO_GL="egl"

# LD_LIBRARY_PATH를 unset하기 전에 현재 값을 저장
# tmux 세션은 fresh shell → unset하면 CUDA 라이브러리 경로가 사라져 JAX가 CPU fallback
SAVED_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
unset LD_LIBRARY_PATH

robot="toddlerbot"
env="_T_Walk"
conda_env="thermalbot"

# 현재 셸(conda 활성 환경)에서 conda 경로를 미리 확정해 tmux에 넘김
# tmux 내부는 non-interactive shell → .bashrc 상단 [ -z "$PS1" ] && return 가드로
# conda init이 건너뛰어지므로, conda.sh를 직접 source하는 방식 사용
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda를 찾을 수 없습니다. conda가 활성화된 환경에서 실행하세요."
    exit 1
fi
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
echo "conda base: $CONDA_BASE"
echo "LD_LIBRARY_PATH (saved): $SAVED_LD_LIBRARY_PATH"

gin_files=(
    # --- Curriculum Ablation ---
    "ablation/cl_baseline_cold"         # 고온 노출 없음
    "ablation/cl_baseline_hot"          # 처음부터 고온, 커리큘럼 없음
    "ablation/cl_current"               # 현재 설정, 고온 노출 부족
    "ablation/cl_modified"              # 제안 설정 / TA-FTC reference
    # --- Component Ablation (Obs × Derate) ---
    "ablation/comp_obs_on_derate_off"   # Obs ON,  Derate OFF
    "ablation/comp_obs_off_derate_on"   # Obs OFF, Derate ON
    "ablation/comp_obs_off_derate_off"  # Obs OFF, Derate OFF (no thermal)
    # --- Safety Reward Ablation ---
    "ablation/comp_no_safety"           # Obs ON, Derate ON, Safety OFF
    # --- Energy Penalty Baseline (no thermal) ---
    "ablation/energy_1x"                # energy = 0.05 (1x)
    "ablation/energy_2x"                # energy = 0.10 (2x)
    "ablation/energy_5x"                # energy = 0.25 (5x)
    "ablation/energy_10x"               # energy = 0.50 (10x)
)

# 4 GPU × 2 슬롯 = 8 동시 실행, 나머지는 대기 후 순차 실행
# 각 GPU당 2개 학습 -> 메모리 분율 0.45 할당 (MPS 제거)
gpu_slots=(0 1 2 3)
MEM_FRACTION=0.25
MAX_PARALLEL=4

timestamp=$(date +"%Y%m%d-%H%M%S")
log_dir="logs/experiment_log_comp_${timestamp}"
mkdir -p "$log_dir"
log_file="${log_dir}/dispatcher.txt"

echo "=== Component Ablation Start: $timestamp ===" | tee -a "$log_file"
echo "Total runs: ${#gin_files[@]}, Parallel slots: $MAX_PARALLEL" | tee -a "$log_file"
echo "Runs: ${gin_files[*]}" | tee -a "$log_file"
echo "Job logs: ${log_dir}/log_<session>.txt" | tee -a "$log_file"
echo "" | tee -a "$log_file"

declare -a active_sessions
declare -a active_gpus

launch_job() {
    local gin_file=$1
    local gpu_id=$2
    local idx=$3

    local clean_gin
    clean_gin=$(echo "$gin_file" | tr '/' '_')
    local session_name="comp_abl_${timestamp}_${idx}_${clean_gin}"
    local job_log="${log_dir}/log_${session_name}.txt"

    echo "[$(date +"%T")] Dispatching [$((idx+1))/${#gin_files[@]}] $gin_file → GPU $gpu_id" \
        "(Session: $session_name)" | tee -a "$log_file"
    echo "              Job log: $job_log" | tee -a "$log_file"

    tmux new-session -d -s "$session_name" \
        "source ${CONDA_SH} && conda activate ${conda_env} && \
         export LD_LIBRARY_PATH=${SAVED_LD_LIBRARY_PATH}:\$LD_LIBRARY_PATH && \
         export MUJOCO_GL=egl && \
         export MUJOCO_EGL_DEVICE_ID=${gpu_id} && \
         export LOKY_MAX_CPU_COUNT=16 && \
         export OMP_NUM_THREADS=1 && \
         export MKL_NUM_THREADS=1 && \
         export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION} && \
         export CUDA_VISIBLE_DEVICES=${gpu_id} && \
         python toddlerbot/locomotion/train_mtjx.py \
            --robot ${robot} \
            --env ${env} \
            --gpu ${gpu_id} \
            --gin-files ${gin_file} 2>&1 | tee ${job_log}"

    active_sessions+=("$session_name")
    active_gpus+=("$gpu_id")
    sleep 2
}

freed_gpu=""   # 전역 변수로 결과 전달 (서브셸 방지)

wait_for_slot() {
    # 완료된 세션을 감지해 freed_gpu 전역 변수에 GPU id를 저장
    # 반드시 서브셸 없이 호출할 것: wait_for_slot (o) / freed=$(wait_for_slot) (x)
    while true; do
        for j in "${!active_sessions[@]}"; do
            local sess="${active_sessions[$j]}"
            if [ -n "$sess" ] && ! tmux has-session -t "$sess" 2>/dev/null; then
                freed_gpu="${active_gpus[$j]}"
                echo "[$(date +"%T")] 세션 $sess 완료 → GPU $freed_gpu 반환" \
                    | tee -a "$log_file"
                unset "active_sessions[$j]"
                unset "active_gpus[$j]"
                return
            fi
        done
        sleep 15
    done
}

# 처음 8개 즉시 실행
for i in $(seq 0 $((MAX_PARALLEL - 1))); do
    if [ $i -lt ${#gin_files[@]} ]; then
        launch_job "${gin_files[$i]}" "${gpu_slots[$i]}" "$i"
    fi
done

# 나머지 작업: 슬롯이 빌 때마다 순차 실행
for i in $(seq $MAX_PARALLEL $((${#gin_files[@]} - 1))); do
    echo "[$(date +"%T")] 슬롯 대기 중 (Job $((i+1))/${#gin_files[@]})..." | tee -a "$log_file"
    wait_for_slot          # 서브셸 없이 직접 호출 → freed_gpu에 결과 저장
    launch_job "${gin_files[$i]}" "$freed_gpu" "$i"
done

echo "" | tee -a "$log_file"
echo "------------------------------------------------" | tee -a "$log_file"
echo "모든 작업이 tmux 세션으로 분산 실행되었습니다." | tee -a "$log_file"
echo "로그 디렉토리: $log_dir" | tee -a "$log_file"
echo "실행 중인 세션 확인: tmux ls" | tee -a "$log_file"