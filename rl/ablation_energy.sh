#!/bin/bash

# Curriculum Ablation: init_hot min 범위 점진적 확대
# init_hot min = 20, 25, 30, 35, 40 (45는 cl_modified.gin으로 이미 실행)
# GPU 0 단일 슬롯, 순차 실행

export MUJOCO_GL="egl"

SAVED_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
unset LD_LIBRARY_PATH

robot="toddlerbot"
env="_T_Walk"
conda_env="thermalbot"

CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda를 찾을 수 없습니다. conda가 활성화된 환경에서 실행하세요."
    exit 1
fi
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
echo "conda base: $CONDA_BASE"
echo "LD_LIBRARY_PATH (saved): $SAVED_LD_LIBRARY_PATH"

gin_files=(
    "ablation/energy_1x"                # energy = 0.05 (1x)
    "ablation/energy_2x"                # energy = 0.10 (2x)
    "ablation/energy_5x"                # energy = 0.25 (5x)
    "ablation/energy_10x"               # energy = 0.50 (10x)
)

gpu_slots=(0)
MEM_FRACTION=0.90
MAX_PARALLEL=1

timestamp=$(date +"%Y%m%d-%H%M%S")
log_dir="logs/experiment_log_hot_min_${timestamp}"
mkdir -p "$log_dir"
log_file="${log_dir}/dispatcher.txt"

echo "=== Hot Min Ablation Start: $timestamp ===" | tee -a "$log_file"
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
    local session_name="hot_min_${timestamp}_${idx}_${clean_gin}"
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
    sleep 30
}

freed_gpu=""

wait_for_slot() {
    while true; do
        for j in "${!active_sessions[@]}"; do
            local sess="${active_sessions[$j]}"
            if [ -n "$sess" ] && ! tmux has-session -t "$sess" 2>/dev/null; then
                freed_gpu="${active_gpus[$j]}"
                echo "[$(date +"%T")] 세션 $sess 완료 → GPU $freed_gpu 반환 (30초 대기 중...)" \
                    | tee -a "$log_file"
                unset "active_sessions[$j]"
                unset "active_gpus[$j]"
                sleep 30   # GPU VRAM 해제 대기
                return
            fi
        done
        sleep 15
    done
}

# 첫 번째 job 즉시 실행
launch_job "${gin_files[0]}" "${gpu_slots[0]}" "0"

# 나머지: 슬롯이 빌 때마다 순차 실행
for i in $(seq 1 $((${#gin_files[@]} - 1))); do
    echo "[$(date +"%T")] 슬롯 대기 중 (Job $((i+1))/${#gin_files[@]})..." | tee -a "$log_file"
    wait_for_slot
    launch_job "${gin_files[$i]}" "$freed_gpu" "$i"
done

echo "" | tee -a "$log_file"
echo "------------------------------------------------" | tee -a "$log_file"
echo "모든 작업이 tmux 세션으로 분산 실행되었습니다." | tee -a "$log_file"
echo "로그 디렉토리: $log_dir" | tee -a "$log_file"
echo "실행 중인 세션 확인: tmux ls" | tee -a "$log_file"
