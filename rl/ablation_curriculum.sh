#!/bin/bash

# Curriculum Ablation Study
# init_cold / init_hot 범위 변경에 따른 커리큘럼 효과 비교
# 참고: toddlerbot/locomotion/ablation/model_ours_with_cl.gin

export MUJOCO_GL="egl"

# LD_LIBRARY_PATH를 unset하기 전에 저장 (tmux 세션에서 CUDA 라이브러리 경로 유지용)
SAVED_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
unset LD_LIBRARY_PATH

# conda.sh 경로 확정 (tmux 내부는 non-interactive shell → conda init이 건너뛰어짐)
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda를 찾을 수 없습니다. conda가 활성화된 환경에서 실행하세요."
    exit 1
fi
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
echo "conda base: $CONDA_BASE"

robots=("toddlerbot")
envs=("_T_Walk")
gin_files=(
    "ablation/cl_baseline_cold"
    "ablation/cl_baseline_hot"
    "ablation/cl_current"
    "ablation/cl_modified"
)

# GPU 설정 (0~3번 사용)
gpus=(0 1 2 3)
num_gpus=${#gpus[@]}
current_idx=0

timestamp=$(date +"%Y%m%d-%H%M%S")
mkdir -p logs
log_file="logs/experiment_log_${timestamp}.txt"

echo "=== Curriculum Ablation Start: $timestamp ===" | tee -a "$log_file"
echo "Runs: ${gin_files[*]}" | tee -a "$log_file"

for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for gin_file in "${gin_files[@]}"; do

            gpu_id=${gpus[$((current_idx % num_gpus))]}

            clean_gin=$(echo "$gin_file" | tr '/' '_')
            session_name="cl_abl_${timestamp}_${current_idx}_${clean_gin}"

            echo "[$(date +"%T")] Dispatching $gin_file to GPU $gpu_id (Session: $session_name)" | tee -a "$log_file"

            tmux new-session -d -s "$session_name" \
                "sleep 5; \
                 source ${CONDA_SH} && conda activate thermalbot && \
                 export LD_LIBRARY_PATH=${SAVED_LD_LIBRARY_PATH}:\$LD_LIBRARY_PATH && \
                 XLA_PYTHON_CLIENT_MEM_FRACTION=.25 \
                 CUDA_VISIBLE_DEVICES=0 \
                 CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${gpu_id} \
                 python toddlerbot/locomotion/train_mtjx.py \
                    --robot $robot \
                    --env $env \
                    --gpu 0 \
                    --gin-files $gin_file 2>&1 | tee ${log_file%.txt}_${clean_gin}.txt"

            ((current_idx++))
            sleep 2
        done
    done
done

echo "------------------------------------------------" | tee -a "$log_file"
echo "모든 작업이 tmux 세션으로 분산 실행되었습니다." | tee -a "$log_file"
echo "로그 파일: $log_file" | tee -a "$log_file"
echo "실행 중인 세션 확인: tmux ls"
