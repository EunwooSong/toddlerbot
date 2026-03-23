#!/bin/bash

# 기본 환경 설정
export MUJOCO_GL="egl"
unset LD_LIBRARY_PATH

robots=("toddlerbot")
envs=("_T_Walk")
gin_files=(
    "ablation/model_baseline"
    "ablation/model_ours"
    "ablation/model_olaf"
    "ablation/model_ours_with_cl"
)

# GPU 설정 (0~3번 사용)
gpus=(0 1 2 3)
num_gpus=${#gpus[@]}
current_idx=0

# 로그 파일 이름에 사용할 날짜-시간 형식 (예: 20240520-143005)
timestamp=$(date +"%Y%m%d-%H%M%S")
log_file="experiment_log_${timestamp}.txt"

echo "=== Experiment Start: $timestamp ===" | tee -a "$log_file"

for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for gin_file in "${gin_files[@]}"; do
            
            # 현재 할당할 GPU 결정
            gpu_id=${gpus[$((current_idx % num_gpus))]}
            
            # tmux 세션 이름 생성 (날짜-시간-순번-설정명)
            # 세션 이름에 슬래시(/)가 들어가면 안 되므로 변환
            clean_gin=$(echo "$gin_file" | tr '/' '_')
            session_name="rl_${timestamp}_${current_idx}_${clean_gin}"
            
            echo "[$(date +"%T")] Dispatching $gin_file to GPU $gpu_id (Session: $session_name)" | tee -a "$log_file"
            
            # tmux 세션 생성 및 백그라운드 실행 (-d)
            # JAX 메모리 독점 방지를 위해 환경변수 세팅 포함
            tmux new-session -d -s "$session_name" \
                "export XLA_PYTHON_CLIENT_MEM_FRACTION=.25; \
                 export CUDA_VISIBLE_DEVICES=$gpu_id; \
                 python toddlerbot/locomotion/train_mtjx.py \
                    --robot '$robot' \
                    --env '$env' \
                    --gpu '$gpu_id' \
                    --gin-files '$gin_file'; \
                 exec bash" # 실행 완료 후 세션이 바로 닫히지 않게 bash 유지

            # 다음 GPU로 넘기기
            ((current_idx++))
            
            # 세션 생성 간 아주 짧은 지연 (포트 충돌 등 방지)
            sleep 2
        done
    done
done

echo "------------------------------------------------" | tee -a "$log_file"
echo "모든 작업이 tmux 세션으로 분산 실행되었습니다." | tee -a "$log_file"
echo "로그 파일: $log_file" | tee -a "$log_file"
echo "실행 중인 세션 확인: tmux ls"
