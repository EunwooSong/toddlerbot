#!/bin/bash
# ============================================================================
#  final_olaf_full.sh — Steelmanned Olaf (all 30 motors) 학습
#  비교군 2: faithful Olaf 와 동일하되 obs/CBF 가 전신 30 모터. "Olaf 가
#  만약 전신 thermal sensing 을 가졌다면" 의 상한 — 우리 method 의
#  contribution 이 monitoring scope 가 아니라 model order/derate/persistence
#  임을 분리 증명.
# ============================================================================
export MUJOCO_GL="egl"
unset LD_LIBRARY_PATH

robots=("toddlerbot")
envs=("_T_Walk")

gin_files=(
    "ablation/model_olaf_full"
)

for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for gin_file in "${gin_files[@]}"; do
            echo "Running experiment with Robot: $robot, Env: $env, Ablation Model Config: $gin_file"
            python toddlerbot/locomotion/train_mtjx.py --robot "$robot" --env "$env" --gpu "0" --gin-files "$gin_file"
            sleep 1
        done
    done
done
