#!/bin/bash
# ============================================================================
#  final_olaf.sh — Olaf baseline (faithful, neck-only) 학습
#  비교군 1: 1st-order LPTN + CBF reward + no derate + obs/CBF on neck only.
#  학습 인프라(curriculum/persistence-block/ambient/노이즈) 는 ours 와 동일,
#  단 thermal persistence 는 OFF (per-episode reset, Olaf 학습 디자인).
#  Eval 은 모든 정책 동일: high-fidelity 2-LPTN + derate 환경, split_eval.
# ============================================================================
export MUJOCO_GL="egl"
unset LD_LIBRARY_PATH

robots=("toddlerbot")
envs=("_T_Walk")

gin_files=(
    "ablation/model_olaf"
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
