#!/bin/bash

# Run RL experiments sequentially
# Define the different configurations for each experiment
export MUJOCO_GL="egl"

#export JAX_TRACEBACK_FILTERING="off"
unset LD_LIBRARY_PATH

robots=("toddlerbot")
envs=("_T_Walk")

gin_files=(
	"ablation/model_ours_with_cl"
)

# Iterate over all configurations
for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for gin_file in "${gin_files[@]}"; do
            echo "Running experiment with Robot: $robot, Env: $env, Ablation Model Config: $gin_file"
            
            # Run the Python script with the current configuration
            # 일단 테스트용!
            python toddlerbot/locomotion/train_mtjx.py --robot "$robot" --env "$env" --gpu "0" --gin-files "$gin_file"
            
            # Optional: Add a small delay between experiments
            sleep 1
        done
    done
done
