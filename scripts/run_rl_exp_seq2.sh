#!/bin/bash

# Run RL experiments sequentially
# Define the different configurations for each experiment

robots=("toddlerbot")
envs=("walk thermal")
config_overrides=(
    "PPOConfig.num_timesteps=50000000,PPOConfig.num_evals=3000,PPOConfig.seed=0"
)
# Iterate over all configurations
for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for config_override in "${config_overrides[@]}"; do
            echo "Running experiment with Robot: $robot, Env: $env, Config Override: $config_override"
            
            # Run the Python script with the current configuration
            python toddlerbot/locomotion/train_mtjx.py --robot "$robot" --env "$env" --config-override "$config_override" --gpu "0"
            
            # Optional: Add a small delay between experiments
            sleep 1
        done
    done
done
