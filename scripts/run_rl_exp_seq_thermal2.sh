#!/bin/bash

# Run RL experiments sequentially
# Define the different configurations for each experiment

robots=("toddlerbot")
envs=("_T_WalkReward2")
config_overrides=(
    "PPOConfig.num_timesteps=10000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,PPOConfig.render_interval=50,HeatRewardScales.warm_up=0.1"
    "PPOConfig.num_timesteps=10000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,PPOConfig.render_interval=50,HeatRewardScales.warm_up=0.11"
    "PPOConfig.num_timesteps=10000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,PPOConfig.render_interval=50,HeatRewardScales.warm_up=0.12"
    "PPOConfig.num_timesteps=10000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,PPOConfig.render_interval=50,HeatRewardScales.warm_up=0.15"
    "PPOConfig.num_timesteps=10000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,PPOConfig.render_interval=50,HeatRewardScales.warm_up=0.18"
    "PPOConfig.num_timesteps=10000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,PPOConfig.render_interval=50,HeatRewardScales.warm_up=0.25"
    "PPOConfig.num_timesteps=10000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,PPOConfig.render_interval=50,HeatRewardScales.warm_up=0.4"
    "PPOConfig.num_timesteps=10000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,PPOConfig.render_interval=50,HeatRewardScales.warm_up=0.65"
)

# Iterate over all configurations
for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for config_override in "${config_overrides[@]}"; do
            echo "Running experiment with Robot: $robot, Env: $env, Config Override: $config_override"
            
            # Run the Python script with the current configuration
            python toddlerbot/locomotion/train_mtjx.py --robot "$robot" --env "$env" --config-override "$config_override" --gpu "0" --restore "results/thermal_basic/60436480"
            
            # Optional: Add a small delay between experiments
            sleep 1
        done
    done
done
 