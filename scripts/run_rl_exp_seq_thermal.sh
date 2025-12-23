#!/bin/bash

# Run RL experiments sequentially
# Define the different configurations for each experiment

robots=("toddlerbot")
envs=("_T_Walk")
#restore=("./results/toddlerbot__T_Walk_ppo_PPOConfig.num_timesteps=300000000,PPOConfig.num_evals=1000,PPOConfig.seed=0_20251120_195130/101068800")
config_overrides=(
    "PPOConfig.num_timesteps=300000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,EnvConfig.use_basic_obs=False"
)

# Iterate over all configurations
for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for config_override in "${config_overrides[@]}"; do
            echo "Running experiment with Robot: $robot, Env: $env, Config Override: $config_override"
            
            # Run the Python script with the current configuration
            python toddlerbot/locomotion/train_mtjx.py --robot "$robot" --env "$env" --config-override "$config_override" --gpu "0" #--restore "$restore"
            
            # Optional: Add a small delay between experiments
            sleep 1
        done
    done
done
