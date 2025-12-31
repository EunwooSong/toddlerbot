#!/bin/bash

# Run RL experiments sequentially
# Define the different configurations for each experiment
export MUJOCO_GL="egl"
export XLA_PYTHON_CLIENT_MEM_FRACTION=".15"
export JAX_TRACEBACK_FILTERING="off"
unset LD_LIBRARY_PATH

robots=("toddlerbot")
envs=("_T_Walk")
#restore=("./results/toddlerbot__T_Walk_ppo_PPOConfig.num_timesteps=300000000,PPOConfig.num_evals=1000,PPOConfig.seed=0_20251120_195130/101068800")

# 이 실험 스크립트는 리워드 함수를 제외하고, 환경적인 제약이 추가되었을 때의 성능을 평가하기 위한 실험입니다.
# 따라서 다음과 같은 학습을 진행합니다.
# 조절할 수 있는 환경적 제약 조건들: 
# TJXEnvConfig.use_basic_obs, TJXEnvConfig.use_derate, TJXEnvConfig.use_group_rand
# 따라서, 3!2 = 6가지의 조합에 대해 실험을 진행합니다.

config_overrides=(
    # bool값인 TJXEnvConfig.use_basic_obs, TJXEnvConfig.use_derate, TJXEnvConfig.use_group_rand를 이용해, 총 6가지의 실험 환경을 세팅합니다.
    # 그 전에, 환경 변수가 잘 동작하는지 확인합니다. group_rand가 동작하나요?
    "PPOConfig.num_timesteps=500000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,TJXEnvConfig.use_basic_obs=True,TJXEnvConfig.use_derate=False,TJXEnvConfig.use_group_rand=False"
    #"PPOConfig.num_timesteps=500000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,TJXEnvConfig.use_basic_obs=True,TJXEnvConfig.use_derate=False,TJXEnvConfig.use_group_rand=True"
    "PPOConfig.num_timesteps=500000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,TJXEnvConfig.use_basic_obs=True,TJXEnvConfig.use_derate=True,TJXEnvConfig.use_group_rand=True"
    #"PPOConfig.num_timesteps=500000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,TJXEnvConfig.use_basic_obs=False,TJXEnvConfig.use_derate=False,TJXEnvConfig.use_group_rand=False"
    "PPOConfig.num_timesteps=500000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,TJXEnvConfig.use_basic_obs=False,TJXEnvConfig.use_derate=True,TJXEnvConfig.use_group_rand=False"
    #"PPOConfig.num_timesteps=500000000,PPOConfig.num_evals=1000,PPOConfig.seed=0,TJXEnvConfig.use_basic_obs=False,TJXEnvConfig.use_derate=True,TJXEnvConfig.use_group_rand=True"
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
