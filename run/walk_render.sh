#!/bin/bash

# Run walk policy by render
# egl render
export MUJOCO_GL="egl"
#export USE_JAX="true"

# for debug
# export JAX_TRACEBACK_FILTERING="off"

# cd
# cd ~/eunwoo/back/toddlerbot

# run policy
python toddlerbot/policies/run_policy.py --robot toddlerbot --policy walk --sim mujoco --vis render  --config-override "TJXEnvConfig.use_basic_obs=True"


#export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#export CUDADIR=/usr/local/cuda-12.6