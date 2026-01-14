#!/bin/bash

# Run walk policy by render
# egl render
export MUJOCO_GL="egl"

# for debug
# export JAX_TRACEBACK_FILTERING="off"

# cd
# cd ~/eunwoo/back/toddlerbot

# run policy
python toddlerbot/policies/run_policy.py --robot toddlerbot --policy thermal_walk --sim mujoco --vis render --config-override "TJXEnvConfig.use_basic_obs=False"