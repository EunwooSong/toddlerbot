#!/bin/bash

# Run walk policy by render
# egl render
export MUJOCO_GL="egl"

# for debug
# export JAX_TRACEBACK_FILTERING="off"

# cd
# cd ~/eunwoo/back/toddlerbot

# run policy
python toddlerbot/policies/run_policy.py --robot toddlerbot\
                                         --policy thermal_walk \
                                         --sim real \
                                         --no-plot \
                                         --gin-file "ablation/model_e_m.gin"