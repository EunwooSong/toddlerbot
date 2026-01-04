# Run walk policy by render
# egl render
export MUJOCO_GL="egl"

# for debug
# export JAX_TRACEBACK_FILTERING="off"

# conda activate
# conda activate toddlerbot_back

# cd
cd ~/eunwoo/back/toddlerbot

# run policy
python toddlerbot/policies/run_policy.py --robot toddlerbot --policy stand --sim real --config-override "TJXEnvConfig.use_basic_obs=True"