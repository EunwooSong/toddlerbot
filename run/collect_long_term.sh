#!/bin/bash
# Long-term data collection: active walk + stand cool-down, repeated N cycles.
#
# Usage:
#   ./collect_long_term.sh [cycles] [run_time_sec] [cool_down_sec] [cool_down_policy]
#
# Defaults:
#   cycles=1, run_time_sec=900 (15 min), cool_down_sec=1200 (45 min), cool_down_policy=none

CYCLES=${1:-1}
RUN_TIME=${2:-900}
COOL_DOWN_TIME=${3:-1200}
COOL_DOWN_POLICY=${4:-stand}

echo "========================================"
echo " Long-term data collection"
echo "  Cycles          : $CYCLES"
echo "  Run time        : ${RUN_TIME}s ($(( RUN_TIME / 60 ))m)"
echo "  Cool-down time  : ${COOL_DOWN_TIME}s ($(( COOL_DOWN_TIME / 60 ))m)"
echo "  Cool-down policy: $COOL_DOWN_POLICY"
echo "  Start           : $(date)"
echo "========================================"

for i in $(seq 1 "$CYCLES"); do
    echo ""
    echo "-------- Cycle $i / $CYCLES  [$(date)] --------"

    python toddlerbot/policies/run_policy.py \
        --robot toddlerbot \
        --policy thermal_walk \
        --sim real \
        --vis none \
        --no-plot \
        --run-time "$RUN_TIME" \
        --cool-down-time "$COOL_DOWN_TIME" \
        --gin-file "ablation/model_c_h.gin"
        #--cool-down-policy "$COOL_DOWN_POLICY" \

    EXIT_CODE=$?
    echo "-------- Cycle $i done [$(date)]  exit=$EXIT_CODE --------"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: run_policy.py exited with code $EXIT_CODE. Aborting collection."
        exit $EXIT_CODE
    fi
done

echo ""
echo "========================================"
echo " All $CYCLES cycles completed."
echo " End: $(date)"
echo "========================================"
