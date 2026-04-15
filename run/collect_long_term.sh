#!/bin/bash
# Long-term data collection: active walk + stand cool-down, repeated N cycles.
#
# Usage:
#   ./collect_long_term.sh [cycles] [run_time_sec] [cool_down_sec] [cool_down_policy]
#
# Defaults:
#   cycles=10, run_time_sec=900 (15 min), cool_down_sec=1200 (20 min), cool_down_policy=stand

CYCLES=${1:-10}
RUN_TIME=${2:-900}
COOL_DOWN_TIME=${3:-1200}
COOL_DOWN_POLICY=${4:-stand}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " Long-term data collection"
echo "  Cycles          : $CYCLES"
echo "  Run time        : ${RUN_TIME}s ($(( RUN_TIME / 60 ))m)"
echo "  Cool-down time  : ${COOL_DOWN_TIME}s ($(( COOL_DOWN_TIME / 60 ))m)"
echo "  Cool-down policy: $COOL_DOWN_POLICY"
echo "  Start           : $(date)"
echo "========================================"

cd "$REPO_ROOT"

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
        --cool-down-policy "$COOL_DOWN_POLICY" \
        --gin-file "ablation/model_c_h.gin"

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
