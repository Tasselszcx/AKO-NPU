#!/bin/bash
# 16-card parallel full-machine utilization demo.
# Each card runs the precision-PASS 4096^3 matmul_leakyrelu binary REPS times.
# Reports per-card PASS count and total wall-clock for all cards combined.
set -uo pipefail
cd "$(dirname "$0")/par16"
REPS=${REPS:-3}
PY=${PY:-/usr/local/conda/bin/python}

t0=$(date +%s.%N)
for c in $(seq 0 15); do
  (
    cd "card_$c"
    export ASCEND_RT_VISIBLE_DEVICES=$c
    ok=0
    for r in $(seq 1 "$REPS"); do
      ./matmul_leakyrelu_custom >"run_$r.log" 2>&1 && ok=$((ok+1))
    done
    # verify precision on the last produced output
    pass=$("$PY" ../../scripts/verify_result.py output/output.bin output/golden.bin 2>/dev/null | grep -c "Overall: PASS")
    echo "card $c launches_ok=$ok/$REPS precision=$([ "$pass" = 1 ] && echo PASS || echo FAIL)" > "status_$c.txt"
  ) &
done
wait
t1=$(date +%s.%N)

echo "===== 16-card x $REPS reps DONE ====="
cat card_*/status_*.txt 2>/dev/null | sort -t' ' -k2 -n
WALL=$(echo "$t1 - $t0" | bc)
echo "wall_clock=${WALL}s  total_launches=$((16*REPS))"
