#!/bin/bash
# big-matmul experiment driver (base = matmul-py kernel, MIX mode, builds OK on CANN 8.3.RC1).
# Build one config -> run -> verify precision -> profile with msprof.
# Usage: bash run_config.sh <tag> <M> <N> <K> <SETDIM> <BLOCKS> <BASEM> <BASEN>
# Env:   ASCEND_RT_VISIBLE_DEVICES (card), SKIP_PROF=1 to skip msprof.
set -eo pipefail
cd "$(dirname "$0")"

TAG=${1:-base}; M=${2:-1024}; N=${3:-640}; K=${4:-256}
CORES=${5:-1}; BLOCKS=${6:-1}; BASEM=${7:-256}; BASEN=${8:-128}
PY=${PY:-/usr/local/conda/bin/python}
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
# Use the real ASCConfig.cmake path (same one run.sh discovers); 'latest' symlink misses kernel_tiling include.
CMAKE_PREFIX=$(dirname "$(find /usr/local/Ascend/ascend-toolkit -name ASCConfig.cmake -print -quit 2>/dev/null)")

BD="build_${TAG}"
echo "==== [$TAG] M=$M N=$N K=$K SetDim=$CORES blocks=$BLOCKS baseM=$BASEM baseN=$BASEN  (card=${ASCEND_RT_VISIBLE_DEVICES:-0}) ===="
rm -rf "$BD"; mkdir -p "$BD"; cd "$BD"
cmake -DASC_DIR="$CMAKE_PREFIX" \
      -DMM_M=$M -DMM_N=$N -DMM_K=$K -DMM_CORES=$CORES -DMM_BLOCKS=$BLOCKS -DMM_BASEM=$BASEM -DMM_BASEN=$BASEN \
      .. >cmake.log 2>&1 || { echo "[FAIL] cmake"; tail -30 cmake.log; exit 1; }
make -j >make.log 2>&1 || { echo "[FAIL] make"; tail -40 make.log; exit 1; }
echo "[ok] build"

MM_M=$M MM_N=$N MM_K=$K "$PY" ../scripts/gen_data.py | tail -1
./matmul_leakyrelu_custom >run.log 2>&1 || { echo "[FAIL] run"; tail -30 run.log; exit 1; }
echo "[ok] run"

"$PY" ../scripts/verify_result.py output/output.bin output/golden.bin | grep -E "Overall|allclose|Safe metrics|MERE|MARE|match ratio" | head -8

if [ "${SKIP_PROF:-0}" != "1" ]; then
  MSPROF_OUT=$(mktemp -d /tmp/msprof_XXXXXX); chmod 700 "$MSPROF_OUT"
  msprof op --warm-up=5 --launch-count=10 --output="$MSPROF_OUT" ./matmul_leakyrelu_custom >msprof.log 2>&1 || true
  OPPROF=$(ls -td "$MSPROF_OUT"/OPPROF_* 2>/dev/null | head -1)
  if [ -n "$OPPROF" ] && [ -f "$OPPROF/OpBasicInfo.csv" ]; then
    echo "--- OpBasicInfo (Task Duration) ---"; cat "$OPPROF/OpBasicInfo.csv"
    [ -f "$OPPROF/ArithmeticUtilization.csv" ] && { echo "--- ArithmeticUtilization ---"; cat "$OPPROF/ArithmeticUtilization.csv"; }
    [ -f "$OPPROF/PipeUtilization.csv" ] && { echo "--- PipeUtilization ---"; cat "$OPPROF/PipeUtilization.csv"; }
    cp -r "$OPPROF" ./msprof_latest
  else
    echo "[warn] no OPPROF csv; tail msprof.log:"; tail -20 msprof.log
  fi
  rm -rf "$MSPROF_OUT"
fi
echo "==== [$TAG] DONE ===="
