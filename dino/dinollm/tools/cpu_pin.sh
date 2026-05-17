#!/bin/sh
# tools/cpu_pin.sh
# Benchmark the inference server with vs without CPU pinning using taskset.
# Works with /bin/sh (POSIX), bash, and dash.
#
# Usage (Linux only):
#   ./tools/cpu_pin.sh               # auto-detects server PID
#   ./tools/cpu_pin.sh 1234          # explicit PID
#
# Install taskset: apt install util-linux  OR  apk add util-linux

set -eu

SERVER_PID=${1:-}
if [ -z "$SERVER_PID" ]; then
    SERVER_PID=$(pgrep -f "uvicorn server.app" 2>/dev/null | head -1 || true)
fi
if [ -z "$SERVER_PID" ]; then
    echo "[ERROR] No PID found. Pass it explicitly: ./tools/cpu_pin.sh <pid>"
    echo "        To find it: pgrep -a python"
    exit 1
fi

NUM_REQUESTS=${2:-50}
CONCURRENCY=${3:-5}
MAX_TOKENS=30
SERVER_URL="http://localhost:8000/generate"
NCORES=$(nproc 2>/dev/null || echo "4")
LAST_CORE=$((NCORES - 1))

echo "======================================================="
echo "  CPU Pinning Benchmark  |  PID: $SERVER_PID"
echo "  Total CPU cores: $NCORES"
echo "======================================================="

if ! command -v taskset >/dev/null 2>&1; then
    echo "[ERROR] taskset not found."
    echo "  Install: apt install util-linux   (Debian/Ubuntu)"
    echo "  Install: apk add util-linux        (Alpine)"
    exit 1
fi

echo ""
echo "[BASELINE] No CPU pinning (all $NCORES cores)..."
taskset -cp "0-${LAST_CORE}" "$SERVER_PID" >/dev/null 2>&1 || true
sleep 1

python3 benchmark/benchmark.py \
    --concurrency "$CONCURRENCY" \
    --requests "$NUM_REQUESTS" \
    --max_tokens "$MAX_TOKENS" \
    --url "$SERVER_URL"

echo ""
echo "[PINNED-2] Restricting server to CPU cores 0-1 only..."
taskset -cp 0-1 "$SERVER_PID"
sleep 1

python3 benchmark/benchmark.py \
    --concurrency "$CONCURRENCY" \
    --requests "$NUM_REQUESTS" \
    --max_tokens "$MAX_TOKENS" \
    --url "$SERVER_URL"

echo ""
echo "[RESTORE] Restoring to all $NCORES cores..."
taskset -cp "0-${LAST_CORE}" "$SERVER_PID" >/dev/null 2>&1 || true

echo ""
echo "======================================================="
echo "  Lower latency when PINNED → better cache locality."
echo "  Higher latency when PINNED → not enough parallelism."
echo "======================================================="
