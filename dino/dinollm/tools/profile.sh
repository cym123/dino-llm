#!/bin/sh
# tools/profile.sh
# Linux OS profiling script for the ML inference server.
# Works with /bin/sh (POSIX), bash, and dash.
# Captures: perf stat CPU counters, /proc memory metrics, taskset CPU-pinning comparison.
#
# Usage:
#   ./tools/profile.sh               # auto-detects server PID via pgrep
#   ./tools/profile.sh 1234          # explicitly pass the server PID
#   ./tools/profile.sh 1234 50       # custom request count
#
# IMPORTANT: replace the literal text "1234" with the actual process ID number.
# Requires: perf, taskset (linux-tools-common), python3 with aiohttp

set -eu

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_PID=${1:-}
if [ -z "$SERVER_PID" ]; then
    SERVER_PID=$(pgrep -f "uvicorn server.app" 2>/dev/null | head -1 || true)
fi
if [ -z "$SERVER_PID" ]; then
    echo "[INFO] No PID supplied and no uvicorn process found."
    echo "[INFO] Pass the server PID explicitly: ./tools/profile.sh <pid>"
    echo "[INFO] To find it: pgrep -a python"
    exit 1
fi

NUM_REQUESTS=${2:-50}
CONCURRENCY=5
MAX_TOKENS=30
SERVER_URL="http://localhost:8000"
REPORT_DIR="./profiling_reports"

mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$REPORT_DIR/profile_${TIMESTAMP}.txt"

echo "============================================"  | tee "$REPORT"
echo " ML Inference System — Linux OS Profile"     | tee -a "$REPORT"
echo " PID: $SERVER_PID | $(date)"                 | tee -a "$REPORT"
echo "  Tip: run 'pgrep -a python' to find PIDs"   | tee -a "$REPORT"
echo "============================================"  | tee -a "$REPORT"
echo ""                                             | tee -a "$REPORT"

# ─── 1. /proc memory snapshot ────────────────────────────────────────────────
echo "=== /proc/${SERVER_PID}/status — Memory Snapshot ===" | tee -a "$REPORT"
if [ -f "/proc/${SERVER_PID}/status" ]; then
    grep -E "^(VmPeak|VmRSS|VmSize|VmSwap|Threads|voluntary_ctxt_switches|nonvoluntary_ctxt_switches)" \
         "/proc/${SERVER_PID}/status" | tee -a "$REPORT"
else
    echo "  [WARN] /proc/${SERVER_PID}/status not found." | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# ─── 2. CPU affinity ──────────────────────────────────────────────────────────
echo "=== CPU Affinity (current) ===" | tee -a "$REPORT"
if command -v taskset >/dev/null 2>&1; then
    taskset -p "$SERVER_PID" | tee -a "$REPORT"
else
    echo "  [WARN] taskset not available. Install: apk add util-linux  OR  apt install util-linux" | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# ─── 3. Benchmark during perf stat ───────────────────────────────────────────
echo "=== Running benchmark ($NUM_REQUESTS requests, concurrency $CONCURRENCY) ===" | tee -a "$REPORT"
if command -v perf >/dev/null 2>&1; then
    echo "  [perf stat attached to PID $SERVER_PID]" | tee -a "$REPORT"
    perf stat -p "$SERVER_PID" \
        -e cycles,instructions,cache-misses,cache-references,branch-misses,context-switches \
        sleep 10 &
    PERF_PID=$!
    python3 benchmark/benchmark.py \
        --concurrency "$CONCURRENCY" \
        --requests "$NUM_REQUESTS" \
        --max_tokens "$MAX_TOKENS" \
        --url "${SERVER_URL}/generate" 2>&1 | tee -a "$REPORT"
    wait "$PERF_PID" 2>&1 | tee -a "$REPORT"
else
    echo "  [WARN] perf not available. Install: apt install linux-tools-generic linux-tools-common" | tee -a "$REPORT"
    python3 benchmark/benchmark.py \
        --concurrency "$CONCURRENCY" \
        --requests "$NUM_REQUESTS" \
        --max_tokens "$MAX_TOKENS" \
        --url "${SERVER_URL}/generate" 2>&1 | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# ─── 4. Post-run memory snapshot ─────────────────────────────────────────────
echo "=== /proc/${SERVER_PID}/status — Memory After Load ===" | tee -a "$REPORT"
if [ -f "/proc/${SERVER_PID}/status" ]; then
    grep -E "^(VmPeak|VmRSS|VmSize|voluntary_ctxt_switches|nonvoluntary_ctxt_switches)" \
         "/proc/${SERVER_PID}/status" | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# ─── 5. CPU pinning comparison ───────────────────────────────────────────────
echo "=== CPU Pinning Comparison ===" | tee -a "$REPORT"
if command -v taskset >/dev/null 2>&1; then
    NCORES=$(nproc 2>/dev/null || echo "4")
    LAST_CORE=$((NCORES - 1))

    echo "  [Baseline — all $NCORES cores]" | tee -a "$REPORT"
    taskset -cp "0-${LAST_CORE}" "$SERVER_PID" >/dev/null 2>&1 || true
    sleep 0.5
    python3 benchmark/benchmark.py \
        --concurrency "$CONCURRENCY" --requests "$NUM_REQUESTS" \
        --max_tokens "$MAX_TOKENS" --url "${SERVER_URL}/generate" 2>&1 | tee -a "$REPORT"

    echo "" | tee -a "$REPORT"
    echo "  [Pinned to cores 0-1 only]" | tee -a "$REPORT"
    taskset -cp 0-1 "$SERVER_PID" >/dev/null 2>&1 || true
    sleep 0.5
    python3 benchmark/benchmark.py \
        --concurrency "$CONCURRENCY" --requests "$NUM_REQUESTS" \
        --max_tokens "$MAX_TOKENS" --url "${SERVER_URL}/generate" 2>&1 | tee -a "$REPORT"

    echo "  [Restoring all cores]" | tee -a "$REPORT"
    taskset -cp "0-${LAST_CORE}" "$SERVER_PID" >/dev/null 2>&1 || true
else
    echo "  [SKIP] taskset not found." | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

echo "============================================" | tee -a "$REPORT"
echo " Profile complete → $REPORT"                 | tee -a "$REPORT"
echo "============================================" | tee -a "$REPORT"
