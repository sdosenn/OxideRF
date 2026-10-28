#!/usr/bin/env bash
# Keep each output for byte-level comparisons with the same RF input.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo 'Usage: scripts/benchmark-native.sh CAPTURE OUTPUT_DIRECTORY [FRAMES=0] [REPEATS=3]' >&2
    exit 1
fi
capture="$1"
results="$2"
frames="${3:-0}"
repeats="${4:-3}"
binary="${VHS_BINARY:-./target/release/oxide-rf}"
[[ -f "$capture" && -x "$binary" ]] || { echo 'Capture or executable not found.' >&2; exit 1; }
[[ "$frames" =~ ^[0-9]+$ && "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo 'Frames and repeats must be nonnegative/positive integers.' >&2; exit 1; }
mkdir -p "$results"
for backend in cpu opencl; do
    for ((run=1; run<=repeats; run++)); do
        base="$results/$backend-$run"
        # Exclusive log creation also prevents reusing previous benchmark results.
        (set -o noclobber; : > "$base.log")
        "$binary" --decode-native --backend "$backend" --input "$capture" \
            --output "$base" --frequency 28.636363636363637 --frames "$frames" \
            > "$base.log" 2>&1
        printf '%s run %s: ' "$backend" "$run"
        rg '^PERFORMANCE ' "$base.log"
    done
done
