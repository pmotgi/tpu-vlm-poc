#!/usr/bin/env bash
# Continuously snapshot a run's TensorBoard event file off a pod.
#
# WHY THIS EXISTS
# On the borrowed cluster /checkpoint is an emptyDir, so metrics only live inside
# the pod. The file can only be read with `kubectl exec`, and exec stops working
# the moment the pod reaches Succeeded -- NOT when it is deleted. On the previous
# run that cost us steps 28-50 of the reward series: the job finished between two
# manual checks and the data was gone before it could be copied.
#
# So: poll on a short interval and keep overwriting a local copy. The last
# successful poll before the pod terminates is what we keep.
#
# Usage: ./snapshot_tb.sh <pod> <run_name> [interval_seconds]
set -uo pipefail

POD="${1:?usage: snapshot_tb.sh <pod> <run_name> [interval]}"
RUN="${2:?usage: snapshot_tb.sh <pod> <run_name> [interval]}"
INTERVAL="${3:-45}"

KUBECONFIG_FILE="${KUBECONFIG:-/tmp/borrowed.kubeconfig}"
NS="${NS:-default}"
# Base checkpoint dir differs per model (qwen3-0.6b uses its own directory so it
# cannot collide with the 4B Orbax cache). Override with BASE=... if needed.
BASE="${BASE:-/checkpoint/maxtext-qwen3-4b}"
TBDIR="${BASE}/${RUN}/tensorboard"
# Anchor on the repo layout (tools/ -> ..) so this works from any CWD.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO_ROOT}/run-logs/${RUN}"
mkdir -p "$OUT"

echo "[snapshot] pod=$POD run=$RUN interval=${INTERVAL}s -> $OUT/events.tfevents"

last_size=0
while true; do
  phase=$(KUBECONFIG="$KUBECONFIG_FILE" kubectl get pod "$POD" -n "$NS" \
            -o jsonpath='{.status.phase}' 2>/dev/null)

  if [ -z "$phase" ]; then
    echo "[snapshot $(date -u +%H:%M:%S)] pod gone; stopping (kept ${last_size}B)"
    break
  fi

  # Only try to read while the container can still be exec'd into.
  if [ "$phase" = "Running" ]; then
    fname=$(KUBECONFIG="$KUBECONFIG_FILE" kubectl exec "$POD" -n "$NS" -- \
              sh -c "ls $TBDIR/ 2>/dev/null | head -1" 2>/dev/null | tr -d '\r')
    if [ -n "$fname" ]; then
      # Write to a temp file first so a mid-copy failure cannot truncate a good
      # snapshot we already have.
      if KUBECONFIG="$KUBECONFIG_FILE" kubectl exec "$POD" -n "$NS" -- \
           cat "$TBDIR/$fname" > "$OUT/.events.tmp" 2>/dev/null; then
        size=$(wc -c < "$OUT/.events.tmp" | tr -d ' ')
        if [ "$size" -gt 0 ] && [ "$size" -ge "$last_size" ]; then
          mv "$OUT/.events.tmp" "$OUT/events.tfevents"
          last_size=$size
          echo "[snapshot $(date -u +%H:%M:%S)] ${size}B"
        fi
      fi
    fi
  else
    echo "[snapshot $(date -u +%H:%M:%S)] phase=$phase; stopping (kept ${last_size}B)"
    break
  fi

  sleep "$INTERVAL"
done

rm -f "$OUT/.events.tmp"
echo "[snapshot] final: $OUT/events.tfevents ($last_size bytes)"
