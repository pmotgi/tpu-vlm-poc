#!/usr/bin/env bash
# Run a set of single-host TPU GRPO manifests strictly serially and archive
# each pod's timestamped log for analysis with analyze_grpo_perf.py.
#
#   ./run_grpo_perf_ab.sh <outdir-root> <manifest> [<manifest> ...]
#
# Example:
#   ./run_grpo_perf_ab.sh ../run-logs \
#       /tmp/control-30.yaml \
#       ../models/qwen3-0.6b/on-demand/qwen3-0.6b-grpo-portable-splash.yaml \
#       ../models/qwen3-0.6b/on-demand/qwen3-0.6b-grpo-portable-splash-remat-full.yaml
#
# Env: KCTX (kube context), NS (namespace, default "default"),
#      TPU_NODE (node to health-check), MAX_RUN_SECONDS (default 9000)
#
# -----------------------------------------------------------------------------
# WHY THIS IS MORE PARANOID THAN IT LOOKS
# -----------------------------------------------------------------------------
# Three failure modes were hit for real while producing the numbers in
# models/qwen3-0.6b/README.md Part 8. Each silently corrupts results rather
# than erroring, so each is defended against here:
#
#  1. `kubectl logs -f` RETURNS CLEANLY on a transient API-server disconnect,
#     long before the pod is done. A driver that treats that as "run finished"
#     will delete a healthy job mid-run (this happened at 40s), or record a
#     30-step run as having completed 1 step (this happened too -- the pod ran
#     to completion and the data was recoverable only post-hoc).
#     => Completion is decided ONLY by the pod's status.phase.
#
#  2. On reconnect, `kubectl logs -f` RESTARTS FROM LINE 1. Concatenating
#     stream fragments therefore duplicates everything -- a 30-step run was
#     archived as 90 steps, which the analyzer happily parsed.
#     => The final non-follow snapshot is authoritative; fragments are a
#        fallback only when the snapshot is empty.
#
#  3. TPU devices can go UNHEALTHY between runs
#     ("Allocate failed due to no healthy devices present"). The pod then sits
#     Pending for an hour and fails, burning the slot.
#     => Node allocatable TPU count is checked before each submission.
#
# Runs are serial because a single-host TPU slice fits exactly one of these
# jobs; overlapping them would both fail to schedule AND invalidate the
# comparison.

set -uo pipefail

KCTX="${KCTX:-$(kubectl config current-context)}"
NS="${NS:-default}"
TPU_NODE="${TPU_NODE:-}"
MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-9000}"
SETTLE_SECONDS="${SETTLE_SECONDS:-90}"

K="kubectl --context=$KCTX -n $NS"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*"; }
count_steps() { [ -s "$1" ] && grep -c 'Train step ' "$1" 2>/dev/null || echo 0; }

[ $# -lt 2 ] && { sed -n '2,14p' "$0"; exit 1; }
ROOT="$1"; shift

check_tpu_health() {
  [ -z "$TPU_NODE" ] && return 0
  local alloc
  alloc=$(kubectl --context="$KCTX" get node "$TPU_NODE" \
          -o jsonpath='{.status.allocatable.google\.com/tpu}' 2>/dev/null)
  say "node $TPU_NODE allocatable google.com/tpu = ${alloc:-unknown}"
  [ -n "$alloc" ] && [ "$alloc" != "0" ]
}

run_one() {
  local manifest="$1"
  local job app out
  job=$(awk '/^metadata:/{m=1} m&&/^  name:/{print $2; exit}' "$manifest")
  app="$job"
  out="$ROOT/$(basename "$manifest" .yaml)"
  rm -rf "$out"; mkdir -p "$out"
  cp "$manifest" "$out/manifest.yaml"

  say "=== START $(basename "$manifest") (job=$job) ==="
  if ! check_tpu_health; then
    say "SKIP: TPU devices not healthy on $TPU_NODE"; return 1
  fi

  $K delete job "$job" --ignore-not-found --wait=true >/dev/null 2>&1
  sleep 10
  $K apply -f "$manifest" > "$out/apply.txt" 2>&1 || { cat "$out/apply.txt"; return 1; }

  local pod="" phase="" waited=0
  while [ $waited -lt 2400 ]; do
    pod=$($K get pods -l "app=$app" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$pod" ]; then
      phase=$($K get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)
      [ -n "$phase" ] && [ "$phase" != "Pending" ] && break
    fi
    sleep 10; waited=$((waited+10))
  done
  if [ -z "$pod" ]; then
    say "NO POD after ${waited}s"
    $K get events --sort-by=.lastTimestamp | tail -40 > "$out/events.txt"
    return 1
  fi
  say "pod=$pod phase=$phase (waited ${waited}s)"
  echo "$pod" > "$out/pod_name.txt"

  # Defence (1)+(2): follower reconnects into numbered parts; never authoritative.
  ( i=1
    while [ ! -f "$out/.stop" ]; do
      $K logs -f "$pod" --timestamps >> "$out/pod.log.part$i" 2>>"$out/follow.err"
      [ -f "$out/.stop" ] && break
      say "  (follower reconnecting; part$i ended -- NOT treating as completion)"
      i=$((i+1)); sleep 5
    done ) &
  local follower=$!

  local elapsed=0 empties=0
  while [ $elapsed -lt "$MAX_RUN_SECONDS" ]; do
    phase=$($K get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ -z "$phase" ]; then
      empties=$((empties+1))
      say "  empty phase read ($empties/6) -- transient, not concluding"
      [ $empties -ge 6 ] && { say "  pod really is gone"; break; }
    else
      empties=0
      case "$phase" in Succeeded|Failed) say "  terminal phase: $phase"; break ;; esac
    fi
    [ $((elapsed % 300)) -eq 0 ] && say "  ...${elapsed}s phase=${phase:-?} steps=$(cat "$out"/pod.log.part* 2>/dev/null | grep -c 'Train step ')"
    sleep 15; elapsed=$((elapsed+15))
  done

  # Snapshot BEFORE deleting anything -- `kubectl logs` stops working once the
  # pod object is gone.
  $K logs "$pod" --timestamps > "$out/pod.log.final" 2>/dev/null
  touch "$out/.stop"; kill "$follower" 2>/dev/null; wait "$follower" 2>/dev/null

  local n_final; n_final=$(count_steps "$out/pod.log.final")
  if [ "$n_final" -gt 0 ]; then
    cp "$out/pod.log.final" "$out/pod.log"
    say "  using final snapshot ($n_final steps)"
  else
    cat "$out"/pod.log.part* > "$out/pod.log" 2>/dev/null
    say "  WARNING snapshot empty; using stream fragments ($(count_steps "$out/pod.log") steps, MAY CONTAIN DUPLICATES)"
  fi

  {
    echo "captured_at: $(ts)"
    echo "job: $job"
    echo "pod: $pod"
    echo "phase: $($K get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)"
    echo "exit_code: $($K get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null)"
    echo "reason: $($K get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state.terminated.reason}' 2>/dev/null)"
    echo "train_steps_logged: $(count_steps "$out/pod.log")"
    echo "attention: $(grep -m1 'Config param attention:' "$out/pod.log" | sed 's/.*attention: //')"
    echo "remat_policy: $(grep -m1 'Config param remat_policy:' "$out/pod.log" | sed 's/.*remat_policy: //')"
  } > "$out/RESULT.txt"
  cat "$out/RESULT.txt"

  $K get events --sort-by=.lastTimestamp 2>/dev/null | grep -- "$job" > "$out/events.txt"
  rm -f "$out"/pod.log.part* "$out/.stop"
  $K delete job "$job" --ignore-not-found --wait=true >/dev/null 2>&1
  say "=== END $(basename "$manifest") ==="
  sleep "$SETTLE_SECONDS"   # let the TPU device plugin release the chips
}

mkdir -p "$ROOT"
say "context=$KCTX ns=$NS outdir=$ROOT runs=$#"
for m in "$@"; do run_one "$m"; done
say "ALL RUNS COMPLETE"
echo
echo "Analyse with:"
echo "  python3 $(dirname "$0")/analyze_grpo_perf.py \\"
for m in "$@"; do
  echo "      $(basename "$m" .yaml)=$ROOT/$(basename "$m" .yaml)/pod.log \\"
done
