#!/usr/bin/env bash
# capture_job_logs.sh -- durable log + metadata capture for GRPO JobSets.
#
# Why this exists: TPU jobs here run on spot/DWS nodes that can vanish at any
# moment, and `kubectl logs` only serves logs while the pod object still exists.
# Once a pod is garbage-collected (or its node is deleted by a preemption) the
# logs are gone. This script continuously follows every pod of the watched
# JobSets and archives the stream to disk, so both failed and SUCCESSFUL runs
# leave behind a permanent record.
#
# Usage:
#   ./capture_job_logs.sh                      # watch the default JobSet list
#   ./capture_job_logs.sh jobset-a jobset-b    # watch specific JobSets
#   OUTDIR=/some/path ./capture_job_logs.sh    # override output location
#
# Layout produced under $OUTDIR:
#   <jobset>/
#     pods/<pod>.log        full timestamped container log
#     pods/<pod>.describe   pod spec/status at capture time
#     pods/<pod>.result     terminal phase + exit code + reason
#     events.log            rolling JobSet/pod events (preemptions, scale-ups)
#     run_args.txt          the exact training args the JobSet was submitted with
#     SUMMARY.md            human-readable index of every attempt and outcome
#   index.md                top-level index across all watched JobSets

set -uo pipefail

NS="${NS:-default}"
CONTAINER="${CONTAINER:-grpo-trainer}"
# This script lives in training-grpo/tools/, so run-logs is one level up.
OUTDIR="${OUTDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run-logs}"
POLL="${POLL:-15}"

JOBSETS=("$@")
if [ ${#JOBSETS[@]} -eq 0 ]; then
  JOBSETS=(
    qwen3-4b-grpo-v6e-8-spot-training
    llama3-1-8b-grpo-v6e-8-spot-training
    llama3-1-8b-grpo-v6e-8-training
  )
fi

mkdir -p "$OUTDIR"
echo "[capture] namespace=$NS container=$CONTAINER outdir=$OUTDIR"
echo "[capture] watching: ${JOBSETS[*]}"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Record the training args a JobSet was submitted with, so a captured log can
# always be tied back to the exact configuration that produced it.
snapshot_args() {
  local js="$1" dir="$2"
  [ -s "$dir/run_args.txt" ] && return 0
  kubectl get jobset "$js" -n "$NS" \
    -o jsonpath='{.spec.replicatedJobs[0].template.spec.template.spec.containers[0].command}' 2>/dev/null \
    | tr ',' '\n' | sed 's/^[][ "]*//; s/[][ "]*$//' > "$dir/run_args.txt" 2>/dev/null
  {
    echo "# captured $(ts)"
    kubectl get jobset "$js" -n "$NS" -o jsonpath='image: {.spec.replicatedJobs[0].template.spec.template.spec.containers[0].image}{"\n"}' 2>/dev/null
    kubectl get jobset "$js" -n "$NS" -o jsonpath='nodeSelector: {.spec.replicatedJobs[0].template.spec.template.spec.nodeSelector}{"\n"}' 2>/dev/null
  } >> "$dir/run_args.txt" 2>/dev/null
}

# Persist the pod's final disposition. This is what distinguishes a successful
# run from one that was preempted or OOM-killed, long after the pod is gone.
record_result() {
  local pod="$1" dir="$2"
  local phase reason exitcode
  phase=$(kubectl get pod "$pod" -n "$NS" -o jsonpath='{.status.phase}' 2>/dev/null)
  [ -z "$phase" ] && phase="Gone(pod deleted)"
  reason=$(kubectl get pod "$pod" -n "$NS" -o jsonpath="{range .status.containerStatuses[?(@.name=='$CONTAINER')]}{.state.terminated.reason}{end}" 2>/dev/null)
  exitcode=$(kubectl get pod "$pod" -n "$NS" -o jsonpath="{range .status.containerStatuses[?(@.name=='$CONTAINER')]}{.state.terminated.exitCode}{end}" 2>/dev/null)
  {
    echo "captured_at: $(ts)"
    echo "pod: $pod"
    echo "phase: $phase"
    echo "terminated_reason: ${reason:-n/a}"
    echo "exit_code: ${exitcode:-n/a}"
    # Classify the outcome from the log body -- the signals we actually care about.
    # NOTE: check for SUCCESS markers *first*. `total_hbm_limit_cap_gb` is printed by
    # every healthy run too, so testing it early mislabels good runs as failures.
    if [ -s "$dir/pods/$pod.log" ]; then
      if grep -q "Finished save (blocking + background)" "$dir/pods/$pod.log"; then
        echo "diagnosis: SUCCESS (trained + checkpoint saved)"
      elif grep -q "RESOURCE_EXHAUSTED\|Error allocating device buffer" "$dir/pods/$pod.log"; then
        echo "diagnosis: OOM (HBM exhausted)"
      elif grep -q "total_hbm_limit_cap_gb" "$dir/pods/$pod.log" \
           && grep -q "Compilation failure\|failed to allocate" "$dir/pods/$pod.log"; then
        echo "diagnosis: vLLM HBM cap below resident model size"
      elif [ "${exitcode:-1}" = "0" ]; then
        echo "diagnosis: SUCCESS"
      else
        echo "diagnosis: terminated (see log tail; often spot preemption)"
      fi
      echo "log_lines: $(wc -l < "$dir/pods/$pod.log" | tr -d ' ')"
    fi
  } > "$dir/pods/$pod.result"
}

# Classify any archived log that has no .result yet. This covers logs recovered
# from pods that were already gone (preempted/GC'd) before capture attached --
# without it those attempts would be missing from SUMMARY.md entirely.
backfill_results() {
  local dir="$1"
  local logf pod
  for logf in "$dir"/pods/*.log; do
    [ -e "$logf" ] || continue
    pod=$(basename "$logf" .log)
    [ -e "$dir/pods/$pod.result" ] && continue
    # Skip pods we are actively streaming: follow_pod writes the authoritative
    # result when the stream ends. Backfilling here would mislabel a live run
    # as "Gone".
    [ -e "${MARKERDIR:-/nonexistent}/$pod" ] && continue
    {
      echo "captured_at: $(ts)"
      echo "pod: $pod"
      echo "phase: Gone(backfilled from archived log)"
      echo "terminated_reason: n/a"
      echo "exit_code: n/a"
      if grep -q "Finished save (blocking + background)" "$logf"; then
        echo "diagnosis: SUCCESS (trained + checkpoint saved)"
      elif grep -q "RESOURCE_EXHAUSTED\|Error allocating device buffer" "$logf"; then
        echo "diagnosis: OOM (HBM exhausted)"
      elif grep -q "total_hbm_limit_cap_gb" "$logf" \
           && grep -q "Compilation failure\|failed to allocate" "$logf"; then
        echo "diagnosis: vLLM HBM cap below resident model size"
      elif grep -qiE "train_loss|rewards/|kl_div|Train step [0-9]" "$logf"; then
        echo "diagnosis: reached training steps"
      else
        echo "diagnosis: incomplete (no terminal marker; likely preempted)"
      fi
      echo "log_lines: $(wc -l < "$logf" | tr -d ' ')"
    } > "$dir/pods/$pod.result"
  done
}

regenerate_summary() {
  local js="$1" dir="$2"
  backfill_results "$dir"
  {
    echo "# $js"
    echo
    echo "_Updated $(ts)_"
    echo
    echo "| Attempt (pod) | Phase | Exit | Diagnosis | Log |"
    echo "|---|---|---|---|---|"
    for r in "$dir"/pods/*.result; do
      [ -e "$r" ] || continue
      local p ph ec dg
      p=$(basename "$r" .result)
      ph=$(grep '^phase:' "$r" | cut -d' ' -f2-)
      ec=$(grep '^exit_code:' "$r" | cut -d' ' -f2-)
      dg=$(grep '^diagnosis:' "$r" | cut -d' ' -f2-)
      echo "| \`${p##*-}\` | $ph | $ec | $dg | [log](pods/$p.log) |"
    done
  } > "$dir/SUMMARY.md"
}

# Follow one pod to completion, archiving stdout/stderr as it streams.
follow_pod() {
  local js="$1" pod="$2" dir="$3"
  local logf="$dir/pods/$pod.log"
  mkdir -p "$dir/pods"
  echo "[capture $(ts)] attaching $js/$pod"
  kubectl get pod "$pod" -n "$NS" -o yaml > "$dir/pods/$pod.describe" 2>/dev/null
  # --timestamps keeps ordering meaningful across restarts; append so a
  # re-attach after a transient disconnect never truncates earlier output.
  kubectl logs -f "$pod" -n "$NS" -c "$CONTAINER" --timestamps >> "$logf" 2>&1
  echo "[capture $(ts)] stream ended $js/$pod"
  record_result "$pod" "$dir"
  regenerate_summary "$js" "$dir"
}

# Track which pods are already being followed with marker files rather than a
# bash 4 associative array: macOS ships bash 3.2, and marker files additionally
# make the capture resume-safe if this script itself is restarted.
MARKERDIR="$OUTDIR/.following"
mkdir -p "$MARKERDIR"
# Markers denote "a follow_pod child of *this* process is streaming that pod", so
# they are meaningless once the daemon exits. Clear them on startup: leftovers from
# a previous run would permanently suppress backfill for those pods (their .result
# would never be written and they'd vanish from SUMMARY.md). Nothing is streaming
# yet at this point, so this cannot race with a live follower.
rm -f "$MARKERDIR"/* 2>/dev/null || true

while true; do
  for js in "${JOBSETS[@]}"; do
    dir="$OUTDIR/$js"
    mkdir -p "$dir/pods"
    kubectl get jobset "$js" -n "$NS" >/dev/null 2>&1 || continue
    snapshot_args "$js" "$dir"

    # Archive events too: preemptions and scale-up failures only ever show up
    # here, never in the container log.
    kubectl get events -n "$NS" --sort-by=.lastTimestamp 2>/dev/null \
      | grep -- "$js" > "$dir/events.log" 2>/dev/null

    for pod in $(kubectl get pods -n "$NS" \
        -o jsonpath="{range .items[?(@.metadata.labels.jobset\.sigs\.k8s\.io/jobset-name=='$js')]}{.metadata.name}{' '}{.status.phase}{'\n'}{end}" 2>/dev/null \
        | awk '$2=="Running"||$2=="Succeeded"||$2=="Failed"{print $1}'); do
      if [ ! -e "$MARKERDIR/$pod" ]; then
        touch "$MARKERDIR/$pod"
        follow_pod "$js" "$pod" "$dir" &
      fi
    done
    regenerate_summary "$js" "$dir"
  done

  {
    echo "# GRPO run log index"
    echo
    echo "_Updated $(ts)_"
    echo
    for js in "${JOBSETS[@]}"; do
      [ -d "$OUTDIR/$js" ] || continue
      n=$(ls "$OUTDIR/$js"/pods/*.log 2>/dev/null | wc -l | tr -d ' ')
      echo "- [$js]($js/SUMMARY.md) — $n captured attempt(s)"
    done
  } > "$OUTDIR/index.md"

  sleep "$POLL"
done
