#!/bin/bash
# ---------------------------------------------------------------------------
# Gate the overnight experiment queue behind the currently-running job.
#
# WHY THIS EXISTS
#   Both clusters have exactly 8 TPU chips and a job already occupying them.
#   experiment_queue.py deletes/creates a resource named "grpo-exp", which does
#   NOT collide by name with the in-flight jobs -- so if the queue started now,
#   its first pod would simply sit Pending forever while the old job held the
#   chips. Rather than hand-timing the launch, this waits for the chips.
#
#   It also deliberately does NOT delete the in-flight job until that job's pod
#   has actually left Running, so we never truncate a run that is mid-training.
#
# Usage: ./start_queue_when_free.sh <borrowed|home>
# ---------------------------------------------------------------------------
set -u

CLUSTER="${1:?usage: start_queue_when_free.sh <borrowed|home>}"
cd "$(dirname "$0")" || exit 1

case "$CLUSTER" in
  borrowed)
    export KUBECONFIG=/tmp/borrowed.kubeconfig
    POD_PREFIX="qwen3-4b-grpo-validation"
    OLD_KIND="job"
    OLD_NAME="qwen3-4b-grpo-validation"
    ;;
  borrowed-adopt)
    # Wait on an experiment the PREVIOUS queue process started. Used when the
    # driver had to be restarted mid-run (bug fix) but the in-flight experiment
    # is healthy and worth keeping. The old driver is killed, which leaves its
    # Job running and unreaped -- we reap it here once its pod finishes.
    export KUBECONFIG=/tmp/borrowed.kubeconfig
    POD_PREFIX="grpo-exp"
    OLD_KIND="job"
    OLD_NAME="grpo-exp"
    CLUSTER="borrowed"
    ;;
  home)
    unset KUBECONFIG   # home is the user's default context -- never switch it
    POD_PREFIX="qwen3-4b-grpo-v6e-8-spot-training"
    OLD_KIND="jobset"
    OLD_NAME="qwen3-4b-grpo-v6e-8-spot-training"
    ;;
  *) echo "unknown cluster $CLUSTER"; exit 1 ;;
esac

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] [gate:$CLUSTER] $*"; }

say "waiting for $POD_PREFIX to stop Running (max 3h)"

DEADLINE=$(( $(date +%s) + 10800 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  # Any pod with this prefix still in Running/Pending/ContainerCreating?
  BUSY=$(kubectl get pods --no-headers 2>/dev/null \
         | awk -v p="$POD_PREFIX" 'index($1,p)==1 {print $3}' \
         | grep -cE 'Running|Pending|Init|ContainerCreating')
  if [ "${BUSY:-0}" -eq 0 ]; then
    say "chips free"
    break
  fi
  sleep 60
done

# Give the final checkpoint save / tfevents flush a moment to land, and let the
# snapshotter take one last pass before the pod object goes away.
say "settling 90s before reclaiming"
sleep 90

say "deleting $OLD_KIND/$OLD_NAME"
kubectl delete "$OLD_KIND" "$OLD_NAME" --ignore-not-found >/dev/null 2>&1
sleep 20

say "launching experiment_queue.py --cluster $CLUSTER"
exec python3 experiment_queue.py --cluster "$CLUSTER"
