#!/usr/bin/env python3
"""Run a queue of GRPO experiments unattended, archiving results as it goes.

Design notes
------------
* Manifests are PATCHED, not regenerated. We load the validated manifest files and
  only replace the training args / names. That keeps the pod spec (TPU selectors,
  tolerations, volumes, sidecars) exactly as validated -- regenerating it by hand is
  how you get a subtle scheduling bug at 3am.

* The borrowed cluster stores metrics on an emptyDir, which is only reachable via
  `kubectl exec` -- and exec STOPS WORKING the moment the pod reaches Succeeded, not
  when it is deleted. So we poll the file continuously and keep the last good copy.
  (This already cost us 22 steps of data once.)

* The home cluster is SPOT and can be preempted mid-run. We treat a vanished pod as
  a retryable event rather than a failure, up to a limit.

* Every experiment archives: the pod log, the tfevents file, and a metrics summary.
  Partial results are kept -- a preempted 30-step run still tells us something.

Usage:
    python3 experiment_queue.py --cluster borrowed
    python3 experiment_queue.py --cluster home
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

# This file lives in training-grpo/tools/, so paths resolve one level up.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARCHIVE = os.path.join(ROOT, "run-logs")
MODELS = os.path.join(ROOT, "models")
SUMMARIZER = os.path.expanduser(
    "~/.gemini/jetski/brain/81537f75-463c-4f8c-8515-c3ebaed1698f/scratch/summarize_run.py"
)

CLUSTERS = {
    "borrowed": {
        "manifest": os.path.join(
            MODELS, "qwen3-4b", "on-demand", "qwen3-4b-grpo-validation-borrowed.yaml"),
        "kubeconfig": "/tmp/borrowed.kubeconfig",
        "kind": "job",
        "container": None,          # single-container pod
        "metrics": "exec",          # emptyDir -> must poll out of the pod
        "max_preemptions": 0,       # on-demand, should not vanish
    },
    "home": {
        "manifest": os.path.join(
            MODELS, "qwen3-4b", "spot", "qwen3-4b-grpo-spot-v6e-8-training.yaml"),
        "kubeconfig": None,         # default context
        "kind": "jobset",
        "container": "grpo-trainer",  # + 2 gcsfuse sidecars
        "metrics": "gcs",           # gcsfuse PVC -> durable in GCS
        "gcs_root": "gs://checkpoint-data-pmotgi-tpu-v7x-04a179d1",
        "max_preemptions": 3,       # spot
    },
}


# --------------------------------------------------------------------------
# Experiment definitions
# --------------------------------------------------------------------------
# The central question: with batch_size=1 each step sees ONE problem, so the reward
# curve plots problem difficulty rather than learning. rl.yml:111 even says
# "Increase batch_size and MAX_STEPS for better results". These sweep the two axes
# that should actually matter -- averaging over problems, and model headroom.

def exp(name, **kw):
    base = dict(
        model="qwen3-4b",
        tokenizer="Qwen/Qwen3-4B",
        outdir="/checkpoint/maxtext-qwen3-4b",
        dataset="openai/gsm8k",
        hf_train_files=None,
        train_split=None,
        maxlen=2048,
        prefill=256,
        hbm=0.6,
        ngen=8,
        batch=1,
        steps=50,
        # Split each batch into micro batches for the training step and the
        # rollout. rl.yml defaults both to -1, which means "do not split" --
        # the whole batch goes through jit__train_step at once. With
        # batch_size=8 and num_generations=8 that is 64 sequences in a single
        # step versus the 8 we actually validated, i.e. an 8x memory blowup.
        # Setting these to 1 keeps the per-step footprint exactly at the
        # validated size while still averaging the gradient over `batch`
        # problems. This is how upstream runs batch_size=128 on the same chips.
        tmicro=1,
        rmicro=1,
        # Arbitrary extra "key=value" MaxText args appended verbatim. Lets an
        # experiment sweep a lever that has no dedicated field above.
        extra=[],
    )
    base.update(kw)
    base["name"] = name
    return base


# Small model on the easy dataset. qwen3-0.6b should have real headroom on GSM8K
# where qwen3-4b was pinned at the ceiling. Cheap and fast, so it runs first --
# if anything produces a clean rising curve, it is this.
QWEN06 = dict(model="qwen3-0.6b", tokenizer="Qwen/Qwen3-0.6B",
              outdir="/checkpoint/maxtext-qwen3-0p6b")

OMI2 = dict(
    dataset="nvidia/OpenMathInstruct-2",
    hf_train_files="hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-00000-of-00003.parquet",
    train_split="train_1M",
)

QUEUES = {
    # Borrowed cluster is on-demand and reliable -> give it the long, high-value runs.
    "borrowed": [
        # 1. Most likely to produce THE demo curve: weak model, easy task, batched.
        #    Runs first and fails fast (~8 min) if qwen3-0.6b is broken outright,
        #    which the first attempt could not tell us because its log was lost.
        #    tmicro=1 is the most conservative possible setting.
        exp("b1r-qwen06-gsm8k-bs8", **QWEN06, batch=8, steps=60),
        # --- revised mid-night, after b1r reported in -------------------------
        # b1r nailed the MEASUREMENT problem: 100% usable gradient, 0% at the
        # ceiling, every step partial credit. But the curve is FLAT --
        # slope -0.00013/step, and KL vs the reference model is only ~0.002
        # after 28 steps. The policy is barely moving. rl.yml's default
        # learning_rate=3e-6 is sized for many hundreds of steps, not the ~60 a
        # workshop can sit through. So the binding constraint is no longer
        # signal quality, it is step size. These two sweep it on the exact
        # recipe b1r validated, changing nothing else.
        exp("b6-qwen06-gsm8k-lr2e5", **QWEN06, batch=8, steps=60,
            extra=["learning_rate=2e-5"]),          # ~7x default
        # --- revised a third time, after b6 (lr=2e-5) came back CLEAN ---------
        # b6 did not collapse: +5% reward, slope +0.00041/step, KL 0.165 (moved
        # meaningfully but stayed on the leash), and answers got SHORTER
        # (715 -> 608 tok) at equal reward. So the collapse threshold sits
        # between 2e-5 and 5e-5, and 2e-5 is usable.
        #
        # That makes a 1e-5 probe pointless -- we no longer need to map the safe
        # band, we need the biggest visible rise. This combines the highest
        # proven-safe learning rate with 2x the steps. 120 steps at ~2.1 min is
        # ~4.2h, which still lands before the user is back.
        #
        # Risk, stated plainly: b6's KL was still climbing at step 59 (0.165).
        # Extrapolated to 120 steps that is ~0.33, and h4 collapsed in the
        # 0.4-1.17 range. This run may therefore collapse late. h5 (150 steps at
        # the conservative 3e-6) is running in parallel on the other cluster as
        # the hedge, so either outcome leaves us with a usable curve.
        exp("b8-qwen06-gsm8k-lr2e5-long", **QWEN06, batch=8, steps=120,
            extra=["learning_rate=2e-5"]),
        # --- added 13:20 UTC, after b8 came back as the headline result -------
        # b8 is the answer: +20%, monotonic across all six segments, 100% usable
        # gradient, 0% truncated. But it costs ~2.1 min/step = ~4.2h, which is
        # longer than a workshop slot.
        #
        # tmicro=2 runs the SAME recipe at ~1.3 min/step (~40% faster) and we
        # know it fits in HBM for 0.6b -- h5 and h6 both ran it without OOM.
        # What we have never seen is tmicro=2 carried to 120 steps, because both
        # runs that used it were on spot and got shredded by preemption (h6's
        # best segment is 49 steps and going backwards).
        #
        # So: the borrowed on-demand cluster is idle and reliable, and I am
        # recommending tmicro=2 in the workshop manifest. Recommending it
        # without ever having seen it finish is exactly the kind of unverified
        # claim that bit us earlier with b2. ~2.6h + ~10 min startup.
        #
        # Expected: the same +20%-ish rise as b8. Gradient accumulation over 4
        # micro-steps instead of 8 is mathematically the same update, so a
        # materially different curve would itself be worth knowing about.
        exp("b9-qwen06-gsm8k-lr2e5-fast", **QWEN06, batch=8, steps=120,
            tmicro=2, rmicro=2, extra=["learning_rate=2e-5"]),
    ],
    # Home cluster is spot and may be preempted -> give it shorter, cheaper runs.
    "home": [
        # --- rewritten mid-night, after b1r and h4 both reported in -----------
        # We now know: lr=3e-6 is stable and rises GENTLY (+7% over 59 steps,
        # slope +0.00039/step), while lr=5e-5 collapses outright. The open
        # question for the workshop is therefore no longer "which LR" but
        # "does the stable setting produce a convincing curve if we simply let
        # it run longer?". At ~1.3 min/step, 150 steps is ~3.2h -- affordable,
        # and 2.5x the longest run we have. If the +7%/59-step trend holds it
        # should land near +18%, which is clearly visible on a plot.
        exp("h5-qwen06-gsm8k-long", **QWEN06, batch=8, steps=150,
            tmicro=2, rmicro=2),
        # --- added after h5 hit a preemption loop -----------------------------
        # h5 was preempted 6 times. Critically, each restart resumed from an
        # OLDER checkpoint (~step 50) than its best segment (step 98), so the
        # run was going backwards, not forwards -- it can never beat the 99-step
        # series already archived. Rather than let it keep thrashing, this is a
        # SHORT cross-cluster reproduction of the recommended configuration on
        # the user's own hardware, which was an explicit ask. 60 steps at
        # ~1.3 min is ~80 min, short enough to have a real chance of surviving
        # spot between preemptions.
        exp("h6-qwen06-gsm8k-lr2e5-repro", **QWEN06, batch=8, steps=60,
            tmicro=2, rmicro=2, extra=["learning_rate=2e-5"]),
        # Control on the model we KNOW works, if there is time left over.
        exp("h2-qwen4b-gsm8k-bs4", batch=4, steps=40),
    ],
}


# --------------------------------------------------------------------------
def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def kube(cfg, *args, check=False, timeout=120):
    env = dict(os.environ)
    if cfg["kubeconfig"]:
        env["KUBECONFIG"] = cfg["kubeconfig"]
    try:
        r = subprocess.run(["kubectl", *args], env=env, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    if check and r.returncode != 0:
        log(f"  kubectl {' '.join(args[:3])} failed: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def build_args(e):
    """Training args. Order mirrors the validated manifests."""
    a = [
        "maxtext/configs/post_train/rl.yml",
        f"model_name={e['model']}",
        f"tokenizer_path={e['tokenizer']}",
        f"run_name={e['name']}",
        f"base_output_directory={e['outdir']}",
        "chips_per_vm=8",
        "data_template_path=maxtext/examples/chat_templates/openmathinstruct2_rl.json",
        f"dataset_name={e['dataset']}",
    ]
    if e["hf_train_files"]:
        a.append(f"hf_train_files={e['hf_train_files']}")
    if e["train_split"]:
        a.append(f"train_split={e['train_split']}")
        a.append(f"eval_dataset_name={e['dataset']}")
    a += [
        f"max_target_length={e['maxlen']}",
        f"max_prefill_predict_length={e['prefill']}",
        f"hbm_utilization_vllm={e['hbm']}",
        f"rl.num_generations={e['ngen']}",
        f"batch_size={e['batch']}",
        f"train_micro_batch_size={e['tmicro']}",
        f"rollout_micro_batch_size={e['rmicro']}",
        "log_period=1",
        f"num_batches={e['steps']}",
        "num_test_batches=0",
        "eval_interval=0",
    ]
    a += list(e.get("extra") or [])
    return ["python3", "-m", "maxtext.trainers.post_train.rl.train_rl"] + a


def render(cfg, e):
    """Patch the validated manifest rather than generating a new one."""
    with open(cfg["manifest"]) as fh:
        doc = yaml.safe_load(fh)
    doc = copy.deepcopy(doc)
    cmd = build_args(e)

    if cfg["kind"] == "job":
        doc["metadata"]["name"] = "grpo-exp"
        spec = doc["spec"]["template"]["spec"]
        spec["containers"][0]["command"] = cmd
    else:
        doc["metadata"]["name"] = "grpo-exp"
        rj = doc["spec"]["replicatedJobs"][0]
        spec = rj["template"]["spec"]["template"]["spec"]
        for c in spec["containers"]:
            if c["name"] == cfg["container"]:
                c["command"] = cmd
    path = f"/tmp/grpo-exp-{e['name']}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, default_flow_style=False)
    return path


def find_pod(cfg):
    out = kube(cfg, "get", "pods", "-n", "default", "--no-headers")
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("grpo-exp"):
            return parts[0], parts[2]
    return None, None


def snapshot_exec(cfg, pod, e, dest):
    """Copy the tfevents file out of a running pod (emptyDir case)."""
    tbdir = f"{e['outdir']}/{e['name']}/tensorboard"
    args = ["exec", pod, "-n", "default"]
    if cfg["container"]:
        args += ["-c", cfg["container"]]
    fname = kube(cfg, *args, "--", "sh", "-c", f"ls {tbdir}/ 2>/dev/null | head -1")
    if not fname:
        return False
    env = dict(os.environ)
    if cfg["kubeconfig"]:
        env["KUBECONFIG"] = cfg["kubeconfig"]
    tmp = dest + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            r = subprocess.run(["kubectl", *args, "--", "cat", f"{tbdir}/{fname}"],
                               env=env, stdout=fh, stderr=subprocess.DEVNULL,
                               timeout=180)
        if r.returncode == 0 and os.path.getsize(tmp) > 0:
            # Never let a short read clobber a longer good snapshot.
            if not os.path.exists(dest) or os.path.getsize(tmp) >= os.path.getsize(dest):
                shutil.move(tmp, dest)
                return True
    except Exception:
        pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return False


def snapshot_gcs(cfg, e, dest):
    """Pull the tfevents file from GCS (gcsfuse PVC case)."""
    sub = e["outdir"].replace("/checkpoint/", "")
    prefix = f"{cfg['gcs_root']}/{sub}/{e['name']}/tensorboard/"
    out = subprocess.run(["gcloud", "storage", "ls", prefix],
                         capture_output=True, text=True, timeout=120)
    files = [l.strip() for l in out.stdout.splitlines() if "tfevents" in l]
    if not files:
        return False
    r = subprocess.run(["gcloud", "storage", "cp", files[-1], dest],
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0


def run_one(cfg, e):
    out = os.path.join(ARCHIVE, e["name"])
    os.makedirs(out, exist_ok=True)
    logf, tbf = os.path.join(out, "pod.log"), os.path.join(out, "events.tfevents")

    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(e, fh, indent=2)

    log(f"=== {e['name']}: {e['model']} {e['dataset'].split('/')[-1]} "
        f"bs={e['batch']} n={e['ngen']} len={e['maxlen']} steps={e['steps']}")

    kube(cfg, "delete", cfg["kind"], "grpo-exp", "-n", "default",
         "--ignore-not-found")
    time.sleep(8)
    kube(cfg, "apply", "-f", render(cfg, e), check=True)

    preemptions = 0
    tailer = None
    deadline = time.time() + 5.5 * 3600   # hard cap so one bad run can't eat the night
    last_pod = None

    while time.time() < deadline:
        pod, phase = find_pod(cfg)

        if pod is None:
            # JobSet recreates pods after preemption; give it a chance.
            if last_pod and preemptions < cfg["max_preemptions"]:
                preemptions += 1
                log(f"  pod vanished (preemption {preemptions}); waiting for restart")
                time.sleep(90)
                continue
            if last_pod:
                log("  pod gone, no restart -- stopping")
                break
            time.sleep(30)
            continue

        if pod != last_pod:
            log(f"  pod {pod} ({phase})")
            last_pod = pod
            if tailer:
                tailer.terminate()
                tailer = None

        # Attach the tailer as soon as the pod is Running. This MUST NOT be
        # nested under `pod != last_pod`: a pod is usually first observed as
        # ContainerCreating, and under the old logic the tailer was then never
        # started at all, so a run that failed later left no log behind.
        if phase == "Running" and tailer is None:
            env = dict(os.environ)
            if cfg["kubeconfig"]:
                env["KUBECONFIG"] = cfg["kubeconfig"]
            largs = ["kubectl", "logs", "-f", pod, "-n", "default"]
            if cfg["container"]:
                largs += ["-c", cfg["container"]]
            tailer = subprocess.Popen(largs, env=env,
                                      stdout=open(logf, "ab"),
                                      stderr=subprocess.STDOUT)
            log(f"  tailing {pod} -> {logf}")

        if phase == "Running":
            if cfg["metrics"] == "exec":
                snapshot_exec(cfg, pod, e, tbf)
        elif phase in ("Succeeded", "Completed"):
            log(f"  {e['name']} COMPLETED")
            break
        elif phase in ("Failed", "Error"):
            log(f"  {e['name']} FAILED")
            break

        time.sleep(45)
    else:
        log(f"  {e['name']} hit the 5.5h cap; moving on")

    if tailer:
        time.sleep(5)
        tailer.terminate()

    # --- post-mortem, BEFORE the resource is deleted -----------------------
    # The tailer can miss the fatal traceback (it dies when the pod leaves
    # Running, and the last chunk is often flushed at exit). Once we delete the
    # resource the log is gone for good, so always take a final pull.
    if last_pod:
        args = ["logs", last_pod, "-n", "default", "--tail=3000"]
        if cfg["container"]:
            args += ["-c", cfg["container"]]
        final = kube(cfg, *args, timeout=180)
        if final:
            with open(os.path.join(out, "final_log.txt"), "w") as fh:
                fh.write(final)
            bad = [l for l in final.splitlines()
                   if any(k in l for k in ("Error", "error:", "Exception",
                                           "Traceback", "RESOURCE_EXHAUSTED",
                                           "OOM", "Attempting to reserve"))]
            if bad:
                with open(os.path.join(out, "ERRORS.txt"), "w") as fh:
                    fh.write("\n".join(bad[-80:]))
                log("  captured error lines -> ERRORS.txt")
                for l in bad[-6:]:
                    log(f"    | {l[:160]}")
        desc = kube(cfg, "describe", "pod", last_pod, "-n", "default", timeout=120)
        if desc:
            with open(os.path.join(out, "describe.txt"), "w") as fh:
                fh.write(desc)

    if cfg["metrics"] == "gcs":
        snapshot_gcs(cfg, e, tbf)

    # Summarize whatever we got, even if partial.
    if os.path.exists(tbf):
        budget = e["maxlen"] - e["prefill"]
        r = subprocess.run([sys.executable, SUMMARIZER, e["name"], tbf, str(budget)],
                           capture_output=True, text=True)
        with open(os.path.join(out, "RESULT.txt"), "w") as fh:
            fh.write(r.stdout)
        print(r.stdout, flush=True)
    else:
        log("  no metrics captured")

    kube(cfg, "delete", cfg["kind"], "grpo-exp", "-n", "default",
         "--ignore-not-found")
    time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True, choices=list(CLUSTERS))
    # The driver has to be restarted whenever the queue definition changes, and
    # a naive restart begins again at index 0 -- which once re-ran a completed
    # 2-hour experiment and started overwriting its archive. These let a
    # restart resume at the right place.
    ap.add_argument("--start-at", default=None,
                    help="Skip experiments until this name, then run from there.")
    ap.add_argument("--skip", default="",
                    help="Comma-separated experiment names to skip entirely.")
    ap.add_argument("--rerun-done", action="store_true",
                    help="Re-run experiments that already have a RESULT.txt.")
    a = ap.parse_args()
    cfg = CLUSTERS[a.cluster]

    todo = list(QUEUES[a.cluster])
    if a.start_at:
        names = [e["name"] for e in todo]
        if a.start_at not in names:
            log(f"FATAL: --start-at {a.start_at} not in queue {names}")
            return
        todo = todo[names.index(a.start_at):]
    skip = {s for s in a.skip.split(",") if s}
    todo = [e for e in todo if e["name"] not in skip]

    # Belt and braces: never silently redo something already archived.
    if not a.rerun_done:
        kept = []
        for e in todo:
            if os.path.exists(os.path.join(ARCHIVE, e["name"], "RESULT.txt")):
                log(f"  skipping {e['name']} (already has RESULT.txt)")
            else:
                kept.append(e)
        todo = kept

    log(f"queue start: {a.cluster} ({len(todo)} experiments: "
        f"{', '.join(e['name'] for e in todo)})")
    for e in todo:
        try:
            run_one(cfg, e)
        except Exception as exc:            # never let one experiment kill the queue
            log(f"  {e['name']} raised {type(exc).__name__}: {exc}")
    log(f"queue done: {a.cluster}")


if __name__ == "__main__":
    main()
