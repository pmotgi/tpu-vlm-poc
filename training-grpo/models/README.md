# Manifests by model

Layout: `models/<model>/<capacity-type>/`

`spot` = ran on **your** cluster (`pmotgi-tpu-v7x`, preemptible).
`on-demand` = ran on the **borrowed** cluster (`borrowed-gke-1`, uninterruptible).

That split is not cosmetic. The two environments differ in ways that change the
manifest itself:

| | your cluster (spot) | borrowed cluster (on-demand) |
|---|---|---|
| Workload kind | `JobSet` (CRD installed) | plain `batch/v1 Job` (**no JobSet CRD**) |
| Containers per pod | **3** (`grpo-trainer` + 2 gcsfuse sidecars) → needs `-c grpo-trainer` | 1 → no `-c` needed |
| `/checkpoint` | gcsfuse PVC → metrics **durable in GCS** | `emptyDir` → metrics **vanish with the pod** |
| Service account | `workload-identity-k8s-sa` | none (omitted) |
| Interruptions | **9 preemptions overnight** | **zero** in 10+ hours |

---

## qwen3-0.6b — ⭐ the workshop model

| File | Status |
|---|---|
| [`spot/qwen3-0.6b-grpo-workshop-v6e-8.yaml`](spot/qwen3-0.6b-grpo-workshop-v6e-8.yaml) | ❌ Failed on spot — exhausted `maxRestarts: 5` before finishing a step |
| [`on-demand/qwen3-0.6b-grpo-workshop-portable.yaml`](on-demand/qwen3-0.6b-grpo-workshop-portable.yaml) | ✅ **Completed, exit 0** — 120 steps, 3h24m, zero OOM |

Recipe: `batch_size=8 · train_micro_batch_size=1 · lr=2e-5 · 120 steps`
Result: **+20% reward, monotonic across all six segments, 100% usable gradient,
0% truncated.**

> The two files are the *same recipe*. The spot one failed purely on
> preemption, not configuration. **Use the on-demand one for the workshop**, or
> move your pool to on-demand/flex-start.

---

## qwen3-4b — tested, not recommended for a training demo

| File | Status |
|---|---|
| [`spot/qwen3-4b-grpo-spot-v6e-8-training.yaml`](spot/qwen3-4b-grpo-spot-v6e-8-training.yaml) | OpenMathInstruct-2, 50 steps, `bs=1` |
| [`on-demand/qwen3-4b-grpo-validation-borrowed.yaml`](on-demand/qwen3-4b-grpo-validation-borrowed.yaml) | portable copy of the above (validation run) |
| [`on-demand/qwen3-4b-probe-borrowed.yaml`](on-demand/qwen3-4b-probe-borrowed.yaml) | **experiment, not a workshop file** — see below |

4B is *too good* at GSM8K: it scores a perfect 0.3667 on ~50% of steps, so all
8 sampled answers agree and there is no gradient. Best case on the harder
OMI2 dataset was **16% usable gradient** vs 0.6B's **100%**.

`qwen3-4b-probe-borrowed.yaml` tests the one 4B config never tried —
`bs=8 + train_micro_batch_size=1 + lr=2e-5` — because that verdict predates both
of those fixes. **Do not use it as a workshop manifest.**

---

## llama3.1-8b — TPU v7x only

| Folder | Contents | Topology |
|---|---|---|
| `spot/` | `llama3.1-8b-grpo-spot-2x2x1-training.yaml` | tpu7x `2x2x1`, 4 chips |
| `spot/` | `llama3.1-8b-grpo-spot-1t-training.yaml` | tpu7x `1x1x1` |
| `dws-flex/` | `llama3.1-8b-grpo-dws-2x2x1-training.yaml` | tpu7x `2x2x1`, DWS |

These are the v7x runs that worked. **The v6e-8 variants were deleted** —
they never trained a single step, and neither did the
`tunix-adapter-patch-configmap.yaml` runtime patch written to try to rescue
them.

> [!NOTE]
> **Why 8B cannot work on a single v6e-8, so nobody retries it:**
> the rollout needs tensor parallelism across all 8 chips (TP=8). TP shards the
> 128256-entry vocabulary, which breaks `tpu_inference`'s JAX sampler. The
> trainer/sampler split that would avoid this requires `num_vms >= 2` with
> Pathways, and a single-host slice has `num_vms == 1`.
> **There is no config-only workaround.** Use 2 hosts, or a smaller model.
>
> Raw evidence is preserved in `../run-logs/llama3-1-8b-grpo-v6e-8-*/`.

---

## Not here

- `../run-logs/` — archived results for every run (kept flat; runs span models)
- `../tools/` — the experiment driver and log/metric capture scripts
- `../docker/` — the container build (`my-maxtext-runner:grpo-v2`)
