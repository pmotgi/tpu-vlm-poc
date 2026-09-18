#!/usr/bin/env python3
"""
Split a GRPO run's wall clock into its two phases, and A/B runs against
each other.

WHY THIS EXISTS
---------------
A GRPO step is two very different things back to back:

    [ vLLM rollout: generate completions ]   ~93% of the step
    [ MaxText: reference logprobs + fwd/bwd ] ~7% of the step

Almost every MaxText tuning flag (attention kernel, remat policy, offloading,
sharding) acts ONLY on the second phase. Comparing total step time across runs
therefore mostly measures how many tokens vLLM happened to generate, which
varies by 25%+ between runs for reasons unrelated to your flag.

Concretely: an A/B in this repo showed total step time dropping 109.3s -> 96.3s
(-12%) after enabling Splash Attention. The real effect on the trainer phase
was +0.14s -- statistically zero. The "win" was the second run generating 27%
fewer tokens. See models/qwen3-0.6b/README.md Part 8.

USAGE
-----
    python3 analyze_grpo_perf.py LABEL=path/to/pod.log [LABEL=... ...]

    # the first run given is treated as the baseline for deltas
    python3 analyze_grpo_perf.py \
        baseline=run-logs/perf-a/pod.log \
        splash=run-logs/perf-b/pod.log

Logs must be captured with timestamps:

    kubectl logs <pod> --timestamps > pod.log

HOW THE SPLIT IS DERIVED
------------------------
  step    = delta between consecutive "Train step N" timestamps
  rollout = sum of that step's vLLM tqdm bars
            ("Processed prompts: 100%|..| 8/8 [MM:SS<")
  trainer = step - rollout

Two parsing traps, both of which silently produce wrong answers:

  1. tqdm RE-RENDERS a finished bar 2-3 times inside one physical log line. A
     naive findall() triple-counts the rollout and yields NEGATIVE trainer
     times.
  2. De-duplicating bars by value is also wrong: two adjacent micro-batches can
     legitimately take the same number of seconds.

Both are avoided by segmenting on the "Rendering prompts" bar that opens each
micro-batch and taking the max within each segment.

Sanity check built in: if the decomposition is right you get exactly
`rollout_micro_batches` bars per step and a trainer residual that is stable
while the rollout swings. The tool reports both so you can verify rather than
trust.
"""

import re
import statistics as st
import sys

TS = re.compile(r"^(\d{4}-\d\d-\d\dT[\d:]+)\.?(\d*)Z")
STEP = re.compile(r"Train step (\d+) training loss:\s*([-\d.eE+]+)")
# group(3)/group(4) = elapsed MM:SS, group(5) = vLLM output throughput
BAR = re.compile(r"Processed prompts:\s*100%.*?\|\s*(\d+)/(\d+)\s*\[(\d+):(\d+)<"
                 r"(?:.*?output:\s*([\d.]+)\s*toks/s)?")
CFG = re.compile(r"Config param (attention|remat_policy|batch_size|num_batches|"
                 r"train_micro_batch_size|rollout_micro_batch_size|"
                 r"max_target_length|learning_rate):\s*(\S+)")
HBM_RESIDENT = re.compile(r"KV cache delete complete.*?hbm_after=\[\(([\d.]+),")
HBM_PEAK = re.compile(r"Deleting kv-cache.*?hbm_before=\[\(([\d.]+),")
OOM = re.compile(r"RESOURCE_EXHAUSTED|Error allocating device buffer|"
                 r"failed to allocate", re.I)

WARMUP = 3   # steps 1..WARMUP are discarded: XLA compile + vLLM cache warm


def tsec(line):
    m = TS.match(line)
    if not m:
        return None
    day = int(m.group(1).split("T")[0].split("-")[2])
    h, mi, s = m.group(1).split("T")[1].split(":")
    frac = float("0." + m.group(2)) if m.group(2) else 0.0
    return day * 86400 + int(h) * 3600 + int(mi) * 60 + int(s) + frac


def bars_in(segment):
    """Rollout bars for one step, delimited by the Rendering bar opening each."""
    out, cur, cur_tp = [], None, None
    for line in segment:
        if "Rendering prompts" in line and cur is not None:
            out.append((cur, cur_tp))
            cur, cur_tp = None, None
        for _, _, mm, ss, tp in BAR.findall(line):
            v = int(mm) * 60 + int(ss)
            if cur is None or v >= cur:
                cur, cur_tp = v, float(tp) if tp else cur_tp
    if cur is not None:
        out.append((cur, cur_tp))
    return out


def parse(path):
    lines = open(path, errors="replace").readlines()
    cfg, resident, peak, oom = {}, [], [], False
    for line in lines:
        for k, v in CFG.findall(line):
            cfg.setdefault(k, v)
        m = HBM_RESIDENT.search(line)
        if m:
            resident.append(float(m.group(1)))
        m = HBM_PEAK.search(line)
        if m:
            peak.append(float(m.group(1)))
        if OOM.search(line):
            oom = True

    idx = [i for i, l in enumerate(lines) if STEP.search(l)]
    steps = []
    for i in range(1, len(idx)):
        n = int(STEP.search(lines[idx[i]]).group(1))
        prev_n = int(STEP.search(lines[idx[i - 1]]).group(1))
        t, pt = tsec(lines[idx[i]]), tsec(lines[idx[i - 1]])
        if t is None or pt is None or n != prev_n + 1:
            continue
        b = bars_in(lines[idx[i - 1] + 1: idx[i] + 1])
        total = t - pt
        roll = sum(x for x, _ in b)
        tps = [y for _, y in b if y]
        # Per-BAR figures, not per-step aggregates. tqdm's "output: N toks/s" is
        # cumulative over the bar, so tokens = rate * elapsed for that same bar.
        # Aggregating per bar (rather than median-rate x mean-elapsed) is what
        # the published numbers in README Part 8 use; keep them consistent.
        steps.append({"step": n, "total": total, "rollout": float(roll),
                      "trainer": total - roll, "nbars": len(b),
                      "bar_tps": tps,
                      "bar_toks": [y * x for x, y in b if y]})

    return {"cfg": cfg, "steps": steps, "resident": resident,
            "peak": peak, "oom": oom, "n_logged": len(idx)}


def steady(d):
    """Steps past warmup whose bar count matches the modal count."""
    cand = [s for s in d["steps"] if s["step"] > WARMUP]
    if not cand:
        return []
    modal = st.mode([s["nbars"] for s in cand])
    return [s for s in cand if s["nbars"] == modal]


def welch(a, b):
    """Returns (diff, stderr, t). No scipy dependency."""
    if len(a) < 2 or len(b) < 2:
        return None
    se = (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5
    d = st.mean(b) - st.mean(a)
    return d, se, (d / se if se else float("inf"))


def main(argv):
    if len(argv) < 2 or any("=" not in a for a in argv[1:]):
        print(__doc__)
        return 1
    runs = []
    for arg in argv[1:]:
        label, path = arg.split("=", 1)
        try:
            runs.append((label, parse(path)))
        except OSError as e:
            print(f"!! {label}: {e}")
    if not runs:
        return 1

    w = max(14, max(len(l) for l, _ in runs) + 2)
    col = lambda vals: "".join(f"{v:>{w}}" for v in vals)

    print("=" * (26 + w * len(runs)))
    print("CONFIG (read back from each run's own dump -- verify the flag applied)")
    print("=" * (26 + w * len(runs)))
    print(f"{'param':<26}" + col([l for l, _ in runs]))
    for k in ["attention", "remat_policy", "batch_size", "train_micro_batch_size",
              "rollout_micro_batch_size", "max_target_length", "num_batches",
              "learning_rate"]:
        print(f"{k:<26}" + col([d["cfg"].get(k, "-") for _, d in runs]))

    print()
    print("=" * (26 + w * len(runs)))
    print(f"PHASE SPLIT (median; first {WARMUP} steps discarded as warmup)")
    print("=" * (26 + w * len(runs)))
    print(f"{'metric':<26}" + col([l for l, _ in runs]))

    def row(name, fn):
        print(f"{name:<26}" + col([fn(d) for _, d in runs]))

    def m(xs, p=1):
        return f"{st.median(xs):.{p}f}" if xs else "n/a"

    row("steps logged", lambda d: d["n_logged"])
    row("steady steps used", lambda d: len(steady(d)))
    row("bars per step", lambda d: steady(d)[0]["nbars"] if steady(d) else "n/a")
    row("median STEP (s)", lambda d: m([s["total"] for s in steady(d)]))
    row("  rollout, vLLM (s)", lambda d: m([s["rollout"] for s in steady(d)]))
    row("  TRAINER, MaxText (s)", lambda d: m([s["trainer"] for s in steady(d)]))
    row("TRAINER share", lambda d: (
        f"{st.median([s['trainer'] for s in steady(d)]) / st.median([s['total'] for s in steady(d)]) * 100:.1f}%"
        if steady(d) else "n/a"))
    row("trainer stdev (s)", lambda d: (
        f"{st.stdev([s['trainer'] for s in steady(d)]):.2f}" if len(steady(d)) > 1 else "n/a"))
    row("vLLM throughput tok/s", lambda d: m(
        [x for s in steady(d) for x in s["bar_tps"]], 0))
    row("tokens per bar", lambda d: m(
        [x for s in steady(d) for x in s["bar_toks"]], 0))

    row("HBM resident (GB)", lambda d: m(d["resident"]))
    row("HBM peak w/ kv (GB)", lambda d: m(d["peak"]))
    row("OOM seen", lambda d: "YES" if d["oom"] else "no")

    if len(runs) > 1:
        base_label, base = runs[0]
        print()
        print("=" * 78)
        print(f"DELTA vs '{base_label}'   (Welch t; |t| < 2 => indistinguishable from noise)")
        print("=" * 78)
        bt = [s["trainer"] for s in steady(base)]
        bs = [s["total"] for s in steady(base)]
        for label, d in runs[1:]:
            tr = [s["trainer"] for s in steady(d)]
            tot = [s["total"] for s in steady(d)]
            print(f"\n{label}")
            r = welch(bt, tr)
            if r:
                diff, se, t = r
                verdict = "INDISTINGUISHABLE FROM ZERO" if abs(t) < 2 else "significant"
                print(f"   TRAINER  {st.median(bt):6.2f}s -> {st.median(tr):6.2f}s   "
                      f"{diff:+.2f}s  95%CI[{diff-1.96*se:+.2f},{diff+1.96*se:+.2f}]  "
                      f"t={t:+.2f}  -> {verdict}")
            if bs and tot:
                print(f"   total    {st.median(bs):6.2f}s -> {st.median(tot):6.2f}s   "
                      f"({(st.median(tot)-st.median(bs))/st.median(bs)*100:+.1f}%)"
                      "   <- DO NOT QUOTE THIS; see note below")
        print("\nNote: total step time is dominated by how many tokens vLLM happened")
        print("to generate, which differs between runs for reasons unrelated to any")
        print("MaxText flag. Compare the TRAINER row. Cross-check the 'tokens per")
        print("bar' row above: if it moved, your total-time delta is generation")
        print("length, not your optimization.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
