#!/usr/bin/env python3
"""Zero-dependency TensorBoard .tfevents inspector & CSV exporter for MaxText GRPO runs.
Supports both local directories and Google Cloud Storage (gs://) run folders!

Usage:
  # Inspect local folder:
  python3 inspect_run.py training-grpo/run-logs/workshop-qwen06-gsm8k-lr2e5-20260916-035550

  # Inspect directly from GCS bucket:
  python3 inspect_run.py gs://$GS_BUCKET/maxtext-qwen3-0p6b/$RUN_NAME
"""
import csv
import glob
import os
import struct
import subprocess
import sys


def read_tfevents(path: str) -> dict[str, list[tuple[int, float]]]:
    """Parses a TensorBoard .tfevents binary file using pure Python standard library."""
    records: dict[str, list[tuple[int, float]]] = {}
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if len(header) < 12:
                break
            length, _ = struct.unpack("<QI", header)
            data = f.read(length)
            f.read(4)  # CRC footer
            step = 0
            idx = 0
            summary_bytes = None
            while idx < len(data):
                tag = data[idx]
                fn = tag >> 3
                wt = tag & 7
                idx += 1
                if wt == 0:
                    val = 0
                    shift = 0
                    while True:
                        b = data[idx]
                        idx += 1
                        val |= (b & 0x7F) << shift
                        if not (b & 0x80):
                            break
                        shift += 7
                    if fn == 2:
                        step = val
                elif wt == 1:
                    idx += 8
                elif wt == 2:
                    l = 0
                    shift = 0
                    while True:
                        b = data[idx]
                        idx += 1
                        l |= (b & 0x7F) << shift
                        if not (b & 0x80):
                            break
                        shift += 7
                    sub = data[idx : idx + l]
                    idx += l
                    if fn == 5:
                        summary_bytes = sub
                elif wt == 5:
                    idx += 4

            if summary_bytes:
                sidx = 0
                while sidx < len(summary_bytes):
                    stag = summary_bytes[sidx]
                    sidx += 1
                    if (stag >> 3) == 1 and (stag & 7) == 2:
                        sl = 0
                        shift = 0
                        while True:
                            b = summary_bytes[sidx]
                            sidx += 1
                            sl |= (b & 0x7F) << shift
                            if not (b & 0x80):
                                break
                            shift += 7
                        val_bytes = summary_bytes[sidx : sidx + sl]
                        sidx += sl
                        vidx = 0
                        vtag_str = None
                        vval = None
                        while vidx < len(val_bytes):
                            vt = val_bytes[vidx]
                            vidx += 1
                            vfn = vt >> 3
                            vwt = vt & 7
                            if vwt == 0:
                                while val_bytes[vidx] & 0x80:
                                    vidx += 1
                                vidx += 1
                            elif vwt == 1:
                                vidx += 8
                            elif vwt == 2:
                                vl = 0
                                shift = 0
                                while True:
                                    b = val_bytes[vidx]
                                    vidx += 1
                                    vl |= (b & 0x7F) << shift
                                    if not (b & 0x80):
                                        break
                                    shift += 7
                                sub = val_bytes[vidx : vidx + vl]
                                vidx += vl
                                if vfn == 1:
                                    vtag_str = sub.decode("utf-8", errors="ignore")
                                elif vfn == 8:
                                    tidx = 0
                                    while tidx < len(sub):
                                        tt = sub[tidx]
                                        tidx += 1
                                        tfn = tt >> 3
                                        twt = tt & 7
                                        if twt == 0:
                                            while sub[tidx] & 0x80:
                                                tidx += 1
                                            tidx += 1
                                        elif twt == 5:
                                            if tfn == 5:
                                                vval = struct.unpack("<f", sub[tidx : tidx + 4])[0]
                                            tidx += 4
                                        elif twt == 2:
                                            tl = 0
                                            shift = 0
                                            while True:
                                                b = sub[tidx]
                                                tidx += 1
                                                tl |= (b & 0x7F) << shift
                                                if not (b & 0x80):
                                                    break
                                                shift += 7
                                            tsub = sub[tidx : tidx + tl]
                                            tidx += tl
                                            if tfn == 4 and len(tsub) == 4:
                                                vval = struct.unpack("<f", tsub)[0]
                                            elif tfn == 5 and len(tsub) >= 4:
                                                vval = struct.unpack("<f", tsub[:4])[0]
                            elif vwt == 5:
                                if vfn == 2:
                                    vval = struct.unpack("<f", val_bytes[vidx : vidx + 4])[0]
                                vidx += 4
                        if vtag_str and vval is not None:
                            records.setdefault(vtag_str, []).append((step, vval))
    return records


def ascii_sparkline(values: list[float], width: int = 40) -> str:
    """Generates a simple ASCII sparkline bar chart."""
    if not values:
        return ""
    blocks = " ▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    if mx == mn:
        return blocks[4] * min(len(values), width)
    step = max(1, len(values) / width)
    sampled = [values[int(i * step)] for i in range(min(len(values), width))]
    return "".join(blocks[min(7, int((v - mn) / (mx - mn) * 7))] for v in sampled)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "training-grpo/run-logs/workshop-qwen06-gsm8k-lr2e5-20260916-035550"
    gcs_source = None

    if target.startswith("gs://"):
        gcs_source = target.rstrip("/")
        run_name = os.path.basename(gcs_source)
        run_dir = os.path.join("training-grpo", "run-logs", run_name)
        os.makedirs(run_dir, exist_ok=True)
        local_dest = os.path.join(run_dir, "events.tfevents")
        print(f"☁️  Fetching latest TensorBoard events from GCS: {gcs_source}/tensorboard/ ...")
        cmd = f"gcloud storage cp '{gcs_source}/tensorboard/*tfevents*' '{local_dest}' 2>/dev/null || gcloud storage cp '{gcs_source}/*tfevents*' '{local_dest}'"
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0 or not os.path.exists(local_dest):
            print(f"❌ Could not find .tfevents file under {gcs_source}")
            sys.exit(1)
        tf_path = local_dest
    elif os.path.isdir(target):
        candidates = glob.glob(os.path.join(target, "**", "*tfevents*"), recursive=True)
        if not candidates:
            print(f"No .tfevents file found in {target}")
            sys.exit(1)
        tf_path = candidates[0]
        run_dir = target
    else:
        tf_path = target
        run_dir = os.path.dirname(target)

    # Ensure standard TensorBoard naming symlink exists for `tensorboard --logdir`
    std_name = os.path.join(run_dir, "events.out.tfevents.grpo")
    if not os.path.exists(std_name) and os.path.basename(tf_path) == "events.tfevents":
        try:
            os.symlink("events.tfevents", std_name)
        except OSError:
            pass

    records = read_tfevents(tf_path)
    key_metrics = [
        "rewards/train/mean",
        "rewards/train/check_numbers",
        "rewards/train/match_format_exactly",
        "completions/train/mean_length",
        "actor/train/kl",
        "actor/train/advantage/nonzero_frac",
    ]

    print("=" * 95)
    print(f"📊 GRPO RUN SUMMARY: {os.path.basename(os.path.abspath(run_dir))}")
    if gcs_source:
        print(f"☁️  GCS Bucket Path : {gcs_source}")
    print(f"📁 Event File      : {tf_path} ({os.path.getsize(tf_path):,} bytes)")
    steps_logged = len(records.get("rewards/train/mean", []))
    print(f"⏱️  Steps Logged    : {steps_logged} full steps (0..{steps_logged - 1})")
    print("=" * 95)
    print(f"{'Metric Tag':<36} | {'First 5 Avg':>11} | {'Last 5 Avg':>11} | {'Change':>9} | {'Trend Sparkline'}")
    print("-" * 95)

    for tag in key_metrics:
        pts = records.get(tag, [])
        if not pts:
            continue
        vals = [v for _, v in pts]
        w = min(5, len(vals))
        first_avg = sum(vals[:w]) / w
        last_avg = sum(vals[-w:]) / w
        pct = ((last_avg - first_avg) / abs(first_avg) * 100.0) if first_avg != 0 else 0.0
        spark = ascii_sparkline(vals, width=20)
        print(f"{tag:<36} | {first_avg:11.4f} | {last_avg:11.4f} | {pct:+8.1f}% | {spark}")
    print("=" * 95)

    # Export to CSV for easy spreadsheet / pandas analysis
    csv_path = os.path.join(run_dir, "metrics.csv")
    all_steps = sorted({s for tag in key_metrics for s, _ in records.get(tag, [])})
    step_map = {tag: dict(records.get(tag, [])) for tag in key_metrics}

    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["step"] + key_metrics)
        for s in all_steps:
            row = [s] + [round(step_map[tag].get(s, float("nan")), 6) for tag in key_metrics]
            writer.writerow(row)

    print(f"✅ Exported step-by-step CSV table to: {csv_path}")
    if gcs_source:
        subprocess.run(
            f"gcloud storage cp '{csv_path}' '{gcs_source}/metrics.csv' >/dev/null 2>&1",
            shell=True,
        )
        print(f"☁️  Uploaded metrics.csv back to GCS: {gcs_source}/metrics.csv")


if __name__ == "__main__":
    main()
