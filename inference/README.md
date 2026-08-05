# TPU & GPU Model Inference using vLLM

This directory contains Kubernetes manifests, benchmarking suite jobs, and comprehensive evaluation reports for serving large language models—specifically **Gemma 4 31B IT** and LoRA-adapted models—on Google Cloud **TPU v6e**, **TPU v5p**, **TPU v7x**, and NVIDIA **RTX 6000 Pro** / **H100** GPU clusters using [vLLM](https://github.com/vllm-project/vllm).

---

## 1. Serving Deployment Manifests

| Manifest File | Target Hardware | Topology / Accelerator | Parallelism Strategy | Target Service Endpoint |
| :--- | :--- | :--- | :--- | :--- |
| **[`gemma-4-31b-it-vllm-v6e.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e.yaml)** | **TPU v6e** | `2x4` slice (8 TPU chips) | `TP=4`, `DP=2` | `http://gemma4-vllm-tpu-v6e-service:8000` |
| **[`gemma-4-31b-it-vllm-v6e-4tpu.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e-4tpu.yaml)** | **TPU v6e** | `2x4` host (4 TPU chips utilized) | `TP=4`, `DP=1` | `http://gemma4-vllm-tpu-v6e-4tpu-service:8000` |
| **[`gemma-4-31b-it-vllm-v5p.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v5p.yaml)** | **TPU v5p** | `2x2x1` slice (4 TPU chips) | `TP=4`, `DP=1` | `http://gemma4-vllm-tpu-v5p-service:8000` |
| **[`gemma-4-31b-it-vllm-v5p-tp2.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v5p-tp2.yaml)** | **TPU v5p** | `2x2x1` host (2 TPU chips utilized) | `TP=2`, `DP=1` | `http://gemma4-vllm-tpu-v5p-tp2-service:8000` |
| **[`gemma4-31b-vllm.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma4-31b-vllm.yaml)** | **TPU v7x** | `1x1x1` pod (1 chip per pod) | `TP=2`, `DP=1` | `http://gemma4-31b-vllm-service:8000` |
| **[`gemma-4-31b-it-vllm-g4.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-g4.yaml)** | **NVIDIA RTX 6000 Pro** | 1 GPU chip (`g4-standard-48`) | `TP=1`, `DP=1` | `http://gemma4-vllm-g4-service:8000` |
| **[`gemma-4-31b-it-vllm-h100.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-h100.yaml)** | **NVIDIA H100** | 1 GPU chip (`a3-mega-8g`) | `TP=1`, `DP=1` | `http://gemma4-vllm-h100-service:8000` |

---

## 2. Automated Benchmark Suite Jobs

Each benchmark suite executes a 9-config sequence matrix (Input / Output token lengths from `128/128` to `8192/1024` and `2048/2048`) with 150 prompts at request rate `inf`:

- **[`benchmark-suite-v6e.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v6e.yaml)**: Benchmarks the 8-chip TPU v6e deployment (`TP=4, DP=2`).
- **[`benchmark-suite-v6e-4tpu.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v6e-4tpu.yaml)**: Benchmarks the 4-chip TPU v6e deployment (`TP=4, DP=1`).
- **[`benchmark-suite-v5p.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v5p.yaml)**: Benchmarks the 4-chip TPU v5p deployment (`TP=4, DP=1`).
- **[`benchmark-suite-v5p-tp2.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v5p-tp2.yaml)**: Benchmarks the 2-chip TPU v5p deployment (`TP=2, DP=1`).
- **[`benchmark-suite-v7x.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v7x.yaml)**: Benchmarks the TPU v7x pod deployment (`TP=2, DP=1`).
- **[`benchmark-suite-g4.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-g4.yaml)**: Benchmarks the NVIDIA RTX 6000 Pro deployment (`TP=1`, `g4-standard-48`).
- **[`benchmark-suite-h100.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-h100.yaml)**: Benchmarks the NVIDIA H100 deployment (`TP=1`, `a3-mega-8g`).

---

## 3. Benchmarking Reports & Analysis

- **[`BENCHMARK_REPORT_TABLES.md`](file:///Users/pmotgi/exploration/cerence/inference/BENCHMARK_REPORT_TABLES.md)**:
  - Complete **7-way comparative tabular report** evaluating **TPU v6e-8 (`TP=4, DP=2`)**, **TPU v6e-4 (`TP=4, DP=1`)**, **TPU v5p-4 (`TP=4, DP=1`)**, **TPU v5p-2 (`TP=2, DP=1`)**, **TPU v7x (`TP=2, DP=1`)**, **NVIDIA RTX 6000 Pro (`TP=1`, 48 GB)**, and **NVIDIA H100 (`TP=1`, 80 GB)**.
  - Includes hardware environment stamps, runtime parameter configurations, output token throughput, total token throughput, TTFT prefill latency, and TPOT decode latency breakdowns.
- **[`BENCHMARK_REPORT_GRAPHS.md`](file:///Users/pmotgi/exploration/cerence/inference/BENCHMARK_REPORT_GRAPHS.md)**:
  - Visual 7-way comparison charts for Output Token Throughput, Total Token Throughput, TTFT Latency, and TPOT Latency.
  - Includes a note explaining that standard out-of-the-box configurations were used to establish an equitable baseline, which can be further optimized for specific production workloads.

---

## 4. How to Deploy and Benchmark

### Step 1: Deploy a vLLM Serving Endpoint
Choose your target hardware deployment:
```bash
# For TPU v6e (8 Chips - TP=4, DP=2)
kubectl apply -f inference/gemma-4-31b-it-vllm-v6e.yaml

# For TPU v7x (1 Pod - TP=2, DP=1)
kubectl apply -f inference/gemma4-31b-vllm.yaml

# For NVIDIA RTX 6000 Pro (1 Chip - TP=1 on g4-standard-48)
kubectl apply -f inference/gemma-4-31b-it-vllm-g4.yaml

# For NVIDIA H100 (1 Chip - TP=1 on a3-mega-8g)
kubectl apply -f inference/gemma-4-31b-it-vllm-h100.yaml
```

Check pod startup and readiness:
```bash
kubectl get pods -n default
kubectl logs -f -l app=gemma4-vllm-h100
```

### Step 2: Run the Benchmark Suite
Deploy the corresponding Kubernetes Job:
```bash
# Benchmark NVIDIA H100 deployment
kubectl apply -f inference/benchmark-suite-h100.yaml
```

Monitor benchmark progress in real-time:
```bash
kubectl logs -f job/benchmark-suite-h100
```
