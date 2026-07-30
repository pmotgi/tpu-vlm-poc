# TPU & GPU Model Inference using vLLM

This directory contains Kubernetes manifests, benchmarking suite jobs, and comprehensive evaluation reports for serving large language models—specifically **Gemma 4 31B IT** and LoRA-adapted models—on Google Cloud **TPU v6e**, **TPU v7x**, and NVIDIA GPU clusters using [vLLM](https://github.com/vllm-project/vllm).

---

## 1. Serving Deployment Manifests

| Manifest File | Target Hardware | Topology / Accelerator | Parallelism Strategy | Target Service Endpoint |
| :--- | :--- | :--- | :--- | :--- |
| **[`gemma-4-31b-it-vllm-v6e.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e.yaml)** | **TPU v6e** | `2x4` slice (8 TPU chips) | `TP=4`, `DP=2` | `http://gemma4-vllm-tpu-v6e-service:8000` |
| **[`gemma-4-31b-it-vllm-v6e-4tpu.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e-4tpu.yaml)** | **TPU v6e** | `2x4` host (4 TPU chips utilized) | `TP=4`, `DP=1` | `http://gemma4-vllm-tpu-v6e-4tpu-service:8000` |
| **[`gemma4-31b-vllm.yaml`](file:///Users/pmotgi/exploration/cerence/inference/gemma4-31b-vllm.yaml)** | **TPU v7x** | `1x1x1` pod (1 chip per pod) | `TP=2`, `DP=1` | `http://gemma4-31b-vllm-service:8000` |
| **[`vllm-deployment.yaml`](file:///Users/pmotgi/exploration/cerence/inference/vllm-deployment.yaml)** | **TPU / GPU** | Multi-purpose vLLM server | `TP=4` | `http://vllm-service:8000` |

---

## 2. Automated Benchmark Suite Jobs

Each benchmark suite runs a 9-config sequence matrix (Input / Output token lengths from `128/128` to `8192/1024` and `2048/2048`) with 150 prompts at request rate `inf`:

- **[`benchmark-suite-v6e.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v6e.yaml)**: Benchmarks the 8-chip TPU v6e deployment (`gemma4-vllm-tpu-v6e-service`).
- **[`benchmark-suite-v6e-4tpu.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v6e-4tpu.yaml)**: Benchmarks the 4-chip TPU v6e deployment (`gemma4-vllm-tpu-v6e-4tpu-service`).
- **[`benchmark-suite-v7x.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v7x.yaml)**: Benchmarks the TPU v7x deployment (`gemma4-31b-vllm-service`).
- **[`benchmark-job.yaml`](file:///Users/pmotgi/exploration/cerence/inference/benchmark-job.yaml)**: Single prompt configuration test run.

---

## 3. Benchmarking Reports & Analysis

- **[`BENCHMARK_REPORT_TABLES.md`](file:///Users/pmotgi/exploration/cerence/inference/BENCHMARK_REPORT_TABLES.md)**:
  - Complete **3-way comparative tabular report** comparing **TPU v6e-8 (`TP=4, DP=2`)**, **TPU v6e-4 (`TP=4, DP=1`)**, and **TPU v7x (`TP=2, DP=1`)**.
  - Includes detailed hardware environment stamps, runtime parameter stamps, throughput (tok/s, req/s), and latency (TTFT, TPOT, ITL) breakdowns.
  - Also includes NVIDIA G4 GPU Cluster (`GLM-5.2 NVFP4`) reference baseline data.
- **[`BENCHMARK_REPORT_GRAPHS.md`](file:///Users/pmotgi/exploration/cerence/inference/BENCHMARK_REPORT_GRAPHS.md)**:
  - Visual 3-way comparison charts for Output Token Throughput, Total Token Throughput, TTFT Latency, and TPOT Latency.

---

## 4. How to Deploy and Benchmark

### Step 1: Deploy a vLLM Serving Endpoint
Choose your target hardware deployment:
```bash
# For TPU v6e (8 Chips - TP=4, DP=2)
kubectl apply -f inference/gemma-4-31b-it-vllm-v6e.yaml

# For TPU v6e (4 Chips - TP=4, DP=1)
kubectl apply -f inference/gemma-4-31b-it-vllm-v6e-4tpu.yaml

# For TPU v7x (TP=2, DP=1)
kubectl apply -f inference/gemma4-31b-vllm.yaml
```

Check server startup and readiness:
```bash
kubectl get pods -n default
kubectl logs -f -l app=gemma4-vllm-tpu-v6e-4tpu
```

### Step 2: Run the Benchmark Suite
Deploy the corresponding Kubernetes Job:
```bash
# Benchmark 4-TPU v6e deployment
kubectl apply -f inference/benchmark-suite-v6e-4tpu.yaml
```

Monitor benchmark progress in real-time:
```bash
kubectl logs -f job/benchmark-suite-v6e-4tpu
```
