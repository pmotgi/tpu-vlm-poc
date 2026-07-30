# Gemma 4 31B IT Inference Benchmark 3-Way Tabular Report

This report presents detailed benchmarking comparison results for serving **`google/gemma-4-31b-it`** using **vLLM** across three Google Cloud TPU hardware/parallelism deployments:
1. **TPU v6e (8 Chips - `TP=4, DP=2`)**
2. **TPU v6e (4 Chips - `TP=4, DP=1`)**
3. **TPU v7x (1 Pod - `TP=2, DP=1`)**

---

## 1. Hardware & Runtime Configuration Stamps

### A. TPU v6e-8 Deployment Stamp (8 Chips)
- **Accelerator Target**: Google Cloud TPU v6e (`tpu-v6e-slice`), Topology: `2x4` (8 TPU chips)
- **Node Pool**: `v6e-8t-dws-flex`
- **Parallelism Strategy**: Tensor Parallelism = 4 (`TP=4`), Data Parallelism = 2 (`DP=2`)
- **KV Cache / Quantization**: `fp8` KV cache, QWIX FP8 weights/activations (`float8_e4m3fn`)
- **Manifest Reference**: [gemma-4-31b-it-vllm-v6e.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e.yaml)
- **Benchmark Suite Manifest**: [benchmark-suite-v6e.yaml](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v6e.yaml)

### B. TPU v6e-4 Deployment Stamp (4 Chips)
- **Accelerator Target**: Google Cloud TPU v6e (`tpu-v6e-slice`), Topology: `2x4` host (4 chips utilized)
- **Node Pool**: `v6e-8t-dws-flex`
- **Parallelism Strategy**: Tensor Parallelism = 4 (`TP=4`), Data Parallelism = 1 (`DP=1`)
- **KV Cache / Quantization**: `fp8` KV cache, QWIX FP8 weights/activations (`float8_e4m3fn`)
- **Manifest Reference**: [gemma-4-31b-it-vllm-v6e-4tpu.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e-4tpu.yaml)
- **Benchmark Suite Manifest**: [benchmark-suite-v6e-4tpu.yaml](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v6e-4tpu.yaml)

### C. TPU v7x Deployment Stamp (1 Pod / Chip Target)
- **Accelerator Target**: Google Cloud TPU v7x (`tpu7x`), Topology: `1x1x1`
- **Node Pool**: `pmotgi-tpu7x-spot-1t`
- **Parallelism Strategy**: Tensor Parallelism = 2 (`TP=2`), Data Parallelism = 1 (`DP=1`)
- **KV Cache / Quantization**: `fp8` KV cache, QWIX FP8 weights/activations (`float8_e4m3fn`)
- **Manifest Reference**: [gemma4-31b-vllm.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma4-31b-vllm.yaml)
- **Benchmark Suite Manifest**: [benchmark-suite-v7x.yaml](file:///Users/pmotgi/exploration/cerence/inference/benchmark-suite-v7x.yaml)

---

## 2. 3-Way Throughput Comparison Tables

### Table 1: Output Token Throughput (tokens/sec) — *Higher is better*

| Input / Output Tokens (ISL / OSL) | v6e-8 (`TP=4, DP=2`, 8 chips) | v6e-4 (`TP=4, DP=1`, 4 chips) | v7x (`TP=2, DP=1`) |
| :---: | :---: | :---: | :---: |
| **128 / 128** | 792.40 | 640.27 | **1,737.35** |
| **512 / 512** | **2,788.31** | 1,365.50 | 2,567.96 |
| **1024 / 1024** | **2,762.94** | 1,438.12 | 2,690.03 |
| **2048 / 128** | **787.23** | 447.78 | 769.70 |
| **4096 / 128** | **486.24** | 266.80 | 464.88 |
| **6000 / 128** | **410.35** | 205.41 | 405.98 |
| **8192 / 1024** | 899.53 | 442.05 | **927.26** |
| **512 / 2048** | 2,999.25 | 1,610.98 | **3,035.83** |
| **2048 / 2048** | 2,052.58 | 1,101.70 | **2,466.84** |

---

### Table 2: Total Token Throughput (tokens/sec) — *Higher is better*

| Input / Output Tokens (ISL / OSL) | v6e-8 (`TP=4, DP=2`, 8 chips) | v6e-4 (`TP=4, DP=1`, 4 chips) | v7x (`TP=2, DP=1`) |
| :---: | :---: | :---: | :---: |
| **128 / 128** | 3,288.56 | 2,657.20 | **7,210.17** |
| **512 / 512** | **7,076.58** | 3,465.56 | 6,517.32 |
| **1024 / 1024** | **6,269.74** | 3,263.43 | 6,104.29 |
| **2048 / 128** | **15,077.76** | 8,576.36 | 14,742.12 |
| **4096 / 128** | **17,092.89** | 9,378.88 | 16,341.91 |
| **6000 / 128** | **20,529.02** | 10,276.18 | 20,310.39 |
| **8192 / 1024** | 8,337.89 | 4,097.42 | **8,594.86** |
| **512 / 2048** | 4,152.41 | 2,230.39 | **4,203.07** |
| **2048 / 2048** | 4,381.36 | 2,351.64 | **5,265.64** |

---

## 3. 3-Way Latency Comparison Tables

### Table 3: Mean Time to First Token (TTFT in seconds) — *Lower is better*

| Input / Output Tokens (ISL / OSL) | v6e-8 (`TP=4, DP=2`, 8 chips) | v6e-4 (`TP=4, DP=1`, 4 chips) | v7x (`TP=2, DP=1`) |
| :---: | :---: | :---: | :---: |
| **128 / 128** | 18.78 s | 20.97 s | **5.42 s** |
| **512 / 512** | **7.30 s** | 11.34 s | 8.07 s |
| **1024 / 1024** | 8.30 s | 23.95 s | **8.26 s** |
| **2048 / 128** | 10.23 s | 19.40 s | **9.93 s** |
| **4096 / 128** | **19.19 s** | 36.20 s | 20.04 s |
| **6000 / 128** | **22.34 s** | 44.88 s | 22.84 s |
| **8192 / 1024** | 70.24 s | 158.33 s | **64.21 s** |
| **512 / 2048** | **7.24 s** | 12.18 s | 7.30 s |
| **2048 / 2048** | **15.18 s** | 83.79 s | 18.04 s |

---

### Table 4: Mean Time per Output Token (TPOT in ms) — *Lower is better*

| Input / Output Tokens (ISL / OSL) | v6e-8 (`TP=4, DP=2`, 8 chips) | v6e-4 (`TP=4, DP=1`, 4 chips) | v7x (`TP=2, DP=1`) |
| :---: | :---: | :---: | :---: |
| **128 / 128** | **40.10 ms** | 69.18 ms | 42.91 ms |
| **512 / 512** | **38.46 ms** | 73.81 ms | 42.27 ms |
| **1024 / 1024** | 44.82 ms | 55.13 ms | **42.56 ms** |
| **2048 / 128** | **86.76 ms** | 105.13 ms | 97.59 ms |
| **4096 / 128** | 104.13 ms | 109.07 ms | **103.69 ms** |
| **6000 / 128** | **95.10 ms** | 104.29 ms | 97.95 ms |
| **8192 / 1024** | 38.13 ms | 39.97 ms | **32.46 ms** |
| **512 / 2048** | 40.16 ms | 65.71 ms | **38.58 ms** |
| **2048 / 2048** | 44.45 ms | 45.09 ms | **36.25 ms** |
