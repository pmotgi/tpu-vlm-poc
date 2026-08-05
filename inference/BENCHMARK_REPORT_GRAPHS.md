# Gemma 4 31B IT Inference 7-Way Benchmark Visual Graph Report

This report contains visual performance comparison charts for serving **`google/gemma-4-31b-it`** using **vLLM** across seven Google Cloud hardware/parallelism deployments (5 TPU deployments + 2 NVIDIA GPU deployments):
1. **TPU v6e-8 (`TP=4, DP=2`, 8 chips)** — Blue Bar (`#3b82f6`)
2. **TPU v6e-4 (`TP=4, DP=1`, 4 chips)** — Light Blue Bar (`#60a5fa`)
3. **TPU v5p-4 (`TP=4, DP=1`, 4 chips)** — Purple Bar (`#a855f7`)
4. **TPU v5p-2 (`TP=2, DP=1`, 2 chips)** — Light Purple Bar (`#c084fc`)
5. **TPU v7x (`TP=2, DP=1`, 1 pod)** — Amber Bar (`#f59e0b`)
6. **NVIDIA RTX 6000 Pro (`TP=1`, 1 chip)** — Emerald Green Bar (`#10b981`)
7. **NVIDIA H100 (`TP=1`, 1 chip)** — Rose Red Bar (`#e11d48`)

> [!NOTE]
> **Standard Configuration Baseline**: These inference benchmark measurements were conducted using out-of-the-box standard serving configurations across all hardware deployments to establish a uniform, equitable comparison. Throughput and latency metrics can be further optimized for intended production workloads through tailored tuning of batching parameters (`--max-num-batched-tokens`, `--max-num-seqs`), KV cache utilization, prefix caching, speculative decoding, and custom compilation flags.

---

## 1. Output Token Throughput (tokens/sec)

*Higher is better. Measures the generation speed of new output tokens across concurrency.*

![Output Token Throughput Chart](/Users/pmotgi/.gemini/jetski/brain/055e94dc-dc45-443a-bcf8-c2d1f821cfb5/output_throughput_chart.svg.png)

---

## 2. Total Token Throughput (tokens/sec)

*Higher is better. Combines prefill input token processing and generation output tokens.*

![Total Token Throughput Chart](/Users/pmotgi/.gemini/jetski/brain/055e94dc-dc45-443a-bcf8-c2d1f821cfb5/total_throughput_chart.svg.png)

---

## 3. Time to First Token (TTFT in seconds)

*Lower is better. Measures response latency before generating the first token.*

![Time to First Token Latency Chart](/Users/pmotgi/.gemini/jetski/brain/055e94dc-dc45-443a-bcf8-c2d1f821cfb5/ttft_latency_chart.svg.png)

---

## 4. Time per Output Token (TPOT in ms)

*Lower is better. Measures average inter-token generation latency.*

![Time per Output Token Latency Chart](/Users/pmotgi/.gemini/jetski/brain/055e94dc-dc45-443a-bcf8-c2d1f821cfb5/tpot_latency_chart.svg.png)

---

## Source Hardware & Manifest References
- TPU v6e-8 Deployment Manifest (8 Chips): [gemma-4-31b-it-vllm-v6e.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e.yaml)
- TPU v6e-4 Deployment Manifest (4 Chips): [gemma-4-31b-it-vllm-v6e-4tpu.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e-4tpu.yaml)
- TPU v5p-4 Deployment Manifest (4 Chips): [gemma-4-31b-it-vllm-v5p.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v5p.yaml)
- TPU v5p-2 Deployment Manifest (2 Chips): [gemma-4-31b-it-vllm-v5p-tp2.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v5p-tp2.yaml)
- TPU v7x Deployment Manifest (1 Pod): [gemma4-31b-vllm.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma4-31b-vllm.yaml)
