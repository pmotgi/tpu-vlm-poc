# Gemma 4 31B IT Inference 3-Way Benchmark Visual Graph Report

This report contains visual performance comparison charts for serving **`google/gemma-4-31b-it`** using **vLLM** across three Google Cloud TPU hardware/parallelism deployments:
1. **TPU v6e-8 (`TP=4, DP=2`, 8 chips)** — Blue Bar
2. **TPU v6e-4 (`TP=4, DP=1`, 4 chips)** — Amber Bar
3. **TPU v7x (`TP=2, DP=1`, 1 pod)** — Emerald Bar

---

## 1. Output Token Throughput (tokens/sec)

*Higher is better. Measures the generation speed of new output tokens across concurrency.*

![Output Token Throughput Chart](/Users/pmotgi/.gemini/jetski/brain/6192c0ae-9da5-41b6-aeb2-18a2dba6f0c3/output_throughput_chart.svg.png)

---

## 2. Total Token Throughput (tokens/sec)

*Higher is better. Combines prefill input token processing and generation output tokens.*

![Total Token Throughput Chart](/Users/pmotgi/.gemini/jetski/brain/6192c0ae-9da5-41b6-aeb2-18a2dba6f0c3/total_throughput_chart.svg.png)

---

## 3. Time to First Token (TTFT in seconds)

*Lower is better. Measures response latency before generating the first token.*

![Time to First Token Latency Chart](/Users/pmotgi/.gemini/jetski/brain/6192c0ae-9da5-41b6-aeb2-18a2dba6f0c3/ttft_latency_chart.svg.png)

---

## 4. Time per Output Token (TPOT in ms)

*Lower is better. Measures average inter-token generation latency.*

![Time per Output Token Latency Chart](/Users/pmotgi/.gemini/jetski/brain/6192c0ae-9da5-41b6-aeb2-18a2dba6f0c3/tpot_latency_chart.svg.png)

---

## Source Hardware & Manifest References
- TPU v6e-8 Deployment Manifest (8 Chips): [gemma-4-31b-it-vllm-v6e.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e.yaml)
- TPU v6e-4 Deployment Manifest (4 Chips): [gemma-4-31b-it-vllm-v6e-4tpu.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma-4-31b-it-vllm-v6e-4tpu.yaml)
- TPU v7x Deployment Manifest (1 Pod): [gemma4-31b-vllm.yaml](file:///Users/pmotgi/exploration/cerence/inference/gemma4-31b-vllm.yaml)
- Full 3-Way Tabular Report: [BENCHMARK_REPORT_TABLES.md](file:///Users/pmotgi/exploration/cerence/inference/BENCHMARK_REPORT_TABLES.md)
