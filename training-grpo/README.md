# MaxText Post-Training RL / GRPO on TPU v7x

This directory contains the production, tested, and verified Kubernetes JobSet manifest for running **Group Relative Policy Optimization (GRPO)** with **vLLM Rollout Sampling** on Google Cloud TPU v7x (`tpu-v7x-spot-2x2x1`).

---

## Verified Manifest

### **LLaMA 3.1-8B GRPO RL (`2x2x1` Topology)**
* **Manifest File**: [`llama3.1-8b-grpo-spot-2x2x1-training.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/llama3.1-8b-grpo-spot-2x2x1-training.yaml)
* **Container Image**: `us-east5-docker.pkg.dev/northam-ce-mlai-tpu/pmotgi-vlm-poc-repo/my-maxtext-runner:grpo-v2`
* **Target Hardware**: `tpu-v7x-spot-2x2x1` (1 VM with 4 TPU chips / 8 Tensor Core devices)
  * **4 Trainer Devices (`TPU_0` to `TPU_3`)**: MaxText Actor Policy gradient optimizer (AdamW) + reference model log-probability computation.
  * **4 Sampler Devices (`TPU_4` to `TPU_7`)**: vLLM online inference with Pallas Ragged Paged Attention (RPA).
* **Dataset**: `openai/gsm8k`
* **Status**: Tested & Verified (`Succeeded`)

### How to Run:
```bash
# From workspace root:
kubectl apply -f training-grpo/llama3.1-8b-grpo-spot-2x2x1-training.yaml

# Or from inside the training-grpo/ directory:
kubectl apply -f llama3.1-8b-grpo-spot-2x2x1-training.yaml
```

---

---

## Container Build Quickstart

You can build the container using either **Google Cloud Build** or local **Docker / Podman**:

### Option 1: Using Google Cloud Build
```bash
cd training-grpo/docker
gcloud builds submit . --tag="REGION-docker.pkg.dev/PROJECT/REPO/maxtext-grpo-runner:v2" --machine-type="e2-highcpu-8"
```

### Option 2: Using Local Docker (Without Cloud Build)
```bash
cd training-grpo/docker

# 1. Authenticate Docker with Artifact Registry
gcloud auth configure-docker REGION-docker.pkg.dev

# 2. Build and push
docker build -t REGION-docker.pkg.dev/PROJECT/REPO/maxtext-grpo-runner:v2 .
docker push REGION-docker.pkg.dev/PROJECT/REPO/maxtext-grpo-runner:v2
```

---

## Detailed Documentation
* **Architecture & Topology**: [`training-grpo/docker/README.md`](file:///Users/pmotgi/exploration/cerence/training-grpo/docker/README.md)
* **MFU Performance Calculator**: [`training-grpo/docker/calculate_mfu.py`](file:///Users/pmotgi/exploration/cerence/training-grpo/docker/calculate_mfu.py)
