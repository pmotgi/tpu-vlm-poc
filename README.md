# Cerence TPU Proof-of-Concept (POC)

This repository contains end-to-end configurations, Kubernetes manifests, and deployment blueprints for running LLM workloads (specifically **Llama 3.3 70B**) on Google Cloud TPUs and GKE.

---

## Repository Structure

```
.
├── deployment/
│   ├── gke-tpu-7x.yaml            # Cluster Toolkit blueprint for provisioning GKE with TPU v7x
│   └── README.md                  # Deployment guide for GKE on GCP
├── training/
│   ├── llama3-3-70b-checkpoint-converter.yaml  # Converts weights to Orbax format
│   ├── llama3-3-70b-lora-training.yaml         # Multi-host TPU LoRA training (MaxText + Tunix)
│   ├── llama3-3-70b-checkpoint-validation.yaml   # Decode verification and validation
│   ├── hf-conversion-pod.yaml                  # Converts trained adapters back to HF format
│   └── README.md                               # Detailed training workflow documentation
└── inference/
    ├── vllm-deployment.yaml       # vLLM OpenAI API deployment with LoRA support
    └── README.md                  # Inference setup and serving guide
```

---

## Streams Overview

### 1. Infrastructure Deployment (`deployment/`)
Showcases how to provision a production-ready Google Kubernetes Engine (GKE) cluster on Google Cloud Platform optimized for TPU workloads using **Cluster Toolkit** (`gcluster`). Includes setup for TPU v7x node pools, Workload Identity, Kueue, JobSet, and GCS FUSE storage.

### 2. Distributed Training & Fine-Tuning (`training/`)
Demonstrates distributed fine-tuning of **Llama 3.3 70B** on Google Cloud TPUs using **MaxText** and **Tunix**:
1. **Checkpoint Conversion**: Convert standard base model weights into MaxText Orbax format.
2. **LoRA Fine-Tuning**: Execute multi-host LoRA fine-tuning across TPU slices (e.g. 2x2x2 mesh topology).
3. **Validation**: Validate decode performance directly on TPU hardware.
4. **Hugging Face Export**: Convert learned LoRA weights into Hugging Face compatible format for downstream serving.

### 3. Model Serving & Inference (`inference/`)
Demonstrates high-throughput serving of large language models and LoRA adapters on TPUs using **vLLM** (`tpu-inference`). Configures tensor parallelism, dynamic LoRA loading, and Kubernetes Service load balancing.

---

## Quick Start

### Prerequisites
1. GCP Project with TPU quota and GKE permissions.
2. `kubectl`, `gcloud`, and `gcluster` (Cluster Toolkit) CLI tools installed.
3. Hugging Face Access Token secret in GKE:
   ```bash
   kubectl create secret generic hf-secret --from-literal=hf_api_token="YOUR_HF_TOKEN"
   ```

### 1. Provision Cluster
```bash
cd deployment
gcluster deploy gke-tpu-7x.yaml
```

### 2. Run Training Pipeline
```bash
# 1. Convert Checkpoints
kubectl apply -f training/llama3-3-70b-checkpoint-converter.yaml

# 2. Run LoRA Training
kubectl apply -f training/llama3-3-70b-lora-training.yaml

# 3. Validate Checkpoint
kubectl apply -f training/llama3-3-70b-checkpoint-validation.yaml

# 4. Export to Hugging Face format
kubectl apply -f training/hf-conversion-pod.yaml
```

### 3. Serve Model via vLLM
```bash
kubectl apply -f inference/vllm-deployment.yaml
```

---

## License
Apache 2.0
