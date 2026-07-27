# TPU Model Inference using vLLM

This directory contains manifests and deployment files for serving large language models (such as Llama 3.3 70B with LoRA adapters) on Google Cloud TPUs / GPUs using [vLLM](https://github.com/vllm-project/vllm).

## Components

- **`vllm-deployment.yaml`**:
  - Kubernetes `Deployment` & `Service` for serving an OpenAI-compatible API server via vLLM.
  - Mounts model weights and fine-tuned LoRA adapters directly from Cloud Storage via GCS FUSE CSI driver.
  - Configures tensor parallelism (`--tensor-parallel-size=4`) and LoRA module loading.

## How to Deploy

```bash
kubectl apply -f inference/vllm-deployment.yaml
```

Check deployment status:
```bash
kubectl get pods -l app=vllm
kubectl get svc vllm-service
```
