# MaxText TPU Distributed Training & Fine-Tuning

This directory contains manifests and configurations for executing distributed fine-tuning (LoRA), checkpoint conversion, and validation for Llama 3.3 70B on TPUs using [MaxText](https://github.com/google/maxtext).

## Workflow Pipeline

1. **Checkpoint Conversion (`llama3-3-70b-checkpoint-converter.yaml`)**:
   - Converts Hugging Face / Meta Llama 3.3 70B weights into MaxText-compatible Orbax checkpoint format across TPU slices using `JobSet`.

2. **Distributed LoRA Training (`llama3-3-70b-lora-training.yaml`)**:
   - Runs multi-host LoRA fine-tuning using MaxText & Tunix on TPU slices (e.g., 2x2x2 topology).
   - Saves checkpoint outputs to GCS.

3. **Checkpoint Validation (`llama3-3-70b-checkpoint-validation.yaml`)**:
   - Executes distributed text generation and decode verification to ensure checkpoint integrity.

4. **Hugging Face Format Export (`hf-conversion-pod.yaml`)**:
   - Exports trained LoRA adapters back into standard Hugging Face format for downstream inference with vLLM.

## Prerequisites

- Active GKE cluster with TPU node pool.
- Kubernetes Secret `hf-secret` containing your `hf_api_token`.
- GCS PVC named `checkpoint-data-pmotgi-tpu-v7x-04a179d1-pvc`.
