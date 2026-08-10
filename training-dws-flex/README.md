# DWS-Flex LoRA to Hugging Face Conversion Guide & Troubleshooting

This document records the issues, fixes, and exact runnable commands required to run the MaxText LoRA to Hugging Face conversion (`hf-conversion-pod.yaml`) in the DWS-Flex GKE environment.

---

## Key Fixes Applied & Complete Commands Executed

### 1. Pod Lifecycle & Log Visibility
- **Issue**: Standard `Job` / `JobSet` resources delete pods upon completion or failure, making logs unavailable if garbage collected.
- **Fix**: Configured `hf-conversion-pod.yaml` as a standalone `Pod` resource (`kind: Pod`). The pod remains in `Completed` or `Error` state after execution, preserving logs for inspection.
- **Complete Commands**:
  ```bash
  # Delete existing pod (if any) and deploy standalone Pod
  kubectl delete pod lora-hf-conversion --ignore-not-found
  kubectl apply -f /Users/pmotgi/exploration/cerence/training-dws-flex/hf-conversion-pod.yaml

  # Check pod status and stream conversion logs
  kubectl get pod lora-hf-conversion
  kubectl logs -f lora-hf-conversion -c converter
  ```

### 2. Missing Kubernetes Secret (`hf-secret`)
- **Issue**: Pod failed on startup with `CreateContainerConfigError` due to missing `hf-secret`.
- **Fix**: Created/updated secret in the `default` namespace containing the Hugging Face API token.
- **Complete Commands**:
  ```bash
  kubectl create secret generic hf-secret --from-literal=hf_api_token="<YOUR_HF_TOKEN>" --dry-run=client -o yaml | kubectl apply -f -
  ```

### 3. GCS Checkpoint Path Alignment & Transfer
- **Issue**: `to_huggingface.py` threw `AttributeError: 'NoneType' object has no attribute 'tree'` because `_CHECKPOINT_METADATA` and checkpoint files were missing from the target DWS Flex GCS bucket (`checkpoint-data-cerence-gke-test-tbd-6e9665fd`).
- **Fix**: Transferred trained LoRA checkpoint files from the source bucket to the target bucket and ensured `_CHECKPOINT_METADATA` was placed at the expected directory level.
- **Complete Commands**:
  ```bash
  # Copy checkpoint files from source bucket to target DWS-Flex bucket
  gcloud storage cp -r gs://checkpoint-data-pmotgi-tpu-v7x-04a179d1/llama3.3-70b/lora_training/llama3-3-70b-lora-2x2x2-bs16/checkpoints/24/ gs://checkpoint-data-cerence-gke-test-tbd-6e9665fd/llama3.3-70b/lora_training/llama3-3-70b-lora-2x2x2-bs16/checkpoints/24/

  # Fix nested directory structure for _CHECKPOINT_METADATA
  gcloud storage mv "gs://checkpoint-data-cerence-gke-test-tbd-6e9665fd/llama3.3-70b/lora_training/llama3-3-70b-lora-2x2x2-bs16/checkpoints/24/24/_CHECKPOINT_METADATA" "gs://checkpoint-data-cerence-gke-test-tbd-6e9665fd/llama3.3-70b/lora_training/llama3-3-70b-lora-2x2x2-bs16/checkpoints/24/_CHECKPOINT_METADATA"

  # Clean up duplicate nested directory
  gcloud storage rm -r "gs://checkpoint-data-cerence-gke-test-tbd-6e9665fd/llama3.3-70b/lora_training/llama3-3-70b-lora-2x2x2-bs16/checkpoints/24/24/"
  ```

### 4. Gated Repository Authentication (Hugging Face 401 Unauthorized)
- **Issue**: Downloading tokenizer config failed with `401 Unauthorized` / `GatedRepoError` for `meta-llama/Llama-3.1-70B`.
- **Fix**: Updated `hf-secret` with an API token associated with a Hugging Face account that has accepted Meta's Llama 3.1 license agreement.
- **Complete Verification Command**:
  ```bash
  curl -i -s -H "Authorization: Bearer <YOUR_HF_TOKEN>" https://huggingface.co/meta-llama/Llama-3.1-70B/resolve/main/config.json
  ```

---

## End-to-End Workflow to Run Again

```bash
# 1. Update/Apply Secret
kubectl create secret generic hf-secret --from-literal=hf_api_token="<YOUR_HF_TOKEN>" --dry-run=client -o yaml | kubectl apply -f -

# 2. Deploy Pod
kubectl delete pod lora-hf-conversion --ignore-not-found
kubectl apply -f /Users/pmotgi/exploration/cerence/training-dws-flex/hf-conversion-pod.yaml

# 3. Monitor Execution
kubectl get pod lora-hf-conversion -w
kubectl logs -f lora-hf-conversion -c converter
```

**Output Artifact Location**:
Upon successful completion (`STATUS: Completed`), converted Hugging Face adapter weights are saved to GCS at:
`/checkpoint/llama3.3-70b/lora_training/llama3-3-70b-lora-2x2x2-bs16/hf_lora_adapter`
(which corresponds to `gs://checkpoint-data-cerence-gke-test-tbd-6e9665fd/llama3.3-70b/lora_training/llama3-3-70b-lora-2x2x2-bs16/hf_lora_adapter`).

---

## Adding a 4x4 `ct6e-standard-4t` (TPU v6e) Node Pool

To add a new **4x4 `ct6e-standard-4t` (TPU v6e - Trillium)** node pool to the cluster and configure Kueue to route jobs to it:

### 1. Create Node Pool in GKE
```bash
gcloud container node-pools create gke-tpu-v6e-4x4-pool \
  --cluster=cerence-gke-test-tbd \
  --location=us-central1 \
  --zone=us-central1-c \
  --node-locations=us-central1-c \
  --machine-type=ct6e-standard-4t \
  --tpu-topology=4x4 \
  --placement-policy=HIGH_THROUGHPUT \
  --enable-queued-provisioning \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=4 \
  --disk-type=hyperdisk-balanced \
  --service-account=gke-np-sa@northam-ce-mlai-tpu.iam.gserviceaccount.com \
  --project=northam-ce-mlai-tpu
```

### 2. Configure Kueue Resources
```bash
kubectl apply -f /Users/pmotgi/exploration/cerence/training-dws-flex/kueue-ct6e-4x4-setup.yaml
```

### 3. Submit Jobs to 4x4 `ct6e` Pool
```bash
kubectl apply -f /Users/pmotgi/exploration/cerence/training-dws-flex/ct6e-4x4-sample-job.yaml

# Monitor admission and pod execution
kubectl get workload -l jobset.sigs.k8s.io/jobset-name=ct6e-4x4-test-job
kubectl get pods -l app=ct6e-4x4-test -w
```

