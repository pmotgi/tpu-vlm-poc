# MaxText Post-Training RL / GRPO on TPU v7x

This directory contains the production, tested, and verified Kubernetes JobSet manifest for running **Group Relative Policy Optimization (GRPO)** with **vLLM Rollout Sampling** on Google Cloud TPU v7x (`tpu-v7x-spot-2x2x1`).

---

## Verified Manifests

### Option A: **Direct Spot Workload (`2x2x1` Topology)**
* **Manifest File**: [`llama3.1-8b-grpo-spot-2x2x1-training.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/llama3.1-8b-grpo-spot-2x2x1-training.yaml)
* **Target Hardware**: `tpu-v7x-spot-2x2x1` (1 VM with 4 TPU chips / 8 Tensor Core devices)
  * **4 Trainer Devices (`TPU_0` to `TPU_3`)**: MaxText Actor Policy gradient optimizer (AdamW) + reference model log-probability computation.
  * **4 Sampler Devices (`TPU_4` to `TPU_7`)**: vLLM online inference with Pallas Ragged Paged Attention (RPA).
* **Dataset**: `openai/gsm8k`
* **Status**: Tested & Verified (`Succeeded`)

#### How to Run (Direct Spot):
```bash
kubectl apply -f training-grpo/llama3.1-8b-grpo-spot-2x2x1-training.yaml
```

---

### Option B: **DWS-Flex & Kueue Orchestration (`2x2x1` Topology)**
* **Kueue Setup File**: [`kueue-tpu7x-2x2x1-setup.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/kueue-tpu7x-2x2x1-setup.yaml)
* **JobSet Manifest File**: [`llama3.1-8b-grpo-dws-2x2x1-training.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/llama3.1-8b-grpo-dws-2x2x1-training.yaml)

#### 1. Create the DWS-Flex 2x2x1 Node Pool (Tested & Verified)
```bash
gcloud container node-pools create <CUSTOMER_NODEPOOL_NAME> \
  --cluster=<CUSTOMER_CLUSTER_NAME> \
  --location=<REGION> \
  --node-locations=<ZONE> \
  --machine-type="tpu7x-standard-4t" \
  --flex-start \
  --reservation-affinity=none \
  --enable-autoscaling \
  --num-nodes=0 \
  --min-nodes=0 \
  --max-nodes=1 \
  --disk-type="hyperdisk-balanced" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --project=<PROJECT_ID>
```

#### 2. Apply the Kueue Configuration
Update `<CUSTOMER_NODEPOOL_NAME>` in `kueue-tpu7x-2x2x1-setup.yaml` and apply:
```bash
kubectl apply -f training-grpo/kueue-tpu7x-2x2x1-setup.yaml
```

#### 3. Submit the GRPO JobSet to Kueue
Update image and volume placeholders in `llama3.1-8b-grpo-dws-2x2x1-training.yaml` and submit:
```bash
kubectl apply -f training-grpo/llama3.1-8b-grpo-dws-2x2x1-training.yaml
```

#### 4. Monitor Provisioning & Execution
```bash
# Check Kueue Workload admission status
kubectl get workloads

# Watch DWS Provisioning Request (ACCEPTED -> PROVISIONED)
kubectl get provisioningrequests -w

# Watch TPU node scaling from 0 to 1
kubectl get nodes -w

# Stream GRPO training logs
kubectl logs -l app=llama3-1-8b-grpo-training -c grpo-trainer -f
```

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
* **MLflow Metrics & Tracking Guide**: [`training-grpo/MLFLOW_METRICS_GUIDE.md`](file:///Users/pmotgi/exploration/cerence/training-grpo/MLFLOW_METRICS_GUIDE.md)
* **In-Cluster MLflow Deployment Manifest**: [`training-grpo/mlflow-in-cluster.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/mlflow-in-cluster.yaml)
* **Custom Dataset & Reward Functions Guide**: [`training-grpo/CUSTOM_DATASET_AND_REWARDS_GUIDE.md`](file:///Users/pmotgi/exploration/cerence/training-grpo/CUSTOM_DATASET_AND_REWARDS_GUIDE.md)
* **TRL to MaxText Mapping**: [`training-grpo/TRL_TO_MAXTEXT_MAPPING.md`](file:///Users/pmotgi/exploration/cerence/training-grpo/TRL_TO_MAXTEXT_MAPPING.md)
* **Custom Reward Functions Template**: [`training-grpo/custom_rewards_template.py`](file:///Users/pmotgi/exploration/cerence/training-grpo/custom_rewards_template.py)
* **Kueue 2x2x1 Setup Manifest**: [`training-grpo/kueue-tpu7x-2x2x1-setup.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/kueue-tpu7x-2x2x1-setup.yaml)
* **DWS 2x2x1 JobSet Manifest**: [`training-grpo/llama3.1-8b-grpo-dws-2x2x1-training.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/llama3.1-8b-grpo-dws-2x2x1-training.yaml)
* **MFU Performance Calculator**: [`training-grpo/docker/calculate_mfu.py`](file:///Users/pmotgi/exploration/cerence/training-grpo/docker/calculate_mfu.py)
