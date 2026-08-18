# MLflow Metrics Export & Deployment Guide for MaxText GRPO

This guide provides instructions for tracking and visualizing **MaxText GRPO RL post-training metrics** using MLflow. Two deployment options are covered:

1. **Option 1: Exporting to an External or In-Cluster MLflow Server**
   * **Method 1A (Recommended)**: 1-File Launcher with TensorBoard Autolog (**Zero MaxText code modifications**).
   * **Method 1B**: Direct native patch inside `src/maxtext/common/metric_logger.py`.
2. **Option 2: Deploying a Local In-Cluster MLflow Server on GKE** (Zero network/firewall friction).

---

## What Metrics Are Logged to MLflow?

MaxText automatically captures and exports the following metrics per training step:
* **Training Dynamics**: `learning/loss`, `learning/current_learning_rate`, `learning/grad_norm`, `learning/total_weights`.
* **RL & GRPO Metrics**: `rewards/exact_match`, `rewards/format`, `learning/dpo_loss` (if applicable), `eval/avg_loss`.
* **Hardware & Throughput Performance**: `perf/step_time_seconds`, `perf/per_device_tflops_per_sec`, `perf/per_device_tokens_per_sec`.

---

## Option 1: Exporting to an MLflow Server

### Method 1A: 1-File Launcher via TensorBoard Autolog (Recommended — Zero Code Changes to MaxText)

Because MaxText writes rich metrics to TensorBoard event files, MLflow's native `mlflow.tensorboard.autolog()` intercepts and streams all scalar summaries to MLflow in real time.

#### 1. The Launcher Script ([`train_with_mlflow.py`](file:///Users/pmotgi/exploration/cerence/training-grpo/train_with_mlflow.py))

Save [`train_with_mlflow.py`](file:///Users/pmotgi/exploration/cerence/training-grpo/train_with_mlflow.py) on your mounted PVC (`/checkpoint/train_with_mlflow.py`) or in your container image:

```python
"""MLflow TensorBoard Autolog Launcher for MaxText GRPO."""
import os
import sys
from absl import app
import mlflow

# 1. Point to MLflow tracking server
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "maxtext-grpo-training")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# 2. Enable TensorBoard Auto-Logging
mlflow.tensorboard.autolog()

# 3. Launch MaxText RL Trainer
from maxtext.trainers.post_train.rl.train_rl import main

if __name__ == "__main__":
    app.run(main)
```

#### 2. Update the JobSet Manifest

In your JobSet manifest ([`llama3.1-8b-grpo-dws-2x2x1-training.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/llama3.1-8b-grpo-dws-2x2x1-training.yaml)):

```yaml
            env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-service:5000"       # Or external IP / hostname
            - name: MLFLOW_EXPERIMENT_NAME
              value: "llama3-1-8b-grpo-training"
            command:
            - python3
            - /checkpoint/train_with_mlflow.py         # <-- Use the launcher script
            - maxtext/configs/post_train/rl.yml
            - model_name=llama3.1-8b-Instruct
            - tokenizer_path=meta-llama/Llama-3.1-8B-Instruct
            - run_name=llama3-1-8b-grpo-training-run
            - base_output_directory=/checkpoint/maxtext
            - chips_per_vm=4
```

---

### Method 1B: Direct Native Patch in `metric_logger.py` (Alternative)

If you prefer modifying the core MaxText library directly, add the following hooks into [`src/maxtext/common/metric_logger.py`](file:///Users/pmotgi/exploration/cerence/maxtext/src/maxtext/common/metric_logger.py):

#### 1. In `MetricLogger.__init__`:
```python
    self.enable_mlflow = getattr(config, "enable_mlflow", False) or bool(os.getenv("MLFLOW_TRACKING_URI"))
    if self.enable_mlflow and jax.process_index() == 0:
      import mlflow
      mlflow_uri = getattr(config, "mlflow_tracking_uri", "") or os.getenv("MLFLOW_TRACKING_URI")
      if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
      experiment_name = getattr(config, "mlflow_experiment_name", "") or os.getenv("MLFLOW_EXPERIMENT_NAME", "maxtext")
      mlflow.set_experiment(experiment_name)
      mlflow.start_run(run_name=config.run_name)
```

#### 2. In `MetricLogger.write_metrics`:
```python
      if self.enable_mlflow and jax.process_index() == 0:
        self.write_metrics_to_mlflow(metrics, step)
```

#### 3. Helper Method:
```python
  def write_metrics_to_mlflow(self, metrics, step):
    """Logs scalar metrics directly to MLflow."""
    import mlflow
    flat_metrics = {}
    for key, val in metrics.get("scalar", {}).items():
      flat_metrics[key] = float(val)
    for key, val in metrics.get("scalars", {}).items():
      for subkey, subval in val.items():
        flat_metrics[f"{key}/{subkey}"] = float(subval)
    mlflow.log_metrics(flat_metrics, step=int(step))
```

#### 4. In `MetricLogger.flush_metrics_and_cleanup`:
```python
    if self.enable_mlflow and jax.process_index() == 0:
      import mlflow
      mlflow.end_run()
```

---

## Option 2: Deploying In-Cluster MLflow on GKE

If your external MLflow server is blocked by VPC firewalls, or if you want a dedicated, zero-friction tracking server inside the cluster, deploy MLflow directly to GKE.

### Step 2.1: Deploy MLflow Server

Apply the included [`mlflow-in-cluster.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/mlflow-in-cluster.yaml) manifest:

```bash
kubectl apply -f training-grpo/mlflow-in-cluster.yaml
```

Verify that the MLflow pod and service are running:
```bash
kubectl get pods,svc -l app=mlflow
```
Expected output:
```
NAME                                 READY   STATUS    RESTARTS   AGE
pod/mlflow-server-7b4f8d66dc-k2m8p   1/1     Running   0          30s

NAME                     TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)    AGE
service/mlflow-service   ClusterIP   10.8.12.34    <none>        5000/TCP   30s
```

---

### Step 2.2: Point Training Pods to the In-Cluster Service

In your JobSet manifest, set:
```yaml
            env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-service:5000"
            - name: MLFLOW_EXPERIMENT_NAME
              value: "llama3-1-8b-grpo"
```

---

### Step 2.3: Access the MLflow Web UI on Your Local Machine

Forward the port to your browser:

```bash
kubectl port-forward svc/mlflow-service 5000:5000
```

Open your browser and navigate to **`http://localhost:5000`** to view real-time metrics!

---

## Summary Comparison

| Feature | Method 1A: Launcher Wrapper | Method 1B: Direct Source Patch |
| :--- | :--- | :--- |
| **MaxText Code Edits** | **None (0 lines modified)** | ~15 lines in `metric_logger.py` |
| **Mechanism** | `mlflow.tensorboard.autolog()` | Direct `mlflow.log_metrics()` |
| **File Required** | `train_with_mlflow.py` | None |
| **Compatibility** | Upstream MaxText safe | Requires maintaining patch |
