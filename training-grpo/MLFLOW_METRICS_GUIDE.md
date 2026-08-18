# MLflow Metrics Export & Deployment Guide for MaxText GRPO

This guide provides instructions for tracking and visualizing **MaxText GRPO RL post-training metrics** using MLflow. Two deployment options are covered:

1. **Option 1: Exporting to an Accessible External MLflow Server** (Cloud Run, VM, or Hosted Server).
2. **Option 2: Deploying a Local In-Cluster MLflow Server on GKE** (Zero network/firewall friction).

---

## What Metrics Are Logged to MLflow?

MaxText automatically captures and exports the following metrics per training step:
* **Training Dynamics**: `learning/loss`, `learning/current_learning_rate`, `learning/grad_norm`, `learning/total_weights`.
* **RL & GRPO Metrics**: `rewards/exact_match`, `rewards/format`, `learning/dpo_loss` (if applicable), `eval/avg_loss`.
* **Hardware & Throughput Performance**: `perf/step_time_seconds`, `perf/per_device_tflops_per_sec`, `perf/per_device_tokens_per_sec`.

---

## Option 1: Exporting to an External MLflow Server

Use this option if your organization already hosts a central MLflow tracking server that is network-accessible from the GKE cluster.

### Step 1.1: Add MLflow Support to MaxText (`src/maxtext/common/metric_logger.py`)

Open [`src/maxtext/common/metric_logger.py`](file:///Users/pmotgi/exploration/cerence/maxtext/src/maxtext/common/metric_logger.py) and add the following hooks:

#### 1. In `MetricLogger.__init__` (around line 122):
```python
    # Auto-initialize MLflow if MLFLOW_TRACKING_URI or enable_mlflow is present
    self.enable_mlflow = getattr(config, "enable_mlflow", False) or bool(os.getenv("MLFLOW_TRACKING_URI"))
    if self.enable_mlflow and jax.process_index() == 0:
      import mlflow  # lazy import
      mlflow_uri = getattr(config, "mlflow_tracking_uri", "") or os.getenv("MLFLOW_TRACKING_URI")
      if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
      experiment_name = getattr(config, "mlflow_experiment_name", "") or os.getenv("MLFLOW_EXPERIMENT_NAME", "maxtext")
      mlflow.set_experiment(experiment_name)
      mlflow.start_run(run_name=config.run_name)
```

#### 2. In `MetricLogger.write_metrics` (around line 147):
```python
      if self.enable_mlflow and jax.process_index() == 0:
        self.write_metrics_to_mlflow(metrics, step)
```

#### 3. Add the `write_metrics_to_mlflow` helper method:
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

#### 4. In `MetricLogger.flush_metrics_and_cleanup` (around line 510):
```python
    if self.enable_mlflow and jax.process_index() == 0:
      import mlflow
      mlflow.end_run()
```

---

### Step 1.2: Configure Environment Variables in the JobSet Manifest

In your JobSet manifest ([`llama3.1-8b-grpo-dws-2x2x1-training.yaml`](file:///Users/pmotgi/exploration/cerence/training-grpo/llama3.1-8b-grpo-dws-2x2x1-training.yaml)), inject the tracking URI and any authentication credentials:

```yaml
            env:
            - name: MLFLOW_TRACKING_URI
              value: "http://<EXTERNAL_MLFLOW_IP_OR_HOSTNAME>:5000"
            - name: MLFLOW_EXPERIMENT_NAME
              value: "llama3-1-8b-grpo-post-training"
            # If your MLflow server requires basic authentication:
            - name: MLFLOW_TRACKING_USERNAME
              valueFrom:
                secretKeyRef:
                  name: mlflow-auth-secret
                  key: username
            - name: MLFLOW_TRACKING_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mlflow-auth-secret
                  key: password
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

### Step 2.2: Configure JobSet to Point to the In-Cluster Service

Since MLflow is inside the cluster, all TPU pods can reach it using the internal DNS name: `http://mlflow-service:5000`.

Update your JobSet manifest:
```yaml
            env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-service:5000"
            - name: MLFLOW_EXPERIMENT_NAME
              value: "llama3-1-8b-grpo"
```

---

### Step 2.3: Access the MLflow Web UI on Your Local Machine

Forward the port from GKE to your local browser:

```bash
kubectl port-forward svc/mlflow-service 5000:5000
```

Open your browser and navigate to:
```
http://localhost:5000
```

You can now monitor live loss curves, reward scores, and hardware TFLOPs per step as the TPU training progresses.

---

## Summary Comparison

| Feature | Option 1: External MLflow | Option 2: In-Cluster GKE MLflow |
| :--- | :--- | :--- |
| **Tracking URI** | `http://<EXTERNAL_IP>:5000` | `http://mlflow-service:5000` |
| **Firewall Setup** | Requires GCP Firewall rule for GKE Pod CIDR | **None** (internal cluster network) |
| **Deployment Effort** | Uses existing server | `kubectl apply -f mlflow-in-cluster.yaml` |
| **UI Access** | Direct external URL | `kubectl port-forward svc/mlflow-service 5000:5000` |
| **Persistence** | Remote database / storage | PersistentVolumeClaim (`mlflow-storage-pvc`) |
