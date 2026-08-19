"""MLflow TensorBoard Autolog Launcher for MaxText GRPO RL Training.

This launcher enables real-time streaming of all MaxText training dynamics,
rewards, and hardware performance metrics to MLflow without modifying any
MaxText source code.

Usage:
  python3 train_with_mlflow.py maxtext/configs/post_train/rl.yml [KEY=VALUE overrides...]
"""

import os
import sys
from absl import app
import mlflow
import mlflow.tensorboard

# 1. Configure MLflow Tracking Server & Experiment
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "maxtext-grpo-training")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# 2. Enable Real-Time TensorBoard Autologging
# Automatically intercepts MaxText's TensorBoard SummaryWriter events
# (loss, learning_rate, step_time, tflops, rewards) and streams to MLflow.
mlflow.tensorboard.autolog()

# 3. Execute MaxText Post-Training Engine
from maxtext.trainers.post_train.rl.train_rl import main

if __name__ == "__main__":
  app.run(main)
