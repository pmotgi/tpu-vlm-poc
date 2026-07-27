# GKE Deployment on GCP using Cluster Toolkit

This directory contains blueprints and documentation for provisioning Google Kubernetes Engine (GKE) clusters configured for TPU workloads on Google Cloud Platform using [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit).

## Overview

Cluster Toolkit simplifies infrastructure creation for AI/ML workloads on GCP. The included blueprint provisions:
- A GKE Cluster with Workload Identity enabled.
- TPU v7x Node Pools with high-throughput placement policies.
- GKE add-ons including JobSet, Kueue, and GCS FUSE CSI driver.

## Deployment Steps

1. **Install Cluster Toolkit (`gcluster`)**:
   Follow instructions to download and install `gcluster` CLI.

2. **Configure Blueprint Variables**:
   Update `gke-tpu-7x.yaml` with your GCP Project ID, region, zone, and cluster options:
   ```yaml
   vars:
     project_id: YOUR_PROJECT_ID
     region: us-east5
     zone: us-east5-a
     num_slices: 1
   ```

3. **Deploy Infrastructure**:
   ```bash
   gcluster deploy gke-tpu-7x.yaml --auto-approve
   ```

4. **Connect to your GKE Cluster**:
   ```bash
   gcloud container clusters get-credentials <cluster-name> --region <region> --project <project-id>
   ```
