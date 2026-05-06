#!/usr/bin/env bash
set -euo pipefail

gcloud compute instances start l4-32vcpu-16core-128ram-250disk \
  --zone us-central1-a \
  --project columbia-492622
