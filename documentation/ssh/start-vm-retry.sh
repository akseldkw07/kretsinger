#!/usr/bin/env bash
# Retry `gcloud compute instances start` until the VM starts.
# - On ZONE_RESOURCE_POOL_EXHAUSTED: sleep 1s and try again.
# - On any other failure: log the error and exit non-zero.
# On success: print retry count and elapsed time.

set -uo pipefail

INSTANCE="l4-32vcpu-16core-128ram-250disk"
ZONE="us-central1-a"
PROJECT="columbia-492622"

attempt=0
start_ts=$(date +%s)

while true; do
  attempt=$((attempt + 1))
  output=$(gcloud compute instances start "$INSTANCE" \
    --zone "$ZONE" \
    --project "$PROJECT" 2>&1)
  status=$?

  if [ $status -eq 0 ]; then
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    retries=$((attempt - 1))
    echo "$output"
    echo ""
    echo "✓ VM started successfully."
    echo "  Retries: $retries"
    echo "  Elapsed: ${elapsed}s"
    exit 0
  fi

  if echo "$output" | grep -q "ZONE_RESOURCE_POOL_EXHAUSTED"; then
    echo "[attempt $attempt] zone exhausted — retrying in 1s..."
    sleep 1
    continue
  fi

  echo "✗ Unexpected failure on attempt $attempt:" >&2
  echo "$output" >&2
  exit $status
done
