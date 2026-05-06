#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [INSTANCE] [ZONE] [PROJECT]

Start a GCP Compute Engine VM. All arguments are positional and optional.

Arguments:
  INSTANCE   VM instance name        (default: l4-32vcpu-16core-128ram-250disk)
  ZONE       GCP zone                (default: us-central1-a)
  PROJECT    GCP project ID          (default: columbia-492622)

Options:
  -h, --help   Show this help and exit

Examples:
  $(basename "$0")
  $(basename "$0") my-vm
  $(basename "$0") my-vm us-east1-b my-project
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

INSTANCE="${1:-l4-32vcpu-16core-128ram-250disk}"
ZONE="${2:-us-central1-a}"
PROJECT="${3:-columbia-492622}"

gcloud compute instances start "$INSTANCE" \
  --zone "$ZONE" \
  --project "$PROJECT"
