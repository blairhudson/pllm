#!/usr/bin/env bash
# Run on the Inference host. Identity/authorization files are provisioned by its owner.
set -euo pipefail
pllm party serve inference \
  --listen 0.0.0.0:7443 \
  --identity /etc/pllm/inference.json \
  --state-dir /var/lib/pllm/inference
