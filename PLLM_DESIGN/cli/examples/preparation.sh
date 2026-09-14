#!/usr/bin/env bash
# Run on the Preparation host. This does not start an Inference worker.
set -euo pipefail
pllm party serve preparation \
  --listen 0.0.0.0:7444 \
  --identity /etc/pllm/preparation.json \
  --state-dir /var/lib/pllm/preparation
