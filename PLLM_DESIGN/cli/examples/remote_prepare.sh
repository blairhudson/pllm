#!/usr/bin/env bash
# Client host; both role services must be available for compilation/preparation.
# The example domains and identity files require actual deployment provisioning.
set -euo pipefail
cd "$(dirname "$0")"
pllm plan compile private_app.py:experiment \
  --deployment remote.yaml --output build/qwen
pllm prepare build/qwen/plan.lock.json --deployment remote.yaml
# After READY, Preparation can be stopped. See remote_run.sh.
