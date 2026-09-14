#!/usr/bin/env bash
# Design example: importing private_app.py executes trusted local Python code.
set -euo pipefail
cd "$(dirname "$0")"
pllm config export private_app.py:experiment --output exported.yaml
pllm plan check private_app.py:experiment
pllm run private_app.py:experiment --input-file question.txt
