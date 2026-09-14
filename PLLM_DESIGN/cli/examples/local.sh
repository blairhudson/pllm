#!/usr/bin/env bash
# Design example: these commands require the proposed PLLM CLI.
set -euo pipefail
cd "$(dirname "$0")"
pllm plan check pllm.yaml
pllm run pllm.yaml --input-file question.txt
