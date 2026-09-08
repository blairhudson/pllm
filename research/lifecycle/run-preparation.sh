#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
python -m pllm_study.benchmark --input 768 --output 256 --rounds 3 --json results/bridge-768-256.json
python -m pllm_study.benchmark --input 1536 --output 512 --rounds 3 --json results/bridge-1536-512.json
python -m pllm_study.benchmark --input 5120 --output 34816 --rounds 1 --json results/qwen-gate-up.json
python -m pllm_study.benchmark --input 17408 --output 5120 --rounds 1 --json results/qwen-down.json
python -m pllm_study.benchmark --input 5120 --output 16480 --rounds 1 --json results/qwen-delta-input.json
