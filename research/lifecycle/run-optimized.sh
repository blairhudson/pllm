set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
python -m pllm_study.online --json results/online-compiled.json
python -m pllm_study.benchmark --input 768 --output 256 --rounds 3 --json results/optimized-768-256.json
python -m pllm_study.benchmark --input 5120 --output 34816 --rounds 2 --json results/optimized-gate-up.json
python -m pllm_study.benchmark --input 17408 --output 5120 --rounds 2 --json results/optimized-down.json
python -m pllm_study.benchmark --input 5120 --output 16480 --rounds 2 --json results/optimized-delta-input.json
python -m pllm_study.benchmark --input 6144 --output 5120 --rounds 2 --json results/optimized-mixer-output.json
python -m pllm_study.benchmark --input 5120 --output 14336 --rounds 2 --json results/optimized-attention-input.json
python -m pllm_study.benchmark --input 5120 --output 1024 --rounds 2 --json results/optimized-head-tile.json
