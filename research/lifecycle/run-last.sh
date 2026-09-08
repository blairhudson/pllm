set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
python -m pllm_study.head
OMP_NUM_THREADS=1 python -m pllm_study.lifecycle --rtt-ms 10 --json results/lifecycle-rtt10.json
OMP_NUM_THREADS=1 python -m pllm_study.lifecycle --rtt-ms 10 --draft 4 --json results/lifecycle-speculative-rtt10.json
