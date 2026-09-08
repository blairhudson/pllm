set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pllm_study.lifecycle --rtt-ms 1 --json results/lifecycle-rtt1.json
python -m pllm_study.lifecycle --rtt-ms 1 --draft 4 --json results/lifecycle-speculative-rtt1.json
python -m pllm_study.lifecycle --draft 4 --json results/lifecycle-speculative.json
