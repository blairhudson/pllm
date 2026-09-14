"""Proposed benchmark API; fresh resources are allocated for every candidate/run."""
from pllm import Benchmark, Study
from pllm.benchmarks import Decode
from private_app import experiment


def main() -> None:
    workload = Decode(input_tokens=32, output_tokens=16, termination="fixed")
    benchmark = Benchmark(
        experiment.with_params(budget__requests=6),
        workload=workload,
        warmup=1,
        repeats=5,
    )
    report = benchmark.run(output="runs/smoke")
    print(report.summary())
    study = Study(
        benchmark,
        space={"pipeline__components__kernels__threads": [1, 2, 4, 8]},
    )
    study.search(output="runs/thread-search")


if __name__ == "__main__":
    main()
