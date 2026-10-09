"""Run a bounded stage study, including interactions and retest coverage."""
from __future__ import annotations

import argparse
import json

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.evidence import EvidenceRegistry
from pllm.metrics import (BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef,
                          Sampling, benchmark_suite)
from pllm.profiles import MaskedLinearCpu
from pllm.search import FrontierStudy, Variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    experiment = Experiment("stage-study", MaskedLinearCpu(Model.tiny()),
        Deployment.local(root="local://benchmark-target-example"), ExecutionBudget(1, 16, 2))
    suite = BenchmarkSuite((BenchmarkCase("prepare", BenchmarkTarget.preparation(experiment), BuildRef.capture()),),
                           Sampling(processes=3))
    study = FrontierStudy(suite, (Variant("digit4", {"kernel": "digit4"}),
                                Variant("chunks", {"chunk_rows": 4})), (("digit4", "chunks"),))
    work = study.plan(max_trials=12)
    result = benchmark_suite(work.suite, output=args.output)
    registry = EvidenceRegistry().register_all(result.results)
    print(json.dumps(study.snapshot(result, metrics={"target.cpu": "min"}).to_dict(), indent=2))
    assert study.plan(registry).suite is None


if __name__ == "__main__":
    main()
