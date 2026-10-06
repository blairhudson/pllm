"""Run SDK beam search and save every measured Experiment as a Python factory.

Each invocation is a separate cohort. Remeasure finalists with their matched
controls before promoting a research score; single-trial search is exploratory.
"""
import argparse
import os
from pathlib import Path
import hashlib
import json

from benchmarks.research.combinations import search_space
from pllm import benchmark_search
from pllm.search import BeamSearch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topology", choices=("prepared", "offset", "client"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--public-prefix", action="store_true")
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--objective", choices=("request_tps", "full_seconds", "covered_bytes", "online_bytes"), default="request_tps")
    args = parser.parse_args()
    root = Path(os.environ["PLLM_RESEARCH_ARTIFACTS"])
    search = BeamSearch(f"qwen25-{args.topology}", search_space(args.topology, public_prefix=args.public_prefix),
        args.objective, "max" if args.objective == "request_tps" else "min",
        width=args.width, max_trials=args.trials)
    report = benchmark_search(search, (root / "prompt.txt").read_text(), max_output_tokens=8,
        progress=lambda candidate, message: print(f"[{candidate.index + 1}/{args.trials}] {message}", flush=True)).to_dict()
    output_dir = Path(__file__).parent / "trials" / args.output.stem.replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in report["candidates"]:
        source = record.pop("python_source")
        path = output_dir / f"trial_{record['configuration_digest'][:16]}.py"
        path.write_text(source)
        record["configuration_file"] = str(path.relative_to(Path(__file__).resolve().parents[2]))
        record["configuration_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: report["search"][key] for key in (
        "best_configuration_digest", "attempted", "stop_reason")}, indent=2))


if __name__ == "__main__":
    main()
