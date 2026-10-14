"""Replay matched cold growing-epoch SDK cohort; no build/install side effects.

Use --preflight-only first. Real defaults: cap 64, two workloads, four policies,
three alternating replicates. --tiny --replicates 1 is functionality only.
"""
from pathlib import Path
import sys


def main():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    from benchmark_growing_epochs_sdk import main as run
    run()


if __name__ == "__main__":
    main()
