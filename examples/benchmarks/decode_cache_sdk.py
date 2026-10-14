"""Decode-cache cohort with harness-only RNG fixture; central integration gate first."""
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from benchmark_decode_cache_sdk import main as run
    run()


if __name__ == "__main__":
    main()
