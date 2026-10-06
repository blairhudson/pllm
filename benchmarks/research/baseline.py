"""Prepared W8A8 baseline; no optional transport, storage or prefix optimizations."""
from benchmarks.research.common import baseline


def experiment():
    return baseline("prepared-baseline")
