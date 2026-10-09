"""Exact prepared prefill transport under calibrated shared access caps."""
from benchmarks.research.prepared_frontier import serial


def control():
    return serial().with_params(name="prepared-prefill-whole-stage", budget__requests=2)


def chunk4():
    return control().with_params(name="prepared-prefill-four-rows", pipeline__linear__prefill_chunk_rows=4)


def chunk16():
    return control().with_params(name="prepared-prefill-sixteen-rows", pipeline__linear__prefill_chunk_rows=16)
