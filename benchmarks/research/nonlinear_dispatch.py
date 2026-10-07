"""Matched whole-response controls for bounded masked-wire dispatch overlap.

The historical decoder factories remain frozen. Both candidates retain their
unfused Q12 numeric mapping, checkpoint, placement and artifact-delivery choices.
"""
from benchmarks.research.nonlinear_decoders import unfused


def serial():
    return unfused().with_params(name="masked-wire-unfused-serial")


def overlap():
    return serial().with_params(name="masked-wire-unfused-overlap",
        pipeline__nonlinear__dispatch="overlap")
