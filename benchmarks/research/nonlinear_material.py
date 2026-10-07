"""Matched complete responses for fresh party-local masked-wire key encoding.

Both candidates use the same optimized native implementation and overlapped
dispatch. Prior dispatch and implementation cohorts retain their own identities.
"""
from benchmarks.research.nonlinear_dispatch import overlap


def full():
    return overlap().with_params(name="masked-wire-unfused-full")


def seeded():
    return full().with_params(name="masked-wire-unfused-seeded",
        pipeline__nonlinear__material_encoding="seeded")
