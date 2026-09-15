#!/usr/bin/env python3
"""Finite public negative controls, not attacks against installed PLLM runtime."""

from __future__ import annotations

from collections import Counter
import json


def run_attacks() -> dict[str, object]:
    findings: list[dict[str, object]] = []

    def add(name: str, successes: int, cases: int, scope: str) -> None:
        findings.append(
            {
                "id": name,
                "outcome": "refuted_in_scope",
                "successes": successes,
                "cases": cases,
                "scope": scope,
                "target": "deliberately_weakened_public_fixture",
            }
        )

    q = 256
    successes = 0
    for x in range(q):
        mask = (97 * x + 13) % q
        first = (x - mask) % q
        second = (x + 31 - mask) % q
        successes += (second - first) % q == 31
    add("mask_reuse", successes, q, "Reused full-ring mask discloses input differences.")

    p = 251
    successes = 0
    for x in range(p):
        delta = [19, 73, 1]
        base = [61, 4, 151]
        first = [(value + x * offset) % p for value, offset in zip(base, delta)]
        second = [(value + (x + 7) * offset) % p for value, offset in zip(base, delta)]
        difference = [(right - left) % p for left, right in zip(first, second)]
        recovered = [value * pow(difference[-1], -1, p) % p for value in difference]
        successes += recovered == delta
    add(
        "affine_label_reuse",
        successes,
        p,
        "Two semantic values on one affine wire expose offset with unit coordinate.",
    )

    successes = 0
    domain = list(range(-9, 10))
    for shift in range(p):
        exposed = {(shift + x) % p for x in domain}
        candidates = [base for base in range(p) if {(base + x) % p for x in domain} == exposed]
        successes += candidates == [shift]
    add(
        "sparse_point_permute",
        successes,
        p,
        "Visible sparse row indices expose shift of proper cyclic interval.",
    )

    mismatches = 0
    for left in range(256):
        for right in range(256):
            mismatches += ((left >> 4) + (right >> 4)) % 16 != ((left + right) % 256) >> 4
    findings.append(
        {
            "id": "missing_truncation_carry",
            "outcome": "refuted_in_scope",
            "mismatches": mismatches,
            "cases": 65536,
            "scope": "Independent local shifts omit carry.",
            "target": "deliberately_weakened_public_fixture",
        }
    )

    successes = sum(((1 if x >= 128 else 8) == 1) == (x >= 128) for x in range(256))
    add("early_exit_metadata", successes, 256, "Unpadded sign-dependent path reveals sign.")

    successes = 0
    for x in range(256):
        mask = (13 * x + 7) % 256
        successes += ((x - mask) - ((x + 19) - mask)) % 256 == (-19) % 256
    add(
        "low_rank_projection",
        successes,
        256,
        "Shared rank-one mask leaves coordinate difference exposed.",
    )

    reference = None
    equal = True
    for x in range(8):
        view = Counter(((x - mask) % 8, (3 * mask - pad) % 8) for mask in range(8) for pad in range(8))
        equal &= len(view) == 64 and all(count == 1 for count in view.values())
        reference = view if reference is None else reference
        equal &= view == reference

    return {
        "schema_version": "pllm.attack_fixture_result.v1",
        "origin": "public_reference_fixtures_executed",
        "findings": findings,
        "ideal_uniform_control": {
            "outcome": "proved_in_model" if equal else "refuted_in_scope",
            "scope": "Exact enumeration over q=8 ideal independent uniform masks.",
            "assignments": 512,
        },
        "limitations": [
            "production runtime not attacked",
            "whole-protocol privacy not established",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_attacks(), indent=2, sort_keys=True))
