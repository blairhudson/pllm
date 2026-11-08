"""Describe the unchanged pinned FuseFSS 15-input/1-output candidate.

From repository root, after fresh resource-exception USER APPROVAL for the actual
trained import (not covered by GKM9 approval), a future ordinary shell invocation
with a 600000 ms timeout is:

    .venv/bin/python -I examples/papers/fusefss_resource_quote.py \
        --output target/fusefss-resource-quote.json

An explicitly approved finite description-only mixed-work ceiling can instead use:

    .venv/bin/python -I examples/papers/fusefss_resource_quote.py \
        --work-policy mixed-work-40 --output target/fss-resource-quote-work40.json

This option changes only the cumulative mixed-work comparison ceiling to 2^40;
it does not grant a parent reservation, execution admission or a hard host bound.
No output file is selected implicitly; existing partial reports are not replaced
unless their path is explicitly supplied.

Separately approved predicted cumulative peer-frame observation can add
--observe-peer-frames with an explicit separate output such as
target/fss-resource-quote-work40-observe-frames.json. It does not emit frames,
relax actual transport or public-copy/native ceilings, or grant execution.

The trusted factory supplies the shared pinned, local-files-only model, checked
dense-RoPE Q12 profile, decoder capacity, tensor-v1 and batch-v1 unchanged.
This calls the direct SourceImport description API: real trained-source I/O and
compilation, not a geometry estimator. It has NO ordinary benchmark memory
admission, watchdog, or hard RSS guarantee. A shell timeout is not such a guarantee.
No Client, roles, material issuance or generation are requested here.

Complete means the descriptive walk finished, NOT execution admission, even when
no material refusal was reported. Partial fields are raw intra-kernel counters,
NOT a coherent kernel prefix or full demand. Preserve the first material refusal
and stop context; unsupported material_wire_bytes remains null, not zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pllm._cli.targets import resolve_target
from pllm.runtime.fss_source import FssSourceError, describe_source


def resource_quote(*, work_policy=None, observe_peer_frames=False):
    """Return public description scalars without granting execution admission."""
    if type(observe_peer_frames) is not bool:
        raise FssSourceError("FuseFSS observe_peer_frames requires bool")
    experiment = resolve_target(
        "examples/papers/fusefss.py:qwen_empty_system",
        factory=True, no_input=True, trust_python=True, output_format="json",
    ).configuration
    options = {}
    if work_policy is not None:
        options["work_policy"] = work_policy
    if observe_peer_frames:
        options["observe_peer_frames"] = True
    description = describe_source(experiment, **options)
    return {**description, "execution_admitted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-policy", choices=["mixed-work-40"], default=None,
                        help="finite source-description-only mixed-work ceiling; not execution admission")
    parser.add_argument("--observe-peer-frames", action="store_true",
                        help="observe cumulative predicted peer frames only; actual transport limits stay hard")
    args = parser.parse_args(argv)
    payload = json.dumps(resource_quote(work_policy=args.work_policy,
                                       observe_peer_frames=args.observe_peer_frames), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)


if __name__ == "__main__":
    main()
