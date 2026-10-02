"""Bounded native vocabulary-page scan; optional cached pinned Qwen source."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
from pllm.metrics import PrivatePageLookupProbe


def pinned_boundary():
    from huggingface_hub import hf_hub_download
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    from probe_rank_interface import _MODEL_ID, _REVISION

    config = Path(
        hf_hub_download(_MODEL_ID, "config.json", revision=_REVISION, local_files_only=True)
    )
    manifest = load_hf_directory(config.parent, model_id=_MODEL_ID)
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(manifest))
    head = engine._model(_MODEL_ID).stages["lm_head"]
    weights = head.weight.values
    scales = head.weight.scales.reshape(-1)
    metadata = {
        "model": _MODEL_ID,
        "revision": _REVISION,
        "head_weight_digest": head.weight_digest,
        "weight_bits": 8,
        "activation_bits": 8,
    }
    return engine, weights, scales, metadata


def packed_rows(weights, scales):
    records, width = weights.shape
    out = np.empty((records, width + 4), dtype=np.uint8)
    out[:, :width] = weights.view(np.uint8)
    out[:, width:] = scales.astype("<f4").view(np.uint8).reshape(records, 4)
    return out.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pinned:
        owner, weights, scales, source = pinned_boundary()
        table = packed_rows(weights, scales)
        records, record_bytes = weights.shape[0], weights.shape[1] + 4
    else:
        owner = None
        source = {"fixture": "bounded-public-random"}
        records, record_bytes, table = 1024, 132, None
    report = {
        "schema": "pllm.private_page_boundary_benchmark.v1",
        "source": source,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    for pages in (1, 8, 32):
        result = PrivatePageLookupProbe(records, record_bytes, pages, 3).run(table)
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            f"page_rows={pages}: {result['body_bytes_per_query']} B; "
            f"client={result['client_cpu_seconds_per_query_median']:.6f}s; "
            f"workers={sum(result['worker_cpu_seconds_per_query_median']):.4f}s",
            flush=True,
        )
    del owner


if __name__ == "__main__":
    main()
