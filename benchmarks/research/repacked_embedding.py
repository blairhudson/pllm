"""Fresh-process FastQuery output-packing comparison with full party-local conversion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "benchmarks/research/repacked_embedding.py", "python/pllm/metrics/repacked_embedding.py",
    "python/pllm/metrics/packed_embedding.py", "python/pllm/runtime/coefficient_repacking.py",
    "python/pllm/protocols/__init__.py", "crates/pllm-core/src/coefficient_repacking.rs",
    "crates/pllm-core/src/packed_embedding.rs", "crates/pllm-garble/src/packed_embedding.rs",
    "crates/pllm-garble/src/shared_circuit.rs", "crates/pllm-garble/src/boolean.rs",
    "crates/pllm-python/src/packed_embedding.rs",
)


def projection(report):
    from pllm.metrics.packed_embedding import _FRAME_BYTES, _WIDTHS
    params = report["configuration"]["component"]["params"]
    groups = math.ceil(896 / len(_WIDTHS[params["precision"]]))
    measured_groups = math.ceil(report["configuration"]["lanes"] / len(_WIDTHS[params["precision"]]))
    operations = {name:(a,b) for name,a,b in report["resources"]["key_payload_bytes_by_operation"]}
    conversion = operations["packed-share-conversion"]
    if any(k % measured_groups for k in conversion):
        raise ValueError("nonintegral conversion projection")
    phases = report["samples"][0]["peer_frame_bytes_by_phase"]
    peer_per_group = sum(sum(p) for p in phases[1:]) // measured_groups
    repacked = params.get("repacking", "none") != "none"
    # Full-width execution is NOT attempted: symbolic blockwise vocabulary sums,
    # followed by groups of <=256 extracted coefficients/interleave inputs.
    replies = math.ceil(groups/256) if repacked else groups
    query_bytes = 70 * math.ceil(151936/4096) * _FRAME_BYTES
    reply_bytes = 70 * replies * _FRAME_BYTES
    conversion_bytes = 70 * groups * peer_per_group
    setup = operations["bfv-context-secret-and-public"][1] + operations["repacking-galois-key-copies"][1]
    return {"precision":params["precision"], "repacking":params.get("repacking", "none"),
        "vocabulary":151936, "hidden_width":896, "lookup_rows":70,
        "he_query_frame_bytes":query_bytes, "he_reply_frame_bytes":reply_bytes,
        "conversion_peer_frame_bytes":conversion_bytes,
        "combined_online_frame_bytes":query_bytes+reply_bytes+conversion_bytes,
        "conversion_key_bytes_by_party":[70*groups*k//measured_groups for k in conversion],
        "setup_delivery_bytes_one_context":setup, "fresh_context_per_row_setup_bytes":70*setup,
        "both_party_context_and_galois_payload_bytes":[operations["bfv-context-secret-and-public"][i]
            +operations["repacking-galois-key-copies"][i] for i in (0,1)],
        "provider_signed_code_bytes":151936*896,
        "provider_packed_coefficient_bytes":math.ceil(151936/4096)*4096*groups*2,
        "embedding_only_released_client_weight_bytes":0,
        "scope":"unadmitted blockwise Qwen 39+32 lookup geometry; no full-width noise/HE execution or decoder transport",
        "full_width_executed":False, "full_wire_bytes":None}


def cost():
    reports=[]
    for rows in (256,4096):
        for precision in ("int8", "int4", "mixed_4_3_3"):
            for mode in ("none", "interleave", "pack_lwes"):
                run=subprocess.run([sys.executable,"-m","benchmarks.research.repacked_embedding",
                    "--worker",str(rows),precision,mode],check=True,capture_output=True,text=True,timeout=300,
                    env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"})
                reports.append(json.loads(run.stdout))
    for rows in (256,4096):
        for precision in ("int8", "int4", "mixed_4_3_3"):
            matched=[r for r in reports if r["configuration"]["component"]["params"]["vocabulary_size"]==rows
                     and r["configuration"]["component"]["params"]["precision"]==precision]
            if len({r["numeric_contract_digest"] for r in matched})!=1:
                raise ValueError("packing changed the numeric contract")
    return {"schema":"pllm.repacked_embedding_cost.v1", "fresh_processes":len(reports),
        "checked_outputs":sum(r["quality"]["checked_outputs"] for r in reports),"reports":reports,
        "projections":[projection(r) for r in reports if r["configuration"]["component"]["params"]["vocabulary_size"]==4096],
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
        "scope":"bounded encrypted output repacking and one-use party-local input/output conversion; local OT dealer",
        "decoder_executable":False, "full_wire_bytes":None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",nargs=3)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.worker:
        from pllm.metrics import benchmark_operator
        from pllm.protocols import PrivateEmbeddingLookup
        rows,precision,mode=args.worker
        result=benchmark_operator(PrivateEmbeddingLookup(vocabulary_size=int(rows),precision=precision,
            repacking=mode,conversion="party_local"),lanes=64,samples=3).to_dict()
        print(json.dumps(result,allow_nan=False))
        return
    result=cost()
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        parser.error("refusing to overwrite research evidence")
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"fresh_processes":result["fresh_processes"],"checked_outputs":result["checked_outputs"]}))


if __name__=="__main__":
    main()
