"""Measured fresh-client 1/10/100-request horizons with shared public object cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.preparation import PreparedInventory
from pllm.kernels import Cpu
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles
from pllm.runtime.prepared_accounting import material_accounting
from pllm.runtime.servers import build_roles
from pllm.state import ClientPrefixReuse


def run(*, tiny=False, horizons=(1, 10), checkpoint=None):
    records = []
    output_control = None
    with tempfile.TemporaryDirectory(prefix="pllm-horizons-") as temporary:
        if tiny:
            from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
            root = create_tiny_llama_checkpoint(Path(temporary) / "model", hidden_size=32,
                intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, seed=812)
            source = Model.path(str(root), model_id="horizon-tiny")
        else:
            source = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")
        for refill in ("idle", "on-demand"):
            cache = Path(temporary) / refill / "public-cache"
            for horizon in horizons:
                experiment = Experiment(f"horizon-{refill}-{horizon}", MaskedLinearCpu(source,
                    kernels=Cpu(threads=1),
                    inventory=PreparedInventory(refill=refill),
                    delivery=ClientBundleTransport("artifacts"),
                    quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
                    placement=ClientLinearRoles(["qkv_projection", "attention_output"]),
                    cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=128)),
                    Deployment.local(root=str(Path(temporary) / "roles")),
                    ExecutionBudget(horizon, 128, 2))
                started = time.perf_counter()
                with build_roles(experiment) as topology:
                    with topology.client(bundle_cache_dir=cache) as client:
                        for request_index in range(horizon):
                            response = client.responses.create(input="Describe a tree briefly.", max_output_tokens=2, temperature=0)
                            digest = hashlib.sha256(response.output_text.encode()).hexdigest()
                            cohort = (digest, response.usage.input_tokens, response.usage.output_tokens)
                            if output_control is None:
                                output_control = cohort
                            assert cohort == output_control
                            # Explicit idle preparation is measured in horizon wall/CPU;
                            # do not race it with the next online request.
                            for state in client._core._transformer_states.values():
                                if state.refill_future is not None:
                                    state.refill_future.result(timeout=300)
                            if (request_index + 1) % 10 == 0:
                                print(f"{refill}: {request_index + 1}/{horizon} requests complete", flush=True)
                        audit_object = client.privacy_audit
                    audit = audit_object.to_dict()
                    ledger = material_accounting(audit)
                    assert ledger["conserved"] and ledger["remaining_stage_rows"] == 0, ledger
                    assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
                records.append({"refill": refill, "requests": horizon, "configuration_digest": experiment.configuration_digest(),
                                "seconds": time.perf_counter() - started, "audit": audit,
                                "material": ledger, "output_digest": cohort[0], "tokens_per_request": list(cohort[1:])})
                print(f"{refill}: {horizon} requests; {ledger['counts']}", flush=True)
                if checkpoint is not None:
                    checkpoint({"schema": "pllm.prepared_horizon_probe.v1", "complete": False,
                                "tiny": tiny, "records": records})
    return {"schema": "pllm.prepared_horizon_probe.v1", "complete": True, "tiny": tiny, "records": records,
            "scope": "measured SDK horizons; shared public artifact cache across fresh clients within each policy; one-use material is not cached",
            "limitations": ["co-located roles", "application bodies, not full wire", "same-prompt workload", "no independent model-quality or compute-cap admission"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 10],
                        help="request counts; 100-request sweeps require explicit opt-in")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not 1 <= value <= 100 for value in args.horizons):
        parser.error("horizons must be in [1,100]")
    def checkpoint(result):
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint(run(tiny=args.tiny, horizons=tuple(args.horizons), checkpoint=checkpoint))
