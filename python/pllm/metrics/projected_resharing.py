"""Bounded three-party product/projection degree-reduction diagnostic."""
from dataclasses import dataclass
import hashlib
import time


@dataclass(frozen=True, slots=True)
class ProjectedResharingProbe:
    def run(self, weights=None) -> dict:
        import numpy as np
        from pllm import _native
        p = 2_013_265_921
        rng = np.random.default_rng(624091)
        w = rng.integers(-31, 32, (16, 64), dtype=np.int8) if weights is None else np.asarray(weights)
        if w.dtype != np.int8 or w.ndim != 2 or not 1 <= w.shape[1] <= 8192 or not 1 <= w.shape[0] <= 4096 or w.size > 32 << 20:
            raise ValueError("projected resharing requires bounded signed-i8 public weights")
        a, b = (rng.integers(0, p, w.shape[1], dtype=np.uint32) for _ in range(2))
        # Independent oracle, excluded from the native CPU window.
        products = (a.astype(np.uint64) * b.astype(np.uint64) % p).astype(np.int64)
        expected = (w.astype(np.int64) @ products) % p
        rows = []
        for before in (False, True):
            start = time.process_time_ns()
            output, peer, distribution = _native.projected_reshare_reference(
                w.tobytes(), a.astype("<u4").tobytes(), b.astype("<u4").tobytes(), before)
            elapsed = (time.process_time_ns() - start) / 1e9
            if not np.array_equal(np.frombuffer(output, "<u4"), expected):
                raise RuntimeError("degree reduction differs from independent field oracle")
            rows.append({"project_before_resharing": before, "peer_frame_bytes": peer,
                         "fixture_input_share_bytes": distribution,
                         "all_role_native_cpu_seconds": elapsed, "exact_field_outputs": True})
        return {"schema": "pllm.projected_resharing_probe.v1", "configurations": rows,
                "weight_digest": hashlib.sha256(w.tobytes()).hexdigest(), "shape": list(w.shape),
                "field_modulus": p, "peer_reduction_factor": rows[0]["peer_frame_bytes"] / rows[1]["peer_frame_bytes"],
                "scope": "local all-share fixture; one-use party-local degree reduction; not decoder execution",
                "client_resident_weight_bytes_required": 0, "fresh_dealer_material_bytes": 0,
                "resident_state_excludes_fixture_distribution": True, "whole_decoder_executable": False,
                "limitations": ["Three semi-honest parties, at most one colludes; not the existing two-party topology",
                    "Each party computes one public projection; 3x clear projection MACs",
                    "No exact fixed-point truncation, SiLU, attention, normalization, or private feedback contract",
                    "No authenticated transport, independent security review or peak client memory measurement"]}

    def project(self, plan, composition, *, response_new_tokens: int) -> dict:
        from pllm.runtime.semantic_stages import scheduled_stage_specs
        from pllm.runtime.region_contract_cost import compiler_region_contract_cost
        base = compiler_region_contract_cost(plan, composition, response_new_tokens=response_new_tokens,
            maximum_online_all_link_body_bytes=1 << 40, maximum_total_all_link_body_bytes=1 << 40)
        rows = base["executed_rows"]
        stages = [s for s in scheduled_stage_specs(plan, composition) if s.role == "mlp_down"]
        if not stages:
            raise ValueError("no compiled gated-output projections")
        return {"plan_digest": plan.digest, "schedule_digest": base["schedule_digest"], "executed_rows": rows,
                "ordinary_peer_frame_bytes": rows * sum(6 * (46 + s.in_features * 4) for s in stages),
                "projected_peer_frame_bytes": rows * sum(6 * (46 + s.out_features * 4) for s in stages),
                "fresh_dealer_material_bytes": 0, "complete_online_body_bytes": None,
                "scope": "isolated field products and public down-projection; resident input shares assumed",
                "missing": ["numeric conversion and rescaling", "SiLU", "normalization", "attention", "private token feedback"]}
