"""Bounded, explicitly scoped network-cost probes for uncomposed methods."""

from __future__ import annotations

import time
import secrets
import sys
from dataclasses import dataclass
from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan


@dataclass(frozen=True, slots=True)
class ResidentMlpCostProbe:
    """Project an optimistic exact two-party MLP lower bound from a compiled plan.

    This is a research admission gate, not a runnable decoder or a privacy proof.
    Even a passing MLP bound cannot activate the share-resident topology.
    """

    fixed_scale_bits: int
    maximum_material_bytes_per_party: int
    maximum_online_all_link_body_bytes: int

    def __post_init__(self) -> None:
        if type(self.fixed_scale_bits) is not int or not 1 <= self.fixed_scale_bits <= 10:
            raise ValueError("reference fixed scale must be between 1 and 10 bits")
        if (
            type(self.maximum_material_bytes_per_party) is not int
            or not 0 < self.maximum_material_bytes_per_party <= 1 << 40
            or type(self.maximum_online_all_link_body_bytes) is not int
            or not 0 < self.maximum_online_all_link_body_bytes <= 1 << 40
        ):
            raise ValueError("bounded material and online body budgets are required")

    def run(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]:
        from pllm.runtime.shared_resources import resident_mlp_resource_gate

        return resident_mlp_resource_gate(
            plan, composition,
            response_new_tokens=response_new_tokens,
            fixed_scale_bits=self.fixed_scale_bits,
            maximum_material_bytes_per_party=self.maximum_material_bytes_per_party,
            maximum_online_all_link_body_bytes=self.maximum_online_all_link_body_bytes,
        )


@dataclass(frozen=True, slots=True)
class ResidentFusedGateCostProbe:
    """Cost-veto dense one-use fused gate from exact semantic layer shapes.

    Only a bounded test-local reference executes; this is not compact FSS or
    an executable whole-decoder component.
    """

    domain_bits: int
    maximum_material_bytes_per_party: int
    maximum_online_all_link_body_bytes: int
    maximum_online_body_bytes_per_layer: int

    def __post_init__(self) -> None:
        if type(self.domain_bits) is not int or self.domain_bits not in (4, 8):
            raise ValueError("fused reference requires Q3 or Q7 input bits")
        for limit in (
            self.maximum_material_bytes_per_party,
            self.maximum_online_all_link_body_bytes,
            self.maximum_online_body_bytes_per_layer,
        ):
            if type(limit) is not int or not 0 < limit <= 1 << 40:
                raise ValueError("fused gate limits must be bounded positive integers")

    def run(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]:
        from pllm.runtime.shared_gate_resources import resident_fused_gate_resource_gate

        return resident_fused_gate_resource_gate(
            plan, composition,
            response_new_tokens=response_new_tokens,
            domain_bits=self.domain_bits,
            maximum_material_bytes_per_party=self.maximum_material_bytes_per_party,
            maximum_online_all_link_body_bytes=self.maximum_online_all_link_body_bytes,
            maximum_online_body_bytes_per_layer=self.maximum_online_body_bytes_per_layer,
        )


@dataclass(frozen=True, slots=True)
class ResidentQuadraticGateCostProbe:
    """Cost-gate only Q7 quadratic gated products with a common masked source."""

    maximum_material_bytes_per_party: int
    maximum_online_all_link_body_bytes: int
    maximum_online_body_bytes_per_layer: int

    def __post_init__(self) -> None:
        for limit in (
            self.maximum_material_bytes_per_party,
            self.maximum_online_all_link_body_bytes,
            self.maximum_online_body_bytes_per_layer,
        ):
            if type(limit) is not int or not 0 < limit <= 1 << 40:
                raise ValueError("quadratic gate limits must be bounded positive integers")

    def run(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]:
        from pllm.runtime.shared_gate_resources import resident_quadratic_gate_resource_gate

        return resident_quadratic_gate_resource_gate(
            plan, composition,
            response_new_tokens=response_new_tokens,
            maximum_material_bytes_per_party=self.maximum_material_bytes_per_party,
            maximum_online_all_link_body_bytes=self.maximum_online_all_link_body_bytes,
            maximum_online_body_bytes_per_layer=self.maximum_online_body_bytes_per_layer,
        )

    def run_two_source_layer_bound(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]:
        """Also count an independent attention-query hidden-source opening."""
        from pllm.runtime.shared_gate_resources import resident_quadratic_layer_resource_gate

        return resident_quadratic_layer_resource_gate(
            plan, composition,
            response_new_tokens=response_new_tokens,
            maximum_material_bytes_per_party=self.maximum_material_bytes_per_party,
            maximum_online_all_link_body_bytes=self.maximum_online_all_link_body_bytes,
            maximum_online_body_bytes_per_layer=self.maximum_online_body_bytes_per_layer,
        )


@dataclass(frozen=True, slots=True)
class LatentResponseCostProbe:
    """Cost-gate non-streaming private-feedback decoder research.

    It never admits a serving composition; selection and feedback remain
    client-local in the currently compiled schedule.
    """

    maximum_online_all_link_body_bytes: int
    maximum_total_all_link_body_bytes: int
    maximum_material_bytes_per_party: int

    def __post_init__(self) -> None:
        for limit in (
            self.maximum_online_all_link_body_bytes,
            self.maximum_total_all_link_body_bytes,
            self.maximum_material_bytes_per_party,
        ):
            if type(limit) is not int or not 0 < limit <= 1 << 40:
                raise ValueError("latent response budgets must be bounded positive integers")

    def run(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]:
        from pllm.runtime.latent_response_cost import latent_response_cost_probe

        return latent_response_cost_probe(
            plan,
            composition,
            response_new_tokens=response_new_tokens,
            maximum_online_all_link_body_bytes=self.maximum_online_all_link_body_bytes,
            maximum_total_all_link_body_bytes=self.maximum_total_all_link_body_bytes,
            maximum_material_bytes_per_party=self.maximum_material_bytes_per_party,
        )


@dataclass(frozen=True, slots=True)
class EncryptedLinearCostProbe:
    """Measure an exact BFV public-weight stage with client-owned secret key.

    No prompt, input values, ciphertext, or secret key enter the returned report.
    This probes one stage, not encrypted nonlinearities or whole-layer execution.
    """

    input_width: int
    output_width: int
    rows: int = 1
    modulus: int = 65_537

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("input_width", self.input_width, 256),
            ("output_width", self.output_width, 256),
            ("rows", self.rows, 32),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if self.modulus != 65_537 or type(self.modulus) is not int:
            raise ValueError("the bounded reference uses only the checked 65537 BFV modulus")
        if self.input_width * self.output_width * self.rows > 1 << 20:
            raise ValueError("encrypted probe exceeds the bounded work product")

    def run(self) -> dict[str, Any]:
        import numpy as np

        from pllm.runtime.tiled_bfv import TiledBFVClient, TiledBFVServer

        # Fixed public fixture shapes and values. Encryption uses fresh SEAL randomness.
        rng = np.random.default_rng(3509)
        weight = rng.integers(
            -7, 8, size=(self.output_width, self.input_width), dtype=np.int8,
        )
        inputs = rng.integers(
            -7, 8, size=(self.rows, self.input_width), dtype=np.int64,
        )
        started = time.process_time_ns()
        client = TiledBFVClient(self.input_width, self.output_width, plain_modulus=self.modulus)
        server = TiledBFVServer(client.public_context, weight)
        setup_cpu_ns = time.process_time_ns() - started
        started = time.process_time_ns()
        requests = client.encrypt_many(inputs)
        encryption_cpu_ns = time.process_time_ns() - started
        started = time.process_time_ns()
        responses = server.evaluate_many(requests)
        evaluation_cpu_ns = time.process_time_ns() - started
        started = time.process_time_ns()
        actual = client.decrypt_many(responses, client.group_sizes(self.rows))
        decryption_cpu_ns = time.process_time_ns() - started
        expected = (inputs @ weight.astype(np.int64).T) % self.modulus
        if not np.array_equal(actual, expected):
            raise RuntimeError("encrypted stage disagrees with exact modular matrix product")
        input_body_bytes = sum(map(len, requests))
        output_body_bytes = sum(map(len, responses))
        return {
            "schema": "pllm.encrypted_linear_cost_probe.v1",
            "scope": "one bounded public-weight BFV linear stage, local roles, encrypted input and output",
            "backend": "tiled-bfv",
            "input_width": self.input_width,
            "output_width": self.output_width,
            "rows": self.rows,
            "modulus": self.modulus,
            "ciphertext_bodies": len(requests) + len(responses),
            "public_context_body_bytes": len(client.public_context),
            "client_to_provider_body_bytes": input_body_bytes,
            "provider_to_client_body_bytes": output_body_bytes,
            "online_all_link_body_bytes": input_body_bytes + output_body_bytes,
            "setup_cpu_seconds": setup_cpu_ns / 1e9,
            "encryption_cpu_seconds": encryption_cpu_ns / 1e9,
            "evaluation_cpu_seconds": evaluation_cpu_ns / 1e9,
            "decryption_cpu_seconds": decryption_cpu_ns / 1e9,
            "exact_modular_parity": True,
            "full_wire_measured": False,
            "whole_decoder_executable": False,
        }


@dataclass(frozen=True, slots=True)
class EncryptedQuadraticShareCostProbe:
    """Measure actual one-use, two-party BFV square-to-additive-shares islands.

    Exact F_65537 polynomial only. No Qwen SiLU, protected transport or
    compiler-admitted decoder is inferred from this bounded local reference.
    """

    width: int = 32
    rows: int = 1
    islands: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.width) is not int or not 1 <= self.width <= 64
            or type(self.rows) is not int or not 1 <= self.rows <= 64
            or self.width * self.rows > 4096
            or type(self.islands) is not int or not 1 <= self.islands <= 32
        ):
            raise ValueError("bounded BFV islands require 1..64 rows/width and 1..32 invocations")

    def run(self) -> dict[str, Any]:
        import numpy as np

        from pllm.runtime.he_quadratic_share import (
            HEQuadraticShareClient, HEQuadraticShareEvaluator,
        )

        def peak_rss_bytes() -> int | None:
            try:
                import resource
            except ImportError:  # Windows has no resource module.
                return None
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(value if sys.platform == "darwin" else value * 1024)

        initial_peak_rss = peak_rss_bytes()
        setup = time.process_time_ns()
        client = HEQuadraticShareClient(self.rows, self.width)
        evaluator = HEQuadraticShareEvaluator(client.public_context, self.rows, self.width)
        setup_cpu = (time.process_time_ns() - setup) / 1e9
        rng = np.random.default_rng(3509)
        upload = download = 0
        encryption_cpu = evaluation_cpu = decryption_cpu = 0.0
        shape = (self.rows, self.width)
        for _ in range(self.islands):
            clear = rng.integers(-7, 8, size=shape, dtype=np.int64)
            share_a = np.asarray(
                [secrets.randbelow(65_537) for _ in range(self.rows * self.width)], dtype=np.int64,
            ).reshape(shape)
            share_b = (clear - share_a) % 65_537
            started = time.process_time_ns()
            request = client.issue(share_a)
            encryption_cpu += (time.process_time_ns() - started) / 1e9
            upload += len(request)
            started = time.process_time_ns()
            evaluated = evaluator.evaluate(request, share_b)
            evaluation_cpu += (time.process_time_ns() - started) / 1e9
            download += len(evaluated.response)
            started = time.process_time_ns()
            output_a = client.finish(evaluated.response)
            decryption_cpu += (time.process_time_ns() - started) / 1e9
            expected = (clear * clear + 3 * clear + 7) % 65_537
            if not np.array_equal((output_a + evaluated.worker_b_share) % 65_537, expected):
                raise RuntimeError("BFV-to-shares polynomial disagrees with exact field arithmetic")
        return {
            "schema": "pllm.encrypted_quadratic_share_cost_probe.v1",
            "scope": "bounded in-process two-party quadratic F_65537 reference; no model quality or distributed privacy claim",
            "backend": "tenseal-bfv",
            "function": "x^2+3x+7 mod 65537",
            "rows": self.rows,
            "width": self.width,
            "islands": self.islands,
            "public_context_body_bytes": len(client.public_context),
            "worker_a_to_b_body_bytes": upload,
            "worker_b_to_a_body_bytes": download,
            "online_all_link_body_bytes": upload + download,
            "covered_cold_body_bytes": upload + download + len(client.public_context),
            "setup_cpu_seconds": setup_cpu,
            "encryption_cpu_seconds": encryption_cpu,
            "evaluation_cpu_seconds": evaluation_cpu,
            "decryption_cpu_seconds": decryption_cpu,
            "process_peak_rss_before_bytes": initial_peak_rss,
            "process_peak_rss_after_bytes": peak_rss_bytes(),
            "exact_modular_parity": True,
            "public_evaluator_has_secret_key": False,
            "whole_decoder_executable": False,
            "full_wire_measured": False,
        }
