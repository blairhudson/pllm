"""Public configuration/native metadata contracts; no checkpoint execution."""

import importlib
from importlib.abc import MetaPathFinder
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def quality(monkeypatch):
    class NoExecutionImports(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == name or fullname.startswith(name + ".") for name in (
                "pllm.runtime.loaders", "pllm.runtime.hf_hub",
                "torch", "transformers", "huggingface_hub",
            )):
                raise AssertionError(f"configuration test attempted execution import: {fullname}")
            return None

    monkeypatch.setattr(sys, "meta_path", [NoExecutionImports(), *sys.meta_path])
    monkeypatch.syspath_prepend(str(ROOT))
    # Keep real Experiment.resolve/native metadata while preventing source-loader
    # imports. The pinned remote identifier needs no filesystem/model resolution.
    source = ModuleType("pllm.model_loader")
    source.expected_model_id = lambda model: model.source
    source.resolve_model = Mock(side_effect=AssertionError("unexpected plaintext/source fallback"))
    monkeypatch.setitem(sys.modules, source.__name__, source)
    return importlib.import_module("examples.papers.gkm_quality")


@pytest.mark.parametrize("factory,sharing", [("experiment", "full"), ("seeded", "seeded")])
def test_factory_preserves_approved_gkm_components_and_source(quality, factory, sharing):
    from pllm import Experiment, Pipeline
    from examples.papers import gkm
    from examples.papers.common import BUDGET, MODEL
    from pllm.profiles import resolve_runtime_composition
    from pllm.protocols import OpenWeightVerifiableOutsourcing

    actual = getattr(quality, factory)()
    original = getattr(gkm, factory)()
    assert isinstance(actual, Experiment)
    assert isinstance(actual.pipeline, Pipeline)
    assert actual.pipeline == original.pipeline
    assert actual.deployment == original.deployment
    assert actual.budget == original.budget == BUDGET
    assert actual.pipeline.model == MODEL
    assert MODEL.to_spec() == {
        "source": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "local_files_only": True,
    }
    assert actual.budget.to_spec() == {
        "requests": 32, "max_input_tokens": 64, "max_new_tokens": 8,
    }
    assert set(actual.pipeline.components) == {"linear", "verification", "topology", "state", "kernels"}
    assert actual.pipeline.components["linear"].params == {
        "sharing": sharing, "generation": "fixed_length_greedy", "capacity": "decoder",
    }
    assert actual.pipeline.components["kernels"].params == {"threads": 1}
    options = resolve_runtime_composition(actual.pipeline)  # Pure recognition, not native resolution.
    assert options.client_runtime == "compiled_gkm_v1"
    assert options.numeric_profile == OpenWeightVerifiableOutsourcing.NUMERIC_PROFILE
    assert options.sharing == sharing
    assert options.generation_policy == "fixed_length_greedy"
    resolved = actual.resolve()  # Real native configuration admission, not issuance.
    assert resolved.client_runtime == "compiled_gkm_v1"
    assert resolved.configuration_digest == actual.configuration_digest()
    assert resolved.composition_digest == actual.pipeline.digest()
    assert resolved.canonical_composition == actual.pipeline.canonical_bytes()
    assert resolved.model == MODEL.source
    assert resolved.verification_component == actual.pipeline.components["verification"].component


def test_factories_round_trip_and_differ_only_in_name_and_sharing(quality):
    from pllm import Experiment

    candidates = [quality.experiment(), quality.seeded()]
    assert len({item.name for item in candidates}) == 2
    assert len({item.configuration_digest() for item in candidates}) == 2
    normalized = []
    for candidate in candidates:
        restored = Experiment.from_spec(json.loads(candidate.canonical_bytes()))
        assert restored.canonical_bytes() == candidate.canonical_bytes()
        assert restored.configuration_digest() == candidate.configuration_digest()
        spec = candidate.to_spec()
        spec.pop("name")
        spec["pipeline"]["components"]["linear"]["params"].pop("sharing")
        normalized.append(spec)
    assert normalized[0] == normalized[1]
    assert quality.experiment().canonical_bytes() == candidates[0].canonical_bytes()


def test_existing_public_workload_stays_within_cli_bounds(quality):
    prompts = json.loads((ROOT / quality.PROMPTS_FILE).read_text(encoding="utf-8"))
    assert len(prompts) == len(set(prompts)) == 12
    assert all(type(prompt) is str and prompt.strip() and len(prompt.encode()) <= 4096
               for prompt in prompts)
    candidate = quality.experiment()
    assert len(prompts) <= candidate.budget.requests
    assert quality.DECODE_STEPS == 3 < candidate.budget.max_new_tokens
    # Token counts require the real pinned tokenizer, not a whitespace estimate.
