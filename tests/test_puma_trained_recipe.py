"""Source recipe regression definitions; no checkpoint admission or execution.

Pure cases never resolve a runtime. The separately guarded ``rust`` case checks
native configuration resolution only, not trained responses or capacity.
"""

import json

import pytest

from pllm import ConfigurationError, ExecutionBudget, Experiment, Model
from pllm.profiles import PumaThreePartyCpu
from pllm.protocols import PumaThreePartyDecoder

from examples.papers.common import BUDGET as PAPER_BUDGET, MODEL
from examples.papers.puma import experiment as original_experiment
from examples.papers.puma_trained import BUDGET, experiment


class TestPureRecipe:
    @pytest.fixture(autouse=True)
    def forbid_runtime_resolution(self, monkeypatch):
        def forbidden(*args, **kwargs):
            pytest.fail("pure recipe checks must not resolve a runtime")

        monkeypatch.setattr(Experiment, "resolve", forbidden)
        monkeypatch.setattr("pllm.profiles.resolve_runtime_composition", forbidden)

    def test_exact_pin_policy_and_budget(self):
        candidate = experiment()
        assert candidate.name == "puma-trained-candidate"
        assert candidate.pipeline.model is MODEL
        assert MODEL.to_spec() == {
            "source": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "local_files_only": True,
        }
        assert candidate.budget is BUDGET
        assert BUDGET.to_spec() == {
            "requests": 1, "max_input_tokens": 32, "max_new_tokens": 2,
        }
        assert isinstance(candidate.pipeline, PumaThreePartyCpu)
        assert candidate.pipeline.linear.get_params() == {
            "weights": "public",
            "range_policy": "checked",
            "max_abs_activation": 256,
            "lookup": "per_token",
            "selection": "expanded",
            "peer_transport": "http",
            "storage": "compact",
        }
        assert candidate.deployment.to_spec() == {
            "kind": "local", "root": ".pllm/papers/puma-trained",
        }

    def test_canonical_roundtrip_without_resolution(self):
        candidate = experiment()
        restored = Experiment.from_spec(json.loads(candidate.canonical_bytes()))
        assert restored.canonical_bytes() == candidate.canonical_bytes()
        assert restored.configuration_digest() == candidate.configuration_digest()
        assert restored.pipeline.components == candidate.pipeline.components

    @pytest.mark.parametrize("override", ["model", "budget", "both"])
    def test_overrides_are_not_qualified_trained_candidates(self, override):
        model = Model.tiny() if override in {"model", "both"} else MODEL
        budget = PAPER_BUDGET if override in {"budget", "both"} else BUDGET
        configured = experiment(model=model, budget=budget)
        assert configured.name == "puma-override"
        assert configured.pipeline.model is model
        assert configured.budget is budget
        assert configured.pipeline.linear == experiment().pipeline.linear

    def test_original_factory_keeps_existing_defaults(self):
        original = original_experiment()
        assert original.name == "puma"
        assert original.pipeline.model is MODEL
        assert original.budget is PAPER_BUDGET
        assert original.budget == ExecutionBudget(
            requests=32, max_input_tokens=64, max_new_tokens=8
        )
        assert original.pipeline.linear == PumaThreePartyDecoder()
        assert original.pipeline.linear.storage == "expanded"

    @pytest.mark.parametrize("change", [
        {"weights": "private"},
        {"range_policy": "static"},
        {"lookup": "tiled"},
        {"lookup": "trie"},
        {"selection": "tiled"},
        {"peer_transport": "streams"},
    ])
    def test_existing_validation_refuses_unsupported_compact_combinations(self, change):
        policy = {"weights": "public", "range_policy": "checked", "storage": "compact"}
        with pytest.raises(ConfigurationError, match="compact PUMA storage requires"):
            PumaThreePartyDecoder(**(policy | change))

    @pytest.mark.parametrize("bound", [0, 1025, True])
    def test_existing_validation_refuses_invalid_checked_bounds(self, bound):
        with pytest.raises(ConfigurationError, match="invalid PUMA public execution policy"):
            PumaThreePartyDecoder(
                weights="public", range_policy="checked", storage="compact",
                max_abs_activation=bound,
            )


@pytest.mark.rust
def test_native_configuration_resolution_only():
    native = pytest.importorskip("pllm._native")
    from pllm.profiles import resolve_runtime_composition

    candidate = experiment()
    resolved = native.resolve_experiment(candidate.canonical_bytes())
    assert resolved.canonical_composition == candidate.pipeline.canonical_bytes()
    assert resolved.composition_digest == candidate.pipeline.digest()
    assert resolved.configuration_digest == candidate.configuration_digest()
    options = resolve_runtime_composition(candidate.pipeline)
    assert options is not None
    assert options.runtime_kind == "replicated"
    assert options.numeric_profile == PumaThreePartyDecoder.CHECKED_NUMERIC_PROFILE
    assert options.generation_policy == "fixed_length_greedy"
