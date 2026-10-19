"""Trie selector and strict public binding. No native execution or source loads."""
import copy
import json

import pytest

from pllm.configuration import Experiment
from pllm.protocols import PumaThreePartyDecoder
from pllm.profiles import PumaThreePartyCpu
from pllm.runtime.replicated_protocol import execution_binding, validate_execution_binding
from test_replicated_execution_policy import experiment_for


def test_trie_canonical_profile_roundtrip_keeps_five_field_execution():
    original = experiment_for(weights="public", lookup="trie")
    restored = Experiment.from_spec(json.loads(original.canonical_bytes()))
    assert restored.canonical_bytes() == original.canonical_bytes()
    assert dict(original.pipeline.components["linear"].params) == {"weights": "public", "lookup": "trie"}
    policy = execution_binding(original)
    assert policy == {"range_policy": "static", "max_abs_activation": 256,
                      "lookup": "trie", "selection": "expanded", "peer_transport": "http"}
    facade = PumaThreePartyCpu(original.pipeline.model, weights="public", lookup="trie")
    assert facade.canonical_bytes() == original.pipeline.canonical_bytes()
    for lookup in ("per_token", "tiled"):
        assert original.configuration_digest() != experiment_for(weights="public", lookup=lookup).configuration_digest()
    assert execution_binding(experiment_for(weights="public")) is None


@pytest.mark.parametrize("replacement", [None, True, 1, "tiled", "per_token", "TRIE", {}])
def test_trie_transport_rejects_selector_substitution(replacement):
    experiment = experiment_for(weights="public", lookup="trie")
    value = copy.deepcopy(execution_binding(experiment))
    value["lookup"] = replacement
    with pytest.raises(ValueError, match="execution binding"):
        validate_execution_binding(experiment, {"execution": value})


def test_trie_transport_rejects_missing_extra_fields_and_private_policy():
    experiment = experiment_for(weights="public", lookup="trie")
    policy = execution_binding(experiment)
    validate_execution_binding(experiment, {"execution": policy})
    for field in policy:
        missing = {k: v for k, v in policy.items() if k != field}
        with pytest.raises(ValueError):
            validate_execution_binding(experiment, {"execution": missing})
    with pytest.raises(ValueError):
        validate_execution_binding(experiment, {"execution": {**policy, "body": "tiled"}})
    with pytest.raises(ValueError):
        PumaThreePartyDecoder(lookup="trie")


def test_trie_discovery_exposes_only_new_lookup_choice():
    descriptor = PumaThreePartyDecoder.describe().to_dict()
    fields = descriptor["parameter_schema"]["properties"]
    assert fields["lookup"]["enum"] == ["tiled", "trie"]
    assert "body" not in fields and "predicate" not in fields
