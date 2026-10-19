"""Installed SDK gate. Run ONLY after coordinated wheel rebuild; never skip stale wheels."""
import json

import pytest

from pllm import OpenAI
from pllm.runtime.replicated_protocol import execution_binding
from pllm.runtime.replicated_source import admit_source
from test_replicated_sdk import experiment_fixture, active_sessions


@pytest.mark.parametrize("transport", ["http", "streams"])
@pytest.mark.parametrize("checked", [False, True])
def test_installed_trie_full_sdk_generated_final_tokens_and_retirement(tmp_path, transport, checked):
    base = experiment_fixture(tmp_path, weights="public", active=True)
    experiment = base.with_params(pipeline__components__linear__lookup="trie",
                                  pipeline__components__linear__peer_transport=transport)
    if checked:
        experiment = experiment.with_params(
            pipeline__components__linear__range_policy="checked",
            pipeline__components__linear__max_abs_activation=256,
            pipeline__components__linear__selection="tiled",
        )
    expected = execution_binding(experiment)
    source = admit_source(experiment)
    assert source.contract["execution"] == expected
    owner, manifest = source.publisher(experiment, (3, 4), 2, lambda: False)
    try:
        assert json.loads(manifest)["execution"] == expected
        assert bool(json.loads(manifest).get("norms")) is checked
    finally:
        owner.close()
    with pytest.raises(ValueError, match="retired"):
        owner.next_chunk()
    captured = []
    with OpenAI(experiment=experiment, timeout=300) as sdk:
        sdk._core._result_hook = captured.append
        for prompt, tokens in [("hello hello", (4, 4)), ("world hello", (1, 1))]:
            response = sdk.responses.create(input=prompt, max_output_tokens=2)
            assert response.usage.output_tokens == 2
            assert captured[-1].selected == tokens
            assert active_sessions(sdk._core) == [0, 0, 0]
        audit = sdk._core.audit.to_dict()
        assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
