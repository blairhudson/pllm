from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.metrics import LatentResponseCostProbe
from pllm.modeling import ModelPlan
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.authenticated_mpc import AuthenticatedMPC, TrustedPreprocessor
from pllm.runtime.secure_selection import secure_argmax
from pllm.runtime.secure_transformer import (
    ClearKVCache,
    SecureDecoder,
    SecureDecoderConfig,
    SecureDecoderWeights,
    SecureKVCache,
)
from pllm.runtime.shared_resources import SharedResourceError

from test_shared_resources import CONFIG

_SHARED_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or str(
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
)


def _composition() -> MaskedLinearCpu:
    return MaskedLinearCpu(
        Model.hf(
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
        ),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )


def _probe(online: int, total: int) -> LatentResponseCostProbe:
    return LatentResponseCostProbe(
        maximum_online_all_link_body_bytes=online,
        maximum_total_all_link_body_bytes=total,
        maximum_material_bytes_per_party=256 << 20,
    )


def _run(new_tokens: int, online: int, total: int) -> dict:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=new_tokens)
    return _probe(online, total).run(
        plan,
        _composition(),
        response_new_tokens=new_tokens,
    )


@pytest.mark.parametrize(
    ("new_tokens", "baseline_online", "baseline_total", "narrow", "keys", "resident"),
    [
        (1, 62_248_704, 98_723_710, 5_870_592, 141_613_056, 10_063_872),
        (8, 73_831_744, 116_843_966, 6_924_288, 167_030_784, 11_870_208),
        (32, 113_545_024, 178_970_558, 10_536_960, 254_177_280, 18_063_360),
    ],
)
def test_qwen_compiled_response_cost_veto(
    new_tokens: int, baseline_online: int, baseline_total: int,
    narrow: int, keys: int, resident: int,
) -> None:
    report = _run(new_tokens, baseline_online // 10, baseline_total // 10)
    assert report["vocabulary_size"] == 151_936
    assert report["composition_digest"] == "bd91e1659773faabcbb998a350893cbbc6df313a830da7cb7bfb87463c8f0b36"
    assert report["remote_mlp_cut"]["minimum_online_input_output_body_bytes"] == narrow
    assert report["remote_mlp_cut"]["quadratic_gate_one_use_dealer_body_bytes_both_parties"] == keys
    assert report["remote_mlp_cut"]["within_online_budget"]
    assert not report["remote_mlp_cut"]["within_all_link_budget"]
    assert report["remote_mlp_cut"]["material_per_party_within_budget"]
    assert report["resident_two_source"]["minimum_online_opening_body_bytes"] == resident
    assert not report["resident_two_source"]["within_online_budget"]
    assert report["resident_two_source"]["material_per_party_within_budget"]
    assert report["resident_two_source"]["optimistic_online_with_existing_feedback_reference_bytes"] == (
        resident + (new_tokens - 1) * 82_653_184
    )
    reference = report["feedback_reference"]
    assert reference["optimistic_peer_only_opened_body_bytes_per_required_selection"] == 82_653_184
    assert reference["optimistic_peer_only_opened_body_bytes_for_response"] == (
        new_tokens - 1
    ) * 82_653_184
    assert reference["client_local_feedback_in_compiled_schedule"]
    assert reference["actual_argmax_reveals_index_to_client"]
    assert reference["lookup_requires_future_token_known_to_issuer"]
    assert reference["share_indexed_embedding_lookup_available"] is False
    assert reference["point_fss_comparison_maximum_domain"] < report["vocabulary_size"]
    assert math.isclose(
        reference["decimal_log10_exhaustive_branches_without_private_feedback"],
        (new_tokens - 1) * math.log10(151_936),
    )
    assert not report["whole_response_executable"]
    assert not report["whole_response_byte_admission"]


def test_reference_argmax_reveals_index_and_consumes_more_than_optimistic_bit_bound() -> None:
    preprocessor = TrustedPreprocessor(seed=972)
    mpc = AuthenticatedMPC(preprocessor)
    scores = np.asarray([[9, 4, 17, 2, 6, 3, 12, 1]], dtype=np.int64)
    selection = secure_argmax(
        mpc,
        preprocessor.share(scores),
        bit_mask=preprocessor.bit_mask(scores.shape),
    )
    np.testing.assert_array_equal(selection.indices, [2])
    assert mpc.stats.opened_values > scores.size * 17
    assert mpc.stats.uploaded_bytes + mpc.stats.downloaded_bytes > scores.size * 17 * 32
    assert mpc.stats.downloaded_bytes > 0  # openings went to the client
    assert mpc.stats.opened_values == 5_249
    assert mpc.stats.uploaded_bytes == 83_984
    assert mpc.stats.downloaded_bytes == 41_992


def test_bounded_autoregressive_reference_needs_client_visible_feedback() -> None:
    config = SecureDecoderConfig(
        vocabulary_size=8, hidden_size=8, head_count=2, head_size=4,
        intermediate_size=12, maximum_context=4,
    )
    weights = SecureDecoderWeights.random(config, seed=801)
    decoder = SecureDecoder(config, weights, TrustedPreprocessor(seed=802))
    secure_state, clear_state = SecureKVCache(), ClearKVCache()
    token = 2
    total_upload = total_download = 0
    for _ in range(3):
        protected = decoder.forward_token([token], cache=secure_state)
        expected, _ = decoder.clear_forward_token([token], cache=clear_state)
        np.testing.assert_array_equal(protected.token_ids, expected)
        # The next input is determined by the client-visible selection.
        token = int(protected.token_ids[0])
        total_upload += protected.uploaded_bytes
        total_download += protected.downloaded_bytes
    assert secure_state.length == clear_state.length == 3
    assert total_upload > 0 and total_download > 0


def test_latent_probe_fails_closed_on_forged_feedback_contract() -> None:
    original = lower_model(CONFIG, batch=1, max_input_tokens=2, max_new_tokens=2)
    composition = _composition()
    for mutate in (
        lambda document: document.update(token_feedback=False),
        lambda document: document["prefill"]["operations"][-3].update(output_shape=[1, 0]),
    ):
        document = copy.deepcopy(original.to_dict())
        mutate(document)
        forged = ModelPlan(json.dumps(document, ensure_ascii=False, sort_keys=True).encode())
        with pytest.raises(SharedResourceError):
            _probe(10_000_000, 20_000_000).run(
                forged, composition, response_new_tokens=2,
            )


@pytest.mark.parametrize("invalid", [0, -1, True, 1 << 41])
def test_latent_probe_rejects_invalid_budgets(invalid: int) -> None:
    with pytest.raises(ValueError):
        _probe(invalid, 20_000_000)


@pytest.mark.skipif(not os.getenv("PLLM_RUN_REAL_QWEN25"), reason="requires pinned cached Qwen source")
def test_real_pinned_qwen25_projections_match_retained_evidence() -> None:
    from huggingface_hub import hf_hub_download

    evidence = json.loads(
        (Path(__file__).parents[1] / "docs/evidence/latent-response-network-qwen25-2026-09-28.json")
        .read_text(encoding="utf-8")
    )
    config_path = hf_hub_download(
        "Qwen/Qwen2.5-0.5B-Instruct", "config.json",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        local_files_only=True,
        cache_dir=_SHARED_HUB_CACHE,
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for cohort in evidence["cohorts"]:
        outputs = cohort["output_tokens"]
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=outputs)
        baseline = cohort["prepared_control"]
        projected = _probe(
            baseline["tenfold_online_budget_bytes"], baseline["tenfold_covered_budget_bytes"],
        ).run(plan, _composition(), response_new_tokens=outputs)
        assert projected["plan_digest"] == cohort["official_plan_digest"]
        assert projected["schedule_digest"] == cohort["official_schedule_digest"]
        assert projected["composition_digest"] == evidence["source"]["pipeline_digest"]
        assert projected["remote_mlp_cut"]["minimum_online_input_output_body_bytes"] == (
            cohort["remote_mlp_cut"]["optimistic_online_narrow_boundary_bytes"]
        )
        assert projected["remote_mlp_cut"]["quadratic_gate_one_use_dealer_body_bytes_both_parties"] == (
            cohort["remote_mlp_cut"]["one_use_dealer_body_bytes_both_parties"]
        )
        assert projected["resident_two_source"]["minimum_online_opening_body_bytes"] == (
            cohort["resident_two_source"]["optimistic_online_opening_bytes_before_attention_math"]
        )
