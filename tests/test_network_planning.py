"""Slice A acceptance: strict offline decisions, native oracle, real selected SDK/benchmark."""

from __future__ import annotations

import copy
import itertools
import json
import math
import pickle
import time
from dataclasses import FrozenInstanceError, replace

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, OpenAI, _native
from pllm.compiler import plan
from pllm.deployment import (
    LinkObservation,
    NetworkError,
    NetworkSnapshot,
    NetworkSpec,
    PartyOffer,
    discover,
    open_execution,
)
from pllm.deployment.network import canonical, strict_load
from pllm.modeling import lower_model
from pllm.plan import PlanningResult
from pllm.profiles import (
    ClientOnlyCpu,
    ClientOnlyMetal,
    MaskedLinearCpu,
    TwoOnlineOffsetCpu,
    VerifiedMaskedLinearCpu,
)
from pllm.roles import ClientLinearRoles
from pllm.search import CandidateCostEvidence, PlanningPolicy, PlanningRequest

pytestmark = pytest.mark.rust

CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
    "max_position_embeddings": 256,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}
PROVIDER_CAPS = ("trusted_preparation", "masked_linear_provider", "public_linear_provider")


def offer(party, caps=PROVIDER_CAPS, **kwargs):
    return PartyOffer(
        party,
        kwargs.pop("operator_id", party),
        kwargs.pop("instance_epoch", "b" * 64),
        kwargs.pop("expires_at_ms", 1000),
        caps,
        kwargs.pop("max_memory_bytes", 64 << 20),
        kwargs.pop("max_weight_bytes", 64 << 20),
        kwargs.pop("max_sessions", 8),
        kwargs.pop("active_sessions", 0),
        kwargs.pop("allowed_models", ("*",)),
        kwargs.pop("allowed_compositions", ("*",)),
        **kwargs,
    )


def snapshot(*, fast="a", offers=None):
    offers = offers or (offer("client", ("trusted_client",)), offer("a"), offer("b"), offer("c"))
    links = tuple(
        LinkObservation(
            left.party_id,
            right.party_id,
            "scenario",
            0,
            1000,
            1_000_000
            if (right.party_id == fast or (left.party_id == fast and right.party_id == "client"))
            else 1_000,
            5,
        )
        for left in offers
        for right in offers
        if left != right
    )
    return NetworkSnapshot("dev", offers, links, "fixture", 0, 1000)


def request(profiles=(MaskedLinearCpu,), **policy_kwargs):
    model = Model.tiny()
    candidates = tuple(
        Experiment(
            profile.__name__,
            profile(model),
            Deployment.local(root="local://plan"),
            ExecutionBudget(1, 3, 2),
        )
        for profile in profiles
    )
    return PlanningRequest(
        lower_model(CONFIG, batch=1, max_input_tokens=3, max_new_tokens=2),
        candidates,
        PlanningPolicy(
            "client",
            100,
            policy_kwargs.pop("max_candidates", 8),
            policy_kwargs.pop("max_assignments", 1000),
            policy_kwargs.pop("objectives", ("link_transfer_ms",)),
            **policy_kwargs,
        ),
    )


def reasons(result):
    return "\n".join(item["reason"] for item in result.to_spec()["rejections"])


def assigned(result):
    return {item["role_id"]: item["party_id"] for item in result.native_placement["roles"]}


@pytest.mark.parametrize(
    "profile", [ClientOnlyCpu, MaskedLinearCpu, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu]
)
def test_all_installed_role_partitions_native_floors_and_replay(profile):
    req = request((profile,), objectives=("client_weight_bytes",))
    result = plan(req, snapshot=snapshot())
    assert result.status == "feasible" and result.exhaustive
    native = result.native_placement
    assert {item["role_id"] for item in native["roles"]} == {
        role.id for role in result.experiment.resolve().role_graph.roles
    }
    assert set(native["required_memory_bytes"]) == set(native["minimum_memory_bytes"])
    for role, floor in native["minimum_weight_bytes"].items():
        assert native["required_weight_bytes"][role] == 6 * floor
        assert native["required_memory_bytes"][role] >= native["minimum_memory_bytes"][role]
    assert assigned(result)["client"] == "client"
    assert native["snapshot_digest"] == result.snapshot.digest
    assert native["model_plan_digest"] == req.model_plan.digest
    assert (
        PlanningResult.from_spec(result.to_spec(), request=req, snapshot=result.snapshot).digest
        == result.digest
    )
    with pytest.raises(TypeError):
        native["minimum_weight_bytes"]["client"] = 0
    with pytest.raises(FrozenInstanceError):
        result.experiment = None


def test_client_budget_changes_selected_composition_and_link_budget_changes_party():
    req = request((ClientOnlyCpu, MaskedLinearCpu), objectives=("online_all_link_body_bytes",))
    loose = plan(req, snapshot=snapshot())
    assert loose.experiment.name == "ClientOnlyCpu"
    tight = plan(
        replace(req, policy=replace(req.policy, max_client_weight_bytes=1700)), snapshot=snapshot()
    )
    assert tight.experiment.name == "MaskedLinearCpu"
    assert tight.costs["client_weight_bytes"] <= 1700 < loose.costs["client_weight_bytes"]
    a = plan(request(), snapshot=snapshot(fast="a"))
    c = plan(request(), snapshot=snapshot(fast="c"))
    assert assigned(a)["inference"] == "a"
    assert assigned(c)["inference"] == "c"
    assert assigned(a) != assigned(c)


def test_small_exhaustive_native_oracle_with_hand_computed_stage_link_cost():
    req, snap = request(), snapshot()
    exp = req.candidates[0]
    requirements = json.loads(
        _native.network_placement_requirements(
            req.model_plan.canonical_bytes(), exp.pipeline.canonical_bytes()
        )
    )
    minimum = requirements["minimum_weight_bytes"]
    link = {
        (item.source_party_id, item.target_party_id): item.bytes_per_second for item in snap.links
    }
    oracle = []
    for inference, preparation in itertools.product(("a", "b", "c"), repeat=2):
        roles = {"client": "client", "inference": inference, "preparation": preparation}
        document = {
            "schema": "pllm.network_placement.v1",
            "snapshot_digest": snap.digest,
            "evaluated_at_ms": 100,
            "client_party_id": "client",
            "roles": [{"role_id": role, "party_id": party} for role, party in roles.items()],
            "parties": [
                item.native_spec() for item in snap.offers if item.party_id in set(roles.values())
            ],
            "required_weight_bytes": {role: floor * 6 for role, floor in minimum.items()},
            "required_memory_bytes": {
                role: floor * 6 + (1 << 20) + (768 if role == "client" else 768)
                for role, floor in minimum.items()
            },
        }
        try:
            _native.validate_network_placement(
                req.model_plan.canonical_bytes(),
                exp.pipeline.canonical_bytes(),
                canonical(document),
            )
        except ValueError:
            continue
        # Independent fixture geometry: QKV 8->16, output 8->8,
        # grouped gate/up 8->32, down 16->8; four evaluated rows, 4-byte arrays.
        score = math.fsum(
            (
                1000 * 640 / link["client", inference],
                1000 * 1024 / link[inference, "client"],
                1000 * 640 / link["client", preparation],
                1000 * 1024 / link[preparation, "client"],
                1000 * 1024 / link[preparation, inference],
            )
        )
        oracle.append((score, tuple(sorted(roles.items()))))
    result = plan(req, snapshot=snap)
    assert result.to_spec()["coverage"]["assignments_evaluated"] == 9
    assert result.to_spec()["coverage"]["feasible_assignments"] == len(oracle) == 6
    assert (result.costs["link_transfer_ms"], tuple(sorted(assigned(result).items()))) == min(
        oracle
    )
    assert result.costs["online_all_link_body_bytes"] == 1664
    assert result.costs["preprocessing_all_link_body_bytes"] == 2688
    assert result.costs["logical_total_linear_macs"] == 2304


def test_deterministic_snapshot_order_and_request_dedup():
    snap = snapshot()
    reordered = replace(
        snap, offers=tuple(reversed(snap.offers)), links=tuple(reversed(snap.links))
    )
    req = request()
    duplicate = replace(req, candidates=(req.candidates[0], req.candidates[0]))
    assert duplicate.digest == req.digest
    assert reordered.digest == snap.digest
    assert (
        plan(req, snapshot=snap).canonical_bytes()
        == plan(duplicate, snapshot=reordered).canonical_bytes()
    )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("require_verified_privacy", True, "UNRESOLVED_VERIFIED_PRIVACY"),
        ("max_client_cpu_ns", 100, "UNKNOWN_REQUIRED_COST"),
        ("max_client_peak_memory_bytes", 100, "UNKNOWN_REQUIRED_COST"),
        ("max_full_wire_bytes", 100, "UNKNOWN_REQUIRED_COST"),
        ("max_client_weight_bytes", 0, "POLICY_BOUND"),
        ("max_client_payload_bytes", 0, "POLICY_BOUND"),
        ("objectives", ("setup_ms",), "UNKNOWN_REQUIRED_COST"),
        ("objectives", ("full_wire_bytes",), "UNKNOWN_REQUIRED_COST"),
    ],
)
def test_unknown_and_hard_policy_fail_closed(field, value, reason):
    result = plan(request(**{field: value}), snapshot=snapshot())
    assert result.status == "infeasible" and result.exhaustive
    assert reason in reasons(result)


def test_unknown_links_not_zero_expired_unused_offers_ignored_and_separation_checked():
    snap = snapshot()
    result = plan(request(), snapshot=replace(snap, links=()))
    assert result.status == "infeasible" and "UNKNOWN_REQUIRED_COST" in reasons(result)
    fresh = snapshot(offers=(*snap.offers, offer("old", expires_at_ms=99)))
    decision = plan(request(), snapshot=fresh)
    assert decision.status == "feasible"
    assert "old" not in assigned(decision).values()
    assert "old" not in {item["party_id"] for item in decision.native_placement["parties"]}
    assert decision.native_placement["snapshot_digest"] == fresh.digest
    shared = snapshot(
        offers=(
            offer("client", ("trusted_client",)),
            offer("a", operator_id="shared"),
            offer("b", operator_id="shared"),
        )
    )
    rejected = plan(request(), snapshot=shared)
    assert rejected.status == "infeasible" and "separate declared operators" in reasons(rejected)


def test_observation_source_expiry_age_and_direction_are_authority():
    snap = snapshot()
    for changes in (
        {"expires_at_ms": 100},
        {"observed_at_ms": 101},
        {"observed_at_ms": 0, "expires_at_ms": 1000},
    ):
        links = tuple(replace(item, **changes) for item in snap.links)
        req = request(max_observation_age_ms=50)
        assert (
            plan(req, snapshot=replace(snap, links=links, observed_at_ms=100)).status
            == "infeasible"
        )
    req = request()
    directed = replace(
        snap, links=tuple(item for item in snap.links if item.source_party_id != "client")
    )
    assert plan(req, snapshot=directed).status == "infeasible"


def test_remote_mac_denominator_and_native_local_placement_geometry():
    base = request()
    local_attention = replace(
        base.candidates[0],
        pipeline=MaskedLinearCpu(
            Model.tiny(), placement=ClientLinearRoles(("qkv_projection", "attention_output"))
        ),
    )
    req = replace(
        base,
        candidates=(local_attention,),
        policy=replace(base.policy, objectives=("client_weight_bytes",)),
    )
    body = plan(req, snapshot=snapshot())
    all_linear = plan(
        replace(req, policy=replace(req.policy, remote_mac_denominator="all_linear")),
        snapshot=snapshot(),
    )
    assert body.status == "feasible"
    assert 0 < all_linear.costs["remote_mac_fraction"] < body.costs["remote_mac_fraction"] < 1
    assert (
        body.costs["client_linear_macs"]
        > plan(base, snapshot=snapshot()).costs["client_linear_macs"]
    )


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"devices": ("metal",)}, "CAPABILITY_MISMATCH"),
        ({"allowed_models": ("f" * 64,)}, "MODEL_NOT_ALLOWED"),
        ({"allowed_compositions": ("e" * 64,)}, "COMPOSITION_NOT_ALLOWED"),
        ({"max_memory_bytes": 1}, "capacity exceeded"),
        ({"max_weight_bytes": 1}, "capacity exceeded"),
        ({"max_sessions": 1, "active_sessions": 1}, "capacity exceeded"),
        ({"expires_at_ms": 100}, "STALE_OFFER"),
    ],
)
def test_offer_devices_models_compositions_memory_weights_slots_expiry(override, reason):
    req = request()
    snap = snapshot(
        offers=(
            offer("client", ("trusted_client",)),
            offer("a", **override),
            offer("b", **override),
        )
    )
    result = plan(req, snapshot=snap)
    assert result.status == "infeasible" and reason in reasons(result)


def test_metal_pipeline_not_assigned_to_cpu_only_and_pending_rejected():
    result = plan(
        request((ClientOnlyMetal,), objectives=("client_weight_bytes",)), snapshot=snapshot()
    )
    assert result.status == "infeasible" and "CAPABILITY_MISMATCH" in reasons(result)
    from pllm.nonlinear import ArithmeticGarblingSiluQ7
    from pllm.configuration import Pipeline

    req = request()
    base = req.candidates[0]
    pending = replace(
        base,
        pipeline=Pipeline(
            model=Model.tiny(),
            components={**base.pipeline.components, "nonlinear": ArithmeticGarblingSiluQ7()},
        ),
    )
    result = plan(replace(req, candidates=(pending,)), snapshot=snapshot())
    assert result.status == "infeasible" and "UNSUPPORTED_COMPOSITION" in reasons(result)
    document = req.to_spec()
    document["candidates"][0]["pipeline"]["components"]["nonlinear"] = {
        "component": "pllm/pending/v1",
        "params": {},
    }
    rejected = plan(PlanningRequest.from_spec(document), snapshot=snapshot())
    assert rejected.status == "infeasible" and "UNSUPPORTED_COMPOSITION" in reasons(rejected)


def test_truncation_outcomes_candidate_and_assignment_caps():
    no_winner = plan(request(max_assignments=1), snapshot=snapshot())
    assert no_winner.status == "inconclusive" and not no_winner.exhaustive
    winner = plan(request(max_assignments=2), snapshot=snapshot())
    assert winner.status == "feasible" and not winner.exhaustive
    assert winner.to_spec()["coverage"]["assignments_evaluated"] == 2
    req = request(
        (ClientOnlyCpu, MaskedLinearCpu), max_candidates=1, objectives=("client_weight_bytes",)
    )
    selected = plan(req, snapshot=snapshot())
    assert not selected.exhaustive and selected.to_spec()["coverage"]["candidates_evaluated"] == 1
    with pytest.raises(NetworkError, match="SEARCH_LIMIT"):
        open_execution(winner, network=NetworkSpec("dev", winner.snapshot), clock=lambda: 100)


def test_native_aggregate_capacity_multiple_roles_same_party():
    caps = ("trusted_client", *PROVIDER_CAPS)
    snap = snapshot(
        offers=(offer("client", caps, max_sessions=1), offer("a", PROVIDER_CAPS, max_sessions=1))
    )
    result = plan(request(objectives=("client_weight_bytes",)), snapshot=snap)
    assert result.status == "infeasible" and "capacity exceeded" in reasons(result)


def test_stale_snapshot_epoch_and_current_admission():
    req, snap = request(), snapshot()
    decision = plan(req, snapshot=snap)
    decision.validate(snapshot=snap, evaluated_at_ms=101)
    with pytest.raises(NetworkError, match="STALE_SNAPSHOT"):
        decision.validate(snapshot=snap, evaluated_at_ms=1000)
    party = assigned(decision)["inference"]
    restarted = replace(
        snap,
        offers=tuple(
            replace(item, instance_epoch="c" * 64) if item.party_id == party else item
            for item in snap.offers
        ),
    )
    with pytest.raises(NetworkError, match="STALE_INSTANCE_EPOCH"):
        decision.validate(snapshot=restarted, evaluated_at_ms=100)
    full = replace(
        snap,
        offers=tuple(
            replace(item, active_sessions=item.max_sessions) if item.party_id == party else item
            for item in snap.offers
        ),
    )
    with pytest.raises(ValueError, match="capacity exceeded"):
        decision.validate(snapshot=full, evaluated_at_ms=100)
    assert (
        plan(replace(req, policy=replace(req.policy, evaluated_at_ms=1000)), snapshot=snap).status
        == "infeasible"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(result_digest="a" * 64),
        lambda d: d["selection"]["costs"].update(link_transfer_ms=0),
        lambda d: d["selection"]["native_placement"].update(placement_digest="0" * 64),
        lambda d: d["coverage"].update(exhaustive=False),
        lambda d: d.update(extra=True),
        lambda d: d["snapshot"]["offers"][0].update(max_weight_bytes=1),
        lambda d: d["request"]["model_plan"].update(config_digest="f" * 64),
    ],
)
def test_result_json_tamper_replays_native_and_score(mutate):
    result = plan(request(), snapshot=snapshot())
    document = result.to_spec()
    mutate(document)
    with pytest.raises(ValueError):
        PlanningResult.from_spec(document)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b" " * ((1 << 20) + 1),
        b"[]",
    ],
)
def test_strict_json_duplicate_nonfinite_and_bound(raw):
    with pytest.raises(NetworkError):
        strict_load(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("instance_epoch", "0" * 64),
        ("instance_epoch", "A" * 64),
        ("max_sessions", True),
        ("max_memory_bytes", 1 << 64),
        ("capabilities", ("protected_provider",)),
        ("allowed_models", ("model-name",)),
        ("authenticated", True),
        ("allowed_compositions", ()),
    ],
)
def test_strict_offer_records(field, value):
    with pytest.raises(ValueError):
        replace(offer("a"), **{field: value})


def test_request_source_budget_precision_and_explicit_limits():
    from pllm.quantization import SymmetricPerRow

    req = request()
    exp = req.candidates[0]
    changed = [
        replace(exp, budget=ExecutionBudget(1, 4, 2)),
        replace(exp, pipeline=MaskedLinearCpu(Model.tiny(model_id="different"))),
        replace(
            exp,
            pipeline=MaskedLinearCpu(
                Model.tiny(), quantization=SymmetricPerRow(weight_bits=4, activation_bits=8)
            ),
        ),
    ]
    for candidate in changed:
        with pytest.raises(NetworkError):
            replace(req, candidates=(exp, candidate))
    for changes in (
        {"max_candidates": 0},
        {"max_assignments": True},
        {"objectives": ("magic-score",)},
        {"remote_mac_denominator": "whole_cpu"},
    ):
        with pytest.raises(NetworkError):
            replace(req.policy, **changes)


def test_declared_cost_evidence_exact_locks_geometry_and_no_measurement_fraud():
    req = replace(request(), source_lock_digest="c" * 64)
    exp = req.candidates[0]
    schedule = req.model_plan.runtime_schedule(exp.pipeline)
    evidence = CandidateCostEvidence(
        exp.configuration_digest(),
        req.model_plan.digest,
        schedule.digest,
        req.source_lock_digest,
        req.workload_cohort_digest,
        "native-geometry-scenario",
        1664,
        2688,
    )
    bound = replace(req, cost_evidence=(evidence,))
    result = plan(bound, snapshot=snapshot())
    assert result.status == "feasible"
    assert result.costs["declared_cost_evidence_digest"] == evidence.digest
    assert result.costs["client_cpu_ns"] is None
    assert PlanningResult.from_spec(result.to_spec()).digest == result.digest
    for name in (
        "model_plan_digest",
        "source_lock_digest",
        "configuration_digest",
        "workload_cohort_digest",
    ):
        with pytest.raises(NetworkError):
            replace(req, cost_evidence=(replace(evidence, **{name: "d" * 64}),))
    for changed in (
        replace(evidence, schedule_digest="d" * 64),
        replace(evidence, online_all_link_body_bytes=0),
    ):
        rejected = plan(replace(req, cost_evidence=(changed,)), snapshot=snapshot())
        assert rejected.status == "infeasible" and "cost evidence" in reasons(rejected)
    with pytest.raises(NetworkError, match="cannot claim measured"):
        replace(evidence, origin="measured")


def test_static_record_size_caps_and_fields():
    with pytest.raises(NetworkError, match="1..64"):
        snapshot(offers=tuple(offer(f"party-{index}") for index in range(65)))
    with pytest.raises(NetworkError, match="duplicate party"):
        snapshot(offers=(offer("a"), offer("a")))
    with pytest.raises(NetworkError, match="only static local"):
        NetworkSpec("dev", snapshot(), backend="remote")
    with pytest.raises(NetworkError, match="disallowed operator"):
        NetworkSpec("dev", snapshot(), allowed_operators=("client",))
    for cls, document in (
        (PartyOffer, offer("a").to_spec()),
        (NetworkSnapshot, snapshot().to_spec()),
        (PlanningPolicy, request().policy.to_spec()),
        (PlanningRequest, request().to_spec()),
    ):
        document["unexpected"] = True
        with pytest.raises(NetworkError, match="fields"):
            cls.from_spec(document)


def test_client_party_role_aggregation_and_logical_remote_work_is_not_duplicate_credit():
    req = request((TwoOnlineOffsetCpu,), objectives=("client_weight_bytes",))
    caps = ("trusted_client", *PROVIDER_CAPS)
    snap = snapshot(offers=(offer("client", caps), offer("a")))
    result = plan(req, snapshot=snap)
    assert result.status == "feasible"
    assert any(assigned(result)[role] == "client" for role in ("worker_a", "worker_b"))
    weights = result.native_placement["required_weight_bytes"]
    assert result.costs["client_weight_bytes"] == sum(
        weights[role] for role, party in assigned(result).items() if party == "client"
    )
    assert result.costs["remote_mac_fraction"] == 0
    rejected = plan(
        replace(req, policy=replace(req.policy, minimum_remote_mac_fraction=0.1)), snapshot=snap
    )
    assert rejected.status == "infeasible"


@pytest.fixture
def local_selected(tmp_path):
    from pllm.model_loader import resolve_model
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    checkpoint = create_tiny_llama_checkpoint(
        tmp_path / "checkpoint",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    config = json.loads((checkpoint / "config.json").read_bytes())
    model = Model.path(str(checkpoint), model_id="selected-tiny")
    budget = ExecutionBudget(1, 32, 2)
    experiment = Experiment(
        "selected-offset",
        TwoOnlineOffsetCpu(model),
        Deployment.local(root="local://selected"),
        budget,
    )
    model_plan = lower_model(config, batch=1, max_input_tokens=32, max_new_tokens=2)
    now = time.time_ns() // 1_000_000
    offers = (
        replace(offer("client", ("trusted_client",)), expires_at_ms=now + 300_000),
        replace(offer("a"), expires_at_ms=now + 300_000),
        replace(offer("b"), expires_at_ms=now + 300_000),
    )
    snap = NetworkSnapshot("selected", offers, (), "fixture", now, now + 300_000)
    req = PlanningRequest(
        model_plan,
        (experiment,),
        PlanningPolicy(
            "client",
            now,
            4,
            32,
            ("client_weight_bytes",),
            minimum_remote_mac_fraction=0.5,
            max_observation_age_ms=300_000,
        ),
        source_lock_digest=resolve_model(model).source_lock_digest,
    )
    return plan(req, snapshot=snap), NetworkSpec("selected", snap), checkpoint


@pytest.mark.integration
def test_selected_sdk_and_ordinary_benchmark_execute_same_exact_experiment(local_selected):
    from pllm.runtime.benchmark_cli import run_loopback_benchmark

    selected, network, checkpoint = local_selected
    with open_execution(selected, network=network) as lease:
        with pytest.raises(TypeError):
            copy.copy(lease)
        with pytest.raises(TypeError):
            pickle.dumps(lease)
        with pytest.raises(ValueError, match="manual client overrides"):
            OpenAI(execution=lease, base_url="http://localhost:9999")
        with OpenAI(execution=lease) as client:
            # Refused requests must not touch any provider or consume the request slot.
            with pytest.raises(NetworkError, match="output cap"):
                client.responses.create(input="Hi", max_output_tokens=3)
            with pytest.raises(NetworkError, match="input exceeds"):
                client.responses.create(input="x" * 1000, max_output_tokens=2)
            assert client.privacy_audit.inference_stage_calls == 0
            response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
            assert response.usage.output_tokens == 2
            assert client.privacy_audit.inference_stage_calls > 0
            with pytest.raises(NetworkError, match="budget exhausted"):
                client.responses.create(input="Hi", max_output_tokens=2)
            with pytest.raises(NetworkError, match="already has a client"):
                OpenAI(execution=lease)
        placement = lease.binding.to_spec()["placement"]
        assert placement["model_plan_digest"] == selected.request.model_plan.digest
        assert placement["composition_digest"] == selected.experiment.pipeline.digest()
    assert lease.closed
    with pytest.raises(NetworkError, match="closed"):
        OpenAI(execution=lease)
    report = run_loopback_benchmark(
        model=str(checkpoint),
        model_id="selected-tiny",
        tiny=False,
        prompt="Hi",
        max_output_tokens=2,
        warmups=0,
        repetitions=1,
        timeout_seconds=120,
        experiment=selected.experiment,
        temperature=0.0,
        capture_output_digest=True,
    )
    assert report["checks"]["passed"]
    assert (
        report["experiment"]["configuration_digest"] == selected.experiment.configuration_digest()
    )
    assert report["summary"]["total_output_tokens"] == 2
    assert report["configuration"]["sampling"]["effective_temperature"] == 0.0


def test_checkpoint_mismatch_and_source_lock_fail_before_any_role_start(
    local_selected, monkeypatch
):
    from pllm.runtime import servers

    selected, network, checkpoint = local_selected
    monkeypatch.setattr(
        servers,
        "build_roles",
        lambda *args, **kwargs: pytest.fail("role startup before lock check"),
    )
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_bytes())
    config["hidden_size"] = 16
    config_path.write_text(json.dumps(config))
    with pytest.raises(NetworkError, match="ModelPlan digest mismatch"):
        open_execution(selected, network=network)
    config["hidden_size"] = 8
    config_path.write_text(json.dumps(config))  # Changed exact source bytes, same semantics.
    with pytest.raises(NetworkError, match="source lock mismatch"):
        open_execution(selected, network=network)


@pytest.mark.integration
def test_stream_cancellation_closes_existing_session_and_lease_children(local_selected):
    selected, network, _ = local_selected
    with open_execution(selected, network=network) as lease:
        with OpenAI(execution=lease) as client:
            with client.responses.create(input="Hi", max_output_tokens=2, stream=True) as stream:
                next(stream)
            assert not client._core._active_offset_transports
    assert lease.closed


@pytest.mark.integration
def test_selected_prepared_stream_cancellation_burns_reserved_rows(local_selected):
    from pllm.preparation import PreparedInventory

    selected, network, _ = local_selected
    experiment = replace(
        selected.experiment,
        pipeline=MaskedLinearCpu(
            selected.experiment.pipeline.model, inventory=PreparedInventory("request-sized")
        ),
    )
    selected = plan(replace(selected.request, candidates=(experiment,)), snapshot=network.snapshot)
    with open_execution(selected, network=network) as lease:
        with OpenAI(execution=lease) as client:
            stream = client.responses.create(input="Hi", max_output_tokens=2, stream=True)
            response_id = next(stream).response["id"]
            reserved = client.prepared_inventory_status()["reserved"]
            assert reserved > 0
            stream.close()
            status = client.prepared_inventory_status()
            assert status["reserved"] == 0
            assert status["burned"] >= reserved
            assert client.responses.retrieve(response_id).status == "cancelled"
            assert client.privacy_audit.preparation_requests_during_online == 0
    assert lease.closed


@pytest.mark.asyncio
@pytest.mark.integration
async def test_selected_async_sdk_cancellation_uses_existing_cleanup(local_selected):
    from pllm import AsyncOpenAI

    selected, network, _ = local_selected
    async with open_execution(selected, network=network) as lease:
        async with AsyncOpenAI(execution=lease) as client:
            stream = await client.responses.create(input="Hi", max_output_tokens=2, stream=True)
            try:
                async for _event in stream:
                    break
            finally:
                await stream.aclose()
            assert not client.sync._core._active_offset_transports
    assert lease.closed


def test_startup_failure_client_failure_and_failed_enter_release_local_roles(
    local_selected, monkeypatch
):
    from pllm.runtime import servers

    selected, network, _ = local_selected

    class Topology:
        closed = False

        def start(self):
            raise RuntimeError("partial start failed")

        def close(self):
            self.closed = True

        def client(self):
            raise RuntimeError("client failed")

    topology = Topology()
    monkeypatch.setattr(servers, "build_roles", lambda *args, **kwargs: topology)
    with pytest.raises(RuntimeError, match="partial start failed"):
        open_execution(selected, network=network)
    assert topology.closed
    topology = Topology()
    monkeypatch.setattr(topology, "start", lambda: topology)
    with open_execution(selected, network=network) as lease:
        with pytest.raises(RuntimeError, match="client failed"):
            OpenAI(execution=lease)
        assert topology.closed and lease.closed
    topology = Topology()
    monkeypatch.setattr(topology, "start", lambda: topology)
    now = [selected.request.policy.evaluated_at_ms]
    lease = open_execution(selected, network=network, clock=lambda: now[0])
    now[0] += 300_000
    with pytest.raises(NetworkError, match="STALE_SNAPSHOT"):
        with lease:
            pytest.fail("expired lease entered")
    assert topology.closed and lease.closed


@pytest.mark.integration
def test_incomplete_selected_execution_needs_and_honors_permission(local_selected):
    selected, network, _ = local_selected
    req = replace(
        selected.request,
        policy=replace(selected.request.policy, max_assignments=2, allow_incomplete_execution=True),
    )
    incomplete = plan(req, snapshot=network.snapshot)
    assert incomplete.status == "feasible" and not incomplete.exhaustive
    with open_execution(incomplete, network=network) as lease:
        with OpenAI(execution=lease) as client:
            assert client.responses.create(input="Hi", temperature=0.0).usage.output_tokens == 2


def test_failed_close_can_retry_cleanup_without_reopening_admission(local_selected, monkeypatch):
    from pllm.runtime import servers

    selected, network, _ = local_selected

    class Topology:
        attempts = 0
        closed = False

        def start(self):
            return self

        def close(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("child stop failed")
            self.closed = True

    topology = Topology()
    monkeypatch.setattr(servers, "build_roles", lambda *args, **kwargs: topology)
    lease = open_execution(selected, network=network)
    with pytest.raises(RuntimeError, match="child stop failed"):
        lease.close()
    assert lease.closed and not topology.closed
    with pytest.raises(NetworkError, match="closed"):
        OpenAI(execution=lease)
    lease.close()
    assert topology.closed and topology.attempts == 2


def test_cli_static_snapshot_create_inspect_explain_validate_and_overwrite(tmp_path, capsys):
    from pllm._cli.app import main

    now = time.time_ns() // 1_000_000
    snap = snapshot()
    snap = replace(
        snap,
        observed_at_ms=now,
        expires_at_ms=now + 60_000,
        offers=tuple(replace(item, expires_at_ms=now + 60_000) for item in snap.offers),
        links=tuple(
            replace(item, observed_at_ms=now, expires_at_ms=now + 60_000) for item in snap.links
        ),
    )
    req = request()
    req = replace(req, policy=replace(req.policy, evaluated_at_ms=now))
    network_path, request_path, snapshot_path, plan_path = [
        tmp_path / f"{name}.json" for name in ("network", "request", "snapshot", "plan")
    ]
    network_path.write_bytes(NetworkSpec("dev", snap).canonical_bytes())
    request_path.write_bytes(req.canonical_bytes())
    for command in ("inspect", "parties"):
        main(["network", command, str(network_path), "--format", "json"])
        assert json.loads(capsys.readouterr().out)["command"] == f"network.{command}"
    main(
        [
            "network",
            "snapshot",
            str(network_path),
            "--output",
            str(snapshot_path),
            "--format",
            "json",
        ]
    )
    capsys.readouterr()
    assert NetworkSnapshot.from_file(snapshot_path).digest == snap.digest
    create = [
        "plan",
        "create",
        str(request_path),
        "--snapshot",
        str(snapshot_path),
        "--output",
        str(plan_path),
        "--format",
        "json",
    ]
    main([*create, "--dry-run"])
    assert not plan_path.exists()
    capsys.readouterr()
    main(create)
    capsys.readouterr()
    for command in ("inspect", "explain", "validate"):
        args = ["plan", command, str(plan_path), "--format", "json"]
        if command == "validate":
            args += ["--snapshot", str(snapshot_path)]
        main(args)
        assert json.loads(capsys.readouterr().out)["command"] == f"plan.{command}"
    with pytest.raises(SystemExit) as error:
        main(create)
    assert error.value.code == 4
    capsys.readouterr()
    main([*create, "--force"])
    capsys.readouterr()
    assert PlanningResult.from_file(plan_path).status == "feasible"
    assert discover(NetworkSpec.from_file(network_path)).digest == snap.digest


def test_client_state_costs_stay_client_local_and_plan_replays():
    from pllm.quantization import SymmetricPerRow
    from pllm.state import ClientPrefixReuse
    from pllm.search import ClientStateCostEvidence

    req = request(objectives=("online_all_link_body_bytes",))
    candidate = replace(req.candidates[0], pipeline=MaskedLinearCpu(Model.tiny(),
        quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=3)))
    req = replace(req, candidates=(candidate,), source_lock_digest="a" * 64)
    cold = plan(req, snapshot=snapshot())
    evidence = ClientStateCostEvidence(candidate.configuration_digest(), req.model_plan.digest,
        req.source_lock_digest, "client", 2, 1024)
    warm_request = replace(req, state_evidence=(evidence,))
    warm = plan(warm_request, snapshot=cold.snapshot)
    assert warm.costs["online_all_link_body_bytes"] * 2 == cold.costs["online_all_link_body_bytes"]
    # Both miss and hit must reserve the complete configured cache capacity.
    assert warm.costs["role_memory_estimates"]["client"] == cold.costs["role_memory_estimates"]["client"]
    assert assigned(warm)["client"] == "client"
    assert "tokens" not in json.dumps(warm_request.to_spec()["state_evidence"])
    assert PlanningRequest.from_spec(warm_request.to_spec()).digest == warm_request.digest
    assert PlanningResult.from_spec(warm.to_spec(), request=warm_request, snapshot=cold.snapshot).digest == warm.digest
    for invalid in (replace(evidence, client_party_id="a"),
                    replace(evidence, source_lock_digest="b" * 64),
                    replace(evidence, reusable_prefill_rows=4)):
        with pytest.raises(NetworkError):
            replace(req, state_evidence=(invalid,))
    with pytest.raises(NetworkError, match="incremental"):
        replace(evidence, state_basis="incremental")
    with pytest.raises(NetworkError, match="measured"):
        replace(evidence, origin="measurement")
    constrained = replace(cold.snapshot, offers=tuple(
        replace(row, max_memory_bytes=cold.costs["role_memory_estimates"]["client"] - 1)
        if row.party_id == "client" else row for row in cold.snapshot.offers))
    assert plan(warm_request, snapshot=constrained).status == "infeasible"


def test_switch_hysteresis_keeps_only_feasible_incumbent():
    req = request((MaskedLinearCpu, TwoOnlineOffsetCpu), objectives=("online_all_link_body_bytes",))
    best = plan(req, snapshot=snapshot())
    incumbent = next(row for row in req.candidates if row.configuration_digest() != best.experiment.configuration_digest())
    sticky = replace(req, policy=replace(req.policy,
        incumbent_configuration_digest=incumbent.configuration_digest(), minimum_switch_improvement_fraction=1.0))
    kept = plan(sticky, snapshot=snapshot())
    assert kept.experiment == incumbent
    assert any(row["reason"] == "SWITCH_HYSTERESIS" for row in kept.to_spec()["alternatives"])
    assert plan(replace(sticky, policy=replace(sticky.policy, minimum_switch_improvement_fraction=0.01)), snapshot=snapshot()).experiment == best.experiment
    impossible = replace(snapshot(), offers=tuple(
        replace(row, allowed_compositions=(best.experiment.pipeline.digest(),)) for row in snapshot().offers))
    assert plan(sticky, snapshot=impossible).experiment == best.experiment
    assert PlanningPolicy.from_spec(sticky.policy.to_spec()).digest == sticky.policy.digest
