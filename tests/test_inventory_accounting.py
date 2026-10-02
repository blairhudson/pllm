"""Partial-stage cancellation must conserve material and revoke leases."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.preparation import PreparedInventory as InventoryPolicy
from pllm.profiles import MaskedLinearCpu
from pllm.runtime import transformer_client
from pllm.runtime.transformer_client import PreparedInventory, PreparedStageRows, TransformerClientError


def test_partial_stage_claim_close_cancel_conserve_and_revoke(monkeypatch):
    monkeypatch.setattr(transformer_client, "derive_online_attempt_id", lambda request, row: str(row))
    counts = defaultdict(int)
    def record(name, count):
        counts[name] += count
    stages = {name: PreparedStageRows(SimpleNamespace(), np.ones((10, 2)), np.ones((10, 3)))
              for name in ("first", "second")}
    inventory = PreparedInventory("bounded", 10, stages, _audit=record)
    lease = inventory.reserve(4)
    lease.take("first", 3)
    lease.take("second", 1)
    lease.close()
    lease.close()
    with pytest.raises(TransformerClientError, match="closed"):
        lease.take("first", 1)
    inventory.cancel()
    inventory.cancel()
    with pytest.raises(TransformerClientError, match="cancelled"):
        inventory.reserve(1)
    assert counts["prepared_stage_rows_issued"] == 20
    assert counts["prepared_stage_rows_claimed"] == 4
    assert counts["prepared_stage_rows_burned"] == 4
    assert counts["prepared_stage_rows_discarded"] == 12
    assert inventory.available == 0
    # Cancellation must not zero a mask view while a caller could be forming x-r.
    assert np.all(stages["first"].input_mask == 1)


def test_cancel_active_lease_rejects_claim_and_burns_remaining(monkeypatch):
    monkeypatch.setattr(transformer_client, "derive_online_attempt_id", lambda request, row: str(row))
    inventory = PreparedInventory("bounded", 4, {"one": PreparedStageRows(
        SimpleNamespace(), np.ones((4, 2)), np.ones((4, 3)))})
    lease = inventory.reserve(2)
    inventory.cancel()
    with pytest.raises(TransformerClientError, match="cancelled"):
        lease.take("one", 1)
    lease.close()
    assert inventory.status()["stage_rows_burned"] == 2
    assert inventory.status()["stage_rows_discarded"] == 2


def test_close_revokes_verifier_claimed_concurrently(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setattr(transformer_client, "derive_online_attempt_id", lambda request, row: str(row))
    entered, release, cancelled = Event(), Event(), Event()
    def claim(*args):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(cancel=cancelled.set)
    stage = PreparedStageRows(SimpleNamespace(seed=b"public-test"), np.ones((2, 2)),
                              np.ones((2, 3)), verification=SimpleNamespace(claim=claim))
    lease = PreparedInventory("concurrent", 2, {"one": stage}).reserve(1)
    with ThreadPoolExecutor(2) as pool:
        take = pool.submit(lease.take, "one", 1)
        assert entered.wait(5)
        close = pool.submit(lease.close)
        release.set()
        take.result(5)
        close.result(5)
    assert cancelled.is_set()
    with pytest.raises(TransformerClientError, match="closed"):
        lease.take_verifier("one")


def test_post_close_ledger_keeps_missing_and_outstanding_rows_visible():
    from pllm.runtime.prepared_accounting import material_accounting
    assert material_accounting({})["conserved"] is None
    counts = {"issued": 10, "reserved": 4, "claimed": 3, "burned": 1, "discarded": 5}
    ledger = material_accounting({"prepared_stage_rows_" + key: value for key, value in counts.items()})
    assert ledger["remaining_stage_rows"] == 1
    assert ledger["conserved"] is False


@pytest.mark.rust
def test_optional_refill_preserves_old_identity_and_binds_new_policy(tmp_path):
    ordinary = InventoryPolicy()
    assert ordinary.params == {"policy": "request-sized", "rows": 1}
    chosen = InventoryPolicy(refill="on-demand")
    experiment = Experiment("demand", MaskedLinearCpu(Model.tiny(), inventory=chosen),
                            Deployment.local(root=str(tmp_path)), ExecutionBudget(1, 32, 2))
    resolved = experiment.resolve()
    assert resolved.inventory_refill == "on-demand"
    assert resolved.background_inventory_refill is False
    assert Experiment.from_spec(experiment.to_spec()) == experiment
    assert chosen != ordinary
