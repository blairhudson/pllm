from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pllm.runtime.correlation_ledger import DurableCorrelationLedger


def test_reserved_entries_never_return_to_available(tmp_path: Path) -> None:
    path = tmp_path / "inventory.db"
    ledger = DurableCorrelationLedger(path)
    ledger.add("m", "s", "k", count=2)
    first = ledger.reserve("m", "s", "k")
    ledger.close()

    reopened = DurableCorrelationLedger(path)
    counts = reopened.counts("m", "s", "k")
    assert counts == {"available": 1, "reserved": 1, "consumed": 0, "burned": 0}
    reopened.burn_stale_reservations(older_than_seconds=-1)
    counts = reopened.counts("m", "s", "k")
    assert counts == {"available": 1, "reserved": 0, "consumed": 0, "burned": 1}
    second = reopened.reserve("m", "s", "k")
    assert second.correlation_id != first.correlation_id
    reopened.consume(second.correlation_id)
    assert reopened.counts("m", "s", "k")["consumed"] == 1


def test_concurrent_reservations_are_unique(tmp_path: Path) -> None:
    path = tmp_path / "inventory.db"
    ledger = DurableCorrelationLedger(path)
    ledger.add("m", "s", "k", count=32)
    ledger.close()

    def reserve_one(_: int) -> int:
        local = DurableCorrelationLedger(path)
        try:
            return local.reserve("m", "s", "k").correlation_id
        finally:
            local.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(reserve_one, range(32)))
    assert len(set(ids)) == 32
