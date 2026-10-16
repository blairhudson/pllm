"""Raw-byte reservations share live SDK capacity without fake planning records."""
import threading

import pytest

from pllm.deployment.network import NetworkError
from pllm.deployment.resources import ClientResources, _ClientCapacity


@pytest.mark.parametrize("amount", [None, True, 0, -1, 1.5, 1 << 64])
def test_incremental_reservation_validates_before_observation(monkeypatch, amount):
    capacity = _ClientCapacity()
    monkeypatch.setattr(capacity, "observe", lambda: pytest.fail("observed invalid reservation"))
    with pytest.raises(NetworkError):
        with capacity.hold_bytes(amount):
            pytest.fail("admitted invalid bytes")
    assert not capacity._owners


@pytest.mark.parametrize("available,reserve,container,amount,allowed", [
    (100, 20, None, 80, True), (100, 20, None, 81, False),
    (100, 20, 30, 30, True), (100, 20, 30, 31, False),
])
def test_incremental_reservation_intersects_post_reserve_host_and_cgroup(
        monkeypatch, available, reserve, container, amount, allowed):
    capacity = _ClientCapacity()
    monkeypatch.setattr(capacity, "observe", lambda: ClientResources(
        100, available, 1, reserve, sum(capacity._owners.values()), container))
    if allowed:
        with capacity.hold_bytes(amount) as observation:
            assert observation.headroom_bytes >= amount
            assert sum(capacity._owners.values()) == amount
    else:
        with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
            with capacity.hold_bytes(amount):
                pytest.fail("overcommitted")
    assert not capacity._owners


def test_incremental_reservation_concurrency_and_failure_release(monkeypatch):
    capacity = _ClientCapacity()
    monkeypatch.setattr(capacity, "observe", lambda: ClientResources(
        100, 120, 1, 20, sum(capacity._owners.values()), 100))
    entered, release = threading.Event(), threading.Event()
    failures = []
    def worker():
        try:
            with capacity.hold_bytes(100):
                entered.set()
                assert release.wait(5)
                raise ValueError("expected worker failure")
        except ValueError as exc:
            failures.append(str(exc))
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
            with capacity.hold_bytes(1):
                pytest.fail("double spent")
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and failures == ["expected worker failure"]
    assert not capacity._owners
    with capacity.hold_bytes(100):
        assert sum(capacity._owners.values()) == 100
    assert not capacity._owners


def test_incremental_observation_failure_cannot_leave_owner(monkeypatch):
    capacity = _ClientCapacity()
    def fail():
        raise NetworkError("CLIENT_MEMORY_OBSERVATION_UNAVAILABLE")
    monkeypatch.setattr(capacity, "observe", fail)
    with pytest.raises(NetworkError, match="OBSERVATION_UNAVAILABLE"):
        with capacity.hold_bytes(1):
            pytest.fail("admitted without observation")
    assert not capacity._owners
