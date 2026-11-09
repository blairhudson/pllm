"""Raw-byte reservations share live SDK capacity without fake planning records."""
from contextlib import contextmanager
import inspect
import sys
import threading
from unittest.mock import Mock
import weakref

import pytest

from pllm.deployment.network import NetworkError
from pllm.deployment import resources
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


class _CloseOwner:
    """Resource-free owner: only explicit close changes its state."""

    def __init__(self):
        self.calls = 0
        self.error = None

    def close(self):
        self.calls += 1
        if self.error is not None:
            raise self.error


@pytest.fixture
def isolated_reservation(monkeypatch):
    capacity = _ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: ClientResources(
        100, 1 << 20, 1, 0, sum(capacity._owners.values())))

    def create():
        reservation = resources.reserve_incremental_memory(64 << 10)
        original_exit = reservation._manager.__exit__
        backend = Mock()

        def observed_exit(*exc):
            backend(*exc)
            return original_exit(*exc)

        # Keep the real backend inside its own manager, not in pytest's patch
        # undo log or the external observation mock (both would root custody).
        reservation._manager.__exit__ = observed_exit
        return capacity, reservation, backend

    # Fixture cache holds only this factory, never a reservation/token/manager.
    return create


@contextmanager
def _interrupt_statement(reservation, method, statement, error):
    """One line fault in the real API; never replace custody/refund machinery."""
    lines, start = inspect.getsourcelines(method)
    locations = [start + offset for offset, line in enumerate(lines)
                 if line.strip() == statement]
    assert locations, statement
    # drain_failed has the same terminal clear in recovery and normal commit.
    location = locations[-1]
    code = method.__func__.__code__
    fired = []
    previous = sys.gettrace()

    def trace(frame, event, arg):
        if (event == "line" and frame.f_code is code
                and frame.f_code.co_filename == code.co_filename
                and frame.f_lineno == location
                and frame.f_locals.get("self") is reservation and not fired):
            fired.append(location)
            assert reservation._token.retained is None
            assert type(reservation._failed) is tuple and 1 <= len(reservation._failed) <= 2
            raise error
        return trace

    try:
        sys.settrace(trace)
        yield
    finally:
        sys.settrace(previous)
        assert fired == [location]


@pytest.mark.parametrize("statement", [
    "if self._failed is owners:",
    "self._token.retained = self  # actual module admission-token root",
], ids=["after_failed_publication", "before_finally_root"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("owner_count", [1, 2])
def test_failed_publication_keeps_original_ledger_custody(
        isolated_reservation, statement, error_type, owner_count):
    capacity, reservation, backend = isolated_reservation()
    token = reservation._token
    owners = tuple(_CloseOwner() for _ in range(owner_count))
    owner_refs = tuple(weakref.ref(owner) for owner in owners)
    reservation_ref = weakref.ref(reservation)
    owners_id = id(owners)
    error = error_type("publication interrupted")
    try:
        with reservation:
            assert capacity._owners[token] == 64 << 10
            with _interrupt_statement(reservation, reservation.retain_failed, statement, error):
                reservation.retain_failed(owners)
    except BaseException as caught:
        assert caught is error
    else:
        pytest.fail("publication fault did not escape")
    assert reservation._token is token
    assert reservation._failed is owners
    assert token.retained is reservation
    assert reservation.pending_retirement
    assert capacity._owners == {token: 64 << 10}
    assert all(owner.calls == 0 for owner in owners)
    backend.assert_not_called()
    # Remove exception frames and all caller strong references to custody.
    error.__traceback__ = None
    del reservation, owners, token
    rooted_token, = capacity._owners
    assert rooted_token.retained is reservation_ref()
    assert id(rooted_token.retained._failed) == owners_id
    assert all(ref() is not None and ref().calls == 0 for ref in owner_refs)
    assert capacity._owners[rooted_token] == 64 << 10
    rooted_token.retained.drain_failed()
    backend.assert_called_once_with(None, None, None)
    assert not capacity._owners


def test_failed_retention_prevalidation_leaves_ordinary_refund(isolated_reservation):
    capacity, reservation, backend = isolated_reservation()
    owner = _CloseOwner()
    with reservation:
        token = reservation._token
        with pytest.raises(NetworkError, match="CUSTODY_INVALID"):
            reservation.retain_failed([owner])
        assert reservation._failed == ()
        assert token.retained is None
        assert capacity._owners[token] == 64 << 10
    assert owner.calls == 0
    assert not capacity._owners
    backend.assert_called_once_with(None, None, None)


def test_empty_failed_custody_keeps_ordinary_exit_refund(isolated_reservation):
    capacity, reservation, backend = isolated_reservation()
    with reservation:
        token = reservation._token
        assert reservation._failed == ()
        assert capacity._owners[token] == 64 << 10
    assert token.retained is None
    assert not capacity._owners
    assert reservation.__exit__(None, None, None) is False
    backend.assert_called_once_with(None, None, None)


def test_repaired_custody_failed_drain_then_single_refund(isolated_reservation):
    capacity, reservation, backend = isolated_reservation()
    owner = _CloseOwner()
    owners = (owner,)
    publication_error = RuntimeError("publication interrupted")
    with pytest.raises(RuntimeError) as publication:
        with reservation:
            with _interrupt_statement(
                    reservation, reservation.retain_failed,
                    "self._token.retained = self  # actual module admission-token root",
                    publication_error):
                reservation.retain_failed(owners)
    assert publication.value is publication_error
    token = reservation._token
    assert token.retained is reservation and reservation._failed is owners
    owner.error = RuntimeError("close failed")
    with pytest.raises(RuntimeError) as retirement:
        reservation.drain_failed()
    assert retirement.value is owner.error
    assert owner.calls == 1
    assert capacity._owners == {token: 64 << 10}
    assert token.retained is reservation
    backend.assert_not_called()
    owner.error = None
    reservation.drain_failed()
    reservation.drain_failed()
    assert reservation.__exit__(None, None, None) is False
    assert owner.calls == 2
    assert not capacity._owners
    assert token.retained is None and reservation._failed == ()
    backend.assert_called_once_with(None, None, None)


def test_repeat_retention_cannot_overwrite_interrupted_tuple(isolated_reservation):
    capacity, reservation, backend = isolated_reservation()
    owners, replacements = (_CloseOwner(),), (_CloseOwner(),)
    with reservation:
        token = reservation._token
        error = RuntimeError("publication interrupted")
        with pytest.raises(RuntimeError) as publication:
            with _interrupt_statement(
                    reservation, reservation.retain_failed,
                    "self._token.retained = self  # actual module admission-token root", error):
                reservation.retain_failed(owners)
        assert publication.value is error
        assert token.retained is None and reservation._failed is owners
        with pytest.raises(NetworkError, match="CUSTODY_INVALID"):
            reservation.retain_failed(replacements)
        assert reservation._failed is owners and token.retained is None
        assert capacity._owners == {token: 64 << 10}
    assert token.retained is reservation and reservation._failed is owners
    assert owners[0].calls == replacements[0].calls == 0
    backend.assert_not_called()
    reservation.drain_failed()
    assert owners[0].calls == 1 and replacements[0].calls == 0
    assert not capacity._owners
    backend.assert_called_once_with(None, None, None)


def test_foreign_retained_root_fails_closed_unchanged(isolated_reservation):
    capacity, reservation, backend = isolated_reservation()
    reservation.__enter__()
    token, foreign = reservation._token, object()
    token.retained = foreign
    with pytest.raises(NetworkError, match="CUSTODY_INVALID"):
        reservation.__exit__(None, None, None)
    assert token.retained is foreign and reservation._failed == ()
    assert capacity._owners == {token: 64 << 10}
    backend.assert_not_called()
    token.retained = None
    reservation.__exit__(None, None, None)
    assert not capacity._owners
    backend.assert_called_once_with(None, None, None)


def test_rootless_postcommit_requires_explicit_drain_recovery(isolated_reservation):
    capacity, reservation, backend = isolated_reservation()
    owner = _CloseOwner()
    owners = (owner,)
    with reservation:
        reservation.retain_failed(owners)
    token = reservation._token
    error = KeyboardInterrupt("terminal custody clear interrupted")
    with pytest.raises(KeyboardInterrupt) as interrupted:
        with _interrupt_statement(
                reservation, reservation.drain_failed,
                "self._failed = ()  # terminal custody", error):
            reservation.drain_failed()
    assert interrupted.value is error
    assert not capacity._owners and token.retained is None
    assert reservation._failed is owners
    assert reservation._drain_finish_state == "committed"
    assert owner.calls == 1
    backend.assert_called_once_with(None, None, None)
    with pytest.raises(NetworkError, match="CUSTODY_INVALID"):
        reservation.__exit__(None, None, None)
    assert not capacity._owners and token.retained is None
    assert reservation._failed is owners and owner.calls == 1
    backend.assert_called_once_with(None, None, None)
    reservation.drain_failed()
    assert not capacity._owners and token.retained is None
    assert reservation._failed == () and reservation._exited
    assert reservation.__exit__(None, None, None) is False
    reservation.drain_failed()
    assert owner.calls == 1
    backend.assert_called_once_with(None, None, None)
