"""Real SQLite definitions for the dormant trusted-local withdrawal core.

These tests do not qualify deployment custody, database rollback protection,
cross-exec transfer, source opening, or operational activation.
"""
import copy
import os
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from pllm.runtime import _operator_source_work as work


def _fixture(tmp_path, initial=20):
    path = tmp_path / "account.sqlite"
    identity = work._provision_account(path, initial)
    return path, identity, work._OperatorController(path, identity)


def _pending(controller, amount=7, lengths=(3, 8)):
    parent = object()
    launch = controller.launch(parent, lengths)
    approval = controller.approve(launch, parent, lengths, amount)
    return launch, parent, approval, lengths, amount


def _issue(controller, pending):
    launch, parent, approval, lengths, amount = pending
    return controller.issue(launch, parent, approval, lengths, amount)


def _consume(controller, seat, pending):
    launch, parent, _, lengths, amount = pending
    return controller.consume(seat, launch, parent, lengths, amount)


def _row(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT * FROM account").fetchall()
    finally:
        connection.close()


def test_finite_account_chosen_work_and_one_use(tmp_path):
    path, identity, controller = _fixture(tmp_path)
    assert len(identity) == 64
    assert os.stat(path).st_mode & 0o777 == 0o600
    pending = _pending(controller, amount=7, lengths=(100, 200))
    seat = _issue(controller, pending)
    assert _row(path) == [(1, identity, 20, 13, 1)]
    assert _consume(controller, seat, pending) == 7  # Not 100, 200, or a cap.
    with pytest.raises(work._SourceWorkError):
        _consume(controller, seat, pending)
    with pytest.raises(work._SourceWorkError):
        _issue(controller, pending)
    assert _row(path) == [(1, identity, 20, 13, 1)]


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, float("inf"),
                                   float("nan"), "7", None, 1 << 63])
def test_initial_and_chosen_work_require_exact_finite_positive_int(tmp_path, bad):
    with pytest.raises(work._SourceWorkError):
        work._provision_account(tmp_path / "bad.sqlite", bad)
    assert not (tmp_path / "bad.sqlite").exists()
    path, _, controller = _fixture(tmp_path)
    parent = object()
    launch = controller.launch(parent, (3,))
    with pytest.raises(work._SourceWorkError):
        controller.approve(launch, parent, (3,), bad)
    assert _row(path)[0][3:] == (20, 0)


@pytest.mark.parametrize("lengths", [(), [], [3], (True,), (0,), (-1,), (1.0,),
                                        (1 << 63,), (3, 3), tuple(range(1, 1026))])
def test_lengths_require_exact_finite_bounded_tuple(tmp_path, lengths):
    _, _, controller = _fixture(tmp_path)
    with pytest.raises(work._SourceWorkError):
        controller.launch(object(), lengths)


def test_provisioning_is_explicit_exclusive_random_and_existing_only(tmp_path):
    path, identity, _ = _fixture(tmp_path)
    with pytest.raises(FileExistsError):
        work._provision_account(path, 500)
    other_id = work._provision_account(tmp_path / "other.sqlite", 20)
    assert other_id != identity
    with pytest.raises(work._SourceWorkError):
        work._OperatorController(path, other_id)
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        work._OperatorController(missing, identity)
    assert not missing.exists()


def test_issue_never_recreates_deleted_account(tmp_path):
    path, _, controller = _fixture(tmp_path)
    pending = _pending(controller)
    path.unlink()
    with pytest.raises(sqlite3.OperationalError):
        _issue(controller, pending)
    assert not path.exists()
    with pytest.raises(work._SourceWorkError):
        _issue(controller, pending)


@pytest.mark.parametrize("field,replacement", [
    (0, object.__new__(work._Launch)), (1, object()),
    (2, object.__new__(work._Approval)), (3, (8, 3)), (4, 8), (4, True),
])
def test_impersonation_or_changed_binding_burns_both_before_sql(
        tmp_path, monkeypatch, field, replacement):
    path, _, controller = _fixture(tmp_path)
    pending = _pending(controller)
    changed = list(pending)
    changed[field] = replacement

    def forbidden(*args):
        raise AssertionError("SQL reached before exact binding refusal")

    with monkeypatch.context() as scoped:
        scoped.setattr(work, "_connect", forbidden)
        with pytest.raises(work._SourceWorkError):
            _issue(controller, changed)
        with pytest.raises(work._SourceWorkError):
            _issue(controller, pending)
    assert _row(path)[0][3:] == (20, 0)


def test_only_one_pending_and_no_actor_or_cap_authority(tmp_path):
    _, _, controller = _fixture(tmp_path)
    pending = _pending(controller)
    launch, parent, _, lengths, amount = pending
    with pytest.raises(work._SourceWorkError, match="pending"):
        controller.approve(launch, parent, lengths, amount)
    with pytest.raises(work._SourceWorkError, match="live"):
        controller.launch(object(), lengths)
    with pytest.raises(TypeError):
        controller.approve(launch, parent, lengths, amount, actor="operator")
    with pytest.raises(TypeError):
        controller.issue(launch, parent, pending[2], lengths, cap=amount)
    with pytest.raises(TypeError):
        work._Seat()


def test_foreign_controller_and_restart_cannot_recover_seat(tmp_path):
    path, identity, controller = _fixture(tmp_path)
    pending = _pending(controller)
    seat = _issue(controller, pending)
    foreign = work._OperatorController(path, identity)
    with pytest.raises(work._SourceWorkError):
        _consume(foreign, seat, pending)
    with pytest.raises(work._SourceWorkError):
        _consume(controller, object.__new__(work._Seat), pending)
    controller.close()
    with pytest.raises(work._SourceWorkError):
        _consume(controller, seat, pending)
    assert _row(path)[0][3:] == (13, 1)


def test_objects_neither_copy_nor_serialize(tmp_path):
    _, _, controller = _fixture(tmp_path)
    pending = _pending(controller)
    seat = _issue(controller, pending)
    for value in (controller, pending[0], pending[2], seat):
        for operation in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(TypeError):
                operation(value)


def test_pid_checked_before_any_inherited_lock(tmp_path, monkeypatch):
    _, _, controller = _fixture(tmp_path)
    pending = _pending(controller)
    seat = _issue(controller, pending)

    class PoisonLock:
        def __enter__(self):
            raise AssertionError("inherited lock reached")

    controller._lock = PoisonLock()
    monkeypatch.setattr(work.os, "getpid", lambda: controller._pid + 1)
    for operation in (lambda: controller.launch(object(), (3,)),
                      lambda: controller.approve(pending[0], pending[1], (3, 8), 7),
                      lambda: _issue(controller, pending),
                      lambda: _consume(controller, seat, pending), controller.close):
        with pytest.raises(work._SourceWorkError, match="process"):
            operation()


def test_two_real_connections_race_same_generation_exactly_one_debit(tmp_path):
    path, identity, left = _fixture(tmp_path)
    right = work._OperatorController(path, identity)
    requests = [(left, _pending(left)), (right, _pending(right))]
    barrier = threading.Barrier(2)

    def withdraw(pair):
        controller, pending = pair
        barrier.wait(timeout=5)
        try:
            return _issue(controller, pending)
        except work._SourceWorkError as exc:
            assert "stale" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(withdraw, requests))
    assert sum(result is not None for result in results) == 1
    assert _row(path) == [(1, identity, 20, 13, 1)]
    for controller, pending in requests:
        with pytest.raises(work._SourceWorkError):
            _issue(controller, pending)


def test_insufficient_and_exact_exhaustion_no_refund(tmp_path):
    path, identity, controller = _fixture(tmp_path, initial=7)
    pending = _pending(controller, amount=8)
    with pytest.raises(work._SourceWorkError, match="insufficient"):
        _issue(controller, pending)
    with pytest.raises(work._SourceWorkError):
        _issue(controller, pending)
    seat = _issue(controller, _pending(controller, amount=7))
    del seat  # Abandonment does not replenish the account.
    controller.close()
    restarted = work._OperatorController(path, identity)
    with pytest.raises(work._SourceWorkError, match="insufficient"):
        _issue(restarted, _pending(restarted, amount=1))
    assert _row(path) == [(1, identity, 7, 0, 1)]


def test_failed_seat_consumption_is_terminal_without_refund(tmp_path):
    path, _, controller = _fixture(tmp_path)
    pending = _pending(controller)
    seat = _issue(controller, pending)
    changed = list(pending)
    changed[1] = object()
    with pytest.raises(work._SourceWorkError):
        _consume(controller, seat, changed)
    with pytest.raises(work._SourceWorkError):
        _consume(controller, seat, pending)
    assert _row(path)[0][3:] == (13, 1)


def test_singleton_schema_and_corrupt_extra_row_refusal(tmp_path):
    path, identity, controller = _fixture(tmp_path)
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO account VALUES (2, ?, 20, 20, 0)",
                               ("b" * 64,))
        pending = _pending(controller)
        # Trusted-local schema tampering is not a supported custody model, but
        # the reader still refuses multiple rows rather than picking one.
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute("INSERT INTO account VALUES (2, ?, 20, 20, 0)",
                           ("b" * 64,))
        with pytest.raises(work._SourceWorkError, match="exactly one"):
            _issue(controller, pending)
    finally:
        connection.close()
    assert _row(path)[0] == (1, identity, 20, 20, 0)


@pytest.mark.parametrize("committed", [False, True])
def test_ambiguous_commit_real_sqlite_no_retry_delivery_or_refund(
        tmp_path, monkeypatch, committed):
    path, identity, controller = _fixture(tmp_path)
    pending = _pending(controller)
    connect = work._connect
    commits = []

    class CommitFault:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, parameters=()):
            if sql == "COMMIT":
                commits.append(sql)
                if committed:
                    self.connection.execute(sql)
                raise sqlite3.OperationalError("lost commit acknowledgement")
            return self.connection.execute(sql, parameters)

        @property
        def in_transaction(self):
            return self.connection.in_transaction

        def close(self):
            self.connection.close()

    with monkeypatch.context() as scoped:
        scoped.setattr(work, "_connect", lambda path: CommitFault(connect(path)))
        with pytest.raises(sqlite3.OperationalError, match="acknowledgement"):
            _issue(controller, pending)
        with pytest.raises(work._SourceWorkError):
            _issue(controller, pending)
    assert commits == ["COMMIT"]
    assert not controller._seats
    assert _row(path) == [(1, identity, 20, 13 if committed else 20,
                           1 if committed else 0)]


def test_precommit_sql_failure_rolls_back_but_burns_attempt(tmp_path):
    path, _, controller = _fixture(tmp_path)
    pending = _pending(controller)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TRIGGER refuse_debit BEFORE UPDATE ON account "
                           "BEGIN SELECT RAISE(ABORT, 'debit refused'); END")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(sqlite3.IntegrityError, match="debit refused"):
        _issue(controller, pending)
    with pytest.raises(work._SourceWorkError):
        _issue(controller, pending)
    assert _row(path)[0][3:] == (20, 0)


def test_real_sqlite_zero_row_compare_and_debit_never_delivers(tmp_path):
    path, _, controller = _fixture(tmp_path)
    pending = _pending(controller)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TRIGGER ignore_debit BEFORE UPDATE ON account "
                           "BEGIN SELECT RAISE(IGNORE); END")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(work._SourceWorkError, match="compare-and-debit"):
        _issue(controller, pending)
    assert not controller._seats
    assert _row(path)[0][3:] == (20, 0)


@pytest.mark.parametrize("mutation,match", [
    ("DELETE FROM account", "exactly one"),
    ("UPDATE account SET account_id = '" + "a" * 64 + "'", "identity"),
    ("UPDATE account SET initial = 21", "initial"),
    ("UPDATE account SET generation = generation + 1", "stale"),
    ("UPDATE account SET remaining = remaining - 1", "stale"),
])
def test_approval_binds_account_initial_generation_and_balance(
        tmp_path, mutation, match):
    path, _, controller = _fixture(tmp_path)
    pending = _pending(controller)
    connection = sqlite3.connect(path)
    try:
        connection.execute(mutation)
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(work._SourceWorkError, match=match):
        _issue(controller, pending)
    assert not controller._seats


def test_generation_exhaustion_refuses_sqlite_integer_overflow(tmp_path):
    path, _, controller = _fixture(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("UPDATE account SET generation = ?", ((1 << 63) - 1,))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(work._SourceWorkError, match="exhausted generation"):
        _issue(controller, _pending(controller))
    assert _row(path)[0][3:] == (20, (1 << 63) - 1)


def test_database_rollback_is_explicitly_not_qualified(tmp_path):
    path, identity, controller = _fixture(tmp_path, initial=7)
    backup = tmp_path / "rollback.sqlite"
    source = sqlite3.connect(path)
    target = sqlite3.connect(backup)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    _issue(controller, _pending(controller, amount=7))
    controller.close()
    # Deliberate operator-controlled offline rollback. This demonstrates the
    # limit, not an anti-rollback claim: restored generation can spend again.
    path.write_bytes(backup.read_bytes())
    restarted = work._OperatorController(path, identity)
    pending = _pending(restarted, amount=7)
    assert _consume(restarted, _issue(restarted, pending), pending) == 7
    assert _row(path) == [(1, identity, 7, 0, 1)]
