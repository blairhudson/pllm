"""Dormant private trusted-local startup withdrawal core (stdlib only).

No operational opener, CLI, import-time activation, source reader, or deployment
trust switch lives here. The operator must privately control the database and
its ancestors. Python object identity is not a sandbox against hostile code in
this interpreter. Database replacement/rollback and cross-exec use remain
UNQUALIFIED: a restored account can spend again. No restart recovers seats.

Audited shared_mpc.SQLitePreprocessingBurnLedger's lock/BEGIN IMMEDIATE/FULL
sync pattern; deliberately do not use preprocessing stock or its create-on-open
semantics. This account pays explicitly chosen integer work, never a cap-derived
estimate. Failed/abandoned seats have no refund path. COMMIT errors are terminal
even when the real commit happened; no retry, reconciliation, or seat delivery.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import sqlite3
import threading


_MAX = (1 << 63) - 1
_MAX_LENGTHS = 1024


class _SourceWorkError(RuntimeError):
    """A local withdrawal or live identity was refused."""


def _positive(value, name):
    if type(value) is not int or not 0 < value <= _MAX:
        raise _SourceWorkError(f"{name} requires a positive finite SQLite integer")
    return value


def _lengths(value):
    if type(value) is not tuple or not 0 < len(value) <= _MAX_LENGTHS:
        raise _SourceWorkError("allowed lengths require an exact bounded tuple")
    for item in value:
        _positive(item, "length")
    if len(set(value)) != len(value):
        raise _SourceWorkError("allowed lengths must be distinct")
    return value


def _path(value):
    path = Path(value)
    if not path.is_absolute():
        raise _SourceWorkError("account path must be absolute")
    return path


def _connect(path):
    # mode=rw is essential: issuance must never silently create an account.
    connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True,
                                 isolation_level=None, timeout=5)
    try:
        connection.execute("PRAGMA synchronous=FULL")
    except BaseException:
        connection.close()
        raise
    return connection


@dataclass(frozen=True, slots=True)
class _Account:
    account_id: str
    initial: int
    remaining: int
    generation: int


def _account(connection):
    # Do not select only the desired ID: a second row is corruption, not stock.
    rows = connection.execute(
        "SELECT singleton, account_id, initial, remaining, generation FROM account"
    ).fetchmany(2)
    if len(rows) != 1:
        raise _SourceWorkError("account requires exactly one row")
    singleton, identity, initial, remaining, generation = rows[0]
    if (type(singleton) is not int or singleton != 1
            or type(identity) is not str or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
            or type(initial) is not int or not 0 < initial <= _MAX
            or type(remaining) is not int or not 0 <= remaining <= initial
            or type(generation) is not int or not 0 <= generation <= _MAX):
        raise _SourceWorkError("invalid finite singleton account")
    return _Account(identity, initial, remaining, generation)


def _provision_account(path, initial_work):
    """Explicit provisioning only; never overwrites or adopts an existing file.

    A provisioning failure may leave a file requiring operator investigation;
    it is never automatically removed, replenished, or adopted.
    """
    initial = _positive(initial_work, "initial work")
    path = _path(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE account ("
            "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
            "account_id TEXT NOT NULL UNIQUE, "
            "initial INTEGER NOT NULL CHECK(typeof(initial) = 'integer' AND initial > 0), "
            "remaining INTEGER NOT NULL CHECK(typeof(remaining) = 'integer' "
            "AND remaining >= 0 AND remaining <= initial), "
            "generation INTEGER NOT NULL CHECK(typeof(generation) = 'integer' "
            "AND generation >= 0))"
        )
        identity = secrets.token_hex(32)
        connection.execute("INSERT INTO account VALUES (1, ?, ?, ?, 0)",
                           (identity, initial, initial))
        connection.execute("COMMIT")
        if connection.in_transaction:
            raise _SourceWorkError("provisioning commit not confirmed")
        return identity
    finally:
        connection.close()


class _Uncopyable:
    __slots__ = ()

    def __copy__(self):
        raise TypeError("live operator objects cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("live operator objects cannot be copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("live operator objects cannot be serialized")


class _Handle(_Uncopyable):
    """Opaque identity only; all authority stays in the owning controller."""
    __slots__ = ()

    def __new__(cls):
        raise TypeError("only the live operator controller can mint handles")


class _Launch(_Handle):
    __slots__ = ()


class _Approval(_Handle):
    __slots__ = ()


class _Seat(_Handle):
    __slots__ = ()


class _OperatorController(_Uncopyable):
    """Live operator custody, not caller-provided actor/cap authority.

    The parent passes a fresh object() as its exact launch identity. One launch
    and one pending approval at a time; separate controllers may race on SQLite.
    Handles are registry-checked by identity, never by supplied attributes.
    Every public operation checks PID before touching a possibly inherited lock.
    """
    __slots__ = ("_pid", "_lock", "_path", "_identity", "_initial", "_launch",
                 "_pending", "_seats", "_closed")

    def __init__(self, path, account_id):
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._path = _path(path)
        if type(account_id) is not str:
            raise _SourceWorkError("requires provisioned account identity")
        connection = _connect(self._path)
        try:
            account = _account(connection)
        finally:
            connection.close()
        if account.account_id != account_id:
            raise _SourceWorkError("wrong account identity")
        self._identity, self._initial = account_id, account.initial
        self._launch = self._pending = None
        self._seats = {}
        self._closed = False

    def _process(self):
        if os.getpid() != self._pid:
            raise _SourceWorkError("operator controller belongs to another process")

    def _live(self):
        if self._closed:
            raise _SourceWorkError("operator controller is closed")

    def _read(self, connection):
        account = _account(connection)
        if (account.account_id, account.initial) != (self._identity, self._initial):
            raise _SourceWorkError("account identity or initial work changed")
        return account

    def launch(self, parent, allowed_lengths):
        self._process()
        with self._lock:
            self._live()
            if type(parent) is not object:
                raise _SourceWorkError("parent requires a fresh object identity")
            lengths = _lengths(allowed_lengths)
            if self._launch is not None:
                raise _SourceWorkError("launch already live")
            launch = object.__new__(_Launch)
            self._launch = (launch, parent, lengths)
            return launch

    def approve(self, launch, parent, allowed_lengths, chosen_work):
        self._process()
        with self._lock:
            self._live()
            lengths = _lengths(allowed_lengths)
            work = _positive(chosen_work, "chosen work")
            if (self._launch is None or self._launch[0] is not launch
                    or self._launch[1] is not parent or self._launch[2] != lengths):
                raise _SourceWorkError("approval requires exact parent launch and lengths")
            if self._pending is not None:
                raise _SourceWorkError("approval already pending")
            connection = _connect(self._path)
            try:
                account = self._read(connection)
            finally:
                connection.close()
            approval = object.__new__(_Approval)
            self._pending = (approval, launch, parent, lengths, work, account)
            return approval

    def issue(self, launch, parent, approval, allowed_lengths, chosen_work):
        self._process()
        with self._lock:
            self._live()
            # Attempt and approval burn BEFORE validation, opening SQLite, or SQL.
            # Even mismatched submissions terminate this pending launch.
            live, pending = self._launch, self._pending
            self._launch = self._pending = None
            lengths = _lengths(allowed_lengths)
            work = _positive(chosen_work, "chosen work")
            if (live is None or pending is None or live[0] is not launch
                    or live[1] is not parent or live[2] != lengths
                    or pending[0] is not approval or pending[1] is not launch
                    or pending[2] is not parent or pending[3] != lengths
                    or pending[4] != work):
                raise _SourceWorkError("withdrawal requires exact pending approval")
            expected = pending[5]
            connection = _connect(self._path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                account = self._read(connection)
                if account != expected:
                    raise _SourceWorkError("stale account generation or balance")
                if account.remaining < work or account.generation == _MAX:
                    raise _SourceWorkError("insufficient work or exhausted generation")
                cursor = connection.execute(
                    "UPDATE account SET remaining = remaining - ?, generation = generation + 1 "
                    "WHERE singleton = 1 AND account_id = ? AND initial = ? "
                    "AND generation = ? AND remaining = ? AND remaining >= ?",
                    (work, self._identity, self._initial, account.generation,
                     account.remaining, work))
                if cursor.rowcount != 1:
                    raise _SourceWorkError("account generation compare-and-debit failed")
                connection.execute("COMMIT")
                if connection.in_transaction:
                    raise _SourceWorkError("withdrawal commit not confirmed")
            finally:
                # Closing an uncommitted transaction rolls it back. If COMMIT
                # raised after committing, closing does not refund that debit.
                # Never catch a commit error to retry or mint a seat.
                connection.close()
            seat = object.__new__(_Seat)
            self._seats[seat] = (launch, parent, lengths, work)
            return seat

    def consume(self, seat, launch, parent, allowed_lengths, chosen_work):
        """Burn one seat, returning only its explicitly approved work count."""
        self._process()
        with self._lock:
            self._live()
            binding = self._seats.pop(seat, None) if type(seat) is _Seat else None
            lengths = _lengths(allowed_lengths)
            work = _positive(chosen_work, "chosen work")
            if (binding is None or binding[0] is not launch
                    or binding[1] is not parent or binding[2] != lengths
                    or binding[3] != work):
                raise _SourceWorkError("seat requires exact same-process launch binding")
            return work

    def close(self):
        self._process()
        with self._lock:
            self._closed = True
            self._launch = self._pending = None
            self._seats.clear()
