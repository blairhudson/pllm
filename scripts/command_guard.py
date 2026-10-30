#!/usr/bin/env python3
"""Cooperative command monitor, NOT containment or an RSS rlimit.

Caller supplies scrubbed env, ordinary stdout/stderr redirection, all writable
roots and the SAME absolute campaign deadline for every command. No retries.
Unknown cleanup retains the canonical coordinator slot: human review required.
Before authority is validated no slot may be created: failure returns 78 and
requires caller HOLD instead. Never mistake that state for an acquired lock.
Short-lived forks/escapes/peaks and between-sample file mutations can be missed.
Arbitrary hostile/daemonizing payloads are unsupported. Source/env/toolchain
pinning and exclusion of rogue external jobs remain caller responsibilities.
Signalling/reaps are bounded; final authority-only filesystem operations can
block with the slot held. There is NO hard full-tool wall-time/OS containment.
Stable lexical aliases (including macOS /var -> /private/var) are pinned, not
whitelisted. Git administration redirects and changes to pinned paths fail.
Campaign init/retire are explicit parent operations, not authorization proof.
--campaign-init requires --campaign-deadline and refuses ANY existing lease;
--campaign-retire takes the same exclusive slot and may retire an expired
same-boot lease. Neither accepts payload argv. Retirement's 30-second admin
control window never creates/extends a campaign. Existing slots need review.
One fixed budget persists cumulative positive observed growth across commands.
Every budget rewrite charges 2 MiB, every slot 64 KiB, and initialization also
prepays 2 MiB for campaign publication. These costs never refund; exhaustion or
an oversized census holds rather than increasing limits. No cache-fit claim.
Fork bootstrap requires standalone single-thread CLI/exclusive reaping. Kernel
fork/syscall scheduling stalls remain outside hard guarantees. READY/GO/exec
EOF never substitute for birth identity or qualified descendant cleanup.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import stat as statmod
import subprocess
import sys
import time
import threading

sys.dont_write_bytecode = True  # monitor imports must not create source-tree caches
try:
    from . import host_swap
except ImportError:
    import host_swap

GIB = 1 << 30
MAX_RESPONSE = 1 << 20


class Blocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    campaign_deadline: float
    seconds: float = 600
    rss: int = 8 * GIB
    reserve: int = 16 * GIB
    disk_growth: int = 20 * GIB
    interval: float = 0.1
    probe_timeout: float = 2
    grace: float = 1
    reap_timeout: float = 1

    @property
    def cleanup_budget(self):
        return 2 * self.probe_timeout + 4 * self.grace + self.reap_timeout

    def validate(self, now):
        values = tuple(self.__dict__.values())
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values):
            raise Blocked("invalid bounds")
        if (self.seconds > 600 or self.rss > 8 * GIB or self.reserve < 16 * GIB
                or self.disk_growth > 20 * GIB or self.interval > 0.1
                or self.probe_timeout > 2 or self.grace != 1 or self.reap_timeout > 1
                or not self.cleanup_budget < self.campaign_deadline - now <= 7200):
            raise Blocked("bounds may tighten only; campaign needs cleanup time")


def identity(stat):
    birth = getattr(stat, "st_birthtime_ns", None)
    if birth is None:
        birth = getattr(stat, "st_birthtime", None)
    if type(birth) not in (int, float) or not math.isfinite(birth) or birth <= 0:
        birth = None
    return stat.st_dev, stat.st_ino, birth


class Pins:
    """Lexical ancestry AND resolved identities, independent of walk-root dedup."""
    def __init__(self, entries=None):
        self.entries = dict(entries or [])

    @staticmethod
    def token(path):
        s = path.lstat()
        kind = statmod.S_IFMT(s.st_mode)
        return [*identity(s), kind, s.st_uid,
                os.readlink(path) if statmod.S_ISLNK(s.st_mode) else None,
                [s.st_size, s.st_mtime_ns, s.st_ctime_ns] if statmod.S_ISREG(s.st_mode) else None,
                statmod.S_IMODE(s.st_mode)]

    def add(self, path):
        path = Path(path).absolute()
        if ".." in path.parts:
            # Pin prefixes BEFORE traversing '..', including any alias whose
            # target determines the real parent. Do not erase with normpath.
            current = Path(path.anchor)
            for part in path.parts[1:]:
                current = self.add(current).parent if part == ".." else current / part
            return self.add(current)
        for item in (*reversed(path.parents), path):
            key, token = str(item), self.token(item)
            if key in self.entries and self.entries[key] != token:
                raise Blocked(f"authority changed while pinning: {item}")
            self.entries[key] = token
        resolved = path.resolve(strict=True)
        if resolved != path:
            self.add(resolved)
        self.check()
        return resolved

    def check(self):
        for path, token in self.entries.items():
            if self.token(Path(path)) != token:
                raise Blocked(f"pinned ancestry/root identity changed: {path}")

    def wire(self):
        return list(self.entries.items())


def owned_file(s):
    if (not statmod.S_ISREG(s.st_mode) or s.st_uid != os.getuid()
            or s.st_nlink != 1 or statmod.S_IMODE(s.st_mode) != 0o600):
        raise Blocked("foreign/uncertain operational file ownership")
    return [*identity(s), s.st_uid, s.st_mode, s.st_nlink, s.st_size, s.st_ctime_ns]


def boot_identity():
    """Worker-only Darwin kern.boottime; never use a caller-supplied boot label."""
    import ctypes
    class Timeval(ctypes.Structure):
        _fields_ = [("seconds", ctypes.c_long), ("microseconds", ctypes.c_int)]
    value = Timeval()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    call = library.sysctlbyname
    call.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                     ctypes.c_void_p, ctypes.c_size_t]
    call.restype = ctypes.c_int
    if (call(b"kern.boottime", ctypes.byref(value), ctypes.byref(size), None, 0) != 0
            or size.value != ctypes.sizeof(value) or value.seconds <= 0
            or not 0 <= value.microseconds < 1000000):
        raise Blocked("OS boot identity unavailable/inconsistent")
    return [value.seconds, value.microseconds]


CAMPAIGN = "native-command.campaign"
SLOT_NAME = "native-command.slot"
BUDGET = "native-command.budget"
# Per MUTATION, not a live reservation. Every budget write pays its own charge
# in that very write. Logical AND allocated footprint must fit these bounds.
STATE_LIMIT = MAX_RESPONSE - 65536
WRITE_CHARGE = 2 * MAX_RESPONSE
SLOT_CHARGE = 65536
CAMPAIGN_CHARGE = WRITE_CHARGE


def footprint(s):
    if any(type(v) is not int or v < 0 for v in (s.st_size, s.st_blocks)):
        raise Blocked("unknown allocation")
    return max(s.st_size, s.st_blocks * 512)


def encode_state(record):
    raw = json.dumps(record, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > STATE_LIMIT:
        raise Blocked("bounded operational state exhausted")
    return raw


def decode_state(raw):
    if len(raw) > STATE_LIMIT:
        raise Blocked("oversize operational state")
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise Blocked("duplicate operational state field")
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(Blocked(v)))


def valid_identity(key):
    return (isinstance(key, list) and len(key) == 3
            and all(type(v) is int and v >= 0 for v in key[:2])
            and type(key[2]) in (int, float) and math.isfinite(key[2]) and key[2] > 0)


def validate_manifest(manifest):
    if (not isinstance(manifest, dict) or set(manifest) != {"declarations", "pins"}
            or not isinstance(manifest["declarations"], list) or not manifest["declarations"]
            or not isinstance(manifest["pins"], list) or not manifest["pins"]
            or len(encode_state(manifest)) > MAX_RESPONSE // 4):
        raise Blocked("invalid/oversize original root manifest")
    paths, pinned = set(), set()
    for item in manifest["declarations"]:
        if (not isinstance(item, list) or len(item) != 3
                or any(not isinstance(p, str) or not Path(p).is_absolute() for p in item[:2])
                or item[0] in paths or not valid_identity(item[2])):
            raise Blocked("invalid/duplicate root declaration")
        paths.add(item[0])
    for item in manifest["pins"]:
        if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
                or not Path(item[0]).is_absolute() or item[0] in pinned
                or not isinstance(item[1], list) or len(item[1]) != 8):
            raise Blocked("invalid/duplicate root ancestry")
        token = item[1]
        if (any(type(token[i]) is not int or token[i] < 0 for i in (0, 1, 3, 4, 7))
                or token[3] not in (statmod.S_IFDIR, statmod.S_IFLNK)
                or (token[2] is not None and (type(token[2]) not in (int, float)
                    or not math.isfinite(token[2]) or token[2] <= 0))
                or (not isinstance(token[5], str) if token[3] == statmod.S_IFLNK else token[5] is not None)
                or token[6] is not None or token[7] > 0o7777):
            raise Blocked("invalid original root ancestry token")
        pinned.add(item[0])
    tokens = dict(manifest["pins"])
    for _, resolved, key in manifest["declarations"]:
        if resolved not in tokens or tokens[resolved][:3] != key or tokens[resolved][3] != statmod.S_IFDIR:
            raise Blocked("root declaration lacks original directory pin")


def validate_campaign(record, boot, now, requested=None, expired=False, pending=False):
    if (not isinstance(boot, list) or len(boot) != 2
            or any(type(v) is not int for v in boot) or boot[0] <= 0 or not 0 <= boot[1] < 1000000
            or not isinstance(record, dict) or set(record) != {
                "boot", "origin", "deadline", "disk_cap_bytes", "budget_identity", "roots"}
            or not isinstance(record["boot"], list) or any(type(v) is not int for v in record["boot"])
            or record["boot"] != boot
            or any(type(record[k]) not in (int, float) or not math.isfinite(record[k])
                   for k in ("origin", "deadline"))
            or not 0 < record["origin"] <= now
            or not 0 < record["deadline"] - record["origin"] <= 7200
            or (not expired and now >= record["deadline"])
            or (requested is not None and requested != record["deadline"])):
        raise Blocked("campaign boot/origin/deadline invalid, expired, or substituted")
    if (type(record["disk_cap_bytes"]) is not int or not 0 < record["disk_cap_bytes"] <= 20 * GIB
            or not (valid_identity(record["budget_identity"])
                    or pending and record["budget_identity"] is None)):
        raise Blocked("campaign cap/budget identity invalid")
    validate_manifest(record["roots"])


def read_campaign(fd, expected=None):
    child = os.open(CAMPAIGN, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        token = owned_file(os.fstat(child))
        raw = os.read(child, STATE_LIMIT + 1)
        record = decode_state(raw)
        if footprint(os.fstat(child)) > CAMPAIGN_CHARGE:
            raise Blocked("campaign allocation exceeds paid bound")
        if owned_file(os.fstat(child)) != token or owned_file(os.stat(
                CAMPAIGN, dir_fd=fd, follow_symlinks=False)) != token:
            raise Blocked("campaign replaced during read")
        if expected is not None and token != expected:
            raise Blocked("campaign identity changed")
        return record, token
    finally:
        os.close(child)


def absent(fd, name):
    try:
        os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise Blocked(f"existing {name}; no automatic adoption/removal")


def disk_roots(paths):
    roots = sorted({Path(p).resolve(strict=True) for p in paths},
                   key=lambda p: (len(p.parts), str(p)))
    kept = []
    for root in roots:
        if root.parent == root or not root.is_dir():
            raise Blocked("filesystem root/non-directory cannot be a disk root")
        if not any(root.is_relative_to(parent) for parent in kept):
            kept.append(root)
    return tuple(kept)


class ObservedDiskGrowth:
    """Cumulative POSITIVE observed deltas, never a refundable net-size meter."""
    def __init__(self):
        self.previous = None
        self.total = 0

    def observe(self, files):
        # identity -> (size, ctime_ns). Hardlinks/renames share one identity.
        if self.previous is not None:
            for key, (size, stamp) in files.items():
                old = self.previous.get(key)
                if old is None or key[2] is None:
                    self.total += size  # new, recreated, or uncertain identity
                else:
                    self.total += max(0, size - old[0])
        self.previous = files.copy()
        return self.total


class Disk:
    def __init__(self, paths):
        self.roots = disk_roots(paths)
        self.identities = {root: identity(root.stat()) for root in self.roots}
        if any(key[2] is None for key in self.identities.values()):
            raise Blocked("disk root birth identity unavailable")
        self.meter = ObservedDiskGrowth()

    def census(self, excluded=()):
        files = {}
        def visit(root):
            with os.scandir(root) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        dest = Path(entry.path).resolve(strict=True)
                        if not any(dest.is_relative_to(p) for p in self.roots):
                            raise Blocked(f"declare outward symlink destination root: {dest}")
                        continue  # destination scanned under declared root
                    stat = entry.stat(follow_symlinks=False)
                    if entry.is_dir(follow_symlinks=False):
                        visit(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        key = identity(stat)
                        if key in excluded:
                            continue  # exact verified owned inode, NEVER a path exemption
                        value = footprint(stat), stat.st_ctime_ns
                        if key in files and files[key] != value:
                            raise Blocked("file changed during disk census")
                        files[key] = value
                    else:
                        raise Blocked(f"unmetered special file: {entry.path}")
        for root in self.roots:
            if root.is_symlink() or identity(root.stat()) != self.identities[root]:
                raise Blocked("disk root replacement/identity unknown")
            visit(root)
            if identity(root.stat()) != self.identities[root]:
                raise Blocked("disk root changed during census")
        return files

    def snapshot(self):
        return self.meter.observe(self.census())


def canonical_slot(cwd, pins=None):
    """Ordinary .git/commondir derivation: Main and linked worktrees share slot."""
    pins = pins if pins is not None else Pins()
    resolved = pins.add(cwd)
    for base in (resolved, *resolved.parents):
        dotgit = base / ".git"
        if dotgit.is_symlink():
            raise Blocked("symlinked Git authority requires separate review")
        if dotgit.is_file():
            pins.add(dotgit)
            text = dotgit.read_text().strip()
            if not text.startswith("gitdir: "):
                raise Blocked("unknown gitdir format")
            git = pins.add(base / text[8:])
            linked = True
        elif dotgit.is_dir():
            git = pins.add(dotgit)
            linked = False
        else:
            continue
        admin = git
        common = admin / "commondir"
        if linked:
            pins.add(common)
            git = pins.add(admin / common.read_text().strip())
            pins.add(admin / "gitdir")
            backlink = (admin / "gitdir").read_text().strip()
            if admin.parent != git / "worktrees" or pins.add(admin / backlink) != dotgit.resolve(strict=True):
                raise Blocked("linked worktree/common-root authority mismatch")
        elif common.exists():
            raise Blocked("unexpected Main commondir")
        if (git.name != ".git" or not git.is_dir() or not (git / "HEAD").is_file()
                or not (git / "objects").is_dir()):
            raise Blocked("nonstandard common Git root needs separate review")
        for path in (git, git / "HEAD", git / "objects"):
            pins.add(path)
        for path, token in pins.entries.items():
            if token[3] == statmod.S_IFLNK and Path(path).parent.resolve().is_relative_to(git):
                raise Blocked("Git administration symlink refused")
        return git.parent / "target/native-command.slot"
    raise Blocked("canonical common Git root missing")


def write_roots(args, env):
    if any(key in env for key in ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE")):
        raise Blocked("Git environment authority redirects refused")
    for key, path in (("TMPDIR", args.temp_dir), ("CARGO_TARGET_DIR", args.target_dir)):
        if not env.get(key) or Path(env[key]).resolve() != path.resolve():
            raise Blocked(f"caller must set {key} to declared path")
    paths = [args.target_dir, args.log_dir, args.temp_dir, *args.disk_root,
             Path(env["TMPDIR"]), Path(env["CARGO_TARGET_DIR"])]
    defaults = {"CARGO_HOME": Path.home() / ".cargo", "RUSTUP_HOME": Path.home() / ".rustup",
                "UV_CACHE_DIR": Path.home() / "Library/Caches/uv"}
    paths += [Path(env.get(key, default)) for key, default in defaults.items()]
    paths += [Path(env[key]) for key in ("TMP", "TEMP", "UV_BUILD_DIR") if env.get(key)]
    return paths


def campaign_binding(record):
    return {key: record[key] for key in ("boot", "origin", "deadline", "disk_cap_bytes")}


def validate_budget(record, campaign, manifest):
    if (not isinstance(record, dict) or set(record) != {
            "version", "campaign", "roots", "sequence", "charged_bytes", "previous"}
            or type(record["version"]) is not int or record["version"] != 1
            or record["campaign"] != campaign_binding(campaign)
            or record["roots"] != manifest
            or type(record["sequence"]) is not int or record["sequence"] < 1
            or type(record["charged_bytes"]) is not int
            or not (record["sequence"] * WRITE_CHARGE + CAMPAIGN_CHARGE + SLOT_CHARGE
                    <= record["charged_bytes"] <= campaign["disk_cap_bytes"])
            or not isinstance(record["previous"], list)):
        raise Blocked("budget fields/campaign/roots/cap invalid")
    binding = record["campaign"]
    if (not isinstance(binding, dict) or set(binding) != {"boot", "origin", "deadline", "disk_cap_bytes"}
            or not isinstance(binding["boot"], list) or len(binding["boot"]) != 2
            or any(type(v) is not int for v in binding["boot"])
            or type(binding["disk_cap_bytes"]) is not int
            or any(type(binding[k]) not in (int, float) or not math.isfinite(binding[k])
                   for k in ("origin", "deadline"))):
        raise Blocked("budget campaign binding types invalid")
    validate_manifest(record["roots"])
    files = {}
    for item in record["previous"]:
        if (not isinstance(item, list) or len(item) != 5
                or any(type(item[i]) is not int or item[i] < 0 for i in (0, 1, 3, 4))
                or (item[2] is not None and (type(item[2]) not in (int, float)
                    or not math.isfinite(item[2]) or item[2] <= 0))):
            raise Blocked("invalid persisted census")
        key = tuple(item[:3])
        if key in files:
            raise Blocked("duplicate persisted census identity")
        files[key] = tuple(item[3:])
    return files


def read_budget(directory, campaign, manifest, expected=None):
    fd = os.open(BUDGET, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    try:
        token = owned_file(os.fstat(fd))
        if list(identity(os.fstat(fd))) != campaign["budget_identity"]:
            raise Blocked("budget inode/birth differs from immutable campaign")
        raw = os.read(fd, STATE_LIMIT + 1)
        record = decode_state(raw)
        validate_budget(record, campaign, manifest)
        if (owned_file(os.fstat(fd)) != token
                or owned_file(os.stat(BUDGET, dir_fd=directory, follow_symlinks=False)) != token
                or expected is not None and token != expected
                or footprint(os.fstat(fd)) > WRITE_CHARGE):
            raise Blocked("budget allocation/content identity changed")
        return record, token
    finally:
        os.close(fd)


class Budget:
    """One stable-inode mutable operational debit, not receipts or grant proof."""
    def __init__(self, authority, fd, record=None, writable=True):
        self.authority, self.fd = authority, fd
        self.writable = writable
        self.stable = list(identity(os.fstat(fd)))
        if not valid_identity(self.stable):
            raise Blocked("budget birth identity unknown")
        self.meter = ObservedDiskGrowth()
        self.record = record
        self.sequence = 0 if record is None else record["sequence"]
        if record is not None:
            self.meter.previous = validate_budget(record, authority.campaign, authority.manifest)
            self.meter.total = record["charged_bytes"]

    @classmethod
    def load(cls, authority):
        record, token = read_budget(authority.fd, authority.campaign, authority.manifest)
        fd = os.open(BUDGET, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=authority.fd)
        try:
            if owned_file(os.fstat(fd)) != token:
                raise Blocked("budget changed while opening owned FD")
            result = cls(authority, fd, record, False)
            result.verify()
            return result
        except Exception:
            os.close(fd)
            raise

    def funded(self, extra):
        if self.meter.total + extra > self.authority.cap:
            raise Blocked("campaign cumulative disk budget cannot fund next mutation; HOLD")

    def verify(self):
        if list(identity(os.fstat(self.fd))) != self.stable:
            raise Blocked("owned budget FD identity changed")
        record, token = read_budget(self.authority.fd, self.authority.campaign,
                                    self.authority.manifest)
        if record != self.record or owned_file(os.fstat(self.fd)) != token:
            raise Blocked("budget replaced/changed or persistence ambiguous")
        return token

    def commit(self, files, extra=0):
        if self.authority.owned_slot is None:
            raise Blocked("no budget mutation without exclusively owned original slot")
        if self.record is not None:
            token = self.verify()
            if not self.writable:
                fd = os.open(BUDGET, os.O_RDWR | os.O_NOFOLLOW, dir_fd=self.authority.fd)
                try:
                    if owned_file(os.fstat(fd)) != token:
                        raise Blocked("budget changed at writable-FD upgrade")
                    os.close(self.fd)
                except Exception:
                    os.close(fd)
                    raise
                self.fd, self.writable = fd, True
        before = footprint(os.fstat(self.fd))
        if before > WRITE_CHARGE:
            raise Blocked("budget allocation before write exceeds bound")
        self.meter.observe(files)
        self.funded(WRITE_CHARGE + extra)  # NEVER an unfunded refusal write
        charged = self.meter.total + WRITE_CHARGE + extra
        record = dict(version=1, campaign=campaign_binding(self.authority.campaign),
                      roots=self.authority.manifest, sequence=self.sequence + 1,
                      charged_bytes=charged,
                      previous=[[*key, *value] for key, value in sorted(
                          files.items(), key=lambda item: repr(item[0]))])
        raw = encode_state(record)
        self.authority.check_directory()
        self.authority.check_slot()
        current = owned_file(os.fstat(self.fd))
        if (list(identity(os.fstat(self.fd))) != self.stable
                or owned_file(os.stat(BUDGET, dir_fd=self.authority.fd,
                                      follow_symlinks=False)) != current):
            raise Blocked("budget changed at write boundary")
        os.lseek(self.fd, 0, os.SEEK_SET)
        if os.write(self.fd, raw) != len(raw):  # ONE overwrite, no retries
            raise Blocked("partial budget write; HOLD even if old record readable")
        if os.fstat(self.fd).st_size > len(raw):
            os.ftruncate(self.fd, len(raw))  # trailing truncate only
        os.fsync(self.fd)
        after = footprint(os.fstat(self.fd))
        if after > WRITE_CHARGE or max(0, after - before) > WRITE_CHARGE:
            raise Blocked("budget allocation exceeds paid mutation bound")
        os.lseek(self.fd, 0, os.SEEK_SET)
        if os.read(self.fd, STATE_LIMIT + 1) != raw:
            raise Blocked("budget complete-content readback failed")
        self.record, self.sequence, self.meter.total = record, record["sequence"], charged
        self.verify()
        return charged

    def proof(self):
        return dict(identity=self.stable, token=self.verify(), sequence=self.sequence,
                    charged_bytes=self.record["charged_bytes"])


class Authority:
    """Worker-owned runtime authority. No caller argument selects trust anchor."""
    def __init__(self, args, env, boot, now):
        self.pins = Pins()
        script = self.pins.add(Path(__file__).absolute())
        self.slot = canonical_slot(script.parent, self.pins)
        coordinator = self.pins.add(args.coordinator_root)
        if (canonical_slot(args.cwd, self.pins) != self.slot
                or canonical_slot(coordinator, self.pins) != self.slot
                or coordinator != self.slot.parent.parent):
            raise Blocked("caller repository differs from trusted script common-dir")
        target = self.slot.parent
        if self.pins.add(target) != target or args.slot.absolute() != self.slot:
            raise Blocked("alternate/redirected coordinator storage refused")
        self.paths = [target, *write_roots(args, env)]
        roots, declarations = Pins(), {}
        for path in self.paths:
            resolved = self.pins.add(path)  # ALL declarations, BEFORE deduplication
            roots.add(path)
            declarations[str(Path(path).absolute())] = [str(Path(path).absolute()),
                                                       str(resolved), list(identity(resolved.stat()))]
            if not resolved.is_dir():
                raise Blocked("declared root is not a directory")
        self.manifest = dict(declarations=[declarations[p] for p in sorted(declarations)],
                             pins=[[path, token] for path, token in sorted(roots.wire())])
        validate_manifest(self.manifest)
        for path in (args.target_dir, args.log_dir, args.temp_dir):
            if not path.resolve(strict=True).is_relative_to(target):
                raise Blocked("target/log/temp must be under canonical ignored target/")
        self.directory = identity(target.stat())
        storage = target.stat()
        if (self.directory[2] is None or storage.st_uid != os.getuid()
                or statmod.S_IMODE(storage.st_mode) & 0o022):
            raise Blocked("coordinator directory birth/ownership/permissions uncertain")
        self.fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.boot = boot
        self.operation = "init" if args.campaign_init else "retire" if args.campaign_retire else "run"
        self.requested = args.campaign_deadline
        self.owned_slot = None
        self.budget = None
        self.pending_slot = 0
        self.cap = args.disk_growth_bytes
        self.campaign_token = None
        try:
            self.check_directory()
            if self.operation == "init":
                absent(self.fd, CAMPAIGN)
                absent(self.fd, BUDGET)
                self.campaign = dict(boot=boot, origin=now, deadline=self.requested,
                                     disk_cap_bytes=self.cap, budget_identity=None, roots=self.manifest)
            else:
                self.campaign, self.campaign_token = read_campaign(self.fd)
                validate_campaign(self.campaign, boot, now, self.requested,
                                  expired=self.operation == "retire")
                if self.campaign["roots"] != self.manifest or not 0 < self.cap <= self.campaign["disk_cap_bytes"]:
                    raise Blocked("original roots/cap mismatch; no root stripping or cap promotion")
                self.budget = Budget.load(self)
            self.check(now)
        except Exception:
            self.close()
            raise

    def close(self):
        try:
            if self.budget is not None:
                os.close(self.budget.fd)
        finally:
            os.close(self.fd)

    def check_directory(self):
        self.pins.check()
        if identity(os.fstat(self.fd)) != self.directory:
            raise Blocked("pinned coordinator directory changed")

    def check(self, now):
        self.check_directory()
        if self.operation == "init":
            absent(self.fd, CAMPAIGN)
            if self.budget is None:
                absent(self.fd, BUDGET)
        else:
            record, _ = read_campaign(self.fd, self.campaign_token)
            if record != self.campaign:
                raise Blocked("campaign contents changed")
        validate_campaign(self.campaign, self.boot, now, self.requested,
                          expired=self.operation == "retire", pending=self.budget is None)
        if self.budget is not None:
            self.budget.verify()
        self.check_slot()

    def check_slot(self):
        if self.owned_slot is not None:
            if owned_file(os.stat(SLOT_NAME, dir_fd=self.fd, follow_symlinks=False)) != self.owned_slot:
                raise Blocked("owned slot replaced/changed")
            if footprint(os.stat(SLOT_NAME, dir_fd=self.fd, follow_symlinks=False)) > SLOT_CHARGE:
                raise Blocked("slot allocation exceeds paid bound")

    def acquire(self, now):
        self.check(now)  # immediate mutation-boundary check, not stale census
        if self.budget is not None:
            self.budget.funded(SLOT_CHARGE + WRITE_CHARGE)
        elif self.cap < SLOT_CHARGE + WRITE_CHARGE + CAMPAIGN_CHARGE:
            raise Blocked("init cannot fund control mutations")
        child = os.open(SLOT_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600, dir_fd=self.fd)
        try:
            text = b"command guard owns canonical slot; retained = HOLD; review before reuse\n"
            if footprint(os.fstat(child)) > SLOT_CHARGE:
                raise Blocked("slot initial allocation exceeds charged bound")
            if os.write(child, text) != len(text):
                raise Blocked("partial slot write")
            self.owned_slot = owned_file(os.fstat(child))
            if footprint(os.fstat(child)) > SLOT_CHARGE:
                raise Blocked("slot allocation exceeds charged bound")
            self.pending_slot = SLOT_CHARGE
        finally:
            os.close(child)
        self.check(now)

    def custody(self):
        return dict(pins=self.pins.wire(), slot=str(self.slot), directory=list(self.directory),
                    owned_slot=self.owned_slot, campaign=self.campaign,
                    campaign_token=self.campaign_token, operation=self.operation,
                    budget=None if self.budget is None else self.budget.proof())

    def excluded(self):
        self.check_directory()
        result = set()
        if self.owned_slot is not None:
            result.add(tuple(self.owned_slot[:3]))
        if self.campaign_token is not None:
            result.add(tuple(self.campaign_token[:3]))
        if self.budget is not None:
            result.add(tuple(self.budget.stable))
        return result

    def debit(self, files, baseline):
        if self.owned_slot is None:
            if self.budget is None:
                return 0
            self.budget.meter.observe(files)
            self.budget.funded(SLOT_CHARGE + WRITE_CHARGE)
            return self.budget.meter.total
        extra = self.pending_slot
        self.check_directory()
        self.check_slot()
        if self.budget is None:
            absent(self.fd, CAMPAIGN)
            absent(self.fd, BUDGET)
            meter = ObservedDiskGrowth()
            meter.previous = baseline
            meter.observe(files)
            if meter.total + extra + WRITE_CHARGE + CAMPAIGN_CHARGE > self.cap:
                raise Blocked("unfunded initial budget creation")
            fd = os.open(BUDGET, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=self.fd)
            try:
                self.budget = Budget(self, fd)
            except Exception:
                os.close(fd)
                raise
            self.budget.meter.previous = baseline
            self.campaign["budget_identity"] = self.budget.stable
            extra += CAMPAIGN_CHARGE  # eventual immutable publication prepaid
        total = self.budget.commit(files, extra)
        self.pending_slot = 0
        return total


def finish_authority(custody, slot, now, commit=True):
    """Post-qualified-shutdown ONLY. No resource census, recursion or subprocess.

    Metadata/unlink can block in the kernel. Cooperative writers are required:
    POSIX has no atomic unlink-if-inode primitive against a malicious replacer.
    """
    clock = now if callable(now) else lambda: now
    if Path(custody["slot"]) != slot:
        raise Blocked("release slot authority mismatch")
    pins = Pins(custody["pins"])
    pins.check()
    fd = os.open(slot.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        def check():
            pins.check()
            if list(identity(os.fstat(fd))) != custody["directory"]:
                raise Blocked("release directory identity changed")
            token = owned_file(os.stat(SLOT_NAME, dir_fd=fd, follow_symlinks=False))
            if token != custody["owned_slot"]:
                raise Blocked("foreign/replaced slot: never unlink")
        check()
        operation, record = custody["operation"], custody["campaign"]
        def check_time():
            validate_campaign(record, record["boot"], clock(), expired=operation == "retire")
        check_time()
        proof = custody["budget"]
        if proof is None or proof["identity"] != record["budget_identity"]:
            raise Blocked("missing original budget custody")
        def check_budget():
            persisted, _ = read_budget(fd, record, record["roots"], proof["token"])
            if (persisted["sequence"] != proof["sequence"]
                    or persisted["charged_bytes"] != proof["charged_bytes"]):
                raise Blocked("final debit not persisted; HOLD")
        check_budget()
        if operation == "init":
            absent(fd, CAMPAIGN)
            if not commit:
                raise Blocked("partial initialization retains budget and slot")
            if commit:
                check()
                check_budget()
                check_time()
                child = os.open(CAMPAIGN, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600, dir_fd=fd)
                try:
                    raw = encode_state(record)
                    if footprint(os.fstat(child)) > CAMPAIGN_CHARGE:
                        raise Blocked("campaign initial allocation exceeds prepaid bound")
                    if os.write(child, raw) != len(raw):
                        raise Blocked("partial campaign initialization; HOLD")
                    created = owned_file(os.fstat(child))
                    os.fsync(child)
                    if footprint(os.fstat(child)) > CAMPAIGN_CHARGE:
                        raise Blocked("campaign allocation exceeds prepaid bound")
                    if owned_file(os.stat(CAMPAIGN, dir_fd=fd, follow_symlinks=False)) != created:
                        raise Blocked("campaign replaced during initialization")
                finally:
                    os.close(child)
        else:
            current, _ = read_campaign(fd, custody["campaign_token"])
            if current != record:
                raise Blocked("release campaign changed")
            if operation == "retire" and commit:
                check()
                # Sole owned administrative slot excludes all other commands;
                # any pre-existing/live/stale global slot prevented acquisition.
                if owned_file(os.stat(CAMPAIGN, dir_fd=fd, follow_symlinks=False)) != custody["campaign_token"]:
                    raise Blocked("campaign replaced before retirement")
                check_budget()
                check_time()
                os.unlink(CAMPAIGN, dir_fd=fd)
                # No retries/automatic recovery if either retirement unlink fails.
                check()
                check_budget()
                os.unlink(BUDGET, dir_fd=fd)
        check()
        check_time()
        if operation != "retire" or not commit:
            check_budget()
            current, _ = read_campaign(fd, created if operation == "init" else custody["campaign_token"])
            if current != record:
                raise Blocked("campaign changed at final release boundary")
        os.unlink(SLOT_NAME, dir_fd=fd)
    finally:
        os.close(fd)


class DarwinProbes:
    """Worker ONLY. Even hung native calls cannot block supervisor's clock."""
    def __init__(self, args):
        if sys.platform != "darwin":
            raise Blocked("real probe adapter requires Darwin")
        import psutil
        self.psutil = psutil
        self.authority = Authority(args, os.environ, boot_identity(), time.monotonic())
        diagnostic(f"worker authority boot_seconds={self.authority.boot[0]} "
                   f"boot_microseconds={self.authority.boot[1]} "
                   f"campaign_deadline={self.authority.campaign['deadline']}")
        self.slot = str(self.authority.slot)
        self.disk = Disk(self.authority.paths)
        self.baseline = None

    def host(self):
        try:
            return host_swap.host_snapshot()
        except RuntimeError as exc:
            raise Blocked(str(exc)) from exc

    def processes(self):
        # Same worker process group; owned helper never starts another session.
        helper = subprocess.Popen(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,stat="],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            birth = self.psutil.Process(helper.pid).create_time()
            text, error = helper.communicate(timeout=1)
            if helper.returncode:
                raise Blocked(f"ps failed: {error}")
        finally:
            if helper.returncode is None:
                os.kill(helper.pid, signal.SIGKILL)  # own unreaped helper PID
                helper.wait(timeout=1)  # worker may hang; supervisor never waits here
        rows = []
        for line in text.splitlines():
            pid, ppid, pgid, rss, state = line.split()
            pid, ppid, pgid, rss = map(int, (pid, ppid, pgid, rss))
            born = birth if pid == helper.pid else None
            if pid != helper.pid:
                try:
                    process = self.psutil.Process(pid)
                    born = process.create_time()
                    if (self.psutil.Process(pid).create_time() != born
                            or process.ppid() != ppid or os.getpgid(pid) != pgid):
                        raise Blocked("process birth/ancestry changed during census")
                except (self.psutil.NoSuchProcess, ProcessLookupError):
                    born = None
            rows.append([pid, born, ppid, pgid, rss * 1024, state.startswith("Z")])
        # ps RSS remains counted even though helper was reaped before response.
        return rows, [[helper.pid, birth]]

    def sample(self, sealed, acquire=False):
        self.authority.check(time.monotonic())
        if acquire:
            self.authority.acquire(time.monotonic())
        host = self.host()
        if self.authority.owned_slot is None:
            # Read-only preflight; any new observed debit happens UNDER custody.
            # Otherwise exhaustion here could lose the canonical HOLD slot.
            growth = 0 if self.authority.budget is None else self.authority.budget.record["charged_bytes"]
        else:
            files = self.disk.census(self.authority.excluded())
            if self.baseline is None:
                self.baseline = files.copy()
            growth = self.authority.debit(files, self.baseline)
        rows, reaped = self.processes()
        self.authority.check(time.monotonic())
        return dict(slot=self.slot, host=host, disk_growth=growth, processes=rows,
                    reaped_helpers=reaped, sealed=sealed, custody=self.authority.custody())


def telemetry_worker(args):
    """Private bounded-wire endpoint; after final census, NO further probes/spawns."""
    try:
        probes = DarwinProbes(args)
        while True:
            command = sys.stdin.buffer.readline(3)
            if command not in (b"S\n", b"F\n", b"A\n"):
                return 78
            sealed = command == b"F\n"
            data = json.dumps(probes.sample(sealed, command == b"A\n"), allow_nan=False,
                              separators=(",", ":")).encode() + b"\n"
            if len(data) > MAX_RESPONSE:
                return 78
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            if sealed:
                while True:
                    signal.pause()  # reviewed source: no live helpers/new spawns
    except Exception as exc:
        diagnostic("telemetry worker failed: exception details omitted")
        return 78


class Wire:
    """Nonblocking parent endpoint: one bounded read/write per supervisor tick."""
    def __init__(self, read_fd, write_fd):
        self.read_fd, self.write_fd = read_fd, write_fd
        self.buffer = bytearray()
        os.set_blocking(write_fd, False)
        os.set_blocking(read_fd, False)

    def request(self, final, acquire=False):
        command = b"A\n" if acquire else b"F\n" if final else b"S\n"
        if self.buffer or os.write(self.write_fd, command) != 2:
            raise Blocked("IPC request framing failure")

    def read(self):
        try:
            chunk = os.read(self.read_fd, 65536)
        except BlockingIOError:
            return None
        if not chunk:
            raise Blocked("telemetry worker crash/EOF")
        self.buffer.extend(chunk)
        if len(self.buffer) > MAX_RESPONSE:
            raise Blocked("oversize telemetry response")
        if b"\n" not in self.buffer:
            return None
        if not self.buffer.endswith(b"\n") or self.buffer.count(b"\n") != 1:
            raise Blocked("malformed IPC framing")
        data = json.loads(self.buffer, parse_constant=lambda value: (_ for _ in ()).throw(Blocked(value)))
        self.buffer.clear()
        if not isinstance(data, dict):
            raise Blocked("malformed telemetry response")
        return data


@dataclass
class OwnedChild:
    pid: int = 0
    status_fd: int = -1
    go_fd: int = -1
    confirmed_group: bool = False
    phase: str = "setup"
    returncode: int | None = None
    go_sent: int = 0
    buffer: bytearray = field(default_factory=bytearray)
    fds: set = field(default_factory=set)

    def close_fd(self, fd):
        if fd in self.fds:
            self.fds.remove(fd)  # never retry a possibly reused descriptor
            os.close(fd)

    def close(self):
        error = None
        for fd in tuple(self.fds):
            try:
                self.close_fd(fd)
            except OSError as exc:
                error = exc
        if error is not None:
            raise error

    def advance(self, allow_go):
        if self.phase == "executed":
            return True  # EOF was exec transition, NOT payload success
        if self.returncode is not None or self.phase == "failed":
            raise Blocked("startup already terminal/uncertain")
        try:
            chunk = os.read(self.status_fd, 512)
        except BlockingIOError:
            chunk = None
        if chunk:
            self.buffer.extend(chunk)
            if len(self.buffer) > 256:
                raise Blocked("oversize startup status")
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer[:] = rest
                if line.startswith(b"E:"):
                    self.phase = "failed"
                    raise Blocked(f"child setup/exec error: {line[:64]!r}")
                if (line != f"R:{self.pid}".encode() or self.confirmed_group
                        or self.go_sent or not allow_go):
                    raise Blocked("partial/duplicate/foreign/late GROUP_READY")
                self.confirmed_group, self.phase = True, "group_ready"
        if chunk == b"":
            if not self.confirmed_group or self.go_sent != 3 or self.buffer:
                raise Blocked("startup EOF before qualified GO/exec transition")
            self.phase = "executed"
            self.close_fd(self.status_fd)
            self.close_fd(self.go_fd)
            return True
        if self.confirmed_group and self.go_sent < 3 and allow_go and not self.buffer:
            try:
                n = os.write(self.go_fd, b"GO\n"[self.go_sent:])
            except BlockingIOError:
                n = 0
            if type(n) is not int or not 0 <= n <= 3 - self.go_sent:
                raise Blocked("invalid GO write")
            self.go_sent += n  # partial/EAGAIN progresses on next bounded tick
        return False


def child_bootstrap(argv, cwd, env, status_fd, go_fd, stdin_fd=None, stdout_fd=None):
    """Fork child only: trusted setup, no imports/probes/spawns before GO."""
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGCHLD, signal.SIGPIPE):
            signal.signal(sig, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_SETMASK, set())
        os.setsid()
        if stdin_fd is not None:
            os.dup2(stdin_fd, 0)
            os.dup2(stdout_fd, 1)
        keep = {0, 1, 2, status_fd, go_fd}
        # Enumeration/close is inside the supervised child, not the parent.
        for value in os.listdir("/dev/fd"):
            fd = int(value)
            if fd not in keep:
                try:
                    os.close(fd)
                except OSError as exc:
                    if exc.errno != 9:  # enumeration's own FD may be closed
                        raise
        ready = f"R:{os.getpid()}\n".encode()
        if os.write(status_fd, ready) != len(ready):
            os._exit(78)
        go = b""
        while len(go) < 3:
            chunk = os.read(go_fd, 3 - len(go))
            if not chunk:
                os._exit(78)
            go += chunk
        if go != b"GO\n":
            os._exit(78)
        os.close(go_fd)
        if cwd is not None:
            os.chdir(cwd)
        os.execvpe(argv[0], argv, env)  # CLOEXEC status EOF is exec transition
    except BaseException as exc:
        try:
            os.write(status_fd, f"E:{getattr(exc, 'errno', 0)}\n".encode()[:64])
        except BaseException:
            pass
        os._exit(78)  # never flush inherited application buffers/finalizers


class Real:
    """No runtime probes/blocking waits; post-shutdown authority-only exception."""
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    def __init__(self):
        self.children = {}
        self.interrupted = False
        self.acquired = False
        self.slot_pending = False

    def acquire(self, slot):
        self.slot_pending = True  # worker performs checked exclusive creation

    def accept(self, data, acquire=False):
        self.custody = data["custody"]
        if acquire:
            if self.custody.get("owned_slot") is None:
                raise Blocked("acquisition not qualified")
            self.acquired, self.slot_pending = True, False

    def release(self, slot, commit=True):
        finish_authority(self.custody, slot, self.now, commit)
        self.acquired = False

    @staticmethod
    def context():
        if (__name__ != "__main__" or threading.active_count() != 1
                or threading.current_thread() is not threading.main_thread()
                or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL
                or {signal.SIGINT, signal.SIGTERM} & signal.pthread_sigmask(signal.SIG_BLOCK, set())):
            raise Blocked("standalone single-thread CLI/default SIGCHLD/exclusive reaper required")

    def spawn(self, name, argv, cwd, env, ipc=False):
        self.context()
        all_fds = set()
        child = OwnedChild()
        pid = 0
        is_child = False
        def pipe():
            pair = os.pipe()  # Python creates CLOEXEC descriptors
            all_fds.update(pair)
            if min(pair) < 3:
                raise Blocked("ordinary open stdin/stdout/stderr required")
            for fd in pair:
                os.set_inheritable(fd, False)
            return pair
        try:
            status_r, status_w = pipe()
            go_r, go_w = pipe()
            child.status_fd, child.go_fd = status_r, go_w
            parent_fds = {status_r, go_w}
            stdin_fd = stdout_fd = None
            if ipc:
                stdin_fd, input_w = pipe()
                output_r, stdout_fd = pipe()
                parent_fds |= {input_w, output_r}
                wire = Wire(output_r, input_w)
            for fd in parent_fds:
                os.set_blocking(fd, False)
            child.fds = parent_fds.copy()
            # Preallocate registration. Child cannot exec/import/spawn until GO.
            self.children[name] = child
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                pid = os.fork()
                if pid == 0:
                    is_child = True
                    child_bootstrap(argv, cwd, env, status_w, go_r, stdin_fd, stdout_fd)
                    os._exit(78)
                child.pid = pid  # FIRST parent action: actual unreaped PID custody
            finally:
                if not is_child:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            for fd in all_fds - parent_fds:
                all_fds.remove(fd)
                os.close(fd)
            if ipc:
                self.wire = wire
            return pid
        except BaseException:
            if pid <= 0:
                self.children.pop(name, None)
                for fd in all_fds:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            else:
                # Registered child remains owned even after partial parent setup.
                child.phase = "failed"
                child.fds |= all_fds
            raise

    def worker(self, options):
        return self.spawn("worker", [sys.executable, "-B", os.path.abspath(__file__),
                                    "--_telemetry", *options], None, dict(os.environ), True)

    def launch(self, argv, cwd, env):
        return self.spawn("payload", argv, cwd, env)

    def startup_step(self, allow_go=True):
        ready = True
        for child in self.children.values():
            ready = child.advance(allow_go) and ready
        return ready

    def close(self, name):
        self.children[name].close()

    def pid(self, name):
        return self.children[name].pid

    def signal(self, name, sig):
        child = self.children[name]
        if child.pid <= 0 or child.returncode is not None:
            raise Blocked("no group signal after owned leader reaped")
        # Unreaped own child pins PID/PGID. No Popen.poll/send_signal/kill.
        try:
            if child.confirmed_group:
                os.killpg(child.pid, sig)
            else:
                os.kill(child.pid, sig)  # no group inference from PID/timeout
        except ProcessLookupError:
            pass  # zombie-only group; final qualified census STILL required

    def reap(self, name):
        child = self.children[name]
        if child.pid <= 0:
            raise Blocked("no reap of unregistered child")
        if child.returncode is None:
            pid, status = os.waitpid(child.pid, os.WNOHANG)
            if not pid:
                return None
            if pid != child.pid:
                raise Blocked("reap identity differs from owned unreaped PID")
            child.returncode = os.waitstatus_to_exitcode(status)
        return child.returncode

    def request(self, final, acquire=False):
        self.wire.request(final, acquire)

    def read(self):
        return self.wire.read()


class Family:
    def __init__(self, monitor, worker, payload=None):
        self.monitor, self.worker, self.payload = monitor, worker, payload
        self.known = {monitor, worker}
        if payload:
            self.known.add(payload)
        self.payload_known = {payload} if payload else set()
        self.worker_known = {worker}
        self.escaped = False
        self.worker_escaped = False

    def observe(self, rows):
        table = {}
        for row in rows:
            if (not isinstance(row, list) or len(row) != 6
                    or any(type(row[i]) is not int or row[i] < 0 for i in (0, 2, 3, 4))
                    or type(row[5]) is not bool
                    or (row[1] is not None and (type(row[1]) not in (int, float)
                        or not math.isfinite(row[1]) or row[1] <= 0))):
                raise Blocked("invalid process telemetry")
            if row[0] in table:
                raise Blocked("duplicate PID telemetry")
            table[row[0]] = row
        active = {(p[0], p[1]) for p in rows if (p[0], p[1]) in self.known}
        if self.monitor not in active or self.worker not in active:
            raise Blocked("supervisor/worker birth lost")
        changed = True
        while changed:
            changed = False
            parents = {pid for pid, _ in active}
            payload_parents = {pid for pid, birth in self.payload_known
                               if pid in table and table[pid][1] == birth}
            worker_parents = {pid for pid, birth in self.worker_known
                              if pid in table and table[pid][1] == birth}
            for p in rows:
                key = p[0], p[1]
                group = self.payload is not None and p[3] == self.payload[0]
                if p[2] in parents or group or p[3] == self.worker[0]:
                    if p[1] is None:
                        raise Blocked("relevant birth unknown")
                    if key not in active:
                        active.add(key)
                        changed = True
                if group or p[2] in payload_parents:
                    if key not in self.payload_known:
                        self.payload_known.add(key)
                        changed = True
                if p[3] == self.worker[0] or p[2] in worker_parents:
                    if key not in self.worker_known:
                        self.worker_known.add(key)
                        changed = True
        self.known |= active
        self.escaped |= any((p[0], p[1]) in self.payload_known and not p[5]
                            and p[3] != self.payload[0] for p in rows) if self.payload else False
        self.worker_escaped |= any((p[0], p[1]) in self.worker_known and not p[5]
                                   and p[3] != self.worker[0] for p in rows)
        return [p for p in rows if (p[0], p[1]) in active]

    def live_payload(self, rows):
        return [p for p in rows if not p[5] and (p[0], p[1]) in self.payload_known]


def validated_headroom(host, baseline=None, previous=None):
    """Single admission selector: no raw-free fallback or alternate estimator."""
    try:
        available = host_swap.validate_headroom_observation(host)
        for other in (baseline, previous):
            if other is None:
                continue
            host_swap.validate_headroom_observation(other)
            if any(host[key] != other[key] for key in (
                    "schema", "counter_source", "page_bytes", "page_size_source", "free_source",
                    "available_estimate_source", "physical_total_bytes", "headroom_metric")):
                raise RuntimeError("Darwin headroom identity changed")
        return available
    except RuntimeError as exc:
        raise Blocked(str(exc)) from exc


def free_denial(reason, host, required):
    return (f"{reason}: metric=available_estimate_bytes; "
            f"measured_estimate_bytes={host['available_estimate_bytes']}; "
            f"raw_unused_free_bytes={host['free_bytes']}; required_bytes={required}; "
            f"estimator={host['available_estimate_source']}; page_size_source={host['page_size_source']}")


def resource_violation(host, baseline, previous, rss, growth, limits):
    host_swap.validate_counter_identity(host, baseline)
    host_swap.validate_counter_identity(host, previous)
    headroom = validated_headroom(host, baseline, previous)
    if host["swap_used_bytes"] < previous["swap_used_bytes"]:
        raise Blocked("swap-used counter fell; review required")
    if host["swap_used_bytes"] > baseline["swap_used_bytes"]:
        return "swap-used growth"
    if host["swapout_bytes"] > baseline["swapout_bytes"]:
        return "actual Swapouts growth"
    if headroom < limits.reserve:
        return free_denial("host available-estimate reserve", host, limits.reserve)
    if rss > limits.rss:
        return "aggregate RSS including supervisor/worker/helpers"
    if growth > limits.disk_growth:
        return "cumulative observed disk growth"
    return None


@dataclass(frozen=True)
class Outcome:
    code: int
    held: bool
    reason: str


def diagnostic(message: str) -> bool:
    """Attempt one bounded write to an already-nonblocking FIFO/socket fd 2.

    External flag changes are excluded; no hard kernel wall-time guarantee.
    Success means a full write, not persistence, authentication or cleanup proof.
    Blocking/default stderr and unsupported sinks intentionally lose diagnostics.
    """
    try:
        mode = os.fstat(2).st_mode
        flags = fcntl.fcntl(2, fcntl.F_GETFL)
        if not (statmod.S_ISFIFO(mode) or statmod.S_ISSOCK(mode)) or not flags & os.O_NONBLOCK:
            return False
        body = message[:496].replace("\n", " ").replace("\r", " ").encode("ascii", "replace")
        record = b"command guard: " + body + b"\n"  # <=512 bytes including newline
        return os.write(2, record) == len(record)
    except Exception:
        return False  # No retry, TextIO, flag mutation or fallback.


def reap_diagnostic(name, terminal_known, returncode, reap_error_observed, wait_timeout):
    expected = (returncode in (-signal.SIGTERM, -signal.SIGKILL)
                if name == "worker" else returncode == 0)
    expected_exit = ("expected" if expected else "unexpected") if terminal_known else "unknown"
    custody = "terminal_known" if terminal_known else "pending/unknown"
    return diagnostic(f"reap {name} terminal_known={terminal_known} "
                      f"returncode={returncode if terminal_known else 'unknown'} "
                      f"reap_error_observed={reap_error_observed} wait_timeout={wait_timeout} "
                      f"expected_exit={expected_exit} custody={custody}")


def run(argv, cwd, env, slot, limits, adapter, monitor_pid, operation="run", start=None):
    """Production control loop; tests replace I/O only, including hung probes."""
    start = adapter.now() if start is None else start
    baseline = previous = family = None
    reason = None
    failure_phase = None
    held = False
    owned = ["worker"]
    stops = {}
    killed = set()
    deadline = start
    validated = False
    grace = 1  # cleanup protocol cannot be weakened by invalid request bounds
    identities_emission_attempted = False
    reap_errors = set()

    def phase(final=False, acquire=False):
        return "A acquisition" if acquire else "F final-seal" if final else "S status"

    def outcome(code, retained, text):
        for name in owned:
            reap_diagnostic(name, name in codes, codes.get(name), name in reap_errors, name not in codes)
        if not identities_emission_attempted:
            diagnostic("identities monitor=unknown worker=unknown boot=unknown")
        custody = ("pending/unknown" if adapter.slot_pending else
                   "retained" if adapter.acquired else "unvalidated" if retained else "released")
        diagnostic(f"slot custody={custody} caller_HOLD={retained}")
        return Outcome(code, retained, text)

    def stop(name):
        nonlocal held
        if name not in stops:
            stops[name] = adapter.now()
            try:
                adapter.signal(name, signal.SIGTERM)
            except Exception:
                held = True

    def progress():
        nonlocal reason, held
        now = adapter.now()
        if "payload" in owned and (now >= deadline or adapter.interrupted):
            reason = reason or "deadline/interrupted"
            stop("payload")
        for name, at in list(stops.items()):
            if name not in killed and now >= at + grace:
                try:
                    adapter.signal(name, signal.SIGKILL)
                except Exception:
                    held = True
                killed.add(name)

    def nap(seconds):
        now = adapter.now()
        due = [now + seconds]
        if "payload" in owned and "payload" not in stops:
            due.append(deadline)
        due += [at + grace for name, at in stops.items() if name not in killed]
        adapter.sleep(max(0, min(due) - now))

    def request(final=False, acquire=False):
        diagnostic(f"telemetry {phase(final, acquire)} start")
        adapter.request(final, acquire)
        until = adapter.now() + limits.probe_timeout
        while True:
            progress()  # independent even when worker never answers
            if adapter.now() >= until:
                raise Blocked("probe timeout/stall")
            data = adapter.read()
            if data is not None:
                if adapter.now() >= until or data.get("sealed") is not final:
                    raise Blocked("late/malformed telemetry response")
                return data
            nap(min(0.05, max(0, until - adapter.now())))

    def sample(final=False, acquire=False):
        nonlocal baseline, previous, family, reason, identities_emission_attempted, failure_phase
        failure_phase = phase(final, acquire)
        data = request(final, acquire)
        if data.get("slot") != str(slot):
            raise Blocked("alternate coordinator slot refused")
        rows, host, growth = data["processes"], data["host"], data["disk_growth"]
        if type(growth) is not int or growth < 0:
            raise Blocked("invalid disk meter")
        def birth(pid, owned=False):
            matches = [p for p in rows if len(p) == 6 and p[0] == pid]
            if (len(matches) != 1 or type(matches[0][1]) not in (int, float)
                    or not math.isfinite(matches[0][1]) or matches[0][1] <= 0
                    or (owned and (matches[0][2] != monitor_pid or matches[0][3] != pid))):
                raise Blocked("owned leader birth/ancestry uncertain; no guessed identity")
            row = matches[0]
            return row[0], row[1]
        if family is None:
            family = Family(birth(monitor_pid), birth(adapter.pid("worker"), True))
        members = family.observe(rows)
        if not identities_emission_attempted:
            diagnostic(f"identities monitor_pid={family.monitor[0]} monitor_birth={family.monitor[1]} "
                       f"worker_pid={family.worker[0]} worker_birth={family.worker[1]} boot=unknown "
                       f"command_deadline={deadline} control_deadline={limits.campaign_deadline}")
            # Boot is validated inside worker authority, not by this supervisor.
            identities_emission_attempted = True
        if baseline is None:
            baseline = previous = host
        headroom = validated_headroom(host, baseline, previous)
        required = max(24 * GIB, limits.rss + limits.reserve)
        if "payload" not in owned and headroom < required:
            reason = reason or free_denial("preflight needs >=24 GiB available-estimate headroom", host, required)
        violation = resource_violation(host, baseline, previous, sum(p[4] for p in members), growth, limits)
        previous = host
        reason = reason or violation
        if family.escaped or family.worker_escaped:
            reason = reason or "observed escape; BLOCKED"
        adapter.accept(data, acquire)
        if not final:
            diagnostic(f"telemetry {phase(final, acquire)} completed")
            failure_phase = None
        return data

    def await_startups():
        until = adapter.now() + limits.probe_timeout
        while True:
            progress()
            if adapter.interrupted or adapter.now() >= deadline:
                raise Blocked("startup deadline/cancellation; no inferred readiness")
            if adapter.now() >= until:
                raise Blocked("startup setup/exec timeout; custody unresolved")
            if adapter.startup_step(True):
                if adapter.now() >= until:
                    raise Blocked("late startup completion")
                if adapter.interrupted or adapter.now() >= deadline:
                    raise Blocked("startup completed after cancellation/deadline")
                return
            nap(min(0.05, until - adapter.now()))

    try:
        limits.validate(start)
        validated = True
        deadline = min(start + limits.seconds, limits.campaign_deadline - limits.cleanup_budget)
        await_startups()  # custody exists BEFORE any setup/exec completion wait
        sample()
        adapter.acquire(slot)
        sample(acquire=True)  # worker rechecks pins/state AT mutation boundary
        if adapter.interrupted or adapter.now() >= deadline:
            reason = reason or "preflight deadline/interrupted"
        if reason is None and operation == "run":
            pid = adapter.launch(argv, cwd, env)
            owned.append("payload")
            await_startups()
            data = sample()
            matches = [p for p in data["processes"] if p[0] == pid]
            if (len(matches) != 1 or type(matches[0][1]) not in (int, float)
                    or not math.isfinite(matches[0][1]) or matches[0][1] <= 0
                    or matches[0][2] != monitor_pid or matches[0][3] != pid):
                raise Blocked("payload birth/ancestry uncertain; no guessed identity")
            row = matches[0]
            family.payload = row[0], row[1]
            family.payload_known.add(family.payload)
            family.known.add(family.payload)
            while True:
                family.observe(data["processes"])
                leader = next(p for p in data["processes"] if (p[0], p[1]) == family.payload)
                if reason or family.escaped or leader[5] or "payload" in stops:
                    break
                nap(limits.interval)
                data = sample()
            stop("payload")
            # Continue resource observations through TERM/full grace/KILL/settle.
            while adapter.now() < stops["payload"] + 2 * grace:
                sample()
                progress()
                nap(limits.interval)
            progress()
        final = sample(True)  # worker seals: no subsequent helpers/probes/spawns
        rows = final["processes"]
        reaped = {tuple(key) for key in final["reaped_helpers"]}
        for key in reaped:
            if not any((p[0], p[1]) == key and p[2] == family.worker[0]
                       and p[3] == family.worker[0] for p in rows):
                raise Blocked("unqualified reaped-helper identity")
        worker_children = [p for p in family.observe(rows) if (p[0], p[1]) in family.worker_known
                           and (p[0], p[1]) != family.worker and not p[5]
                           and (p[0], p[1]) not in reaped]
        if family.live_payload(rows) or family.escaped or family.worker_escaped or worker_children:
            held = True
            reason = reason or "final qualified census has descendants/escape"
        diagnostic(f"telemetry {phase(True)} completed")
        failure_phase = None
    except Exception as exc:
        held = True  # no qualified final census, even if leaders later reap
        phase_label = f" [{failure_phase}]" if failure_phase is not None else ""
        reason = f"telemetry/control failure{phase_label}: {exc}"
    finally:
        # Emergency path never calls a probe, pipe read, or blocking helper wait.
        for name, child in getattr(adapter, "children", {}).items():
            if child.pid > 0 and name not in owned:
                owned.append(name)  # launch failed AFTER immediate registration
        for name in owned:
            stop(name)
        until = max(stops.values()) + grace
        while adapter.now() < until:
            progress()
            nap(min(0.05, max(0, until - adapter.now())))
        progress()  # every owned leader receives final signal BEFORE any reap
        until = adapter.now() + (limits.reap_timeout if validated else 1)
        codes = {}
        while len(codes) != len(owned):
            for name in owned:
                if name not in codes:
                    try:
                        code = adapter.reap(name)  # WNOHANG only
                        if code is not None:
                            codes[name] = code
                    except Exception:
                        reap_errors.add(name)
                        held = True
            if len(codes) == len(owned) or adapter.now() >= until:
                break
            nap(min(0.05, max(0, until - adapter.now())))
        if len(codes) != len(owned):
            held = True
            reason = reason or "owned leader terminal wait incomplete"
        if codes.get("worker") not in (-signal.SIGTERM, -signal.SIGKILL):
            held = True
            reason = reason or "observer did not exit through controlled shutdown"
        for name in owned:
            try:
                adapter.close(name)
            except Exception:
                held = True
                reason = reason or "startup/IPC descriptor custody uncertain"
    if held:
        custody = ("slot retained/ownership unresolved" if adapter.acquired or adapter.slot_pending
                   else "no validated slot acquired; caller HOLD required")
        return outcome(78, True, f"BLOCKED {custody}: {reason or 'cleanup uncertain'}")
    try:
        adapter.release(slot, commit=reason is None)  # sealed census + ALL reaps
    except Exception as exc:
        return outcome(78, True, f"BLOCKED slot retained; final authority check failed: {exc}")
    if operation != "run":
        return outcome(0 if reason is None else 78, False, reason or f"explicit campaign {operation}")
    if "payload" not in owned:
        return outcome(78, False, reason or "preflight refused")
    return outcome(1 if reason or codes["payload"] != 0 else 0, False,
                   reason or f"command exit {codes['payload']}; qualified cleanup")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--_telemetry", action="store_true", help=argparse.SUPPRESS)
    for name in ("cwd", "coordinator-root", "slot", "target-dir", "log-dir", "temp-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--disk-root", type=Path, action="append", default=[])
    p.add_argument("--campaign-deadline", type=float)
    management = p.add_mutually_exclusive_group()
    management.add_argument("--campaign-init", action="store_true")
    management.add_argument("--campaign-retire", action="store_true")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--rss-bytes", type=int, default=8 * GIB)
    p.add_argument("--reserve-bytes", type=int, default=16 * GIB)
    p.add_argument("--disk-growth-bytes", type=int, default=20 * GIB)
    p.add_argument("--cooperative-build", action="store_true")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    return p


def main():
    try:
        Real.context()  # refuse embedding BEFORE handlers/children/native probes
    except Exception as exc:
        diagnostic("BLOCKED: execution context refused; exception details omitted")
        return 78
    args = parser().parse_args()
    if args._telemetry:
        return telemetry_worker(args)
    operation = "init" if args.campaign_init else "retire" if args.campaign_retire else "run"
    if ((operation == "run" and (not args.cooperative_build or args.argv[:1] != ["--"] or len(args.argv) < 2))
            or (operation != "run" and args.argv)):
        diagnostic("BLOCKED: reviewed cooperative argv after -- required")
        return 78
    adapter = Real()
    # No PID-based stale-lock removal. Existing fixed slot always needs review.
    try:
        # Retirement has a bounded administrative shutdown window, never a new
        # campaign/payload budget. It can retire an expired SAME-boot lease.
        control_deadline = adapter.now() + 30 if operation == "retire" else args.campaign_deadline
        limits = Limits(control_deadline, args.timeout, args.rss_bytes,
                        args.reserve_bytes, args.disk_growth_bytes)
        limits.validate(adapter.now())  # invalid CLI bounds create NO children/files
        options = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
        def interrupted(_signum, _frame):
            adapter.interrupted = True
        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
        started = adapter.now()  # no new command allowance after fork/bootstrap
        try:
            adapter.worker(options)
        except Exception:
            if "worker" not in adapter.children:
                raise
            # A partially initialized wire must still retire its owned worker
            # through the SAME independent control loop and retain the slot.
        result = run(args.argv[1:] if operation == "run" else [], args.cwd, dict(os.environ),
                     args.slot.absolute(), limits, adapter, os.getpid(), operation, started)
        diagnostic(result.reason)
        return result.code
    except Exception as exc:
        diagnostic("BLOCKED slot retained if acquired: control failure; exception details omitted")
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
