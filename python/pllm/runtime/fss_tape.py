"""One-request encrypted OFFLINE material spool, never a durable/resumable cache.

The online worker reads only after its dealer has retired. Role/session/auth and
dealer isolation are the caller's responsibility; this file does not implement
that topology. Metadata is opaque public canonical bytes, stored in the clear and
authenticated with request, role, ordinal and lengths. Packets are AES-GCM sealed.

Pricing describes logical disk and Python-visible payload buffers, NOT a native
reservation, allocator/RSS bound, filesystem reservation or OpenSSL scratch bound.
Caller input bytes, returned immutable packets, transport/native copies, interpreter
and cryptographic owners require separate reservations held across callbacks.
Returned bytes survive retirement; neither Python nor OS secure erasure is claimed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import threading

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAX_RECORDS = 4096
MAX_METADATA_BYTES = 4096
MAX_PACKET_BYTES = 16 << 20
MAX_TOTAL_PACKET_BYTES = 256 << 20
MAX_FILE_BYTES = 1 << 30
DISK_HEADROOM_BYTES = 1 << 20
_DOMAIN = b"pllm.fss.offline-material-tape.aesgcm.v1\0"
_HEADER = struct.Struct(">IQ12s")  # metadata length, ciphertext+tag length, ordinal nonce
_TAG_BYTES = 16
_BUFFER_SLACK = 64 << 10


class MaterialTapeError(ValueError):
    """Fixed, payload-free failure; the active tape is consumed on every error."""


@dataclass(frozen=True)
class TapeResources:
    records: int
    packet_bytes: int
    total_packet_bytes: int
    metadata_bytes_per_record: int
    framing_bytes_per_record: int
    max_file_bytes: int
    required_disk_free_bytes: int
    max_ciphertext_bytes: int
    append_buffer_bytes: int
    take_buffer_bytes: int
    max_internal_buffer_bytes: int
    caller_input_bytes: int
    max_returned_packet_bytes: int
    max_retained_output_bytes: int


class MaterialTape:
    """Sequential consuming material owner; no path, key, seek or resume API.

    I/O is synchronous and serialized. close/cancel waits for the current I/O
    operation, then drops the AEAD/key reference and closes the unlinked file.
    Callers must retire their own input/output copies independently.
    """

    @staticmethod
    def price(*, records: int, packet_bytes: int, total_packet_bytes: int) -> TapeResources:
        """Pure public preflight, before opening a file or generating any key.

        total_packet_bytes is a cumulative ceiling, not a required exact total.
        Empty tapes require all three limits to be zero. Each nonempty record
        needs at least one metadata byte and one packet byte.
        """
        if (type(records) is not int or type(packet_bytes) is not int
                or type(total_packet_bytes) is not int or not 0 <= records <= MAX_RECORDS
                or not 0 <= packet_bytes <= MAX_PACKET_BYTES
                or not 0 <= total_packet_bytes <= MAX_TOTAL_PACKET_BYTES
                or (records == 0 and (packet_bytes != 0 or total_packet_bytes != 0))
                or (records != 0 and (packet_bytes == 0 or total_packet_bytes < records))):
            raise MaterialTapeError("material tape limits exceed bounds")
        total = min(total_packet_bytes, records * packet_bytes)
        # A partial writer may exhaust its cumulative cap before all records;
        # price that legal append too, even though seal must then reject.
        packet = min(packet_bytes, total_packet_bytes) if records else 0
        framing = _HEADER.size + _TAG_BYTES
        disk = total + records * (MAX_METADATA_BYTES + framing)
        if disk > MAX_FILE_BYTES:
            raise MaterialTapeError("material tape file bound exceeds limit")
        ciphertext = packet + _TAG_BYTES if records else 0
        # append retains one ciphertext and bounded AAD/header; take retains a
        # ciphertext bytearray, its bytes conversion and returned clear packet.
        # Slack also covers bounded memoryviews/hash/header/object staging. Crypto
        # library internal copies and allocator metadata remain caller-priced.
        append = ciphertext + 2 * MAX_METADATA_BYTES + _BUFFER_SLACK
        take = 2 * ciphertext + packet + 3 * MAX_METADATA_BYTES + _BUFFER_SLACK
        return TapeResources(records, packet_bytes, total_packet_bytes, MAX_METADATA_BYTES,
                             framing, disk, disk + DISK_HEADROOM_BYTES, ciphertext,
                             append, take, max(append, take),
                             packet + MAX_METADATA_BYTES, packet, total)

    def __init__(self, request_id: str, role: str, *, records: int, packet_bytes: int,
                 total_packet_bytes: int, directory: Path | None = None):
        resources = self.price(records=records, packet_bytes=packet_bytes,
                               total_packet_bytes=total_packet_bytes)
        if (type(request_id) is not str or len(request_id) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in request_id)
                or int(request_id, 16) == 0 or type(role) is not str
                or role not in ("worker_a", "worker_b")
                or (directory is not None and not isinstance(directory, Path))):
            raise MaterialTapeError("material tape request or role is invalid")
        self._resources = resources
        self._lock = threading.RLock()
        self._file = self._aead = None
        self._state = "creating"
        self._written = self._read = self._packets_written = self._packets_read = 0
        self._records_written = self._records_read = 0
        self._sealed_bytes = 0
        self._sealed_digest = None
        self._write_hash = hashlib.sha256()
        self._read_hash = None
        self._binding = (_DOMAIN + bytes.fromhex(request_id)
                         + bytes([0 if role == "worker_a" else 1])
                         + struct.pack(">IQQ", records, packet_bytes, total_packet_bytes))
        try:
            root = directory if directory is not None else Path(tempfile.gettempdir())
            if shutil.disk_usage(root).free < resources.required_disk_free_bytes:
                raise MaterialTapeError("material tape disk headroom is insufficient")
            self._file = tempfile.TemporaryFile(mode="w+b", buffering=0, dir=root)
            info = os.fstat(self._file.fileno())
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_nlink != 0 or info.st_uid != os.geteuid()):
                raise MaterialTapeError("material tape temporary file is not private and unlinked")
            # No key exists until public pricing, disk check and private-file
            # ownership checks have completed. Never expose/reuse this key.
            self._aead = AESGCM(AESGCM.generate_key(bit_length=256))
            self._state = "writing"
        except BaseException as error:
            self._retire("failed")
            if isinstance(error, (MaterialTapeError, KeyboardInterrupt, SystemExit)):
                raise
            raise MaterialTapeError("material tape creation failed") from None

    @property
    def resources(self) -> TapeResources:
        return self._resources

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._state not in ("writing", "sealed")

    def receipt(self) -> dict:
        """Only public prices, completed counts and ciphertext-file commitment."""
        with self._lock:
            return {"state": self._state, "encrypted_file_bytes_written": self._written,
                    "encrypted_file_bytes_read": self._read,
                    "packet_bytes_written": self._packets_written,
                    "packet_bytes_read": self._packets_read,
                    "records_written": self._records_written, "records_read": self._records_read,
                    "sealed_file_bytes": self._sealed_bytes,
                    "sealed_digest": self._sealed_digest}

    def _retire(self, state):
        handle = self._file
        self._file = self._aead = self._binding = None
        self._write_hash = self._read_hash = None
        self._state = state
        if handle is not None:
            try:
                handle.close()
            except Exception:
                # Unbuffered FileIO retirement is best effort on OS failure;
                # no payload/error text enters receipts, no resume remains.
                pass

    @contextmanager
    def _operation(self):
        with self._lock:
            try:
                yield
            except BaseException as error:
                self._retire("failed")
                if isinstance(error, (MaterialTapeError, KeyboardInterrupt, SystemExit)):
                    raise
                raise MaterialTapeError("material tape operation failed") from None

    @staticmethod
    def _metadata(metadata):
        if type(metadata) is not bytes or not 0 < len(metadata) <= MAX_METADATA_BYTES:
            raise MaterialTapeError("material tape metadata exceeds bounds")

    def _aad(self, ordinal, header, metadata):
        return self._binding + ordinal.to_bytes(4, "big") + bytes(header) + metadata

    def _write_all(self, raw):
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            count = self._file.write(view[offset:])
            if type(count) is not int or not 0 < count <= len(view) - offset:
                raise MaterialTapeError("material tape write failed")
            self._written += count
            self._write_hash.update(view[offset:offset + count])
            offset += count

    def append(self, metadata: bytes, packet: bytes) -> None:
        with self._operation():
            if self._state != "writing":
                raise MaterialTapeError("material tape is not writable")
            self._metadata(metadata)
            r = self._resources
            if (self._records_written >= r.records or type(packet) is not bytes
                    or not 0 < len(packet) <= r.packet_bytes
                    or self._packets_written + len(packet) > r.total_packet_bytes):
                raise MaterialTapeError("material tape packet or record budget exceeded")
            frame_bytes = _HEADER.size + len(metadata) + len(packet) + _TAG_BYTES
            if self._written + frame_bytes > r.max_file_bytes:
                raise MaterialTapeError("material tape file budget exceeded")
            ordinal = self._records_written
            nonce = ordinal.to_bytes(12, "big")
            header = _HEADER.pack(len(metadata), len(packet) + _TAG_BYTES, nonce)
            ciphertext = self._aead.encrypt(nonce, packet, self._aad(ordinal, header, metadata))
            if len(ciphertext) != len(packet) + _TAG_BYTES:
                raise MaterialTapeError("material tape encryption size differs")
            self._write_all(header)
            self._write_all(metadata)
            self._write_all(ciphertext)
            self._packets_written += len(packet)
            self._records_written += 1

    def seal(self) -> None:
        with self._operation():
            if self._state != "writing" or self._records_written != self._resources.records:
                raise MaterialTapeError("material tape record count or sealing order differs")
            self._file.flush()
            os.fsync(self._file.fileno())
            if os.fstat(self._file.fileno()).st_size != self._written:
                raise MaterialTapeError("material tape file size differs")
            self._sealed_bytes = self._written
            self._sealed_digest = self._write_hash.hexdigest()
            self._write_hash = None
            self._read_hash = hashlib.sha256()
            self._file.seek(0)
            self._state = "sealed"

    def _read_exact(self, size):
        if size > self._sealed_bytes - self._read:
            raise MaterialTapeError("material tape is truncated")
        raw = bytearray(size)
        view = memoryview(raw)
        offset = 0
        while offset < size:
            count = self._file.readinto(view[offset:])
            if type(count) is not int or not 0 < count <= size - offset:
                raise MaterialTapeError("material tape is truncated")
            self._read += count
            self._read_hash.update(view[offset:offset + count])
            offset += count
        return raw

    def take(self, metadata: bytes) -> bytes:
        with self._operation():
            if self._state != "sealed" or self._records_read >= self._resources.records:
                raise MaterialTapeError("material tape read order or record count differs")
            self._metadata(metadata)
            header = self._read_exact(_HEADER.size)
            meta_bytes, ciphertext_bytes, nonce = _HEADER.unpack(header)
            r = self._resources
            packet_bytes = ciphertext_bytes - _TAG_BYTES
            if (not 0 < meta_bytes <= MAX_METADATA_BYTES or not 0 < packet_bytes <= r.packet_bytes
                    or self._packets_read + packet_bytes > r.total_packet_bytes
                    or nonce != self._records_read.to_bytes(12, "big")):
                raise MaterialTapeError("material tape record geometry differs")
            stored_metadata = self._read_exact(meta_bytes)
            if not hmac.compare_digest(metadata, stored_metadata):
                raise MaterialTapeError("material tape metadata differs")
            ciphertext = self._read_exact(ciphertext_bytes)
            try:
                packet = self._aead.decrypt(nonce, bytes(ciphertext),
                                            self._aad(self._records_read, header, metadata))
            except InvalidTag:
                raise MaterialTapeError("material tape authentication failed") from None
            if len(packet) != packet_bytes:
                raise MaterialTapeError("material tape decrypted size differs")
            self._records_read += 1
            self._packets_read += packet_bytes
            return packet

    def finish(self) -> None:
        with self._operation():
            if self._state != "sealed" or self._records_read != self._resources.records:
                raise MaterialTapeError("material tape has unread records or invalid retirement order")
            extra = self._file.read(1)
            self._read += len(extra)
            if (extra or self._read != self._sealed_bytes
                    or self._read_hash.hexdigest() != self._sealed_digest
                    or self._packets_read != self._packets_written):
                raise MaterialTapeError("material tape EOF or digest differs")
            self._retire("finished")

    def close(self) -> None:
        with self._lock:
            if self._state in ("creating", "writing", "sealed"):
                self._retire("closed")

    cancel = close

    def __enter__(self):
        with self._operation():
            if self._state not in ("writing", "sealed"):
                raise MaterialTapeError("material tape is closed")
            return self

    def __exit__(self, *_):
        self.close()

    def __getstate__(self):
        self.close()
        raise TypeError("material tapes cannot be copied or resumed")

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass


__all__ = ["MaterialTape", "MaterialTapeError", "TapeResources"]
