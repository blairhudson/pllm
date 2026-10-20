"""One-request encrypted OFFLINE material spool, never a durable/resumable cache.

The online worker reads only after its dealer has retired. Role/session/auth and
dealer isolation are the caller's responsibility; this file does not implement
that topology. Metadata is opaque public canonical bytes, authenticated in full
with request, role, ordinal and lengths. V1 stores it in the clear; opt-in v2
omits it from disk and requires the same original bytes at take(). Packets are
AES-GCM sealed; neither mode substitutes a metadata hash for the original AAD.

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

MAX_METADATA_BYTES = 4096
MAX_PACKET_BYTES = 16 << 20
MAX_TOTAL_PACKET_BYTES = 256 << 20
MAX_FILE_BYTES = 1 << 30
DECODER_MAX_FILE_BYTES = 32 << 30
DECODER_MAX_TOTAL_PACKET_BYTES = DECODER_MAX_FILE_BYTES
DISK_HEADROOM_BYTES = 1 << 20
_DOMAIN = b"pllm.fss.offline-material-tape.aesgcm.v1\0"
_EXTERNAL_AAD_DOMAIN = b"pllm.fss.offline-material-tape.aesgcm.v2\0"
_HEADER = struct.Struct(">IQ12s")  # full metadata INPUT length, ciphertext+tag length, ordinal nonce
_LIMITS = struct.Struct(">IQQ")  # shared v1/v2 count, packet ceiling, cumulative packet ceiling
_ORDINAL_BYTES = 4
_NONCE_BYTES = 12
_TAG_BYTES = 16
_BUFFER_SLACK = 64 << 10
# Preserve the v1 count/AAD/nonce encodings. The count itself must fit u32;
# ordinals are zero-based and must fit both AAD and nonce. Every record needs at
# least one packet byte, and must reserve all selected on-disk metadata/framing.
# MAX_RECORDS preserves the Capsule/inline/default-metadata ceiling. price() derives
# the selected capacity's ceiling from M and also checks the complete packet sum.
_ENCODED_RECORDS = min((1 << 32) - 1, 1 << (8 * _ORDINAL_BYTES), 1 << (8 * _NONCE_BYTES))
_RECORD_OVERHEAD = MAX_METADATA_BYTES + _HEADER.size + _TAG_BYTES
MAX_RECORDS = min(_ENCODED_RECORDS, MAX_TOTAL_PACKET_BYTES,
                  MAX_FILE_BYTES // (_RECORD_OVERHEAD + 1))


class MaterialTapeError(ValueError):
    """Fixed, payload-free failure; the active tape is consumed on every error."""


def capacity_limits(capacity="capsule"):
    """Independent hard ceilings, not native admission or a disk reservation."""
    if type(capacity) is not str or capacity not in ("capsule", "decoder"):
        raise MaterialTapeError("material tape capacity differs")
    if capacity == "decoder":
        return DECODER_MAX_TOTAL_PACKET_BYTES, DECODER_MAX_FILE_BYTES
    return MAX_TOTAL_PACKET_BYTES, MAX_FILE_BYTES


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
    capacity: str = "capsule"
    metadata_storage: str = "inline"

    @property
    def stored_metadata_bytes_per_record(self) -> int:
        """Disk-only metadata bound; input and AAD retain their full bound."""
        return self.metadata_bytes_per_record if self.metadata_storage == "inline" else 0


class MaterialTape:
    """Sequential consuming material owner; no path, key, seek or resume API.

    I/O is synchronous and serialized. close/cancel waits for the current I/O
    operation, then drops the AEAD/key reference and closes the unlinked file.
    Callers must retire their own input/output copies independently.
    Only counters and streaming hashes survive each operation; no record index,
    metadata array or nonce ledger is retained. Native owns material-nonce replay
    protection; the tape's independent AEAD nonces are its sequential ordinals.
    """

    @staticmethod
    def price(*, records: int, packet_bytes: int, total_packet_bytes: int,
              capacity: str = "capsule", metadata_bytes_per_record: int = MAX_METADATA_BYTES,
              metadata_storage: str = "inline") -> TapeResources:
        """Pure public preflight, before opening a file or generating any key.

        total_packet_bytes is a cumulative ceiling, not a required exact total.
        Empty tapes require all three limits to be zero. Each nonempty record
        needs at least one metadata byte and one packet byte.

        For N records, packet ceiling P, cumulative ceiling T and metadata bound
        M (default 4096), the full file price is min(T, N*P) + N*(S + 24 + 16),
        where S=M for inline/v1, or S=0 for external_aad/v2. V2 retains the
        metadata length in each header and authenticates the FULL caller metadata
        as AAD. It saves only duplicated disk bytes, not input/AAD/live ownership.
        Decoder accepts a tighter native-admitted M, but never larger packets or
        metadata. Capacity and M are storage policy, not new wire/AAD fields.
        They do not authorize a decoder request or reserve any caller's storage.
        The absolute record ceiling follows from P >= 1 and T >= N, not a
        separate request-count policy.
        The actual native-admitted N must still fit that full file price; native
        admission is the caller's responsibility, never inferred from this price.
        Live payload buffers depend on P/T, not N, even at the record ceiling.
        """
        total_limit, file_limit = capacity_limits(capacity)
        if type(metadata_storage) is not str or metadata_storage not in ("inline", "external_aad"):
            raise MaterialTapeError("material tape metadata storage differs")
        if type(metadata_bytes_per_record) is not int or not 1 <= metadata_bytes_per_record <= MAX_METADATA_BYTES:
            raise MaterialTapeError("material tape metadata limit exceeds bounds")
        stored_metadata = metadata_bytes_per_record if metadata_storage == "inline" else 0
        overhead = stored_metadata + _HEADER.size + _TAG_BYTES
        record_limit = min(_ENCODED_RECORDS, total_limit, file_limit // (overhead + 1))
        if (type(records) is not int or type(packet_bytes) is not int
                or type(total_packet_bytes) is not int or not 0 <= records <= record_limit
                or not 0 <= packet_bytes <= MAX_PACKET_BYTES
                or not 0 <= total_packet_bytes <= total_limit
                or (records == 0 and (packet_bytes != 0 or total_packet_bytes != 0))
                or (records != 0 and (packet_bytes == 0 or total_packet_bytes < records))):
            raise MaterialTapeError("material tape limits exceed bounds")
        total = min(total_packet_bytes, records * packet_bytes)
        # A partial writer may exhaust its cumulative cap before all records;
        # price that legal append too, even though seal must then reject.
        packet = min(packet_bytes, total_packet_bytes) if records else 0
        framing = _HEADER.size + _TAG_BYTES
        # Python integer arithmetic cannot wrap. Prove the complete file fits
        # before any file/key, then bind these exact limits into every record.
        if records > (file_limit - total) // overhead:
            raise MaterialTapeError("material tape file bound exceeds limit")
        disk = total + records * overhead
        ciphertext = packet + _TAG_BYTES if records else 0
        # append retains one ciphertext and bounded AAD/header; take retains a
        # ciphertext bytearray, its bytes conversion and returned clear packet.
        # Slack also covers bounded memoryviews/hash/header/object staging. Crypto
        # library internal copies and allocator metadata remain caller-priced.
        # Keep these live bounds in v2 too: full metadata is still input/AAD.
        append = ciphertext + 2 * metadata_bytes_per_record + _BUFFER_SLACK
        take = 2 * ciphertext + packet + 3 * metadata_bytes_per_record + _BUFFER_SLACK
        return TapeResources(records, packet_bytes, total_packet_bytes, metadata_bytes_per_record,
                             framing, disk, disk + DISK_HEADROOM_BYTES, ciphertext,
                             append, take, max(append, take),
                              packet + metadata_bytes_per_record, packet, total, capacity, metadata_storage)

    def __init__(self, request_id: str, role: str, *, records: int, packet_bytes: int,
                 total_packet_bytes: int, directory: Path | None = None,
                 capacity: str = "capsule", metadata_bytes_per_record: int = MAX_METADATA_BYTES,
                 metadata_storage: str = "inline"):
        resources = self.price(records=records, packet_bytes=packet_bytes,
                               total_packet_bytes=total_packet_bytes, capacity=capacity,
                                metadata_bytes_per_record=metadata_bytes_per_record,
                                metadata_storage=metadata_storage)
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
        domain = _DOMAIN if resources.metadata_storage == "inline" else _EXTERNAL_AAD_DOMAIN
        self._binding = (domain + bytes.fromhex(request_id)
                         + bytes([0 if role == "worker_a" else 1])
                         + _LIMITS.pack(records, packet_bytes, total_packet_bytes))
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

    def _metadata(self, metadata):
        if type(metadata) is not bytes or not 0 < len(metadata) <= self._resources.metadata_bytes_per_record:
            raise MaterialTapeError("material tape metadata exceeds bounds")

    def _aad(self, ordinal, header, metadata):
        return self._binding + ordinal.to_bytes(_ORDINAL_BYTES, "big") + bytes(header) + metadata

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
            stored_metadata = len(metadata) if r.metadata_storage == "inline" else 0
            frame_bytes = _HEADER.size + stored_metadata + len(packet) + _TAG_BYTES
            if self._written + frame_bytes > r.max_file_bytes:
                raise MaterialTapeError("material tape file budget exceeded")
            ordinal = self._records_written
            nonce = ordinal.to_bytes(_NONCE_BYTES, "big")
            header = _HEADER.pack(len(metadata), len(packet) + _TAG_BYTES, nonce)
            ciphertext = self._aead.encrypt(nonce, packet, self._aad(ordinal, header, metadata))
            if len(ciphertext) != len(packet) + _TAG_BYTES:
                raise MaterialTapeError("material tape encryption size differs")
            self._write_all(header)
            if r.metadata_storage == "inline":
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
            if (not 0 < meta_bytes <= r.metadata_bytes_per_record or not 0 < packet_bytes <= r.packet_bytes
                    or self._packets_read + packet_bytes > r.total_packet_bytes
                    or nonce != self._records_read.to_bytes(_NONCE_BYTES, "big")):
                raise MaterialTapeError("material tape record geometry differs")
            if r.metadata_storage == "inline":
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


__all__ = ["MaterialTape", "MaterialTapeError", "TapeResources", "capacity_limits"]
