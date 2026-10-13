"""Small deterministic public-object delta screen; no model imports or RSS claims.

Descriptors are trusted inputs from authenticated manifests, not authentication
created by this codec. Patch SHA-256 is likewise supplied by a trusted publisher.
Only immutable, caller-owned base files may be used during reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import struct
import sys
import tempfile
import zlib

BLOCK = 65536
MAX_OBJECT = 8 << 20
MAX_HEADER = 16384
MAX_PATCH = MAX_OBJECT + (1 << 20)
MAGIC = b"PLLMDEL1"
RECORD = struct.Struct(">BII")
DOMAIN = b"pllm-public-object/1\0"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def descriptor(data, metadata=None):
    metadata = metadata or {"scope": "synthetic-public", "dtype": "u1"}
    content, domain = sha(data), sha(canonical(metadata))
    return {"sha256": sha(DOMAIN + bytes.fromhex(domain) + bytes.fromhex(content)),
            "content_sha256": content, "domain_hash": domain,
            "size": len(data), "metadata": metadata}


def validate_row(row):
    if set(row) != {"sha256", "content_sha256", "domain_hash", "size", "metadata"}:
        raise ValueError("invalid object descriptor")
    if type(row["size"]) is not int or not 0 < row["size"] <= MAX_OBJECT:
        raise ValueError("object size bound")
    for key in ("sha256", "content_sha256", "domain_hash"):
        if not isinstance(row[key], str) or len(row[key]) != 64 or any(
                c not in "0123456789abcdef" for c in row[key]):
            raise ValueError("invalid digest")
    domain = sha(canonical(row["metadata"]))
    if domain != row["domain_hash"] or row["sha256"] != sha(
            DOMAIN + bytes.fromhex(domain) + bytes.fromhex(row["content_sha256"])):
        raise ValueError("object domain mismatch")


def verify_bytes(data, row):
    validate_row(row)
    if len(data) != row["size"] or sha(data) != row["content_sha256"]:
        raise ValueError("object content mismatch")


def make_patch(old, new, base_row, new_row, codec="xor"):
    """Publisher fixture encoder. Client reconstruction never loads whole objects."""
    verify_bytes(old, base_row)
    verify_bytes(new, new_row)
    if codec not in {"xor", "replace"}:
        raise ValueError("unknown codec")
    header = canonical({"base": base_row, "new": new_row, "block": BLOCK,
                        "codec": codec, "version": 1})
    if len(header) > MAX_HEADER:
        raise ValueError("header bound")
    output = bytearray(MAGIC + struct.pack(">I", len(header)) + header)
    for offset in range(0, len(new), BLOCK):
        target = new[offset:offset + BLOCK]
        source = old[offset:offset + len(target)]
        if source == target:
            output.extend(RECORD.pack(0, len(target), 0))
            continue
        literal = zlib.compress(target, 9)
        residual = bytes(a ^ b for a, b in zip(target, source.ljust(len(target), b"\0")))
        delta = zlib.compress(residual, 9)
        tag, encoded = (1, delta) if codec == "xor" and len(delta) < len(literal) else (2, literal)
        output.extend(RECORD.pack(tag, len(target), len(encoded)))
        output.extend(encoded)
    if len(output) > MAX_PATCH:
        raise ValueError("patch bound")
    return bytes(output)


class Reader:
    def __init__(self, stream):
        self.stream, self.count, self.digest = stream, 0, hashlib.sha256()

    def take(self, count):
        if count < 0 or count > BLOCK + MAX_HEADER or self.count + count > MAX_PATCH:
            raise ValueError("wire bound")
        result = bytearray()
        while len(result) < count:
            part = self.stream.read(count - len(result))
            if not part:
                raise ValueError("truncated patch")
            if len(part) > count - len(result):
                raise ValueError("transport read bound")
            result.extend(part)
        self.count += count
        self.digest.update(result)
        return bytes(result)


def reconstruct(base, stream, destination, *, base_row, new_row, patch_sha256):
    """Verify immutable base, stream patch into unpublished file, atomically replace.

    Owns neither base nor stream. Caller must ensure base is immutable throughout;
    ordinary writable cache paths are NOT production-safe native snapshot owners.
    Destination must be distinct from base. Failure removes only pending file.
    """
    validate_row(base_row)
    validate_row(new_row)
    destination = Path(destination)
    if Path(base).resolve() == destination.resolve() or (
            destination.exists() and os.path.samefile(base, destination)):
        raise ValueError("base cannot be publication target")
    with Path(base).open("rb") as source:
        hashed, size = hashlib.sha256(), 0
        while block := source.read(BLOCK):
            size += len(block)
            if size > base_row["size"]:
                raise ValueError("base oversize")
            hashed.update(block)
        if size != base_row["size"] or hashed.hexdigest() != base_row["content_sha256"]:
            raise ValueError("base mismatch")
        reader = Reader(stream)
        if reader.take(8) != MAGIC:
            raise ValueError("patch magic")
        length = struct.unpack(">I", reader.take(4))[0]
        if not 0 < length <= MAX_HEADER:
            raise ValueError("header bound")
        raw_header = reader.take(length)
        header = json.loads(raw_header)
        if not isinstance(header, dict) or canonical(header) != raw_header or header != {
                "base": base_row, "new": new_row, "block": BLOCK,
                "codec": header.get("codec"), "version": 1} or header.get("codec") not in {"xor", "replace"}:
            raise ValueError("patch identity mismatch")
        fd, pending = tempfile.mkstemp(prefix=".delta-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                hashed, offset = hashlib.sha256(), 0
                while offset < new_row["size"]:
                    tag, raw_size, encoded_size = RECORD.unpack(reader.take(RECORD.size))
                    expected = min(BLOCK, new_row["size"] - offset)
                    if raw_size != expected or tag not in (0, 1, 2):
                        raise ValueError("record output bound")
                    source.seek(offset)
                    old = source.read(raw_size)
                    if tag == 0:
                        if encoded_size != 0 or len(old) != raw_size:
                            raise ValueError("invalid copy")
                        block = old
                    else:
                        if not 0 < encoded_size <= raw_size + 1024 or (tag == 1 and header["codec"] != "xor"):
                            raise ValueError("record encoded bound")
                        decoder = zlib.decompressobj()
                        block = decoder.decompress(reader.take(encoded_size), raw_size + 1)
                        if len(block) != raw_size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                            raise ValueError("invalid compressed record")
                        if tag == 1:
                            block = bytes(a ^ b for a, b in zip(block, old.ljust(raw_size, b"\0")))
                    output.write(block)
                    hashed.update(block)
                    offset += raw_size
                if stream.read(1):
                    raise ValueError("trailing patch")
                if reader.digest.hexdigest() != patch_sha256:
                    raise ValueError("patch commitment mismatch")
                if hashed.hexdigest() != new_row["content_sha256"]:
                    raise ValueError("new object mismatch")
                output.flush()
                os.fsync(output.fileno())
            os.replace(pending, destination)
        finally:
            if os.path.exists(pending):
                os.unlink(pending)
    return {"wire_bytes": reader.count, "content_sha256": hashed.hexdigest(),
            "object_sha256": new_row["sha256"]}


def full_controls(data):
    """Independent full-new zlib encoder: strongest measured bounded control."""
    return {f"zlib-{level}-{chunk}": sum(
        8 + len(zlib.compress(data[i:i + chunk], level))
        for i in range(0, len(data), chunk))
        for level in (1, 6, 9) for chunk in (BLOCK, 1 << 20)}


def fixtures(seed, size):
    rng = random.Random(seed)
    old = rng.randbytes(size)
    sparse = bytearray(old)
    for offset in range(17, size, 4096):
        sparse[offset] ^= 1
    return [("sparse", old, bytes(sparse)),
            ("dense", old, rng.randbytes(size)),
            ("dense_low_entropy", old, b"\0" * size),
            ("shifted", old, b"public shift" + old[:-12]),
            ("append", old, old + b"public addition" * 32)]


def run():
    training = fixtures(8108, 32768)
    calibration = {codec: sum(len(make_patch(a, b, descriptor(a), descriptor(b), codec))
                              for _, a, b in training) for codec in ("replace", "xor")}
    # Choice freezes here, before held-out fixtures are constructed or evaluated.
    chosen = min(calibration, key=lambda codec: (calibration[codec], codec))
    rows = []
    with tempfile.TemporaryDirectory(prefix="artifact-delta-") as directory:
        root = Path(directory)
        for name, old, new in fixtures(9108, 262144):
            base_row, new_row = descriptor(old), descriptor(new)
            patch = make_patch(old, new, base_row, new_row, chosen)
            (root / "base").write_bytes(old)
            result = reconstruct(root / "base", io.BytesIO(patch), root / "new",
                                 base_row=base_row, new_row=new_row, patch_sha256=sha(patch))
            assert (root / "new").read_bytes() == new
            assert (root / "base").read_bytes() == old
            controls = full_controls(new)
            control = min(controls.values())
            base_control = min(full_controls(old).values())
            manifest = len(canonical(new_row))
            request = len(canonical({"base": base_row["sha256"], "new": new_row["sha256"]}))
            offer = len(canonical({"base": base_row["sha256"], "new": new_row["sha256"],
                                   "patch_sha256": sha(patch), "patch_size": len(patch)}))
            base_delivery = base_control + len(canonical(base_row))
            delta = len(patch) + request + offer
            savings = control - delta
            rows.append({"case": name, "synthetic": True, "base": base_row, "new": new_row,
                         "full_controls_bytes": controls, "strongest_full_bytes": control,
                         "production_zlib1_1MiB_bytes": controls["zlib-1-1048576"],
                         "new_manifest_bytes_common": manifest, "delta_request_bytes": request,
                         "authenticated_patch_offer_bytes": offer,
                         "patch_bytes_including_table": len(patch), "patch_sha256": sha(patch),
                         "delta_hit_bytes": delta + manifest, "full_bytes": control + manifest,
                         "cold_base_then_delta_bytes": base_delivery + delta + manifest,
                         "cold_base_then_full_bytes": base_delivery + control + manifest,
                         "delta_miss_then_full_bytes": delta + control + manifest,
                         "net_hit_saving_bytes": savings, "base_first_delivery_bytes": base_delivery,
                         "break_even_hit_rate": delta / control,
                         "extra_base_amortization_updates": base_delivery // savings + 1 if savings > 0 else None,
                         "exact_reconstruction": result,
                         "disk_payload_owners": {"base_cache": len(old), "new_pending_or_cache": len(new),
                             "existing_destination_if_any": len(new), "patch_spool_optional": len(patch),
                             "old_native_snapshot_if_live": len(old), "new_native_import_if_copied": len(new)},
                         "publisher_payloads": {"old_source": len(old), "new_source": len(new),
                                                "patch": len(patch)}})
    sources = ["python/pllm/runtime/bundle_artifacts.py", "python/pllm/runtime/bundle_compression.py",
               "docs/research/native-snapshot-reference.md", "docs/research/qwen3-sdk-artifacts.md",
               "docs/plans/low-tier-all-link-hypotheses.md", "scripts/probe_artifact_update_delta.py",
               "tests/test_artifact_update_delta_probe.py", "docs/research/artifact-update-delta-reference.md"]
    return {"schema": 1, "status": "synthetic-only; actual-pair evidence missing",
            "runtime": {"python": sys.version, "zlib": zlib.ZLIB_RUNTIME_VERSION},
            "whole_client_low_eligibility": "unknown", "timing_rss_or_model_quality_claim": False,
            "authentication": "externally trusted old/new manifest rows and patch commitment required",
            "actual_pair_search": {"scope": "directory entries only; not exhaustive cache scan",
                "roots": ["target/prepared-frontier-artifacts", "target/qwen3-memory-artifacts"],
                "credible_authenticated_pair_found": False},
            "freeze": {"training_seed": 8108, "held_out_seed": 9108, "training_object_bytes": 32768,
                       "held_out_object_bytes": 262144, "calibration_patch_bytes": calibration,
                       "selected_codec": chosen, "block": BLOCK, "zlib_level": 9},
            "bounds": {"object_bytes": MAX_OBJECT, "header_bytes": MAX_HEADER, "patch_bytes": MAX_PATCH,
                       "client_explicit_live_buffers_conservative_bytes": 8 * BLOCK + 4 * MAX_HEADER,
                       "excluded": "Python/runtime/zlib state, filesystem cache, native import and full client"},
            "accounting": {"scope": "single publisher-to-client application bodies; other links unchanged, unmeasured",
                "hit_formula": "manifest + request + authenticated-patch-offer + patch",
                "miss_formula": "manifest + request + offer + attempted patch + full; worst complete attempt",
                "hit_rate_gate": "h > (patch + request + offer + per-update-extra-setup)/full",
                "updates_gate": "U*(h*full-patch-request-offer-extra-setup) > extra-base-first-delivery+one-time-distribution",
                "setup": "No learned tables. Calibration local public synthetic inputs; CPU/scratch unmeasured. No free real-base assumption.",
                "unchanged_objects": "existing object/snapshot reuse; zero delta opportunity counted",
                "transport_exclusions": "TLS, headers, retries outside explicit miss, TCP framing, checkpoint distribution"},
            "source_and_own_code_sha256": {path: sha(Path(path).read_bytes()) for path in sources},
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
