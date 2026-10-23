"""Bounded, read-only inspection of pinned Maverick-candidate BF16 scale slices.

Stdlib plus the existing pure metadata/parser/file owners only. No model loader,
PLLM/native model code, production decoder, serving or numeric profile is adopted.
HTTPS/pinned inputs remain trusted; ranges do not verify whole-shard LFS hashes.
Caller-supplied prior consumption is accounting, not a source acceptance rule.
Run only with explicit source-I/O authorization; importing this module does no I/O.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import struct
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from scripts import probe_gkm_checkpoint_bounds as metadata


PIN = "96b30dc13593a244a5e59e84687309f53c375cfa"
ASSET = "model-00001-of-00002.safetensors"
CANONICAL = f"https://huggingface.co/Qwen/Qwen3-4B-FP8/resolve/{PIN}/{ASSET}"
FILE_SIZE = 4_412_140_424
HEADER_BYTES = 76_544
HEADER_SHA256 = "9203d0277e3986e90216c106ebdba821ea34acef7fcea49e88b5e5660def7cca"
BODY_LIMIT = 4 << 20
REQUEST_LIMIT = 300
MAX_RANGE = 64 << 10
MAX_BF16_WORD = 0x5600  # Exactly the existing 2**45 premise, not a new bound.
SHAPES = {"self_attn.q_proj": [32, 20], "self_attn.k_proj": [8, 20],
          "self_attn.v_proj": [8, 20], "self_attn.o_proj": [20, 32],
          "mlp.gate_proj": [76, 20], "mlp.up_proj": [76, 20], "mlp.down_proj": [20, 76]}
EXPECTED = {f"model.layers.{layer}.{role}.weight_scale_inv": shape
            for layer in range(36) for role, shape in SHAPES.items()}


class Reject(ValueError):
    pass


def find_cached_shard(roots=None):
    """Only the exact official snapshot; never scan other model/cache payloads."""
    if roots is None:
        roots = {Path.home() / ".cache/huggingface/hub"}
        for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            if os.environ.get(key):
                roots.add(Path(os.environ[key]).expanduser())
        if os.environ.get("HF_HOME"):
            roots.add(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    relative = Path("models--Qwen--Qwen3-4B-FP8/snapshots") / PIN / ASSET
    return next((Path(root) / relative for root in sorted(roots, key=str)
                 if (Path(root) / relative).is_file()), None)


def parse_header(source):
    """Reuse the bounded parser, with a scoped metadata-width entry, no decoder."""
    sizes = metadata.SIZES
    try:
        metadata.SIZES = {**sizes, "F8_E4M3": 1}
        return metadata.parse_header(source)
    finally:
        metadata.SIZES = sizes  # No permanent mutation of the shared owner.


def scale_groups(rows):
    actual = {name: row for name, row in rows.items() if name.endswith(".weight_scale_inv")}
    if set(actual) != set(EXPECTED) or len(rows) != 650:
        raise Reject("first-shard scale/tensor inventory mismatch")
    intervals = []
    for name, shape in EXPECTED.items():
        row = actual[name]
        if row["dtype"] != "BF16" or row["shape"] != shape:
            raise Reject("scale dtype/shape mismatch")
        intervals.append((row["absolute_offset"], row["elements"] * 2, name))
    if sum(length for _, length, _ in intervals) != 443_520:
        raise Reject("scale byte count mismatch")
    groups = []
    for start, length, name in sorted(intervals):
        if groups and groups[-1][0] + groups[-1][1] == start and groups[-1][1] + length <= MAX_RANGE:
            groups[-1][1] += length
            groups[-1][2].append((start, length, name))
        else:
            groups.append([start, length, [(start, length, name)]])
    return groups


def summarize_words(raw):
    """Audit BF16 words, not a model tensor decoder; never use host floats."""
    if len(raw) % 2:
        raise Reject("partial BF16 word")
    counts, low, high = Counter(), None, None
    for (word,) in struct.iter_unpack("<H", raw):
        exponent = (word >> 7) & 255
        if exponent == 255:
            category = "nan" if word & 127 else "infinity"
        elif word & 0x7FFF == 0:
            category = "negative_zero" if word >> 15 else "positive_zero"
        elif word >> 15:
            category = "negative_finite"
        elif word > MAX_BF16_WORD:
            category = "above_2_to_45"
        else:
            category = "positive_finite_within_bound"
        counts[category] += 1
        if 0 < word < 0x7F80:
            low = word if low is None else min(low, word)
            high = word if high is None else max(high, word)
    return counts, low, high


def exact_positive(word):
    if word is None:
        return None
    exponent, fraction = (word >> 7) & 255, word & 127
    n, shift = (fraction, -133) if exponent == 0 else (128 + fraction, exponent - 134)
    value = Fraction(n << shift) if shift >= 0 else Fraction(n, 1 << -shift)
    return {"bf16_word_hex": f"0x{word:04x}", "exact_fraction": str(value)}


class NoReadAhead(http.client.HTTPResponse):
    def __init__(self, sock, *args, **kwargs):
        super().__init__(sock, *args, **kwargs)
        self.fp.close()
        self.fp = sock.makefile("rb", buffering=0)


class Connection(http.client.HTTPSConnection):
    response_class = NoReadAhead


class HTTPS(HTTPSHandler):
    def https_open(self, req):
        return self.do_open(Connection, req, context=ssl.create_default_context())


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None  # Never let urllib drain redirect bodies.


class Audit:
    def __init__(self, prior=0, *, prior_requests=0, opener=None, cache_roots=None):
        if type(prior) is not int or not 0 <= prior <= BODY_LIMIT:
            raise Reject("invalid prior byte accounting")
        if type(prior_requests) is not int or not 0 <= prior_requests <= REQUEST_LIMIT:
            raise Reject("invalid prior request accounting")
        self.prior, self.prior_requests = prior, prior_requests
        self.body = prior  # Conservative inspection cap includes cache reads too.
        self.network_bytes = self.cache_bytes = self.requests = 0
        self.events, self.results = [], []
        self.counts = Counter()
        self.minimum = self.maximum = self.final_url = None
        self.allowed_scales = set()
        self.opener = opener if opener is not None else build_opener(ProxyHandler({}), Redirects(), HTTPS())
        self.cache_roots, self.cached, self.stack = cache_roots, None, None

    def check(self):
        """Light budget guard for the shared InputFile owner; no host/model probe."""
        if self.body > BODY_LIMIT or self.requests + self.prior_requests > REQUEST_LIMIT:
            raise Reject("inspection budget exceeded")

    def __enter__(self):
        self.stack = ExitStack()
        try:
            path = find_cached_shard(self.cache_roots)
            if path is not None:
                self.cached = self.stack.enter_context(metadata.InputFile(path, Counter(), self))
                if self.cached.size != FILE_SIZE:
                    raise Reject("cached shard size mismatch; no network fallback")
            return self
        except BaseException:
            self.stack.close()
            self.cached = None
            raise

    def __exit__(self, *args):
        try:
            return self.stack.__exit__(*args)
        finally:
            self.cached = None

    def fetch(self, start, length, kind):
        if kind == "header":
            if (start, length) not in {(0, 8), (8, HEADER_BYTES)}:
                raise Reject("unexpected header slice")
            url = CANONICAL
        elif kind == "scales" and (start, length) in self.allowed_scales:
            url = self.final_url
        else:
            raise Reject("non-scale payload read refused")
        return self._read_range(start, length, kind, url)

    def _remaining_body_budget(self):
        return BODY_LIMIT - self.body

    def _before_request(self):
        if self.requests + self.prior_requests >= REQUEST_LIMIT:
            raise Reject("request budget exhausted")

    def _read_range(self, start, length, kind, url):
        """Shared bounded transport; callers must first enforce their named scope."""
        if not 0 < length <= max(MAX_RANGE, HEADER_BYTES) or length > self._remaining_body_budget():
            raise Reject("body budget exceeded before read/request")
        end = start + length - 1
        if not 0 <= start <= end < FILE_SIZE:
            raise Reject("out-of-file slice")
        if self.cached is not None:
            self.cached.check_identity()
            before = sum(self.cached.io.values())
            try:
                raw = self.cached.read(start, length, kind)
            finally:
                consumed = sum(self.cached.io.values()) - before
                self.body += consumed
                self.cache_bytes += consumed
                self.events.append({"kind": kind, "start": start, "end_inclusive": end,
                                    "transport": "cache", "inspected_bytes": consumed})
            self.cached.check_identity()
            return raw
        if not url:
            raise Reject("no pinned redirect binding for scale reads")
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        for _ in range(8):
            host = urlsplit(url)
            if (host.scheme != "https" or host.username or host.password or host.port not in (None, 443)
                    or not (host.hostname == "huggingface.co"
                            or (host.hostname or "").endswith(".huggingface.co")
                            or (host.hostname or "").endswith(".hf.co"))):
                raise Reject("redirect outside public official hosts")
            self._before_request()
            self.requests += 1
            try:
                response = self.opener.open(Request(url, headers=headers), timeout=30)
            except HTTPError as error:
                response = error
            with response:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise Reject("redirect without location; no body read")
                    url = urljoin(url, location)
                    continue
                event = {"kind": kind, "start": start, "end_inclusive": end,
                         "status": response.status, "body_bytes": 0}
                self.events.append(event)
                if response.status != 206:
                    raise Reject("HTTP status is not 206; response closed before body")
                lengths = response.headers.get_all("Content-Length", [])
                ranges = response.headers.get_all("Content-Range", [])
                encodings = response.headers.get_all("Content-Encoding", [])
                if (len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0])
                        or int(lengths[0]) != length
                        or ranges != [f"bytes {start}-{end}/{FILE_SIZE}"]
                        or response.headers.get("Transfer-Encoding")
                        or encodings not in ([], ["identity"])):
                    raise Reject("range/size/encoding mismatch; response closed before body")
                if length > self._remaining_body_budget():
                    raise Reject("aggregate cap exceeded before body")
                chunks, remaining = [], length
                while remaining:
                    chunk = response.read(min(16 << 10, remaining, self._remaining_body_budget()))
                    if not chunk:
                        raise Reject("truncated bounded body")
                    self.body += len(chunk)
                    self.network_bytes += len(chunk)
                    event["body_bytes"] += len(chunk)
                    remaining -= len(chunk)
                    chunks.append(chunk)
                self.final_url = url  # Never output signed CDN query credentials.
                return b"".join(chunks)
        raise Reject("redirect limit reached")

    def run(self):
        with self:
            self._inspect()

    def _inspect(self):
        audit = self

        class Source:
            size = FILE_SIZE

            def read(self, offset, length, purpose):
                if purpose != "headers":
                    raise Reject("nonheader parser read refused")
                raw = audit.fetch(offset, length, "header")
                if offset == 0 and int.from_bytes(raw, "little") != HEADER_BYTES:
                    raise Reject("pinned header length mismatch")
                if offset == 8 and hashlib.sha256(raw).hexdigest() != HEADER_SHA256:
                    raise Reject("pinned header hash mismatch")
                return raw

        rows, _ = parse_header(Source())
        groups = scale_groups(rows)
        self.allowed_scales = {(start, length) for start, length, _ in groups}
        for start, length, members in groups:
            raw = self.fetch(start, length, "scales")
            for offset, count, name in members:
                local, low, high = summarize_words(raw[offset-start:offset-start+count])
                self.counts.update(local)
                if low is not None:
                    self.minimum = low if self.minimum is None else min(self.minimum, low)
                    self.maximum = high if self.maximum is None else max(self.maximum, high)
                self.results.append({"name": name, "start": offset, "bytes": count,
                                     "elements": sum(local.values()), "counts": dict(local)})
        if len(self.results) != 252 or sum(self.counts.values()) != 221760:
            raise Reject("incomplete aggregate scale inspection")

    def output(self, error=None):
        complete = (len(self.results) == 252 and {r["name"] for r in self.results} == set(EXPECTED)
                    and sum(self.counts.values()) == 221760)
        passed = complete and self.counts["positive_finite_within_bound"] == 221760
        return {"repository": "Qwen/Qwen3-4B-FP8", "revision": PIN, "asset": ASSET,
                "result": "observed_all_scales_satisfy_bound" if passed and error is None else "unresolved_or_failed",
                "safe_error": error, "inspected_tensors": len(self.results),
                "inspected_elements": sum(self.counts.values()), "counts": dict(self.counts),
                "positive_minimum": exact_positive(self.minimum), "positive_maximum": exact_positive(self.maximum),
                "prior_body_bytes_debited": self.prior, "new_body_bytes": self.network_bytes,
                "aggregate_body_bytes": self.prior + self.network_bytes, "body_limit": BODY_LIMIT,
                "new_cache_bytes": self.cache_bytes, "aggregate_inspection_bytes": self.body,
                "new_http_requests_including_redirects": self.requests,
                "prior_requests_debited": self.prior_requests,
                "aggregate_requests": self.requests + self.prior_requests,
                "request_limit": REQUEST_LIMIT, "ranges": self.events,
                "tensor_ranges_and_counts": self.results,
                "whole_shard_integrity": "unverified", "source_no_wrap": "unproved",
                "head_embedding_equality": "unverified", "numeric_or_runtime_adoption": "none"}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prior-body-bytes", type=int, required=True)
    cli.add_argument("--prior-requests", type=int, required=True)
    args = cli.parse_args(argv)
    audit = None
    error = None
    try:
        audit = Audit(args.prior_body_bytes, prior_requests=args.prior_requests)
        audit.run()
    except Exception as exc:
        error = str(exc) if isinstance(exc, Reject) else type(exc).__name__ + ": details suppressed"
    result = audit.output(error) if audit is not None else {"result": "unresolved_or_failed", "safe_error": error}
    print(json.dumps(result, indent=2))
    return 0 if result["result"] == "observed_all_scales_satisfy_bound" else 2


if __name__ == "__main__":
    raise SystemExit(main())
