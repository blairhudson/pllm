"""Fake transport/cache and exact BF16 audit tests; never contact a model source."""
from collections import Counter
from email.message import Message
import importlib
import importlib.abc
import io
import json
from pathlib import Path
import struct
import sys
from urllib.error import HTTPError

import pytest


@pytest.fixture(scope="module")
def source():
    class NoRuntime(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {
                "pllm", "torch", "numpy", "transformers", "safetensors", "tokenizers", "psutil",
            }:
                raise AssertionError("research module imported runtime/host probe: " + fullname)
            return None

    guard = NoRuntime()
    root = str(Path(__file__).resolve().parents[1])
    before = tuple(sys.meta_path)
    sys.path.insert(0, root)
    sys.meta_path.insert(0, guard)
    try:
        module = importlib.import_module("benchmarks.research.maverick_scale_source")
        assert Path(module.__file__).resolve() == Path(root) / "benchmarks/research/maverick_scale_source.py"
        assert tuple(sys.meta_path) == (guard, *before)  # no persistent import trap installed by module
        yield module
    finally:
        sys.path.remove(root)
        sys.meta_path.remove(guard)


@pytest.fixture(autouse=True)
def forbid_network(source, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real HTTP connection forbidden in unit tests")
    monkeypatch.setattr(source.Connection, "connect", forbidden)


class Reply:
    def __init__(self, source, *, status=206, start=0, body=b"12345678", headers=None):
        self.status, self.body = status, body
        self.headers = Message()
        initial = [("Content-Length", str(len(body))),
                   ("Content-Range", f"bytes {start}-{start+len(body)-1}/{source.FILE_SIZE}")]
        for key, value in initial if headers is None else headers:
            self.headers[key] = value
        self.reads, self.closed = [], False

    def read(self, count):
        self.reads.append(count)
        block, self.body = self.body[:count], self.body[count:]
        return block

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class Opener:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def open(self, request, timeout):
        self.calls.append(request)
        if not self.replies:
            raise AssertionError("unexpected network/fallback request")
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def cache_file(source, root, raw=b"12345678"):
    path = root / "models--Qwen--Qwen3-4B-FP8/snapshots" / source.PIN / source.ASSET
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    return path


@pytest.mark.parametrize("word,category", [
    (0x0000, "positive_zero"), (0x8000, "negative_zero"),
    (0x0001, "positive_finite_within_bound"), (0x3F80, "positive_finite_within_bound"),
    (0x5600, "positive_finite_within_bound"), (0x5601, "above_2_to_45"),
    (0x7F7F, "above_2_to_45"), (0xBF80, "negative_finite"),
    (0x7F80, "infinity"), (0xFF80, "infinity"),
    (0x7F81, "nan"), (0x7FC0, "nan"), (0xFFFF, "nan"),
])
def test_bf16_edge_values_are_classified_without_floats(source, word, category):
    counts, low, high = source.summarize_words(struct.pack("<H", word))
    assert counts == Counter({category: 1})
    assert (low, high) == ((word, word) if 0 < word < 0x7F80 else (None, None))


def test_exact_extrema_and_partial_word(source):
    assert source.exact_positive(0x0001)["exact_fraction"] == "1/10889035741470030830827987437816582766592"
    assert source.exact_positive(0x376D)["exact_fraction"] == "237/16777216"
    assert source.exact_positive(0x3B35)["exact_fraction"] == "181/65536"
    assert source.exact_positive(0x5600)["exact_fraction"] == str(1 << 45)
    assert source.exact_positive(None) is None
    with pytest.raises(source.Reject, match="partial"):
        source.summarize_words(b"\x01")


@pytest.mark.parametrize("status", [200, 403, 404, 500])
def test_non_206_closes_before_any_body_read(source, status):
    response = Reply(source, status=status)
    audit = source.Audit(opener=Opener(response), cache_roots=())
    with pytest.raises(source.Reject, match="not 206"):
        audit.fetch(0, 8, "header")
    assert response.closed and response.reads == []
    assert audit.body == audit.network_bytes == 0


@pytest.mark.parametrize("headers", [
    [], [("Content-Length", "7")], [("Content-Length", "9999999999")],
    [("Content-Length", "8"), ("Content-Length", "8")],
    [("Content-Length", "8"), ("Content-Range", "bytes 0-7/1")],
    [("Content-Length", "8"), ("Content-Range", "bytes 1-8/4412140424")],
    [("Content-Length", "8"), ("Content-Range", "bytes 0-7/4412140424"), ("Transfer-Encoding", "chunked")],
    [("Content-Length", "8"), ("Content-Range", "bytes 0-7/4412140424"), ("Content-Encoding", "gzip")],
    [("Content-Length", "8"), ("Content-Range", "bytes 0-7/4412140424"),
     ("Content-Encoding", "identity"), ("Content-Encoding", "gzip")],
])
def test_range_length_encoding_rejection_precedes_read(source, headers):
    response = Reply(source, headers=headers)
    audit = source.Audit(opener=Opener(response), cache_roots=())
    with pytest.raises(source.Reject, match="range/size"):
        audit.fetch(0, 8, "header")
    assert response.closed and response.reads == [] and audit.body == 0


def test_body_and_request_caps_precede_transport(source):
    opener = Opener()
    for audit in (source.Audit(source.BODY_LIMIT - 7, opener=opener),
                  source.Audit(prior_requests=source.REQUEST_LIMIT, opener=opener)):
        with pytest.raises(source.Reject, match="budget"):
            audit.fetch(0, 8, "header")
    assert opener.calls == []
    assert source.Audit(0, prior_requests=0).prior == 0  # no historical acceptance gate


def test_redirects_close_without_body_and_final_url_stays_private(source):
    redirect = Reply(source, status=302, headers=[("Location", "https://cdn-lfs.huggingface.co/object?token=fake-secret")])
    final = Reply(source)
    audit = source.Audit(opener=Opener(redirect, final), cache_roots=())
    assert audit.fetch(0, 8, "header") == b"12345678"
    assert redirect.closed and redirect.reads == [] and final.closed
    assert audit.requests == 2 and audit.body == 8
    assert "fake-secret" not in json.dumps(audit.output())
    assert source.Redirects().redirect_request(None, None, None, None, None, None) is None


@pytest.mark.parametrize("location", ["http://huggingface.co/object", "https://example.com/object", "https://user@huggingface.co/object"])
def test_nonpublic_redirects_are_refused_without_second_request(source, location):
    redirect = Reply(source, status=302, headers=[("Location", location)])
    opener = Opener(redirect)
    audit = source.Audit(opener=opener, cache_roots=())
    with pytest.raises(source.Reject, match="public official"):
        audit.fetch(0, 8, "header")
    assert len(opener.calls) == 1 and redirect.reads == [] and redirect.closed


def test_http_error_redirect_has_no_automatic_body_drain(source):
    class Body(io.BytesIO):
        reads = 0

        def read(self, *args):
            self.reads += 1
            return super().read(*args)

    body = Body(b"must not read")
    headers = Message()
    headers["Location"] = "https://cdn-lfs.huggingface.co/object"
    error = HTTPError(source.CANONICAL, 302, "redirect", headers, body)
    audit = source.Audit(opener=Opener(error, Reply(source)), cache_roots=())
    assert audit.fetch(0, 8, "header") == b"12345678"
    assert body.closed and body.reads == 0


def test_redirect_request_budget_and_stream_cap(source):
    redirect = Reply(source, status=302, headers=[("Location", source.CANONICAL)])
    opener = Opener(redirect)
    audit = source.Audit(prior_requests=299, opener=opener, cache_roots=())
    with pytest.raises(source.Reject, match="request budget"):
        audit.fetch(0, 8, "header")
    assert len(opener.calls) == 1 and audit.body == 0
    response = Reply(source, start=8, body=b"x" * source.HEADER_BYTES)
    audit = source.Audit(opener=Opener(response), cache_roots=())
    assert len(audit.fetch(8, source.HEADER_BYTES, "header")) == source.HEADER_BYTES
    assert max(response.reads) <= 16 << 10
    assert audit.body == audit.network_bytes == source.HEADER_BYTES


def test_truncated_body_retains_actual_consumption(source):
    response = Reply(source)
    response.body = b"123"
    audit = source.Audit(opener=Opener(response), cache_roots=())
    with pytest.raises(source.Reject, match="truncated"):
        audit.fetch(0, 8, "header")
    assert response.closed and audit.body == audit.network_bytes == 3
    assert audit.output()["result"] == "unresolved_or_failed"


def test_unauthorized_weight_payload_never_opens_transport(source):
    opener = Opener()
    audit = source.Audit(opener=opener, cache_roots=())
    for start, length, kind in [(0, 9, "header"), (100, 2, "scales"), (0, 8, "weights")]:
        with pytest.raises(source.Reject):
            audit.fetch(start, length, kind)
    assert opener.calls == []


def test_cache_preferred_exact_snapshot_and_shared_file_owner(source, monkeypatch, tmp_path):
    monkeypatch.setattr(source, "FILE_SIZE", 8)
    path = cache_file(source, tmp_path)
    assert source.find_cached_shard([tmp_path]) == path
    opener = Opener()
    audit = source.Audit(opener=opener, cache_roots=[tmp_path])
    with audit:
        assert isinstance(audit.cached, source.metadata.InputFile)
        assert audit.fetch(0, 8, "header") == b"12345678"
    assert audit.cached is None and opener.calls == []
    assert audit.cache_bytes == audit.body == 8 and audit.network_bytes == 0
    assert audit.output()["aggregate_inspection_bytes"] == 8


def test_bad_or_mutating_cache_rejects_without_http_fallback(source, monkeypatch, tmp_path):
    monkeypatch.setattr(source, "FILE_SIZE", 9)
    path = cache_file(source, tmp_path)
    opener = Opener()
    with pytest.raises(source.Reject, match="size mismatch"):
        with source.Audit(opener=opener, cache_roots=[tmp_path]):
            pytest.fail("wrong-size source entered")
    monkeypatch.setattr(source, "FILE_SIZE", 8)
    with source.Audit(opener=opener, cache_roots=[tmp_path]) as audit:
        path.write_bytes(b"changed!!")
        with pytest.raises(source.metadata.Rejected, match="changed"):
            audit.fetch(0, 8, "header")
    assert opener.calls == []


def test_cache_cap_precedes_pread(source, monkeypatch, tmp_path):
    monkeypatch.setattr(source, "FILE_SIZE", 8)
    cache_file(source, tmp_path)
    with source.Audit(source.BODY_LIMIT - 7, opener=Opener(), cache_roots=[tmp_path]) as audit:
        monkeypatch.setattr(audit.cached, "read", lambda *args: pytest.fail("pread before cap"))
        with pytest.raises(source.Reject, match="body budget"):
            audit.fetch(0, 8, "header")


def test_existing_pure_header_parser_is_reused_and_width_restored(source):
    header = json.dumps({
        "scale": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        "weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [2, 3]},
    }).encode()
    header += b" " * (-len(header) % 8)
    raw = struct.pack("<Q", len(header)) + header

    class TinySource:
        size = len(raw) + 3

        def read(self, start, count, purpose):
            assert purpose == "headers" and start + count <= len(raw)
            return raw[start:start+count]

    original = source.metadata.SIZES
    rows, observation = source.parse_header(TinySource())
    assert rows["weight"]["elements"] == 1 and observation["tensor_count"] == 2
    assert source.metadata.SIZES is original and "F8_E4M3" not in original
    with pytest.raises(source.metadata.Rejected):
        source.parse_header(type("Broken", (), {"read": lambda *args: b"\0" * 8})())
    assert source.metadata.SIZES is original


def test_http_adapter_disables_buffered_body_read_ahead(source):
    class Socket:
        calls = []
        files = []

        def makefile(self, mode, buffering=-1):
            self.calls.append((mode, buffering))
            stream = io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nSECRET")
            self.files.append(stream)
            return stream

    sock = Socket()
    response = source.NoReadAhead(sock)
    response.begin()
    assert sock.calls == [("rb", -1), ("rb", 0)] and sock.files[0].closed
    assert response.fp.tell() == len(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n")
    response.close()  # no body read even though server ignored Range


def test_default_cache_lookup_uses_configured_root_without_other_models(source, monkeypatch, tmp_path):
    path = cache_file(source, tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(source.Path, "home", lambda: tmp_path / "empty-home")
    assert source.find_cached_shard() == path


def test_real_cached_dispatch_rejects_tiny_source_as_incomplete(source, monkeypatch, tmp_path):
    # Synthetic metadata/words, not model values; the extra FP8 sentinel is never read.
    header = json.dumps({
        "tiny_scale": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
        "weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [4, 5]},
    }).encode()
    header += b" " * (-len(header) % 8)
    start = len(header) + 8
    raw = struct.pack("<Q", len(header)) + header + struct.pack("<HH", 0x3F80, 0x5600) + b"\xff"
    cache_file(source, tmp_path, raw)
    monkeypatch.setattr(source, "FILE_SIZE", len(raw))
    monkeypatch.setattr(source, "HEADER_BYTES", len(header))
    monkeypatch.setattr(source, "HEADER_SHA256", source.hashlib.sha256(header).hexdigest())
    monkeypatch.setattr(source, "scale_groups", lambda rows: [[start, 4, [(start, 4, "tiny_scale")]]])
    audit = source.Audit(opener=Opener(), cache_roots=[tmp_path])
    with pytest.raises(source.Reject, match="incomplete"):
        audit.run()
    assert audit.counts == Counter({"positive_finite_within_bound": 2})
    assert audit.body == audit.cache_bytes == len(raw) - 1  # untouched weight sentinel
    assert audit.output()["result"] == "unresolved_or_failed"


def test_scale_planner_preserves_gaps_and_all_names(source):
    rows, cursor = {}, 1000
    for name, shape in source.EXPECTED.items():
        elements = shape[0] * shape[1]
        rows[name] = {"dtype": "BF16", "shape": shape, "elements": elements, "absolute_offset": cursor}
        cursor += 2 * elements + 2  # forbidden intervening bytes must not be fetched
    rows.update({f"non_scale_{n}": {} for n in range(398)})
    groups = source.scale_groups(rows)
    assert len(groups) == 252
    assert sum(count for _, count, _ in groups) == 443520
    assert {name for _, _, members in groups for _, _, name in members} == set(source.EXPECTED)
    del rows[next(iter(source.EXPECTED))]
    with pytest.raises(source.Reject, match="inventory"):
        source.scale_groups(rows)


def test_report_and_cli_keep_source_integrity_unverified(source, monkeypatch, capsys):
    audit = source.Audit(opener=Opener(), cache_roots=())
    assert audit.output()["result"] == "unresolved_or_failed"
    monkeypatch.setattr(source.Audit, "run", lambda self: (_ for _ in ()).throw(source.Reject("synthetic stop")))
    assert source.main(["--prior-body-bytes", "0", "--prior-requests", "0"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["safe_error"] == "synthetic stop"
    assert result["whole_shard_integrity"] == result["head_embedding_equality"] == "unverified"
    assert result["source_no_wrap"] == "unproved" and result["numeric_or_runtime_adoption"] == "none"
    with pytest.raises(SystemExit) as exc:
        source.main(["--help"])
    assert exc.value.code == 0
