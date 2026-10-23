"""Synthetic Gamma/range tests only; no checkpoint observations or real HTTP."""
from email.message import Message
import importlib
import importlib.abc
import json
from pathlib import Path
import re
import struct
import sys

import pytest


@pytest.fixture(scope="module")
def gamma():
    class NoRuntime(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"pllm", "torch", "numpy", "transformers", "safetensors", "tokenizers", "psutil"}:
                raise AssertionError("runtime import forbidden: " + fullname)
            return None
    guard = NoRuntime()
    root = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, root)
    sys.meta_path.insert(0, guard)
    try:
        yield importlib.import_module("benchmarks.research.maverick_gamma_source")
    finally:
        sys.meta_path.remove(guard)
        sys.path.remove(root)


@pytest.fixture(autouse=True)
def no_network(gamma, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real connection forbidden in offline tests")
    monkeypatch.setattr(gamma.owner.Connection, "connect", forbidden)


class Reply:
    def __init__(self, module, start, body, *, status=206, location=None):
        self.status, self.body = status, body
        self.headers = Message()
        self.headers["Content-Length"] = str(len(body))
        self.headers["Content-Range"] = f"bytes {start}-{start+len(body)-1}/{module.owner.FILE_SIZE}"
        if location:
            self.headers["Location"] = location
        self.closed, self.reads = False, []

    def read(self, n):
        self.reads.append(n)
        block, self.body = self.body[:n], self.body[n:]
        return block

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class Server:
    def __init__(self, module, redirects=0, word=0x4100):
        self.module, self.redirects, self.word = module, redirects, word
        self.calls, self.responses, self.seen = [], [], {}
        self.audit = None

    def open(self, request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        self.calls.append((start, end))
        group = next(g for g in self.audit.groups if (g["start"], g["start"]+g["length"]-1) == (start, end))
        seen = self.seen.get(start, 0)
        self.seen[start] = seen+1
        if seen < self.redirects:
            response = Reply(self.module, start, b"must-not-read", status=302,
                             location="https://cdn-lfs.huggingface.co/fixture?token=fake-secret")
        else:
            raw = bytearray(b"\xff" * group["length"])  # overlap is NaN if mistakenly classified
            for name in group["gamma_names"]:
                row = self.audit.norms[name]
                offset = row["start"] - start
                words = [0xBF80, 0x8000, 0x0000, self.word] * (row["length"]//8)
                raw[offset:offset+row["length"]] = struct.pack("<"+"H"*len(words), *words)
            response = Reply(self.module, start, bytes(raw))
        self.responses.append(response)
        return response


def fixture_audit(gamma, *, redirects=0, word=0x4100, **kwargs):
    server = Server(gamma, redirects, word)
    audit = gamma.GammaAudit(kwargs.pop("prior", gamma.PRIOR_BODY),
                             prior_requests=kwargs.pop("prior_requests", gamma.PRIOR_REQUESTS),
                             opener=server)
    server.audit = audit
    return audit, server


@pytest.mark.parametrize("redirects", [0, 1])
def test_complete_synthetic_inspection_signed_gamma_and_exact_budgets(gamma, redirects):
    audit, server = fixture_audit(gamma, redirects=redirects)
    audit.run()
    result = audit.output()
    assert result["result"] == "observed_gamma_envelope"
    assert result["maximum_absolute"]["exact_fraction"] == "8"
    assert result["sufficient_gamma_exponent_g"] == 3
    assert result["inspected_tensors"] == 145 and result["inspected_coefficients"] == 196096
    assert result["new_body_bytes"] == 507392 and result["aggregate_body_bytes"] == 1353368
    assert result["completed_group_scale_overlap_bytes"] == 115200
    assert result["new_http_attempts_including_redirects"] == 37*(redirects+1)
    assert result["aggregate_request_reservation"] == 217+37*(redirects+1)
    assert result["counts"]["negative_finite"] > 0 and result["counts"]["negative_zero"] > 0
    assert result["unread_vectors"] == [] and "fake-secret" not in json.dumps(result)
    assert all(response.closed for response in server.responses)
    assert all(not response.reads for response in server.responses if response.status == 302)
    assert result["whole_shard_integrity"] == "unverified"
    assert result["embedding_bound"] == result["source_no_wrap"] == result["whole_graph_totality"] == "unproved"
    assert result["implementation_or_SDK_authority"] == "none"


def test_excess_redirects_abort_before_remaining_request_reservation_fails(gamma):
    audit, server = fixture_audit(gamma, redirects=2)
    with pytest.raises(gamma.owner.Reject, match="cannot fit") as exc:
        audit.run()
    result = audit.output(str(exc.value))
    assert 0 < result["completed_groups"] < 37
    assert result["new_http_attempts_including_redirects"] == len(server.calls) <= 83
    assert result["aggregate_request_reservation"] <= 300
    assert result["new_body_bytes"] <= 507392
    assert result["unread_vectors"] and result["result"] == "unresolved_or_failed"
    assert all(r.closed for r in server.responses)


@pytest.mark.parametrize("word", [0x4A81, 0x7F80, 0xFF80, 0x7FC1])
def test_complete_invalid_gamma_is_not_a_certificate(gamma, word):
    audit, _ = fixture_audit(gamma, word=word)
    audit.run()
    result = audit.output()
    assert result["complete_observation"] and result["result"] == "unresolved_or_failed"
    assert result["sufficient_gamma_exponent_g"] is None
    assert result["unread_vectors"] == []


def test_exact_gamma_boundary_and_signed_zeros(gamma):
    counts, maximum = gamma.signed_gamma_summary(struct.pack("<5H", 0, 0x8000, 0xCA80, 0x4A80, 0x0001))
    assert maximum == 0x4A80 and sum(counts.values()) == 5
    assert counts["negative_finite"] == counts["negative_zero"] == counts["positive_zero"] == 1
    audit, _ = fixture_audit(gamma, word=0xCA80)
    audit.run()
    assert audit.output()["sufficient_gamma_exponent_g"] == 22
    with pytest.raises(gamma.owner.Reject, match="partial"):
        gamma.signed_gamma_summary(b"x")


@pytest.mark.parametrize("kwargs", [{"prior": 4194304-507392+1}, {"prior_requests": 264}])
def test_aggregate_budget_preflight_precedes_any_request(gamma, kwargs):
    audit, server = fixture_audit(gamma, **kwargs)
    with pytest.raises(gamma.owner.Reject, match="before first"):
        audit.run()
    assert server.calls == [] and audit.network_bytes == audit.requests == 0


@pytest.mark.parametrize("prior,requests", [(845975, 217), (845976, 216)])
def test_old_debits_cannot_be_reclaimed(gamma, prior, requests):
    with pytest.raises(gamma.owner.Reject, match="stale"):
        gamma.GammaAudit(prior, prior_requests=requests, opener=object())


@pytest.mark.parametrize("mode", ["200", "oversize", "wrongrange", "truncated", "foreignredirect"])
def test_bad_response_closes_without_refresh_or_fallback(gamma, mode):
    audit, server = fixture_audit(gamma)
    first = audit.groups[0]
    response = Reply(gamma, first["start"], b"x" * first["length"])
    if mode == "200":
        response.status = 200
    elif mode == "oversize":
        response.headers.replace_header("Content-Length", "9999999999")
    elif mode == "wrongrange":
        response.headers.replace_header("Content-Range", "bytes 0-1/2")
    elif mode == "truncated":
        response.body = b"123"
    else:
        response.status = 302
        response.headers["Location"] = "https://example.com/forbidden?token=secret"

    def opening(request, timeout):
        server.calls.append(request)
        if len(server.calls) != 1:
            pytest.fail("unexpected retry/refresh/fallback")
        return response
    audit.opener.open = opening
    with pytest.raises(gamma.owner.Reject) as exc:
        audit.run()
    result = audit.output(str(exc.value))
    assert response.closed and len(server.calls) == 1
    assert result["new_body_bytes"] == (3 if mode == "truncated" else 0)
    assert len(result["unread_vectors"]) == 145
    assert "secret" not in json.dumps(result)
    if mode != "truncated":
        assert response.reads == []


def test_read_allowlist_and_drift_fail_before_io(gamma, monkeypatch):
    audit, server = fixture_audit(gamma)
    for kind in ("header", "scales", "weights"):
        with pytest.raises(gamma.owner.Reject, match="non-plan"):
            audit.fetch(0, 8, kind)
    assert server.calls == []
    original = gamma.planner.plan

    def drifted():
        data = original()
        data["plans"]["proposed_known_scale_overlap"]["groups"][0]["start"] += 1
        return data
    monkeypatch.setattr(gamma.planner, "plan", drifted)
    with pytest.raises(gamma.owner.Reject, match="ranges changed"):
        fixture_audit(gamma)


def test_cli_requires_explicit_action_and_reports_errors(gamma, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        gamma.main(["--help"])
    assert exc.value.code == 0
    capsys.readouterr()
    monkeypatch.setattr(gamma.GammaAudit, "run", lambda self: (_ for _ in ()).throw(gamma.owner.Reject("offline stop")))
    assert gamma.main(["--inspect", "--prior-body-bytes", "845976", "--prior-requests", "217"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["new_body_bytes"] == result["new_http_attempts_including_redirects"] == 0
    assert len(result["unread_vectors"]) == 145
