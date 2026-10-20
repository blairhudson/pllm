"""Shared decoder metadata owner precedes inspection and clears failed owners."""
from contextlib import contextmanager
import weakref

import pytest

from pllm.deployment import resources
from pllm.runtime import delegated_source, fss_source, tokenizer_admission
from test_fss_worker import configuration


@pytest.mark.parametrize("kind", ["fss", "gkm"])
@pytest.mark.parametrize("fail", [False, True])
def test_shared_decoder_metadata_reservation_precedes_inspection_and_retires_frames(monkeypatch, kind, fail):
    events, references = [], []
    class Payload:
        pass
    @contextmanager
    def reserve(amount):
        assert amount == tokenizer_admission.DECODER_METADATA_MEMORY_BYTES == 384 << 20
        events.append("reserve")
        try:
            yield
        finally:
            assert all(reference() is None for reference in references)
            events.append("release")
    monkeypatch.setattr(resources, "reserve_incremental_memory", reserve)
    def inspect(*args, **kwargs):
        assert events == ["validate", "reserve"]
        payload = Payload()
        references.append(weakref.ref(payload))
        events.append("inspect")
        if fail:
            raise RuntimeError("metadata parse failure")
        return "resolved"
    if kind == "fss":
        monkeypatch.setattr(fss_source, "_experiment", lambda *_: events.append("validate"))
        monkeypatch.setattr(fss_source, "_resolve_client_source", inspect)
        resolve = fss_source.resolve_client_source
    else:
        monkeypatch.setattr(delegated_source, "_experiment_bytes", lambda *_: events.append("validate"))
        monkeypatch.setattr(delegated_source, "_admit_source", inspect)
        resolve = delegated_source.admit_source
    if fail:
        with pytest.raises(RuntimeError, match="metadata parse failure"):
            resolve(configuration("decoder"))
    else:
        assert resolve(configuration("decoder")) == "resolved"
    assert events == ["validate", "reserve", "inspect", "release"]


@pytest.mark.parametrize("kind", ["fss", "gkm"])
def test_capsule_source_retains_existing_metadata_policy(monkeypatch, kind):
    monkeypatch.setattr(resources, "reserve_incremental_memory", lambda *_: pytest.fail("decoder reserve for Capsule"))
    module, inner, outer = ((fss_source, "_resolve_client_source", "resolve_client_source") if kind == "fss"
                            else (delegated_source, "_admit_source", "admit_source"))
    monkeypatch.setattr(module, inner, lambda *_, **__: "resolved")
    assert getattr(module, outer)(configuration("capsule")) == "resolved"


def test_decoder_metadata_pressure_rejects_before_source_inspection(monkeypatch):
    from pllm.deployment.network import NetworkError
    monkeypatch.setattr(fss_source, "_experiment", lambda *_: None)
    monkeypatch.setattr(fss_source, "_resolve_client_source", lambda *_, **__: pytest.fail("unreserved source inspection"))
    def reject(*_):
        raise NetworkError("CLIENT_CAPACITY_CHANGED")
    monkeypatch.setattr(resources, "reserve_incremental_memory", reject)
    with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
        fss_source.resolve_client_source(configuration("decoder"))
