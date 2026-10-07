import struct

import pytest

from pllm import _native
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.protocols import PrivateEmbeddingLookup


def test_optional_options_preserve_defaults_and_replay():
    component = PrivateEmbeddingLookup()
    assert component.to_spec()["params"] == {"vocabulary_size": 64, "precision": "mixed_4_3_3"}
    changed = component.with_params(repacking="interleave", conversion="party_local")
    assert changed != component
    for kwargs in ({"repacking": "PackLWEs"}, {"conversion": "SiRNN"}):
        with pytest.raises(ConfigurationError):
            PrivateEmbeddingLookup(**kwargs)


@pytest.mark.parametrize("precision", ["int8", "int4", "mixed_4_3_3"])
@pytest.mark.parametrize("repacking", ["interleave", "pack_lwes"])
@pytest.mark.parametrize("rows,columns", [(1, 7), (256, 64), (4096, 7)])
def test_encrypted_repacking_and_party_conversion_match_signed_rows(precision, rows, columns, repacking):
    pytest.importorskip("_sealapi_cpp")
    from pllm.metrics.packed_embedding import _WIDTHS
    widths = _WIDTHS[precision]
    table = [((i * 19 + i//columns * 7) % (1 << widths[i % columns % len(widths)]))
             - (1 << (widths[i % columns % len(widths)] - 1)) for i in range(rows*columns)]
    component = PrivateEmbeddingLookup(vocabulary_size=rows, precision=precision,
                                       repacking=repacking, conversion="party_local")
    report = benchmark_operator(component, lanes=columns, samples=1, inputs={"table": table, "token": [rows-1]})
    assert report.quality.exact_integer_outputs == columns
    replay = replay_operator_benchmark(report.configuration)
    assert replay.configuration_digest == report.configuration_digest
    assert replay.resources == report.resources
    phases = report.samples[0].peer_frame_bytes_by_phase
    groups = (columns+len(widths)-1)//len(widths)
    assert len(phases) == 1+groups*2
    assert tuple(sum(p[i] for p in phases) for i in (0,1)) == report.resources.peer_frame_bytes_by_direction
    assert phases[0][0] == phases[0][1]  # One request and one repacked reply.
    assert int(dict(report.environment)["minimum_bfv_noise_budget_bits"]) > 0


@pytest.mark.parametrize("precision", ["int8", "int4", "mixed_4_3_3"])
@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_dense_constant_extreme_tables_and_zero_ciphertexts_preserve_noise(precision, sign):
    pytest.importorskip("_sealapi_cpp")
    from pllm.metrics.packed_embedding import _WIDTHS
    widths = _WIDTHS[precision]
    table = [sign * ((1 << (widths[i%64%len(widths)]-1)) - (sign > 0)) for i in range(256*64)]
    for repacking in ("interleave", "pack_lwes"):
        report = benchmark_operator(PrivateEmbeddingLookup(vocabulary_size=256, precision=precision,
            repacking=repacking, conversion="party_local"), lanes=64, samples=1,
            inputs={"table":table, "token":[0]})
        assert report.quality.exact_integer_outputs == 64
        assert int(dict(report.environment)["minimum_bfv_noise_budget_bits"]) > 0


def test_party_input_type_and_foreign_context_burn_before_conversion():
    def issue(session):
        return _native.packed_embedding_issue_parties("mixed_4_3_3", bytes(32), session, 0)[:2]
    a,b = issue(bytes(32))
    with pytest.raises((TypeError,ValueError)):
        a.start(1)
    with pytest.raises(ValueError, match="consumed"):
        a.start(bytes(4))
    b.cancel()
    a,b = issue(bytes(32))
    x,y = issue(bytes([1])*32)
    a.start(bytes(4))
    foreign = y.start(bytes(4))
    with pytest.raises(ValueError):
        a.advance(foreign)
    with pytest.raises(ValueError,match="consumed"):
        a.advance(foreign)
    for p in (a,b,x,y):
        p.cancel()


def test_native_rns_rejects_noncanonical_ciphertext_and_geometry():
    with pytest.raises(ValueError):
        _native.coefficient_repacking_divide(bytes(2*4096*8), [16], 2)
    with pytest.raises(ValueError):
        _native.coefficient_repacking_divide(struct.pack("<Q", 17)*8192, [17], 2)
    with pytest.raises(ValueError):
        _native.coefficient_repacking_divide(bytes(2*4096*8), [17], 3)


@pytest.mark.parametrize("repacking", ["interleave", "pack_lwes"])
def test_repacking_admission_precedes_material(monkeypatch, repacking):
    def forbidden(*args, **kwargs):
        raise AssertionError("native allocation before admission")
    monkeypatch.setattr(_native, "packed_embedding_table", forbidden)
    with pytest.raises(ConfigurationError, match="allocation"):
        benchmark_operator(PrivateEmbeddingLookup(vocabulary_size=4096, precision="int8",
            repacking=repacking, conversion="party_local"), lanes=256, samples=1)
