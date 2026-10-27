"""Authored only: no backend execution or numerical evidence claimed."""
from fractions import Fraction
from collections import Counter
import itertools

import pytest

from pllm.runtime import encrypted_phase_index as phase


class _ClearBits:
    """Independent Boolean semantics, not an encrypted comparison claim."""
    constant = staticmethod(int)
    product = staticmethod(lambda a, b: a & b)
    xor = staticmethod(lambda a, b: a ^ b)
    negate_bit = staticmethod(lambda a: 1 - a)


def _division_oracle(p, q, cells=4096):
    # Different expression from implementation: quotient/remainder then decide
    # nearest integer by distance. Odd q makes remainder*2 == q impossible.
    integer, remainder = divmod(cells * (p % q), q)
    return (integer + int(remainder > q // 2)) % cells


def _clear_circuit(circuit, bits):
    result = phase._interpret_clear_bits(circuit, bits, _ClearBits())
    assert len(result) == 12 and set(result) <= {0, 1}
    return sum(value << i for i, value in enumerate(result))


@pytest.mark.parametrize("q", [3, 5, 17, 257])
def test_whole_phase_boolean_circuit(q):
    for b, a in [(0, (0, 0)), (q - 1, (q - 1, q - 1)), (1, (2 % q, 3 % q))]:
        circuit = phase.compile_phase_index(phase.PhaseIndexSpec(q, 2), b, a)
        for bits in itertools.product((0, 1), repeat=2):
            p = (b + sum(x * s for x, s in zip(a, bits))) % q
            assert _clear_circuit(circuit, bits) == _division_oracle(p, q)


def test_q17_n2_b1_a1_3_all_four_inputs():
    circuit = phase.compile_phase_index(phase.PhaseIndexSpec(17, 2), 1, (1, 3))
    for bits in itertools.product((0, 1), repeat=2):
        assert _clear_circuit(circuit, bits) == _division_oracle(1 + bits[0] + 3 * bits[1], 17)


def test_compiled_generic_phase_boundaries_and_wrap():
    q = (1 << 20) + 1
    phases = {0, 1, q - 2, q - 1}
    for cell in (0, 1, 2, 2047, 2048, 4094, 4095):
        floor_boundary = ((2 * cell + 1) * q) // 8192
        phases.update(p for p in (floor_boundary - 1, floor_boundary, floor_boundary + 1)
                      if 0 <= p < q)
    for b in sorted(phases):
        circuit = phase.compile_phase_index(phase.PhaseIndexSpec(q, 1), b, (q - 1,))
        for bit in (0, 1):
            assert _clear_circuit(circuit, (bit,)) == _division_oracle(b + (q - 1) * bit, q)


@pytest.mark.parametrize("bits", [(2,), (-1,), (True,), (0.0,), (None,), (), (0, 1), [0]])
def test_nonbinary_clear_inputs_rejected_before_interpretation(bits):
    circuit = phase.compile_phase_index(phase.PhaseIndexSpec(17, 1), 1, (1,))
    class ForbiddenOperations:
        def __getattr__(self, name):
            pytest.fail("malformed clear inputs reached interpreter operations")
    with pytest.raises(ValueError, match="clear"):
        phase._interpret_clear_bits(circuit, bits, ForbiddenOperations())


def test_phase_half_cell_boundaries_plus_minus_grid_units():
    # J has 4096 half-cell boundaries per canonical phase period.
    q = (1 << 20) + 1
    visited = 0
    for cell in range(4096):
        boundary = Fraction((2 * cell + 1) * q, 8192)
        base = boundary.numerator // boundary.denominator
        for delta in (-1, 0, 1):
            p = base + delta
            if 0 <= p < q:
                visited += 1
                assert phase.quantize_phase_reference(p, q) == _division_oracle(p, q)
    assert visited == 4096 * 3


def test_all_8192_local_half_cells_plus_minus_grid_units():
    # The four 2048-entry segments have 8192 shifted half-cell thresholds.
    # z_s reaches i-1/2 at X=K_s+i*B_s-1/2, between grid points.
    # Reject rather than clamp fixtures outside each canonical segment.
    visited = 0
    for s in range(4):
        width, start, end = 1 << (9 + s), (1 << 20) << s, (1 << 20) << (s + 1)
        for index in range(2048):
            visited += 1
            floor_boundary = start + index * width - 1
            for delta in (-1, 0, 1):
                x = floor_boundary + delta
                if start <= x < end:
                    # INDEX oracle comes only from integer X/start/width;
                    # never consumes candidate local_center or its numerator.
                    expected = (x - start) // width
                    assert phase.segment_index(x, 1 << 20) == (s, expected)
                    z = phase.local_center(x, s)
                    assert phase.nearest_even(z.numerator, z.denominator) == expected
    assert visited == 8192


@pytest.mark.parametrize("q", [3, 17, 4097, (1 << 61) - 1])
def test_wrap_center_and_canonical_first(q):
    for p in [0, q - 1, q, -1, -q, 7 * q + 1]:
        assert phase.quantize_phase_reference(p, q) == _division_oracle(p, q)
    assert phase.centered_phase(q - 1, q) == -1
    assert phase.centered_phase(q // 2, q) == q // 2
    assert phase.centered_phase(q // 2 + 1, q) == -(q // 2)


@pytest.mark.parametrize("n,d,want", [(-5, 2, -2), (-3, 2, -2), (-1, 2, 0),
                                      (1, 2, 0), (3, 2, 2), (5, 2, 2)])
def test_signed_nearest_even_ties(n, d, want):
    assert phase.nearest_even(n, d) == want


def test_direct_integer_grid_ties_and_shifted_local_no_ties():
    for s in range(4):
        width, start = 1 << (9 + s), (1 << 20) << s
        for index in range(2048):
            x = start + index * width + width // 2
            # Unshifted grid midpoint ties to EVEN directly from integer X.
            assert phase.nearest_even(x - start, width) == index + index % 2
            # Shifted z_s never lands on a half-integer for integer X:
            # its unreduced numerator is odd, but width B_s is even.
            numerator = 2 * (x - start) - width + 1
            assert numerator % (2 * width) != width
            z = phase.local_center(x, s)
            assert phase.nearest_even(z.numerator, z.denominator) == (x - start) // width


@pytest.mark.parametrize("q", [3, 17, 8193, (1 << 63) - 1])
def test_odd_modulus_excludes_half_ties(q):
    # A modular tie requires 8192*p = (2*k+1)*q. Left is even,
    # right odd. Independently sample every canonical p for small q.
    for p in range(min(q, 8193)):
        assert 2 * ((4096 * p) % q) != q


def test_component_rounding_counterexample_and_dimension_bound():
    # Requested q=17, M=8: rounding b=1,a=1 separately loses one cell.
    assert (_division_oracle(1, 17, 8) * 2) % 8 == 0
    assert _division_oracle(2, 17, 8) == 1
    assert phase.COMPONENT_ERROR_CELLS == Fraction(1025, 2)
    assert phase.COMPONENT_ERROR_CELLS == Fraction(1024 + 1, 2)


def test_all_segment_cell_edges_and_strict_error():
    for s in range(4):
        width, start = 1 << (9 + s), (1 << 20) << s
        for index in range(2048):
            for offset in (0, 1, width - 1):
                x = start + index * width + offset
                assert phase.segment_index(x, phase.GRID) == (s, index)
                z = phase.local_center(x, s)
                assert phase.nearest_even(z.numerator, z.denominator) == index


def test_strict_index_error_promises():
    for s in range(4):
        phase_bound, local_bound = Fraction(1, 8192), Fraction(1, 2 * (1 << (9 + s)))
        for sign in (-1, 1):
            phase.require_index_error(normalized_phase_error=sign * phase_bound * Fraction(999, 1000),
                                      local_coordinate_error=sign * local_bound * Fraction(999, 1000),
                                      segment=s)
            for factor in (Fraction(1), Fraction(1001, 1000)):
                with pytest.raises(phase.PhaseIndexRefused, match="normalized"):
                    phase.require_index_error(normalized_phase_error=sign * phase_bound * factor,
                                              local_coordinate_error=Fraction(0), segment=s)
                with pytest.raises(phase.PhaseIndexRefused, match="per-segment"):
                    phase.require_index_error(normalized_phase_error=Fraction(0),
                                              local_coordinate_error=sign * local_bound * factor, segment=s)
        if s < 3:
            # Legal local-coordinate error can exceed normalized phase bound.
            phase.require_index_error(normalized_phase_error=Fraction(0),
                                      local_coordinate_error=(phase_bound + local_bound) / 2, segment=s)


@pytest.mark.parametrize("normalized,local,segment", [(0, Fraction(0), 0),
                                                     (Fraction(0), 0.0, 0),
                                                     (Fraction(0), Fraction(0), True),
                                                     (Fraction(0), Fraction(0), 4)])
def test_error_units_require_named_exact_rationals(normalized, local, segment):
    with pytest.raises(ValueError):
        phase.require_index_error(normalized_phase_error=normalized,
                                  local_coordinate_error=local, segment=segment)


def test_negacyclic_sign():
    assert phase.negacyclic_rotation(0) == (0, 1)
    assert phase.negacyclic_rotation(1) == (2047, -1)
    assert phase.negacyclic_rotation(2048) == (0, -1)
    assert phase.negacyclic_rotation(2049) == (2047, 1)
    assert phase.negacyclic_rotation(4095) == (1, 1)


def test_homcomp_strict_quarter_cell():
    q = 4096 * 4096 + 1
    assert phase.homcomp_error_cells(q, 0) < Fraction(1, 4)
    phase.require_quarter_cell(q, 0)
    with pytest.raises(phase.PhaseIndexRefused, match="quarter-cell"):
        phase.require_quarter_cell(q, q)


@pytest.mark.parametrize("q,n", [(2, 1), (0, 1), (17, 0), (17, 1025),
                                 (1 << 10000, 1), (True, 1)])
def test_malformed_specs_and_huge_inputs(q, n):
    with pytest.raises(ValueError):
        phase.PhaseIndexSpec(q, n)


def test_noncanonical_and_shape_refuse():
    spec = phase.PhaseIndexSpec(17, 1)
    for b, a in [(17, (1,)), (0, (-1,)), (0, (17,)), (0, ()), (0, [1])]:
        with pytest.raises(ValueError):
            phase.compile_phase_index(spec, b, a)
    for kwargs in [dict(phase_sign="minus"), dict(level=1), dict(scale=1), dict(key_epoch=-1)]:
        with pytest.raises(ValueError):
            phase.PhaseIndexSpec(17, 1, **kwargs)
    for numerator, denominator in [(16, 1), (0, 1), (1, 3), (1, 0), (-1, 1)]:
        with pytest.raises(ValueError):
            phase.segment_index(numerator, denominator)
    with pytest.raises(ValueError):
        phase.quantize_phase_reference(1 << 10000, 17)


@pytest.mark.parametrize("width,nodes", [(61, 864545), (63, 893419)])
def test_ir_preflight_refuses_before_builder(monkeypatch, width, nodes):
    def forbidden(*args, **kwargs):
        pytest.fail("oversized shape allocated IR builder")
    monkeypatch.setattr(phase, "_Builder", forbidden)
    # Separate legal ceilings are not joint support. Current formula differs
    # from review's 728001; pin the actual unchanged generic gate construction.
    with pytest.raises(phase.PhaseIndexRefused, match=f"static shape {nodes} nodes"):
        phase.compile_phase_index(phase.PhaseIndexSpec((1 << width) - 1, 1024), 0, (0,) * 1024)


@pytest.mark.parametrize("q", [(1 << 61) - 1, (1 << 63) - 1])
def test_wide_modulus_bounded_dimension_is_clear_compilable(q):
    circuit = phase.compile_phase_index(phase.PhaseIndexSpec(q, 1), q - 1, (q - 1,))
    for bit in (0, 1):
        assert _clear_circuit(circuit, (bit,)) == _division_oracle(q - 1 + (q - 1) * bit, q)
    assert circuit.quote.nodes <= phase.MAX_NODES
    with pytest.raises(phase.PhaseIndexRefused):
        circuit.quote.require_admission()


def test_maximum_width_boundary():
    assert phase.PhaseIndexSpec((1 << 63) - 1, 1).q.bit_length() == 63
    with pytest.raises(ValueError, match="bounded integer"):
        phase.PhaseIndexSpec((1 << 63) + 1, 1)


def _independent_ssa_accounting(nodes, outputs):
    """Separate IR walk; no builder depths, _parents, uses or quote reused."""
    counts = Counter(op for op, _, _ in nodes)
    depths = []
    last_read = [-1] * len(nodes)
    references = Counter(outputs)
    for ordinal, (op, left, right) in enumerate(nodes):
        if op in ("and", "xor"):
            parents = (left, right)
            depth = 1 + max(depths[left], depths[right])
        elif op == "not":
            parents, depth = (left,), depths[left]
        else:
            assert op in ("const", "input")
            parents, depth = (), 0
        for parent in parents:
            assert 0 <= parent < ordinal
            last_read[parent] = ordinal
            references[parent] += 1
        depths.append(depth)
    for output in outputs:
        last_read[output] = len(nodes)  # pinned by final result
    expiry = [[] for _ in nodes]
    for ordinal, last in enumerate(last_read):
        if last < len(nodes):
            expiry[max(ordinal, last)].append(ordinal)
    entries, peak = set(), 0
    for ordinal in range(len(nodes)):
        entries.add(ordinal)
        peak = max(peak, len(entries))
        entries.difference_update(expiry[ordinal])
    assert entries == set(outputs)
    return counts, max(depths), peak, tuple(references[i] for i in range(len(nodes)))


@pytest.mark.parametrize("q,n", [(17, 1), (17, 2), (17, 3), ((1 << 63) - 1, 1)])
def test_independent_gate_depth_and_dictionary_liveness_accounting(q, n):
    circuit = phase.compile_phase_index(phase.PhaseIndexSpec(q, n), 1, (3,) * n)
    counts, depth, dictionary_peak, references = _independent_ssa_accounting(circuit.nodes, circuit.outputs)
    quote = circuit.quote
    assert quote.nodes == sum(counts.values())
    assert (quote.and_gates, quote.xor_gates, quote.not_gates) == (counts["and"], counts["xor"], counts["not"])
    assert quote.products == quote.relinearizations == counts["and"] + counts["xor"]
    assert quote.multiplicative_depth == depth
    assert quote.live_bit_planes == dictionary_peak
    assert circuit.uses == references
    assert quote.working_set_bytes is None  # dictionary peak is NOT backend peak


def test_accounting_oracle_includes_dead_work_depth():
    # Unused multiplication chain, output remains constant; all work executes.
    nodes = (("const", 0, -1), ("const", 1, -1),
             ("and", 1, 1), ("xor", 2, 1), ("and", 3, 3))
    counts, depth, peak, uses = _independent_ssa_accounting(nodes, (0,))
    assert depth == 3 and peak == 4
    assert counts["and"] == 2 and counts["xor"] == 1
    assert uses == (1, 3, 1, 2, 0)


def test_quote_matches_actual_gates_and_refuses_before_backend(monkeypatch):
    circuit = phase.compile_phase_index(phase.PhaseIndexSpec(17, 1), 1, (1,))
    quote = circuit.quote
    assert quote.nodes == len(circuit.nodes)
    assert quote.products == quote.relinearizations == quote.and_gates + quote.xor_gates
    assert quote.ir_heap_estimate_bytes > 0  # heuristic only; no proven byte bound
    assert quote.multiplicative_depth > 0 and quote.live_bit_planes >= 12
    assert quote.noise_margin_bits is None and quote.working_set_bytes is None
    assert quote.context_bytes is None and quote.evaluation_key_bytes is None
    assert quote.serialization_bytes is None and quote.import_overlap_bytes is None
    assert not quote.tc128_parameters_checked and not quote.joint_key_privacy_checked
    def forbidden():
        pytest.fail("backend imported before admission")
    monkeypatch.setattr(phase, "_backend", forbidden)
    with pytest.raises(phase.PhaseIndexRefused, match="no HE allocation"):
        phase.evaluate_encrypted_phase_index(circuit, (object(),))


@pytest.mark.he
def test_encrypted_final_index_client_only():
    # Unconditionally unavailable. No latent private-executor/client-context
    # experiment may spring open when admission changes. Generic SEAL source
    # is authored, NOT qualified, and cannot validate ciphertext binary bits.
    # A separately reviewed future test needs: seal_frames public-only import;
    # retained owner and consumed request; fresh independent key; evaluator
    # without decryptor; final bits decrypted client-side only; failure/replay
    # retirement tests; TC128 parameters/noise/resources and hard-limit review.
    # Reuse admitted owners; do not copy CKKS authority onto BFV.
    pytest.skip("HE primitive unavailable: no reviewed BFV profile/public-owner import, "
                "request consumption/retirement or qualified noise/resource enforcement; "
                "encrypted correctness NOT established")
