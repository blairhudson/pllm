"""Exact small Galois-ring external transform control and analytical ledger.

Not a PCG, FSS, encryption implementation, security audit, or RSS measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20
WORD = struct.Struct("<QQ")
LOCKS = {
    "papers/2025-1223.pdf": "130178519ec48cbbe64b201e42bb581b513f77503f671e1a56e4b0909b4aa375",
    "papers/2025-1660.pdf": "4f05f4b6b8bfea36a65913ac6ed532d734e231bc803311a7933bf76083c7bee8",
    "target/compact-setup-fulltext/2025-1223.txt": "4a91404c17a157c0449f625c2c3532bcb722447ad23d03d9e58b53d2262b5821",
    "target/compact-setup-fulltext/2025-1660.txt": "fab182e60cc9ed03697a9a6381e831e419653fff49c136f4fdcf646eae763f3d",
}


def integer(value: int, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside supported bounds")


class GR:
    """GR(2^bits,2) = Z/(2^bits)[z]/(z^2+z+1), z^3=1.

    Polynomial is the Hensel lift of the irreducible primitive degree-two
    polynomial over F2. Scalars are pairs of canonical base-ring coefficients.
    """

    def __init__(self, bits: int):
        integer(bits, 1, 64)
        self.bits, self.mod = bits, 1 << bits
        self.zero, self.one, self.z = (0, 0), (1, 0), (0, 1)

    def validate(self, x: tuple[int, int]) -> None:
        if not isinstance(x, tuple) or len(x) != 2:
            raise ValueError("expected two coefficients")
        for limb in x:
            integer(limb, 0, self.mod - 1)

    def add(self, x, y):
        return ((x[0] + y[0]) % self.mod, (x[1] + y[1]) % self.mod)

    def mul(self, x, y):
        a, b = x
        c, d = y
        return ((a*c - b*d) % self.mod, (a*d + b*c - b*d) % self.mod)

    def sigma(self, x):
        # Frobenius sends z to z^2=-1-z, NOT x -> x^2 in characteristic 2^k.
        return ((x[0] - x[1]) % self.mod, -x[1] % self.mod)

    def trace(self, x):
        return (2*x[0] - x[1]) % self.mod

    def roots(self):
        return (self.one, self.z, self.mul(self.z, self.z))

    def fiber(self, values, inverse=False):
        roots = self.roots()
        result = []
        for out in range(3):
            products = [self.mul(values[j], roots[((-1 if inverse else 1)*out*j) % 3])
                        for j in range(3)]
            total = self.add(self.add(products[0], products[1]), products[2])
            if inverse:
                total = self.mul(total, (pow(3, -1, self.mod), 0))
            result.append(total)
        return result


def geometry(n: int, toy=False) -> int:
    integer(n, 1, 5 if toy else 18)
    return 3**n


def fibers(n: int, axis: int):
    size, stride = geometry(n), 3**axis
    for block in range(0, size, 3*stride):
        for offset in range(stride):
            yield tuple(block + offset + j*stride for j in range(3))


def full_transform(values, ring: GR, n: int, inverse=False):
    size = geometry(n, toy=True)
    if len(values) != size:
        raise ValueError("wrong transform length")
    for x in values:
        ring.validate(x)
    current = list(values)
    for axis in range(n):
        nxt = [ring.zero] * size
        for indices in fibers(n, axis):
            for index, x in zip(indices, ring.fiber([current[i] for i in indices], inverse), strict=True):
                nxt[index] = x
        current = nxt
    return current


def external_transform(source: Path, output: Path, scratch: Path, ring: GR, n: int):
    """Plaintext toy files only. No secret spill or authenticated paging claim.

    Two alternating files; three input and three output elements per butterfly.
    Random local accesses replace full resident arrays. Output must not exist.
    """
    size = geometry(n, toy=True)
    if source.stat().st_size != size * WORD.size:
        raise ValueError("wrong packed source length")
    if output.exists() or source.resolve() == output.resolve():
        raise ValueError("output already exists or aliases source")
    scratch.mkdir(parents=True, exist_ok=True)
    reads = writes = 0
    with tempfile.TemporaryDirectory(prefix="low-correlation-streaming-", dir=scratch) as folder:
        current = source
        for axis in range(n):
            nxt = Path(folder) / f"stage-{axis % 2}.bin"
            with current.open("rb", buffering=0) as src, nxt.open("wb", buffering=0) as dst:
                dst.truncate(size * WORD.size)
                for indices in fibers(n, axis):
                    values = []
                    for index in indices:
                        src.seek(index * WORD.size)
                        raw = src.read(WORD.size)
                        if len(raw) != WORD.size:
                            raise ValueError("truncated packed element")
                        value = WORD.unpack(raw)
                        ring.validate(value)
                        values.append(value)
                        reads += 1
                    for index, value in zip(indices, ring.fiber(values), strict=True):
                        dst.seek(index * WORD.size)
                        dst.write(WORD.pack(*value))
                        writes += 1
            current = nxt
        # No overwrite and no whole-file read. Final copy cost explicitly counted.
        with current.open("rb", buffering=0) as src, output.open("xb", buffering=0) as dst:
            for _ in range(size):
                raw = src.read(WORD.size)
                if len(raw) != WORD.size:
                    raise ValueError("truncated final stage")
                dst.write(raw)
    return {"stage_element_reads": reads, "stage_element_writes": writes,
            "stage_io_bytes": (reads + writes)*WORD.size,
            "final_copy_io_bytes": 2*size*WORD.size,
            "logical_butterfly_live_elements": 6,
            "scope": "logical records, not Python allocation or process RSS"}


def digits(index: int, n: int):
    return tuple((index // 3**axis) % 3 for axis in range(n))


def direct_transform(values, ring: GR, n: int):
    """Independent quadratic evaluation oracle, not radix-stage implementation."""
    roots = ring.roots()
    result = []
    for out in range(3**n):
        total = ring.zero
        for index, value in enumerate(values):
            exponent = sum(a*b for a, b in zip(digits(out, n), digits(index, n), strict=True))
            total = ring.add(total, ring.mul(value, roots[exponent % 3]))
        result.append(total)
    return result


def convolution(left, right, ring: GR, n: int):
    result = [ring.zero] * 3**n
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            index = sum(((a+b) % 3)*3**axis for axis, (a, b) in
                        enumerate(zip(digits(i, n), digits(j, n), strict=True)))
            result[index] = ring.add(result[index], ring.mul(x, y))
    return result


def ole_reconstruction_control(ring: GR, n: int):
    """Construction 2 Eq.(3), ideal additive cross-term shares, no FSS keys."""
    size = geometry(n, toy=True)
    def fixture(salt):
        return [((salt*13+i*7) % ring.mod, (salt*19+i*11) % ring.mod)
                for i in range(size)]
    a = fixture(1)
    s0, e0, s1, e1 = (fixture(salt) for salt in (2, 3, 4, 5))
    ahat = full_transform(a, ring, n)
    b0 = full_transform([ring.add(x, y) for x, y in
                         zip(convolution(a, s0, ring, n), e0, strict=True)], ring, n)
    b1 = full_transform([ring.add(x, y) for x, y in
                         zip(convolution(a, s1, ring, n), e1, strict=True)], ring, n)
    accumulators = [[[0]*size for _ in range(2)] for _ in range(2)]
    bases = (ring.z, ring.mul(ring.z, ring.z))
    for frobenius in (0, 1):
        def sigma_poly(values):
            if not frobenius:
                return values
            result = [ring.zero]*size
            for index, x in enumerate(values):
                target = sum((2*d % 3)*3**axis for axis, d in enumerate(digits(index, n)))
                result[target] = ring.sigma(x)
            return result
        cross = [convolution(s0, sigma_poly(s1), ring, n),
                 convolution(e0, sigma_poly(s1), ring, n),
                 convolution(s0, sigma_poly(e1), ring, n),
                 convolution(e0, sigma_poly(e1), ring, n)]
        for term, values in enumerate(cross):
            share0 = fixture(6+4*frobenius+term)
            share1 = [ring.add(x, ((-y[0]) % ring.mod, (-y[1]) % ring.mod))
                      for x, y in zip(values, share0, strict=True)]
            for party, share in enumerate((share0, share1)):
                transformed = full_transform(share, ring, n)
                for index, u in enumerate(transformed):
                    ap = ring.sigma(ahat[index]) if frobenius else ahat[index]
                    multiplier = (ring.mul(ahat[index], ap), ap, ahat[index], ring.one)[term]
                    value = ring.mul(multiplier, u)
                    for lane, basis in enumerate(bases):
                        coefficient = ring.mul(basis, ring.sigma(basis) if frobenius else basis)
                        accumulators[party][lane][index] = (
                            accumulators[party][lane][index] + ring.trace(ring.mul(coefficient, value))
                        ) % ring.mod
    for lane, basis in enumerate(bases):
        for index in range(size):
            left = ring.trace(ring.mul(basis, b0[index]))
            right = ring.trace(ring.mul(basis, b1[index]))
            assert left*right % ring.mod == (
                accumulators[0][lane][index]+accumulators[1][lane][index]) % ring.mod
    return 2*size


def resource_model(n=15, window_mib=8, demand=1):
    """Unoptimized c=5 generalization envelope; not published implementation cost.

    Keep public Table 3 point (128,5,27,m=2,k=64,N=3^15). Generalizing the
    c=2 cross-term schedule gives m*c^2 separate t^2 FSS arrays. Cost one
    forward transform per array, then local trace accumulation on disk.
    """
    size = geometry(n)
    # Table 3 keeps c=5,t=27 at n=15,16 only. Never call n=13/14/17/18 secure.
    integer(n, 15, 16)
    integer(window_mib, 1, 256)
    integer(demand, 1, 2*size)
    c, t, m, k, lam = 5, 27, 2, 64, 128
    vector = size * m * (k//8)
    transforms = m*c*c
    # Integer conservative envelope of approximate p34 seed formula:
    # ceil(log2 N)-floor(log2 t)+1 >= log2 N-log2 t+1.
    logn_upper, logt_lower = (size - 1).bit_length(), t.bit_length()-1
    seed_formula_bits = (m*c*c*t*t*(logn_upper-logt_lower+1)*(lam+2)
                         + lam + k*m + c*t*(logn_upper+k*m))
    seed_bytes = (seed_formula_bits+7)//8
    output_per_index = 2*(k//8)  # one scalar x,z pair, basis-major logical order
    page_payload, page_overhead = 4096, 12+16  # hypothetical stored nonce + AEAD tag
    pages_per_vector = (vector+page_payload-1)//page_payload
    return {
        "parameters": {"lambda": lam, "c": c, "t": t, "m": m, "k": k,
                       "d": 3, "n": n, "N": size, "capacity_scalar_oles": m*size},
        "source_pages": {"2025/1223": [19, 20, 22, 28, 29, 34, 35, 36]},
        "security_label": "paper Table 3 128-bit regular-noise parameter point; not audited here",
        "gr_vector_bytes": vector,
        "public_a_vectors": c-1,
        "public_precomputed_transform_disk_bytes": (c-1)*vector,
        "secret_transform_pingpong_disk_bytes": 2*vector,
        "secret_trace_accumulator_disk_bytes": vector,
        "secret_disk_payload_peak_bytes": 3*vector,
        "full_x_z_output_bytes": m*size*output_per_index,
        "cross_term_forward_transforms": transforms,
        "cross_term_transform_gr_multiplications": transforms*3*n*size,
        "cross_term_transform_gr_additions": transforms*2*n*size,
        "cross_term_transform_stage_io_bytes": transforms*2*n*vector,
        "cross_term_ingest_write_bytes": transforms*vector,
        "cross_term_transformed_scan_read_bytes": transforms*vector,
        "cross_term_accumulator_read_write_bytes": transforms*2*vector,
        "cross_term_public_multiplier_reads_bytes": transforms*2*vector,
        "public_precompute_transform_stage_io_bytes": (c-1)*2*n*vector,
        "approximate_seed_formula_integer_envelope_bytes": seed_bytes,
        "table3_rounded_seed_log2_bits": "26.60" if n == 15 else "26.70",
        "table3_seed_projection_warning": "rounded log2 bits, not serialized key bound",
        "renewal_two_seed_formula_envelopes_bytes": 2*seed_bytes,
        "scratch_hypothesis_bytes": window_mib*MIB,
        "window_scalar_pairs": window_mib*MIB//output_per_index,
        "requested_scalar_oles": demand,
        "global_expansion_unrequested_scalar_oles": m*size-demand,
        "whole_low_budget_bytes": 256*MIB,
        "remaining_for_sdk_state_keys_buffers_outputs_overlap_bytes":
            256*MIB-window_mib*MIB-2*seed_bytes,
        "public_tables_resident_bytes": None,
        "secret_fss_frontier_and_serialized_key_bytes": None,
        "encrypted_spill_aead_metadata_and_io_bytes": None,
        "hypothetical_spill_page_payload_bytes": page_payload,
        "hypothetical_spill_nonce_tag_bytes_per_page": page_overhead,
        "hypothetical_secret_padded_authenticated_pages_bytes":
            3*pages_per_vector*(page_payload+page_overhead),
        "hypothetical_uncached_cross_term_transform_page_io_bytes":
            transforms*2*n*size*(page_payload+page_overhead),
        "hypothetical_spill_scope": "page accounting only; no AEAD implementation or measured I/O",
        "sparse_x_generation_and_local_trace_work": None,
        "native_process_peak_rss_bytes": None,
        "complete_pcg_operation_count": None,
        "final_z_and_output_consumption_io_bytes": None,
        "final_consumption_scope": "mapping-dependent: charge accumulator reads, public/sparse x inputs and output handling separately",
        "ledger_owner": "one party/client expansion; not combined two-party expansion",
        "fixed_W_specialization": "missing; this ledger is OLE expansion only",
        "admission": "unqualified: owner costs and authenticated spill missing",
        "count_boundary": "exact arithmetic for stated unoptimized schedule; not a full PCG cost",
    }


def source_locks(root=ROOT):
    records = []
    for name, expected in LOCKS.items():
        data = (root/name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"source hash mismatch: {name}")
        records.append({"path": name, "sha256": digest, "bytes": len(data)})
    return records


def controls(scratch: Path):
    comparisons, local_traces, reconstructed = 0, 0, 0
    with tempfile.TemporaryDirectory(prefix="low-correlation-streaming-control-", dir=scratch) as folder:
        directory = Path(folder)
        for bits in (1, 8, 32, 64):
            ring = GR(bits)
            for n in (1, 2, 3):
                size = 3**n
                left = [((17*i+3) % ring.mod, (31*i+5) % ring.mod) for i in range(size)]
                right = [((7*i+1) % ring.mod, (11*i+2) % ring.mod) for i in range(size)]
                expected = direct_transform(left, ring, n)
                assert full_transform(left, ring, n) == expected
                assert full_transform(expected, ring, n, inverse=True) == left
                multiplied = [ring.mul(a, b) for a, b in
                              zip(expected, full_transform(right, ring, n), strict=True)]
                assert full_transform(convolution(left, right, ring, n), ring, n) == multiplied
                src, dst = directory/f"{bits}-{n}.in", directory/f"{bits}-{n}.out"
                with src.open("xb") as stream:
                    for x in left:
                        stream.write(WORD.pack(*x))
                io = external_transform(src, dst, directory, ring, n)
                with dst.open("rb") as stream:
                    disk = [WORD.unpack(stream.read(WORD.size)) for _ in range(size)]
                assert disk == expected
                assert io["stage_io_bytes"] == 2*n*size*WORD.size
                # sigma_R coefficient Frobenius plus exponent-doubling permutation.
                sigma_poly = [ring.zero]*size
                for i, x in enumerate(left):
                    index = sum((2*d % 3)*3**axis for axis, d in enumerate(digits(i, n)))
                    sigma_poly[index] = ring.sigma(x)
                assert full_transform(sigma_poly, ring, n) == [ring.sigma(x) for x in expected]
                for x in disk:
                    assert ring.add(x, ring.sigma(x)) == (ring.trace(x), 0)
                    local_traces += 1
                comparisons += 1
                reconstructed += ole_reconstruction_control(ring, n)
    # Exhaustive bijection at a tiny algebra-only domain, not lowered PCG security.
    ring = GR(1)
    elems = list(itertools.product(range(2), repeat=2))
    images = {tuple(full_transform(list(v), ring, 1)) for v in itertools.product(elems, repeat=3)}
    assert len(images) == 64
    trace_counts = []
    for bits in (1, 2, 3):
        ring = GR(bits)
        images = {(ring.trace(ring.mul(ring.z, x)),
                   ring.trace(ring.mul(ring.mul(ring.z, ring.z), x)))
                  for x in itertools.product(range(ring.mod), repeat=2)}
        assert len(images) == ring.mod**2
        trace_counts.append(len(images))
    return {"full_direct_external_roundtrip_convolution_cases": comparisons,
            "ideal_cross_share_reconstructed_scalar_oles": reconstructed,
            "local_trace_elements": local_traces,
            "exhaustive_transform_bijection_inputs": 64,
            "exhaustive_normal_basis_trace_bijections": trace_counts,
            "bits": [1, 8, 32, 64], "n": [1, 2, 3],
            "scope": "deterministic algebra and plaintext local record I/O only"}


def evidence(scratch: Path):
    locks = source_locks()
    scratch.mkdir(parents=True, exist_ok=True)
    return {
        "schema": "pllm.low_correlation_streaming.v1",
        "scope": "source-locked algebra/storage controls and analytical schedule; no secure PCG",
        "sources": locks,
        "controls": controls(scratch),
        "model": resource_model(),
        "window_sensitivity": [
            {key: resource_model(window_mib=w)[key] for key in
             ("scratch_hypothesis_bytes", "window_scalar_pairs",
              "remaining_for_sdk_state_keys_buffers_outputs_overlap_bytes",
              "cross_term_transform_stage_io_bytes")}
            for w in (8, 16, 64)],
        "dory_reference": {"source": "2025/1660", "pdf_pages": [11, 12, 14, 15, 21],
                           "lambda": 128, "N": 51478528, "t": 1571, "log2_N_over_t": 15,
                           "ell": 7, "capacity": 10295705, "threads": 8,
                           "security": "malicious", "paper_memory_MB": 20,
                           "paper_base_communication_MB": "0.419",
                           "paper_silent_communication_MB": "0.389",
                           "domain": "F2 into F_(2^128) COT; not exact-ring matrix correlation",
                           "whole_client_qualification": False},
        "cryptographic_implementation": False,
        "process_peak_measurement": None,
        "stop_condition": "No Low admission until fixed-W specialization, complete native owner ledger, authenticated local spill and rollback-safe issuance exist",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=ROOT/"target/low-correlation-streaming")
    args = parser.parse_args()
    if not args.scratch.name.startswith("low-correlation-streaming"):
        parser.error("scratch basename must start low-correlation-streaming")
    if args.output.exists():
        parser.error("output exists; evidence is immutable")
    result = evidence(args.scratch)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
