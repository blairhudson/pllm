"""Function-specific joint-correlation screen; no runtime or security admission.

Default: bounded exact algebra plus complete-region known floors. --bfv measures
dealerless, role-separated BFV generation of joint mask functions. Unsanitized
BFV evaluation is NOT circuit private: its transcript is a numeric/cost probe,
not the secure generator specified in the accompanying evidence report.
No downloads, full-scale material, or output files. All evidence goes to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np

P = 65537
ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 64 << 20
MAX_RSS = 1 << 30


def random_residues(shape: tuple[int, ...], p: int = P) -> np.ndarray:
    return np.asarray(
        [secrets.randbelow(p) for _ in range(int(np.prod(shape)))], dtype=np.int64
    ).reshape(shape)


def projections(n: int, m: int, h: int) -> tuple[np.ndarray, ...]:
    if not 1 <= n <= 64 or not 1 <= m <= 64 or not 1 <= h <= 4:
        raise ValueError("synthetic public-map dimensions exceed bounded probe")
    rng = np.random.default_rng(20261001)
    return tuple(
        rng.integers(-3, 4, size=shape, dtype=np.int64) for shape in ((m, n), (m, n), (h, m))
    )


def functions(
    r: np.ndarray, b: np.ndarray, maps: tuple[np.ndarray, ...], p: int = P
) -> tuple[np.ndarray, ...]:
    g, u, d = maps
    rg, ru = (r @ g.T) % p, (r @ u.T) % p
    return (np.sum(r * r, axis=1) % p, (r * b[:, None]) % p, (((rg * ru) % p) @ d.T) % p)


def frame(session: str, digest: str, kind: str, party: int, payload: bytes) -> bytes:
    return msgpack.packb(
        {
            "schema": "structured.v1",
            "session": session,
            "digest": digest,
            "kind": kind,
            "party": party,
            "payload": payload,
        },
        use_bin_type=True,
    )


def unframe(
    body: bytes, session: str, digest: str, kind: str, party: int, size: int | None = None
) -> bytes:
    if type(body) is not bytes or not 0 < len(body) <= MAX_BODY:
        raise ValueError("body bound")
    row = msgpack.unpackb(body, raw=False)
    if (
        type(row) is not dict
        or set(row) != {"schema", "session", "digest", "kind", "party", "payload"}
        or row["schema"] != "structured.v1"
        or row["session"] != session
        or row["digest"] != digest
        or row["kind"] != kind
        or type(row["party"]) is not int
        or row["party"] != party
        or type(row["payload"]) is not bytes
        or (size is not None and len(row["payload"]) != size)
    ):
        raise ValueError("binding or size mismatch")
    return row["payload"]


def words(x: np.ndarray) -> bytes:
    return x.astype("<u4").tobytes()


def view(body: bytes, shape: tuple[int, ...], p: int) -> np.ndarray:
    x = np.frombuffer(body, dtype="<u4").astype(np.int64).reshape(shape)
    if np.any(x >= p):
        raise ValueError("invalid residue")
    return x


@dataclass
class JointMaterial:
    """One party's joint (r,b,<r,r>,rb,D((Gr)*(Ur))) shares; no dealer view."""

    session: str
    digest: str
    party: int
    r: np.ndarray
    b: np.ndarray
    coefficients: tuple[np.ndarray, ...]
    maps: tuple[np.ndarray, ...]
    p: int = P
    started: bool = False
    cancelled: bool = False
    pending: np.ndarray | None = None
    opened: np.ndarray | None = None
    scalar_pending: np.ndarray | None = None
    scalar_started: bool = False

    @property
    def storage_bytes(self) -> int:
        # Actual transport representation: little-endian u32, not 17-bit packing.
        return 4 * (self.r.size + self.b.size + sum(x.size for x in self.coefficients))

    def cancel(self) -> None:
        self.cancelled = True
        self.pending = self.opened = self.scalar_pending = None
        self.r.fill(0)
        self.b.fill(0)
        for x in self.coefficients:
            x.fill(0)

    def begin_source(self, share: np.ndarray) -> bytes:
        if self.started or self.cancelled:
            raise ValueError("source spent")
        self.started = True  # reserve/burn before validation
        try:
            if (
                not isinstance(share, np.ndarray)
                or share.shape != self.r.shape
                or share.dtype != np.int64
                or np.any(share < 0)
                or np.any(share >= self.p)
            ):
                raise ValueError("invalid source share")
            self.pending = (share - self.r) % self.p
            return frame(self.session, self.digest, "source", self.party, words(self.pending))
        except BaseException:
            self.cancel()
            raise

    def finish_source(self, peer: bytes) -> tuple[np.ndarray, np.ndarray]:
        if self.pending is None or self.cancelled:
            raise ValueError("source spent or cancelled")
        pending, self.pending = self.pending, None
        try:
            payload = unframe(
                peer, self.session, self.digest, "source", 1 - self.party, self.r.size * 4
            )
            x = (pending + view(payload, self.r.shape, self.p)) % self.p
            self.opened = x
            g, u, d = self.maps
            xr, ur = (x @ g.T) % self.p, (x @ u.T) % self.p
            gr, vr = (self.r @ g.T) % self.p, (self.r @ u.T) % self.p
            q = (self.coefficients[0] + 2 * np.sum(x * self.r, axis=1)) % self.p
            z = (self.coefficients[2] + (((xr * vr + gr * ur) % self.p) @ d.T)) % self.p
            if self.party == 0:
                q = (q + np.sum(x * x, axis=1)) % self.p
                z = (z + (((xr * ur) % self.p) @ d.T)) % self.p
            return q, z
        except BaseException:
            self.cancel()
            raise

    def begin_scalar(self, share: np.ndarray) -> bytes:
        if self.opened is None or self.scalar_started or self.cancelled:
            raise ValueError("scalar unavailable or spent")
        self.scalar_started = True
        try:
            if (
                not isinstance(share, np.ndarray)
                or share.shape != self.b.shape
                or share.dtype != np.int64
                or np.any(share < 0)
                or np.any(share >= self.p)
            ):
                raise ValueError("invalid scalar share")
            self.scalar_pending = (share - self.b) % self.p
            return frame(
                self.session, self.digest, "scalar", self.party, words(self.scalar_pending)
            )
        except BaseException:
            self.cancel()
            raise

    def finish_scalar(self, peer: bytes) -> np.ndarray:
        if self.scalar_pending is None or self.opened is None or self.cancelled:
            raise ValueError("scalar spent or cancelled")
        try:
            payload = unframe(
                peer, self.session, self.digest, "scalar", 1 - self.party, self.b.size * 4
            )
            e = (self.scalar_pending + view(payload, self.b.shape, self.p)) % self.p
            d = self.opened
            result = (self.coefficients[1] + d * self.b[:, None] + self.r * e[:, None]) % self.p
            if self.party == 0:
                result = (result + d * e[:, None]) % self.p
            self.cancel()
            return result
        except BaseException:
            self.cancel()
            raise


def material_pair(
    r0: np.ndarray,
    r1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    c0: tuple[np.ndarray, ...],
    c1: tuple[np.ndarray, ...],
    maps: tuple[np.ndarray, ...],
    session: str,
    p: int = P,
) -> tuple[JointMaterial, ...]:
    digest = hashlib.sha256(
        b"".join(x.tobytes() for x in maps) + str((r0.shape, p)).encode()
    ).hexdigest()
    return tuple(
        JointMaterial(session, digest, i, r.copy(), b.copy(), tuple(x.copy() for x in cs), maps, p)
        for i, r, b, cs in ((0, r0, b0, c0), (1, r1, b1, c1))
    )


def consume(pair: tuple[JointMaterial, ...]) -> dict:
    rows, n = pair[0].r.shape
    p, maps = pair[0].p, pair[0].maps
    # Oracle harness only. Workers receive one input/material share each.
    x = np.arange(rows * n, dtype=np.int64).reshape(rows, n) % p
    s = (np.arange(rows, dtype=np.int64) + 17) % p
    x0, s0 = random_residues(x.shape, p), random_residues(s.shape, p)
    xo = [pair[0].begin_source(x0), pair[1].begin_source((x - x0) % p)]
    results = [pair[0].finish_source(xo[1]), pair[1].finish_source(xo[0])]
    expected = functions(x, s, maps, p)
    np.testing.assert_array_equal((results[0][0] + results[1][0]) % p, expected[0])
    np.testing.assert_array_equal((results[0][1] + results[1][1]) % p, expected[2])
    so = [pair[0].begin_scalar(s0), pair[1].begin_scalar((s - s0) % p)]
    products = [pair[0].finish_scalar(so[1]), pair[1].finish_scalar(so[0])]
    np.testing.assert_array_equal((products[0] + products[1]) % p, expected[1])
    return {
        "exact_joint_consumer_parity": True,
        "source_openings_both_directions_bytes": sum(map(len, xo)),
        "scalar_openings_both_directions_bytes": sum(map(len, so)),
        "arithmetic_opening_bytes": 2 * rows * (n + 1) * 4,
        "inverse_square_root_implemented": False,
        "private_scalar_is_synthetic": True,
    }


def dealer_reference(rows: int = 4, n: int = 16, m: int = 24, h: int = 2) -> dict:
    maps = projections(n, m, h)
    r0, r1 = random_residues((rows, n)), random_residues((rows, n))
    b0, b1 = random_residues((rows,)), random_residues((rows,))
    full = functions((r0 + r1) % P, (b0 + b1) % P, maps)
    c0 = tuple(random_residues(x.shape) for x in full)
    c1 = tuple((x - a) % P for x, a in zip(full, c0, strict=True))
    pair = material_pair(r0, r1, b0, b1, c0, c1, maps, "test-dealer")
    storage = sum(x.storage_bytes for x in pair)
    return {
        "scope": "test-only omniscient algebra oracle",
        "rows": rows,
        "n": n,
        "m": m,
        "h": h,
        "field": P,
        "material_u32_bytes_both_parties": storage,
        **consume(pair),
    }


def rank_mod(a: np.ndarray, p: int = P) -> int:
    a = (a.copy().astype(np.int64)) % p
    rank = 0
    for col in range(a.shape[1]):
        pivots = np.flatnonzero(a[rank:, col])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        a[[rank, pivot]] = a[[pivot, rank]]
        a[rank] = a[rank] * pow(int(a[rank, col]), -1, p) % p
        for row in range(a.shape[0]):
            if row != rank:
                a[row] = (a[row] - a[row, col] * a[rank]) % p
        rank += 1
        if rank == a.shape[0]:
            break
    return rank


def region_screen() -> dict:
    evidence = ROOT / "docs/evidence/compiler-region-contract-qwen25-2026-09-30.json"
    raw = evidence.read_bytes()
    locked = json.loads(raw)
    cohorts = []
    n, m, h, layers, heads = 896, 4864, 896, 24, 14
    for source in locked["cohorts"]:
        rows, output = source["executed_rows"], source["output_tokens"]
        budget = source["matched_prepared"]
        norm_rows = 2 * layers * rows
        boundary = source["two_worker_resident"]["token_boundary_body_bytes"]
        scenarios = []
        for bits in (32, 24, 12):
            # Arithmetic-only optimistic packing, not serialized envelope sizes.
            opening = (2 * norm_rows * (n + 1) * bits + 7) // 8
            storage = (2 * norm_rows * (2 * n + 2) * bits + 7) // 8
            online = opening + boundary
            scenarios.append(
                {
                    "bits": bits,
                    "joint_norm_online_floor_bytes": opening,
                    "explicit_joint_norm_material_storage_bytes": storage,
                    "explicit_dealer_issuance_if_transmitted_bytes": storage,
                    "norm_plus_boundary_online_floor_bytes": online,
                    "dealer_layout_known_all_link_floor_bytes": online + storage,
                    "exceeds_online_budget": online > budget["tenfold_online_budget_bytes"],
                    "exceeds_all_link_budget_before_material": online
                    > budget["tenfold_all_link_budget_bytes"],
                    "numeric_admitted": False,
                }
            )
        causal_edges = 39 * 40 // 2 + sum(range(40, 39 + output))
        cohorts.append(
            {
                "output_tokens": output,
                "executed_rows": rows,
                "norm_rows": norm_rows,
                "budgets": budget,
                "norm_scenarios": scenarios,
                "projected_quadratic_mlp_u32_material_bytes": 2 * (n + h) * 4 * layers * rows,
                "projected_cubic_mlp_u32_material_bytes": 2 * (n + 2 * m + h) * 4 * layers * rows,
                "causal_attention_scores": layers * heads * causal_edges,
                "explicit_attention_score_constant_u32_share_bytes": 2
                * layers
                * heads
                * causal_edges
                * 4,
                "secure_unpacked_paillier_3072_norm_generator_ciphertext_floor_bytes": 2
                * (n + 1)
                * 768
                * norm_rows,
                "complete_region_admitted": False,
            }
        )
    g, u, _ = projections(16, 24, 2)
    return {
        "locked_region_evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "source": locked["source"],
        "cohorts": cohorts,
        "dimensions": {"hidden": n, "intermediate": m, "layers": layers, "query_heads": heads},
        "synthetic_stacked_public_map_field_rank": rank_mod(np.vstack((g, u))),
        "unknowns": [
            "private RMSNorm inverse and exact rounding",
            "W8A8 dynamic quantization and small-ring signed carry",
            "protected SiLU with checkpoint fidelity",
            "causal softmax, probability-value product and KV share state",
            "secure generator transcript, authentication and full wire",
        ],
        "verdict": "stop tested norm-opening layouts; pursue only a new opening-eliminating complete-region protocol",
    }


def bfv_probe(batches: tuple[int, ...] = (1, 4, 16)) -> dict:
    import tenseal as ts

    if not batches or any(type(x) is not int or not 1 <= x <= 16 for x in batches):
        raise ValueError("batch bound")
    n, m, h = 16, 24, 2
    maps = projections(n, m, h)
    g, u, d = maps
    cpu = time.process_time()
    secret = ts.context(ts.SCHEME_TYPE.BFV, 8192, P, n_threads=1)
    secret.generate_relin_keys()
    secret.generate_galois_keys()
    context_body = secret.serialize(
        save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True
    )
    if len(context_body) > MAX_BODY:
        raise RuntimeError("context byte bound")
    evaluator = ts.context_from(context_body, n_threads=1)
    if evaluator.has_secret_key():
        raise RuntimeError("evaluator received secret key")
    secret_storage = len(
        secret.serialize(
            save_public_key=True, save_secret_key=True, save_galois_keys=True, save_relin_keys=True
        )
    )
    setup_cpu = time.process_time() - cpu
    digest = hashlib.sha256(context_body).hexdigest()
    consumed: set[str] = set()

    def evaluate(request: bytes, r1: np.ndarray, b1: np.ndarray, session: str):
        # Only public evaluator context plus party B's local data enters here.
        if session in consumed:
            raise ValueError("generation replay")
        consumed.add(session)  # burn before request decoding
        payload = unframe(request, session, digest, "generation-request", 0)
        packed = msgpack.unpackb(payload, raw=False)
        if type(packed) is not list or len(packed) != 4:
            raise ValueError("generation request shape")
        encrypted = [ts.bfv_vector_from(evaluator, x) for x in packed]
        rows = r1.shape[0]
        r = encrypted[0] + r1.reshape(-1).tolist()
        b = encrypted[1] + np.repeat(b1, n).tolist()
        gr = encrypted[2] + ((r1 @ g.T) % P).reshape(-1).tolist()
        ur = encrypted[3] + ((r1 @ u.T) % P).reshape(-1).tolist()
        masks = (random_residues((rows,)), random_residues((rows, n)), random_residues((rows, h)))
        square, product, gate = r * r, r * b, gr * ur
        outputs = [product - masks[1].reshape(-1).tolist()]
        for row in range(rows):
            selector = np.zeros(rows * n, dtype=np.int64)
            selector[row * n : (row + 1) * n] = 1
            outputs.append(square.dot(selector.tolist()) - [int(masks[0][row])])
            for col in range(h):
                weights = np.zeros(rows * m, dtype=np.int64)
                weights[row * m : (row + 1) * m] = d[col]
                outputs.append(gate.dot(weights.tolist()) - [int(masks[2][row, col])])
        # Returning each reduced ciphertext preserves measured parity.
        # Repacking evaluated scalars failed parity in this parameter set.
        bodies = [x.serialize() for x in outputs]
        response = frame(
            session, digest, "generation-response", 1, msgpack.packb(bodies, use_bin_type=True)
        )
        return response, masks, [len(x) for x in bodies]

    samples = []
    for rows in batches:
        session = secrets.token_hex(16)
        r0, r1 = random_residues((rows, n)), random_residues((rows, n))
        b0, b1 = random_residues((rows,)), random_residues((rows,))
        cpu = time.process_time()
        local = (
            r0.reshape(-1),
            np.repeat(b0, n),
            ((r0 @ g.T) % P).reshape(-1),
            ((r0 @ u.T) % P).reshape(-1),
        )
        ciphers = [ts.bfv_vector(secret, x.tolist()).serialize() for x in local]
        request = frame(
            session, digest, "generation-request", 0, msgpack.packb(ciphers, use_bin_type=True)
        )
        response, c1, response_lengths = evaluate(request, r1, b1, session)
        payload = unframe(response, session, digest, "generation-response", 1)
        results = [
            np.asarray(ts.bfv_vector_from(secret, x).decrypt(), dtype=np.int64) % P
            for x in msgpack.unpackb(payload, raw=False)
        ]
        q0, k0 = np.empty(rows, dtype=np.int64), np.empty((rows, h), dtype=np.int64)
        for row in range(rows):
            q0[row] = results[1 + row * (h + 1)][0]
            for col in range(h):
                k0[row, col] = results[2 + row * (h + 1) + col][0]
        c0 = (q0, results[0].reshape(rows, n), k0)
        expected = functions((r0 + r1) % P, (b0 + b1) % P, maps)
        for a, b, oracle in zip(c0, c1, expected, strict=True):
            np.testing.assert_array_equal((a + b) % P, oracle)
        try:
            evaluate(request, r1, b1, session)
        except ValueError as exc:
            replay = str(exc) == "generation replay"
        else:
            replay = False
        if not replay:
            raise RuntimeError("generation replay accepted")
        generation_cpu = time.process_time() - cpu
        pair = material_pair(r0, r1, b0, b1, c0, c1, maps, session)
        storage = sum(x.storage_bytes for x in pair)
        consumer = consume(pair)
        online_bodies = (
            consumer["source_openings_both_directions_bytes"]
            + consumer["scalar_openings_both_directions_bytes"]
        )
        samples.append(
            {
                "rows": rows,
                "n": n,
                "m": m,
                "h": h,
                "request_ciphertext_bytes": list(map(len, ciphers)),
                "response_ciphertext_bytes": response_lengths,
                "request_envelope_bytes": len(request),
                "response_envelope_bytes": len(response),
                "generator_body_bytes_warm_context": len(request) + len(response),
                "generator_body_bytes_cold_context": len(context_body)
                + len(request)
                + len(response),
                "generator_plus_consumer_bodies_warm_bytes": len(request)
                + len(response)
                + online_bodies,
                "generator_plus_consumer_bodies_cold_bytes": len(context_body)
                + len(request)
                + len(response)
                + online_bodies,
                "warm_generator_bytes_per_row": (len(request) + len(response)) / rows,
                "B_local_output_masks_u32_bytes_not_transmitted": sum(x.size * 4 for x in c1),
                "each_party_local_r_b_u32_bytes": (n + 1) * rows * 4,
                "one_use_material_u32_bytes_both_parties": storage,
                "exact_generator_parity": True,
                "generation_replay_rejected": replay,
                "generation_cpu_seconds": generation_cpu,
                **consumer,
            }
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    if rss_bytes > MAX_RSS:
        raise RuntimeError("observed peak RSS exceeded 1 GiB")
    return {
        "backend": f"TenSEAL {ts.__version__}",
        "poly_modulus_degree": 8192,
        "plain_modulus": P,
        "n_threads_per_context": 1,
        "public_context_with_rotation_relinearization_keys_bytes": len(context_body),
        "A_private_context_serialization_bytes_not_transmitted": secret_storage,
        "evaluator_has_secret_key": evaluator.has_secret_key(),
        "context_generation_and_import_cpu_seconds": setup_cpu,
        "process_observed_peak_rss_bytes": rss_bytes,
        "scope": "dealerless numeric and serialized-body cost probe; BFV circuit privacy NOT implemented",
        "cryptographically_secure_generator_admitted": False,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfv", action="store_true")
    args = parser.parse_args()
    result = {
        "schema": "structured_correlations_screen.v1",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "algebra": dealer_reference(),
        "complete_region_screen": region_screen(),
    }
    if args.bfv:
        result["bfv"] = bfv_probe()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
