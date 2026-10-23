"""One bounded, source-only Gamma inspection; no importer, model or SDK authority.

Reuse the scale source's exact pinned HTTPS/range reader. Retained header identity
and an immutable trusted publisher bind slices, not whole-shard LFS authentication.
Only scalar BF16 Gamma statistics are retained; overlapping scales are not decoded.
Do not repeat an inspection using stale prior debits. There is no retry/refresh.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import struct

from benchmarks.research import maverick_gamma_range_plan as planner
from benchmarks.research import maverick_scale_source as owner

MAX_BODY = 507392
MAX_REQUESTS = 83
PRIOR_BODY = 845976
PRIOR_REQUESTS = 217
PIN = "96b30dc13593a244a5e59e84687309f53c375cfa"
HEADER = "9203d0277e3986e90216c106ebdba821ea34acef7fcea49e88b5e5660def7cca"


def signed_gamma_summary(raw):
    if len(raw) % 2:
        raise owner.Reject("partial Gamma BF16 word")
    counts, maximum = Counter(), None
    for (word,) in struct.iter_unpack("<H", raw):
        magnitude = word & 0x7FFF
        if magnitude >= 0x7F80:
            category = "infinity" if magnitude == 0x7F80 else "nan"
        else:
            category = ("negative_zero" if word >> 15 else "positive_zero") if magnitude == 0 else (
                "negative_finite" if word >> 15 else "positive_finite")
            maximum = magnitude if maximum is None else max(maximum, magnitude)
        counts[category] += 1
    return counts, maximum


class GammaAudit(owner.Audit):
    def __init__(self, prior, *, prior_requests, opener=None):
        super().__init__(prior, prior_requests=prior_requests, opener=opener, cache_roots=())
        # This authorized continuation cannot reclaim already recorded debits.
        if prior < PRIOR_BODY or prior_requests < PRIOR_REQUESTS:
            raise owner.Reject("stale prior source accounting")
        self.manifest = planner.plan()
        selected = self.manifest["plans"]["proposed_known_scale_overlap"]
        if (owner.PIN != PIN or owner.HEADER_SHA256 != HEADER or owner.FILE_SIZE != 4412140424
                or self.manifest["header_sha256"] != HEADER or self.manifest["gamma_tensors"] != 145
                or self.manifest["gamma_coefficients"] != 196096
                or selected["payload_bytes"] != MAX_BODY
                or selected["previously_inspected_scale_overlap_bytes"] != 115200):
            raise owner.Reject("authorized Gamma plan prerequisites changed")
        self.groups = selected["groups"]
        expected = [(777988872, 5120)] + [(778003112 + 23072*j, 13952) for j in range(36)]
        if [(g["start"], g["length"]) for g in self.groups] != expected:
            raise owner.Reject("authorized Gamma ranges changed")
        self.norms = {row["name"]: row for row in self.manifest["norms"]}
        self.completed_groups = 0
        self.gamma_maximum = None

    def _remaining_body_budget(self):
        return min(super()._remaining_body_budget(), MAX_BODY - self.network_bytes)

    def _before_request(self):
        remaining = len(self.groups) - self.completed_groups
        if self.requests + remaining > MAX_REQUESTS:
            raise owner.Reject("redirect accounting cannot fit remaining Gamma ranges")
        if self.prior_requests + self.requests + remaining > owner.REQUEST_LIMIT:
            raise owner.Reject("cumulative requests cannot fit remaining Gamma ranges")
        super()._before_request()

    def fetch(self, start, length, kind):
        if self.completed_groups >= len(self.groups):
            raise owner.Reject("Gamma range sequence exhausted")
        expected = self.groups[self.completed_groups]
        if kind != "gamma" or (start, length) != (expected["start"], expected["length"]):
            raise owner.Reject("non-plan or out-of-order Gamma read refused")
        return self._read_range(start, length, "gamma", self.final_url or owner.CANONICAL)

    def _inspect(self):
        if (MAX_BODY > self._remaining_body_budget()
                or self.prior_requests + len(self.groups) > owner.REQUEST_LIMIT):
            raise owner.Reject("aggregate Gamma plan cannot fit before first request")
        for group in self.groups:
            raw = self.fetch(group["start"], group["length"], "gamma")
            for name in group["gamma_names"]:
                row = self.norms[name]
                offset = row["start"] - group["start"]
                counts, maximum = signed_gamma_summary(raw[offset:offset+row["length"]])
                self.counts.update(counts)
                if maximum is not None:
                    self.gamma_maximum = maximum if self.gamma_maximum is None else max(self.gamma_maximum, maximum)
                self.results.append({**row, "elements": sum(counts.values()), "counts": dict(counts),
                                     "maximum_absolute": owner.exact_positive(maximum)})
            self.completed_groups += 1

    def output(self, error=None):
        names = {row["name"] for row in self.results}
        unread = sorted(set(self.norms) - names)
        complete = (len(self.results) == 145 and not unread and sum(self.counts.values()) == 196096
                    and self.completed_groups == 37 and self.network_bytes == MAX_BODY)
        finite = self.counts["nan"] + self.counts["infinity"] == 0
        g = next((g for g in range(23) if self.gamma_maximum is not None
                  and self.gamma_maximum <= (127+g)*128), None)
        passed = complete and finite and g is not None and error is None
        return {"repository": "Qwen/Qwen3-4B-FP8", "revision": PIN, "asset": owner.ASSET,
                "retained_header_sha256": HEADER,
                "result": "observed_gamma_envelope" if passed else "unresolved_or_failed",
                "safe_error": error, "complete_observation": complete, "counts": dict(self.counts),
                "inspected_tensors": len(self.results), "inspected_coefficients": sum(self.counts.values()),
                "maximum_absolute": owner.exact_positive(self.gamma_maximum),
                "sufficient_gamma_exponent_g": g if passed else None, "unread_vectors": unread,
                "prior_body_bytes_debited": self.prior, "new_body_bytes": self.network_bytes,
                "aggregate_body_bytes": self.body, "additional_body_cap": MAX_BODY,
                "prior_requests_reserved": self.prior_requests, "new_http_attempts_including_redirects": self.requests,
                "aggregate_request_reservation": self.prior_requests+self.requests,
                "additional_request_cap": MAX_REQUESTS, "completed_groups": self.completed_groups,
                "completed_group_scale_overlap_bytes": sum(self.groups[j]["length"] for j in range(self.completed_groups))
                    - sum(row["length"] for row in self.results),
                "ranges": self.events, "vectors": self.results,
                "source_assumption": "trusted pinned HTTPS publisher is immutable; retained metadata unchanged",
                "whole_shard_integrity": "unverified", "embedding_bound": "unproved",
                "head_embedding_equality": "unverified", "source_no_wrap": "unproved",
                "whole_graph_totality": "unproved", "implementation_or_SDK_authority": "none"}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inspect", action="store_true", required=True)
    cli.add_argument("--prior-body-bytes", type=int, required=True)
    cli.add_argument("--prior-requests", type=int, required=True)
    args = cli.parse_args(argv)
    audit, error = None, None
    try:
        audit = GammaAudit(args.prior_body_bytes, prior_requests=args.prior_requests)
        audit.run()
    except Exception as exc:
        error = str(exc) if isinstance(exc, owner.Reject) else type(exc).__name__ + ": details suppressed"
    result = audit.output(error) if audit is not None else {
        "result": "unresolved_or_failed", "safe_error": error, "new_body_bytes": 0,
        "new_http_attempts_including_redirects": 0, "unread_vectors": sorted(planner.NORMS)}
    print(json.dumps(result, indent=2))
    return 0 if result["result"] == "observed_gamma_envelope" else 2


if __name__ == "__main__":
    raise SystemExit(main())
