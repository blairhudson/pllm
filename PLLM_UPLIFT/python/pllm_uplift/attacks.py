"""Small public negative controls. Not attacks against an installed PLLM runtime.

Real adversary adapters must receive only the corrupted party's permitted view;
these fixtures intentionally expose the variable named in each weakened scheme.
"""
from __future__ import annotations
from collections import Counter
from math import gcd

def run_attacks() -> dict:
    findings=[]
    def add(name, successes, cases, scope):
        findings.append({"id": name, "outcome": "expected_counterexample_reproduced",
                         "successes": successes, "cases": cases, "scope": scope,
                         "target": "deliberately_weakened_public_fixture"})
    q=256
    successes=0
    for x in range(q):
        r=(97*x+13)%q
        d0=(x-r)%q; d1=((x+31)-r)%q
        successes += (d1-d0)%q == 31
    add("mask_reuse", successes, q, "Input differences are disclosed by reusing a full-ring mask.")
    p=251;ok=0
    for x in range(p):
        delta=[19,73,1];base=[61,4,151]
        a=[(b+x*d)%p for b,d in zip(base,delta)]
        b=[(z+(x+7)*d)%p for z,d in zip(base,delta)]
        diff=[(v-u)%p for u,v in zip(a,b)]
        recovered=[v*pow(diff[-1],-1,p)%p for v in diff]
        ok+=recovered==delta
    add("affine_label_reuse",ok,p,"Two different semantic values on one affine wire expose the offset with a unit point-and-permute coordinate.")
    ok=0;domain=list(range(-9,10))
    for shift in range(p):
        exposed={(shift+x)%p for x in domain}
        candidates=[b for b in range(p) if {(b+x)%p for x in domain}==exposed]
        ok+=candidates==[shift]
    add("sparse_point_permute",ok,p,"Visible sparse row indices expose the shift of a proper cyclic interval.")
    bad=0
    for a in range(256):
        for b in range(256):
            bad+=((a>>4)+(b>>4))%16 != ((a+b)%256)>>4
    findings.append({"id":"missing_truncation_carry","outcome":"expected_counterexample_reproduced",
                     "mismatches":bad,"cases":65536,"scope":"Independent local shifts omit the carry.","target":"deliberately_weakened_public_fixture"})
    ok=0
    for x in range(256):
        public_depth=1 if x>=128 else 8
        predicted_negative=public_depth==1
        ok+=predicted_negative==(x>=128)
    add("early_exit_metadata",ok,256,"Unpadded sign-dependent path length in the explicit fixture reveals sign.")
    ok=0
    for x in range(256):
        r=(13*x+7)%256
        ok+=((x-r)-((x+19)-r))%256 == (-19)%256
    add("low_rank_projection",ok,256,"Masking both coordinates with the same rank-one mask leaves their difference exposed.")
    # Positive control in the ideal, finite uniform model; NOT a failed statistical attack.
    q=8;reference=None;equal=True
    for x in range(q):
        h=Counter(((x-r)%q,(3*r-s)%q) for r in range(q) for s in range(q))
        equal &= len(h)==q*q and all(v==1 for v in h.values())
        if reference is None:reference=h
        equal &= h==reference
    return {"schema_version":"pllm.attack_fixture_result.v1",
            "origin":"public_reference_fixtures_executed", "findings":findings,
            "ideal_uniform_control":{"q":8,"assignments":512,"exact_equal_joint_views":bool(equal)},
            "production_runtime_attacked":False,"whole_protocol_privacy_proven":False}
