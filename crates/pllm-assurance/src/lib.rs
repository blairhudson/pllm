//! Deterministic finite checks of public assurance fixtures.
//!
//! These checks target deliberately weakened examples, not production PLLM code.

use pllm_types::{
    canonical_bytes, canonical_digest, AssuranceOrigin, AssuranceOutcome, AssuranceResult, Digest,
    ImplementationRefinement, ASSURANCE_RESULT_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const ASSURANCE_REPORT_SCHEMA_VERSION: &str = "pllm.assurance_report.v1";
pub const REPORT_ORIGIN: &str = "public_reference_fixtures_executed";
pub const WEAKENED_FIXTURE_TARGET: &str = "deliberately_weakened_public_fixture";

pub const LIMITATIONS: [&str; 3] = [
    "production runtime not attacked",
    "whole-protocol privacy not established",
    "computational-security proof not established",
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AssuranceReport {
    pub schema_version: String,
    pub origin: String,
    pub findings: Vec<Finding>,
    pub ideal_uniform_control: IdealUniformControl,
    pub limitations: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Finding {
    pub id: String,
    pub outcome: AssuranceOutcome,
    pub cases: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub successes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mismatches: Option<u64>,
    pub scope: String,
    pub target: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IdealUniformControl {
    pub outcome: AssuranceOutcome,
    pub scope: String,
    pub q: u64,
    pub assignments: u64,
}

impl AssuranceReport {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        canonical_bytes(self)
    }

    pub fn digest(&self) -> Digest {
        canonical_digest(ASSURANCE_REPORT_SCHEMA_VERSION, self)
    }

    pub fn to_assurance_results(&self) -> Vec<AssuranceResult> {
        let mut results: Vec<_> = self
            .findings
            .iter()
            .map(|finding| AssuranceResult {
                schema_version: ASSURANCE_RESULT_SCHEMA_VERSION.to_owned(),
                id: format!("assurance.public_fixture.{}", finding.id),
                claim_id: format!("public_fixture.{}", finding.id),
                outcome: finding.outcome,
                scope: finding.scope.clone(),
                origin: AssuranceOrigin::AdversaryExecuted,
                implementation_refinement: ImplementationRefinement::NotEstablished,
                assumptions: vec![finding.target.clone()],
                evidence_paths: Vec::new(),
                tool: Some(env!("CARGO_PKG_NAME").to_owned()),
                tool_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
                code_digest: None,
            })
            .collect();

        results.push(AssuranceResult {
            schema_version: ASSURANCE_RESULT_SCHEMA_VERSION.to_owned(),
            id: "assurance.ideal_uniform_control.q8".to_owned(),
            claim_id: "ideal_uniform_control.q8".to_owned(),
            outcome: self.ideal_uniform_control.outcome,
            scope: self.ideal_uniform_control.scope.clone(),
            origin: AssuranceOrigin::AdversaryExecuted,
            implementation_refinement: ImplementationRefinement::NotEstablished,
            assumptions: vec!["independent uniform masks in the enumerated finite model".to_owned()],
            evidence_paths: Vec::new(),
            tool: Some(env!("CARGO_PKG_NAME").to_owned()),
            tool_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
            code_digest: None,
        });
        results
    }
}

/// Executes exact finite enumerations matching `research/assurance/attacks/fixtures.py`.
pub fn run() -> AssuranceReport {
    let mut findings = Vec::with_capacity(6);

    let q = 256_i32;
    let mask_reuse_successes = (0..q)
        .filter(|&x| {
            let mask = (97 * x + 13).rem_euclid(q);
            let first = (x - mask).rem_euclid(q);
            let second = (x + 31 - mask).rem_euclid(q);
            (second - first).rem_euclid(q) == 31
        })
        .count() as u64;
    findings.push(success_finding(
        "mask_reuse",
        mask_reuse_successes,
        q as u64,
        "Reused full-ring mask discloses input differences.",
    ));

    let p = 251_i32;
    let delta = [19, 73, 1];
    let base = [61, 4, 151];
    let affine_label_reuse_successes = (0..p)
        .filter(|&x| {
            let first = std::array::from_fn::<_, 3, _>(|index| {
                (base[index] + x * delta[index]).rem_euclid(p)
            });
            let second = std::array::from_fn::<_, 3, _>(|index| {
                (base[index] + (x + 7) * delta[index]).rem_euclid(p)
            });
            let difference = std::array::from_fn::<_, 3, _>(|index| {
                (second[index] - first[index]).rem_euclid(p)
            });
            let inverse = modular_inverse(difference[2], p);
            difference.map(|value| (value * inverse).rem_euclid(p)) == delta
        })
        .count() as u64;
    findings.push(success_finding(
        "affine_label_reuse",
        affine_label_reuse_successes,
        p as u64,
        "Two semantic values on one affine wire expose offset with unit coordinate.",
    ));

    let domain: Vec<_> = (-9..=9).collect();
    let sparse_point_permute_successes = (0..p)
        .filter(|&shift| {
            let exposed: BTreeSet<_> = domain.iter().map(|x| (shift + x).rem_euclid(p)).collect();
            let candidates: Vec<_> = (0..p)
                .filter(|&candidate| {
                    domain
                        .iter()
                        .map(|x| (candidate + x).rem_euclid(p))
                        .collect::<BTreeSet<_>>()
                        == exposed
                })
                .collect();
            candidates == [shift]
        })
        .count() as u64;
    findings.push(success_finding(
        "sparse_point_permute",
        sparse_point_permute_successes,
        p as u64,
        "Visible sparse row indices expose shift of proper cyclic interval.",
    ));

    let missing_truncation_carry_mismatches = (0_u32..256)
        .flat_map(|left| (0_u32..256).map(move |right| (left, right)))
        .filter(|&(left, right)| ((left >> 4) + (right >> 4)) % 16 != ((left + right) % 256) >> 4)
        .count() as u64;
    findings.push(mismatch_finding(
        "missing_truncation_carry",
        missing_truncation_carry_mismatches,
        65_536,
        "Independent local shifts omit carry.",
    ));

    let early_exit_metadata_successes = (0_u32..256)
        .filter(|&x| ((if x >= 128 { 1 } else { 8 }) == 1) == (x >= 128))
        .count() as u64;
    findings.push(success_finding(
        "early_exit_metadata",
        early_exit_metadata_successes,
        256,
        "Unpadded sign-dependent path reveals sign.",
    ));

    let low_rank_projection_successes = (0..q)
        .filter(|&x| {
            let mask = (13 * x + 7).rem_euclid(q);
            ((x - mask) - ((x + 19) - mask)).rem_euclid(q) == (-19_i32).rem_euclid(q)
        })
        .count() as u64;
    findings.push(success_finding(
        "low_rank_projection",
        low_rank_projection_successes,
        q as u64,
        "Shared rank-one mask leaves coordinate difference exposed.",
    ));

    let ideal_q = 8_usize;
    let mut reference = None;
    let mut equal = true;
    for x in 0..ideal_q {
        let mut view = [0_u8; 64];
        for mask in 0..ideal_q {
            for pad in 0..ideal_q {
                let first = (x + ideal_q - mask) % ideal_q;
                let second = (3 * mask + ideal_q - pad) % ideal_q;
                view[first * ideal_q + second] += 1;
            }
        }
        equal &= view.iter().all(|&count| count == 1);
        match reference {
            Some(previous) => equal &= view == previous,
            None => reference = Some(view),
        }
    }

    AssuranceReport {
        schema_version: ASSURANCE_REPORT_SCHEMA_VERSION.to_owned(),
        origin: REPORT_ORIGIN.to_owned(),
        findings,
        ideal_uniform_control: IdealUniformControl {
            outcome: if equal {
                AssuranceOutcome::ProvedInModel
            } else {
                AssuranceOutcome::RefutedInScope
            },
            scope: "Exact enumeration over q=8 ideal independent uniform masks.".to_owned(),
            q: ideal_q as u64,
            assignments: (ideal_q * ideal_q * ideal_q) as u64,
        },
        limitations: LIMITATIONS
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
    }
}

pub fn report_bytes() -> Vec<u8> {
    run().canonical_bytes()
}

pub fn report_digest() -> Digest {
    run().digest()
}

pub fn assurance_results() -> Vec<AssuranceResult> {
    run().to_assurance_results()
}

fn success_finding(id: &str, successes: u64, cases: u64, scope: &str) -> Finding {
    Finding {
        id: id.to_owned(),
        outcome: AssuranceOutcome::RefutedInScope,
        cases,
        successes: Some(successes),
        mismatches: None,
        scope: scope.to_owned(),
        target: WEAKENED_FIXTURE_TARGET.to_owned(),
    }
}

fn mismatch_finding(id: &str, mismatches: u64, cases: u64, scope: &str) -> Finding {
    Finding {
        id: id.to_owned(),
        outcome: AssuranceOutcome::RefutedInScope,
        cases,
        successes: None,
        mismatches: Some(mismatches),
        scope: scope.to_owned(),
        target: WEAKENED_FIXTURE_TARGET.to_owned(),
    }
}

fn modular_inverse(value: i32, modulus: i32) -> i32 {
    (1..modulus)
        .find(|candidate| (value * candidate).rem_euclid(modulus) == 1)
        .expect("fixture denominator is invertible")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_public_fixture_counts() {
        let report = run();
        let counts: Vec<_> = report
            .findings
            .iter()
            .map(|finding| {
                (
                    finding.id.as_str(),
                    finding.successes,
                    finding.mismatches,
                    finding.cases,
                )
            })
            .collect();
        assert_eq!(
            counts,
            [
                ("mask_reuse", Some(256), None, 256),
                ("affine_label_reuse", Some(251), None, 251),
                ("sparse_point_permute", Some(251), None, 251),
                ("missing_truncation_carry", None, Some(30_720), 65_536),
                ("early_exit_metadata", Some(256), None, 256),
                ("low_rank_projection", Some(256), None, 256),
            ]
        );
        assert_eq!(report.ideal_uniform_control.q, 8);
        assert_eq!(report.ideal_uniform_control.assignments, 512);
    }

    #[test]
    fn report_status_and_limitations_are_narrow() {
        let report = run();
        assert_eq!(report.schema_version, ASSURANCE_REPORT_SCHEMA_VERSION);
        assert!(report.findings.iter().all(|finding| {
            finding.outcome == AssuranceOutcome::RefutedInScope
                && finding.target == WEAKENED_FIXTURE_TARGET
                && (finding.successes.is_some() ^ finding.mismatches.is_some())
        }));
        assert_eq!(
            report.ideal_uniform_control.outcome,
            AssuranceOutcome::ProvedInModel
        );
        assert_eq!(report.limitations, LIMITATIONS);
    }

    #[test]
    fn canonical_encoding_and_digest_are_stable() {
        let first = run();
        let second = run();
        assert_eq!(first.canonical_bytes(), second.canonical_bytes());
        assert_eq!(first.digest(), second.digest());
        assert_eq!(
            first.digest().as_str(),
            "d76e814d77fd726e127448c738495c181a3081bd85df2904c4a0b67b4d3bcffa"
        );
    }

    #[test]
    fn assurance_result_conversion_does_not_claim_code_refinement() {
        let results = assurance_results();
        assert_eq!(results.len(), 7);
        assert!(results.iter().all(|result| {
            result.schema_version == ASSURANCE_RESULT_SCHEMA_VERSION
                && result.origin == AssuranceOrigin::AdversaryExecuted
                && result.implementation_refinement == ImplementationRefinement::NotEstablished
                && result.code_digest.is_none()
        }));
        assert_eq!(
            results
                .iter()
                .map(|result| (result.id.as_str(), result.claim_id.as_str()))
                .collect::<Vec<_>>(),
            [
                (
                    "assurance.public_fixture.mask_reuse",
                    "public_fixture.mask_reuse"
                ),
                (
                    "assurance.public_fixture.affine_label_reuse",
                    "public_fixture.affine_label_reuse"
                ),
                (
                    "assurance.public_fixture.sparse_point_permute",
                    "public_fixture.sparse_point_permute"
                ),
                (
                    "assurance.public_fixture.missing_truncation_carry",
                    "public_fixture.missing_truncation_carry"
                ),
                (
                    "assurance.public_fixture.early_exit_metadata",
                    "public_fixture.early_exit_metadata"
                ),
                (
                    "assurance.public_fixture.low_rank_projection",
                    "public_fixture.low_rank_projection"
                ),
                (
                    "assurance.ideal_uniform_control.q8",
                    "ideal_uniform_control.q8"
                ),
            ]
        );
        assert!(results[..6]
            .iter()
            .all(|result| result.outcome == AssuranceOutcome::RefutedInScope));
        assert_eq!(results[6].outcome, AssuranceOutcome::ProvedInModel);
    }

    #[test]
    fn canonical_report_round_trips_without_shape_loss() {
        let report = run();
        let decoded: AssuranceReport = serde_json::from_slice(&report.canonical_bytes()).unwrap();
        assert_eq!(decoded, report);
    }
}
