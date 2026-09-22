//! Canonical JSON-safe identities, plans, locks, and assurance records.

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const LOGICAL_PLAN_SCHEMA_VERSION: &str = "pllm.logical_plan.v2";
pub const EXECUTION_PLAN_SCHEMA_VERSION: &str = "pllm.execution_plan.v1";
pub const PLAN_LOCK_SCHEMA_VERSION: &str = "pllm.plan_lock.v1";
pub const ASSURANCE_RESULT_SCHEMA_VERSION: &str = "pllm.assurance_result.v1";
pub const PRIVACY_CONTRACT_SCHEMA_VERSION: &str = "pllm.privacy_contract.v1";
pub const LOCKED_CONTEXT_SCHEMA_VERSION: &str = "pllm.locked_context.v2";
pub const PIPELINE_DIGEST_DOMAIN: &str = "pllm.pipeline.v2";

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Digest(String);

impl Digest {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn from_sha256(value: [u8; 32]) -> Self {
        encode_digest(value)
    }

    pub fn try_clone(&self) -> Result<Self, std::collections::TryReserveError> {
        let mut value = String::new();
        value.try_reserve_exact(self.0.len())?;
        value.push_str(&self.0);
        Ok(Self(value))
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(de::Error::custom(
                "digest must contain exactly 64 lowercase hexadecimal characters",
            ));
        }
        Ok(Self(value))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NamedDigest {
    pub id: String,
    pub digest: Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VersionedArtifact {
    pub id: String,
    pub version: String,
    pub digest: Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceReference {
    pub id: String,
    pub digest: Digest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AssuranceOutcome {
    ProvedInModel,
    RefutedInScope,
    NotRefuted,
    Inconclusive,
    OutsideContract,
    Unchecked,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplementationRefinement {
    NotEstablished,
    TestedDifferential,
    ProvedForLockedCode,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AssuranceOrigin {
    SolverExecuted,
    AdversaryExecuted,
    SourceReported,
    HumanReview,
    NotRun,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AssuranceResult {
    pub schema_version: String,
    pub id: String,
    pub claim_id: String,
    pub outcome: AssuranceOutcome,
    pub scope: String,
    pub origin: AssuranceOrigin,
    pub implementation_refinement: ImplementationRefinement,
    pub assumptions: Vec<String>,
    pub evidence_paths: Vec<String>,
    pub tool: Option<String>,
    pub tool_version: Option<String>,
    pub code_digest: Option<Digest>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimRequirement {
    pub claim_id: String,
    pub accepted_outcomes: BTreeSet<AssuranceOutcome>,
    pub accepted_implementation_refinements: BTreeSet<ImplementationRefinement>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrivacyContract {
    pub schema_version: String,
    pub id: String,
    pub online_parties: u16,
    pub allow_online_preparation: bool,
    pub allow_client_weights: bool,
    pub allow_he: bool,
    pub allow_experimental: bool,
    pub required_claims: Vec<ClaimRequirement>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LockedContext {
    pub schema_version: String,
    pub composition_digest: Digest,
    pub model: NamedDigest,
    pub tokenizer: NamedDigest,
    pub semantic_graph: NamedDigest,
    pub numeric_graph: NamedDigest,
    pub protected_graph: NamedDigest,
    pub roles: Vec<String>,
    pub privacy_contract: NamedDigest,
    pub workload: NamedDigest,
    pub target: NamedDigest,
    pub material_requests: Vec<NamedDigest>,
    pub resource_forecast: NamedDigest,
    pub compiler: VersionedArtifact,
    pub execution_role: String,
    pub static_role_plans: Vec<RolePlanReference>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalPlan {
    pub schema_version: String,
    pub configuration_digest: Digest,
    pub composition_digest: Digest,
    pub model: NamedDigest,
    pub tokenizer: NamedDigest,
    pub semantic_graph: NamedDigest,
    pub numeric_graph: NamedDigest,
    pub protected_graph: NamedDigest,
    pub roles: Vec<String>,
    pub privacy_contract: NamedDigest,
    pub workload: NamedDigest,
    pub required_claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedComponent {
    pub slot: String,
    pub component: String,
    pub version: String,
    pub artifact_digest: Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RolePlanReference {
    pub role: String,
    pub digest: Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlan {
    pub schema_version: String,
    pub logical_plan_digest: Digest,
    pub target: NamedDigest,
    pub components: Vec<ResolvedComponent>,
    pub conversions: Vec<NamedDigest>,
    pub role_plans: Vec<RolePlanReference>,
    pub material_requests: Vec<NamedDigest>,
    pub resource_forecast: NamedDigest,
    pub evidence_references: Vec<EvidenceReference>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlanLock {
    pub schema_version: String,
    pub configuration_digest: Digest,
    pub logical_plan_digest: Digest,
    pub execution_plan_digest: Digest,
    pub model_digest: Digest,
    pub tokenizer_digest: Digest,
    pub numeric_graph_digest: Digest,
    pub privacy_contract_digest: Digest,
    pub workload_digest: Digest,
    pub compiler: VersionedArtifact,
    pub components: Vec<VersionedArtifact>,
    pub evidence_references: Vec<EvidenceReference>,
}

pub fn canonical_bytes<T: Serialize>(value: &T) -> Vec<u8> {
    let value = serde_json::to_value(value)
        .expect("canonical PLLM records contain serializable JSON values");
    let mut bytes = Vec::new();
    write_canonical_value(&value, &mut bytes);
    bytes
}

fn write_canonical_value(value: &serde_json::Value, bytes: &mut Vec<u8>) {
    match value {
        serde_json::Value::Null => bytes.extend_from_slice(b"null"),
        serde_json::Value::Bool(value) => {
            bytes.extend_from_slice(if *value { b"true" } else { b"false" })
        }
        serde_json::Value::Number(value) => bytes.extend_from_slice(value.to_string().as_bytes()),
        serde_json::Value::String(value) => bytes.extend_from_slice(
            &serde_json::to_vec(value).expect("JSON string value is serializable"),
        ),
        serde_json::Value::Array(values) => {
            bytes.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    bytes.push(b',');
                }
                write_canonical_value(value, bytes);
            }
            bytes.push(b']');
        }
        serde_json::Value::Object(values) => {
            bytes.push(b'{');
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    bytes.push(b',');
                }
                bytes.extend_from_slice(
                    &serde_json::to_vec(key).expect("JSON object key is serializable"),
                );
                bytes.push(b':');
                write_canonical_value(value, bytes);
            }
            bytes.push(b'}');
        }
    }
}

pub fn digest_bytes(domain: &str, bytes: &[u8]) -> Digest {
    let mut hash = Sha256::new();
    hash.update(domain.as_bytes());
    hash.update([0]);
    hash.update(bytes);
    encode_digest(hash.finalize())
}

pub fn configuration_digest_bytes(bytes: &[u8]) -> Digest {
    digest_bytes("pllm.configuration.v1", bytes)
}

pub fn pipeline_digest_bytes(bytes: &[u8]) -> Digest {
    digest_bytes(PIPELINE_DIGEST_DOMAIN, bytes)
}

pub fn pipeline_digest<T: Serialize>(pipeline: &T) -> Digest {
    canonical_digest(PIPELINE_DIGEST_DOMAIN, pipeline)
}

fn encode_digest(value: impl AsRef<[u8]>) -> Digest {
    let mut encoded = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in value.as_ref() {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Digest(encoded)
}

pub fn canonical_digest<T: Serialize>(domain: &str, value: &T) -> Digest {
    digest_bytes(domain, &canonical_bytes(value))
}

pub fn logical_plan_bytes(plan: &LogicalPlan) -> Vec<u8> {
    canonical_bytes(plan)
}

pub fn logical_plan_digest(plan: &LogicalPlan) -> Digest {
    canonical_digest(LOGICAL_PLAN_SCHEMA_VERSION, plan)
}

pub fn execution_plan_bytes(plan: &ExecutionPlan) -> Vec<u8> {
    canonical_bytes(plan)
}

pub fn execution_plan_digest(plan: &ExecutionPlan) -> Digest {
    canonical_digest(EXECUTION_PLAN_SCHEMA_VERSION, plan)
}

pub fn plan_lock_bytes(lock: &PlanLock) -> Vec<u8> {
    canonical_bytes(lock)
}

pub fn plan_lock_digest(lock: &PlanLock) -> Digest {
    canonical_digest(PLAN_LOCK_SCHEMA_VERSION, lock)
}

pub fn assurance_result_bytes(result: &AssuranceResult) -> Vec<u8> {
    canonical_bytes(result)
}

pub fn assurance_result_digest(result: &AssuranceResult) -> Digest {
    canonical_digest(ASSURANCE_RESULT_SCHEMA_VERSION, result)
}

pub fn privacy_contract_digest(contract: &PrivacyContract) -> Digest {
    canonical_digest(PRIVACY_CONTRACT_SCHEMA_VERSION, contract)
}

pub fn valid_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"./_:-".contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_serde_is_strict() {
        let valid = format!("\"{}\"", "a".repeat(64));
        let digest: Digest = serde_json::from_str(&valid).unwrap();
        assert_eq!(digest.as_str(), "a".repeat(64));
        assert_eq!(digest.try_clone().unwrap(), digest);
        assert_eq!(serde_json::to_string(&digest).unwrap(), valid);
        for invalid in [
            format!("\"{}\"", "a".repeat(63)),
            format!("\"{}\"", "A".repeat(64)),
            format!("\"{}\"", "g".repeat(64)),
        ] {
            assert!(serde_json::from_str::<Digest>(&invalid).is_err());
        }
    }

    #[test]
    fn assurance_vocabularies_are_exact() {
        assert_eq!(
            canonical_bytes(&[
                AssuranceOutcome::ProvedInModel,
                AssuranceOutcome::RefutedInScope,
                AssuranceOutcome::NotRefuted,
                AssuranceOutcome::Inconclusive,
                AssuranceOutcome::OutsideContract,
                AssuranceOutcome::Unchecked,
            ]),
            br#"["proved_in_model","refuted_in_scope","not_refuted","inconclusive","outside_contract","unchecked"]"#
        );
        assert_eq!(
            canonical_bytes(&[
                ImplementationRefinement::NotEstablished,
                ImplementationRefinement::TestedDifferential,
                ImplementationRefinement::ProvedForLockedCode,
            ]),
            br#"["not_established","tested_differential","proved_for_locked_code"]"#
        );
    }

    #[test]
    fn experiment_configuration_digest_matches_python_identity() {
        let fixture = include_str!("../../../schemas/fixtures/experiment.valid.json");
        let value: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let bytes = canonical_bytes(&value);
        assert_eq!(
            configuration_digest_bytes(&bytes).as_str(),
            "cd051de9c3dfdfe2e3f13d1316582a844d5494529cf772c58502a44021875329"
        );
        assert_eq!(
            pipeline_digest(&value["pipeline"]),
            pipeline_digest_bytes(&canonical_bytes(&value["pipeline"]))
        );
        assert_eq!(
            pipeline_digest(&value["pipeline"]).as_str(),
            "b8531d93f0bdc4ca759981903aa67ccc726ff5652ba59c714af468b60593cd8c"
        );
    }
}
