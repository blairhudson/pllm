use crate::{
    canonical_digest, digest_array, validate_rms_norm_q10_direct_region, Digest,
    ModelRmsNormQ10DirectRegion,
};
use pllm_garble::boolean_stream::{
    estimate_rms_norm_q10_direct_stream, evaluate_rms_norm_q10_direct_stream,
    garble_rms_norm_q10_direct_stream, RmsNormQ10StreamClient, RmsNormQ10StreamDecoder,
    RmsNormQ10StreamInputs, RmsNormQ10StreamOutputs, UnreviewedRmsNormQ10Stream,
    RMS_NORM_Q10_STREAM_METHOD_ID, RMS_NORM_Q10_STREAM_TOPOLOGY_ID,
};
use pllm_models::DecoderPlan;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Seek, Write};
use std::sync::{Mutex, OnceLock};

const SOURCE_IDENTITY_DOMAIN: &str = "pllm.protected.rms_norm_q10_stream.source.v1";
const TICKET_DIGEST_DOMAIN: &str = "pllm.protected.rms_norm_q10_stream.ticket.v1";
const BODY_DIGEST_DOMAIN: &str = "pllm.protected.rms_norm_q10_stream.body.v1";
const WEIGHTS_DIGEST_DOMAIN: &str = "pllm.protected.rms_norm_q10_stream.weights.v1";
const REGION_DIGEST_DOMAIN: &str = "pllm.protected.rms_norm_q10_stream.region.v1";
const MAX_BODY_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_ADMITTED_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_ISSUANCES: usize = 64;

#[cfg(test)]
pub(crate) static RMS_REGISTRY_TEST_LOCK: Mutex<()> = Mutex::new(());
const MAX_PLAN_OPERATIONS: usize = 100_000;
const MAX_PLAN_STATES: usize = 100_000;
const MAX_PLAN_DIGEST_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_JSON_DEPTH: usize = 64;

/// Explicit acknowledgement for public weights and unreviewed reference cryptography.
#[derive(Clone, Copy, Eq, PartialEq, Serialize)]
pub struct ExperimentalRmsNormQ10StreamPolicy {
    max_body_bytes: u64,
}

/// Resource bound carried by the explicit unreviewed streamed-RMSNorm acknowledgement.
pub type RmsNormQ10StreamResourcePolicy = ExperimentalRmsNormQ10StreamPolicy;

pub(crate) fn rms_norm_q10_stream_body_bytes(
    weights: &[i16],
    policy: &RmsNormQ10StreamResourcePolicy,
) -> Result<u64, String> {
    if policy.max_body_bytes == 0 || policy.max_body_bytes > MAX_BODY_BYTES {
        return Err("streamed protected Q10 RMSNorm policy is invalid".into());
    }
    let estimate =
        estimate_rms_norm_q10_direct_stream(weights).map_err(|error| error.to_string())?;
    if estimate.evaluator_body_bytes > policy.max_body_bytes {
        return Err(format!(
            "streamed protected Q10 RMSNorm requires {} body bytes, exceeding policy limit {}",
            estimate.evaluator_body_bytes, policy.max_body_bytes
        ));
    }
    Ok(estimate.evaluator_body_bytes)
}

pub(crate) fn preflight_rms_norm_q10_stream_registry(
    additional_issuances: usize,
    additional_bytes: u64,
) -> Result<(), String> {
    // Every reserved, published, and actively claimed issuance remains in this registry.
    let registry = registry()?;
    validate_registry_capacity(&registry, additional_issuances, additional_bytes)
}

impl ExperimentalRmsNormQ10StreamPolicy {
    pub fn acknowledge_unreviewed_public_weights(max_body_bytes: u64) -> Result<Self, String> {
        if max_body_bytes == 0 || max_body_bytes > MAX_BODY_BYTES {
            return Err(
                "streamed protected Q10 RMSNorm body limit must be between 1 byte and 4 GiB".into(),
            );
        }
        Ok(Self { max_body_bytes })
    }

    pub fn max_body_bytes(&self) -> u64 {
        self.max_body_bytes
    }
}

#[derive(Serialize)]
struct SourceBinding<'a> {
    plan: &'a DecoderPlan,
    region: &'a ModelRmsNormQ10DirectRegion,
    row_index: usize,
    weights: &'a [i16],
    max_body_bytes: u64,
    and_gate_count: u64,
    evaluator_body_bytes: u64,
    method_id: &'static str,
    topology_id: &'static str,
    artifact_digest: &'a Digest,
    issuance_nonce: [u8; 32],
}

#[derive(Serialize)]
struct TicketCommitment<'a> {
    plan_digest: &'a Digest,
    region_digest: &'a Digest,
    row_index: usize,
    weights_digest: &'a Digest,
    width: usize,
    max_body_bytes: u64,
    and_gate_count: u64,
    body_bytes: u64,
    method_id: &'static str,
    topology_id: &'static str,
    artifact_digest: &'a Digest,
    issuance_nonce: [u8; 32],
    source_identity: [u8; 32],
    body_digest: &'a Digest,
}

/// Process-local, one-use evaluator authorization. Deliberately not Clone or serializable.
pub struct RmsNormQ10StreamTicket {
    plan_digest: Digest,
    region_digest: Digest,
    row_index: usize,
    weights_digest: Digest,
    width: usize,
    max_body_bytes: u64,
    and_gate_count: u64,
    body_bytes: u64,
    artifact_digest: Digest,
    issuance_nonce: [u8; 32],
    source_identity: [u8; 32],
    body_digest: Digest,
    commitment_digest: Option<Digest>,
    drop_commitment: [u8; 32],
    registered: bool,
}

impl RmsNormQ10StreamTicket {
    pub fn body_bytes(&self) -> u64 {
        self.body_bytes
    }

    pub fn and_gate_count(&self) -> u64 {
        self.and_gate_count
    }
}

impl Drop for RmsNormQ10StreamTicket {
    fn drop(&mut self) {
        if let (true, Some(digest)) = (
            self.registered && drop_commitment(self) == self.drop_commitment,
            self.commitment_digest.as_ref(),
        ) {
            let _ = unregister_issuance(self.issuance_nonce, digest);
        }
    }
}

pub struct BoundRmsNormQ10StreamMaterial {
    client: Option<RmsNormQ10StreamClient>,
    ticket: Option<RmsNormQ10StreamTicket>,
}

pub struct BoundRmsNormQ10StreamInputs(RmsNormQ10StreamInputs);
pub struct BoundRmsNormQ10StreamDecoder(RmsNormQ10StreamDecoder);
pub struct BoundRmsNormQ10StreamOutputs(RmsNormQ10StreamOutputs);

impl BoundRmsNormQ10StreamMaterial {
    pub fn ticket(&self) -> &RmsNormQ10StreamTicket {
        self.ticket.as_ref().expect("ticket exists until encode")
    }

    pub fn encode(
        mut self,
        input: &[i16],
    ) -> Result<
        (
            RmsNormQ10StreamTicket,
            BoundRmsNormQ10StreamInputs,
            BoundRmsNormQ10StreamDecoder,
        ),
        String,
    > {
        let client = self
            .client
            .take()
            .ok_or("streamed protected Q10 RMSNorm client material disappeared")?;
        let ticket = self
            .ticket
            .take()
            .ok_or("streamed protected Q10 RMSNorm ticket disappeared")?;
        let (inputs, decoder) = client.encode(input).map_err(|error| error.to_string())?;
        Ok((
            ticket,
            BoundRmsNormQ10StreamInputs(inputs),
            BoundRmsNormQ10StreamDecoder(decoder),
        ))
    }
}

impl BoundRmsNormQ10StreamDecoder {
    pub fn decode(self, outputs: BoundRmsNormQ10StreamOutputs) -> Result<Vec<i32>, String> {
        self.0.decode(outputs.0).map_err(|error| error.to_string())
    }
}

/// Claimed evaluator state retains global admitted bytes through evaluation or drop.
pub struct BoundRmsNormQ10StreamEvaluator {
    ticket: RmsNormQ10StreamTicket,
    weights: Vec<i16>,
    inputs: Option<RmsNormQ10StreamInputs>,
    active_claim: Option<ActiveClaim>,
}

impl BoundRmsNormQ10StreamEvaluator {
    /// Atomically burn authentic issuance before validating caller context.
    pub fn claim(
        plan: &DecoderPlan,
        region: &ModelRmsNormQ10DirectRegion,
        row_index: usize,
        weights: &[i16],
        policy: &ExperimentalRmsNormQ10StreamPolicy,
        mut ticket: RmsNormQ10StreamTicket,
        inputs: BoundRmsNormQ10StreamInputs,
    ) -> Result<Self, String> {
        let digest = ticket
            .commitment_digest
            .as_ref()
            .ok_or("streamed protected Q10 RMSNorm ticket has no commitment")?;
        let active_claim = consume_issuance(ticket.issuance_nonce, digest)?;
        ticket.registered = false;

        if inputs.0.source_identity() != ticket.source_identity {
            return Err("streamed protected Q10 RMSNorm inputs have a different source".into());
        }
        preflight_plan_digest_input(plan)?;
        let (rows, width) = validate_rms_norm_q10_direct_region(plan, region)?;
        if row_index >= rows || weights.len() != width {
            return Err("streamed protected Q10 RMSNorm context has invalid row or weights".into());
        }
        let estimate =
            estimate_rms_norm_q10_direct_stream(weights).map_err(|error| error.to_string())?;
        let artifact_digest = crate::protected_rms_norm_q10_artifact_digest();
        let region_digest = canonical_digest(REGION_DIGEST_DOMAIN, region);
        let weights_digest = canonical_digest(WEIGHTS_DIGEST_DOMAIN, &weights);
        let source_identity = source_identity(&SourceBinding {
            plan,
            region,
            row_index,
            weights,
            max_body_bytes: policy.max_body_bytes,
            and_gate_count: estimate.and_gate_count,
            evaluator_body_bytes: estimate.evaluator_body_bytes,
            method_id: RMS_NORM_Q10_STREAM_METHOD_ID,
            topology_id: RMS_NORM_Q10_STREAM_TOPOLOGY_ID,
            artifact_digest: &artifact_digest,
            issuance_nonce: ticket.issuance_nonce,
        })?;
        if ticket.plan_digest != plan.digest()
            || ticket.region_digest != region_digest
            || ticket.row_index != row_index
            || ticket.weights_digest != weights_digest
            || ticket.width != width
            || ticket.max_body_bytes != policy.max_body_bytes
            || ticket.and_gate_count != estimate.and_gate_count
            || ticket.body_bytes != estimate.evaluator_body_bytes
            || ticket.body_bytes > policy.max_body_bytes
            || ticket.artifact_digest != artifact_digest
            || ticket.source_identity != source_identity
        {
            return Err("streamed protected Q10 RMSNorm ticket context does not match".into());
        }
        let mut owned_weights = Vec::new();
        owned_weights
            .try_reserve_exact(weights.len())
            .map_err(|_| "streamed protected Q10 RMSNorm weight allocation failed")?;
        owned_weights.extend_from_slice(weights);
        Ok(Self {
            ticket,
            weights: owned_weights,
            inputs: Some(inputs.0),
            active_claim: Some(active_claim),
        })
    }

    /// Open reader only after claim burn; authenticate complete bytes and EOF before evaluation.
    pub fn evaluate_with_reader<R, F>(
        mut self,
        open_reader: F,
    ) -> Result<BoundRmsNormQ10StreamOutputs, String>
    where
        R: Read,
        F: FnOnce() -> Result<R, String>,
    {
        let mut reader = open_reader()?;
        let mut spool = authenticate_body(&mut reader, &self.ticket)?;
        spool
            .rewind()
            .map_err(|error| format!("failed to rewind streamed RMSNorm body spool: {error}"))?;
        let inputs = self
            .inputs
            .take()
            .ok_or("streamed protected Q10 RMSNorm inputs disappeared")?;
        let outputs = evaluate_rms_norm_q10_direct_stream(
            &UnreviewedRmsNormQ10Stream::acknowledge_unreviewed_public_weights(),
            &self.weights,
            self.ticket.source_identity,
            self.ticket.max_body_bytes,
            inputs,
            &mut spool,
        );
        drop(self.active_claim.take());
        outputs
            .map(BoundRmsNormQ10StreamOutputs)
            .map_err(|error| error.to_string())
    }
}

/// Garble into anonymous spool, commit complete body, register ticket, then publish.
/// Any publication error or panic requires caller to discard sink contents.
pub fn prepare_bound_rms_norm_q10_stream_row<W: Write>(
    plan: &DecoderPlan,
    region: &ModelRmsNormQ10DirectRegion,
    row_index: usize,
    weights: &[i16],
    policy: &ExperimentalRmsNormQ10StreamPolicy,
    writer: &mut W,
) -> Result<BoundRmsNormQ10StreamMaterial, String> {
    preflight_plan_digest_input(plan)?;
    let (rows, width) = validate_rms_norm_q10_direct_region(plan, region)?;
    if row_index >= rows {
        return Err("streamed protected Q10 RMSNorm row is outside its semantic tensor".into());
    }
    if weights.len() != width {
        return Err(format!(
            "streamed protected Q10 RMSNorm requires {width} weights, received {}",
            weights.len()
        ));
    }
    let estimate =
        estimate_rms_norm_q10_direct_stream(weights).map_err(|error| error.to_string())?;
    if rms_norm_q10_stream_body_bytes(weights, policy)? != estimate.evaluator_body_bytes {
        return Err("streamed protected Q10 RMSNorm estimate is inconsistent".into());
    }
    let reservation = reserve_capacity(estimate.evaluator_body_bytes)?;
    let mut issuance_nonce = [0_u8; 32];
    getrandom::fill(&mut issuance_nonce)
        .map_err(|_| "streamed protected Q10 RMSNorm nonce generation failed")?;
    let artifact_digest = crate::protected_rms_norm_q10_artifact_digest();
    let source_identity = source_identity(&SourceBinding {
        plan,
        region,
        row_index,
        weights,
        max_body_bytes: policy.max_body_bytes,
        and_gate_count: estimate.and_gate_count,
        evaluator_body_bytes: estimate.evaluator_body_bytes,
        method_id: RMS_NORM_Q10_STREAM_METHOD_ID,
        topology_id: RMS_NORM_Q10_STREAM_TOPOLOGY_ID,
        artifact_digest: &artifact_digest,
        issuance_nonce,
    })?;
    let mut spool = tempfile::tempfile()
        .map_err(|error| format!("failed to create streamed RMSNorm preparation spool: {error}"))?;
    let mut hashed = HashingWriter::new(&mut spool);
    let client = garble_rms_norm_q10_direct_stream(
        &UnreviewedRmsNormQ10Stream::acknowledge_unreviewed_public_weights(),
        weights,
        source_identity,
        policy.max_body_bytes,
        &mut hashed,
    )
    .map_err(|error| error.to_string())?;
    let (body_digest, body_bytes) = hashed.finish();
    if body_bytes != estimate.evaluator_body_bytes {
        return Err("streamed protected Q10 RMSNorm body length differs from estimate".into());
    }
    let mut ticket = RmsNormQ10StreamTicket {
        plan_digest: plan.digest(),
        region_digest: canonical_digest(REGION_DIGEST_DOMAIN, region),
        row_index,
        weights_digest: canonical_digest(WEIGHTS_DIGEST_DOMAIN, &weights),
        width,
        max_body_bytes: policy.max_body_bytes,
        and_gate_count: estimate.and_gate_count,
        body_bytes,
        artifact_digest,
        issuance_nonce,
        source_identity,
        body_digest,
        commitment_digest: None,
        drop_commitment: [0_u8; 32],
        registered: false,
    };
    let digest = ticket_digest(&ticket);
    ticket.commitment_digest = Some(digest.clone());
    ticket.drop_commitment = drop_commitment(&ticket);
    let mut published = reservation.finalize(issuance_nonce, digest)?;
    ticket.registered = true;
    spool
        .rewind()
        .map_err(|error| format!("failed to rewind streamed RMSNorm preparation spool: {error}"))?;
    if let Err(error) = std::io::copy(&mut spool, writer) {
        return Err(format!(
            "failed to publish streamed RMSNorm body; discard the writer: {error}"
        ));
    }
    if let Err(error) = writer.flush() {
        return Err(format!(
            "failed to flush streamed RMSNorm body; discard the writer: {error}"
        ));
    }
    published.active = false;
    Ok(BoundRmsNormQ10StreamMaterial {
        client: Some(client),
        ticket: Some(ticket),
    })
}

fn source_identity(binding: &SourceBinding<'_>) -> Result<[u8; 32], String> {
    digest_array(&canonical_digest(SOURCE_IDENTITY_DOMAIN, binding))
}

fn preflight_plan_digest_input(plan: &DecoderPlan) -> Result<(), String> {
    let mut budget = MAX_PLAN_DIGEST_INPUT_BYTES;
    charge_string(&mut budget, &plan.schema_version)?;
    charge_string(&mut budget, &plan.model_family)?;
    charge_string(&mut budget, &plan.adapter)?;
    if plan.transformations.len() > 1024 {
        return Err("streamed protected Q10 RMSNorm plan has too many transformations".into());
    }
    for transformation in &plan.transformations {
        charge_string(&mut budget, &transformation.component)?;
        charge_string(&mut budget, &transformation.implementation)?;
        charge_string(&mut budget, &transformation.method_id)?;
    }
    for graph in [&plan.prefill, &plan.decode] {
        if graph.operations.len() > MAX_PLAN_OPERATIONS
            || graph.state_inputs.len() > MAX_PLAN_STATES
            || graph.state_outputs.len() > MAX_PLAN_STATES
        {
            return Err("streamed protected Q10 RMSNorm plan exceeds graph count limits".into());
        }
        charge_string(&mut budget, &graph.output)?;
        for operation in &graph.operations {
            charge_string(&mut budget, &operation.id)?;
            if operation.inputs.len() > 1024 || operation.output_shape.len() > 16 {
                return Err("streamed protected Q10 RMSNorm operation exceeds shape limits".into());
            }
            for input in &operation.inputs {
                charge_string(&mut budget, input)?;
            }
            charge_json(&mut budget, &operation.attributes, 0)?;
        }
        for state in graph.state_inputs.iter().chain(&graph.state_outputs) {
            charge_string(&mut budget, &state.id)?;
            if state.shape.len() > 16 {
                return Err("streamed protected Q10 RMSNorm state rank exceeds limit".into());
            }
        }
    }
    Ok(())
}

fn charge_string(budget: &mut usize, value: &str) -> Result<(), String> {
    *budget = budget
        .checked_sub(value.len())
        .ok_or("streamed protected Q10 RMSNorm plan exceeds digest-input limit")?;
    Ok(())
}

fn charge_json(budget: &mut usize, value: &Value, depth: usize) -> Result<(), String> {
    if depth > MAX_JSON_DEPTH {
        return Err("streamed protected Q10 RMSNorm plan JSON exceeds depth limit".into());
    }
    *budget = budget
        .checked_sub(16)
        .ok_or("streamed protected Q10 RMSNorm plan exceeds digest-input limit")?;
    match value {
        Value::String(value) => charge_string(budget, value),
        Value::Array(values) => {
            for value in values {
                charge_json(budget, value, depth + 1)?;
            }
            Ok(())
        }
        Value::Object(values) => {
            for (key, value) in values {
                charge_string(budget, key)?;
                charge_json(budget, value, depth + 1)?;
            }
            Ok(())
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(()),
    }
}

fn ticket_digest(ticket: &RmsNormQ10StreamTicket) -> Digest {
    canonical_digest(
        TICKET_DIGEST_DOMAIN,
        &TicketCommitment {
            plan_digest: &ticket.plan_digest,
            region_digest: &ticket.region_digest,
            row_index: ticket.row_index,
            weights_digest: &ticket.weights_digest,
            width: ticket.width,
            max_body_bytes: ticket.max_body_bytes,
            and_gate_count: ticket.and_gate_count,
            body_bytes: ticket.body_bytes,
            method_id: RMS_NORM_Q10_STREAM_METHOD_ID,
            topology_id: RMS_NORM_Q10_STREAM_TOPOLOGY_ID,
            artifact_digest: &ticket.artifact_digest,
            issuance_nonce: ticket.issuance_nonce,
            source_identity: ticket.source_identity,
            body_digest: &ticket.body_digest,
        },
    )
}

fn drop_commitment(ticket: &RmsNormQ10StreamTicket) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"pllm.protected.rms_norm_q10.stream.drop_commitment.v1");
    hasher.update(ticket.plan_digest.as_str().as_bytes());
    hasher.update(ticket.region_digest.as_str().as_bytes());
    hasher.update(ticket.row_index.to_le_bytes());
    hasher.update(ticket.weights_digest.as_str().as_bytes());
    hasher.update(ticket.width.to_le_bytes());
    hasher.update(ticket.max_body_bytes.to_le_bytes());
    hasher.update(ticket.and_gate_count.to_le_bytes());
    hasher.update(ticket.body_bytes.to_le_bytes());
    hasher.update(ticket.artifact_digest.as_str().as_bytes());
    hasher.update(ticket.issuance_nonce);
    hasher.update(ticket.source_identity);
    hasher.update(ticket.body_digest.as_str().as_bytes());
    if let Some(digest) = ticket.commitment_digest.as_ref() {
        hasher.update(digest.as_str().as_bytes());
    }
    hasher.finalize().into()
}

fn authenticate_body<R: Read>(
    reader: &mut R,
    ticket: &RmsNormQ10StreamTicket,
) -> Result<std::fs::File, String> {
    let mut spool = tempfile::tempfile().map_err(|error| {
        format!("failed to create streamed RMSNorm authentication spool: {error}")
    })?;
    let mut hasher = domain_hasher(BODY_DIGEST_DOMAIN);
    let mut remaining = ticket.body_bytes;
    let mut buffer = [0_u8; 64 * 1024];
    while remaining != 0 {
        let request = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "streamed RMSNorm body read exceeds usize")?;
        let read = match reader.read(&mut buffer[..request]) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.to_string()),
        };
        if read == 0 {
            return Err("streamed RMSNorm body is truncated".into());
        }
        spool.write_all(&buffer[..read]).map_err(|error| {
            format!("failed to write streamed RMSNorm authentication spool: {error}")
        })?;
        hasher.update(&buffer[..read]);
        remaining -= u64::try_from(read).map_err(|_| "streamed RMSNorm read exceeds u64")?;
    }
    ensure_eof(reader)?;
    if finish_digest(hasher) != ticket.body_digest {
        return Err("streamed RMSNorm body commitment is invalid".into());
    }
    Ok(spool)
}

fn ensure_eof<R: Read>(reader: &mut R) -> Result<(), String> {
    let mut byte = [0_u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err("streamed RMSNorm body has trailing bytes".into()),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.to_string()),
        }
    }
}

struct HashingWriter<'a> {
    writer: &'a mut std::fs::File,
    hasher: Sha256,
    bytes: u64,
}

impl<'a> HashingWriter<'a> {
    fn new(writer: &'a mut std::fs::File) -> Self {
        Self {
            writer,
            hasher: domain_hasher(BODY_DIGEST_DOMAIN),
            bytes: 0,
        }
    }

    fn finish(self) -> (Digest, u64) {
        (finish_digest(self.hasher), self.bytes)
    }
}

impl Write for HashingWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let written = self.writer.write(buffer)?;
        self.hasher.update(&buffer[..written]);
        self.bytes = self
            .bytes
            .checked_add(written as u64)
            .ok_or_else(|| std::io::Error::other("streamed RMSNorm body length overflow"))?;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

fn domain_hasher(domain: &str) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher
}

fn finish_digest(hasher: Sha256) -> Digest {
    Digest::from_sha256(hasher.finalize().into())
}

#[derive(Default)]
struct Registry {
    entries: BTreeMap<[u8; 32], Entry>,
    reservations: usize,
    active_claims: usize,
    admitted_bytes: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RmsRegistrySnapshot {
    entries: usize,
    reservations: usize,
    active_claims: usize,
    admitted_bytes: u64,
}

#[cfg(test)]
pub(crate) fn rms_registry_snapshot() -> Result<RmsRegistrySnapshot, String> {
    let registry = registry()?;
    Ok(RmsRegistrySnapshot {
        entries: registry.entries.len(),
        reservations: registry.reservations,
        active_claims: registry.active_claims,
        admitted_bytes: registry.admitted_bytes,
    })
}

struct Entry {
    ticket_digest: Digest,
    admitted_bytes: u64,
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();

struct Reservation {
    bytes: u64,
    active: bool,
}

struct PublishedIssuance {
    nonce: [u8; 32],
    digest: Digest,
    active: bool,
}

struct ActiveClaim {
    bytes: u64,
    active: bool,
}

impl Reservation {
    fn finalize(mut self, nonce: [u8; 32], digest: Digest) -> Result<PublishedIssuance, String> {
        let mut registry = registry()?;
        registry.reservations = registry
            .reservations
            .checked_sub(1)
            .ok_or("streamed RMSNorm reservation count underflowed")?;
        self.active = false;
        if registry.entries.contains_key(&nonce) {
            registry.admitted_bytes = registry
                .admitted_bytes
                .checked_sub(self.bytes)
                .ok_or("streamed RMSNorm admission accounting underflowed")?;
            return Err("streamed RMSNorm issuance nonce collision".into());
        }
        registry.entries.insert(
            nonce,
            Entry {
                ticket_digest: digest.clone(),
                admitted_bytes: self.bytes,
            },
        );
        Ok(PublishedIssuance {
            nonce,
            digest,
            active: true,
        })
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.active {
            let _ = release_reservation(self.bytes);
        }
    }
}

impl Drop for PublishedIssuance {
    fn drop(&mut self) {
        if self.active {
            let _ = unregister_issuance(self.nonce, &self.digest);
        }
    }
}

impl Drop for ActiveClaim {
    fn drop(&mut self) {
        if self.active {
            let _ = release_active_claim(self.bytes);
        }
    }
}

fn registry() -> Result<std::sync::MutexGuard<'static, Registry>, String> {
    REGISTRY
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "streamed RMSNorm issuance registry is poisoned".into())
}

fn reserve_capacity(bytes: u64) -> Result<Reservation, String> {
    let mut registry = registry()?;
    validate_registry_capacity(&registry, 1, bytes)?;
    registry.reservations += 1;
    registry.admitted_bytes = registry
        .admitted_bytes
        .checked_add(bytes)
        .ok_or("streamed RMSNorm admitted-byte count overflowed")?;
    Ok(Reservation {
        bytes,
        active: true,
    })
}

fn validate_registry_capacity(
    registry: &Registry,
    additional_issuances: usize,
    additional_bytes: u64,
) -> Result<(), String> {
    let count = registry
        .entries
        .len()
        .checked_add(registry.reservations)
        .and_then(|count| count.checked_add(registry.active_claims))
        .and_then(|count| count.checked_add(additional_issuances))
        .ok_or("streamed RMSNorm issuance count overflowed")?;
    let admitted_bytes = registry
        .admitted_bytes
        .checked_add(additional_bytes)
        .ok_or("streamed RMSNorm admitted-byte count overflowed")?;
    if count > MAX_ISSUANCES || admitted_bytes > MAX_ADMITTED_BYTES {
        return Err("streamed RMSNorm global admission capacity is exhausted".into());
    }
    Ok(())
}

fn release_reservation(bytes: u64) -> Result<(), String> {
    let mut registry = registry()?;
    registry.reservations = registry
        .reservations
        .checked_sub(1)
        .ok_or("streamed RMSNorm reservation count underflowed")?;
    registry.admitted_bytes = registry
        .admitted_bytes
        .checked_sub(bytes)
        .ok_or("streamed RMSNorm admission accounting underflowed")?;
    Ok(())
}

fn unregister_issuance(nonce: [u8; 32], digest: &Digest) -> Result<bool, String> {
    let mut registry = registry()?;
    if registry
        .entries
        .get(&nonce)
        .is_none_or(|entry| &entry.ticket_digest != digest)
    {
        return Ok(false);
    }
    let entry = registry
        .entries
        .remove(&nonce)
        .ok_or("streamed RMSNorm issuance disappeared")?;
    registry.admitted_bytes = registry
        .admitted_bytes
        .checked_sub(entry.admitted_bytes)
        .ok_or("streamed RMSNorm admission accounting underflowed")?;
    Ok(true)
}

fn consume_issuance(nonce: [u8; 32], digest: &Digest) -> Result<ActiveClaim, String> {
    let mut registry = registry()?;
    if registry
        .entries
        .get(&nonce)
        .is_none_or(|entry| &entry.ticket_digest != digest)
    {
        return Err("streamed RMSNorm material was not issued or was already claimed".into());
    }
    let entry = registry
        .entries
        .remove(&nonce)
        .ok_or("streamed RMSNorm issuance disappeared")?;
    registry.active_claims = registry
        .active_claims
        .checked_add(1)
        .ok_or("streamed RMSNorm active claim count overflowed")?;
    Ok(ActiveClaim {
        bytes: entry.admitted_bytes,
        active: true,
    })
}

fn release_active_claim(bytes: u64) -> Result<(), String> {
    let mut registry = registry()?;
    registry.active_claims = registry
        .active_claims
        .checked_sub(1)
        .ok_or("streamed RMSNorm active claim count underflowed")?;
    registry.admitted_bytes = registry
        .admitted_bytes
        .checked_sub(bytes)
        .ok_or("streamed RMSNorm admission accounting underflowed")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower_rms_norm_q10_direct_regions, DecoderMode};
    use pllm_models::{lower_model_json, DecoderWorkload};
    use std::cell::Cell;
    use std::io::Cursor;

    type Issued = (
        DecoderPlan,
        ModelRmsNormQ10DirectRegion,
        ExperimentalRmsNormQ10StreamPolicy,
        Vec<u8>,
        RmsNormQ10StreamTicket,
        BoundRmsNormQ10StreamInputs,
        BoundRmsNormQ10StreamDecoder,
    );

    fn qwen_plan(width: usize) -> DecoderPlan {
        lower_model_json(
            format!(
                r#"{{
                    "model_type":"qwen2","hidden_size":{width},"intermediate_size":4,
                    "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
                    "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
                    "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
                }}"#
            )
            .as_bytes(),
            DecoderWorkload {
                batch: 1,
                max_input_tokens: 1,
                max_new_tokens: 1,
            },
        )
        .unwrap()
    }

    fn issue(input: &[i16], weights: &[i16]) -> Issued {
        let plan = qwen_plan(weights.len());
        let region =
            lower_rms_norm_q10_direct_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
        let policy = ExperimentalRmsNormQ10StreamPolicy::acknowledge_unreviewed_public_weights(
            32 * 1024 * 1024,
        )
        .unwrap();
        let mut body = Vec::new();
        let material =
            prepare_bound_rms_norm_q10_stream_row(&plan, &region, 0, weights, &policy, &mut body)
                .unwrap();
        assert_eq!(material.ticket().body_bytes(), body.len() as u64);
        let (ticket, inputs, decoder) = material.encode(input).unwrap();
        (plan, region, policy, body, ticket, inputs, decoder)
    }

    fn duplicate_ticket(ticket: &RmsNormQ10StreamTicket) -> RmsNormQ10StreamTicket {
        RmsNormQ10StreamTicket {
            plan_digest: ticket.plan_digest.clone(),
            region_digest: ticket.region_digest.clone(),
            row_index: ticket.row_index,
            weights_digest: ticket.weights_digest.clone(),
            width: ticket.width,
            max_body_bytes: ticket.max_body_bytes,
            and_gate_count: ticket.and_gate_count,
            body_bytes: ticket.body_bytes,
            artifact_digest: ticket.artifact_digest.clone(),
            issuance_nonce: ticket.issuance_nonce,
            source_identity: ticket.source_identity,
            body_digest: ticket.body_digest.clone(),
            commitment_digest: ticket.commitment_digest.clone(),
            drop_commitment: ticket.drop_commitment,
            registered: ticket.registered,
        }
    }

    fn registry_state() -> (usize, usize, usize, u64) {
        let registry = registry().unwrap();
        (
            registry.entries.len(),
            registry.reservations,
            registry.active_claims,
            registry.admitted_bytes,
        )
    }

    #[test]
    fn exact_width_eight_qwen_row_matches_core_and_binds_unique_sources() {
        let _guard = RMS_REGISTRY_TEST_LOCK.lock().unwrap();
        let input = [1536_i16, -768, 256, -128, 1024, 2048, -1536, 512];
        let weights = [1024_i16, 896, 960, 1088, 1024, 1000, 1040, 992];
        let (plan, region, policy, body, ticket, inputs, decoder) = issue(&input, &weights);
        let expected = pllm_core::rms_norm_q10_direct(&input, &weights).unwrap();
        assert_eq!(ticket.body_bytes(), ticket.and_gate_count() * 32);
        let evaluator = BoundRmsNormQ10StreamEvaluator::claim(
            &plan, &region, 0, &weights, &policy, ticket, inputs,
        )
        .unwrap();
        let outputs = evaluator
            .evaluate_with_reader(|| Ok(Cursor::new(body)))
            .unwrap();
        assert_eq!(decoder.decode(outputs).unwrap(), expected);

        let (_, _, _, _, first, _, _) = issue(&input, &weights);
        let (_, _, _, _, second, _, _) = issue(&input, &weights);
        assert_ne!(first.source_identity, second.source_identity);
    }

    #[test]
    fn forged_ticket_does_not_burn_authentic_issuance_and_replay_fails() {
        let _guard = RMS_REGISTRY_TEST_LOCK.lock().unwrap();
        let weights = [1024_i16, 1024];
        let (plan, region, policy, _, ticket, inputs, _) = issue(&[17, 17], &weights);
        let mut forged = duplicate_ticket(&ticket);
        forged.row_index = 1;
        let forged_digest = ticket_digest(&forged);
        assert!(consume_issuance(forged.issuance_nonce, &forged_digest).is_err());
        drop(forged);

        let replay = duplicate_ticket(&ticket);
        let evaluator = BoundRmsNormQ10StreamEvaluator::claim(
            &plan, &region, 0, &weights, &policy, ticket, inputs,
        )
        .unwrap();
        assert!(consume_issuance(replay.issuance_nonce, &ticket_digest(&replay)).is_err());
        drop(evaluator);
        assert_eq!(registry_state(), (0, 0, 0, 0));
    }

    #[test]
    fn authentic_region_and_weight_mismatches_burn_issuance() {
        let _guard = RMS_REGISTRY_TEST_LOCK.lock().unwrap();
        let weights = [1024_i16, 1024];
        let (plan, region, policy, _, ticket, inputs, _) = issue(&[17, 17], &weights);
        let replay = duplicate_ticket(&ticket);
        let mut wrong_region = region.clone();
        wrong_region.maximum_encoded_error = 2;
        assert!(BoundRmsNormQ10StreamEvaluator::claim(
            &plan,
            &wrong_region,
            0,
            &weights,
            &policy,
            ticket,
            inputs,
        )
        .is_err());
        assert!(consume_issuance(replay.issuance_nonce, &ticket_digest(&replay)).is_err());

        let (plan, region, policy, _, ticket, inputs, _) = issue(&[17, 17], &weights);
        let replay = duplicate_ticket(&ticket);
        assert!(BoundRmsNormQ10StreamEvaluator::claim(
            &plan,
            &region,
            0,
            &[1023, 1024],
            &policy,
            ticket,
            inputs,
        )
        .is_err());
        assert!(consume_issuance(replay.issuance_nonce, &ticket_digest(&replay)).is_err());
        assert_eq!(registry_state(), (0, 0, 0, 0));
    }

    #[test]
    fn corruption_truncation_and_trailing_bytes_produce_no_outputs() {
        let _guard = RMS_REGISTRY_TEST_LOCK.lock().unwrap();
        let weights = [1024_i16, 1024];
        for mutation in 0..3 {
            let (plan, region, policy, mut body, ticket, inputs, _decoder) =
                issue(&[1536, 1536], &weights);
            match mutation {
                0 => body[0] ^= 0x80,
                1 => {
                    body.pop();
                }
                _ => body.push(0),
            }
            let evaluator = BoundRmsNormQ10StreamEvaluator::claim(
                &plan, &region, 0, &weights, &policy, ticket, inputs,
            )
            .unwrap();
            assert!(evaluator
                .evaluate_with_reader(|| Ok(Cursor::new(body)))
                .is_err());
        }
        assert_eq!(registry_state(), (0, 0, 0, 0));
    }

    struct InterruptingReader {
        body: Cursor<Vec<u8>>,
        interrupt_next: bool,
    }

    impl Read for InterruptingReader {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            if self.interrupt_next {
                self.interrupt_next = false;
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            self.interrupt_next = true;
            self.body.read(buffer)
        }
    }

    #[test]
    fn reader_callback_runs_after_burn_and_interrupted_reads_retry() {
        let _guard = RMS_REGISTRY_TEST_LOCK.lock().unwrap();
        let input = [1536_i16, 1536];
        let weights = [1024_i16, 1024];
        let (plan, region, policy, body, ticket, inputs, decoder) = issue(&input, &weights);
        let replay = duplicate_ticket(&ticket);
        let callback_ran = Cell::new(false);
        let evaluator = BoundRmsNormQ10StreamEvaluator::claim(
            &plan, &region, 0, &weights, &policy, ticket, inputs,
        )
        .unwrap();
        let outputs = evaluator
            .evaluate_with_reader(|| {
                callback_ran.set(true);
                assert!(consume_issuance(replay.issuance_nonce, &ticket_digest(&replay)).is_err());
                Ok(InterruptingReader {
                    body: Cursor::new(body),
                    interrupt_next: true,
                })
            })
            .unwrap();
        assert!(callback_ran.get());
        assert_eq!(
            decoder.decode(outputs).unwrap(),
            pllm_core::rms_norm_q10_direct(&input, &weights).unwrap()
        );
        assert_eq!(registry_state(), (0, 0, 0, 0));
    }

    struct FailingWriter {
        calls: usize,
    }

    impl Write for FailingWriter {
        fn write(&mut self, _buffer: &[u8]) -> std::io::Result<usize> {
            self.calls += 1;
            Err(std::io::Error::other("intentional publication failure"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn publication_failure_releases_reservation_and_issuance() {
        let _guard = RMS_REGISTRY_TEST_LOCK.lock().unwrap();
        let plan = qwen_plan(2);
        let region =
            lower_rms_norm_q10_direct_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
        let policy = ExperimentalRmsNormQ10StreamPolicy::acknowledge_unreviewed_public_weights(
            32 * 1024 * 1024,
        )
        .unwrap();
        let before = registry_state();
        let mut writer = FailingWriter { calls: 0 };
        let error = match prepare_bound_rms_norm_q10_stream_row(
            &plan,
            &region,
            0,
            &[1024, 1024],
            &policy,
            &mut writer,
        ) {
            Ok(_) => panic!("publication failure was accepted"),
            Err(error) => error,
        };
        assert!(error.contains("discard the writer"));
        assert_eq!(writer.calls, 1);
        assert_eq!(registry_state(), before);
    }

    struct PanicWriter;

    impl Write for PanicWriter {
        fn write(&mut self, _buffer: &[u8]) -> std::io::Result<usize> {
            panic!("writer must not be called before policy admission")
        }

        fn flush(&mut self) -> std::io::Result<()> {
            panic!("writer must not be flushed before policy admission")
        }
    }

    #[test]
    fn body_policy_preflight_precedes_writer_callbacks() {
        let _guard = RMS_REGISTRY_TEST_LOCK.lock().unwrap();
        let plan = qwen_plan(2);
        let region =
            lower_rms_norm_q10_direct_regions(&plan, DecoderMode::Decode).unwrap()[0].clone();
        let policy =
            ExperimentalRmsNormQ10StreamPolicy::acknowledge_unreviewed_public_weights(1).unwrap();
        let mut writer = PanicWriter;
        let error = match prepare_bound_rms_norm_q10_stream_row(
            &plan,
            &region,
            0,
            &[1024, 1024],
            &policy,
            &mut writer,
        ) {
            Ok(_) => panic!("undersized body policy was accepted"),
            Err(error) => error,
        };
        assert!(error.contains("exceeding policy limit"));
        assert_eq!(registry_state(), (0, 0, 0, 0));
    }
}
