use super::{
    digest_array, gated_multiply_q7_method, model_gated_multiply_q7_region_digest, tensor_elements,
    validate_model_gated_multiply_q7_region_against_plan, ModelGatedMultiplyQ7Region,
    GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
    GATED_MULTIPLY_Q7_ISSUANCE_CAPACITY,
};
use pllm_models::DecoderPlan;
use pllm_types::{canonical_bytes, canonical_digest, digest_bytes, Digest};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::io::{BufReader, Read, Seek, Write};
use std::sync::{Mutex, OnceLock};

const TENSOR_TICKET_SCHEMA_VERSION: &str = "pllm.gated_multiply_q7_tensor_ticket.v2";
const TENSOR_TICKET_DIGEST_DOMAIN: &str = "pllm.gated_multiply_q7_tensor_ticket.digest.v2";
const TENSOR_ORCHESTRATION_DIGEST_DOMAIN: &str = "pllm.gated_multiply_q7_tensor_orchestration.v1";
const TENSOR_ISSUANCE_ID_DOMAIN: &str = "pllm.gated_multiply_q7_tensor_issuance.v1";
const TENSOR_LANE_CONTEXT_DOMAIN: &str = "pllm.gated_multiply_q7_tensor_lane.v1";
const TENSOR_BODY_DIGEST_DOMAIN: &str = "pllm.gated_multiply_q7_tensor_body.v1";
const TENSOR_CHUNK_DIGEST_DOMAIN: &str = "pllm.gated_multiply_q7_tensor_chunk.v1";
const TENSOR_BODY_MAGIC: &[u8; 8] = b"PLLMGQT1";
const TENSOR_FRAME_HEADER_BYTES: usize = 8 + 8 + 4 + 8 + 32;
const TENSOR_MAX_READY_BYTES: u64 = 1_u64 << 40;
const TENSOR_MAX_LABEL_BYTES: usize = 1_024;
const TENSOR_LANE_ORDER: &str = "row_major_last_axis_fastest";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorResourcePolicy {
    pub max_elements: u64,
    pub max_chunk_elements: u32,
    pub max_chunk_bytes: u64,
    pub max_body_bytes: u64,
    pub max_output_bytes: u64,
    pub max_client_material_bytes: u64,
    pub max_working_bytes: u64,
}

impl TensorResourcePolicy {
    fn validate(&self) -> Result<(), String> {
        if self.max_elements == 0
            || self.max_chunk_elements == 0
            || self.max_chunk_bytes == 0
            || self.max_body_bytes < TENSOR_BODY_MAGIC.len() as u64
            || self.max_output_bytes == 0
            || self.max_client_material_bytes == 0
            || self.max_working_bytes == 0
        {
            return Err(
                "gated Q7 tensor resource policy contains a zero or impossible bound".into(),
            );
        }
        Ok(())
    }

    fn fits_within(&self, host: &Self) -> bool {
        self.max_elements <= host.max_elements
            && self.max_chunk_elements <= host.max_chunk_elements
            && self.max_chunk_bytes <= host.max_chunk_bytes
            && self.max_body_bytes <= host.max_body_bytes
            && self.max_output_bytes <= host.max_output_bytes
            && self.max_client_material_bytes <= host.max_client_material_bytes
            && self.max_working_bytes <= host.max_working_bytes
    }
}

/// Same-process opaque authorization ticket; only its streamed body is transportable.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GatedMultiplyQ7TensorTicket {
    schema_version: String,
    issuance_id: Digest,
    orchestration_digest: Digest,
    region_digest: Digest,
    method_component_id: String,
    schedule_component_id: String,
    tensor_shape: Vec<u64>,
    lane_order: String,
    element_count: u64,
    chunk_elements: u32,
    chunk_count: u64,
    evaluator_program_bytes: u64,
    largest_chunk_bytes: u64,
    body_bytes: u64,
    output_bytes: u64,
    client_material_bytes: u64,
    working_bytes: u64,
    body_digest: Digest,
    resource_policy: TensorResourcePolicy,
}

impl GatedMultiplyQ7TensorTicket {
    /// Exact finite frame length required by the evaluator body reader.
    pub fn body_bytes(&self) -> u64 {
        self.body_bytes
    }

    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }
}

#[derive(Serialize)]
struct TensorOrchestration<'a> {
    region_digest: &'a Digest,
    method_component_id: &'a str,
    schedule_component_id: &'a str,
    tensor_shape: &'a [u64],
    lane_order: &'static str,
    element_count: u64,
    chunk_elements: u32,
    resource_policy: &'a TensorResourcePolicy,
}

#[derive(Clone)]
struct TensorIssuance {
    ticket_digest: Digest,
    reserved_bytes: u64,
}

#[derive(Default)]
struct TensorIssuanceRegistry {
    entries: BTreeMap<[u8; 32], TensorIssuance>,
    reservations: usize,
    active_claims: usize,
    ready_bytes: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GatedTensorRegistrySnapshot {
    entries: usize,
    reservations: usize,
    active_claims: usize,
    ready_bytes: u64,
}

#[cfg(test)]
pub(crate) fn gated_tensor_registry_snapshot() -> Result<GatedTensorRegistrySnapshot, String> {
    let registry = tensor_registry()?;
    Ok(GatedTensorRegistrySnapshot {
        entries: registry.entries.len(),
        reservations: registry.reservations,
        active_claims: registry.active_claims,
        ready_bytes: registry.ready_bytes,
    })
}

#[cfg(test)]
thread_local! {
    static TEST_GATED_REGISTRY_SATURATED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) struct TestGatedTensorRegistrySaturation;

#[cfg(test)]
pub(crate) fn saturate_gated_tensor_registry_for_test(
) -> Result<TestGatedTensorRegistrySaturation, String> {
    let already_saturated = TEST_GATED_REGISTRY_SATURATED.with(|saturated| saturated.replace(true));
    if already_saturated {
        return Err("gated Q7 tensor test registry is already saturated".into());
    }
    Ok(TestGatedTensorRegistrySaturation)
}

#[cfg(test)]
impl Drop for TestGatedTensorRegistrySaturation {
    fn drop(&mut self) {
        TEST_GATED_REGISTRY_SATURATED.with(|saturated| saturated.set(false));
    }
}

static TENSOR_ISSUANCE_REGISTRY: OnceLock<Mutex<TensorIssuanceRegistry>> = OnceLock::new();

struct TensorReservation {
    reserved_bytes: u64,
    active: bool,
}

struct TensorActiveClaim {
    reserved_bytes: u64,
    active: bool,
}

struct PublishedTensorIssuance {
    issuance_id: [u8; 32],
    ticket_digest: Digest,
    active: bool,
}

impl Drop for PublishedTensorIssuance {
    fn drop(&mut self) {
        if self.active {
            let _ = unregister_tensor_issuance(self.issuance_id, &self.ticket_digest);
        }
    }
}

impl Drop for TensorActiveClaim {
    fn drop(&mut self) {
        if self.active {
            let _ = release_active_tensor_claim(self.reserved_bytes);
        }
    }
}

impl TensorReservation {
    fn finalize(
        mut self,
        issuance_id: [u8; 32],
        ticket_digest: Digest,
        issuance_bytes: u64,
    ) -> Result<(), String> {
        let mut registry = tensor_registry()?;
        registry.reservations = registry
            .reservations
            .checked_sub(1)
            .ok_or("gated Q7 tensor reservation count underflowed")?;
        self.active = false;
        if issuance_bytes > self.reserved_bytes {
            registry.ready_bytes = registry
                .ready_bytes
                .checked_sub(self.reserved_bytes)
                .ok_or("gated Q7 tensor reserved-byte accounting underflowed")?;
            return Err("gated Q7 tensor issuance exceeds its reservation".into());
        }
        if registry.entries.contains_key(&issuance_id) {
            registry.ready_bytes = registry
                .ready_bytes
                .checked_sub(self.reserved_bytes)
                .ok_or("gated Q7 tensor reserved-byte accounting underflowed")?;
            return Err("gated Q7 tensor issuance identifier collision".into());
        }
        registry.ready_bytes = registry
            .ready_bytes
            .checked_sub(self.reserved_bytes - issuance_bytes)
            .ok_or("gated Q7 tensor reserved-byte accounting underflowed")?;
        registry.entries.insert(
            issuance_id,
            TensorIssuance {
                ticket_digest,
                reserved_bytes: issuance_bytes,
            },
        );
        Ok(())
    }
}

impl Drop for TensorReservation {
    fn drop(&mut self) {
        if self.active {
            let _ = release_tensor_reservation(self.reserved_bytes);
        }
    }
}

pub struct BoundGatedMultiplyQ7TensorMaterial {
    materials: Vec<pllm_garble::GatedMultiplyQ7ClientMaterial>,
    ticket: GatedMultiplyQ7TensorTicket,
    issuance_id: [u8; 32],
    ticket_digest: Digest,
}

impl BoundGatedMultiplyQ7TensorMaterial {
    pub fn ticket(&self) -> &GatedMultiplyQ7TensorTicket {
        &self.ticket
    }

    pub fn element_count(&self) -> usize {
        self.materials.len()
    }

    pub fn encode_gates(&self, values: &[i16]) -> Result<Vec<Vec<u8>>, pllm_garble::GarbleError> {
        if values.len() != self.materials.len() {
            return Err(pllm_garble::GarbleError::InvalidProgram);
        }
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(values.len())
            .map_err(|_| pllm_garble::GarbleError::InvalidProgram)?;
        for (material, value) in self.materials.iter().zip(values) {
            encoded.push(material.encode_gate(*value)?);
        }
        Ok(encoded)
    }

    pub fn encode_ups(&self, values: &[i16]) -> Result<Vec<Vec<u8>>, pllm_garble::GarbleError> {
        if values.len() != self.materials.len() {
            return Err(pllm_garble::GarbleError::InvalidProgram);
        }
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(values.len())
            .map_err(|_| pllm_garble::GarbleError::InvalidProgram)?;
        for (material, value) in self.materials.iter().zip(values) {
            encoded.push(material.encode_up(*value)?);
        }
        Ok(encoded)
    }

    pub fn decode_tensor(&self, labels: &[Vec<u8>]) -> Result<Vec<i16>, pllm_garble::GarbleError> {
        if labels.len() != self.materials.len() {
            return Err(pllm_garble::GarbleError::InvalidProgram);
        }
        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(labels.len())
            .map_err(|_| pllm_garble::GarbleError::InvalidProgram)?;
        for (material, label) in self.materials.iter().zip(labels) {
            decoded.push(material.decode(label)?);
        }
        Ok(decoded)
    }

    pub fn cancel(&self) -> Result<bool, String> {
        unregister_tensor_issuance(self.issuance_id, &self.ticket_digest)
    }
}

impl Drop for BoundGatedMultiplyQ7TensorMaterial {
    fn drop(&mut self) {
        let _ = unregister_tensor_issuance(self.issuance_id, &self.ticket_digest);
    }
}

pub(crate) struct GatedTensorPreparationPreflight {
    elements: usize,
    chunk_elements: usize,
    element_count: u64,
    chunk_count: u64,
    region_digest: Digest,
    orchestration_digest: Digest,
    method: pllm_garble::GatedMultiplyQ7Method,
    first_material: pllm_garble::GatedMultiplyQ7Material,
    evaluator_program_bytes: u64,
    largest_chunk_bytes: u64,
    expected_body_bytes: u64,
    output_label_bytes: u64,
    output_bytes: u64,
    client_material_bytes: u64,
    working_bytes: u64,
}

pub(crate) fn preflight_bound_gated_multiply_q7_tensor_material(
    plan: &DecoderPlan,
    region: &ModelGatedMultiplyQ7Region,
    policy: &TensorResourcePolicy,
) -> Result<GatedTensorPreparationPreflight, String> {
    // The lane-zero probe garbles exact-size material but never registers issuance or touches the writer.
    validate_model_gated_multiply_q7_region_against_plan(plan, region)?;
    if region.schedule_component_id
        != GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID
    {
        return Err("gated Q7 tensor streaming requires the chunked scheduler component".into());
    }
    policy.validate()?;
    if region.input.shape.len() > 8 {
        return Err("gated Q7 tensor rank exceeds 8".into());
    }
    let elements = tensor_elements(&region.input.shape)?;
    if elements == 0
        || elements > region.max_tensor_elements
        || u64::try_from(elements).map_err(|_| "tensor element count exceeds u64")?
            > policy.max_elements
    {
        return Err("gated Q7 tensor element count exceeds its plan or resource bound".into());
    }
    let chunk_elements = usize::try_from(policy.max_chunk_elements)
        .map_err(|_| "chunk element bound exceeds usize")?
        .min(elements);
    let element_count = u64::try_from(elements).map_err(|_| "tensor element count exceeds u64")?;
    let chunk_count = element_count.div_ceil(chunk_elements as u64);
    let region_digest = model_gated_multiply_q7_region_digest(region);
    let orchestration_digest = canonical_digest(
        TENSOR_ORCHESTRATION_DIGEST_DOMAIN,
        &TensorOrchestration {
            region_digest: &region_digest,
            method_component_id: &region.method_component_id,
            schedule_component_id: &region.schedule_component_id,
            tensor_shape: &region.input.shape,
            lane_order: TENSOR_LANE_ORDER,
            element_count,
            chunk_elements: chunk_elements as u32,
            resource_policy: policy,
        },
    );
    let method = gated_multiply_q7_method(region)?;
    preflight_tensor_registry_count()?;
    let first_context = tensor_lane_context(&orchestration_digest, 0)?;
    // This unregistered probe becomes lane zero if preparation proceeds.
    let first_material =
        pllm_garble::prepare_gated_multiply_q7_with_method_and_context(method, first_context)
            .map_err(|error| error.to_string())?;
    let estimate = first_material
        .resource_estimate()
        .map_err(|error| error.to_string())?;
    let evaluator_program_bytes = u64::try_from(estimate.evaluator_program_bytes)
        .map_err(|_| "gated Q7 tensor program length exceeds u64")?;
    let entry_bytes = evaluator_program_bytes
        .checked_add(4)
        .ok_or("gated Q7 tensor program entry length overflows")?;
    let largest_chunk_bytes = u64::try_from(chunk_elements)
        .map_err(|_| "chunk element bound exceeds u64")?
        .checked_mul(entry_bytes)
        .ok_or("gated Q7 tensor chunk length overflows")?;
    let expected_body_bytes = (TENSOR_BODY_MAGIC.len() as u64)
        .checked_add(
            chunk_count
                .checked_mul(TENSOR_FRAME_HEADER_BYTES as u64)
                .ok_or("gated Q7 tensor frame length overflows")?,
        )
        .and_then(|bytes| bytes.checked_add(element_count.checked_mul(entry_bytes)?))
        .ok_or("gated Q7 tensor body length overflows")?;
    let working_bytes = largest_chunk_bytes
        .checked_mul(pllm_garble::PROGRAM_DECODE_WORKING_BYTES_PER_WIRE_BYTE)
        .ok_or("gated Q7 tensor working-memory estimate overflows")?;
    let client_material_bytes = u64::try_from(estimate.client_material_bytes)
        .map_err(|_| "gated Q7 tensor client material exceeds u64")?
        .checked_mul(element_count)
        .ok_or("gated Q7 tensor client-material size overflows")?;
    let output_label_bytes = u64::try_from(estimate.output_label_bytes)
        .map_err(|_| "gated Q7 tensor output label exceeds u64")?;
    let output_bytes = output_label_bytes
        .checked_mul(element_count)
        .ok_or("gated Q7 tensor output size overflows")?;
    if largest_chunk_bytes > policy.max_chunk_bytes
        || expected_body_bytes > policy.max_body_bytes
        || output_bytes > policy.max_output_bytes
        || working_bytes > policy.max_working_bytes
        || client_material_bytes > policy.max_client_material_bytes
    {
        return Err("gated Q7 tensor preparation exceeds its resource policy".into());
    }
    preflight_tensor_registry_capacity(expected_body_bytes)?;
    Ok(GatedTensorPreparationPreflight {
        elements,
        chunk_elements,
        element_count,
        chunk_count,
        region_digest,
        orchestration_digest,
        method,
        first_material,
        evaluator_program_bytes,
        largest_chunk_bytes,
        expected_body_bytes,
        output_label_bytes,
        output_bytes,
        client_material_bytes,
        working_bytes,
    })
}

/// Stream evaluator programs while retaining only compact client encodings in memory.
/// If publication fails, the caller must discard that writer or transport frame because it may
/// contain an unauthenticated prefix; the corresponding issuance is cancelled before return.
pub fn prepare_bound_gated_multiply_q7_tensor_material<W: Write>(
    plan: &DecoderPlan,
    region: &ModelGatedMultiplyQ7Region,
    policy: TensorResourcePolicy,
    writer: &mut W,
) -> Result<BoundGatedMultiplyQ7TensorMaterial, String> {
    let preflight = preflight_bound_gated_multiply_q7_tensor_material(plan, region, &policy)?;
    prepare_bound_gated_multiply_q7_tensor_material_with_preflight(
        region, policy, preflight, writer,
    )
}

pub(crate) fn prepare_bound_gated_multiply_q7_tensor_material_with_preflight<W: Write>(
    region: &ModelGatedMultiplyQ7Region,
    policy: TensorResourcePolicy,
    preflight: GatedTensorPreparationPreflight,
    writer: &mut W,
) -> Result<BoundGatedMultiplyQ7TensorMaterial, String> {
    let GatedTensorPreparationPreflight {
        elements,
        chunk_elements,
        element_count,
        chunk_count,
        region_digest,
        orchestration_digest,
        method,
        first_material,
        evaluator_program_bytes,
        largest_chunk_bytes,
        expected_body_bytes,
        output_label_bytes,
        output_bytes,
        client_material_bytes,
        working_bytes,
    } = preflight;
    // A race after preflight can still fail here without publishing or leaking the probe.
    let reservation = reserve_tensor_capacity(expected_body_bytes)?;
    let (first_client, first_program, first_material_id) = first_material.into_client_material();
    let program_bytes = u64::try_from(first_program.len())
        .map_err(|_| "gated Q7 tensor program length exceeds u64")?;
    let observed_client_material_bytes = u64::try_from(first_client.heap_bytes())
        .map_err(|_| "gated Q7 tensor client material exceeds u64")?
        .checked_mul(element_count)
        .ok_or("gated Q7 tensor client-material size overflows u64")?;
    if program_bytes != evaluator_program_bytes
        || observed_client_material_bytes != client_material_bytes
    {
        return Err("gated Q7 tensor exact preflight drifted before preparation".into());
    }
    let mut materials = Vec::new();
    materials
        .try_reserve_exact(elements)
        .map_err(|_| "gated Q7 tensor client-material allocation failed")?;
    let mut issuance_hasher = domain_hasher(TENSOR_ISSUANCE_ID_DOMAIN);
    issuance_hasher.update(digest_array(&orchestration_digest)?);
    issuance_hasher.update(element_count.to_le_bytes());
    let mut first_client = Some(first_client);
    let mut first_program = Some(first_program);
    let mut first_material_id = Some(first_material_id);
    let mut body_hasher = domain_hasher(TENSOR_BODY_DIGEST_DOMAIN);
    let mut body_bytes = 0_u64;
    let mut body_spool = tempfile::tempfile()
        .map_err(|error| format!("failed to create gated Q7 tensor preparation spool: {error}"))?;
    write_hashed(
        &mut body_spool,
        TENSOR_BODY_MAGIC,
        &mut body_hasher,
        &mut body_bytes,
    )?;

    let mut next_lane = 0_usize;
    for chunk_index in 0..chunk_count {
        let lane_count = chunk_elements.min(elements - next_lane);
        let mut chunk = Vec::new();
        for lane in next_lane..next_lane + lane_count {
            let (client, program, material_id) = if lane == 0 {
                (
                    first_client
                        .take()
                        .ok_or("gated Q7 tensor first client material disappeared")?,
                    first_program
                        .take()
                        .ok_or("gated Q7 tensor first program disappeared")?,
                    first_material_id
                        .take()
                        .ok_or("gated Q7 tensor first material ID disappeared")?,
                )
            } else {
                let context = tensor_lane_context(&orchestration_digest, lane)?;
                let prepared =
                    pllm_garble::prepare_gated_multiply_q7_with_method_and_context(method, context)
                        .map_err(|error| error.to_string())?;
                let lane_estimate = prepared
                    .resource_estimate()
                    .map_err(|error| error.to_string())?;
                if u64::try_from(lane_estimate.evaluator_program_bytes)
                    .map_err(|_| "gated Q7 tensor evaluator size exceeds u64")?
                    != evaluator_program_bytes
                    || u64::try_from(lane_estimate.client_material_bytes)
                        .map_err(|_| "gated Q7 tensor client material size exceeds u64")?
                        != client_material_bytes / element_count
                    || u64::try_from(lane_estimate.output_label_bytes)
                        .map_err(|_| "gated Q7 tensor output label size exceeds u64")?
                        != output_label_bytes
                {
                    return Err(
                        "gated Q7 tensor lane resource estimate changed during preparation".into(),
                    );
                }
                prepared.into_client_material()
            };
            if program.len() as u64 != program_bytes {
                return Err("gated Q7 tensor program lengths are not canonical".into());
            }
            let lane_client_bytes = u64::try_from(client.heap_bytes())
                .map_err(|_| "gated Q7 tensor client material exceeds u64")?;
            if lane_client_bytes
                .checked_mul(element_count)
                .is_none_or(|bytes| bytes != client_material_bytes)
            {
                return Err("gated Q7 tensor client-material lengths are not canonical".into());
            }
            let length =
                u32::try_from(program.len()).map_err(|_| "gated Q7 tensor program exceeds u32")?;
            let next_chunk_bytes = chunk
                .len()
                .checked_add(4)
                .and_then(|value| value.checked_add(program.len()))
                .ok_or("gated Q7 tensor chunk length overflows usize")?;
            if next_chunk_bytes as u64 > policy.max_chunk_bytes {
                return Err("gated Q7 tensor chunk exceeds its resource bound".into());
            }
            chunk
                .try_reserve_exact(4 + program.len())
                .map_err(|_| "gated Q7 tensor chunk allocation failed")?;
            chunk.extend_from_slice(&length.to_le_bytes());
            chunk.extend_from_slice(&program);
            materials.push(client);
            issuance_hasher.update(material_id);
        }
        let chunk_digest = digest_bytes(TENSOR_CHUNK_DIGEST_DOMAIN, &chunk);
        let first_lane = u64::try_from(next_lane).map_err(|_| "lane index exceeds u64")?;
        let lane_count_u32 = u32::try_from(lane_count).map_err(|_| "lane count exceeds u32")?;
        let chunk_bytes = u64::try_from(chunk.len()).map_err(|_| "chunk length exceeds u64")?;
        let mut frame = Vec::with_capacity(TENSOR_FRAME_HEADER_BYTES);
        frame.extend_from_slice(&chunk_index.to_le_bytes());
        frame.extend_from_slice(&first_lane.to_le_bytes());
        frame.extend_from_slice(&lane_count_u32.to_le_bytes());
        frame.extend_from_slice(&chunk_bytes.to_le_bytes());
        frame.extend_from_slice(&digest_array(&chunk_digest)?);
        let projected = body_bytes
            .checked_add(frame.len() as u64)
            .and_then(|value| value.checked_add(chunk_bytes))
            .ok_or("gated Q7 tensor body length overflows u64")?;
        if projected > policy.max_body_bytes {
            return Err("gated Q7 tensor body exceeds its resource bound".into());
        }
        write_hashed(&mut body_spool, &frame, &mut body_hasher, &mut body_bytes)?;
        write_hashed(&mut body_spool, &chunk, &mut body_hasher, &mut body_bytes)?;
        next_lane += lane_count;
    }
    if body_bytes != expected_body_bytes {
        return Err("gated Q7 tensor body length is not canonical".into());
    }
    let body_digest = finish_digest(body_hasher);
    let issuance_id = finish_digest(issuance_hasher);
    let ticket = GatedMultiplyQ7TensorTicket {
        schema_version: TENSOR_TICKET_SCHEMA_VERSION.into(),
        issuance_id: issuance_id.clone(),
        orchestration_digest,
        region_digest,
        method_component_id: region.method_component_id.clone(),
        schedule_component_id: region.schedule_component_id.clone(),
        tensor_shape: region.input.shape.clone(),
        lane_order: TENSOR_LANE_ORDER.into(),
        element_count,
        chunk_elements: chunk_elements as u32,
        chunk_count,
        evaluator_program_bytes,
        largest_chunk_bytes,
        body_bytes,
        output_bytes,
        client_material_bytes,
        working_bytes,
        body_digest,
        resource_policy: policy,
    };
    let ticket_digest = canonical_digest(TENSOR_TICKET_DIGEST_DOMAIN, &ticket);
    let issuance_id_bytes = digest_array(&issuance_id)?;
    body_spool
        .rewind()
        .map_err(|error| format!("failed to rewind gated Q7 tensor preparation spool: {error}"))?;
    validate_ticket_structure(&ticket)?;
    reservation.finalize(issuance_id_bytes, ticket_digest.clone(), body_bytes)?;
    let mut published = PublishedTensorIssuance {
        issuance_id: issuance_id_bytes,
        ticket_digest: ticket_digest.clone(),
        active: true,
    };
    if let Err(error) = std::io::copy(&mut body_spool, writer) {
        return Err(format!(
            "failed to publish gated Q7 tensor body; discard the writer: {error}"
        ));
    }
    if let Err(error) = writer.flush() {
        return Err(format!(
            "failed to flush gated Q7 tensor body; discard the writer: {error}"
        ));
    }
    published.active = false;
    Ok(BoundGatedMultiplyQ7TensorMaterial {
        materials,
        ticket,
        issuance_id: issuance_id_bytes,
        ticket_digest,
    })
}

pub struct GatedMultiplyQ7TensorEvaluator {
    ticket: GatedMultiplyQ7TensorTicket,
    active_claim: Option<TensorActiveClaim>,
    consumed: bool,
}

impl GatedMultiplyQ7TensorEvaluator {
    /// Authenticate and atomically burn whole-tensor issuance before any body reader is invoked.
    pub fn claim(
        region: &ModelGatedMultiplyQ7Region,
        ticket: &GatedMultiplyQ7TensorTicket,
        host_policy: &TensorResourcePolicy,
    ) -> Result<Self, String> {
        validate_ticket_structure(ticket)?;
        host_policy.validate()?;
        let ticket_digest = canonical_digest(TENSOR_TICKET_DIGEST_DOMAIN, ticket);
        let active_claim =
            consume_tensor_issuance(digest_array(&ticket.issuance_id)?, &ticket_digest)?;
        let expected_region = model_gated_multiply_q7_region_digest(region);
        if region.schedule_component_id
            != GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID
            || ticket.region_digest != expected_region
            || ticket.method_component_id != region.method_component_id
            || ticket.schedule_component_id != region.schedule_component_id
            || ticket.tensor_shape != region.input.shape
            || ticket.element_count
                != u64::try_from(tensor_elements(&region.input.shape)?)
                    .map_err(|_| "tensor element count exceeds u64")?
            || !ticket.resource_policy.fits_within(host_policy)
        {
            return Err("gated Q7 tensor ticket does not match region or host policy".into());
        }
        let expected_orchestration = canonical_digest(
            TENSOR_ORCHESTRATION_DIGEST_DOMAIN,
            &TensorOrchestration {
                region_digest: &ticket.region_digest,
                method_component_id: &ticket.method_component_id,
                schedule_component_id: &ticket.schedule_component_id,
                tensor_shape: &ticket.tensor_shape,
                lane_order: TENSOR_LANE_ORDER,
                element_count: ticket.element_count,
                chunk_elements: ticket.chunk_elements,
                resource_policy: &ticket.resource_policy,
            },
        );
        if ticket.orchestration_digest != expected_orchestration {
            return Err("gated Q7 tensor orchestration commitment is invalid".into());
        }
        Ok(Self {
            ticket: ticket.clone(),
            active_claim: Some(active_claim),
            consumed: false,
        })
    }

    /// Evaluate one exact finite body frame; no output is returned until commitment and EOF pass.
    /// Transport callers must bound the reader to [`GatedMultiplyQ7TensorTicket::body_bytes`].
    pub fn evaluate<R: Read>(
        &mut self,
        reader: &mut R,
        gate_labels: &[Vec<u8>],
        up_labels: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, String> {
        self.burn()?;
        let _active_claim = self
            .active_claim
            .take()
            .ok_or("gated Q7 tensor active claim disappeared")?;
        let elements = usize::try_from(self.ticket.element_count)
            .map_err(|_| "tensor element count exceeds usize")?;
        if gate_labels.len() != elements || up_labels.len() != elements {
            return Err("gated Q7 tensor label count does not match ticket".into());
        }
        if gate_labels
            .iter()
            .chain(up_labels)
            .any(|label| label.is_empty() || label.len() > TENSOR_MAX_LABEL_BYTES)
        {
            return Err("gated Q7 tensor input label exceeds its byte bound".into());
        }
        let mut spool = authenticate_tensor_body(reader, &self.ticket)?;
        spool
            .rewind()
            .map_err(|error| format!("failed to rewind gated Q7 tensor spool: {error}"))?;
        let mut reader = BufReader::new(spool);
        let mut magic = [0_u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|error| error.to_string())?;
        if &magic != TENSOR_BODY_MAGIC {
            return Err("gated Q7 tensor body magic is invalid".into());
        }
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(elements)
            .map_err(|_| "gated Q7 tensor output allocation failed")?;
        let mut output_bytes = 0_usize;
        let mut expected_lane = 0_u64;
        for expected_chunk in 0..self.ticket.chunk_count {
            let chunk_index = read_u64(&mut reader)?;
            let first_lane = read_u64(&mut reader)?;
            let lane_count = read_u32(&mut reader)?;
            let chunk_bytes = read_u64(&mut reader)?;
            let mut chunk_digest_bytes = [0_u8; 32];
            reader
                .read_exact(&mut chunk_digest_bytes)
                .map_err(|error| error.to_string())?;
            if chunk_index != expected_chunk
                || first_lane != expected_lane
                || lane_count == 0
                || lane_count > self.ticket.chunk_elements
                || chunk_bytes > self.ticket.resource_policy.max_chunk_bytes
                || chunk_bytes
                    .checked_mul(pllm_garble::PROGRAM_DECODE_WORKING_BYTES_PER_WIRE_BYTE)
                    .is_none_or(|bytes| bytes > self.ticket.resource_policy.max_working_bytes)
            {
                return Err("gated Q7 tensor chunk sequence is invalid".into());
            }
            let lane_end =
                checked_tensor_lane_end(first_lane, lane_count, self.ticket.element_count)?;
            let chunk_len = usize::try_from(chunk_bytes)
                .map_err(|_| "gated Q7 tensor chunk length exceeds usize")?;
            let mut chunk = Vec::new();
            chunk
                .try_reserve_exact(chunk_len)
                .map_err(|_| "gated Q7 tensor chunk allocation failed")?;
            chunk.resize(chunk_len, 0);
            reader
                .read_exact(&mut chunk)
                .map_err(|error| error.to_string())?;
            if digest_array(&digest_bytes(TENSOR_CHUNK_DIGEST_DOMAIN, &chunk))?
                != chunk_digest_bytes
            {
                return Err("gated Q7 tensor chunk digest is invalid".into());
            }
            let mut position = 0_usize;
            for offset in 0..lane_count {
                let length_end = position
                    .checked_add(4)
                    .ok_or("gated Q7 tensor program length overflows")?;
                let length = u32::from_le_bytes(
                    chunk
                        .get(position..length_end)
                        .ok_or("gated Q7 tensor program length is truncated")?
                        .try_into()
                        .map_err(|_| "gated Q7 tensor program length is truncated")?,
                ) as usize;
                position = length_end;
                let end = position
                    .checked_add(length)
                    .ok_or("gated Q7 tensor program length overflows")?;
                let program_bytes = chunk
                    .get(position..end)
                    .filter(|bytes| !bytes.is_empty())
                    .ok_or("gated Q7 tensor program is truncated")?;
                if u64::try_from(program_bytes.len())
                    .map_err(|_| "gated Q7 tensor program length exceeds u64")?
                    != self.ticket.evaluator_program_bytes
                {
                    return Err("gated Q7 tensor program length is not canonical".into());
                }
                let lane = first_lane
                    .checked_add(u64::from(offset))
                    .ok_or("gated Q7 tensor lane index overflows")?;
                let lane_usize = usize::try_from(lane)
                    .map_err(|_| "gated Q7 tensor lane index exceeds usize")?;
                let program = pllm_garble::GarbledProgram::from_bytes(program_bytes)
                    .map_err(|error| error.to_string())?;
                if program.context_digest()
                    != tensor_lane_context(&self.ticket.orchestration_digest, lane_usize)?
                    || program.input_moduli()
                        != [
                            pllm_garble::SILU_QUADRATIC_Q7_MODULUS,
                            pllm_garble::SILU_QUADRATIC_Q7_MODULUS,
                        ]
                {
                    return Err("gated Q7 tensor program authentication is invalid".into());
                }
                let output = program
                    .evaluate(&[&gate_labels[lane_usize], &up_labels[lane_usize]])
                    .map_err(|error| error.to_string())?;
                output_bytes = output_bytes
                    .checked_add(output.len())
                    .ok_or("gated Q7 tensor output length overflows")?;
                if output_bytes as u64 > self.ticket.resource_policy.max_output_bytes {
                    return Err("gated Q7 tensor output exceeds its resource bound".into());
                }
                outputs.push(output);
                position = end;
            }
            if position != chunk.len() {
                return Err("gated Q7 tensor chunk has trailing bytes".into());
            }
            expected_lane = lane_end;
        }
        if expected_lane != self.ticket.element_count
            || outputs.len() != elements
            || u64::try_from(output_bytes)
                .map_err(|_| "gated Q7 tensor output length exceeds u64")?
                != self.ticket.output_bytes
        {
            return Err("gated Q7 tensor body omits lanes".into());
        }
        ensure_reader_eof(&mut reader, "gated Q7 tensor body has trailing bytes")?;
        Ok(outputs)
    }

    pub fn burn(&mut self) -> Result<(), String> {
        if self.consumed {
            return Err("gated Q7 tensor evaluator material was already consumed".into());
        }
        self.consumed = true;
        Ok(())
    }
}

fn authenticate_tensor_body<R: Read>(
    reader: &mut R,
    ticket: &GatedMultiplyQ7TensorTicket,
) -> Result<std::fs::File, String> {
    let mut spool = tempfile::tempfile()
        .map_err(|error| format!("failed to create gated Q7 tensor spool: {error}"))?;
    let mut hasher = domain_hasher(TENSOR_BODY_DIGEST_DOMAIN);
    let mut remaining = ticket.body_bytes;
    let mut buffer = [0_u8; 64 * 1024];
    while remaining != 0 {
        let request = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "gated Q7 tensor body read exceeds usize")?;
        let read = match reader.read(&mut buffer[..request]) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.to_string()),
        };
        if read == 0 {
            return Err("gated Q7 tensor body is truncated".into());
        }
        spool
            .write_all(&buffer[..read])
            .map_err(|error| format!("failed to write gated Q7 tensor spool: {error}"))?;
        hasher.update(&buffer[..read]);
        remaining = remaining
            .checked_sub(u64::try_from(read).map_err(|_| "body read exceeds u64")?)
            .ok_or("gated Q7 tensor body read accounting underflowed")?;
    }
    ensure_reader_eof(reader, "gated Q7 tensor body has trailing bytes")?;
    if finish_digest(hasher) != ticket.body_digest {
        return Err("gated Q7 tensor body commitment is invalid".into());
    }
    Ok(spool)
}

fn validate_ticket_structure(ticket: &GatedMultiplyQ7TensorTicket) -> Result<(), String> {
    ticket.resource_policy.validate()?;
    let entry_bytes = ticket
        .evaluator_program_bytes
        .checked_add(4)
        .ok_or("gated Q7 tensor ticket entry length overflows")?;
    let expected_largest_chunk = u64::from(ticket.chunk_elements)
        .checked_mul(entry_bytes)
        .ok_or("gated Q7 tensor ticket chunk length overflows")?;
    let expected_body = (TENSOR_BODY_MAGIC.len() as u64)
        .checked_add(
            ticket
                .chunk_count
                .checked_mul(TENSOR_FRAME_HEADER_BYTES as u64)
                .ok_or("gated Q7 tensor ticket frame length overflows")?,
        )
        .and_then(|bytes| bytes.checked_add(ticket.element_count.checked_mul(entry_bytes)?))
        .ok_or("gated Q7 tensor ticket body length overflows")?;
    let expected_working = expected_largest_chunk
        .checked_mul(pllm_garble::PROGRAM_DECODE_WORKING_BYTES_PER_WIRE_BYTE)
        .ok_or("gated Q7 tensor ticket working bound overflows")?;
    if ticket.schema_version != TENSOR_TICKET_SCHEMA_VERSION
        || canonical_bytes(ticket).len() > 16_384
        || ticket.schedule_component_id
            != GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID
        || ticket.tensor_shape.is_empty()
        || ticket.tensor_shape.len() > 8
        || ticket.tensor_shape.contains(&0)
        || ticket.method_component_id.len() > 128
        || ticket.schedule_component_id.len() > 128
        || ticket.lane_order.len() > 128
        || ticket.lane_order != TENSOR_LANE_ORDER
        || ticket.element_count == 0
        || ticket.element_count > ticket.resource_policy.max_elements
        || ticket.chunk_elements == 0
        || ticket.chunk_elements > ticket.resource_policy.max_chunk_elements
        || ticket.chunk_count
            != ticket
                .element_count
                .div_ceil(u64::from(ticket.chunk_elements))
        || ticket.evaluator_program_bytes == 0
        || ticket.largest_chunk_bytes != expected_largest_chunk
        || ticket.body_bytes != expected_body
        || ticket.body_bytes > ticket.resource_policy.max_body_bytes
        || ticket.body_bytes < TENSOR_BODY_MAGIC.len() as u64
        || ticket.output_bytes == 0
        || ticket.output_bytes > ticket.resource_policy.max_output_bytes
        || ticket.client_material_bytes == 0
        || ticket.client_material_bytes > ticket.resource_policy.max_client_material_bytes
        || ticket.working_bytes != expected_working
        || ticket.working_bytes > ticket.resource_policy.max_working_bytes
    {
        return Err("gated Q7 tensor ticket is invalid".into());
    }
    Ok(())
}

fn tensor_registry() -> Result<std::sync::MutexGuard<'static, TensorIssuanceRegistry>, String> {
    TENSOR_ISSUANCE_REGISTRY
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "gated Q7 tensor issuance registry is poisoned".into())
}

fn preflight_tensor_registry_count() -> Result<(), String> {
    #[cfg(test)]
    if TEST_GATED_REGISTRY_SATURATED.with(std::cell::Cell::get) {
        return Err("gated Q7 tensor issuance capacity is exhausted".into());
    }
    let registry = tensor_registry()?;
    validate_tensor_registry_capacity(&registry, 1, 0)
}

pub(crate) fn preflight_tensor_registry_capacity(additional_bytes: u64) -> Result<(), String> {
    let registry = tensor_registry()?;
    validate_tensor_registry_capacity(&registry, 1, additional_bytes)
}

fn validate_tensor_registry_capacity(
    registry: &TensorIssuanceRegistry,
    additional_issuances: usize,
    additional_bytes: u64,
) -> Result<(), String> {
    let count = registry
        .entries
        .len()
        .checked_add(registry.reservations)
        .and_then(|count| count.checked_add(registry.active_claims))
        .and_then(|count| count.checked_add(additional_issuances))
        .ok_or("gated Q7 tensor issuance count overflows")?;
    let ready_bytes = registry
        .ready_bytes
        .checked_add(additional_bytes)
        .ok_or("gated Q7 tensor reserved-byte count overflows")?;
    if count > GATED_MULTIPLY_Q7_ISSUANCE_CAPACITY || ready_bytes > TENSOR_MAX_READY_BYTES {
        return Err("gated Q7 tensor issuance capacity is exhausted".into());
    }
    Ok(())
}

fn reserve_tensor_capacity(reserved_bytes: u64) -> Result<TensorReservation, String> {
    let mut registry = tensor_registry()?;
    validate_tensor_registry_capacity(&registry, 1, reserved_bytes)?;
    let next_bytes = registry.ready_bytes + reserved_bytes;
    registry.reservations = registry
        .reservations
        .checked_add(1)
        .ok_or("gated Q7 tensor reservation count overflows")?;
    registry.ready_bytes = next_bytes;
    Ok(TensorReservation {
        reserved_bytes,
        active: true,
    })
}

fn release_tensor_reservation(reserved_bytes: u64) -> Result<(), String> {
    let mut registry = tensor_registry()?;
    registry.reservations = registry
        .reservations
        .checked_sub(1)
        .ok_or("gated Q7 tensor reservation count underflowed")?;
    registry.ready_bytes = registry
        .ready_bytes
        .checked_sub(reserved_bytes)
        .ok_or("gated Q7 tensor reserved-byte accounting underflowed")?;
    Ok(())
}

fn unregister_tensor_issuance(
    issuance_id: [u8; 32],
    ticket_digest: &Digest,
) -> Result<bool, String> {
    let mut registry = tensor_registry()?;
    if registry
        .entries
        .get(&issuance_id)
        .is_none_or(|entry| &entry.ticket_digest != ticket_digest)
    {
        return Ok(false);
    }
    let entry = registry
        .entries
        .remove(&issuance_id)
        .ok_or("gated Q7 tensor issuance disappeared")?;
    registry.ready_bytes = registry
        .ready_bytes
        .checked_sub(entry.reserved_bytes)
        .ok_or("gated Q7 tensor reserved-byte accounting underflowed")?;
    Ok(true)
}

fn consume_tensor_issuance(
    issuance_id: [u8; 32],
    ticket_digest: &Digest,
) -> Result<TensorActiveClaim, String> {
    let mut registry = tensor_registry()?;
    if registry
        .entries
        .get(&issuance_id)
        .is_none_or(|entry| &entry.ticket_digest != ticket_digest)
    {
        return Err("gated Q7 tensor material was not issued or was already claimed".into());
    }
    let entry = registry
        .entries
        .remove(&issuance_id)
        .ok_or("gated Q7 tensor issuance disappeared")?;
    registry.active_claims = registry
        .active_claims
        .checked_add(1)
        .ok_or("gated Q7 tensor active-claim count overflows")?;
    Ok(TensorActiveClaim {
        reserved_bytes: entry.reserved_bytes,
        active: true,
    })
}

fn release_active_tensor_claim(reserved_bytes: u64) -> Result<(), String> {
    let mut registry = tensor_registry()?;
    registry.active_claims = registry
        .active_claims
        .checked_sub(1)
        .ok_or("gated Q7 tensor active-claim count underflowed")?;
    registry.ready_bytes = registry
        .ready_bytes
        .checked_sub(reserved_bytes)
        .ok_or("gated Q7 tensor reserved-byte accounting underflowed")?;
    Ok(())
}

fn tensor_lane_context(orchestration_digest: &Digest, lane: usize) -> Result<[u8; 32], String> {
    digest_array(&canonical_digest(
        TENSOR_LANE_CONTEXT_DOMAIN,
        &(orchestration_digest, lane),
    ))
}

fn checked_tensor_lane_end(
    first_lane: u64,
    lane_count: u32,
    element_count: u64,
) -> Result<u64, String> {
    first_lane
        .checked_add(u64::from(lane_count))
        .filter(|end| *end <= element_count)
        .ok_or_else(|| "gated Q7 tensor chunk exceeds the committed lane count".into())
}

fn domain_hasher(domain: &str) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher
}

fn finish_digest(hasher: Sha256) -> Digest {
    let bytes: [u8; 32] = hasher.finalize().into();
    Digest::from_sha256(bytes)
}

fn ensure_reader_eof<R: Read>(reader: &mut R, trailing_error: &str) -> Result<(), String> {
    let mut trailing = [0_u8; 1];
    loop {
        match reader.read(&mut trailing) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(trailing_error.into()),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.to_string()),
        }
    }
}

fn write_hashed<W: Write>(
    writer: &mut W,
    bytes: &[u8],
    hasher: &mut Sha256,
    count: &mut u64,
) -> Result<(), String> {
    writer.write_all(bytes).map_err(|error| error.to_string())?;
    hasher.update(bytes);
    *count = count
        .checked_add(u64::try_from(bytes.len()).map_err(|_| "body write exceeds u64")?)
        .ok_or("body byte count overflows u64")?;
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, String> {
    let mut bytes = [0_u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, String> {
    let mut bytes = [0_u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::checked_tensor_lane_end;

    #[test]
    fn chunk_extent_rejects_overflow_and_out_of_bounds() {
        assert_eq!(checked_tensor_lane_end(3, 2, 5).unwrap(), 5);
        assert!(checked_tensor_lane_end(u64::MAX, 1, u64::MAX).is_err());
        assert!(checked_tensor_lane_end(4, 2, 5).is_err());
    }
}
