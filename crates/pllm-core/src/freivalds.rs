use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes256,
};
use sha2::{Digest as _, Sha256};
use std::{
    collections::HashSet,
    fmt,
    sync::{Mutex, OnceLock},
};
use subtle::ConstantTimeEq;
use zeroize::{Zeroize, Zeroizing};

pub const FREIVALDS_FIELD_MODULUS: u32 = 4_294_967_291;
pub const FREIVALDS_MAX_CHECKS: usize = 8;
pub const FREIVALDS_MAX_BINDING_BYTES: usize = 4_096;
pub const FREIVALDS_MAX_SOUNDNESS_BITS: u32 = 128;
pub const FREIVALDS_MAX_PROCESS_SESSIONS: usize = 65_536;
const BINDING_DOMAIN: &[u8] = b"pllm.freivalds.binding.v1";
const AUTHENTICATION_DOMAIN: &[u8] = b"pllm.freivalds.projection_authentication.v1";
const CHALLENGE_DOMAIN: &[u8] = b"pllm.freivalds.challenge.v1";
const CHALLENGE_BLOCK_DOMAIN: &[u8] = b"pllm.freivalds.challenge_block.v1";
const CONSERVATIVE_BITS_PER_CHECK: u32 = 31;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FreivaldsError {
    Allocation,
    Binding,
    Consumed,
    Dimension,
    FieldElement,
    Integrity,
    Policy,
    Random,
    Range,
}

impl fmt::Display for FreivaldsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Allocation => "Freivalds allocation failed",
            Self::Binding => "Freivalds material binding does not match",
            Self::Consumed => "Freivalds verifier was already consumed",
            Self::Dimension => "Freivalds dimensions are invalid",
            Self::FieldElement => "Freivalds projection contains a noncanonical field element",
            Self::Integrity => "Freivalds verification failed",
            Self::Policy => "Freivalds resource or soundness policy is invalid",
            Self::Random => "Freivalds challenge derivation exceeded its rejection budget",
            Self::Range => "Freivalds signed output is outside its declared bound",
        })
    }
}

impl std::error::Error for FreivaldsError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FreivaldsResourcePolicy {
    max_matrix_elements: usize,
    max_projection_elements: usize,
    max_challenge_elements: usize,
    max_output_elements: usize,
    max_multiply_accumulates: u64,
    max_batch_rows: usize,
}

impl FreivaldsResourcePolicy {
    pub fn new(
        max_matrix_elements: usize,
        max_projection_elements: usize,
        max_challenge_elements: usize,
        max_output_elements: usize,
        max_multiply_accumulates: u64,
        max_batch_rows: usize,
    ) -> Result<Self, FreivaldsError> {
        if max_matrix_elements == 0
            || max_projection_elements == 0
            || max_challenge_elements == 0
            || max_output_elements == 0
            || max_multiply_accumulates == 0
            || max_batch_rows == 0
        {
            return Err(FreivaldsError::Policy);
        }
        Ok(Self {
            max_matrix_elements,
            max_projection_elements,
            max_challenge_elements,
            max_output_elements,
            max_multiply_accumulates,
            max_batch_rows,
        })
    }

    pub const fn max_matrix_elements(&self) -> usize {
        self.max_matrix_elements
    }

    pub const fn max_projection_elements(&self) -> usize {
        self.max_projection_elements
    }

    pub const fn max_batch_rows(&self) -> usize {
        self.max_batch_rows
    }

    pub const fn max_challenge_elements(&self) -> usize {
        self.max_challenge_elements
    }

    pub const fn max_output_elements(&self) -> usize {
        self.max_output_elements
    }

    pub const fn max_multiply_accumulates(&self) -> u64 {
        self.max_multiply_accumulates
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FreivaldsPolicy {
    checks: usize,
    target_failure_bits: u32,
    max_attempts: u64,
}

impl FreivaldsPolicy {
    pub fn sized_for_attempts(
        target_failure_bits: u32,
        max_attempts: u64,
    ) -> Result<Self, FreivaldsError> {
        if target_failure_bits == 0 || max_attempts == 0 {
            return Err(FreivaldsError::Policy);
        }
        let attempt_bits = 64 - (max_attempts - 1).leading_zeros();
        let required_bits = target_failure_bits
            .checked_add(attempt_bits)
            .ok_or(FreivaldsError::Policy)?;
        if required_bits > FREIVALDS_MAX_SOUNDNESS_BITS {
            return Err(FreivaldsError::Policy);
        }
        let checks = required_bits.div_ceil(CONSERVATIVE_BITS_PER_CHECK) as usize;
        if checks == 0 || checks > FREIVALDS_MAX_CHECKS {
            return Err(FreivaldsError::Policy);
        }
        Ok(Self {
            checks,
            target_failure_bits,
            max_attempts,
        })
    }

    pub const fn checks(&self) -> usize {
        self.checks
    }

    pub const fn conservative_failure_bits(&self) -> u32 {
        self.checks as u32 * CONSERVATIVE_BITS_PER_CHECK
    }

    pub const fn target_failure_bits(&self) -> u32 {
        self.target_failure_bits
    }

    pub const fn max_attempts(&self) -> u64 {
        self.max_attempts
    }
}

static REGISTERED_SESSION_IDS: OnceLock<Mutex<HashSet<[u8; 32]>>> = OnceLock::new();

pub struct FreivaldsSession {
    id: [u8; 32],
    policy: FreivaldsPolicy,
    remaining_attempts: u64,
    material_ids: HashSet<[u8; 32]>,
}

impl FreivaldsSession {
    pub fn register(id: [u8; 32], policy: FreivaldsPolicy) -> Result<Self, FreivaldsError> {
        if id == [0_u8; 32] {
            return Err(FreivaldsError::Binding);
        }
        let sessions = REGISTERED_SESSION_IDS.get_or_init(|| Mutex::new(HashSet::new()));
        let mut sessions = sessions.lock().map_err(|_| FreivaldsError::Consumed)?;
        if sessions.len() >= FREIVALDS_MAX_PROCESS_SESSIONS || !sessions.insert(id) {
            return Err(FreivaldsError::Consumed);
        }
        Ok(Self {
            id,
            policy,
            remaining_attempts: policy.max_attempts,
            material_ids: HashSet::new(),
        })
    }

    pub const fn id(&self) -> &[u8; 32] {
        &self.id
    }

    pub const fn policy(&self) -> FreivaldsPolicy {
        self.policy
    }

    pub const fn remaining_attempts(&self) -> u64 {
        self.remaining_attempts
    }

    fn register_material(&mut self, id: [u8; 32]) -> Result<(), FreivaldsError> {
        if self.material_ids.len() as u64 >= self.policy.max_attempts
            || !self.material_ids.insert(id)
        {
            return Err(FreivaldsError::Consumed);
        }
        Ok(())
    }

    fn claim(&mut self, rows: usize) -> Result<(), FreivaldsError> {
        let rows = u64::try_from(rows).map_err(|_| FreivaldsError::Dimension)?;
        self.remaining_attempts = self
            .remaining_attempts
            .checked_sub(rows)
            .ok_or(FreivaldsError::Consumed)?;
        Ok(())
    }
}

impl Drop for FreivaldsSession {
    fn drop(&mut self) {
        if let Some(registry) = REGISTERED_SESSION_IDS.get() {
            if let Ok(mut registry) = registry.lock() {
                registry.remove(&self.id);
            }
        }
    }
}

pub struct FreivaldsProjectionBatch {
    inventory_rows: usize,
    in_features: usize,
    out_features: usize,
    policy: FreivaldsPolicy,
    session_id: [u8; 32],
    material_id: [u8; 32],
    binding_digest: [u8; 32],
    signed_input_bound: i64,
    signed_output_bound: i64,
    max_row_l1: u64,
    claimable: bool,
    values: Zeroizing<Vec<u32>>,
    consumed: Vec<bool>,
}

impl FreivaldsProjectionBatch {
    pub const fn inventory_rows(&self) -> usize {
        self.inventory_rows
    }

    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    pub const fn checks(&self) -> usize {
        self.policy.checks
    }

    pub const fn signed_input_bound(&self) -> i64 {
        self.signed_input_bound
    }

    pub const fn signed_output_bound(&self) -> i64 {
        self.signed_output_bound
    }

    pub const fn material_id(&self) -> &[u8; 32] {
        &self.material_id
    }

    pub const fn binding_digest(&self) -> &[u8; 32] {
        &self.binding_digest
    }

    pub const fn max_row_l1(&self) -> u64 {
        self.max_row_l1
    }

    pub fn encoded_len(&self) -> Result<usize, FreivaldsError> {
        self.values
            .len()
            .checked_mul(4)
            .ok_or(FreivaldsError::Dimension)
    }

    pub fn encode(&self) -> Result<Zeroizing<Vec<u8>>, FreivaldsError> {
        if self.consumed.iter().any(|consumed| *consumed) {
            return Err(FreivaldsError::Consumed);
        }
        let mut encoded = Zeroizing::new(allocate(self.encoded_len()?)?);
        for (chunk, value) in encoded.chunks_exact_mut(4).zip(self.values.iter()) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        Ok(encoded)
    }

    pub fn authentication_tag(&self, root_seed: &[u8; 32]) -> Result<[u8; 16], FreivaldsError> {
        let encoded = self.encode()?;
        let context = challenge_context_from_binding_digest(
            &self.binding_digest,
            &self.session_id,
            &self.material_id,
            self.policy,
            self.signed_input_bound,
            self.signed_output_bound,
            self.max_row_l1,
            self.in_features,
            self.out_features,
        )?;
        projection_authentication_tag(root_seed, &context, &encoded)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn claim_verifier(
        &mut self,
        root_seed: &[u8; 32],
        binding: &[u8],
        row_start: usize,
        batch_rows: usize,
        session: &mut FreivaldsSession,
        resources: FreivaldsResourcePolicy,
    ) -> Result<FreivaldsVerifier, FreivaldsError> {
        if !self.claimable {
            return Err(FreivaldsError::Policy);
        }
        if session.id != self.session_id || session.policy != self.policy {
            return Err(FreivaldsError::Binding);
        }
        if binding_digest(binding)? != self.binding_digest {
            return Err(FreivaldsError::Binding);
        }
        validate_selection(
            self.inventory_rows,
            row_start,
            batch_rows,
            self.in_features,
            self.out_features,
            self.policy.checks,
            self.signed_output_bound,
            resources,
        )?;
        let row_width = self
            .policy
            .checks
            .checked_mul(self.in_features)
            .ok_or(FreivaldsError::Dimension)?;
        let start = row_start
            .checked_mul(row_width)
            .ok_or(FreivaldsError::Dimension)?;
        let length = batch_rows
            .checked_mul(row_width)
            .ok_or(FreivaldsError::Dimension)?;
        let end = start
            .checked_add(length)
            .filter(|end| *end <= self.values.len())
            .ok_or(FreivaldsError::Dimension)?;
        if self.consumed[row_start..row_start + batch_rows]
            .iter()
            .any(|consumed| *consumed)
        {
            return Err(FreivaldsError::Consumed);
        }
        session.claim(batch_rows)?;
        self.consumed[row_start..row_start + batch_rows].fill(true);
        let mut projections = match allocate(length) {
            Ok(values) => Zeroizing::new(values),
            Err(error) => {
                self.values[start..end].zeroize();
                return Err(error);
            }
        };
        projections.copy_from_slice(&self.values[start..end]);
        self.values[start..end].zeroize();
        FreivaldsVerifier::new(
            root_seed,
            binding,
            &self.session_id,
            &self.material_id,
            row_start,
            batch_rows,
            self.in_features,
            self.out_features,
            self.policy,
            self.signed_input_bound,
            self.signed_output_bound,
            self.max_row_l1,
            projections,
            resources,
        )
    }
}

impl fmt::Debug for FreivaldsProjectionBatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FreivaldsProjectionBatch")
            .field("inventory_rows", &self.inventory_rows)
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("checks", &self.policy.checks)
            .finish_non_exhaustive()
    }
}

pub struct FreivaldsVerifier {
    batch_rows: usize,
    in_features: usize,
    out_features: usize,
    checks: usize,
    signed_input_bound: i64,
    signed_output_bound: i64,
    max_row_l1: u64,
    challenges: Zeroizing<Vec<u32>>,
    projections: Zeroizing<Vec<u32>>,
    consumed: bool,
}

impl FreivaldsVerifier {
    #[allow(clippy::too_many_arguments)]
    fn new(
        root_seed: &[u8; 32],
        binding: &[u8],
        session_id: &[u8; 32],
        material_id: &[u8; 32],
        row_start: usize,
        batch_rows: usize,
        in_features: usize,
        out_features: usize,
        policy: FreivaldsPolicy,
        signed_input_bound: i64,
        signed_output_bound: i64,
        max_row_l1: u64,
        projections: Zeroizing<Vec<u32>>,
        resources: FreivaldsResourcePolicy,
    ) -> Result<Self, FreivaldsError> {
        validate_selection(
            row_start
                .checked_add(batch_rows)
                .ok_or(FreivaldsError::Dimension)?,
            row_start,
            batch_rows,
            in_features,
            out_features,
            policy.checks,
            signed_output_bound,
            resources,
        )?;
        let challenge_len = batch_rows
            .checked_mul(policy.checks)
            .and_then(|value| value.checked_mul(out_features))
            .ok_or(FreivaldsError::Dimension)?;
        let mut challenges = Zeroizing::new(allocate(challenge_len)?);
        derive_challenges(
            root_seed,
            binding,
            session_id,
            material_id,
            row_start,
            batch_rows,
            policy,
            signed_input_bound,
            signed_output_bound,
            max_row_l1,
            in_features,
            out_features,
            &mut challenges,
        )?;
        Ok(Self {
            batch_rows,
            in_features,
            out_features,
            checks: policy.checks,
            signed_input_bound,
            signed_output_bound,
            max_row_l1,
            challenges,
            projections,
            consumed: false,
        })
    }

    pub const fn batch_rows(&self) -> usize {
        self.batch_rows
    }

    pub const fn shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    pub const fn checks(&self) -> usize {
        self.checks
    }

    pub fn verify(
        &mut self,
        input: &[i32],
        output: &[i64],
    ) -> Result<VerifiedAccumulatorBatch, FreivaldsError> {
        if self.consumed {
            return Err(FreivaldsError::Consumed);
        }
        self.consumed = true;
        let result = self.verify_claimed(input, output);
        self.erase();
        result
    }

    fn verify_claimed(
        &self,
        input: &[i32],
        output: &[i64],
    ) -> Result<VerifiedAccumulatorBatch, FreivaldsError> {
        let expected_input = self
            .batch_rows
            .checked_mul(self.in_features)
            .ok_or(FreivaldsError::Dimension)?;
        let expected_output = self
            .batch_rows
            .checked_mul(self.out_features)
            .ok_or(FreivaldsError::Dimension)?;
        if input.len() != expected_input || output.len() != expected_output {
            return Err(FreivaldsError::Dimension);
        }
        let input_max = input
            .iter()
            .map(|value| u64::from(value.unsigned_abs()))
            .max()
            .unwrap_or(0);
        if input_max > self.signed_input_bound as u64
            || u128::from(input_max) * u128::from(self.max_row_l1)
                > self.signed_output_bound as u128
        {
            return Err(FreivaldsError::Range);
        }
        let mut verified = Zeroizing::new(allocate(expected_output)?);
        verified.copy_from_slice(output);
        let mut integrity = subtle::Choice::from(1);
        for (row, (input_row, output_row)) in input
            .chunks_exact(self.in_features)
            .zip(output.chunks_exact(self.out_features))
            .enumerate()
        {
            if output_row
                .iter()
                .any(|value| value.unsigned_abs() > self.signed_output_bound as u64)
            {
                verified.zeroize();
                return Err(FreivaldsError::Range);
            }
            for check in 0..self.checks {
                let challenge_offset = (row * self.checks + check) * self.out_features;
                let projection_offset = (row * self.checks + check) * self.in_features;
                let left = dot_signed_field(
                    output_row,
                    &self.challenges[challenge_offset..challenge_offset + self.out_features],
                );
                let right = dot_signed_field(
                    input_row,
                    &self.projections[projection_offset..projection_offset + self.in_features],
                );
                integrity &= left.ct_eq(&right);
            }
        }
        if !bool::from(integrity) {
            verified.zeroize();
            return Err(FreivaldsError::Integrity);
        }
        Ok(VerifiedAccumulatorBatch {
            rows: self.batch_rows,
            cols: self.out_features,
            values: verified,
        })
    }

    fn erase(&mut self) {
        self.challenges.zeroize();
        self.projections.zeroize();
    }
}

impl fmt::Debug for FreivaldsVerifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FreivaldsVerifier")
            .field("batch_rows", &self.batch_rows)
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("checks", &self.checks)
            .field("state", &if self.consumed { "consumed" } else { "ready" })
            .finish_non_exhaustive()
    }
}

pub struct VerifiedAccumulatorBatch {
    rows: usize,
    cols: usize,
    values: Zeroizing<Vec<i64>>,
}

#[allow(clippy::too_many_arguments)]
pub fn import_freivalds_projections(
    encoded: &[u8],
    authentication_tag: &[u8; 16],
    root_seed: &[u8; 32],
    session: &mut FreivaldsSession,
    inventory_rows: usize,
    out_features: usize,
    in_features: usize,
    binding: &[u8],
    material_id: [u8; 32],
    signed_input_bound: i64,
    signed_output_bound: i64,
    max_row_l1: u64,
    policy: FreivaldsPolicy,
    resources: FreivaldsResourcePolicy,
) -> Result<FreivaldsProjectionBatch, FreivaldsError> {
    if session.policy != policy {
        return Err(FreivaldsError::Binding);
    }
    let expected_elements = inventory_rows
        .checked_mul(policy.checks)
        .and_then(|value| value.checked_mul(in_features))
        .ok_or(FreivaldsError::Dimension)?;
    let expected_bytes = expected_elements
        .checked_mul(4)
        .ok_or(FreivaldsError::Dimension)?;
    if encoded.len() != expected_bytes {
        return Err(FreivaldsError::Dimension);
    }
    validate_import(
        inventory_rows,
        out_features,
        in_features,
        signed_input_bound,
        signed_output_bound,
        max_row_l1,
        policy,
        resources,
    )?;
    let digest = binding_digest(binding)?;
    let context = challenge_context_from_binding_digest(
        &digest,
        session.id(),
        &material_id,
        policy,
        signed_input_bound,
        signed_output_bound,
        max_row_l1,
        in_features,
        out_features,
    )?;
    let expected_tag = Zeroizing::new(projection_authentication_tag(root_seed, &context, encoded)?);
    if !bool::from(expected_tag.ct_eq(authentication_tag)) {
        return Err(FreivaldsError::Integrity);
    }
    session.register_material(material_id)?;
    let mut values = Zeroizing::new(allocate(expected_elements)?);
    for (value, chunk) in values.iter_mut().zip(encoded.chunks_exact(4)) {
        let decoded = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if decoded >= FREIVALDS_FIELD_MODULUS {
            return Err(FreivaldsError::FieldElement);
        }
        *value = decoded;
    }
    Ok(FreivaldsProjectionBatch {
        inventory_rows,
        in_features,
        out_features,
        policy,
        session_id: *session.id(),
        material_id,
        binding_digest: digest,
        signed_input_bound,
        signed_output_bound,
        max_row_l1,
        claimable: true,
        consumed: allocate(inventory_rows)?,
        values,
    })
}

impl VerifiedAccumulatorBatch {
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn as_slice(&self) -> &[i64] {
        &self.values
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_freivalds_projections(
    weights: &[i8],
    out_features: usize,
    in_features: usize,
    inventory_rows: usize,
    root_seed: &[u8; 32],
    binding: &[u8],
    session_id: [u8; 32],
    material_id: [u8; 32],
    signed_input_bound: i64,
    signed_output_bound: i64,
    policy: FreivaldsPolicy,
    resources: FreivaldsResourcePolicy,
) -> Result<FreivaldsProjectionBatch, FreivaldsError> {
    let digest = binding_digest(binding)?;
    validate_preparation(
        weights,
        out_features,
        in_features,
        inventory_rows,
        signed_input_bound,
        signed_output_bound,
        policy,
        resources,
    )?;
    let max_row_l1 = weights
        .chunks_exact(in_features)
        .map(|row| {
            row.iter()
                .map(|weight| u64::from(weight.unsigned_abs()))
                .sum()
        })
        .max()
        .ok_or(FreivaldsError::Dimension)?;
    let projection_len = inventory_rows
        .checked_mul(policy.checks)
        .and_then(|value| value.checked_mul(in_features))
        .ok_or(FreivaldsError::Dimension)?;
    let challenge_len = policy
        .checks
        .checked_mul(out_features)
        .ok_or(FreivaldsError::Dimension)?;
    let mut values = Zeroizing::new(allocate(projection_len)?);
    let mut challenges = Zeroizing::new(allocate(challenge_len)?);
    for row in 0..inventory_rows {
        derive_challenges(
            root_seed,
            binding,
            &session_id,
            &material_id,
            row,
            1,
            policy,
            signed_input_bound,
            signed_output_bound,
            max_row_l1,
            in_features,
            out_features,
            &mut challenges,
        )?;
        for check in 0..policy.checks {
            let challenge = &challenges[check * out_features..(check + 1) * out_features];
            let output = &mut values[(row * policy.checks + check) * in_features
                ..(row * policy.checks + check + 1) * in_features];
            for input_index in 0..in_features {
                let mut accumulator = 0_u128;
                for output_index in 0..out_features {
                    let weight = weights[output_index * in_features + input_index];
                    accumulator += u128::from(signed_to_field(i64::from(weight)))
                        * u128::from(challenge[output_index]);
                }
                output[input_index] = reduce(accumulator);
            }
        }
        challenges.fill(0);
    }
    let consumed = allocate(inventory_rows)?;
    Ok(FreivaldsProjectionBatch {
        inventory_rows,
        in_features,
        out_features,
        policy,
        session_id,
        material_id,
        binding_digest: digest,
        signed_input_bound,
        signed_output_bound,
        max_row_l1,
        claimable: false,
        values,
        consumed,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_import(
    inventory_rows: usize,
    out_features: usize,
    in_features: usize,
    signed_input_bound: i64,
    signed_output_bound: i64,
    max_row_l1: u64,
    policy: FreivaldsPolicy,
    resources: FreivaldsResourcePolicy,
) -> Result<(), FreivaldsError> {
    let matrix_elements = out_features
        .checked_mul(in_features)
        .ok_or(FreivaldsError::Dimension)?;
    let projection_elements = inventory_rows
        .checked_mul(policy.checks)
        .and_then(|value| value.checked_mul(in_features))
        .ok_or(FreivaldsError::Dimension)?;
    let projection_work = u64::try_from(inventory_rows)
        .ok()
        .and_then(|value| value.checked_mul(policy.checks as u64))
        .and_then(|value| value.checked_mul(matrix_elements as u64))
        .ok_or(FreivaldsError::Dimension)?;
    if out_features == 0
        || in_features == 0
        || inventory_rows == 0
        || inventory_rows as u64 > policy.max_attempts
        || matrix_elements > resources.max_matrix_elements
        || projection_elements > resources.max_projection_elements
        || projection_work > resources.max_multiply_accumulates
        || inventory_rows > resources.max_batch_rows
    {
        return Err(FreivaldsError::Policy);
    }
    validate_declared_bound(signed_input_bound, signed_output_bound, max_row_l1)
}

#[allow(clippy::too_many_arguments)]
fn validate_preparation(
    weights: &[i8],
    out_features: usize,
    in_features: usize,
    inventory_rows: usize,
    signed_input_bound: i64,
    signed_output_bound: i64,
    policy: FreivaldsPolicy,
    resources: FreivaldsResourcePolicy,
) -> Result<(), FreivaldsError> {
    if out_features == 0
        || in_features == 0
        || inventory_rows == 0
        || signed_input_bound <= 0
        || signed_output_bound <= 0
        || signed_output_bound >= i64::from(FREIVALDS_FIELD_MODULUS) / 2
    {
        return Err(FreivaldsError::Dimension);
    }
    let matrix_elements = out_features
        .checked_mul(in_features)
        .ok_or(FreivaldsError::Dimension)?;
    if weights.len() != matrix_elements {
        return Err(FreivaldsError::Dimension);
    }
    let projection_elements = inventory_rows
        .checked_mul(policy.checks)
        .and_then(|value| value.checked_mul(in_features))
        .ok_or(FreivaldsError::Dimension)?;
    let challenge_elements = policy
        .checks
        .checked_mul(out_features)
        .ok_or(FreivaldsError::Dimension)?;
    let output_elements = inventory_rows
        .checked_mul(out_features)
        .ok_or(FreivaldsError::Dimension)?;
    let multiply_accumulates = u64::try_from(inventory_rows)
        .ok()
        .and_then(|rows| rows.checked_mul(policy.checks as u64))
        .and_then(|value| value.checked_mul(matrix_elements as u64))
        .ok_or(FreivaldsError::Dimension)?;
    if matrix_elements > resources.max_matrix_elements
        || projection_elements > resources.max_projection_elements
        || challenge_elements > resources.max_challenge_elements
        || output_elements > resources.max_output_elements
        || multiply_accumulates > resources.max_multiply_accumulates
        || inventory_rows > resources.max_batch_rows
    {
        return Err(FreivaldsError::Policy);
    }
    let max_row_l1 = weights
        .chunks_exact(in_features)
        .try_fold(0_u64, |maximum, row| {
            let row_l1 = row.iter().try_fold(0_u64, |sum, weight| {
                sum.checked_add(u64::from(weight.unsigned_abs()))
                    .ok_or(FreivaldsError::Dimension)
            })?;
            Ok::<u64, FreivaldsError>(maximum.max(row_l1))
        })?;
    validate_declared_bound(signed_input_bound, signed_output_bound, max_row_l1)
}

fn validate_declared_bound(
    signed_input_bound: i64,
    signed_output_bound: i64,
    max_row_l1: u64,
) -> Result<(), FreivaldsError> {
    if signed_input_bound <= 0
        || signed_output_bound <= 0
        || signed_output_bound >= i64::from(FREIVALDS_FIELD_MODULUS) / 2
    {
        return Err(FreivaldsError::Range);
    }
    if u128::from(signed_input_bound as u64) * u128::from(max_row_l1) > signed_output_bound as u128
    {
        return Err(FreivaldsError::Range);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_selection(
    inventory_rows: usize,
    row_start: usize,
    batch_rows: usize,
    in_features: usize,
    out_features: usize,
    checks: usize,
    signed_output_bound: i64,
    resources: FreivaldsResourcePolicy,
) -> Result<(), FreivaldsError> {
    let row_end = row_start
        .checked_add(batch_rows)
        .filter(|end| *end <= inventory_rows)
        .ok_or(FreivaldsError::Dimension)?;
    if row_end == row_start
        || in_features == 0
        || out_features == 0
        || checks == 0
        || checks > FREIVALDS_MAX_CHECKS
        || batch_rows > resources.max_batch_rows
        || signed_output_bound <= 0
        || signed_output_bound >= i64::from(FREIVALDS_FIELD_MODULUS) / 2
    {
        return Err(FreivaldsError::Policy);
    }
    let matrix_elements = out_features
        .checked_mul(in_features)
        .ok_or(FreivaldsError::Dimension)?;
    let projection_elements = batch_rows
        .checked_mul(checks)
        .and_then(|value| value.checked_mul(in_features))
        .ok_or(FreivaldsError::Dimension)?;
    let challenge_elements = batch_rows
        .checked_mul(checks)
        .and_then(|value| value.checked_mul(out_features))
        .ok_or(FreivaldsError::Dimension)?;
    let output_elements = batch_rows
        .checked_mul(out_features)
        .ok_or(FreivaldsError::Dimension)?;
    let verification_accumulates = u64::try_from(batch_rows)
        .ok()
        .and_then(|rows| rows.checked_mul(checks as u64))
        .and_then(|value| {
            in_features
                .checked_add(out_features)
                .and_then(|width| u64::try_from(width).ok())
                .and_then(|width| value.checked_mul(width))
        })
        .ok_or(FreivaldsError::Dimension)?;
    if matrix_elements > resources.max_matrix_elements
        || projection_elements > resources.max_projection_elements
        || challenge_elements > resources.max_challenge_elements
        || output_elements > resources.max_output_elements
        || verification_accumulates > resources.max_multiply_accumulates
    {
        return Err(FreivaldsError::Policy);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn derive_challenges(
    root_seed: &[u8; 32],
    binding: &[u8],
    session_id: &[u8; 32],
    material_id: &[u8; 32],
    row_start: usize,
    rows: usize,
    policy: FreivaldsPolicy,
    signed_input_bound: i64,
    signed_output_bound: i64,
    max_row_l1: u64,
    in_features: usize,
    out_features: usize,
    output: &mut [u32],
) -> Result<(), FreivaldsError> {
    let expected = rows
        .checked_mul(policy.checks)
        .and_then(|value| value.checked_mul(out_features))
        .ok_or(FreivaldsError::Dimension)?;
    if output.len() != expected {
        return Err(FreivaldsError::Dimension);
    }
    let context = challenge_context_digest(
        binding,
        session_id,
        material_id,
        policy,
        signed_input_bound,
        signed_output_bound,
        max_row_l1,
        in_features,
        out_features,
    )?;
    let cipher = Aes256::new_from_slice(root_seed).map_err(|_| FreivaldsError::Random)?;
    for row in 0..rows {
        for check in 0..policy.checks {
            let absolute_row = row_start
                .checked_add(row)
                .ok_or(FreivaldsError::Dimension)?;
            let mut hasher = Sha256::new();
            sha2::Digest::update(&mut hasher, CHALLENGE_DOMAIN);
            sha2::Digest::update(&mut hasher, context);
            sha2::Digest::update(
                &mut hasher,
                u64::try_from(absolute_row)
                    .map_err(|_| FreivaldsError::Dimension)?
                    .to_le_bytes(),
            );
            sha2::Digest::update(
                &mut hasher,
                u32::try_from(check)
                    .map_err(|_| FreivaldsError::Dimension)?
                    .to_le_bytes(),
            );
            let stream_identity: [u8; 32] = hasher.finalize().into();
            let target = &mut output[(row * policy.checks + check) * out_features
                ..(row * policy.checks + check + 1) * out_features];
            let max_blocks = target.len().div_ceil(8).saturating_add(64);
            let mut filled = 0;
            for counter in 0..max_blocks {
                let counter = u32::try_from(counter).map_err(|_| FreivaldsError::Dimension)?;
                let mut block_hasher = Sha256::new();
                sha2::Digest::update(&mut block_hasher, CHALLENGE_BLOCK_DOMAIN);
                sha2::Digest::update(&mut block_hasher, stream_identity);
                sha2::Digest::update(&mut block_hasher, counter.to_le_bytes());
                let block_identity = block_hasher.finalize();
                let mut left = aes::cipher::Block::<Aes256>::default();
                let mut right = aes::cipher::Block::<Aes256>::default();
                left.copy_from_slice(&block_identity[..16]);
                right.copy_from_slice(&block_identity[16..]);
                cipher.encrypt_block(&mut left);
                cipher.encrypt_block(&mut right);
                for chunk in left.chunks_exact(4).chain(right.chunks_exact(4)) {
                    let candidate = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if candidate < FREIVALDS_FIELD_MODULUS {
                        target[filled] = candidate;
                        filled += 1;
                        if filled == target.len() {
                            break;
                        }
                    }
                }
                if filled == target.len() {
                    left.as_mut_slice().zeroize();
                    right.as_mut_slice().zeroize();
                    break;
                }
                left.as_mut_slice().zeroize();
                right.as_mut_slice().zeroize();
            }
            if filled != target.len() {
                return Err(FreivaldsError::Random);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn challenge_context_digest(
    binding: &[u8],
    session_id: &[u8; 32],
    material_id: &[u8; 32],
    policy: FreivaldsPolicy,
    signed_input_bound: i64,
    signed_output_bound: i64,
    max_row_l1: u64,
    in_features: usize,
    out_features: usize,
) -> Result<[u8; 32], FreivaldsError> {
    let mut hasher = Sha256::new();
    sha2::Digest::update(&mut hasher, CHALLENGE_DOMAIN);
    sha2::Digest::update(&mut hasher, binding_digest(binding)?);
    sha2::Digest::update(&mut hasher, session_id);
    sha2::Digest::update(&mut hasher, material_id);
    sha2::Digest::update(&mut hasher, policy.target_failure_bits.to_le_bytes());
    sha2::Digest::update(&mut hasher, policy.max_attempts.to_le_bytes());
    sha2::Digest::update(
        &mut hasher,
        u32::try_from(policy.checks)
            .map_err(|_| FreivaldsError::Dimension)?
            .to_le_bytes(),
    );
    sha2::Digest::update(&mut hasher, signed_input_bound.to_le_bytes());
    sha2::Digest::update(&mut hasher, signed_output_bound.to_le_bytes());
    sha2::Digest::update(&mut hasher, max_row_l1.to_le_bytes());
    sha2::Digest::update(
        &mut hasher,
        u64::try_from(in_features)
            .map_err(|_| FreivaldsError::Dimension)?
            .to_le_bytes(),
    );
    sha2::Digest::update(
        &mut hasher,
        u64::try_from(out_features)
            .map_err(|_| FreivaldsError::Dimension)?
            .to_le_bytes(),
    );
    Ok(hasher.finalize().into())
}

#[allow(clippy::too_many_arguments)]
fn challenge_context_from_binding_digest(
    binding: &[u8; 32],
    session_id: &[u8; 32],
    material_id: &[u8; 32],
    policy: FreivaldsPolicy,
    signed_input_bound: i64,
    signed_output_bound: i64,
    max_row_l1: u64,
    in_features: usize,
    out_features: usize,
) -> Result<[u8; 32], FreivaldsError> {
    let mut hasher = Sha256::new();
    sha2::Digest::update(&mut hasher, CHALLENGE_DOMAIN);
    sha2::Digest::update(&mut hasher, binding);
    sha2::Digest::update(&mut hasher, session_id);
    sha2::Digest::update(&mut hasher, material_id);
    sha2::Digest::update(&mut hasher, policy.target_failure_bits.to_le_bytes());
    sha2::Digest::update(&mut hasher, policy.max_attempts.to_le_bytes());
    sha2::Digest::update(
        &mut hasher,
        u32::try_from(policy.checks)
            .map_err(|_| FreivaldsError::Dimension)?
            .to_le_bytes(),
    );
    sha2::Digest::update(&mut hasher, signed_input_bound.to_le_bytes());
    sha2::Digest::update(&mut hasher, signed_output_bound.to_le_bytes());
    sha2::Digest::update(&mut hasher, max_row_l1.to_le_bytes());
    sha2::Digest::update(
        &mut hasher,
        u64::try_from(in_features)
            .map_err(|_| FreivaldsError::Dimension)?
            .to_le_bytes(),
    );
    sha2::Digest::update(
        &mut hasher,
        u64::try_from(out_features)
            .map_err(|_| FreivaldsError::Dimension)?
            .to_le_bytes(),
    );
    Ok(hasher.finalize().into())
}

fn binding_digest(binding: &[u8]) -> Result<[u8; 32], FreivaldsError> {
    if binding.is_empty() || binding.len() > FREIVALDS_MAX_BINDING_BYTES {
        return Err(FreivaldsError::Binding);
    }
    let mut hasher = Sha256::new();
    sha2::Digest::update(&mut hasher, BINDING_DOMAIN);
    sha2::Digest::update(
        &mut hasher,
        u64::try_from(binding.len())
            .map_err(|_| FreivaldsError::Binding)?
            .to_le_bytes(),
    );
    sha2::Digest::update(&mut hasher, binding);
    Ok(hasher.finalize().into())
}

fn projection_authentication_tag(
    root_seed: &[u8; 32],
    context: &[u8; 32],
    encoded: &[u8],
) -> Result<[u8; 16], FreivaldsError> {
    let message_len = AUTHENTICATION_DOMAIN
        .len()
        .checked_add(context.len())
        .and_then(|value| value.checked_add(encoded.len()))
        .ok_or(FreivaldsError::Dimension)?;
    let mut message = Zeroizing::new(allocate(message_len)?);
    let context_start = AUTHENTICATION_DOMAIN.len();
    let encoded_start = context_start + context.len();
    message[..context_start].copy_from_slice(AUTHENTICATION_DOMAIN);
    message[context_start..encoded_start].copy_from_slice(context);
    message[encoded_start..].copy_from_slice(encoded);
    aes_cmac(root_seed, &message)
}

fn double_cmac_subkey(input: &[u8; 16]) -> [u8; 16] {
    let mut output = [0_u8; 16];
    let mut carry = 0_u8;
    for index in (0..16).rev() {
        output[index] = (input[index] << 1) | carry;
        carry = input[index] >> 7;
    }
    if input[0] & 0x80 != 0 {
        output[15] ^= 0x87;
    }
    output
}

fn aes_cmac(root_seed: &[u8; 32], message: &[u8]) -> Result<[u8; 16], FreivaldsError> {
    let cipher = Aes256::new_from_slice(root_seed).map_err(|_| FreivaldsError::Random)?;
    let mut block = aes::cipher::Block::<Aes256>::default();
    cipher.encrypt_block(&mut block);
    let mut first = Zeroizing::new([0_u8; 16]);
    first.copy_from_slice(&block);
    let first_subkey = Zeroizing::new(double_cmac_subkey(&first));
    first.zeroize();
    let second = Zeroizing::new(double_cmac_subkey(&first_subkey));
    block.as_mut_slice().zeroize();

    let block_count = message.len().div_ceil(16).max(1);
    let mut state = Zeroizing::new([0_u8; 16]);
    for source in message.chunks_exact(16).take(block_count.saturating_sub(1)) {
        for index in 0..16 {
            block[index] = state[index] ^ source[index];
        }
        cipher.encrypt_block(&mut block);
        state.copy_from_slice(&block);
        block.as_mut_slice().zeroize();
    }
    finish_cmac(
        &cipher,
        message,
        block_count,
        &first_subkey,
        &second,
        &mut state,
        &mut block,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_cmac(
    cipher: &Aes256,
    message: &[u8],
    block_count: usize,
    first: &[u8; 16],
    second: &[u8; 16],
    state: &mut [u8; 16],
    block: &mut aes::cipher::Block<Aes256>,
) -> Result<[u8; 16], FreivaldsError> {
    let complete = !message.is_empty() && message.len() % 16 == 0;
    let mut last = Zeroizing::new([0_u8; 16]);
    if complete {
        last.copy_from_slice(&message[(block_count - 1) * 16..block_count * 16]);
        for index in 0..16 {
            last[index] ^= first[index];
        }
    } else {
        let remainder = message.len() % 16;
        if remainder != 0 {
            last[..remainder].copy_from_slice(&message[message.len() - remainder..]);
        }
        last[remainder] = 0x80;
        for index in 0..16 {
            last[index] ^= second[index];
        }
    }
    finish_cmac_block(cipher, state, block, &last)
}

fn finish_cmac_block(
    cipher: &Aes256,
    state: &[u8; 16],
    block: &mut aes::cipher::Block<Aes256>,
    last: &[u8; 16],
) -> Result<[u8; 16], FreivaldsError> {
    for index in 0..16 {
        block[index] = state[index] ^ last[index];
    }
    cipher.encrypt_block(block);
    let mut tag = [0_u8; 16];
    tag.copy_from_slice(block);
    block.as_mut_slice().zeroize();
    Ok(tag)
}

fn dot_signed_field<T: Copy + Into<i128>>(signed: &[T], field: &[u32]) -> u32 {
    let mut accumulator = 0_u128;
    for (signed_value, field_value) in signed.iter().zip(field) {
        let value = signed_to_field_i128((*signed_value).into());
        accumulator += u128::from(value) * u128::from(*field_value);
    }
    reduce(accumulator)
}

fn signed_to_field(value: i64) -> u32 {
    signed_to_field_i128(i128::from(value))
}

fn signed_to_field_i128(value: i128) -> u32 {
    value.rem_euclid(i128::from(FREIVALDS_FIELD_MODULUS)) as u32
}

fn reduce(value: u128) -> u32 {
    (value % u128::from(FREIVALDS_FIELD_MODULUS)) as u32
}

fn allocate<T: Default + Clone>(length: usize) -> Result<Vec<T>, FreivaldsError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| FreivaldsError::Allocation)?;
    values.resize(length, T::default());
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resources() -> FreivaldsResourcePolicy {
        FreivaldsResourcePolicy::new(1_000, 1_000, 1_000, 1_000, 1_000_000, 8).unwrap()
    }

    fn policy() -> FreivaldsPolicy {
        FreivaldsPolicy::sized_for_attempts(40, 1_000).unwrap()
    }

    fn binding() -> &'static [u8] {
        b"model/body/stage/weight/shape/q7"
    }

    fn session(seed: [u8; 32]) -> FreivaldsSession {
        FreivaldsSession::register(seed, policy()).unwrap()
    }

    fn imported(
        weights: &[i8],
        out_features: usize,
        in_features: usize,
        rows: usize,
        seed: [u8; 32],
    ) -> (FreivaldsProjectionBatch, FreivaldsSession) {
        let prepared = prepared(weights, out_features, in_features, rows, &seed);
        let payload = prepared.encode().unwrap();
        let tag = prepared.authentication_tag(&seed).unwrap();
        let mut session = session(seed);
        let imported = import_freivalds_projections(
            &payload,
            &tag,
            &seed,
            &mut session,
            rows,
            out_features,
            in_features,
            binding(),
            [11_u8; 32],
            10,
            1_000,
            prepared.max_row_l1(),
            policy(),
            resources(),
        )
        .unwrap();
        (imported, session)
    }

    fn prepared(
        weights: &[i8],
        out_features: usize,
        in_features: usize,
        inventory_rows: usize,
        seed: &[u8; 32],
    ) -> FreivaldsProjectionBatch {
        prepare_freivalds_projections(
            weights,
            out_features,
            in_features,
            inventory_rows,
            seed,
            binding(),
            *seed,
            [11_u8; 32],
            10,
            1_000,
            policy(),
            resources(),
        )
        .unwrap()
    }

    #[test]
    fn verifies_non_square_signed_runtime_matrix_for_multiple_rows() {
        let weights = [1_i8, -2, 3, 4, -5, 6];
        let seed = [7_u8; 32];
        let (mut projections, mut session) = imported(&weights, 2, 3, 3, seed);
        let mut verifier = projections
            .claim_verifier(&seed, binding(), 1, 2, &mut session, resources())
            .unwrap();
        let input = [2, -3, 5, -1, 4, 2];
        let output = [23, 53, -3, -12];
        let verified = verifier.verify(&input, &output).unwrap();
        assert_eq!(verified.shape(), (2, 2));
        assert_eq!(verified.as_slice(), output);
        assert!(matches!(
            verifier.verify(&input, &output),
            Err(FreivaldsError::Consumed)
        ));
    }

    #[test]
    fn corruption_and_range_failures_burn_material() {
        let weights = [1_i8, 2, 3, 4];
        let seed = [9_u8; 32];
        let (mut projections, mut session) = imported(&weights, 2, 2, 1, seed);
        let mut corrupted = projections
            .claim_verifier(&seed, binding(), 0, 1, &mut session, resources())
            .unwrap();
        assert!(matches!(
            corrupted.verify(&[5, 6], &[17, 40]),
            Err(FreivaldsError::Integrity)
        ));
        assert!(matches!(
            corrupted.verify(&[5, 6], &[17, 39]),
            Err(FreivaldsError::Consumed)
        ));
        let range_error = prepare_freivalds_projections(
            &weights,
            2,
            2,
            1,
            &seed,
            binding(),
            seed,
            [12_u8; 32],
            10,
            10,
            policy(),
            resources(),
        )
        .unwrap_err();
        assert_eq!(range_error, FreivaldsError::Range);
        let range_seed = [10_u8; 32];
        let (mut ranged_material, mut range_session) = imported(&weights, 2, 2, 1, range_seed);
        let mut ranged = ranged_material
            .claim_verifier(
                &range_seed,
                binding(),
                0,
                1,
                &mut range_session,
                resources(),
            )
            .unwrap();
        assert!(matches!(
            ranged.verify(&[50, 6], &[62, 174]),
            Err(FreivaldsError::Range)
        ));
    }

    #[test]
    fn binding_row_and_seed_change_material() {
        let weights = [1_i8, 2, 3, 4];
        let seed = [4_u8; 32];
        let (mut first, mut session) = imported(&weights, 2, 2, 2, seed);
        let second = prepared(&weights, 2, 2, 2, &[5_u8; 32]);
        assert_ne!(
            first.encode().unwrap().as_slice(),
            second.encode().unwrap().as_slice()
        );
        assert!(matches!(
            first.claim_verifier(&seed, b"wrong", 0, 1, &mut session, resources()),
            Err(FreivaldsError::Binding)
        ));
        let encoded = first.encode().unwrap();
        let row_width = first.checks() * first.in_features() * 4;
        assert_ne!(&encoded[..row_width], &encoded[row_width..row_width * 2]);
    }

    #[test]
    fn claimed_rows_cannot_be_reissued() {
        let weights = [1_i8, 2, 3, 4];
        let seed = [6_u8; 32];
        let (mut projections, mut session) = imported(&weights, 2, 2, 1, seed);
        let mut verifier = projections
            .claim_verifier(&seed, binding(), 0, 1, &mut session, resources())
            .unwrap();
        assert!(matches!(
            projections.claim_verifier(&seed, binding(), 0, 1, &mut session, resources(),),
            Err(FreivaldsError::Consumed)
        ));
        assert!(matches!(
            projections.encode(),
            Err(FreivaldsError::Consumed)
        ));
        assert_eq!(
            verifier.verify(&[5, 6], &[17, 39]).unwrap().as_slice(),
            [17, 39]
        );
    }

    #[test]
    fn soundness_policy_composes_over_attempt_budget() {
        assert_eq!(
            FreivaldsPolicy::sized_for_attempts(40, 1).unwrap().checks(),
            2
        );
        assert_eq!(
            FreivaldsPolicy::sized_for_attempts(40, 1_u64 << 22)
                .unwrap()
                .checks(),
            2
        );
        assert_eq!(
            FreivaldsPolicy::sized_for_attempts(40, 1_u64 << 23)
                .unwrap()
                .checks(),
            3
        );
        assert!(FreivaldsPolicy::sized_for_attempts(0, 1).is_err());
        assert!(FreivaldsPolicy::sized_for_attempts(100, u64::MAX).is_err());
    }

    #[test]
    fn completed_session_releases_its_process_registration() {
        let id = [21_u8; 32];
        drop(FreivaldsSession::register(id, policy()).unwrap());
        assert!(FreivaldsSession::register(id, policy()).is_ok());
    }

    #[test]
    fn declared_integer_bound_must_cover_the_weight_l1_bound() {
        assert!(matches!(
            prepare_freivalds_projections(
                &[127_i8, 127],
                1,
                2,
                1,
                &[1_u8; 32],
                binding(),
                [2_u8; 32],
                [1_u8; 32],
                100,
                1_000,
                policy(),
                resources(),
            ),
            Err(FreivaldsError::Range)
        ));
    }

    #[test]
    fn malformed_dimensions_and_policies_fail_before_work() {
        let seed = [0_u8; 32];
        assert!(prepare_freivalds_projections(
            &[],
            0,
            1,
            1,
            &seed,
            binding(),
            seed,
            [0_u8; 32],
            10,
            100,
            policy(),
            resources(),
        )
        .is_err());
        assert!(prepare_freivalds_projections(
            &[1],
            1,
            1,
            9,
            &seed,
            binding(),
            seed,
            [0_u8; 32],
            10,
            100,
            policy(),
            resources(),
        )
        .is_err());
        assert!(FreivaldsResourcePolicy::new(0, 1, 1, 1, 1, 1).is_err());
        assert!(prepare_freivalds_projections(
            &[1],
            1,
            1,
            1,
            &seed,
            &[],
            seed,
            [0_u8; 32],
            10,
            100,
            policy(),
            resources(),
        )
        .is_err());
    }

    #[test]
    fn projection_debug_is_secret_free() {
        let projections = prepared(&[1_i8, 2, 3, 4], 2, 2, 1, &[3_u8; 32]);
        assert_eq!(
            projections.encoded_len().unwrap(),
            policy().checks() * 2 * 4
        );
        let rendered = format!("{projections:?}");
        assert!(!rendered.contains("binding"));
        assert!(!rendered.contains("values"));
    }

    #[test]
    fn projection_matches_locked_independent_vector() {
        let material = prepare_freivalds_projections(
            &[1_i8, -2, 3, 4],
            2,
            2,
            1,
            &[0x11_u8; 32],
            b"locked-vector",
            [0x33_u8; 32],
            [0x22_u8; 32],
            10,
            100,
            FreivaldsPolicy::sized_for_attempts(40, 1).unwrap(),
            resources(),
        )
        .unwrap();
        assert_eq!(
            material.encode().unwrap().as_slice(),
            &[
                0xbc, 0x60, 0x61, 0x27, 0xb3, 0x81, 0x38, 0x8a, 0x6c, 0x49, 0x12, 0xcb, 0x4f, 0x9d,
                0x57, 0x77,
            ]
        );
    }

    #[test]
    fn aes_cmac_matches_nist_aes256_vectors() {
        let key = [
            0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe, 0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d,
            0x77, 0x81, 0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7, 0x2d, 0x98, 0x10, 0xa3,
            0x09, 0x14, 0xdf, 0xf4,
        ];
        assert_eq!(
            aes_cmac(&key, &[]).unwrap(),
            [
                0x02, 0x89, 0x62, 0xf6, 0x1b, 0x7b, 0xf8, 0x9e, 0xfc, 0x6b, 0x55, 0x1f, 0x46, 0x67,
                0xd9, 0x83,
            ]
        );
        assert_eq!(
            aes_cmac(
                &key,
                &[
                    0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73,
                    0x93, 0x17, 0x2a,
                ],
            )
            .unwrap(),
            [
                0x28, 0xa7, 0x02, 0x3f, 0x45, 0x2e, 0x8f, 0x82, 0xbd, 0x4b, 0xf2, 0x8d, 0x8c, 0x37,
                0xc3, 0x5c,
            ]
        );
    }

    #[test]
    fn imported_projection_material_preserves_binding_and_one_use() {
        let seed = [0x33_u8; 32];
        let prepared = prepared(&[1_i8, 2, 3, 4], 2, 2, 1, &seed);
        let encoded = prepared.encode().unwrap();
        let tag = prepared.authentication_tag(&seed).unwrap();
        let mut session = session(seed);
        let mut malformed = encoded.to_vec();
        malformed[..4].copy_from_slice(&FREIVALDS_FIELD_MODULUS.to_le_bytes());
        assert!(matches!(
            import_freivalds_projections(
                &malformed,
                &tag,
                &seed,
                &mut session,
                1,
                2,
                2,
                binding(),
                [11_u8; 32],
                10,
                1_000,
                7,
                policy(),
                resources(),
            ),
            Err(FreivaldsError::Integrity)
        ));
        let mut imported = import_freivalds_projections(
            &encoded,
            &tag,
            &seed,
            &mut session,
            1,
            2,
            2,
            binding(),
            [11_u8; 32],
            10,
            1_000,
            7,
            policy(),
            resources(),
        )
        .unwrap();
        let mut verifier = imported
            .claim_verifier(&seed, binding(), 0, 1, &mut session, resources())
            .unwrap();
        assert_eq!(
            verifier.verify(&[5, 6], &[17, 39]).unwrap().as_slice(),
            &[17, 39]
        );
    }
}
