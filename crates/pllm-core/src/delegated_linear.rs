use std::fmt;

use zeroize::{Zeroize, Zeroizing};

pub const SLALOM_FIELD_MODULUS: u32 = (1 << 24) - 3;
pub const SLALOM_CHALLENGE_REPETITIONS: usize = 2;
pub const SLALOM_MAX_MATRIX_ELEMENTS: usize = 1_048_576;
const CHALLENGE_ABS_BOUND: i32 = 1 << 19;
const CHALLENGE_WIDTH: u64 = (CHALLENGE_ABS_BOUND as u64) * 2 + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DelegatedLinearError {
    Allocation,
    Consumed,
    Dimension,
    FieldElement,
    Random,
    State,
    VerificationFailed,
}

pub struct MaskedFieldVector {
    values: Zeroizing<Vec<u32>>,
}

impl MaskedFieldVector {
    pub fn as_slice(&self) -> &[u32] {
        &self.values
    }
}

pub struct VerifiedFieldVector {
    values: Zeroizing<Vec<u32>>,
}

impl VerifiedFieldVector {
    pub fn as_slice(&self) -> &[u32] {
        &self.values
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum MaterialState {
    Ready,
    Masked,
    Consumed,
}

pub struct SlalomPreparedMatVec {
    rows: usize,
    cols: usize,
    mask: Zeroizing<Vec<u32>>,
    output_mask: Zeroizing<Vec<u32>>,
    challenges: Zeroizing<Vec<u32>>,
    projected_weights: Zeroizing<Vec<u32>>,
    input: Zeroizing<Vec<u32>>,
    state: MaterialState,
}

impl fmt::Debug for SlalomPreparedMatVec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SlalomPreparedMatVec")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field(
                "state",
                &match self.state {
                    MaterialState::Ready => "ready",
                    MaterialState::Masked => "masked",
                    MaterialState::Consumed => "consumed",
                },
            )
            .finish_non_exhaustive()
    }
}

impl SlalomPreparedMatVec {
    pub fn prepare(
        weights: &[u32],
        rows: usize,
        cols: usize,
    ) -> Result<Self, DelegatedLinearError> {
        validate_matrix(weights, rows, cols)?;
        let mask = random_field_vector(rows)?;
        let challenges = random_challenges(cols)?;
        Self::from_material(weights, rows, cols, mask, challenges)
    }

    fn from_material(
        weights: &[u32],
        rows: usize,
        cols: usize,
        mask: Vec<u32>,
        challenges: Vec<u32>,
    ) -> Result<Self, DelegatedLinearError> {
        let mask = Zeroizing::new(mask);
        let challenges = Zeroizing::new(challenges);
        validate_matrix(weights, rows, cols)?;
        validate_vector(&mask, rows)?;
        let challenge_len = cols
            .checked_mul(SLALOM_CHALLENGE_REPETITIONS)
            .ok_or(DelegatedLinearError::Dimension)?;
        validate_vector(&challenges, challenge_len)?;
        for challenge in challenges.chunks_exact(cols) {
            if challenge.iter().all(|value| *value == 0) {
                return Err(DelegatedLinearError::FieldElement);
            }
        }
        let output_mask = Zeroizing::new(matvec(weights, rows, cols, &mask)?);
        let projection_len = rows
            .checked_mul(SLALOM_CHALLENGE_REPETITIONS)
            .ok_or(DelegatedLinearError::Dimension)?;
        let mut projected_weights = Zeroizing::new(allocate(projection_len)?);
        for (repetition, challenge) in challenges.chunks_exact(cols).enumerate() {
            let projection = &mut projected_weights[repetition * rows..(repetition + 1) * rows];
            for row in 0..rows {
                projection[row] = dot(&weights[row * cols..(row + 1) * cols], challenge);
            }
        }
        Ok(Self {
            rows,
            cols,
            mask,
            output_mask,
            challenges,
            projected_weights,
            input: Zeroizing::new(Vec::new()),
            state: MaterialState::Ready,
        })
    }

    pub fn mask_input(&mut self, input: &[u32]) -> Result<MaskedFieldVector, DelegatedLinearError> {
        if self.state == MaterialState::Consumed {
            return Err(DelegatedLinearError::Consumed);
        }
        if self.state != MaterialState::Ready {
            return Err(DelegatedLinearError::State);
        }
        validate_vector(input, self.rows)?;
        let mut saved_input = Zeroizing::new(allocate(self.rows)?);
        saved_input.copy_from_slice(input);
        let mut masked = allocate(self.rows)?;
        for ((output, value), mask) in masked.iter_mut().zip(input).zip(self.mask.iter()) {
            *output = add_mod(*value, *mask);
        }
        self.input = saved_input;
        self.mask.zeroize();
        self.state = MaterialState::Masked;
        Ok(MaskedFieldVector {
            values: Zeroizing::new(masked),
        })
    }

    pub fn recover_and_verify(
        &mut self,
        masked_output: &[u32],
    ) -> Result<VerifiedFieldVector, DelegatedLinearError> {
        if self.state == MaterialState::Consumed {
            return Err(DelegatedLinearError::Consumed);
        }
        if self.state != MaterialState::Masked {
            return Err(DelegatedLinearError::State);
        }
        self.state = MaterialState::Consumed;
        let result = self.recover_and_verify_inner(masked_output);
        self.output_mask.zeroize();
        self.challenges.zeroize();
        self.projected_weights.zeroize();
        self.input.zeroize();
        result
    }

    fn recover_and_verify_inner(
        &self,
        masked_output: &[u32],
    ) -> Result<VerifiedFieldVector, DelegatedLinearError> {
        validate_vector(masked_output, self.cols)?;
        let mut output = allocate(self.cols)?;
        for ((value, masked), mask) in output
            .iter_mut()
            .zip(masked_output)
            .zip(self.output_mask.iter())
        {
            *value = sub_mod(*masked, *mask);
        }
        for repetition in 0..SLALOM_CHALLENGE_REPETITIONS {
            let challenge = &self.challenges[repetition * self.cols..(repetition + 1) * self.cols];
            let projection =
                &self.projected_weights[repetition * self.rows..(repetition + 1) * self.rows];
            if dot(&output, challenge) != dot(&self.input, projection) {
                output.zeroize();
                return Err(DelegatedLinearError::VerificationFailed);
            }
        }
        Ok(VerifiedFieldVector {
            values: Zeroizing::new(output),
        })
    }
}

pub fn slalom_evaluate_masked(
    weights: &[u32],
    rows: usize,
    cols: usize,
    masked_input: &[u32],
) -> Result<Vec<u32>, DelegatedLinearError> {
    validate_matrix(weights, rows, cols)?;
    validate_vector(masked_input, rows)?;
    matvec(weights, rows, cols, masked_input)
}

fn validate_matrix(weights: &[u32], rows: usize, cols: usize) -> Result<(), DelegatedLinearError> {
    if rows == 0 || cols == 0 {
        return Err(DelegatedLinearError::Dimension);
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or(DelegatedLinearError::Dimension)?;
    if elements > SLALOM_MAX_MATRIX_ELEMENTS || weights.len() != elements {
        return Err(DelegatedLinearError::Dimension);
    }
    if weights.iter().any(|value| *value >= SLALOM_FIELD_MODULUS) {
        return Err(DelegatedLinearError::FieldElement);
    }
    Ok(())
}

fn validate_vector(values: &[u32], length: usize) -> Result<(), DelegatedLinearError> {
    if values.len() != length {
        return Err(DelegatedLinearError::Dimension);
    }
    if values.iter().any(|value| *value >= SLALOM_FIELD_MODULUS) {
        return Err(DelegatedLinearError::FieldElement);
    }
    Ok(())
}

fn allocate(length: usize) -> Result<Vec<u32>, DelegatedLinearError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| DelegatedLinearError::Allocation)?;
    values.resize(length, 0);
    Ok(values)
}

fn matvec(
    weights: &[u32],
    rows: usize,
    cols: usize,
    input: &[u32],
) -> Result<Vec<u32>, DelegatedLinearError> {
    let mut output = allocate(cols)?;
    for row in 0..rows {
        let input_value = input[row];
        for col in 0..cols {
            output[col] = multiply_add_mod(output[col], input_value, weights[row * cols + col]);
        }
    }
    Ok(output)
}

fn dot(left: &[u32], right: &[u32]) -> u32 {
    left.iter()
        .zip(right)
        .fold(0, |sum, (left, right)| multiply_add_mod(sum, *left, *right))
}

fn add_mod(left: u32, right: u32) -> u32 {
    ((u64::from(left) + u64::from(right)) % u64::from(SLALOM_FIELD_MODULUS)) as u32
}

fn sub_mod(left: u32, right: u32) -> u32 {
    ((u64::from(left) + u64::from(SLALOM_FIELD_MODULUS) - u64::from(right))
        % u64::from(SLALOM_FIELD_MODULUS)) as u32
}

fn multiply_add_mod(sum: u32, left: u32, right: u32) -> u32 {
    ((u64::from(sum) + u64::from(left) * u64::from(right)) % u64::from(SLALOM_FIELD_MODULUS)) as u32
}

fn random_field_vector(length: usize) -> Result<Vec<u32>, DelegatedLinearError> {
    let mut output = Zeroizing::new(allocate(length)?);
    fill_uniform(&mut output, u64::from(SLALOM_FIELD_MODULUS))?;
    Ok(std::mem::take(&mut *output))
}

fn random_challenges(cols: usize) -> Result<Vec<u32>, DelegatedLinearError> {
    let length = cols
        .checked_mul(SLALOM_CHALLENGE_REPETITIONS)
        .ok_or(DelegatedLinearError::Dimension)?;
    let mut output = Zeroizing::new(allocate(length)?);
    loop {
        fill_uniform(&mut output, CHALLENGE_WIDTH)?;
        for value in output.iter_mut() {
            let signed = i64::from(*value) - i64::from(CHALLENGE_ABS_BOUND);
            *value = if signed < 0 {
                (i64::from(SLALOM_FIELD_MODULUS) + signed) as u32
            } else {
                signed as u32
            };
        }
        if output
            .chunks_exact(cols)
            .all(|challenge| challenge.iter().any(|value| *value != 0))
        {
            return Ok(std::mem::take(&mut *output));
        }
    }
}

fn fill_uniform(output: &mut [u32], modulus: u64) -> Result<(), DelegatedLinearError> {
    let zone = ((u64::from(u32::MAX) + 1) / modulus) * modulus;
    let mut filled = 0;
    let mut bytes = Zeroizing::new([0_u8; 4096]);
    while filled < output.len() {
        getrandom::fill(&mut *bytes).map_err(|_| DelegatedLinearError::Random)?;
        for chunk in bytes.chunks_exact(4) {
            let sample = u64::from(u32::from_le_bytes(chunk.try_into().unwrap()));
            if sample < zone {
                output[filled] = (sample % modulus) as u32;
                filled += 1;
                if filled == output.len() {
                    break;
                }
            }
        }
    }
    bytes.zeroize();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn material(weights: &[u32], rows: usize, cols: usize) -> SlalomPreparedMatVec {
        let mask = (0..rows)
            .map(|index| (index as u32 * 7 + 3) % SLALOM_FIELD_MODULUS)
            .collect();
        let mut challenges = Vec::new();
        for repetition in 0..SLALOM_CHALLENGE_REPETITIONS {
            for col in 0..cols {
                challenges.push((repetition as u32 + 2) * (col as u32 + 1));
            }
        }
        SlalomPreparedMatVec::from_material(weights, rows, cols, mask, challenges).unwrap()
    }

    #[test]
    fn masks_evaluates_recovers_and_verifies_exact_field_matvec() {
        let weights = [2, 3, 5, 7, 11, 13];
        let input = [17, 19];
        let clear = matvec(&weights, 2, 3, &input).unwrap();
        let mut prepared = material(&weights, 2, 3);
        let masked = prepared.mask_input(&input).unwrap();
        assert_ne!(masked.as_slice(), input);
        let remote = slalom_evaluate_masked(&weights, 2, 3, masked.as_slice()).unwrap();
        let verified = prepared.recover_and_verify(&remote).unwrap();
        assert_eq!(verified.as_slice(), clear);
        assert!(matches!(
            prepared.recover_and_verify(&remote),
            Err(DelegatedLinearError::Consumed)
        ));
    }

    #[test]
    fn tampering_fails_verification_and_burns_material() {
        let weights = [2, 3, 5, 7, 11, 13];
        let input = [17, 19];
        let mut prepared = material(&weights, 2, 3);
        let masked = prepared.mask_input(&input).unwrap();
        let mut remote = slalom_evaluate_masked(&weights, 2, 3, masked.as_slice()).unwrap();
        remote[1] = add_mod(remote[1], 1);
        assert!(matches!(
            prepared.recover_and_verify(&remote),
            Err(DelegatedLinearError::VerificationFailed)
        ));
        assert!(matches!(
            prepared.mask_input(&input),
            Err(DelegatedLinearError::Consumed)
        ));
    }

    #[test]
    fn state_and_field_bounds_fail_closed() {
        let weights = [1, 2, 3, 4];
        let mut prepared = material(&weights, 2, 2);
        assert!(matches!(
            prepared.recover_and_verify(&[0, 0]),
            Err(DelegatedLinearError::State)
        ));
        assert!(matches!(
            prepared.mask_input(&[1]),
            Err(DelegatedLinearError::Dimension)
        ));
        assert!(matches!(
            prepared.mask_input(&[SLALOM_FIELD_MODULUS, 1]),
            Err(DelegatedLinearError::FieldElement)
        ));
        assert!(matches!(
            slalom_evaluate_masked(&[SLALOM_FIELD_MODULUS], 1, 1, &[0]),
            Err(DelegatedLinearError::FieldElement)
        ));
        assert!(matches!(
            slalom_evaluate_masked(&[], 0, 1, &[]),
            Err(DelegatedLinearError::Dimension)
        ));
    }

    #[test]
    fn modular_matvec_matches_scalar_reference_across_boundaries() {
        let weights = [
            0,
            1,
            SLALOM_FIELD_MODULUS - 1,
            SLALOM_FIELD_MODULUS - 2,
            17,
            99,
        ];
        for first in [0, 1, 17, SLALOM_FIELD_MODULUS - 1] {
            for second in [0, 2, 101, SLALOM_FIELD_MODULUS - 2] {
                let output = slalom_evaluate_masked(&weights, 2, 3, &[first, second]).unwrap();
                for col in 0..3 {
                    let expected = ((u64::from(first) * u64::from(weights[col])
                        + u64::from(second) * u64::from(weights[3 + col]))
                        % u64::from(SLALOM_FIELD_MODULUS))
                        as u32;
                    assert_eq!(output[col], expected);
                }
            }
        }
    }

    #[test]
    fn prepared_debug_does_not_expose_secret_material() {
        let prepared = material(&[1, 2, 3, 4], 2, 2);
        let rendered = format!("{prepared:?}");
        assert_eq!(
            rendered,
            "SlalomPreparedMatVec { rows: 2, cols: 2, state: \"ready\", .. }"
        );
        assert!(!rendered.contains('3'));
    }
}
