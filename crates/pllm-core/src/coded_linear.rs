//! Small, source-independent coded-verification reference for public field matvecs.
//!
//! This implements the *verification* half of a coded delegation protocol. It
//! has no private-vector masking, model schedule, network transport or batch
//! projection delegation; none of those can be inferred from an accepted claim.
use std::sync::atomic::{AtomicU32, Ordering};

use sha2::{Digest as _, Sha256};

/// The BabyBear field used by the bounded reference, not the normal u16/u24/u32 rings.
pub const CODED_LINEAR_FIELD: u32 = 2_013_265_921;
pub const CODED_LINEAR_CHALLENGE_WEIGHT: usize = 41;
pub const CODED_LINEAR_REPETITIONS: usize = 2;
pub const CODED_LINEAR_MAX_CLAIMS: u32 = 65_536;

/// Public offline encoding of a signed-int8 matrix, with fresh private checks
/// sampled only *after* a claimed result is fixed.
pub struct CodedMatVecVerifier {
    rows: usize,
    cols: usize,
    encoded_rows: Vec<u32>,
    matrix_digest: [u8; 32],
    claims: AtomicU32,
}

impl CodedMatVecVerifier {
    /// Walsh-linear code over an odd prime field. Each nonzero codeword has
    /// relative distance >= 1/2: pairing positions that differ in one bit
    /// cannot give two zero values when that bit's coefficient is nonzero.
    /// Limited to 6..=10 matrix rows; beyond this the exponential block length
    /// is inappropriate and a reviewed linear-time code is required.
    pub fn preprocess(weights: &[i8], rows: usize, cols: usize) -> Result<Self, String> {
        if !(6..=10).contains(&rows) || cols == 0 || cols > 16_384 {
            return Err("coded reference requires 6..=10 rows and 1..=16384 columns".into());
        }
        if rows.checked_mul(cols) != Some(weights.len()) {
            return Err("coded reference matrix shape does not match weight length".into());
        }
        let length = 1_usize << rows;
        let mut encoded_rows = vec![0_u32; length * cols];
        for position in 0..length {
            for row in 0..rows {
                let negative = (position >> row) & 1 != 0;
                for col in 0..cols {
                    let weight = i64::from(weights[row * cols + col]);
                    let term = if negative { -weight } else { weight };
                    let index = position * cols + col;
                    encoded_rows[index] = field_add(encoded_rows[index], field_from_signed(term));
                }
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.coded_linear.walsh_reference.v1");
        hash.update((rows as u64).to_le_bytes());
        hash.update((cols as u64).to_le_bytes());
        hash.update(
            weights
                .iter()
                .map(|weight| *weight as u8)
                .collect::<Vec<_>>(),
        );
        Ok(Self {
            rows,
            cols,
            encoded_rows,
            matrix_digest: hash.finalize().into(),
            claims: AtomicU32::new(0),
        })
    }

    pub fn matrix_digest(&self) -> [u8; 32] {
        self.matrix_digest
    }

    /// Return `false` for a wrong answer, `Err` for invalid claims or depleted
    /// attempt budget. Every call consumes an attempt, including malformed ones.
    pub fn verify(&self, input: &[u32], claimed: &[u32]) -> Result<bool, String> {
        self.claims
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < CODED_LINEAR_MAX_CLAIMS).then_some(count + 1)
            })
            .map_err(|_| "coded reference attempt budget exhausted".to_string())?;
        if input.len() != self.cols || claimed.len() != self.rows {
            return Err("coded reference input/output length mismatch".into());
        }
        if input
            .iter()
            .chain(claimed)
            .any(|&element| element >= CODED_LINEAR_FIELD)
        {
            return Err("coded reference noncanonical field element".into());
        }
        let length = 1_usize << self.rows;
        for _ in 0..CODED_LINEAR_REPETITIONS {
            // The output and all validation are already fixed before the secret
            // support and nonzero coefficients are generated.
            let mut support: Vec<usize> = (0..length).collect();
            let mut projected_row = vec![0_u32; self.cols];
            let mut projected_output = 0_u32;
            for index in 0..CODED_LINEAR_CHALLENGE_WEIGHT {
                let offset = random_below((length - index) as u32)? as usize;
                support.swap(index, index + offset);
                let position = support[index];
                let coefficient = random_below(CODED_LINEAR_FIELD - 1)? + 1;
                let mut encoded_output = 0_u32;
                for (row, &value) in claimed.iter().enumerate() {
                    let term = if (position >> row) & 1 != 0 {
                        field_neg(value)
                    } else {
                        value
                    };
                    encoded_output = field_add(encoded_output, term);
                }
                projected_output =
                    field_add(projected_output, field_mul(coefficient, encoded_output));
                for (col, entry) in projected_row.iter_mut().enumerate() {
                    *entry = field_add(
                        *entry,
                        field_mul(coefficient, self.encoded_rows[position * self.cols + col]),
                    );
                }
            }
            let projected_input = projected_row
                .iter()
                .zip(input)
                .fold(0_u32, |sum, (&weight, &value)| {
                    field_add(sum, field_mul(weight, value))
                });
            if projected_input != projected_output {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn field_from_signed(value: i64) -> u32 {
    value.rem_euclid(i64::from(CODED_LINEAR_FIELD)) as u32
}

fn field_add(a: u32, b: u32) -> u32 {
    ((u64::from(a) + u64::from(b)) % u64::from(CODED_LINEAR_FIELD)) as u32
}

fn field_neg(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        CODED_LINEAR_FIELD - value
    }
}

fn field_mul(a: u32, b: u32) -> u32 {
    ((u64::from(a) * u64::from(b)) % u64::from(CODED_LINEAR_FIELD)) as u32
}

fn random_below(upper: u32) -> Result<u32, String> {
    if upper == 0 {
        return Err("coded reference randomness domain is empty".into());
    }
    let domain = u64::from(u32::MAX) + 1;
    let limit = domain - domain % u64::from(upper);
    loop {
        let mut bytes = [0_u8; 4];
        getrandom::fill(&mut bytes).map_err(|_| "coded reference randomness unavailable")?;
        let sample = u64::from(u32::from_le_bytes(bytes));
        if sample < limit {
            return Ok((sample % u64::from(upper)) as u32);
        }
    }
}

/// Public numerical RAA code from Maverick's repeat/permute/scale/accumulate
/// construction. This reference carries no distance certificate, LPN security
/// parameter, or private-mask state and cannot authorize a verifier session.
pub struct RaaNumericCode {
    rows: usize,
    first_permutation: Vec<usize>,
    second_permutation: Vec<usize>,
    first_scale: Vec<u32>,
    second_scale: Vec<u32>,
}

impl RaaNumericCode {
    pub fn from_public_seed(rows: usize, seed: [u8; 32]) -> Result<Self, String> {
        if !(1..=128).contains(&rows) {
            return Err("RAA numeric reference supports 1..=128 rows".into());
        }
        let length = rows * 8;
        let mut random = PublicCodeSampler { seed, counter: 0 };
        let mut permutation = || -> Result<Vec<usize>, String> {
            let mut indices: Vec<usize> = (0..length).collect();
            for i in (1..length).rev() {
                let selected = random.below(i as u32 + 1)? as usize;
                indices.swap(i, selected);
            }
            Ok(indices)
        };
        let first_permutation = permutation()?;
        let second_permutation = permutation()?;
        let mut first_scale = Vec::with_capacity(length);
        let mut second_scale = Vec::with_capacity(length);
        for _ in 0..length {
            first_scale.push(random.below(CODED_LINEAR_FIELD - 1)? + 1);
            second_scale.push(random.below(CODED_LINEAR_FIELD - 1)? + 1);
        }
        Ok(Self {
            rows,
            first_permutation,
            second_permutation,
            first_scale,
            second_scale,
        })
    }

    pub fn encoded_rows(&self) -> usize {
        self.rows * 8
    }

    pub fn encode(&self, values: &[u32]) -> Result<Vec<u32>, String> {
        if values.len() != self.rows || values.iter().any(|&v| v >= CODED_LINEAR_FIELD) {
            return Err("RAA input must contain the declared number of field elements".into());
        }
        let length = self.encoded_rows();
        let repeated: Vec<u32> = values.iter().flat_map(|&v| [v; 8]).collect();
        let mut first = vec![0u32; length];
        let mut prefix = 0;
        for i in 0..length {
            prefix = field_add(
                prefix,
                field_mul(repeated[self.first_permutation[i]], self.first_scale[i]),
            );
            first[i] = prefix;
        }
        let mut encoded = vec![0u32; length];
        prefix = 0;
        for i in 0..length {
            prefix = field_add(
                prefix,
                field_mul(first[self.second_permutation[i]], self.second_scale[i]),
            );
            encoded[i] = prefix;
        }
        Ok(encoded)
    }

    /// Numeric transpose G e for a sparse vector e in the encoded domain.
    /// Sparse coefficients are caller-owned; this does not create private masks.
    pub fn syndrome(&self, sparse: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        let length = self.encoded_rows();
        if sparse.is_empty() || sparse.len() > length {
            return Err("RAA syndrome sparsity exceeds encoded dimensions".into());
        }
        let mut current = vec![0u32; length];
        for &(position, value) in sparse {
            if position >= length || value == 0 || value >= CODED_LINEAR_FIELD {
                return Err("RAA syndrome contains invalid position or coefficient".into());
            }
            if current[position] != 0 {
                return Err("RAA syndrome positions must be distinct".into());
            }
            current[position] = value;
        }
        // The transpose of a prefix accumulation is a suffix accumulation.
        let mut sum = 0;
        for value in current.iter_mut().rev() {
            sum = field_add(sum, *value);
            *value = sum;
        }
        let mut first = vec![0u32; length];
        for (position, &value) in current.iter().enumerate() {
            let mapped = self.second_permutation[position];
            first[mapped] = field_add(first[mapped], field_mul(value, self.second_scale[position]));
        }
        sum = 0;
        for value in first.iter_mut().rev() {
            sum = field_add(sum, *value);
            *value = sum;
        }
        let mut repeated = vec![0u32; length];
        for (position, &value) in first.iter().enumerate() {
            let mapped = self.first_permutation[position];
            repeated[mapped] = field_add(
                repeated[mapped],
                field_mul(value, self.first_scale[position]),
            );
        }
        Ok(repeated
            .chunks_exact(8)
            .map(|repeat| repeat.iter().copied().fold(0, field_add))
            .collect())
    }

    /// Q = G^T M. Return row-major field values; caller owns resource policy.
    pub fn preprocess_i8(&self, weights: &[i8], columns: usize) -> Result<Vec<u32>, String> {
        if columns == 0 || columns > 256 || weights.len() != self.rows * columns {
            return Err("RAA reference weight shape exceeds 128 x 256".into());
        }
        let mut encoded = vec![0u32; self.encoded_rows() * columns];
        for column in 0..columns {
            let source = (0..self.rows)
                .map(|row| field_from_signed(i64::from(weights[row * columns + column])))
                .collect::<Vec<_>>();
            for (row, value) in self.encode(&source)?.into_iter().enumerate() {
                encoded[row * columns + column] = value;
            }
        }
        Ok(encoded)
    }
}

struct PublicCodeSampler {
    seed: [u8; 32],
    counter: u64,
}

impl PublicCodeSampler {
    fn below(&mut self, size: u32) -> Result<u32, String> {
        if size == 0 {
            return Err("RAA sampling bound must be positive".into());
        }
        let range = u64::from(u32::MAX) + 1;
        let ceiling = range - range % u64::from(size);
        loop {
            let mut hasher = Sha256::new();
            hasher.update(b"pllm.public-raa-code.v1");
            hasher.update(self.seed);
            hasher.update(self.counter.to_le_bytes());
            self.counter = self
                .counter
                .checked_add(1)
                .ok_or("RAA seed stream exhausted")?;
            let hash: [u8; 32] = hasher.finalize().into();
            let number = u32::from_le_bytes(hash[..4].try_into().unwrap());
            if u64::from(number) < ceiling {
                return Ok(number % size);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coded_verification_accepts_field_matvec_and_rejects_wrong_sparse_outputs() {
        let rows = 8;
        let cols = 13;
        let weights: Vec<i8> = (0..rows * cols)
            .map(|index| (index % 17) as i8 - 8)
            .collect();
        let verifier = CodedMatVecVerifier::preprocess(&weights, rows, cols).unwrap();
        let input: Vec<u32> = (0..cols).map(|index| (index * 31) as u32).collect();
        let result: Vec<u32> = weights
            .chunks_exact(cols)
            .map(|row| {
                field_from_signed(
                    row.iter()
                        .zip(&input)
                        .map(|(&weight, &value)| i64::from(weight) * i64::from(value))
                        .sum(),
                )
            })
            .collect();
        assert!(verifier.verify(&input, &result).unwrap());
        for changed_row in 0..rows {
            let mut forged = result.clone();
            forged[changed_row] = field_add(forged[changed_row], 1);
            assert!(!verifier.verify(&input, &forged).unwrap());
        }
        let mut altered = weights;
        altered[1] += 1;
        assert_ne!(
            verifier.matrix_digest(),
            CodedMatVecVerifier::preprocess(&altered, rows, cols)
                .unwrap()
                .matrix_digest()
        );
    }

    #[test]
    fn coded_verification_rejects_invalid_domains_and_shapes() {
        assert!(CodedMatVecVerifier::preprocess(&[], 5, 1).is_err());
        assert!(CodedMatVecVerifier::preprocess(&[], 11, 1).is_err());
        let verifier = CodedMatVecVerifier::preprocess(&[1; 6], 6, 1).unwrap();
        assert!(verifier.verify(&[], &[0; 6]).is_err());
        assert!(verifier.verify(&[CODED_LINEAR_FIELD], &[0; 6]).is_err());
        assert!(verifier.verify(&[0], &[CODED_LINEAR_FIELD; 6]).is_err());
        assert!(verifier.verify(&[0], &[0; 6]).unwrap());
    }

    #[test]
    fn walsh_code_has_half_distance_for_all_small_nonzero_messages() {
        for rows in 1..=4 {
            let length = 1 << rows;
            for message in 1..(1 << rows) {
                let nonzero = (0..length)
                    .filter(|&position| {
                        let value: i64 = (0..rows)
                            .filter(|&row| (message >> row) & 1 != 0)
                            .map(|row| if (position >> row) & 1 != 0 { -1 } else { 1 })
                            .sum();
                        value != 0
                    })
                    .count();
                assert!(nonzero >= length / 2);
            }
        }
    }

    #[test]
    fn public_raa_preprocessing_commutes_with_real_matrix_products() {
        let code = RaaNumericCode::from_public_seed(64, [27; 32]).unwrap();
        let weights = (0..64 * 17)
            .map(|i| ((i * 13 % 255) - 127) as i8)
            .collect::<Vec<_>>();
        let input = (0..17)
            .map(|i| ((i as u64 * 600_000_017) % u64::from(CODED_LINEAR_FIELD)) as u32)
            .collect::<Vec<_>>();
        let preprocessed = code.preprocess_i8(&weights, input.len()).unwrap();
        let output = weights
            .chunks_exact(input.len())
            .map(|row| {
                row.iter().zip(&input).fold(0, |sum, (&w, &x)| {
                    field_add(sum, field_mul(field_from_signed(i64::from(w)), x))
                })
            })
            .collect::<Vec<_>>();
        let expected = code.encode(&output).unwrap();
        for (row, &value) in expected.iter().enumerate() {
            let projected = preprocessed[row * input.len()..][..input.len()]
                .iter()
                .zip(&input)
                .fold(0, |sum, (&q, &x)| field_add(sum, field_mul(q, x)));
            assert_eq!(projected, value);
        }
        let other = RaaNumericCode::from_public_seed(64, [28; 32]).unwrap();
        assert_ne!(other.encode(&output).unwrap(), expected);
        assert!(RaaNumericCode::from_public_seed(129, [27; 32]).is_err());
        assert!(code.preprocess_i8(&weights, 0).is_err());
        assert!(code.encode(&vec![CODED_LINEAR_FIELD; 64]).is_err());
    }

    #[test]
    fn public_raa_numeric_mask_and_preprocessing_match_field_product() {
        let code = RaaNumericCode::from_public_seed(64, [31; 32]).unwrap();
        let sparse = (0..160)
            .map(|index| (index * 3, (index as u32 * 81_234 + 1) % CODED_LINEAR_FIELD))
            .collect::<Vec<_>>();
        let weights = (0..8 * 64)
            .map(|index| (index % 127) as i8 - 63)
            .collect::<Vec<_>>();
        let hidden_input = (0..64)
            .map(|index| (index as u32 * 129_981 + 17) % CODED_LINEAR_FIELD)
            .collect::<Vec<_>>();
        let mask = code.syndrome(&sparse).unwrap();
        let prepared = weights
            .chunks_exact(64)
            .map(|row| {
                let field_row = row
                    .iter()
                    .map(|&weight| field_from_signed(i64::from(weight)))
                    .collect::<Vec<_>>();
                let encoded = code.encode(&field_row).unwrap();
                sparse.iter().fold(0, |sum, &(position, value)| {
                    field_add(sum, field_mul(encoded[position], value))
                })
            })
            .collect::<Vec<_>>();
        let encrypted_input = hidden_input
            .iter()
            .zip(&mask)
            .map(|(&input, &mask)| field_add(input, mask))
            .collect::<Vec<_>>();
        for (row, &expected_mask) in weights.chunks_exact(64).zip(&prepared) {
            let returned = row.iter().zip(&encrypted_input).fold(0, |sum, (&w, &x)| {
                field_add(sum, field_mul(field_from_signed(i64::from(w)), x))
            });
            let plain = row.iter().zip(&hidden_input).fold(0, |sum, (&w, &x)| {
                field_add(sum, field_mul(field_from_signed(i64::from(w)), x))
            });
            assert_eq!(field_add(plain, expected_mask), returned);
        }
        assert!(code.syndrome(&[(0, 1), (0, 2)]).is_err());
        assert!(code.syndrome(&[(code.encoded_rows(), 1)]).is_err());
        assert!(code.syndrome(&[(0, CODED_LINEAR_FIELD)]).is_err());
    }
}
