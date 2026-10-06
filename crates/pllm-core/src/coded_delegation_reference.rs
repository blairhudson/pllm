//! Bounded, in-process RAA masked delegation from Maverick Protocol 3.
//!
//! This reference exercises the algebra, exact sparse sampling and one-use
//! client ownership. RAA distance and dual-LPN security are NOT certified.
//! Acceptance is a conditional coded check, not an admitted verifier contract.
//! Public preprocessing stays client-local and is charged before allocation.
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use crate::coded_linear::{
    field_add, field_from_signed, field_mul, field_neg, random_below, RaaNumericCode,
    CODED_LINEAR_CHALLENGE_WEIGHT, CODED_LINEAR_FIELD, CODED_LINEAR_MAX_CLAIMS,
    CODED_LINEAR_REPETITIONS, RAA_MAX_DIMENSION, RAA_MAX_PREPROCESS_BYTES,
};

pub const RAA_MAX_LIVE_QUERIES: u32 = 8;

/// Payload accounting, excluding allocator overhead, caller weights and scratch.
#[derive(Debug, Clone, Copy)]
pub struct RaaDelegationResources {
    pub privacy_matrix_bytes: usize,
    pub verification_matrix_bytes: usize,
    pub code_payload_bytes: usize,
    pub persistent_payload_bytes: usize,
    pub max_live_query_payload_bytes: usize,
    pub privacy_sparsity: usize,
}

impl RaaDelegationResources {
    /// Geometry only: a successful result is not a privacy or runtime admission.
    pub fn for_shape(rows: usize, columns: usize) -> Result<Self, String> {
        if !(64..=RAA_MAX_DIMENSION).contains(&rows) || !(64..=RAA_MAX_DIMENSION).contains(&columns)
        {
            return Err("RAA delegation reference requires dimensions in 64..=16384".into());
        }
        let matrix_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(8))
            .and_then(|encoded_elements| encoded_elements.checked_mul(size_of::<u32>()))
            .ok_or("RAA preprocessing size overflow")?;
        // Each code stores two permutations and two nonzero field-scale arrays.
        let code_payload_bytes =
            8 * (rows + columns) * (2 * size_of::<usize>() + 2 * size_of::<u32>());
        // Paper Eq. (1), lambda=128, target delta=1/2. This is a heuristic,
        // conditional on the code distribution, not 128-bit security evidence.
        let privacy_sparsity =
            (2.0 * std::f64::consts::LN_2 * (128.0 - ((columns * 8) as f64).log2())).ceil()
                as usize;
        Ok(Self {
            privacy_matrix_bytes: matrix_bytes,
            verification_matrix_bytes: matrix_bytes,
            code_payload_bytes,
            persistent_payload_bytes: matrix_bytes
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(code_payload_bytes))
                .ok_or("RAA preprocessing size overflow")?,
            max_live_query_payload_bytes: RAA_MAX_LIVE_QUERIES as usize * (rows + columns) * 4,
            privacy_sparsity,
        })
    }

    pub fn within_reference_bound(&self) -> bool {
        self.persistent_payload_bytes
            .checked_add(self.max_live_query_payload_bytes)
            .is_some_and(|bytes| bytes <= RAA_MAX_PREPROCESS_BYTES)
    }
}

struct SparseChallenge {
    entries: Vec<(usize, u32)>,
}

impl SparseChallenge {
    /// Uniform support without replacement; independent uniform nonzero values.
    /// Neither a public seed nor a structured fixed-support substitute is used.
    fn sample(length: usize, weight: usize) -> Result<Self, String> {
        if length > RAA_MAX_DIMENSION * 8 || weight == 0 || weight > length {
            return Err("sparse reference challenge exceeds its bounded domain".into());
        }
        let mut pool = Zeroizing::new((0..length).collect::<Vec<_>>());
        let mut result = Self {
            entries: Vec::with_capacity(weight),
        };
        for index in 0..weight {
            let selected = index + random_below((length - index) as u32)? as usize;
            pool.swap(index, selected);
            result
                .entries
                .push((pool[index], random_below(CODED_LINEAR_FIELD - 1)? + 1));
        }
        Ok(result)
    }
}

impl Drop for SparseChallenge {
    fn drop(&mut self) {
        for (position, value) in &mut self.entries {
            position.zeroize();
            value.zeroize();
        }
    }
}

/// The caller may send only `masked_input()` to the matrix evaluator.
/// The correction remains client-owned. Moving this value into `finish` burns
/// it on every return path; dropping it cancels and erases retained material.
/// This type deliberately has no Clone, Debug or serialization implementation.
///
/// ```compile_fail
/// use pllm_core::coded_delegation_reference::{RaaDelegationReference, RaaMaskedQuery};
/// fn replay(reference: &RaaDelegationReference, query: RaaMaskedQuery, output: &[u32]) {
///     let _ = reference.finish(query, output);
///     let _ = reference.finish(query, output); // query was consumed, even on error
/// }
/// ```
pub struct RaaMaskedQuery {
    binding: [u8; 32],
    masked_input: Zeroizing<Vec<u32>>,
    correction: Zeroizing<Vec<u32>>,
    _lease: QueryLease,
}

struct QueryLease(Arc<AtomicU32>);

impl Drop for QueryLease {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl RaaMaskedQuery {
    pub fn masked_input(&self) -> &[u32] {
        &self.masked_input
    }
}

pub struct RaaDelegationReference {
    rows: usize,
    columns: usize,
    resources: RaaDelegationResources,
    binding: [u8; 32],
    privacy_code: RaaNumericCode,
    verification_code: RaaNumericCode,
    // P is [m,8n]; Q is [8m,n], both public but retained by the client here.
    privacy_matrix: Vec<u32>,
    verification_matrix: Vec<u32>,
    issued: AtomicU32,
    live_queries: Arc<AtomicU32>,
}

impl RaaDelegationReference {
    /// Seeds describe public codes only. Each issue samples fresh secret noise.
    /// Full P+Q+code payload is checked before hashing or allocating either code.
    pub fn preprocess(
        weights: &[i8],
        rows: usize,
        columns: usize,
        public_seed: [u8; 32],
    ) -> Result<Self, String> {
        let resources = RaaDelegationResources::for_shape(rows, columns)?;
        if !resources.within_reference_bound() {
            return Err("RAA delegation preprocessing exceeds the 256 MiB payload bound".into());
        }
        if weights.len() != rows * columns {
            return Err("RAA delegation reference weight shape mismatch".into());
        }
        let code_seed = |role: &[u8]| -> [u8; 32] {
            let mut hash = Sha256::new();
            hash.update(b"pllm.raa_delegation_reference.code.v1");
            hash.update(public_seed);
            hash.update(role);
            hash.finalize().into()
        };
        let privacy_code = RaaNumericCode::from_public_seed(columns, code_seed(b"privacy"))?;
        let verification_code = RaaNumericCode::from_public_seed(rows, code_seed(b"verification"))?;
        let mut privacy_matrix = Vec::with_capacity(resources.privacy_matrix_bytes / 4);
        for row in weights.chunks_exact(columns) {
            let field_row = row
                .iter()
                .map(|&w| field_from_signed(i64::from(w)))
                .collect::<Vec<_>>();
            privacy_matrix.extend(privacy_code.encode(&field_row)?);
        }
        let verification_matrix = verification_code.preprocess_i8(weights, columns)?;
        let mut hash = Sha256::new();
        hash.update(b"pllm.raa_delegation_reference.binding.v1");
        hash.update(public_seed);
        hash.update((rows as u64).to_le_bytes());
        hash.update((columns as u64).to_le_bytes());
        for chunk in weights.chunks(4096) {
            hash.update(chunk.iter().map(|&w| w as u8).collect::<Vec<_>>());
        }
        Ok(Self {
            rows,
            columns,
            resources,
            binding: hash.finalize().into(),
            privacy_code,
            verification_code,
            privacy_matrix,
            verification_matrix,
            issued: AtomicU32::new(0),
            live_queries: Arc::new(AtomicU32::new(0)),
        })
    }

    pub fn resources(&self) -> RaaDelegationResources {
        self.resources
    }

    pub fn binding(&self) -> [u8; 32] {
        self.binding
    }

    pub fn issue(&self, input: &[i8]) -> Result<RaaMaskedQuery, String> {
        self.issued
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < CODED_LINEAR_MAX_CLAIMS).then_some(count + 1)
            })
            .map_err(|_| "RAA reference issuance budget exhausted".to_string())?;
        if input.len() != self.columns {
            return Err("RAA reference input shape mismatch".into());
        }
        self.live_queries
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < RAA_MAX_LIVE_QUERIES).then_some(count + 1)
            })
            .map_err(|_| "RAA reference live-query budget exhausted".to_string())?;
        let lease = QueryLease(Arc::clone(&self.live_queries));
        let sparse = SparseChallenge::sample(self.columns * 8, self.resources.privacy_sparsity)?;
        let mask = Zeroizing::new(self.privacy_code.syndrome(&sparse.entries)?);
        let masked_input = Zeroizing::new(
            input
                .iter()
                .zip(mask.iter())
                .map(|(&x, &r)| field_add(field_from_signed(i64::from(x)), r))
                .collect(),
        );
        let correction = Zeroizing::new(
            self.privacy_matrix
                .chunks_exact(self.columns * 8)
                .map(|row| {
                    sparse.entries.iter().fold(0, |sum, &(position, value)| {
                        field_add(sum, field_mul(row[position], value))
                    })
                })
                .collect(),
        );
        Ok(RaaMaskedQuery {
            binding: self.binding,
            masked_input,
            correction,
            _lease: lease,
        })
    }

    /// The response is fixed before fresh secret verification challenges exist.
    /// Both valid and invalid responses consume the query. A conditional pass
    /// does not certify RAA distance or establish the paper's privacy assumption.
    pub fn finish(&self, query: RaaMaskedQuery, claimed: &[u32]) -> Result<Vec<i32>, String> {
        if query.binding != self.binding {
            return Err("RAA reference query belongs to another matrix/code binding".into());
        }
        if claimed.len() != self.rows || claimed.iter().any(|&v| v >= CODED_LINEAR_FIELD) {
            return Err("RAA reference response shape or field element is invalid".into());
        }
        let encoded_output = self.verification_code.encode(claimed)?;
        let mut mismatch = 0u32;
        for _ in 0..CODED_LINEAR_REPETITIONS {
            let challenge = SparseChallenge::sample(self.rows * 8, CODED_LINEAR_CHALLENGE_WEIGHT)?;
            let mut projected_row = Zeroizing::new(vec![0u32; self.columns]);
            let mut right = 0;
            for &(position, value) in &challenge.entries {
                right = field_add(right, field_mul(encoded_output[position], value));
                for (entry, &q) in projected_row
                    .iter_mut()
                    .zip(&self.verification_matrix[position * self.columns..][..self.columns])
                {
                    *entry = field_add(*entry, field_mul(q, value));
                }
            }
            let left = projected_row
                .iter()
                .zip(query.masked_input.iter())
                .fold(0, |sum, (&w, &x)| field_add(sum, field_mul(w, x)));
            mismatch |= left ^ right;
        }
        if mismatch != 0 {
            return Err("RAA reference coded check rejected response".into());
        }
        let mut result = Zeroizing::new(Vec::with_capacity(self.rows));
        for (&value, &mask) in claimed.iter().zip(query.correction.iter()) {
            let residue = field_add(value, field_neg(mask));
            let centered = if residue > CODED_LINEAR_FIELD / 2 {
                i64::from(residue) - i64::from(CODED_LINEAR_FIELD)
            } else {
                i64::from(residue)
            };
            if centered.unsigned_abs() > (self.columns * 128 * 128) as u64 {
                return Err("RAA reference decoded result exceeds signed-i8 product bound".into());
            }
            result.push(centered as i32);
        }
        Ok(std::mem::take(&mut *result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Executor, Matrix};

    fn fixture(rows: usize, cols: usize, seed: u8) -> (Matrix, RaaDelegationReference) {
        let weights = (0..rows * cols)
            .map(|i| (i * 31 + i / cols * 7) as u8)
            .collect::<Vec<_>>();
        let matrix = Matrix::new(&weights, rows, cols).unwrap();
        let reference =
            RaaDelegationReference::preprocess(matrix.as_signed_slice(), rows, cols, [seed; 32])
                .unwrap();
        (matrix, reference)
    }

    #[test]
    fn full_width_queries_match_independent_signed_oracle_with_fresh_masks() {
        let (rows, cols) = (64, 1024);
        let (matrix, reference) = fixture(rows, cols, 37);
        let executor = Executor::new(1, true).unwrap();
        for input in [
            vec![-128; cols],
            vec![127; cols],
            (0..cols).map(|i| i as i8).collect(),
        ] {
            let expected = matrix
                .as_signed_slice()
                .chunks_exact(cols)
                .map(|row| {
                    row.iter()
                        .zip(&input)
                        .map(|(&w, &x)| i32::from(w) * i32::from(x))
                        .sum::<i32>()
                })
                .collect::<Vec<_>>();
            let first = reference.issue(&input).unwrap();
            let second = reference.issue(&input).unwrap();
            assert_ne!(first.masked_input(), second.masked_input());
            for query in [first, second] {
                let claimed = matrix
                    .modular(&executor, query.masked_input(), 1, CODED_LINEAR_FIELD)
                    .unwrap();
                assert_eq!(reference.finish(query, &claimed).unwrap(), expected);
            }
        }
    }

    #[test]
    fn rejects_forgery_wrong_binding_and_malformed_claims() {
        let (matrix, reference) = fixture(64, 64, 7);
        let (_, other) = fixture(64, 64, 8);
        let executor = Executor::new(1, true).unwrap();
        for row in 0..64 {
            let query = reference.issue(&[127; 64]).unwrap();
            let mut claimed = matrix
                .modular(&executor, query.masked_input(), 1, CODED_LINEAR_FIELD)
                .unwrap();
            claimed[row] = field_add(claimed[row], 1);
            assert!(reference
                .finish(query, &claimed)
                .unwrap_err()
                .contains("coded check"));
        }
        let query = reference.issue(&[0; 64]).unwrap();
        assert!(other
            .finish(query, &[0; 64])
            .unwrap_err()
            .contains("binding"));
        let query = reference.issue(&[0; 64]).unwrap();
        assert!(reference.finish(query, &[CODED_LINEAR_FIELD; 64]).is_err());
        let query = reference.issue(&[0; 64]).unwrap();
        assert!(reference.finish(query, &[]).is_err());
        assert_eq!(reference.live_queries.load(Ordering::Acquire), 0);
    }

    #[test]
    fn allocation_and_issuance_limits_fail_before_material_creation() {
        // Empty weights prove that full shape admission runs before weight import.
        for (rows, cols) in [(9728, 896), (896, 4864), (16384, 16384)] {
            assert!(RaaDelegationReference::preprocess(&[], rows, cols, [0; 32])
                .err()
                .unwrap()
                .contains("256 MiB"));
        }
        assert!(RaaDelegationResources::for_shape(usize::MAX, 64).is_err());
        let (_, reference) = fixture(64, 64, 2);
        reference
            .issued
            .store(CODED_LINEAR_MAX_CLAIMS - 1, Ordering::Release);
        assert!(reference.issue(&[]).is_err());
        assert!(reference.issue(&[0; 64]).err().unwrap().contains("budget"));
        let (_, reference) = fixture(64, 64, 3);
        reference
            .issued
            .store(CODED_LINEAR_MAX_CLAIMS - 1, Ordering::Release);
        drop(reference.issue(&[0; 64]).unwrap());
        assert!(reference.issue(&[0; 64]).err().unwrap().contains("budget"));
        let (_, reference) = fixture(64, 64, 4);
        let mut live = (0..RAA_MAX_LIVE_QUERIES)
            .map(|_| reference.issue(&[0; 64]).unwrap())
            .collect::<Vec<_>>();
        assert!(reference
            .issue(&[0; 64])
            .err()
            .unwrap()
            .contains("live-query"));
        drop(live.pop());
        drop(reference.issue(&[0; 64]).unwrap());
        assert_eq!(
            reference.live_queries.load(Ordering::Acquire),
            RAA_MAX_LIVE_QUERIES - 1
        );
    }

    #[test]
    fn sparse_sampling_has_exact_weight_and_canonical_nonzero_values() {
        let mut supports = std::collections::BTreeSet::new();
        for _ in 0..64 {
            let sparse = SparseChallenge::sample(16, 5).unwrap();
            let support = sparse
                .entries
                .iter()
                .map(|&(p, _)| p)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(support.len(), 5);
            assert!(sparse
                .entries
                .iter()
                .all(|&(p, v)| p < 16 && v > 0 && v < CODED_LINEAR_FIELD));
            supports.insert(support);
        }
        assert!(supports.len() > 1);
        assert!(SparseChallenge::sample(16, 17).is_err());
    }
}
