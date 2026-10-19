//! Exact frozen descriptor adapter. Public construction and private compilation
//! have separate reservations; evaluator owners never borrow a dealer plan.

use super::{
    Budget, Context, DealerPartition, Error, Lease, LookupBackend, Party, Resources, RingTable,
};
use crate::fusefss::{self, Descriptor};

/// Trusted-dealer owner for the admitted source-derived GELU scalar only.
/// It does not implement source graph ranges or an authenticated dealer role.
pub struct CompiledGelu<'a> {
    // Drop public allocations before returning their logical reservation.
    descriptor: Descriptor,
    public_table: Option<RingTable>,
    source_digest: [u8; 32],
    lease: Lease<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct CompiledResources {
    pub descriptor_reservation_bytes: usize,
    pub issuance: Resources,
    pub combined_reservation_bytes: usize,
}

fn compiler_error(error: fusefss::Error) -> Error {
    match error {
        fusefss::Error::Allocation => Error::Allocation,
        fusefss::Error::UnsupportedProfile | fusefss::Error::UnsupportedPhi => Error::Unsupported,
        _ => Error::Domain,
    }
}

impl<'a> CompiledGelu<'a> {
    /// Reserve before constructing the fixed public descriptor. Generic caller
    /// descriptors are deliberately not accepted by this bounded adapter.
    pub fn new(budget: &'a Budget) -> Result<Self, Error> {
        Self::with_backend(budget, LookupBackend::PrivateVector)
    }

    /// Select the private-vector control or public-original-boundary reuse.
    /// Both are derived from the same frozen descriptor; public reuse never
    /// publishes a mask-shifted partition or the selected coefficients.
    pub fn with_backend(budget: &'a Budget, backend: LookupBackend) -> Result<Self, Error> {
        let lease = budget.reserve(super::COMPILER_OVERLAP_BYTES)?;
        let descriptor = Descriptor::gelu().map_err(compiler_error)?;
        let public_table = match backend {
            LookupBackend::PrivateVector => None,
            LookupBackend::PublicBoundary => {
                // Zero rotation is public and input-independent. The fixed
                // first-eligible duplicate retains the compiler's padded shape.
                let public = descriptor.compile(0).map_err(compiler_error)?;
                Some(
                    public
                        .with_dealer_handoff(|h| {
                            Ok(RingTable::new(
                                h.word.bits,
                                h.shifted_boundaries,
                                h.coefficients,
                                super::COMPILER_OVERLAP_BYTES,
                            ))
                        })
                        .map_err(compiler_error)?
                        .map_err(super::primitive_error)?,
                )
            }
        };
        let mut source_digest = [0; 32];
        for (i, byte) in source_digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&fusefss::SOURCE_LOCK_SHA256[2 * i..2 * i + 2], 16)
                .map_err(|_| Error::Domain)?;
        }
        Ok(Self {
            descriptor,
            public_table,
            source_digest,
            lease,
        })
    }

    pub fn descriptor_digest(&self) -> [u8; 32] {
        self.descriptor.digest()
    }

    fn spec(&self, lanes: usize) -> super::Spec {
        super::Spec {
            bits: self.descriptor.word().bits,
            source_intervals: self.descriptor.shape().source_intervals,
            lanes,
            descriptor_digest: self.descriptor.digest(),
            source_digest: self.source_digest,
            boolean_outputs: self.descriptor.shape().boolean_outputs,
            identity_phi: true,
        }
    }

    /// Public descriptor ownership remains charged in addition to each concurrent
    /// issuance's compiler scratch, keys, state, frames and outputs.
    pub fn resources(&self, lanes: usize) -> Result<CompiledResources, Error> {
        let issuance = super::resources_for(self.spec(lanes), self.public_table.as_ref())?;
        let combined_reservation_bytes = self
            .lease
            .bytes
            .checked_add(issuance.combined_reservation_bytes)
            .filter(|bytes| *bytes <= super::MAX_ALLOCATION_BYTES)
            .ok_or(Error::Resource)?;
        Ok(CompiledResources {
            descriptor_reservation_bytes: self.lease.bytes,
            issuance,
            combined_reservation_bytes,
        })
    }

    /// Every lane receives a fresh mask, FSS keys and Beaver correlation. The
    /// original shared x enters Horner. There is no post-product Q12 shift.
    pub fn issue(&self, lanes: usize, context: Context) -> Result<[Party<'a>; 2], Error> {
        self.resources(lanes)?;
        if let Some(table) = &self.public_table {
            return super::issue_public(self.spec(lanes), context, self.lease.budget, table);
        }
        super::issue(self.spec(lanes), context, self.lease.budget, |_, mask| {
            let plan = self.descriptor.compile(mask).map_err(compiler_error)?;
            plan.with_dealer_handoff(|handoff| {
                if handoff.descriptor_digest != self.descriptor.digest()
                    || handoff.word != self.descriptor.word()
                    || handoff.shape != self.descriptor.shape()
                    || !handoff.predicate_queries.is_empty()
                    || !handoff.boolean_program.is_empty()
                {
                    return Err(fusefss::Error::Shape);
                }
                Ok(DealerPartition::from_slices(
                    mask,
                    handoff.shifted_boundaries,
                    handoff.coefficients,
                ))
            })
            .map_err(compiler_error)?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    const MASK: u64 = (1u64 << 50) - 1;

    fn context(index: u64) -> Context {
        Context {
            plan: Sha256::digest(b"fusefss-native-compiled-gelu-test").into(),
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: index,
        }
    }

    #[test]
    fn frozen_gelu_vectors_use_actual_compiler_and_fresh_protected_owners() {
        let vectors: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../paper/reproduction-contracts/fusefss/vectors.json"
        ))
        .unwrap();
        let mut inputs: Vec<i64> = vectors["gelu_inputs_signed_q12"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        let mut expected: Vec<i64> = vectors["gelu_expected_signed_q12"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        // Distinguish the admitted mathematical signed50 extension from an
        // unproved signed38 narrowing. Graph provenance is not asserted here.
        inputs.push((1i64 << 37) + 64);
        expected.push((1i64 << 37) + 64);
        let budget = Budget::new(super::super::MAX_ALLOCATION_BYTES).unwrap();
        for backend in [LookupBackend::PrivateVector, LookupBackend::PublicBoundary] {
            let gate = CompiledGelu::with_backend(&budget, backend).unwrap();
            let resident = budget.used_bytes();
            for (chunk_index, (input, expected)) in
                inputs.chunks(4).zip(expected.chunks(4)).enumerate()
            {
                let resources = gate.resources(input.len()).unwrap();
                assert_eq!(resources.issuance.padded_intervals, 512);
                assert_eq!(resources.issuance.lookup_backend, backend);
                let [mut a, mut b] = gate
                    .issue(input.len(), context(4 * chunk_index as u64))
                    .unwrap();
                let left: Vec<u64> = input
                    .iter()
                    .map(|_| super::super::random().unwrap() & MASK)
                    .collect();
                let right: Vec<u64> = input
                    .iter()
                    .zip(&left)
                    .map(|(x, share)| (*x as u64).wrapping_sub(*share) & MASK)
                    .collect();
                let af = a.start(&left).unwrap();
                let bf = b.start(&right).unwrap();
                let ap = a.advance(bf.as_bytes()).unwrap();
                let bp = b.advance(af.as_bytes()).unwrap();
                assert_eq!(
                    af.as_bytes().len()
                        + bf.as_bytes().len()
                        + ap.as_bytes().len()
                        + bp.as_bytes().len(),
                    resources.issuance.total_peer_frame_bytes
                );
                let ao = a.finish(bp.as_bytes()).unwrap();
                let bo = b.finish(ap.as_bytes()).unwrap();
                for ((a, b), expected) in ao.shares().iter().zip(bo.shares()).zip(expected) {
                    assert_eq!(a.wrapping_add(*b) & MASK, *expected as u64 & MASK);
                }
                assert!(a.is_consumed() && b.is_consumed());
                assert!(matches!(a.start(&left), Err(Error::Consumed)));
                drop((a, b, af, bf, ap, bp, ao, bo));
                assert_eq!(budget.used_bytes(), resident);
            }
            drop(gate);
            assert_eq!(budget.used_bytes(), 0);
        }
    }

    #[test]
    fn descriptor_and_exported_frames_have_independent_budget_lifetimes() {
        for backend in [LookupBackend::PrivateVector, LookupBackend::PublicBoundary] {
            let budget = Budget::new(super::super::MAX_ALLOCATION_BYTES).unwrap();
            let gate = CompiledGelu::with_backend(&budget, backend).unwrap();
            let resources = gate.resources(1).unwrap();
            let [mut a, mut b] = gate.issue(1, context(0)).unwrap();
            let frame = a.start(&[0]).unwrap();
            b.cancel();
            drop((gate, a, b));
            assert_eq!(budget.used_bytes(), resources.issuance.masked_frame_bytes);
            drop(frame);
            assert_eq!(budget.used_bytes(), 0);

            let too_small = Budget::new(super::super::COMPILER_OVERLAP_BYTES - 1).unwrap();
            assert!(matches!(
                CompiledGelu::new(&too_small),
                Err(Error::Resource)
            ));
            assert_eq!(too_small.used_bytes(), 0);
            let descriptor_only = Budget::new(resources.descriptor_reservation_bytes).unwrap();
            let gate = CompiledGelu::with_backend(&descriptor_only, backend).unwrap();
            assert!(matches!(gate.issue(1, context(0)), Err(Error::Resource)));
            assert_eq!(
                descriptor_only.used_bytes(),
                resources.descriptor_reservation_bytes
            );
        }
    }

    #[test]
    fn lookup_backend_binding_rejects_cross_backend_frames_after_nonce_control() {
        let budget = Budget::new(super::super::MAX_ALLOCATION_BYTES).unwrap();
        let private = CompiledGelu::new(&budget).unwrap();
        let public = CompiledGelu::with_backend(&budget, LookupBackend::PublicBoundary).unwrap();
        let [mut a, _b] = private.issue(1, context(0)).unwrap();
        let [_c, mut d] = public.issue(1, context(0)).unwrap();
        d.nonce = a.nonce;
        let _own = a.start(&[0]).unwrap();
        let foreign = d.start(&[64]).unwrap();
        assert!(matches!(a.advance(foreign.as_bytes()), Err(Error::Context)));
        assert!(a.is_consumed());
        let cost = public.resources(1).unwrap().issuance;
        assert_eq!(cost.dcf_keys_per_party, 1);
        assert_eq!(cost.dcf_evaluations_per_party, 512);
        assert_eq!(cost.party_key_logical_bytes, 1372);
        assert_eq!(cost.public_table_bytes_per_party, 12296);
        assert_eq!(cost.public_source_table_bytes, 12296);
    }
}
