//! Thin private-payload vector adapter over shared compact_dcf. No PRG,
//! correction, group arithmetic or traversal is implemented in this module.
//! Generic-control material only; not a full-model execution default.
use super::{primitive_error, Budget, Error};
use crate::{
    compact_dcf::{Resources, VectorComparison},
    shared_rescale_reference::Error as PrimitiveError,
};

#[repr(transparent)]
pub(super) struct Key {
    inner: VectorComparison,
}
impl Key {
    pub(super) fn write_material(&self, out: &mut Vec<u8>) {
        self.inner.write_material(out);
    }
    pub(super) fn read_material(bytes: &[u8], bits: u8, budget: &Budget) -> Result<Self, Error> {
        Ok(Self {
            inner: VectorComparison::read_material(bytes, bits, bits, budget)?,
        })
    }
    pub(super) fn logical_bytes(bits: u8) -> usize {
        32 + 34 * usize::from(bits)
    }
    /// Source-managed owner/capacity/scratch sizes. Parent must use these helpers
    /// instead of depending on shared correction representation/layout.
    pub(super) fn resources(bits: u8) -> Result<Resources, Error> {
        VectorComparison::resources(bits, bits).map_err(primitive_error)
    }
    pub(super) fn issue(
        bits: u8,
        threshold: u64,
        payload: [u64; 2],
        budget: &Budget,
    ) -> Result<[Self; 2], Error> {
        let cost = Self::resources(bits)?;
        let mut allocation_failure = None;
        let result = VectorComparison::issue_with_reservation(
            bits,
            bits,
            threshold,
            payload,
            cost.combined_allocation_bytes,
            |_| {
                budget.allocation().map_err(|error| {
                    allocation_failure = Some(error);
                    PrimitiveError::Resource
                })
            },
        );
        result
            .map(|keys| keys.map(|inner| Self { inner }))
            .map_err(|error| allocation_failure.unwrap_or_else(|| primitive_error(error)))
    }
    pub(super) fn eval(&self, party: u8, input: u64) -> Result<[u64; 2], Error> {
        self.inner.eval(party, input).map_err(primitive_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_vector_exhaustive_and_all_width_endpoints() {
        let budget = Budget::new(super::super::MAX_ALLOCATION_BYTES).unwrap();
        for bits in 1..=50 {
            let mask = (1u64 << bits) - 1;
            let thresholds: Vec<_> = if bits <= 5 {
                (0..=mask).collect()
            } else {
                vec![0, 1, mask / 2, mask]
            };
            for threshold in thresholds {
                for payload in [[0, 0], [1, mask], [mask, mask / 2]] {
                    let [a, b] = Key::issue(bits, threshold, payload, &budget).unwrap();
                    let inputs: Vec<_> = if bits <= 5 {
                        (0..=mask).collect()
                    } else {
                        vec![
                            0,
                            1,
                            threshold.wrapping_sub(1) & mask,
                            threshold,
                            (threshold + 1) & mask,
                            mask,
                        ]
                    };
                    for x in inputs {
                        let aa = a.eval(0, x).unwrap();
                        let bb = b.eval(1, x).unwrap();
                        for j in 0..2 {
                            assert_eq!(
                                (aa[j] + bb[j]) & mask,
                                if x < threshold { payload[j] } else { 0 }
                            );
                        }
                    }
                    let cost = Key::resources(bits).unwrap();
                    assert_eq!(cost.encoded_key_bytes, Key::logical_bytes(bits));
                    assert_eq!(a.inner.allocation(), cost);
                    assert_eq!(std::mem::size_of::<Key>(), cost.key_owner_bytes);
                }
            }
        }
    }
    #[test]
    fn shared_allocator_failpoints_and_checked_eval() {
        let budget = Budget::new(super::super::MAX_ALLOCATION_BYTES).unwrap();
        for fail in 0..2 {
            budget
                .fail_after
                .store(fail, std::sync::atomic::Ordering::SeqCst);
            assert!(matches!(
                Key::issue(50, 0, [0, 1], &budget),
                Err(Error::Allocation)
            ));
        }
        budget
            .fail_after
            .store(-1, std::sync::atomic::Ordering::SeqCst);
        let [a, _] = Key::issue(50, 0, [0, 1], &budget).unwrap();
        assert_eq!(a.eval(2, 0), Err(Error::Domain));
        assert_eq!(a.eval(0, 1 << 50), Err(Error::Domain));
    }
}
