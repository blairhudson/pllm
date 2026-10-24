//! Fixed PAPER inverse table, certified before hashing or native issuance.
use super::*;

#[cfg_attr(not(test), allow(dead_code))] // Private fixed table only issued by qualification until parent completion.
pub(super) const ENTRIES: usize = 8192;
#[cfg_attr(not(test), allow(dead_code))] // Exact certified integer numerator, not a public profile.
const NUMERATOR: u128 = 1 << 57;

/// Certified original fixed table; NOT a source/profile/replay capability.
/// No constructor adopts caller table bytes or a claimed digest.
#[cfg_attr(not(test), allow(dead_code))] // Genuine owned table/control retained by the private native path.
pub(in crate::masked_wire_reference) struct CertifiedTableFor<C: AllocationContext> {
    values: Zeroizing<Vec<u64>>,
    digest: [u8; 32],
    lease: LeaseFor<C>,
}
#[cfg_attr(not(test), allow(dead_code))] // Closed table loans; no caller table/digest adoption.
impl<C: AllocationContext> CertifiedTableFor<C> {
    pub(in crate::masked_wire_reference) fn values(&self) -> &[u64] {
        &self.values
    }
    pub(in crate::masked_wire_reference) fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub(in crate::masked_wire_reference) fn budget(&self) -> &Budget {
        self.lease.budget.budget()
    }
}
#[allow(dead_code)]
struct Controls<C: AllocationContext> {
    value: Option<CertifiedTableFor<C>>,
    result: Result<CertifiedTableFor<C>, Error>,
    words: [u128; 8],
    hash: Sha256,
    word: [u8; 8],
    vector: Vec<u64>,
    digest: sha2::digest::Output<Sha256>,
    lease: Option<LeaseFor<C>>,
}
/// Table-only admission, counted ONCE. Source/Arc/parent/native evaluator
/// tables remain additional charges in the owning region's aggregate.
#[cfg_attr(not(test), allow(dead_code))] // Table-only quote pending full parent aggregate.
pub(super) fn reservation_bytes<C: AllocationContext>() -> Result<usize, Error> {
    [
        8usize.checked_mul(ENTRIES).ok_or(Error::Resource)?,
        64,
        size_of::<CertifiedTableFor<C>>(),
        size_of::<Controls<C>>(),
        C::TRANSIENT_LINK_BYTES,
    ]
    .into_iter()
    .try_fold(0usize, |a, b| a.checked_add(b).ok_or(Error::Resource))
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // Actual future interval metadata, not an active scanner or signed certificate.
pub(super) struct IntervalEntry {
    pub start: u64,
    pub end: u64,
    pub exponent: u64,
    pub multiplier: u64,
}

#[cfg_attr(not(test), allow(dead_code))] // Fixed interval inventory; paired scanner not yet issued.
pub(super) fn intervals() -> [IntervalEntry; 51] {
    std::array::from_fn(|i| {
        if i == 0 {
            IntervalEntry {
                start: 0,
                end: 1,
                exponent: 0,
                multiplier: 0,
            }
        } else {
            let e = i - 1;
            IntervalEntry {
                start: 1 << e,
                end: 1 << (e + 1),
                exponent: e as u64,
                multiplier: 1 << (49 - e),
            }
        }
    })
}

#[cfg_attr(not(test), allow(dead_code))] // Shared fixed-table certification arithmetic; private issuer only.
fn denominator(m: usize, e: usize) -> Result<u128, Error> {
    if m >= 128 || e >= 64 {
        return Err(Error::Domain);
    }
    (128u128 + m as u128)
        .checked_shl(e as u32)
        .ok_or(Error::Resource)
}

#[cfg_attr(not(test), allow(dead_code))] // Certificate checks actual owned entries before native issuance.
fn certify(value: u64, d: u128) -> Result<(), Error> {
    let square = |a: u128| {
        a.checked_mul(a)
            .and_then(|v| v.checked_mul(d))
            .ok_or(Error::Resource)
    };
    if square(value as u128)? > NUMERATOR || square(value as u128 + 1)? <= NUMERATOR {
        return Err(Error::Domain);
    }
    Ok(())
}

/// Exactly 26 iterations even when the entry is zero. The certificate checks
/// the integer rational directly; neither floating sqrt nor rounded division
/// supplies a premise. 2^25 is the largest possible result.
#[cfg_attr(not(test), allow(dead_code))] // Fixed certified entry function, not a general rsqrt API.
pub(super) fn entry(m: usize, e: usize) -> Result<u64, Error> {
    let d = denominator(m, e)?;
    let quotient = NUMERATOR / d;
    let (mut low, mut high) = (0u64, (1u64 << 25) + 1);
    for _ in 0..26 {
        let middle = low + (high - low) / 2;
        if u128::from(middle) * u128::from(middle) <= quotient {
            low = middle;
        } else {
            high = middle;
        }
    }
    certify(low, d)?;
    Ok(low)
}

/// Caller has reserved actual Vec/certification/move controls on the ORIGINAL
/// counter. No source authority derives from these public table bytes.
#[cfg_attr(not(test), allow(dead_code))] // Real charged private constructor; production parent still unavailable.
pub(super) fn construct<C: AllocationContext>(
    lease: LeaseFor<C>,
) -> Result<CertifiedTableFor<C>, Error> {
    if lease.bytes != reservation_bytes::<C>()? {
        return Err(Error::Resource);
    }
    let budget = lease.budget.budget();
    let mut values = Zeroizing::new(allocate(ENTRIES, budget)?);
    if values.capacity() != ENTRIES {
        return Err(Error::Resource);
    }
    for m in 0..128 {
        for e in 0..64 {
            values.push(entry(m, e)?);
        }
    }
    // entry certified EVERY owned entry before this first hash operation. No
    // duplicate certification pass hidden outside the fixed work inventory.
    let mut hash = Sha256::new();
    for value in values.iter() {
        hash.update(value.to_le_bytes());
    }
    Ok(CertifiedTableFor {
        values,
        digest: hash.finalize().into(),
        lease,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // Independent restoring square-root algorithm, NOT the production search.
    fn reference(mut n: u128) -> u128 {
        let mut bit = 1u128 << 126;
        while bit > n {
            bit >>= 2;
        }
        let mut root = 0;
        while bit != 0 {
            if n >= root + bit {
                n -= root + bit;
                root = (root >> 1) + bit;
            } else {
                root >>= 1;
            }
            bit >>= 2;
        }
        root
    }
    #[test]
    fn all_8192_entries_match_independent_integer_reference() {
        for m in 0..128 {
            for e in 0..64 {
                assert_eq!(
                    u128::from(entry(m, e).unwrap()),
                    reference((1u128 << 57) / ((128u128 + m as u128) << e))
                );
            }
        }
        assert_eq!(entry(0, 0).unwrap(), 1 << 25);
        assert_eq!(entry(127, 63).unwrap(), 0);
    }
    #[test]
    fn fixed_intervals_cover_ring_and_exact_exponent_boundaries() {
        let bins = intervals();
        assert_eq!(bins[0].start, 0);
        assert_eq!(bins[50].end, 1 << 50);
        for i in 1..51 {
            assert_eq!(bins[i - 1].end, bins[i].start);
            assert_eq!(bins[i].multiplier, 1 << (49 - bins[i].exponent));
            for s in [bins[i].start, bins[i].end - 1] {
                let t = u128::from(s) * u128::from(bins[i].multiplier);
                assert!(t < 1u128 << 50);
                let q = (t >> 42) as u64;
                assert_eq!(
                    q & 127,
                    (((128u128 * u128::from(s)) >> bins[i].exponent) - 128) as u64
                );
            }
        }
    }
}
