//! Fixed PAPER inverse table, certified before hashing or native issuance.
use super::*;
use crate::shared_allocation::ReserveErrorMap;

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
struct SquareWitness<'d> {
    d: &'d u128,
}
impl SquareWitness<'_> {
    fn checked(&self, a: u128) -> Result<u128, Error> {
        let first = a.checked_mul(a);
        let product = match first {
            Some(v) => v.checked_mul(*self.d),
            None => None,
        };
        product.ok_or(Error::Resource)
    }
}

type AddChecked = fn(usize, usize) -> Result<usize, Error>;
fn add_checked(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}

// Logical source/move inventories only: no extra runtime record initialization,
// closure-layout equivalence, stack-slot reuse or destination/overlap proof.
#[allow(dead_code)]
struct SquareFrame {
    argument: u128,
    first: Option<u128>,
    extracted: u128,
    product: Option<u128>,
    mapped: Result<u128, Error>,
}
#[allow(dead_code)]
struct AddFrame {
    a: usize,
    b: usize,
    option: Option<usize>,
    returned: Result<usize, Error>,
}
#[allow(dead_code)]
struct CoreQuoteFrame {
    payload_option: Option<usize>,
    payload_result: Result<usize, Error>,
    terms: [usize; 5],
    iterator: std::array::IntoIter<usize, 5>,
    callback: AddChecked,
    accumulator: usize,
    item: usize,
    callback_frame: AddFrame,
    returned: Result<usize, Error>,
}
#[allow(dead_code)]
struct CoreNamedControls<'d> {
    square: SquareWitness<'d>,
    lower: SquareFrame,
    upper: SquareFrame,
    quote: CoreQuoteFrame,
}
type ReserveResult = Result<(), std::collections::TryReserveError>;
type AllocationResult = Result<Vec<u64>, Error>;
#[allow(dead_code)]
struct AllocationControls {
    capacity: usize,
    permission: Result<(), Error>,
    reserved: ReserveResult,
    mapper: ReserveErrorMap,
    mapped: Result<(), Error>,
    exact_capacity_argument: usize,
    observed_capacity: usize,
    returned: AllocationResult,
}
#[allow(dead_code)]
struct DenominatorControls {
    m: usize,
    e: usize,
    shifted: Option<u128>,
    returned: Result<u128, Error>,
}
#[allow(dead_code)]
struct EntryFormals {
    m: usize,
    e: usize,
}
#[allow(dead_code)]
struct CertifyFormals {
    value: u64,
    d: u128,
}
#[allow(dead_code)]
struct ValueHashControls<'v> {
    iterator: core::slice::Iter<'v, u64>,
}
#[allow(dead_code)]
struct FillControls {
    outer: core::ops::Range<usize>,
    inner: core::ops::Range<usize>,
    m: usize,
    e: usize,
    entry_result: Result<u64, Error>,
}
#[allow(dead_code)]
struct SearchControls {
    iterator: core::ops::Range<i32>,
    denominator: u128,
    quotient: u128,
    low: u64,
    high: u64,
    middle: u64,
    certificate_result: Result<(), Error>,
}

// LN4 subset only, not the deferred shared CoreControls/Premises/pair/facade.
// Independent lifetimes describe certificate and value-iterator activations;
// no self-referential runtime Controls value is constructed.
#[allow(dead_code)]
struct Controls<'d, 'v, C: AllocationContext> {
    value: Option<CertifiedTableFor<C>>,
    result: Result<CertifiedTableFor<C>, Error>,
    words: [u128; 8],
    hash: Sha256,
    word: [u8; 8],
    vector: Vec<u64>,
    digest: sha2::digest::Output<Sha256>,
    lease: Option<LeaseFor<C>>,
    named: CoreNamedControls<'d>,
    allocation: AllocationControls,
    denominator_call: DenominatorControls,
    entry_formals: EntryFormals,
    certify_formals: CertifyFormals,
    value_hash: ValueHashControls<'v>,
    fill: FillControls,
    search: SearchControls,
    constructed_capacity: usize,
}
/// Table-only admission, counted ONCE. Source/Arc/parent/native evaluator
/// tables remain additional charges in the owning region's aggregate.
#[cfg_attr(not(test), allow(dead_code))] // Table-only quote pending full parent aggregate.
pub(super) fn reservation_bytes<C: AllocationContext>() -> Result<usize, Error> {
    let terms = [
        8usize.checked_mul(ENTRIES).ok_or(Error::Resource)?,
        64,
        size_of::<CertifiedTableFor<C>>(),
        size_of::<Controls<'_, '_, C>>(),
        C::TRANSIENT_LINK_BYTES,
    ];
    let add: AddChecked = add_checked;
    terms.into_iter().try_fold(0usize, add)
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
    let square = SquareWitness { d: &d };
    if square.checked(value as u128)? > NUMERATOR
        || square.checked(value as u128 + 1)? <= NUMERATOR
    {
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
mod layout_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_square_preserves_overflow_and_refusal_prefixes() {
        let zero = 0;
        let square = SquareWitness { d: &zero };
        // First overflow must not be hidden by the zero second operand.
        assert!(matches!(square.checked(u128::MAX), Err(Error::Resource)));
        assert_eq!(square.checked(1).unwrap(), 0);
        let maximum = u128::MAX;
        let square = SquareWitness { d: &maximum };
        assert_eq!(square.checked(1).unwrap(), maximum);
        assert!(matches!(square.checked(2), Err(Error::Resource)));
        // Lower predicate refuses before the upper multiplication would overflow.
        assert!(matches!(certify(1, maximum), Err(Error::Domain)));
        assert!(matches!(certify(2, maximum), Err(Error::Resource)));
        assert!(matches!(certify(0, 0), Err(Error::Domain)));
        assert!(certify(0, maximum).is_ok());
        assert!(matches!(denominator(128, 0), Err(Error::Domain)));
        assert!(matches!(denominator(0, 64), Err(Error::Domain)));
    }

    #[test]
    fn named_add_checked_preserves_resource_overflow() {
        let add: AddChecked = add_checked;
        assert_eq!(add(usize::MAX, 0).unwrap(), usize::MAX);
        assert!(matches!(add(usize::MAX, 1), Err(Error::Resource)));
    }

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
