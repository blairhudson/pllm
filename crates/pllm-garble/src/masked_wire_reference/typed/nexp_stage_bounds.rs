//! Fixed D29 clear-root numerical preflight. Mount `mod nexp_stage_bounds;` in
//! typed.rs; only that parent and its descendants can consume these functions.
//!
//! No table owner/copy/cache, private construction, allocation, hash, entropy or
//! new proof authority. Verified means ALL512 exact literal floors succeeded;
//! it does not authenticate source/profile, prove privacy or issue Budget. The
//! actual clear-root slice is checked by the existing D29 preflight, not inferred
//! from a uniform mask. Caller must still enforce original Binding/counter/lease
//! identity, source qualification, canonical region records and both-role burn.
//!
//! Half-open true-value bounds at the18 reads:
//!  0,1: X signed50 Q12 [0,2^38); 2,5: X40 signed40 Q12 same.
//!  3: A=X-65536 signed40 Q12 [-65536,2^38-65536).
//!  4,6: H=[A>=0], B=H XOR1 are Boolean [0,2).
//!  7,8,9,10: W,V,C unsigned16 Q12 [0,65536).
//!  11,12,13: J1,J0,J1 unsigned8 Q0 [0,256).
//!  14: U0 signed50 Q12 [3848,4097); 15: U1 same [0,4097).
//!  16: P=U0*U1 signed50 Q24 [0,2^24+1).
//!  17: floor(P/4096) signed50 Q12 [0,4097).
//!
//! A is an unwrapped signed40 subtraction. W=(X40-65535) mod2^16,
//! V=B*W mod2^16 and C=(V+65535) mod2^16 are INTENTIONALLY modular,
//! not signed16 no-wrap assertions. X<65536 gives B=1,C=X; otherwise B=0,
//! C=65535. This proves C=min(X,65535) without disclosing a private range.
//! J0=C mod256 and the UNSIGNED TR16.8 J1=floor(C/256) include128..255.
//! Public typed TR would instead require signed8 Q4: it is not this approved
//! coordinate adapter, and no generic relabel is authorized by these bounds.
//!
//! The11 exchanges: H; V; TR carry; J1; E0 signed table opening/correction;
//! reconstructed U0; E1 opening/correction; reconstructed U1; P; Gap carry;
//! reconstructed E. Carry/correction clear values are bits. In either D19 lookup,
//! the mathematical signed table term sigma*T is in [-4096,4097), and selected
//! term is in [0,4097). The source then reconstructs T+rho with rho=2u-r.
//! This is NOT a bound on the opened Rhat/Shat/common masked words: those remain
//! full canonical ring50 words; their independent native validation is unchanged.
//! P is nonnegative and <2^48, so GapARS50.12 has its whole half-open quarter
//! premise. Floor is Euclidean, never nearest; P=2^24 at X=0 is included. Zero
//! E1/P/E is valid; there is no reciprocal/Softmax positivity claim here.

use super::{EdgeDescriptor, Error, Interpretation, Interval, Numeric, Representation};
use crate::fixed_exp_tables::{Table, Verified};
use std::mem::size_of;

pub(super) const READ_COUNT: usize = 18;
pub(super) const TABLE_WORDS: usize = 512;
/// Existing B owner edges: X,X40,A,H,B,W,V,C,J0,J1,U0,U1,P,E.
pub(super) const EDGE_BY_USE: [usize; READ_COUNT] =
    [0, 0, 1, 2, 3, 1, 4, 5, 6, 7, 7, 9, 8, 9, 10, 11, 12, 13];
pub(super) type BindingIntervals = [Option<Interval>; 14];
const INPUT_UPPER: i128 = 1i128 << 38;

/// Plain public numerical metadata, not a private-value range certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ReadBound {
    pub numeric: Numeric,
    pub interval: Interval,
}

/// Exact expected metadata/ranges; no caller-supplied floor/source certificate.
pub(super) fn read_bound(slot: usize) -> Result<ReadBound, Error> {
    use Interpretation::{Predicate, Signed, Unsigned};
    let (bits, fractional, interpretation, lower, upper) = match slot {
        0 | 1 => (50, 12, Signed, 0, INPUT_UPPER),
        2 | 5 => (40, 12, Signed, 0, INPUT_UPPER),
        3 => (40, 12, Signed, -65536, INPUT_UPPER - 65536),
        4 | 6 => (1, 0, Predicate, 0, 2),
        7..=10 => (16, 12, Unsigned, 0, 65536),
        11..=13 => (8, 0, Unsigned, 0, 256),
        14 => (50, 12, Signed, 3848, 4097),
        15 => (50, 12, Signed, 0, 4097),
        16 => (50, 24, Signed, 0, (1i128 << 24) + 1),
        17 => (50, 12, Signed, 0, 4097),
        _ => return Err(Error::Domain),
    };
    let result = ReadBound {
        numeric: Numeric::new(bits, fractional, interpretation).map_err(|_| Error::Domain)?,
        interval: Interval { lower, upper },
    };
    if !fits_numeric(result) {
        return Err(Error::Domain);
    }
    Ok(result)
}

fn fits_numeric(bound: ReadBound) -> bool {
    let interval = bound.interval;
    if interval.lower >= interval.upper {
        return false;
    }
    match bound.numeric.interpretation() {
        Interpretation::Signed => interval.fits(bound.numeric.ring_bits()),
        Interpretation::Unsigned => {
            interval.lower >= 0 && interval.upper <= 1i128 << bound.numeric.ring_bits()
        }
        Interpretation::Predicate => interval.lower >= 0 && interval.upper <= 2,
    }
}

fn quarter_fits(interval: Interval, bits: u8) -> bool {
    if !(3..=64).contains(&bits) || interval.lower >= interval.upper {
        return false;
    }
    let quarter = 1i128 << (bits - 2);
    interval.lower >= -quarter && interval.upper <= quarter
}

fn product_interval(a: Interval, b: Interval) -> Result<Interval, Error> {
    if a.lower < 0 || b.lower < 0 || a.lower >= a.upper || b.lower >= b.upper {
        return Err(Error::Domain);
    }
    Ok(Interval {
        lower: a.lower.checked_mul(b.lower).ok_or(Error::Resource)?,
        upper: (a.upper - 1)
            .checked_mul(b.upper - 1)
            .and_then(|x| x.checked_add(1))
            .ok_or(Error::Resource)?,
    })
}

fn extrema(values: &[u64; 256]) -> Result<Interval, Error> {
    let mut lower = u64::MAX;
    let mut upper = 0u64;
    for &value in values {
        // No silent reduction, saturation, table repair or exp substitution.
        if value > 4096 {
            return Err(Error::Domain);
        }
        lower = lower.min(value);
        upper = upper.max(value);
    }
    Ok(Interval {
        lower: i128::from(lower),
        upper: i128::from(upper).checked_add(1).ok_or(Error::Resource)?,
    })
}

fn certified_ranges(verified: &Verified) -> Result<(), Error> {
    let mut words_checked = 0usize;
    for (table, denominator, slot) in [(Table::LowByte, 12, 14), (Table::HighByte, 4, 15)] {
        let declaration = table.definition();
        if declaration.entries != 256
            || declaration.argument_denominator_bits != denominator
            || declaration.payload_scale != 4096
            || declaration.coordinate_bits != 8
            || declaration.coordinate_fractional_bits != 0
            || declaration.payload_bits != 50
            || declaration.payload_fractional_bits != 12
        {
            return Err(Error::Domain);
        }
        if extrema(verified.values(table))? != read_bound(slot)?.interval {
            return Err(Error::Domain);
        }
        words_checked = words_checked
            .checked_add(verified.values(table).len())
            .ok_or(Error::Resource)?;
    }
    if words_checked != TABLE_WORDS {
        return Err(Error::Domain);
    }
    let product = product_interval(read_bound(14)?.interval, read_bound(15)?.interval)?;
    if product != read_bound(16)?.interval
        || !quarter_fits(product, 50)
        || !product.fits(50)
        || product.shift(12) != read_bound(17)?.interval
    {
        return Err(Error::Domain);
    }
    // Approved private TR16.8 coordinate bridge, NOT public typed TR retagging.
    let c = read_bound(10)?.numeric;
    let j1 = read_bound(11)?.numeric;
    let p = read_bound(16)?.numeric;
    let e = read_bound(17)?.numeric;
    let tr_shift = 8;
    let gap_shift = 12;
    if tr_shift == 0
        || tr_shift >= c.ring_bits()
        || c.ring_bits().checked_sub(tr_shift) != Some(j1.ring_bits())
        || j1.fractional_bits() != 0
        || gap_shift == 0
        || gap_shift >= p.ring_bits() - 1
        || p.ring_bits() != e.ring_bits()
        || p.fractional_bits().checked_sub(gap_shift) != Some(e.fractional_bits())
    {
        return Err(Error::Domain);
    }
    Ok(())
}

/// Source-specific numerical check BEFORE entropy/private construction. Whole
/// values/shape validation is reused, not a new caller premise or inherited root.
/// `reads` borrows existing canonical records; it never clones descriptors/tables.
/// Returns plain intervals for the EXISTING14 bindings, not a new owner/token.
/// U0 retains B's existing conservative [0,4097) binding; its certified true
/// minimum3848 is not used to change existing binding digests/range declarations.
/// The caller can replace its literal interval array, in its already quoted slot,
/// with this result before entropy. No Budget/source/Binding authority is issued.
pub(super) fn preflight(
    input: &EdgeDescriptor,
    values: &[i64],
    verified: &Verified,
    reads: [&EdgeDescriptor; READ_COUNT],
) -> Result<BindingIntervals, Error> {
    super::nexp::validate_nexp_clear_root(input, values)?;
    certified_ranges(verified)?;
    for (slot, edge) in reads.iter().enumerate() {
        let expected = read_bound(slot)?;
        let representation = if expected.numeric.interpretation() == Interpretation::Predicate {
            Representation::CommonXorMasked
        } else {
            Representation::CommonAdditiveMasked
        };
        if edge.numeric() != expected.numeric
            || edge.representation() != representation
            || edge.shape() != input.shape()
            || edge.identity().first_tensor_index() != input.identity().first_tensor_index()
        {
            return Err(Error::Domain);
        }
    }
    // Same declared parent, not coincident numbers/types on a replacement edge.
    // Original runtime Binding/mask/counter identity remains the containing owner.
    if reads[0] != input
        || reads[1] != input
        || reads[2] != reads[5]
        || reads[9] != reads[10]
        || reads[11] != reads[13]
    {
        return Err(Error::Context);
    }
    let mut bindings = [None; 14];
    for (slot, edge) in EDGE_BY_USE.iter().enumerate() {
        bindings[*edge] = Some(read_bound(slot)?.interval);
    }
    bindings[10] = Some(Interval {
        lower: 0,
        upper: 4097,
    });
    Ok(bindings)
}

/// Logical fixed metadata phase slots only; NOT total physical stack/RSS, a
/// native factory peak or a reservation grant. Add to the containing caller's
/// real pre-entropy layout quote (its own records/callers remain additional).
#[allow(dead_code)]
struct ValidationSlots<'a> {
    input: &'a EdgeDescriptor,
    values: &'a [i64],
    verified: &'a Verified,
    reads: [&'a EdgeDescriptor; READ_COUNT],
    declaration: crate::fixed_exp_tables::Definition,
    bound: ReadBound,
    bound_result: Result<ReadBound, Error>,
    table_extrema: Interval,
    interval_result: Result<Interval, Error>,
    product: Interval,
    bindings: BindingIntervals,
    lower_word: u64,
    upper_word: u64,
    value_word: u64,
    slot: usize,
    returned: Result<BindingIntervals, Error>,
}
pub(super) fn validation_slots_bytes() -> usize {
    size_of::<ValidationSlots<'static>>()
}

#[cfg(test)]
#[path = "nexp_stage_bounds_tests.rs"]
mod tests;
