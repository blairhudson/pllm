//! Exact integer matrix and row-energy bounds, independent of numeric protocols.
//!
//! Certificates describe ordinary signed coefficients, never ring residues or
//! secret observations. Source authentication, certificate provenance and no-wrap
//! proofs belong to callers. Rounding envelopes are explicit absolute errors in
//! output-code units; an envelope of one covers either signed floor or ceiling.
use std::mem::size_of;

pub const MAX_ROWS: usize = 16_384;
pub const MAX_COLUMNS: usize = 16_384;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Shape,
    Bounds,
    Allocation,
    Retired,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Shape => "integer bound shape or ordered scan length differs",
            Self::Bounds => "integer bound arithmetic or payload limit exceeded",
            Self::Allocation => "integer bound column allocation failed",
            Self::Retired => "integer bound scan is retired",
        })
    }
}
impl std::error::Error for Error {}

fn shape(rows: usize, columns: usize) -> Result<(), Error> {
    if rows == 0 || rows > MAX_ROWS || columns == 0 || columns > MAX_COLUMNS {
        return Err(Error::Shape);
    }
    rows.checked_mul(columns).ok_or(Error::Bounds)?;
    Ok(())
}

/// Read-only complete-matrix bounds. A scan computes exact statistics; imported
/// bounds require independent caller-owned provenance and authentication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixCertificate {
    rows: usize,
    columns: usize,
    maximum_row_l1: u128,
    maximum_column_l1: u128,
    frobenius_squared: u128,
    maximum_row_squared_energy: u128,
}

impl MatrixCertificate {
    /// Check geometry for caller-authenticated bounds. This does not certify that
    /// the supplied statistics describe any particular source matrix. Without an
    /// exact row scan, the Frobenius bound also bounds every complete row's energy.
    pub fn from_bounds(
        rows: usize,
        columns: usize,
        maximum_row_l1: u128,
        maximum_column_l1: u128,
        frobenius_squared: u128,
    ) -> Result<Self, Error> {
        shape(rows, columns)?;
        Ok(Self {
            rows,
            columns,
            maximum_row_l1,
            maximum_column_l1,
            frobenius_squared,
            maximum_row_squared_energy: frobenius_squared,
        })
    }

    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn columns(self) -> usize {
        self.columns
    }
    pub fn maximum_row_l1(self) -> u128 {
        self.maximum_row_l1
    }
    pub fn maximum_column_l1(self) -> u128 {
        self.maximum_column_l1
    }
    pub fn frobenius_squared(self) -> u128 {
        self.frobenius_squared
    }
    /// Exact for scanned matrices; conservatively Frobenius for `from_bounds`.
    pub fn maximum_row_squared_energy(self) -> u128 {
        self.maximum_row_squared_energy
    }

    /// ||W||_2^2 <= min(||W||_infinity * ||W||_1, ||W||_F^2).
    pub fn operator_squared(self) -> Result<u128, Error> {
        shape(self.rows, self.columns)?;
        // An overflowing product exceeds every representable Frobenius bound.
        Ok(self
            .maximum_row_l1
            .checked_mul(self.maximum_column_l1)
            .map_or(self.frobenius_squared, |bound| {
                bound.min(self.frobenius_squared)
            }))
    }
}

/// Exact maximum energy of a complete scanned row. No caller-supplied energy
/// constructor exists; only a completed ordered scan can create this value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowEnergyCertificate {
    rows: usize,
    columns: usize,
    maximum_row_squared_energy: u128,
}

impl RowEnergyCertificate {
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn columns(self) -> usize {
        self.columns
    }
    pub fn maximum_row_squared_energy(self) -> u128 {
        self.maximum_row_squared_energy
    }
}

struct Pending {
    columns: Vec<u128>,
    width: usize,
    rows: usize,
    offset: usize,
    current_row: u128,
    maximum_row: u128,
    frobenius_squared: u128,
    current_row_squared: u128,
    maximum_row_squared: u128,
}

impl Pending {
    fn complete(&self) -> Result<(), Error> {
        if self.offset != self.rows * self.width
            || self.current_row != 0
            || self.current_row_squared != 0
        {
            return Err(Error::Shape);
        }
        Ok(())
    }
}

/// Ordered row-major public scan retaining at most one u128 total per column.
/// Row-energy-only mode retains constant state and never sums across rows.
/// Every append error drops its state. Finish consumes the scan, and incomplete
/// scans cannot mint certificates. Dropping the owner releases all payload.
pub struct MatrixScan(Option<Pending>);

impl MatrixScan {
    /// Preflight column payload plus owner/working-state sizes, excluding
    /// allocator bookkeeping and caller-owned input slices. No allocation occurs.
    pub fn payload_bytes(rows: usize, columns: usize) -> Result<usize, Error> {
        shape(rows, columns)?;
        columns
            .checked_mul(size_of::<u128>())
            .and_then(|n| n.checked_add(size_of::<Self>() + size_of::<Pending>()))
            .ok_or(Error::Bounds)
    }

    /// The caller reserves the explicit payload limit before this allocation.
    pub fn new(rows: usize, columns: usize, payload_limit: usize) -> Result<Self, Error> {
        if Self::payload_bytes(rows, columns)? > payload_limit {
            return Err(Error::Bounds);
        }
        let mut sums = Vec::new();
        sums.try_reserve_exact(columns)
            .map_err(|_| Error::Allocation)?;
        sums.resize(columns, 0);
        Ok(Self::start(rows, columns, sums))
    }

    /// Constant-state row-energy preflight. Unlike a full matrix certificate,
    /// this supports any nonempty checked geometry, including large vocabularies.
    /// The enclosing producer is responsible for its own total byte/work limits.
    pub fn row_energy_payload_bytes(rows: usize, columns: usize) -> Result<usize, Error> {
        if rows == 0 || columns == 0 {
            return Err(Error::Shape);
        }
        rows.checked_mul(columns).ok_or(Error::Bounds)?;
        Ok(size_of::<Self>() + size_of::<Pending>())
    }

    pub fn new_row_energy(
        rows: usize,
        columns: usize,
        payload_limit: usize,
    ) -> Result<Self, Error> {
        if Self::row_energy_payload_bytes(rows, columns)? > payload_limit {
            return Err(Error::Bounds);
        }
        Ok(Self::start(rows, columns, Vec::new()))
    }

    fn start(rows: usize, width: usize, columns: Vec<u128>) -> Self {
        Self(Some(Pending {
            columns,
            width,
            rows,
            offset: 0,
            current_row: 0,
            maximum_row: 0,
            frobenius_squared: 0,
            current_row_squared: 0,
            maximum_row_squared: 0,
        }))
    }

    pub fn append(&mut self, offset: usize, values: &[i64]) -> Result<(), Error> {
        self.append_iter(offset, values.iter().copied())
    }

    // FixedFile decodes canonical byte chunks through this same scan without an
    // extra decoded buffer. Check iterator length too, before restoring state.
    pub(crate) fn append_iter(
        &mut self,
        offset: usize,
        values: impl ExactSizeIterator<Item = i64>,
    ) -> Result<(), Error> {
        let mut pending = self.0.take().ok_or(Error::Retired)?;
        let end = offset.checked_add(values.len()).ok_or(Error::Shape)?;
        if values.len() == 0 || offset != pending.offset || end > pending.rows * pending.width {
            return Err(Error::Shape);
        }
        for value in values {
            if pending.offset == end {
                return Err(Error::Shape);
            }
            let magnitude = u128::from(value.unsigned_abs());
            let square = squared(magnitude)?;
            pending.current_row_squared = pending
                .current_row_squared
                .checked_add(square)
                .ok_or(Error::Bounds)?;
            if !pending.columns.is_empty() {
                let column = pending.offset % pending.width;
                pending.columns[column] = pending.columns[column]
                    .checked_add(magnitude)
                    .ok_or(Error::Bounds)?;
                pending.current_row = pending
                    .current_row
                    .checked_add(magnitude)
                    .ok_or(Error::Bounds)?;
                pending.frobenius_squared = pending
                    .frobenius_squared
                    .checked_add(square)
                    .ok_or(Error::Bounds)?;
            }
            pending.offset += 1;
            if pending.offset % pending.width == 0 {
                pending.maximum_row = pending.maximum_row.max(pending.current_row);
                pending.current_row = 0;
                pending.maximum_row_squared =
                    pending.maximum_row_squared.max(pending.current_row_squared);
                pending.current_row_squared = 0;
            }
        }
        if pending.offset != end {
            return Err(Error::Shape);
        }
        self.0 = Some(pending);
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.0 = None;
    }

    pub fn finish(mut self) -> Result<MatrixCertificate, Error> {
        let pending = self.0.take().ok_or(Error::Retired)?;
        pending.complete()?;
        // Row-only scans cannot fabricate column statistics or Frobenius sums.
        if pending.columns.is_empty() {
            return Err(Error::Shape);
        }
        let mut certificate = MatrixCertificate::from_bounds(
            pending.rows,
            pending.width,
            pending.maximum_row,
            pending.columns.into_iter().max().ok_or(Error::Shape)?,
            pending.frobenius_squared,
        )?;
        certificate.maximum_row_squared_energy = pending.maximum_row_squared;
        Ok(certificate)
    }

    pub fn finish_row_energy(mut self) -> Result<RowEnergyCertificate, Error> {
        let pending = self.0.take().ok_or(Error::Retired)?;
        pending.complete()?;
        Ok(RowEnergyCertificate {
            rows: pending.rows,
            columns: pending.width,
            maximum_row_squared_energy: pending.maximum_row_squared,
        })
    }
}

/// Exact ceiling of the square root over the entire u128 domain.
pub fn ceil_sqrt(value: u128) -> u128 {
    let floor = value.isqrt();
    floor + u128::from(floor * floor != value)
}

fn squared(radius: u128) -> Result<u128, Error> {
    radius.checked_mul(radius).ok_or(Error::Bounds)
}

/// Exact squared energy of ordinary signed codes, checked even for i64::MIN.
/// An empty vector has zero energy; no scan state or allocation is involved.
pub fn row_energy(values: &[i64]) -> Result<u128, Error> {
    values.iter().try_fold(0u128, |sum, value| {
        sum.checked_add(squared(u128::from(value.unsigned_abs()))?)
            .ok_or(Error::Bounds)
    })
}

/// Bound a row after division by 2^shift and a per-coordinate absolute error of
/// at most `rounding_units`. Shift is 0..=127. Zero error represents exact scaling;
/// one conservatively covers both signed floor and signed ceiling (<1 error).
pub fn rescaled_energy(
    raw_squared: u128,
    coordinates: usize,
    shift: u32,
    rounding_units: u128,
) -> Result<u128, Error> {
    if coordinates == 0 || coordinates > MAX_COLUMNS {
        return Err(Error::Shape);
    }
    if shift >= 128 {
        return Err(Error::Bounds);
    }
    let rounding = ceil_sqrt(coordinates as u128)
        .checked_mul(rounding_units)
        .ok_or(Error::Bounds)?;
    squared(
        ceil_sqrt(raw_squared)
            .div_ceil(1u128 << shift)
            .checked_add(rounding)
            .ok_or(Error::Bounds)?,
    )
}

/// Bound a complete output row of a matrix stored as [output, input], with one
/// caller-specified final-coordinate rounding envelope after the complete sum.
pub fn linear_energy(
    input_squared: u128,
    matrix: MatrixCertificate,
    shift: u32,
    rounding_units: u128,
) -> Result<u128, Error> {
    let raw = input_squared
        .checked_mul(matrix.operator_squared()?)
        .ok_or(Error::Bounds)?;
    rescaled_energy(raw, matrix.rows, shift, rounding_units)
}

/// Minkowski bound for addition of two equal-width rows.
pub fn sum_energy(left: u128, right: u128) -> Result<u128, Error> {
    squared(
        ceil_sqrt(left)
            .checked_add(ceil_sqrt(right))
            .ok_or(Error::Bounds)?,
    )
}

/// Bound half-split two-coordinate rotations with operator norm
/// sqrt(c^2+s^2)/2^shift. The envelope is per FINAL coordinate: use two for two
/// separately floored/ceiled product terms, or one for a single final truncation.
pub fn rotary_energy(
    input_squared: u128,
    cos: &[i64],
    sin: &[i64],
    shift: u32,
    rounding_units: u128,
) -> Result<u128, Error> {
    if cos.is_empty() || cos.len() != sin.len() || cos.len() > MAX_ROWS / 2 {
        return Err(Error::Shape);
    }
    let mut maximum = 0u128;
    for (&c, &s) in cos.iter().zip(sin) {
        maximum = maximum.max(
            squared(u128::from(c.unsigned_abs()))?
                .checked_add(squared(u128::from(s.unsigned_abs()))?)
                .ok_or(Error::Bounds)?,
        );
    }
    let raw = input_squared.checked_mul(maximum).ok_or(Error::Bounds)?;
    rescaled_energy(raw, 2 * cos.len(), shift, rounding_units)
}

/// Cauchy--Schwarz radius for a complete untruncated dot. Ring/no-wrap admission
/// remains a separate caller obligation.
pub fn dot_radius(left_squared: u128, right_squared: u128) -> Result<u128, Error> {
    Ok(ceil_sqrt(
        left_squared
            .checked_mul(right_squared)
            .ok_or(Error::Bounds)?,
    ))
}

/// Absolute update-error codes, each representing its value divided by 2^f.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReciprocalStepError {
    pub delta_d: u128,
    pub delta_n: u128,
}

/// Public reference envelope only: f=0..=127, m=0..=8 are not evaluated settings.
/// All numeric codes represent value / 2^fraction_bits. Every inactive step must
/// have both fields zero; this also applies to all eight steps when m=0.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReciprocalBoundSpec {
    pub fraction_bits: u8,
    pub iterations: u8,
    pub xmin: u128,
    pub xmax: u128,
    pub e0: u128,
    pub w0: u128,
    pub steps: [ReciprocalStepError; 8],
}

/// Conditional nonstrict absolute error bound, not an execution capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReciprocalBound {
    code: u128,
    fraction_bits: u8,
}

impl ReciprocalBound {
    pub fn code(self) -> u128 {
        self.code
    }

    pub fn fraction_bits(self) -> u8 {
        self.fraction_bits
    }
}

// Private caller contract: divisor > 0. Quotient/remainder avoid n+d-1 overflow.
// Increment cannot overflow: remainder!=0 implies divisor>=2 and q<u128::MAX;
// divisor=1 instead gives remainder=0. Keep the increment checked nonetheless.
fn reciprocal_ceil(numerator: u128, divisor: u128) -> Result<u128, Error> {
    let quotient = numerator / divisor;
    let remainder = numerator % divisor;
    quotient
        .checked_add(u128::from(remainder != 0))
        .ok_or(Error::Bounds)
}

/// Bound simultaneous old-state Goldschmidt updates for a fixed positive x:
/// d' = d*(2-d)+eps_d, n' = n*(2-d)+eps_n, |eps_*| <= delta_*/S.
/// The caller must independently justify xmin/S <= x <= xmax/S,
/// |1-d0| <= e0/S and |d0-x*n0| <= w0/S, where S=2^fraction_bits.
/// For OLD E,W, propagate E'=ceil(E*E/S)+delta_d and
/// W'=ceil((S+E)*W/S)+delta_d+ceil(xmax*delta_n/S).
/// The returned B/S bounds |n-1/x| nonstrictly, B=ceil((E+W)*S/xmin).
/// Indeed 1-d'=(1-d)^2-eps_d and
/// d'-x*n'=(2-d)*(d-x*n)+eps_d-x*eps_n, with |2-d|<=1+E/S.
/// Finally |n-1/x|<=((E+W)/S)/x<=(E+W)/xmin before outward rounding.
/// Supplied premises are not authenticated: this API has no x,d0,n0 inputs.
/// Refresh/gain are excluded; a post-refresh invocation needs independently
/// justified new initial premises, never an automatic error reset. No convergence,
/// tolerance, paper iteration count or numerical profile is selected.
///
/// Every invalid domain, nonzero inactive field or intermediate overflow returns
/// Error::Bounds, even when a reduced answer would fit. Only complete success
/// constructs a bound; the borrowed specification is immutable on every path.
///
/// External admission, as with MatrixScan: BEFORE entry the caller must fund
/// validation, full work, input/original backing and output/error overlap. This
/// helper neither authenticates funding/premises nor quotes/adopts a runtime.
/// There is no heap allocation, history, callback or imported capability.
///
/// Source-level work inventory (successful m-step path; errors short-circuit):
/// checked multiply 3m+1; division AND remainder each 3m+1; checked ceiling
/// increment 3m+1; other checked add 4m+1; checked shift 1. Scalar validation
/// has <=4 comparisons, then <=8 borrowed step reads/classifications, <=8 index
/// validity comparisons and <=16 zero comparisons (short-circuit OR). Validation
/// has <=9 loop termination checks; recurrence <=8 iterations/9 termination checks.
/// A single prefix slice bounds check is justified by m<=8. No recurrence indexing.
/// Each step has 7 checked-arithmetic Option->Result/? sites and 3 helper /?
/// sites; finalization has 2 arithmetic sites and 1 helper /?, plus entry shift.
/// Thus main has 10m+4 propagated Result sites on success; each of 3m+1 helper
/// calls adds one Option->Result return. Two explicit validation Err returns and
/// one final Ok return exist; helper returns its checked increment Result.
/// Branches also include short-circuit validation and helper remainder!=0; each
/// checked operation may exit early. Local E,W update only after all step checks.
/// Named assignments: entry binds S,E,W; each recurrence binds nine magnitudes
/// then assigns E,W; finalization binds sum,numerator,code. Each helper binds q,r.
///
/// Remaining source-control inventory (successful path):
/// - Two slice-iterator constructions, one enumerate construction; validation
///   next 9 / recurrence next m+1, separately Some/None classifications 9 / m+1;
///   bind 8 validation pairs and m steps.
/// - Scalar OR 1..4 comparisons, <=3 OR continuations. Validation 8 index
///   comparisons / AND RHS 8-m, inactive fields 2(8-m) comparisons on success
///   and short-circuit OR on error. Helper: one remainder comparison and u128
///   Bool conversion per call.
/// - usize::from 9, u32::from 1; prefix bounds check 1. Iterator internals are
///   not additional SOURCE indexing.
/// - Arithmetic Options 7m+3 main, 3m+1 helpers classified by ok_or. Main
///   10m+4 propagated Result classifications; helper returns Result without ?;
///   iterator Option classifications are separate.
/// - Bind S/E/W once, nine magnitudes/step plus 2 commits, 3 final magnitudes,
///   helper q/r. Two explicit validation Err sites, propagation exits, helper
///   returns and final Ok are distinct. Accessors return one copied field.
///
/// All error prefixes use independent maximum m=8: validation next <=9,
/// recurrence next <=9, 8 pair/step bindings, 4 scalar / 8 index / 16 inactive
/// comparisons. Invalid supplied m never determines work. Arithmetic follows
/// validation; errors omit suffixes.
///
/// Conservative LOGICAL source-slot expression; aliases are NOT new Rust types
/// or API. With sz<T> = size_of::<T>(), the logical envelope L is:
///
/// ```text
/// I = std::slice::Iter<'a, ReciprocalStepError>
/// J = std::iter::Enumerate<I>
/// P = (usize, &'a ReciprocalStepError)
/// A = Option<u128>
/// R = Result<u128, Error>
/// O = Result<ReciprocalBound, Error>
/// L = sz<&ReciprocalBoundSpec>
///   + sz<&[ReciprocalStepError; 8]> + sz<&[ReciprocalStepError]>
///   + 3*sz<I> + sz<J> + sz<P> + sz<Option<P>>
///   + 2*sz<&ReciprocalStepError> + sz<Option<&ReciprocalStepError>>
///   + sz<std::ops::RangeTo<usize>> + 3*sz<usize>
///   + 4*sz<u8> + sz<u32> + 13*sz<bool>
///   + 76*sz<u128> + 11*sz<A> + 15*sz<R>
///   + 17*sz<O> + 27*sz<Error> + sz<ReciprocalBound>
/// ```
///
/// The 76 magnitudes cover 20 named main/helper argument/local slots, 3 operand/
/// result slots for each of 16 conservative expression sites, and 8 extra helper
/// call argument slots. The 16 sites are 11 distinct checked-arithmetic
/// expressions, 4 ceiling-call source sites and 1 final bound construction:
/// source-audited conservative operand/result allowance, NOT a new fee. Sum-type
/// allowances cover checked operations, helper returns and propagated error
/// outputs, intentionally conservatively overlapping payloads. Sequential
/// iterations/helper calls reuse SOURCE-SITE slots.
///
/// Spec view extent size_of::<ReciprocalBoundSpec>() is contained in, NOT billed
/// additionally over, caller-charged FULL original backing; borrow references
/// are separate logical slots. Caller entry/work/original backing must be funded
/// externally BEFORE validation. Source-call slots EXCLUDE private standard
/// library internals; fixed width/bounded iteration do not establish physical
/// costs. No stack/RSS/payment/tariff/quote API/admitted-execution claim follows.
/// Independent caller binding is still required.
pub fn reciprocal_bound(spec: &ReciprocalBoundSpec) -> Result<ReciprocalBound, Error> {
    if spec.fraction_bits > 127 || spec.iterations > 8 || spec.xmin == 0 || spec.xmin > spec.xmax {
        return Err(Error::Bounds);
    }
    for (index, step) in spec.steps.iter().enumerate() {
        if index >= usize::from(spec.iterations) && (step.delta_d != 0 || step.delta_n != 0) {
            return Err(Error::Bounds);
        }
    }
    // This checked shift cannot fail after f<=127; S and xmin are positive.
    let scale = 1u128
        .checked_shl(u32::from(spec.fraction_bits))
        .ok_or(Error::Bounds)?;
    let mut e = spec.e0;
    let mut w = spec.w0;
    for step in &spec.steps[..usize::from(spec.iterations)] {
        let square = e.checked_mul(e).ok_or(Error::Bounds)?;
        let next_e = reciprocal_ceil(square, scale)?
            .checked_add(step.delta_d)
            .ok_or(Error::Bounds)?;
        // After E^2 succeeds, E<=2^64-1, so S+E cannot overflow for S<=2^127.
        let gain = scale.checked_add(e).ok_or(Error::Bounds)?;
        let product = gain.checked_mul(w).ok_or(Error::Bounds)?;
        let propagated_w = reciprocal_ceil(product, scale)?;
        let with_delta = propagated_w
            .checked_add(step.delta_d)
            .ok_or(Error::Bounds)?;
        let numerator_error = spec
            .xmax
            .checked_mul(step.delta_n)
            .ok_or(Error::Bounds)?;
        let scaled_error = reciprocal_ceil(numerator_error, scale)?;
        let next_w = with_delta.checked_add(scaled_error).ok_or(Error::Bounds)?;
        e = next_e;
        w = next_w;
    }
    let sum = e.checked_add(w).ok_or(Error::Bounds)?;
    let numerator = sum.checked_mul(scale).ok_or(Error::Bounds)?;
    let code = reciprocal_ceil(numerator, spec.xmin)?;
    Ok(ReciprocalBound {
        code,
        fraction_bits: spec.fraction_bits,
    })
}

#[cfg(test)]
mod reciprocal_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    fn scan(rows: usize, columns: usize, values: &[i64], chunk: usize) -> MatrixCertificate {
        let bytes = MatrixScan::payload_bytes(rows, columns).unwrap();
        let mut scan = MatrixScan::new(rows, columns, bytes).unwrap();
        for (i, values) in values.chunks(chunk).enumerate() {
            scan.append(i * chunk, values).unwrap();
        }
        scan.finish().unwrap()
    }

    fn energy(values: &[i64]) -> u128 {
        values
            .iter()
            .map(|v| (*v as i128 * *v as i128) as u128)
            .sum()
    }

    #[test]
    fn matrix_statistics_are_exact_across_chunks_and_extended_geometry() {
        let values = [1, -2, 3, -4, 0, 5];
        let certificate = scan(2, 3, &values, 4);
        assert_eq!((certificate.rows(), certificate.columns()), (2, 3));
        assert_eq!(certificate.maximum_row_l1(), 9);
        assert_eq!(certificate.maximum_column_l1(), 8);
        assert_eq!(certificate.frobenius_squared(), 55);
        assert_eq!(certificate.maximum_row_squared_energy(), 41);
        assert_eq!(certificate.operator_squared(), Ok(55));
        for chunk in [1, 2, 3, 6] {
            assert_eq!(scan(2, 3, &values, chunk), certificate);
        }
        assert_eq!(scan(2, 3, &values.map(|v| -v), 5), certificate);
        let tall = scan(MAX_ROWS, 1, &vec![1; MAX_ROWS], 127);
        assert_eq!(tall.maximum_column_l1(), MAX_ROWS as u128);
        assert_eq!(tall.operator_squared(), Ok(MAX_ROWS as u128));
        assert_eq!(tall.maximum_row_squared_energy(), 1);
        let declared = MatrixCertificate::from_bounds(2, 3, 9, 8, 55).unwrap();
        assert_eq!(declared.maximum_row_squared_energy(), 55);
        assert_eq!(declared.operator_squared(), certificate.operator_squared());
        assert_eq!(
            MatrixCertificate::from_bounds(1, 1, u128::MAX, 2, 13)
                .unwrap()
                .operator_squared(),
            Ok(13)
        );
    }

    #[test]
    fn row_only_scan_is_constant_payload_and_does_not_require_total_square_sum() {
        let bytes = MatrixScan::row_energy_payload_bytes(MAX_ROWS + 1, 2).unwrap();
        assert_eq!(bytes, MatrixScan::row_energy_payload_bytes(1, 1).unwrap());
        assert!(matches!(
            MatrixScan::new_row_energy(2, 2, bytes - 1),
            Err(Error::Bounds)
        ));
        assert_eq!(
            MatrixScan::row_energy_payload_bytes(usize::MAX, 2),
            Err(Error::Bounds)
        );
        let mut rows = MatrixScan::new_row_energy(2, 2, bytes).unwrap();
        rows.append(0, &[i64::MIN; 3]).unwrap();
        rows.append(3, &[i64::MIN]).unwrap();
        let certificate = rows.finish_row_energy().unwrap();
        assert_eq!((certificate.rows(), certificate.columns()), (2, 2));
        assert_eq!(certificate.maximum_row_squared_energy(), 1 << 127);
        let mut matrix = MatrixScan::new(2, 2, MatrixScan::payload_bytes(2, 2).unwrap()).unwrap();
        assert_eq!(matrix.append(0, &[i64::MIN; 4]), Err(Error::Bounds));
        assert_eq!(matrix.finish(), Err(Error::Retired));
        let mut rows = MatrixScan::new_row_energy(1, 1, bytes).unwrap();
        rows.append(0, &[1]).unwrap();
        assert_eq!(rows.finish(), Err(Error::Shape));

        let mut large = MatrixScan::new_row_energy(MAX_ROWS + 1, 2, bytes).unwrap();
        for row in 0..MAX_ROWS + 1 {
            large.append(row * 2, &[-3, 4]).unwrap();
        }
        assert_eq!(
            large
                .finish_row_energy()
                .unwrap()
                .maximum_row_squared_energy(),
            25
        );
    }

    #[test]
    fn requested_row_energy_overflow_order_errors_and_cancellation_retire_scan() {
        let bytes = MatrixScan::row_energy_payload_bytes(1, 4).unwrap();
        let mut scan = MatrixScan::new_row_energy(1, 4, bytes).unwrap();
        scan.append(0, &[i64::MIN; 3]).unwrap();
        assert_eq!(scan.append(3, &[i64::MIN]), Err(Error::Bounds));
        assert_eq!(scan.finish_row_energy(), Err(Error::Retired));
        let mut scan = MatrixScan::new_row_energy(1, 4, bytes).unwrap();
        scan.append(0, &[1]).unwrap();
        assert_eq!(scan.append(0, &[2]), Err(Error::Shape));
        assert_eq!(scan.finish_row_energy(), Err(Error::Retired));
        let mut scan = MatrixScan::new_row_energy(1, 4, bytes).unwrap();
        scan.append(0, &[1]).unwrap();
        assert_eq!(scan.finish_row_energy(), Err(Error::Shape));
        let mut scan = MatrixScan::new_row_energy(1, 4, bytes).unwrap();
        scan.cancel();
        assert_eq!(scan.append(0, &[1]), Err(Error::Retired));
        assert_eq!(scan.finish_row_energy(), Err(Error::Retired));
    }

    #[test]
    fn scans_preflight_payload_and_consume_every_malformed_or_overflowed_append() {
        let bytes = MatrixScan::payload_bytes(MAX_ROWS, MAX_COLUMNS).unwrap();
        assert!(bytes < (257 << 10));
        assert!(matches!(
            MatrixScan::new(MAX_ROWS, MAX_COLUMNS, bytes - 1),
            Err(Error::Bounds)
        ));
        for shape in [
            (0, 1),
            (1, 0),
            (MAX_ROWS + 1, 1),
            (1, MAX_COLUMNS + 1),
            (usize::MAX, 1),
        ] {
            assert_eq!(
                MatrixScan::payload_bytes(shape.0, shape.1),
                Err(Error::Shape)
            );
        }
        for (offset, values) in [
            (1, vec![0]),
            (0, vec![]),
            (0, vec![0; 5]),
            (usize::MAX, vec![0]),
            (0, vec![i64::MIN; 4]),
        ] {
            let mut owner = MatrixScan::new(2, 2, bytes).unwrap();
            assert!(owner.append(offset, &values).is_err());
            assert_eq!(owner.append(0, &[0; 4]), Err(Error::Retired));
            assert_eq!(owner.finish(), Err(Error::Retired));
        }
        let mut owner = MatrixScan::new(2, 2, bytes).unwrap();
        owner.append(0, &[1, -1, 2]).unwrap();
        assert_eq!(owner.finish(), Err(Error::Shape));
        let mut owner = MatrixScan::new(1, 1, bytes).unwrap();
        owner.append(0, &[0]).unwrap();
        assert_eq!(owner.append(1, &[0]), Err(Error::Shape));
        assert_eq!(owner.finish(), Err(Error::Retired));
    }

    #[test]
    fn roots_and_row_energy_cover_full_integer_domain_and_fail_closed_on_overflow() {
        for value in [
            0,
            1,
            2,
            3,
            4,
            15,
            16,
            17,
            (1 << 126) - 1,
            1 << 126,
            u128::MAX,
        ] {
            let root = ceil_sqrt(value);
            let big = BigUint::from(value);
            assert!(BigUint::from(root).pow(2) >= big);
            if root != 0 {
                assert!(BigUint::from(root - 1).pow(2) < big);
            }
        }
        assert_eq!(ceil_sqrt(u128::MAX), 1 << 64);
        assert_eq!(row_energy(&[]), Ok(0));
        assert_eq!(row_energy(&[i64::MIN]), Ok(1 << 126));
        assert_eq!(row_energy(&[i64::MIN; 4]), Err(Error::Bounds));
        assert_eq!(dot_radius(u128::MAX, 2), Err(Error::Bounds));
        assert_eq!(sum_energy(u128::MAX, 1), Err(Error::Bounds));
        assert_eq!(rescaled_energy(1, 1, 128, 0), Err(Error::Bounds));
        assert_eq!(rescaled_energy(1, 0, 1, 1), Err(Error::Shape));
        assert_eq!(rescaled_energy(1, 2, 1, u128::MAX), Err(Error::Bounds));
        assert_eq!(
            rotary_energy(u128::MAX, &[2], &[0], 18, 2),
            Err(Error::Bounds)
        );
    }

    #[test]
    fn independent_signed_floor_and_ceiling_matrix_outputs_obey_shared_envelope() {
        for code in 0..81usize {
            let mut code = code;
            let mut weights = [0i64; 4];
            for weight in &mut weights {
                *weight = (code % 3) as i64 - 1;
                code /= 3;
            }
            let certificate = scan(2, 2, &weights, 3);
            for x in -2..=2i64 {
                for y in -2..=2i64 {
                    let raw = [
                        weights[0] * x + weights[1] * y,
                        weights[2] * x + weights[3] * y,
                    ];
                    let input = energy(&[x, y]);
                    assert!(energy(&raw) <= linear_energy(input, certificate, 0, 0).unwrap());
                    for shift in [1, 3] {
                        let bound = linear_energy(input, certificate, shift, 1).unwrap();
                        for choices in 0..4 {
                            let output = std::array::from_fn::<_, 2, _>(|i| {
                                raw[i].div_euclid(1 << shift)
                                    + i64::from(
                                        choices & (1 << i) != 0
                                            && raw[i].rem_euclid(1 << shift) != 0,
                                    )
                            });
                            assert!(energy(&output) <= bound);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rotary_final_and_separate_product_rounding_envelopes_and_dot_bounds() {
        for x in -3..=3i64 {
            for y in -3..=3i64 {
                for (c, s) in [(8, 0), (0, 8), (5, 6), (-5, 6)] {
                    let input = energy(&[x, y]);
                    let final_floor =
                        [(x * c - y * s).div_euclid(8), (y * c + x * s).div_euclid(8)];
                    assert!(
                        energy(&final_floor) <= rotary_energy(input, &[c], &[s], 3, 1).unwrap()
                    );
                    let bound = rotary_energy(input, &[c], &[s], 3, 2).unwrap();
                    for choices in 0..16 {
                        let raw = [x * c, -y * s, y * c, x * s];
                        let terms = std::array::from_fn::<_, 4, _>(|i| {
                            raw[i].div_euclid(8)
                                + i64::from(choices & (1 << i) != 0 && raw[i].rem_euclid(8) != 0)
                        });
                        let out = [terms[0] + terms[1], terms[2] + terms[3]];
                        assert!(energy(&out) <= bound);
                        let bias = [-2, 3];
                        assert!(
                            energy(&[out[0] - 2, out[1] + 3])
                                <= sum_energy(bound, energy(&bias)).unwrap()
                        );
                        let dot = out[0] as i128 * -2 + out[1] as i128 * 3;
                        assert!(dot.unsigned_abs() <= dot_radius(bound, energy(&bias)).unwrap());
                    }
                }
            }
        }
    }
}
