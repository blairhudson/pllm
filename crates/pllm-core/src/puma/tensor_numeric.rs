//! Source-ordered reciprocal/rsqrt graphs and public conditional range proofs.
//!
//! A partition by the ORIGINAL input exponent is public proof machinery only.
//! Every runtime executes the same graph; no exponent or selected box is opened.
use super::*;

const ONE: i64 = 1 << 18;
const DOMAINS: usize = 36;

#[derive(Clone)]
struct Ranged {
    value: Value,
    boxes: [Option<Interval>; DOMAINS],
}

fn hull(boxes: &[Option<Interval>; DOMAINS]) -> Result<Interval, Error> {
    let lower = boxes
        .iter()
        .flatten()
        .map(|b| b.lower())
        .min()
        .ok_or(Error::Bounds)?;
    let upper = boxes
        .iter()
        .flatten()
        .map(|b| b.upper())
        .max()
        .ok_or(Error::Bounds)?;
    Interval::new(lower, upper)
}

impl Ranged {
    fn input(value: Value, bounds: Interval, width: u8) -> Result<Self, Error> {
        if bounds.lower() <= 0 || bounds.upper() >= 1_i64 << width {
            return Err(Error::Bounds);
        }
        let boxes = std::array::from_fn(|j| {
            let lower = bounds.lower().max(1 << j);
            let upper = bounds.upper().min((1 << (j + 1)) - 1);
            (lower <= upper).then(|| Interval::new(lower, upper).unwrap())
        });
        Ok(Self { value, boxes })
    }

    fn exact_map(
        &self,
        b: &mut Builder,
        value: Value,
        f: impl Fn(usize) -> i64,
    ) -> Result<Self, Error> {
        let boxes =
            std::array::from_fn(|j| self.boxes[j].map(|_| Interval::new(f(j), f(j)).unwrap()));
        b.0.nodes[value.0 as usize].as_mut().unwrap().bounds = hull(&boxes)?;
        Ok(Self { value, boxes })
    }

    fn affine(
        &self,
        b: &mut Builder,
        other: Option<&Self>,
        coefficients: Affine,
    ) -> Result<Self, Error> {
        let mut boxes = [None; DOMAINS];
        for (j, slot) in boxes.iter_mut().enumerate() {
            if let Some(left) = self.boxes[j] {
                let right = match other {
                    Some(other) => other.boxes[j].ok_or(Error::Binding)?,
                    None => Interval::new(0, 0)?,
                };
                *slot = Some(coefficients.output_bounds(InputBounds { left, right })?);
            }
        }
        let value = b.affine(self.value, other.map(|r| r.value), coefficients)?;
        b.0.nodes[value.0 as usize].as_mut().unwrap().bounds = hull(&boxes)?;
        Ok(Self { value, boxes })
    }

    fn product(&self, b: &mut Builder, other: &Self, rescale: bool) -> Result<Self, Error> {
        let mut raw = [None; DOMAINS];
        let mut boxes = [None; DOMAINS];
        for j in 0..DOMAINS {
            if let Some(left) = self.boxes[j] {
                let right = other.boxes[j].ok_or(Error::Binding)?;
                let corners = [
                    i128::from(left.lower()) * i128::from(right.lower()),
                    i128::from(left.lower()) * i128::from(right.upper()),
                    i128::from(left.upper()) * i128::from(right.lower()),
                    i128::from(left.upper()) * i128::from(right.upper()),
                ];
                let range = Interval::new(
                    i64::try_from(*corners.iter().min().unwrap()).map_err(|_| Error::Bounds)?,
                    i64::try_from(*corners.iter().max().unwrap()).map_err(|_| Error::Bounds)?,
                )?;
                raw[j] = Some(range);
                boxes[j] = Some(if rescale {
                    rescaled_bounds(range, 18)?
                } else {
                    range
                });
            }
        }
        let value = b.push(
            Instruction::Product {
                left: self.value,
                right: other.value,
                rescale,
                certified: Some(hull(&raw)?),
            },
            hull(&boxes)?,
            false,
        )?;
        Ok(Self { value, boxes })
    }

    fn rescale(&self, b: &mut Builder, shift: u8) -> Result<Self, Error> {
        let mut boxes = [None; DOMAINS];
        for (slot, source) in boxes.iter_mut().zip(self.boxes) {
            *slot = source.map(|r| rescaled_bounds(r, shift)).transpose()?;
        }
        let value = b.rescale(self.value, shift)?;
        b.0.nodes[value.0 as usize].as_mut().unwrap().bounds = hull(&boxes)?;
        Ok(Self { value, boxes })
    }
}

fn affine(left: i64, right: i64, constant: i64) -> Affine {
    Affine {
        left,
        right,
        constant,
    }
}

impl Program {
    /// Finite binary32 default split-half coefficients, then the Q18
    /// toward-zero boundary. Fixed libm implementation; no private positions.
    pub fn rotary_coefficients(
        theta: f32,
        position: u32,
        width: usize,
    ) -> Result<(Vec<i64>, Vec<i64>), Error> {
        if !theta.is_finite()
            || theta < 1.0
            || width == 0
            || width > MAX_TILE_LANES
            || width % 2 != 0
            || position > 512
        {
            return Err(Error::Bounds);
        }
        let mut cosine = Vec::new();
        let mut sine = Vec::new();
        cosine
            .try_reserve_exact(width / 2)
            .map_err(|_| Error::Allocation)?;
        sine.try_reserve_exact(width / 2)
            .map_err(|_| Error::Allocation)?;
        for i in 0..width / 2 {
            let frequency = 1.0_f32 / libm::powf(theta, (2 * i) as f32 / width as f32);
            let angle = position as f32 * frequency;
            cosine.push((libm::cosf(angle) * ONE as f32) as i64);
            sine.push((libm::sinf(angle) * ONE as f32) as i64);
        }
        Ok((cosine, sine))
    }
    /// Half-split rotary embedding with explicit public Q18 coefficient rows.
    /// Each row supplies width/2 cosines and sines, repeated across its halves.
    /// The two signed coefficient products rescale separately before addition.
    /// Source/position validation and coefficient encoding precede this API.
    pub fn rotary_half(
        rows: usize,
        width: usize,
        input: Interval,
        cosine: &[i64],
        sine: &[i64],
    ) -> Result<Self, Error> {
        let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
        if rows == 0
            || width == 0
            || width % 2 != 0
            || cosine.len() != lanes / 2
            || sine.len() != lanes / 2
        {
            return Err(Error::Shape);
        }
        let mut b = Builder::new(lanes, input)?;
        if cosine.iter().chain(sine).any(|c| !(-ONE..=ONE).contains(c)) {
            return Err(Error::Bounds);
        }
        let x = b.input();
        let half = width / 2;
        let mut coefficients = [affine(0, 0, 0); MAX_TILE_LANES];
        for (i, c) in coefficients[..lanes].iter_mut().enumerate() {
            *c = affine(cosine[(i / width) * half + i % half], 0, 0);
        }
        let cosine_product = b.affine_lanes(x, &coefficients[..lanes])?;
        let cosine_product = b.rescale(cosine_product, 18)?;
        let swapped = b.row_map(x, width, RowMap::Rotate(half))?;
        for (i, c) in coefficients[..lanes].iter_mut().enumerate() {
            let s = sine[(i / width) * half + i % half];
            *c = affine(if i % width < half { -s } else { s }, 0, 0);
        }
        let sine_product = b.affine_lanes(swapped, &coefficients[..lanes])?;
        let sine_product = b.rescale(sine_product, 18)?;
        let output = b.affine(cosine_product, Some(sine_product), affine(1, 1, 0))?;
        b.finish(output)
    }

    /// Shifted, strictly clipped P-N1 softmax. Every eagerly evaluated power is
    /// admitted, including the lanes subsequently masked below -14.
    pub fn softmax(rows: usize, width: usize, input: Interval) -> Result<Self, Error> {
        Ok(Self::softmax_schedule(rows, width, input, false, false)?.0)
    }

    /// Range-safe shifted softmax. Secretly replace clipped inputs by zero
    /// before exponentiation, then zero those exponentials before summation.
    /// Retained-domain stochastic equations and the strict cutoff are unchanged;
    /// the additional exact shared product has a distinct program identity.
    pub fn softmax_guarded(rows: usize, width: usize, input: Interval) -> Result<Self, Error> {
        Ok(Self::softmax_schedule(rows, width, input, true, false)?.0)
    }

    /// Original guarded schedule with a public proof of each complete row's
    /// nonnegative encoded mass. Denominator partitions cover every rounding;
    /// the certificate does not disclose a selected partition or score.
    pub fn softmax_certified(
        rows: usize,
        width: usize,
        input: Interval,
    ) -> Result<(Self, u64), Error> {
        let (program, mass) = Self::softmax_schedule(rows, width, input, true, true)?;
        Ok((program, mass.ok_or(Error::Binding)?))
    }

    fn softmax_schedule(
        rows: usize,
        width: usize,
        input: Interval,
        guard_tail: bool,
        certify_mass: bool,
    ) -> Result<(Self, Option<u64>), Error> {
        let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
        if width == 0 || rows == 0 {
            return Err(Error::Bounds);
        }
        let mut b = Builder::row(lanes, input)?;
        let x = b.input();
        let mut maximum = x;
        let mut offset = 1;
        while offset < width {
            let other = b.row_map(maximum, width, RowMap::Rotate(offset))?;
            let difference = b.affine(maximum, Some(other), affine(1, -1, 0))?;
            let bit = b.sign(difference)?;
            let bit = b.from_bit(bit)?;
            maximum = select(&mut b, bit, other, maximum)?;
            // The secret selector chooses one of the two candidates exactly.
            b.0.nodes[maximum.0 as usize].as_mut().unwrap().bounds = input;
            offset *= 2;
        }
        let z = b.affine(x, Some(maximum), affine(1, -1, 0))?;
        let lower = input
            .lower()
            .checked_sub(input.upper())
            .ok_or(Error::Bounds)?;
        b.0.nodes[z.0 as usize].as_mut().unwrap().bounds = Interval::new(lower, 0)?;
        let keep = if guard_tail {
            Some(softmax_keep(&mut b, z)?)
        } else {
            None
        };
        let safe_z = if let Some(keep) = keep {
            let safe = b.product(z, keep, false)?;
            // keep=1 iff z>-14*ONE; otherwise safe=0. This proof uses
            // the shared predicate, never a revealed value or private range.
            b.0.nodes[safe.0 as usize].as_mut().unwrap().bounds =
                Interval::new(lower.max(-14 * ONE + 1), 0)?;
            safe
        } else {
            z
        };
        let reduced = b.rescale(safe_z, 5)?;
        let mut exp = b.affine(reduced, None, affine(1, 0, ONE))?;
        for _ in 0..5 {
            exp = b.square(exp)?;
        }
        let keep = match keep {
            Some(keep) => keep,
            None => softmax_keep(&mut b, z)?,
        };
        let exp = b.product(exp, keep, false)?;
        let total = b.row_map(exp, width, RowMap::Sum)?;
        let upper = b.node(total, false)?.bounds.upper();
        // At least one lane has Z=0, exact exp=ONE, and survives clipping.
        let total_range = Interval::new(ONE, upper)?;
        b.0.nodes[total.0 as usize].as_mut().unwrap().bounds = total_range;
        let inverse = reciprocal(&mut b, total, total_range)?;
        let mass = if certify_mass {
            let (range, product) = reciprocal_mass_bounds(total_range)?;
            b.0.nodes[inverse.0 as usize].as_mut().unwrap().bounds = range;
            // e_i>=0, d=sum(e_i), r>=0. Each p_i=TR(e_i*r) is at
            // most (e_i*r+Q-1)/Q, so sum(p_i)<=floor((d*r+n*(Q-1))/Q).
            Some(
                u64::try_from(
                    (i128::from(product.upper()) + width as i128 * i128::from(ONE - 1))
                        / i128::from(ONE),
                )
                .map_err(|_| Error::Bounds)?,
            )
        } else {
            None
        };
        let inverse = b.row_map(inverse, width, RowMap::Broadcast)?;
        let output = b.product(exp, inverse, true)?;
        Ok((b.finish(output)?, mass))
    }

    /// Centered, averaged LayerNorm with E(1e-5)=2. Gamma/beta are explicit
    /// public per-column Q18 constants, applied in the approved two-step order.
    pub fn layer_norm(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        beta: &[i64],
    ) -> Result<Self, Error> {
        let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
        if width == 0 || rows == 0 || gamma.len() != width || beta.len() != width {
            return Err(Error::Shape);
        }
        let mut b = Builder::row(lanes, input)?;
        let x = b.input();
        let inverse_width = ((1.0_f32 / width as f32) * ONE as f32) as i64;
        let sum = b.row_map(x, width, RowMap::Sum)?;
        let raw_mean = b.affine(sum, None, affine(inverse_width, 0, 0))?;
        let mean = b.rescale(raw_mean, 18)?;
        let mean = b.row_map(mean, width, RowMap::Broadcast)?;
        let delta = b.affine(x, Some(mean), affine(1, -1, 0))?;
        let square = b.square(delta)?;
        let variance_sum = b.row_map(square, width, RowMap::Sum)?;
        let raw_variance = b.affine(variance_sum, None, affine(inverse_width, 0, 0))?;
        let variance = b.rescale(raw_variance, 18)?;
        let variance = b.affine(variance, None, affine(1, 0, 2))?;
        let bounds = b.node(variance, false)?.bounds;
        let inverse = rsqrt(&mut b, variance, bounds)?;
        let inverse = b.row_map(inverse, width, RowMap::Broadcast)?;
        let normalized = b.product(delta, inverse, true)?;
        let scales: [Affine; MAX_ROW_LANES] =
            std::array::from_fn(|i| affine(gamma[i % width], 0, 0));
        let scaled = b.affine_lanes(normalized, &scales[..lanes])?;
        let rounded = b.rescale(scaled, 18)?;
        let offsets: [Affine; MAX_ROW_LANES] =
            std::array::from_fn(|i| affine(1, 0, beta[i % width]));
        let output = b.affine_lanes(rounded, &offsets[..lanes])?;
        b.finish(output)
    }

    /// Uncentered RMSNorm under the separately named epsilon-one adaptation.
    /// Square rescaling, reciprocal-width rescaling, rsqrt, normalization and
    /// public gamma rescaling remain distinct stochastic operations. Zero rows
    /// have a positive denominator without opening or selecting on their value.
    pub fn rms_norm(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
    ) -> Result<Self, Error> {
        Self::rms_norm_proof(rows, width, input, gamma, false, None).map(|(program, _)| program)
    }

    /// The same RMSNorm arithmetic with a finer public proof of its rsqrt
    /// polynomial. Partitions are compiler-only; runtime never opens a bucket.
    /// Returns a certified squared Euclidean bound for each complete output row.
    pub fn rms_norm_certified(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
    ) -> Result<(Self, u128), Error> {
        Self::rms_norm_proof(rows, width, input, gamma, true, None)
    }

    /// Same arithmetic under an additional caller-proved upper bound on each
    /// complete input row's sum of squared encoded integers. Like `input`, this
    /// is a public range obligation; execution never reconstructs it to check.
    pub fn rms_norm_with_input_norm(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        squared_norm: u128,
    ) -> Result<(Self, u128), Error> {
        Self::rms_norm_proof(rows, width, input, gamma, true, Some(squared_norm))
    }

    fn rms_norm_proof(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        partitioned: bool,
        input_norm: Option<u128>,
    ) -> Result<(Self, u128), Error> {
        let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
        if width == 0 || rows == 0 || gamma.len() != width {
            return Err(Error::Shape);
        }
        let (mut b, normalized, radius) =
            rms_normalization(rows, width, input, partitioned, input_norm)?;
        let scales: [Affine; MAX_ROW_LANES] =
            std::array::from_fn(|i| affine(gamma[i % width], 0, 0));
        let scaled = b.affine_lanes(normalized, &scales[..lanes])?;
        let output = b.rescale(scaled, 18)?;
        let squared = if partitioned {
            let gamma = gamma
                .iter()
                .map(|v| v.unsigned_abs())
                .max()
                .ok_or(Error::Shape)?;
            // Each public-gamma rescale contributes an error of magnitude < 1
            // per lane. Minkowski and Cauchy-Schwarz retain the complete row.
            let radius = radius
                .checked_mul(u128::from(gamma))
                .ok_or(Error::Bounds)?
                .div_ceil(ONE as u128)
                .checked_add(ceil_sqrt(width as u128))
                .ok_or(Error::Bounds)?;
            radius.checked_mul(radius).ok_or(Error::Bounds)?
        } else {
            0
        };
        Ok((b.finish(output)?, squared))
    }

    pub fn negative(lanes: usize, input: Interval) -> Result<Self, Error> {
        let mut b = Builder::new(lanes, input)?;
        let bit = b.sign(b.input())?;
        let out = b.from_bit(bit)?;
        b.finish(out)
    }

    /// Arithmetic equality to zero, with every decision retained as a share.
    pub fn equal_zero(lanes: usize, input: Interval) -> Result<Self, Error> {
        let mut b = Builder::new(lanes, input)?;
        let mut bits = b.bits(b.input())?;
        for shift in [32, 16, 8, 4, 2, 1] {
            let other = b.bit_map(bits, BitMap::ShiftRight(shift))?;
            bits = b.bit_or(bits, other)?;
        }
        let bit = b.bit_map(bits, BitMap::LowBits(1))?;
        let nonzero = b.from_bit(bit)?;
        let out = b.affine(nonzero, None, affine(-1, 0, 1))?;
        b.finish(out)
    }

    /// Positive source Q18 reciprocal. Exponent selection remains shared; the
    /// native range proof covers every public input bucket and every rounding.
    pub fn reciprocal(lanes: usize, input: Interval) -> Result<Self, Error> {
        let mut b = Builder::new(lanes, input)?;
        let output = reciprocal(&mut b, Value(0), input)?;
        b.finish(output)
    }

    /// Source Q18 rsqrt, including sequential U powers, one accumulated
    /// polynomial rescale, and the 724/512 integer exponent compensation.
    pub fn rsqrt(lanes: usize, input: Interval) -> Result<Self, Error> {
        let mut b = Builder::new(lanes, input)?;
        let output = rsqrt(&mut b, Value(0), input)?;
        b.finish(output)
    }
}

fn softmax_keep(b: &mut Builder, z: Value) -> Result<Value, Error> {
    let threshold = b.affine(z, None, affine(-1, 0, -14 * ONE))?;
    let keep = b.sign(threshold)?;
    b.from_bit(keep)
}

fn rms_normalization(
    rows: usize,
    width: usize,
    input: Interval,
    partitioned: bool,
    input_norm: Option<u128>,
) -> Result<(Builder, Value, u128), Error> {
    let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
    if rows == 0 || width == 0 {
        return Err(Error::Shape);
    }
    let mut b = Builder::row(lanes, input)?;
    b.0.numeric_profile = RMS_NORM_PROFILE;
    let x = b.input();
    let square = b.square(x)?;
    let sum = b.row_map(square, width, RowMap::Sum)?;
    if let Some(squared_norm) = input_norm {
        let minimum = if input.lower() <= 0 && input.upper() >= 0 {
            0
        } else {
            input
                .lower()
                .unsigned_abs()
                .min(input.upper().unsigned_abs())
        };
        if squared_norm < u128::from(minimum).pow(2) * width as u128 {
            return Err(Error::Bounds);
        }
        // s_i=TR(x_i^2), so sum(s_i)<=floor((B+n*(Q-1))/Q).
        // Keep the exact squares, ordered sum and original mean rescale.
        let upper = squared_norm
            .checked_add(width as u128 * (ONE - 1) as u128)
            .ok_or(Error::Bounds)?
            / ONE as u128;
        let ordinary = b.node(sum, false)?.bounds;
        let upper = upper.min(ordinary.upper() as u128) as i64;
        b.0.nodes[sum.0 as usize].as_mut().unwrap().bounds =
            Interval::new(ordinary.lower(), upper)?;
    }
    let inverse_width = ((1.0_f32 / width as f32) * ONE as f32) as i64;
    if inverse_width <= 0 {
        return Err(Error::Bounds);
    }
    let raw_mean = b.affine(sum, None, affine(inverse_width, 0, 0))?;
    let mean = b.rescale(raw_mean, 18)?;
    let denominator = b.affine(mean, None, affine(1, 0, RMS_NORM_EPSILON_Q18))?;
    let bounds = b.node(denominator, false)?.bounds;
    let mut inverse = rsqrt_ranged(&mut b, denominator, bounds, partitioned)?;
    inverse.value = b.row_map(inverse.value, width, RowMap::Broadcast)?;
    // Correlate each row value with its own denominator before multiplying.
    // For Q=2^18, c=E(1/width)>0, S=sum(TR(x_i^2)), d=TR(S*c)+epsilon:
    // d<=hi => S*c < (hi-epsilon+1)*Q and x_i^2 < (S+1)*Q.
    // Every stochastic truncation is floor or ceil, so these inequalities
    // cover every mask. Buckets are public proof cases, never runtime branches.
    let denominator_boxes = Ranged::input(denominator, bounds, 35)?.boxes;
    let mut boxes = [None; DOMAINS];
    let mut radius = 0;
    for (j, bucket) in denominator_boxes.into_iter().enumerate() {
        if let Some(bucket) = bucket {
            let sum_max = ((i128::from(bucket.upper()) - i128::from(RMS_NORM_EPSILON_Q18) + 1)
                * i128::from(ONE)
                - 1)
                / i128::from(inverse_width);
            let square_max =
                u128::try_from((sum_max + 1) * i128::from(ONE) - 1).map_err(|_| Error::Bounds)?;
            let magnitude = i64::try_from(square_max.isqrt()).map_err(|_| Error::Bounds)?;
            let lower = input.lower().max(-magnitude);
            let upper = input.upper().min(magnitude);
            if lower > upper {
                // A loose denominator hull can contain an unreachable bucket.
                inverse.boxes[j] = None;
            } else {
                boxes[j] = Some(Interval::new(lower, upper)?);
                if partitioned {
                    // Sum x_i^2 <= Q*S + width*(Q-1). This is a row proof,
                    // not width copies of the maximum component proof above.
                    let sum_squares = u128::try_from(
                        sum_max * i128::from(ONE) + width as i128 * i128::from(ONE - 1),
                    )
                    .map_err(|_| Error::Bounds)?;
                    let magnitude = u128::from(
                        input
                            .lower()
                            .unsigned_abs()
                            .max(input.upper().unsigned_abs()),
                    );
                    let sum_squares = sum_squares.min(
                        magnitude
                            .checked_mul(magnitude)
                            .and_then(|v| v.checked_mul(width as u128))
                            .ok_or(Error::Bounds)?,
                    );
                    let inverse = inverse.boxes[j].ok_or(Error::Bounds)?;
                    let inverse = u128::from(
                        inverse
                            .lower()
                            .unsigned_abs()
                            .max(inverse.upper().unsigned_abs()),
                    );
                    let square = sum_squares
                        .checked_mul(inverse)
                        .and_then(|v| v.checked_mul(inverse))
                        .ok_or(Error::Bounds)?;
                    let bound = ceil_sqrt(square)
                        .div_ceil(ONE as u128)
                        .checked_add(ceil_sqrt(width as u128))
                        .ok_or(Error::Bounds)?;
                    radius = radius.max(bound);
                }
            }
        }
    }
    let normalized = Ranged { value: x, boxes }
        .product(&mut b, &inverse, true)?
        .value;
    Ok((b, normalized, radius))
}

fn ceil_sqrt(value: u128) -> u128 {
    let root = value.isqrt();
    root + u128::from(root * root != value)
}

pub(super) fn reciprocal(b: &mut Builder, input: Value, bounds: Interval) -> Result<Value, Error> {
    let x = Ranged::input(input, bounds, 36)?;
    let highest = b.highest_bit(input, 36)?;
    let reversed = b.bit_map(highest, BitMap::ReverseLowBits(36))?;
    let factor = b.from_boolean(reversed)?;
    let factor = x.exact_map(b, factor, |j| 1 << (35 - j))?;
    let c = x.product(b, &factor, true)?;
    let guess = c.affine(b, None, affine(-2, 0, 763940))?;
    let w = guess.product(b, &factor, true)?;
    let xw = x.product(b, &w, true)?;
    let e = xw.affine(b, None, affine(-1, 0, ONE))?;
    let one_e = e.affine(b, None, affine(1, 0, ONE))?;
    let r = w.product(b, &one_e, true)?;
    let e2 = e.product(b, &e, true)?;
    let one_e2 = e2.affine(b, None, affine(1, 0, ONE))?;
    Ok(r.product(b, &one_e2, true)?.value)
}

/// The six source reciprocal truncations over bounded public subintervals.
/// These are proof cells only; the executable highest-bit circuit is unchanged.
/// Splitting each exponent bucket into at most 256 cells retains the relation
/// between d and r(d), instead of multiplying unrelated whole-domain extrema.
fn reciprocal_mass_bounds(bounds: Interval) -> Result<(Interval, Interval), Error> {
    if bounds.lower() < ONE || bounds.upper() >= 1_i64 << 36 {
        return Err(Error::Bounds);
    }
    let mut inverse_lower = i64::MAX;
    let mut inverse_upper = i64::MIN;
    let mut product_lower = i64::MAX;
    let mut product_upper = i64::MIN;
    for j in 18..36 {
        let lower = bounds.lower().max(1 << j);
        let upper = bounds.upper().min((1 << (j + 1)) - 1);
        if lower > upper {
            continue;
        }
        let step = (upper - lower + 256) / 256;
        for start in (lower..=upper).step_by(step as usize) {
            let d = Interval::new(start, upper.min(start + step - 1))?;
            let r = reciprocal_cell(d, j)?;
            let product = InputBounds { left: d, right: r }.product()?;
            inverse_lower = inverse_lower.min(r.lower());
            inverse_upper = inverse_upper.max(r.upper());
            product_lower = product_lower.min(product.lower());
            product_upper = product_upper.max(product.upper());
        }
    }
    if inverse_lower < 0 {
        return Err(Error::Bounds);
    }
    Ok((
        Interval::new(inverse_lower, inverse_upper)?,
        Interval::new(product_lower, product_upper)?,
    ))
}

fn reciprocal_cell(d: Interval, exponent: usize) -> Result<Interval, Error> {
    let multiply = |left, right| InputBounds { left, right }.output();
    let linear = |input, coefficient, constant| {
        affine(coefficient, 0, constant).output_bounds(InputBounds {
            left: input,
            right: Interval::new(0, 0)?,
        })
    };
    let factor = Interval::new(1 << (35 - exponent), 1 << (35 - exponent))?;
    let c = multiply(d, factor)?;
    let guess = linear(c, -2, 763940)?;
    let w = multiply(guess, factor)?;
    let xw = multiply(d, w)?;
    let e = linear(xw, -1, ONE)?;
    let r = multiply(w, linear(e, 1, ONE)?)?;
    let e2 = multiply(e, e)?;
    multiply(r, linear(e2, 1, ONE)?)
}

pub(super) fn rsqrt(b: &mut Builder, input: Value, bounds: Interval) -> Result<Value, Error> {
    Ok(rsqrt_ranged(b, input, bounds, false)?.value)
}

fn rsqrt_ranged(
    b: &mut Builder,
    input: Value,
    bounds: Interval,
    partitioned: bool,
) -> Result<Ranged, Error> {
    let x = Ranged::input(input, bounds, 35)?;
    let highest = b.highest_bit(input, 35)?;
    let z = b.bit_map(highest, BitMap::ShiftLeft(1))?;
    let reversed = b.bit_map(z, BitMap::ReverseLowBits(36))?;
    let factor = b.from_boolean(reversed)?;
    let factor = x.exact_map(b, factor, |j| 1 << (34 - j))?;
    let u = x.product(b, &factor, true)?;
    let u2 = u.product(b, &u, true)?;
    let u3 = u2.product(b, &u, true)?;
    let u4 = u3.product(b, &u, true)?;
    let mut raw = u.affine(b, None, affine(-4057974, 0, 0))?;
    for (power, coefficient) in [(&u2, 10085068), (&u3, -13072088), (&u4, 6823457)] {
        raw = raw.affine(b, Some(power), affine(1, coefficient, 0))?;
    }
    let mut p = raw.rescale(b, 18)?.affine(b, None, affine(1, 0, 1086023))?;
    if partitioned {
        for (slot, domain) in p.boxes.iter_mut().zip(u.boxes) {
            *slot = domain.map(rsqrt_polynomial_bounds).transpose()?;
        }
        b.0.nodes[p.value.0 as usize].as_mut().unwrap().bounds = hull(&p.boxes)?;
    }

    // XOR-linear pair folding and parity work on shares; the one-hot invariant
    // follows from the fixed highest-bit circuit, never a caller assertion.
    let parity = b.bit_map(z, BitMap::Parity(0x5555_5555_5555_5555))?;
    let parity = b.from_bit(parity)?;
    let parity = x.exact_map(b, parity, |j| i64::from((j + 1) % 2 == 0))?;
    let coefficient = parity.affine(b, None, affine(212, 0, 512))?;
    let folded = b.bit_map(z, BitMap::FoldPairs)?;
    let reversed = b.bit_map(folded, BitMap::ReverseLowBits(18))?;
    let scale = b.from_boolean(reversed)?;
    let scale = x.exact_map(b, scale, |j| 1 << (17 - (j + 1) / 2))?;
    let compensation = coefficient.product(b, &scale, false)?;
    p.product(b, &compensation, true)
}

/// Cover every normalized integer U and every floor/ceil choice at all three
/// powers and the final polynomial rescale. Within each 256-word public cell,
/// ordinary interval arithmetic is sound; unioning cells retains their shared U.
fn rsqrt_polynomial_bounds(input: Interval) -> Result<Interval, Error> {
    if input.lower() < ONE / 4 || input.upper() > ONE / 2 {
        return Err(Error::Bounds);
    }
    let mut lower = i64::MAX;
    let mut upper = i64::MIN;
    for start in (input.lower()..=input.upper()).step_by(256) {
        let u = Interval::new(start, input.upper().min(start + 255))?;
        let u2 = InputBounds { left: u, right: u }.output()?;
        let u3 = InputBounds { left: u2, right: u }.output()?;
        let u4 = InputBounds { left: u3, right: u }.output()?;
        let zero = Interval::new(0, 0)?;
        let mut raw = affine(-4057974, 0, 0).output_bounds(InputBounds {
            left: u,
            right: zero,
        })?;
        for (power, coefficient) in [(u2, 10085068), (u3, -13072088), (u4, 6823457)] {
            raw = affine(1, coefficient, 0).output_bounds(InputBounds {
                left: raw,
                right: power,
            })?;
        }
        let p = affine(1, 0, 1086023).output_bounds(InputBounds {
            left: rescaled_bounds(raw, 18)?,
            right: zero,
        })?;
        lower = lower.min(p.lower());
        upper = upper.max(p.upper());
    }
    Interval::new(lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_mass_proof_covers_denominators_and_all_rounding_choices() {
        let choices = |v: i128| {
            let q = i128::from(ONE);
            [v.div_euclid(q) as i64, (-(-v).div_euclid(q)) as i64]
        };
        for j in 18..36 {
            let lo = 1_i64 << j;
            let hi = (1_i64 << (j + 1)) - 1;
            let (inverse, product) =
                reciprocal_mass_bounds(Interval::new(lo, hi).unwrap()).unwrap();
            // Exhaust the first exponent bucket. Other exponents cover endpoints
            // and non-aligned public cells; every case checks all 64 roundings.
            let stride = if j == 18 { 1 } else { (hi - lo) / 257 };
            for d in (lo..=hi).step_by(stride as usize).chain([hi - 1, hi]) {
                let factor = 1_i128 << (35 - j);
                for c in choices(i128::from(d) * factor) {
                    for w in choices((763_940 - 2 * i128::from(c)) * factor) {
                        for xw in choices(i128::from(d) * i128::from(w)) {
                            let e = i128::from(ONE - xw);
                            for r in choices(i128::from(w) * (i128::from(ONE) + e)) {
                                for e2 in choices(e * e) {
                                    for out in choices(i128::from(r) * i128::from(ONE + e2)) {
                                        assert!((inverse.lower()..=inverse.upper()).contains(&out));
                                        let raw = d * out;
                                        assert!((product.lower()..=product.upper()).contains(&raw));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(reciprocal_mass_bounds(Interval::new(ONE - 1, ONE).unwrap()).is_err());
        assert!(reciprocal_mass_bounds(Interval::new(ONE, 1 << 36).unwrap()).is_err());
    }

    #[test]
    fn normalized_rsqrt_proof_covers_every_integer_and_rounding_choice() {
        let whole = rsqrt_polynomial_bounds(Interval::new(ONE / 4, ONE / 2).unwrap()).unwrap();
        assert!(whole.lower() > ONE && whole.upper() < 3 * ONE);
        let choices = |v: i128| {
            let q = i128::from(ONE);
            [v.div_euclid(q) as i64, (-(-v).div_euclid(q)) as i64]
        };
        // Different cell boundaries from the implementation; every integer U
        // and all 16 stochastic floor/ceil combinations are independently checked.
        for start in (ONE / 4..=ONE / 2).step_by(251) {
            let end = (start + 250).min(ONE / 2);
            let bound = rsqrt_polynomial_bounds(Interval::new(start, end).unwrap()).unwrap();
            for u in start..=end {
                let u = i128::from(u);
                for u2 in choices(u * u) {
                    for u3 in choices(i128::from(u2) * u) {
                        for u4 in choices(i128::from(u3) * u) {
                            let numerator = -4_057_974 * u + 10_085_068 * i128::from(u2)
                                - 13_072_088 * i128::from(u3)
                                + 6_823_457 * i128::from(u4);
                            for scaled in choices(numerator) {
                                let value = scaled + 1_086_023;
                                assert!((bound.lower()..=bound.upper()).contains(&value));
                                assert!((whole.lower()..=whole.upper()).contains(&value));
                            }
                        }
                    }
                }
            }
        }
        for (lo, hi) in [(0, ONE / 2), (ONE / 4 - 1, ONE / 2), (ONE / 4, ONE / 2 + 1)] {
            assert!(rsqrt_polynomial_bounds(Interval::new(lo, hi).unwrap()).is_err());
        }
    }
}
