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
    /// Shifted, strictly clipped P-N1 softmax. Every eagerly evaluated power is
    /// admitted, including the lanes subsequently masked below -14.
    pub fn softmax(rows: usize, width: usize, input: Interval) -> Result<Self, Error> {
        let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
        if width == 0 || rows == 0 {
            return Err(Error::Bounds);
        }
        let mut b = Builder::new(lanes, input)?;
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
        let reduced = b.rescale(z, 5)?;
        let mut exp = b.affine(reduced, None, affine(1, 0, ONE))?;
        for _ in 0..5 {
            exp = b.square(exp)?;
        }
        let threshold = b.affine(z, None, affine(-1, 0, -14 * ONE))?;
        let keep = b.sign(threshold)?;
        let keep = b.from_bit(keep)?;
        let exp = b.product(exp, keep, false)?;
        let total = b.row_map(exp, width, RowMap::Sum)?;
        let upper = b.node(total, false)?.bounds.upper();
        // At least one lane has Z=0, exact exp=ONE, and survives clipping.
        let total_range = Interval::new(ONE, upper)?;
        b.0.nodes[total.0 as usize].as_mut().unwrap().bounds = total_range;
        let inverse = reciprocal(&mut b, total, total_range)?;
        let inverse = b.row_map(inverse, width, RowMap::Broadcast)?;
        let output = b.product(exp, inverse, true)?;
        b.finish(output)
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
        let mut b = Builder::new(lanes, input)?;
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
        let scales: [Affine; MAX_TILE_LANES] =
            std::array::from_fn(|i| affine(gamma[i % width], 0, 0));
        let scaled = b.affine_lanes(normalized, &scales[..lanes])?;
        let rounded = b.rescale(scaled, 18)?;
        let offsets: [Affine; MAX_TILE_LANES] =
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
        let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
        if width == 0 || rows == 0 || gamma.len() != width {
            return Err(Error::Shape);
        }
        let mut b = Builder::new(lanes, input)?;
        b.0.numeric_profile = RMS_NORM_PROFILE;
        let x = b.input();
        let square = b.square(x)?;
        let sum = b.row_map(square, width, RowMap::Sum)?;
        let inverse_width = ((1.0_f32 / width as f32) * ONE as f32) as i64;
        let raw_mean = b.affine(sum, None, affine(inverse_width, 0, 0))?;
        let mean = b.rescale(raw_mean, 18)?;
        let denominator = b.affine(mean, None, affine(1, 0, RMS_NORM_EPSILON_Q18))?;
        let bounds = b.node(denominator, false)?.bounds;
        let inverse = rsqrt(&mut b, denominator, bounds)?;
        let inverse = b.row_map(inverse, width, RowMap::Broadcast)?;
        let normalized = b.product(x, inverse, true)?;
        let scales: [Affine; MAX_TILE_LANES] =
            std::array::from_fn(|i| affine(gamma[i % width], 0, 0));
        let scaled = b.affine_lanes(normalized, &scales[..lanes])?;
        let output = b.rescale(scaled, 18)?;
        b.finish(output)
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

pub(super) fn rsqrt(b: &mut Builder, input: Value, bounds: Interval) -> Result<Value, Error> {
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
    let p = raw.rescale(b, 18)?.affine(b, None, affine(1, 0, 1086023))?;

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
    Ok(p.product(b, &compensation, true)?.value)
}
