use super::*;
use num_bigint::{BigInt, BigUint, Sign};
use std::cmp::Ordering;

// Independent oracle: exact signed bigint rationals on the universal 2^-149
// grid. Rounding searches ALL target encodings (BF16 or F32), with the exact
// virtual 2^128 successor at overflow. No candidate primitive/helper supplies an
// expected bit, and there is no host floating arithmetic, libm or new dependency.
fn grid(bits: u32) -> BigUint {
    assert!(bits <= 0x7f80_0000);
    if bits == 0x7f80_0000 {
        return BigUint::from(1_u8) << 277_usize;
    }
    let exponent = bits / 0x0080_0000;
    let fraction = bits % 0x0080_0000;
    if exponent == 0 {
        BigUint::from(fraction)
    } else {
        BigUint::from(fraction + 0x0080_0000) << (exponent - 1) as usize
    }
}

fn signed_grid(bits: u32) -> Result<BigInt, Error> {
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return Err(Error::NonFinite);
    }
    Ok(BigInt::from_biguint(
        if bits >> 31 == 0 {
            Sign::Plus
        } else {
            Sign::Minus
        },
        grid(bits & 0x7fff_ffff),
    ))
}

fn promoted(word: u16) -> Result<u32, Error> {
    let bits = u32::from(word) << 16;
    signed_grid(bits)?;
    Ok(bits)
}

fn round_fraction(
    numerator: BigInt,
    denominator: BigUint,
    bf16: bool,
    negative_zero: bool,
) -> Result<u32, Error> {
    assert_ne!(denominator, BigUint::from(0_u8));
    let negative =
        numerator.sign() == Sign::Minus || (numerator.sign() == Sign::NoSign && negative_zero);
    let sign = u32::from(negative) << 31;
    if numerator.sign() == Sign::NoSign {
        return Ok(sign);
    }
    let shift = if bf16 { 16 } else { 0 };
    let infinity = 0x7f80_0000 >> shift;
    let scaled = numerator.magnitude() << 149_usize;
    let doubled = &scaled << 1_usize;
    let overflow_midpoint =
        (grid((infinity - 1) << shift) + grid(infinity << shift)) * &denominator;
    if doubled >= overflow_midpoint {
        return Err(Error::NonFinite);
    }
    let mut lo = 0_u32;
    let mut hi = infinity;
    while hi - lo > 1 {
        let middle = lo + (hi - lo) / 2;
        if grid(middle << shift) * &denominator <= scaled {
            lo = middle;
        } else {
            hi = middle;
        }
    }
    let midpoint = (grid(lo << shift) + grid(hi << shift)) * denominator;
    let chosen = match doubled.cmp(&midpoint) {
        Ordering::Less => lo,
        Ordering::Greater => hi,
        Ordering::Equal => {
            if lo & 1 == 0 {
                lo
            } else {
                hi
            }
        }
    };
    Ok(sign | (chosen << shift))
}

#[derive(Clone, Copy)]
enum Op {
    Add,
    Mul,
    Div,
}

fn arithmetic(op: Op, a: u32, b: u32) -> Result<u32, Error> {
    let left = signed_grid(a)?;
    let right = signed_grid(b)?;
    let one = BigUint::from(1_u8);
    let (numerator, denominator, negative_zero) = match op {
        Op::Add => (left + right, one << 149_usize, a & b & 0x8000_0000 != 0),
        Op::Mul => (left * right, one << 298_usize, (a ^ b) >> 31 != 0),
        Op::Div => {
            if right.sign() == Sign::NoSign {
                return Err(Error::NonFinite);
            }
            let numerator = if right.sign() == Sign::Minus {
                -left
            } else {
                left
            };
            (numerator, right.magnitude().clone(), (a ^ b) >> 31 != 0)
        }
    };
    round_fraction(numerator, denominator, false, negative_zero)
}

fn cast(bits: u32) -> Result<u16, Error> {
    Ok((round_fraction(
        signed_grid(bits)?,
        BigUint::from(1_u8) << 149_usize,
        true,
        bits >> 31 != 0,
    )? >> 16) as u16)
}

fn reciprocal_root(a: u32) -> u32 {
    assert!(a > 0 && a < 0x7f80_0000);
    let source = grid(a);
    let one = BigUint::from(1_u8) << 447_usize;
    // Independent full-encoding search, not the candidate's normal bracket.
    let mut lo = 0_u32;
    let mut hi = 0x7f80_0000_u32;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let value = grid(mid);
        if &source * &value * value <= one {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let midpoint = grid(lo) + grid(hi);
    match (&source * &midpoint * midpoint).cmp(&(BigUint::from(1_u8) << 449_usize)) {
        Ordering::Less => hi,
        Ordering::Greater => lo,
        Ordering::Equal => {
            if lo & 1 == 0 {
                lo
            } else {
                hi
            }
        }
    }
}

struct Trace {
    width: u32,
    squares: Vec<u32>,
    sums: Vec<u32>,
    mean: u32,
    adjusted: u32,
    normalizer: u32,
    normalized: Vec<u32>,
    casts: Vec<u16>,
    products: Vec<u32>,
    output: Vec<u16>,
}

fn reference(input: &[u16], gamma: &[u16]) -> Trace {
    assert!(!input.is_empty());
    assert_eq!(input.len(), gamma.len());
    let width =
        round_fraction(BigInt::from(input.len()), BigUint::from(1_u8), false, false).unwrap();
    let mut squares = Vec::new();
    let mut sums = Vec::new();
    let mut sum = 0;
    for &word in input {
        let value = promoted(word).unwrap();
        let square = arithmetic(Op::Mul, value, value).unwrap();
        sum = arithmetic(Op::Add, sum, square).unwrap();
        squares.push(square);
        sums.push(sum);
    }
    let mean = arithmetic(Op::Div, sum, width).unwrap();
    let adjusted = arithmetic(Op::Add, mean, 0x3586_37bd).unwrap();
    let normalizer = reciprocal_root(adjusted);
    let mut normalized = Vec::new();
    let mut casts = Vec::new();
    let mut products = Vec::new();
    let mut output = Vec::new();
    for (&word, &weight) in input.iter().zip(gamma) {
        let value = arithmetic(Op::Mul, promoted(word).unwrap(), normalizer).unwrap();
        let before_gamma = cast(value).unwrap();
        let product = arithmetic(
            Op::Mul,
            promoted(before_gamma).unwrap(),
            promoted(weight).unwrap(),
        )
        .unwrap();
        normalized.push(value);
        casts.push(before_gamma);
        products.push(product);
        output.push(cast(product).unwrap());
    }
    Trace {
        width,
        squares,
        sums,
        mean,
        adjusted,
        normalizer,
        normalized,
        casts,
        products,
        output,
    }
}

// Every stage is checked against independent expected bits, then the complete
// helper is compared with that SAME independently constructed trace. Candidate
// arithmetic is never fed back into reference generation.
fn certify_row(input: &[u16], gamma: &[u16]) -> Trace {
    let expected = reference(input, gamma);
    let mut sum = f32::from_bits(0);
    for (j, &word) in input.iter().enumerate() {
        let value = promote_bf16(word).unwrap();
        assert_eq!(value.to_bits(), promoted(word).unwrap());
        let square = mul_rn32(value, value).unwrap();
        assert_eq!(square.to_bits(), expected.squares[j]);
        sum = add_rn32(sum, square).unwrap();
        assert_eq!(sum.to_bits(), expected.sums[j]);
    }
    let mean = div_rn32(sum, f32::from_bits(expected.width)).unwrap();
    assert_eq!(mean.to_bits(), expected.mean);
    let adjusted = add_rn32(mean, f32::from_bits(EPSILON_BITS)).unwrap();
    assert_eq!(adjusted.to_bits(), expected.adjusted);
    let normalizer = rsqrt_rn32(adjusted).unwrap();
    assert_eq!(normalizer.to_bits(), expected.normalizer);
    for (j, (&word, &weight)) in input.iter().zip(gamma).enumerate() {
        let normalized = mul_rn32(promote_bf16(word).unwrap(), normalizer).unwrap();
        assert_eq!(normalized.to_bits(), expected.normalized[j]);
        let before_gamma = f32_to_bf16(normalized).unwrap();
        assert_eq!(before_gamma, expected.casts[j]);
        assert_eq!(
            promote_bf16(before_gamma).unwrap().to_bits(),
            promoted(before_gamma).unwrap()
        );
        assert_eq!(
            promote_bf16(weight).unwrap().to_bits(),
            promoted(weight).unwrap()
        );
        let product = mul_rn32(
            promote_bf16(before_gamma).unwrap(),
            promote_bf16(weight).unwrap(),
        )
        .unwrap();
        assert_eq!(product.to_bits(), expected.products[j]);
        assert_eq!(f32_to_bf16(product).unwrap(), expected.output[j]);
    }
    let mut output = vec![0x5555; input.len()];
    rms_norm_bf16_row_into(input, gamma, &mut output).unwrap();
    assert_eq!(output, expected.output);
    expected
}

#[test]
fn rms_norm_integer_width_preflight_and_untouched_shapes() {
    for (width, bits) in [
        (1, 0x3f80_0000),
        (3, 0x4040_0000),
        (128, 0x4300_0000),
        (2560, 0x4520_0000),
        (16_777_216, 0x4b80_0000),
        (16_777_218, 0x4b80_0001),
    ] {
        assert_eq!(exact_width(width).unwrap().to_bits(), bits);
    }
    for width in [16_777_217, 16_777_219, 33_554_434] {
        assert_eq!(preflight(width, width, width), Err(Error::IntegerRange));
    }
    assert_eq!(preflight(0, 0, 0), Err(Error::Geometry));
    // Invalid data would fail on the first visit; shape error has precedence.
    for (input, gamma, output_len, error) in [
        (&[][..], &[0x7fc1][..], 2, Error::Geometry),
        (&[0x7fc1][..], &[][..], 2, Error::Length),
        (&[0x7fc1][..], &[0x7fc1][..], 2, Error::Length),
        (&[0x7fc1; 2][..], &[0x7fc1; 2][..], 1, Error::Length),
    ] {
        let mut output = [0x5555; 2];
        assert_eq!(
            rms_norm_bf16_row_into(input, gamma, &mut output[..output_len]),
            Err(error)
        );
        assert_eq!(output, [0x5555; 2]);
    }
}

#[test]
fn rms_norm_work_quote_inventory_fit_short_and_overflow() {
    assert_eq!(FINITE_R32_WORK, 8192);
    assert_eq!(RSQRT_WORK, 4096);
    assert_eq!(OTHER_PER_COORDINATE, 256 + 128 + 256 + 128 + 256);
    assert_eq!(OTHER_FIXED, 4096);
    for width in [1_usize, 3, 128, 2560] {
        let primitives = (4 * width + 2) * 8192 + 4096;
        let other = 1024 * width + 4096;
        let required = rms_norm_bf16_row_work(width).unwrap();
        assert_eq!(required, 33_792 * width + 24_576);
        assert_eq!(required, primitives + other);
        // Fit/short are quote DEFINITIONS for a future caller, not an admission
        // test or a claim that the no-budget numeric helper charges a ledger.
        assert!(primitives + other <= required);
        assert!(primitives + other > required - 1);
    }
    assert_eq!(rms_norm_bf16_row_work(128), Ok(4_349_952));
    assert_eq!(rms_norm_bf16_row_work(0), Err(Error::Geometry));
    assert_eq!(rms_norm_bf16_row_work(16_777_217), Err(Error::IntegerRange));
    let enormous = 1_usize << (usize::BITS - 1);
    assert_eq!(
        preflight(enormous, enormous, enormous),
        Err(Error::WorkLimit)
    );
    assert_eq!(rms_norm_bf16_row_work(enormous), Err(Error::WorkLimit));
}

#[test]
fn rms_norm_bigint_every_stage_cast_before_gamma_discriminator() {
    let expected = certify_row(&[0x3f80, 0x4000], &[0x3f82, 0x3f80]);
    assert_eq!(expected.casts[0], 0x3f22); // 81/128
    assert_eq!(expected.output, [0x3f25, 0x3fa2]);
    // Cast path: (81/128)*(65/64) = 5265/8192, above 5264/8192 midpoint.
    // Without the intermediate cast the independent rational oracle lands below
    // that midpoint. Neither expected branch is generated by the row helper.
    let no_cast = cast(arithmetic(Op::Mul, expected.normalized[0], 0x3f82_0000).unwrap()).unwrap();
    assert_eq!(no_cast, 0x3f24);
    assert_ne!(no_cast, expected.output[0]);
    certify_row(&[0xbf80, 0x4040, 0x3e80], &[0x3fc0, 0xbf00, 0x4000]);
}

#[test]
fn rms_norm_bigint_subnormal_signed_zero_and_epsilon_stages() {
    assert_eq!(
        round_fraction(
            BigInt::from(1_u8),
            BigUint::from(1_000_000_u32),
            false,
            false
        )
        .unwrap(),
        EPSILON_BITS
    );
    assert_eq!(reciprocal_root(EPSILON_BITS), 0x447a_0000);
    let tiny = certify_row(&[1, 0x8001, 0, 0x8000], &[0x3f80, 0x3f80, 0xbf80, 0xbf80]);
    assert_eq!(tiny.squares, [0; 4]);
    assert_eq!(tiny.mean, 0);
    assert_eq!(tiny.adjusted, EPSILON_BITS);
    assert_eq!(tiny.normalizer, 0x447a_0000);
    assert_eq!(tiny.output, [0x01fa, 0x81fa, 0x8000, 0]);
    let subnormal_square = certify_row(&[0x1f80], &[0x3f80]);
    assert_eq!(subnormal_square.squares, [0x0020_0000]);
    let subnormal_product = certify_row(&[1, 0x8001], &[0x3a80, 0x3a00]);
    assert!(subnormal_product
        .products
        .iter()
        .all(|&bits| bits & 0x7fff_ffff < 0x0080_0000));
    assert_eq!(subnormal_product.output, [1, 0x8000]);
    for width in [128, 2560] {
        let zero = certify_row(&vec![0; width], &vec![0x3f80; width]);
        assert_eq!(zero.adjusted, EPSILON_BITS);
        assert_eq!(zero.normalizer, 0x447a_0000);
        assert!(zero.output.iter().all(|&word| word == 0));
    }
}

#[test]
fn rms_norm_coordinate_order_literal_sum_discriminator() {
    // Each small square is 1, exactly half an ULP at 2^24: starting with the
    // large square loses every one by even rounding. Reordering collects 4128
    // first. 4129 > (257/4)^2, but 4129/(1+4128/2^24) < (257/4)^2.
    // The independent rounded stages certify that the BF16 midpoint 64.25
    // remains crossed even after RN32 division, epsilon, rsqrt and multiplication.
    let mut input = vec![0x3f80; 4129];
    input[0] = 0x4580; // 4096, square 2^24.
    let gamma = vec![0x3f80; input.len()];
    let first = certify_row(&input, &gamma);
    assert_eq!(first.sums.last(), Some(&0x4b80_0000));
    assert_eq!(first.output[0], 0x4281); // 64.5
    assert!(first.output[1..].iter().all(|&word| word == 0x3c81));
    input.rotate_left(1);
    let last = certify_row(&input, &gamma);
    assert_eq!(last.sums.last(), Some(&0x4b80_0810));
    assert_eq!(last.output[4128], 0x4280); // 64
    assert!(last.output[..4128].iter().all(|&word| word == 0x3c80));
}

#[test]
fn rms_norm_late_gamma_failure_wipes_entire_row_and_suffix() {
    for invalid in [0x7f80, 0xff80, 0x7f81, 0x7fc1, 0xffff] {
        for position in [0, 1, 2] {
            let mut gamma = [0x3f80; 3];
            gamma[position] = invalid;
            let mut output = [0x5555; 3];
            assert_eq!(
                rms_norm_bf16_row_into(&[0x3f80; 3], &gamma, &mut output),
                Err(Error::NonFinite)
            );
            assert_eq!(output, [0; 3]);
        }
    }
    // Valid input and gamma, but the late final multiplication overflows.
    let valid = reference(&[0x3f80, 0x4000], &[0x3f80; 2]);
    assert_eq!(
        arithmetic(
            Op::Mul,
            promoted(valid.casts[1]).unwrap(),
            promoted(0x7f7f).unwrap()
        ),
        Err(Error::NonFinite)
    );
    let mut output = [0x5555; 2];
    assert_eq!(
        rms_norm_bf16_row_into(&[0x3f80, 0x4000], &[0x3f80, 0x7f7f], &mut output),
        Err(Error::NonFinite)
    );
    assert_eq!(output, [0; 2]);
    let mut backing = [0x5555; 5];
    assert_eq!(
        rms_norm_bf16_row_into(&[0x3f80; 3], &[0x3f80, 0x3f80, 0x7fc1], &mut backing[1..4]),
        Err(Error::NonFinite)
    );
    assert_eq!(backing, [0x5555, 0, 0, 0, 0x5555]);
}

#[test]
fn rms_norm_late_bf16_cast_overflow_wipes_full_row() {
    let input = [0x42fe, 0x4300]; // [127,128], second cast normalizer product >1.
    let valid = reference(&input, &[0x3f80; 2]);
    assert_eq!(valid.casts[1], 0x3f81); // 129/128

    // (254/128 * 2^127)*(129/128) is finite F32, but rounds to BF16 infinity.
    let product = arithmetic(
        Op::Mul,
        promoted(valid.casts[1]).unwrap(),
        promoted(0x7f7e).unwrap(),
    )
    .unwrap();
    assert_eq!(product, 0x7f7f_fc00);
    assert_eq!(cast(product), Err(Error::NonFinite));
    let mut output = [0x5555; 2];
    assert_eq!(
        rms_norm_bf16_row_into(&input, &[0x3f80, 0x7f7e], &mut output),
        Err(Error::NonFinite)
    );
    assert_eq!(output, [0; 2]);
}

#[test]
fn rms_norm_input_and_arithmetic_failures_wipe_unwritten_output() {
    assert_eq!(
        arithmetic(Op::Mul, 0x7f7f_0000, 0x7f7f_0000),
        Err(Error::NonFinite)
    );
    let square = arithmetic(Op::Mul, 0x5f00_0000, 0x5f00_0000).unwrap();
    assert_eq!(square, 0x7e80_0000); // 2^126
    let twice = arithmetic(Op::Add, square, square).unwrap();
    let thrice = arithmetic(Op::Add, twice, square).unwrap();
    assert_eq!(arithmetic(Op::Add, thrice, square), Err(Error::NonFinite));
    for input in [
        [0x7fc1, 0x3f80, 0x3f80, 0x3f80],
        [0x3f80, 0x3f80, 0x3f80, 0xff80],
        [0x7f7f, 0x3f80, 0x3f80, 0x3f80], // square overflow
        [0x5f00; 4],                      // each square 2^126 finite; ordered sum overflows
    ] {
        let mut output = [0x5555; 4];
        assert_eq!(
            rms_norm_bf16_row_into(&input, &[0x3f80; 4], &mut output),
            Err(Error::NonFinite)
        );
        assert_eq!(output, [0; 4]);
    }
}

#[test]
fn rms_norm_bigint_cast_midpoints_and_signed_underflow() {
    for (bits, word) in [
        (0x3f80_8000, 0x3f80),
        (0x3f81_8000, 0x3f82),
        (0x0000_8000, 0),
        (0x8000_8000, 0x8000),
        (0x0001_8000, 2),
        (0x8001_8000, 0x8002),
    ] {
        assert_eq!(cast(bits), Ok(word));
        assert_eq!(f32_to_bf16(f32::from_bits(bits)), Ok(word));
    }
}

// Fallible envelope oracle: independent exact rational arithmetic, including
// both casts and refusal paths. The successful-row reference above deliberately
// retains its unwraps; it is not used to predict envelope failures.
fn envelope_reference(
    width: usize,
    input_abs: u16,
    gamma_abs: u16,
    work_limit: usize,
) -> Result<u16, Error> {
    if width == 0 {
        return Err(Error::Geometry);
    }
    let width_bits = round_fraction(BigInt::from(width), BigUint::from(1_u8), false, false)?;
    if grid(width_bits) != (BigUint::from(width) << 149_usize) {
        return Err(Error::IntegerRange);
    }
    let required = width
        .checked_mul(33_792)
        .and_then(|n| n.checked_add(24_576))
        .ok_or(Error::WorkLimit)?;
    if required > work_limit {
        return Err(Error::WorkLimit);
    }
    let magnitude = |word: u16| -> Result<u32, Error> {
        let bits = promoted(word)?;
        if signed_grid(bits)?.sign() == Sign::Minus {
            return Err(Error::IntegerRange);
        }
        Ok(bits & 0x7fff_ffff)
    };
    let h = magnitude(input_abs)?;
    let g = magnitude(gamma_abs)?;
    let square = arithmetic(Op::Mul, h, h)?;
    let mut sum = 0;
    for _ in 0..width {
        sum = arithmetic(Op::Add, sum, square)?;
    }
    let mean = arithmetic(Op::Div, sum, width_bits)?;
    arithmetic(Op::Add, mean, 0x3586_37bd)?;
    let normalized = arithmetic(Op::Mul, h, reciprocal_root(0x3586_37bd))?;
    let first_cast = cast(normalized)?;
    let product = arithmetic(Op::Mul, promoted(first_cast)?, g)?;
    Ok(cast(product)? & 0x7fff)
}

#[test]
fn public_bound_independent_oracle() {
    assert_eq!(reciprocal_root(0x3586_37bd), 0x447a_0000);
    for width in [1, 2, 3, 17] {
        let work = crate::field_float::rms_norm_bf16_bound_work(width).unwrap();
        for h in [0, 0x8000, 1, 0x007f, 0x0080, 0x1f80, 0x3a84, 0x3f81] {
            for g in [0, 0x8000, 1, 0x007f, 0x0080, 0x3a80, 0x3f82] {
                let expected = envelope_reference(width, h, g, work);
                assert_eq!(
                    crate::field_float::rms_norm_bf16_bound(width, h, g, work),
                    expected,
                    "width={width}, H={h:04x}, G={g:04x}"
                );
                let n = expected.unwrap();
                assert_eq!(n & 0x8000, 0);
                assert!(n < 0x7f80);
            }
        }
    }
    // Subnormal promotion must not flush to zero; both BF16 casts matter.
    let work = rms_norm_bf16_bound_work(1).unwrap();
    assert_eq!(rms_norm_bf16_bound(1, 1, 0x3f80, work), Ok(0x01fa));
    assert_eq!(rms_norm_bf16_bound(1, 1, 0x3a80, work), Ok(1));
    let normalized = arithmetic(Op::Mul, promoted(0x3a84).unwrap(), 0x447a_0000).unwrap();
    assert_ne!(normalized, promoted(cast(normalized).unwrap()).unwrap());
    assert_eq!(cast(normalized), Ok(0x3f81));
    let product = arithmetic(Op::Mul, promoted(0x3f81).unwrap(), promoted(0x3f82).unwrap()).unwrap();
    assert_ne!(product, promoted(cast(product).unwrap()).unwrap());
    assert_eq!(rms_norm_bf16_bound(1, 0x3a84, 0x3f82, work), cast(product));
    // Envelope-specific discriminator: rounding before gamma crosses a BF16 tie.
    let no_cast = cast(arithmetic(Op::Mul, normalized, promoted(0x3fc0).unwrap()).unwrap());
    assert_eq!(no_cast, Ok(0x3fc1));
    assert_eq!(envelope_reference(1, 0x3a84, 0x3fc0, work), Ok(0x3fc2));
    assert_eq!(rms_norm_bf16_bound(1, 0x3a84, 0x3fc0, work), Ok(0x3fc2));
}

#[test]
fn public_bound_covers_signed_rows_and_cast_order() {
    for (input, gamma) in [
        (vec![0x3f80, 0x4000], vec![0x3f82, 0x3f80]),
        (
            vec![0xbf80, 0x4040, 0x3e80],
            vec![0x3fc0, 0xbf00, 0x4000],
        ),
        (
            vec![1, 0x8001, 0, 0x8000],
            vec![0x3f80, 0x3f80, 0xbf80, 0xbf80],
        ),
        (
            vec![0, 0xc000, 0, 0x8000, 0x3f00],
            vec![0xc040, 0x3f80, 0x8001, 0, 1],
        ),
    ] {
        let width = input.len();
        let h = input.iter().map(|word| word & 0x7fff).max().unwrap();
        let g = gamma.iter().map(|word| word & 0x7fff).max().unwrap();
        let work = rms_norm_bf16_bound_work(width).unwrap();
        let n = rms_norm_bf16_bound(width, h, g, work).unwrap();
        assert_eq!(envelope_reference(width, h, g, work), Ok(n));
        let expected = reference(&input, &gamma);
        let mut output = vec![0x5555; width];
        rms_norm_bf16_row_into(&input, &gamma, &mut output).unwrap();
        assert_eq!(output, expected.output);
        for word in output {
            assert!((word & 0x7fff) <= n);
        }
    }
    // Preserve the independently established cast-before-gamma discriminator.
    let trace = reference(&[0x3f80, 0x4000], &[0x3f82, 0x3f80]);
    assert_eq!(trace.casts[0], 0x3f22);
    assert_eq!(trace.output[0], 0x3f25);
    let no_cast = cast(arithmetic(Op::Mul, trace.normalized[0], 0x3f82_0000).unwrap()).unwrap();
    assert_eq!(no_cast, 0x3f24);
    assert_ne!(trace.output[0], no_cast);
}

#[test]
fn public_bound_overflow_and_q8_boundary() {
    for (width, h, g) in [
        (1, 0x7f7f, 0x3f80), // square overflow
        (1, 0x7f7f, 0),      // zero gamma cannot hide earlier overflow
        (4, 0x5f00, 0x3f80), // finite square, ordered sum overflow
        (4, 0x5f00, 0),
        (1, 0x3a84, 0x7f7f), // F32 product overflow
        (1, 0x3a84, 0x7f7e), // finite F32 product, BF16 cast overflow
    ] {
        let work = rms_norm_bf16_bound_work(width).unwrap();
        assert_eq!(envelope_reference(width, h, g, work), Err(Error::NonFinite));
        assert_eq!(rms_norm_bf16_bound(width, h, g, work), Err(Error::NonFinite));
    }
    assert_eq!(
        arithmetic(
            Op::Mul,
            promoted(0x3f81).unwrap(),
            promoted(0x7f7f).unwrap()
        ),
        Err(Error::NonFinite)
    );
    let product = arithmetic(Op::Mul, promoted(0x3f81).unwrap(), promoted(0x7f7e).unwrap()).unwrap();
    assert_eq!(product, 0x7f7f_fc00);
    assert_eq!(cast(product), Err(Error::NonFinite));
    // Conservative refusal does not prove every covered row fails.
    let mut output = [0x5555; 4];
    rms_norm_bf16_row_into(&[0; 4], &[0x3f80; 4], &mut output).unwrap();
    assert_eq!(output, [0; 4]);

    let work = rms_norm_bf16_bound_work(1).unwrap();
    for (g, expected) in [
        (0x5aff, Ok(0x7f80_0000_0000_0000_i64)),
        (0x5b00, Err(Error::IntegerRange)),
    ] {
        let n = rms_norm_bf16_bound(1, 0x3a83, g, work).unwrap();
        assert_eq!(envelope_reference(1, 0x3a83, g, work), Ok(n));
        assert_eq!(n, g); // First cast is exactly 1; N straddles positive 2^55.
        assert_eq!(
            crate::field_float::quantize_q8(promote_bf16(n).unwrap()),
            expected
        );
    }
    // Negative 2^55 fits i64::MIN; checking that sign would miss positive failure.
    assert_eq!(
        crate::field_float::quantize_q8(promote_bf16(0xdb00).unwrap()),
        Ok(i64::MIN)
    );
}

#[test]
fn public_bound_geometry_work_and_zero() {
    for width in [1, 3, 128, 2560] {
        let work = rms_norm_bf16_bound_work(width).unwrap();
        assert_eq!(work, rms_norm_bf16_row_work(width).unwrap());
        assert_eq!(work, 33_792 * width + 24_576);
        assert!(width + 5 <= 4 * width + 2);
        for (h, g) in [(0, 0), (0x8000, 0x8000), (0, 0x7f7f), (0x3f80, 0x8000)] {
            assert_eq!(rms_norm_bf16_bound(width, h, g, work), Ok(0));
        }
        assert_eq!(
            rms_norm_bf16_bound(width, 0x7fc1, 0xffff, work - 1),
            Err(Error::WorkLimit)
        );
        assert_eq!(
            rms_norm_bf16_bound(width, 0x7fc1, 0xffff, work),
            Err(Error::NonFinite)
        );
    }
    for (width, expected) in [
        (0, Error::Geometry),
        (16_777_217, Error::IntegerRange),
        (1_usize << (usize::BITS - 1), Error::WorkLimit),
    ] {
        assert_eq!(rms_norm_bf16_bound_work(width), Err(expected));
        // All fail before inspection or any enormous-width loop, even at MAX budget.
        assert_eq!(
            rms_norm_bf16_bound(width, 0x7fc1, 0xffff, usize::MAX),
            Err(expected)
        );
        assert_eq!(
            envelope_reference(width, 0x7fc1, 0xffff, usize::MAX),
            Err(expected)
        );
    }
    let work = rms_norm_bf16_bound_work(1).unwrap();
    for invalid in [
        0x7f80, 0xff80, 0x7f81, 0x7fc1, 0xff81, 0xffff, 0x8001, 0xbf80,
    ] {
        let expected = if invalid & 0x7f80 == 0x7f80 {
            Error::NonFinite
        } else {
            Error::IntegerRange
        };
        for (h, g) in [(invalid, 0x3f80), (0x3f80, invalid)] {
            assert_eq!(envelope_reference(1, h, g, work), Err(expected));
            assert_eq!(rms_norm_bf16_bound(1, h, g, work), Err(expected));
        }
    }
    // H classification precedes G; both classifications precede H arithmetic.
    assert_eq!(
        rms_norm_bf16_bound(1, 0xbf80, 0x7fc1, work),
        Err(Error::IntegerRange)
    );
    assert_eq!(
        rms_norm_bf16_bound(1, 0x7fc1, 0xbf80, work),
        Err(Error::NonFinite)
    );
    assert_eq!(
        rms_norm_bf16_bound(1, 0x7f7f, 0xbf80, work),
        Err(Error::IntegerRange)
    );
}

// New relational expectations use only the independent rational oracle above.
// Retain every prefix maximum so exhaustive small-H checks share one oracle
// scan; normal-domain tests scan a full prefix, never a Cartesian set of rows.
fn relational_prefix_reference(width: usize, h: u16) -> Result<(Vec<u16>, u32), Error> {
    let d = round_fraction(BigInt::from(width), BigUint::from(1_u8), false, false)?;
    let square = arithmetic(Op::Mul, promoted(h)?, promoted(h)?)?;
    let mut sum = 0;
    for _ in 0..width {
        sum = arithmetic(Op::Add, sum, square)?;
    }
    let mean = arithmetic(Op::Div, sum, d)?;
    arithmetic(Op::Add, mean, 0x3586_37bd)?;
    let mut prefixes = Vec::new();
    let mut maximum = 0;
    let mut uncast_maximum = 0;
    for word in 0..=h {
        let value = promoted(word)?;
        let square = arithmetic(Op::Mul, value, value)?;
        let mean = arithmetic(Op::Div, square, d)?;
        let adjusted = arithmetic(Op::Add, mean, 0x3586_37bd)?;
        let normalized = arithmetic(Op::Mul, value, reciprocal_root(adjusted))?;
        maximum = maximum.max(cast(normalized)? & 0x7fff);
        uncast_maximum = uncast_maximum.max(normalized);
        prefixes.push(maximum);
    }
    Ok((prefixes, uncast_maximum))
}

fn relational_final_reference(c: u16, g: u16) -> Result<u16, Error> {
    Ok(cast(arithmetic(Op::Mul, promoted(c)?, promoted(g)?)?)? & 0x7fff)
}

fn relational_reference(
    width: usize,
    h: u16,
    g: u16,
    work_limit: usize,
) -> Result<u16, Error> {
    if width == 0 {
        return Err(Error::Geometry);
    }
    let d = round_fraction(BigInt::from(width), BigUint::from(1_u8), false, false)?;
    if grid(d) != (BigUint::from(width) << 149_usize) {
        return Err(Error::IntegerRange);
    }
    // Independently expanded formula with fixed primitive allowances, not the
    // candidate quote or constants: 9216*d + 37888*32640 + 36864.
    let work = width
        .checked_mul(9216)
        .and_then(|n| n.checked_add(1_236_701_184))
        .ok_or(Error::WorkLimit)?;
    if work > work_limit {
        return Err(Error::WorkLimit);
    }
    for word in [h, g] {
        let bits = promoted(word)?;
        if signed_grid(bits)?.sign() == Sign::Minus {
            return Err(Error::IntegerRange);
        }
    }
    let (prefixes, _) = relational_prefix_reference(width, h & 0x7fff)?;
    relational_final_reference(*prefixes.last().unwrap(), g & 0x7fff)
}

#[test]
fn relational_bound_independent_oracle() {
    for width in [1_usize, 3, 17] {
        let work = crate::field_float::rms_norm_bf16_relational_bound_work(width).unwrap();
        let (prefixes, _) = relational_prefix_reference(width, 0x0080).unwrap();
        // Every small/subnormal H, every encoding in its prefix, gradual
        // underflow and both casts; expected bits never use candidate arithmetic.
        for h in 0..=0x0080_u16 {
            for g in [0, 0x8000, 1, 0x3a80, 0x3f82] {
                let expected = relational_final_reference(prefixes[usize::from(h)], g & 0x7fff);
                assert_eq!(
                    crate::field_float::rms_norm_bf16_relational_bound(width, h, g, work),
                    expected,
                    "width={width}, H={h:04x}, G={g:04x}"
                );
            }
        }
        assert_eq!(
            rms_norm_bf16_relational_bound(width, 0x8000, 0x7f7f, work),
            Ok(0)
        );
    }
    // A normal input whose square is the smallest F32 subnormal; division by
    // three rounds it to zero. Also scan every smaller BF16 encoding.
    let value = promoted(0x1a00).unwrap(); // 2^-75: square ties to even zero.
    assert_eq!(arithmetic(Op::Mul, value, value), Ok(0));
    let value = promoted(0x1a40).unwrap();
    let square = arithmetic(Op::Mul, value, value).unwrap();
    assert_eq!(square, 1);
    assert_eq!(arithmetic(Op::Div, square, 0x4040_0000), Ok(0));
    for (width, h, g) in [(3, 0x1a40, 0x3f82), (2, 0x4000, 0x3fc0)] {
        let work = rms_norm_bf16_relational_bound_work(width).unwrap();
        let expected = relational_reference(width, h, g, work);
        assert!(expected.is_ok());
        assert_eq!(rms_norm_bf16_relational_bound(width, h, g, work), expected);
    }
    // Bound-specific cast-order discriminator: independently scan one full
    // normal prefix, then a small gamma interval. Require an actual difference,
    // not an assumption that delaying the first cast changes every fixture.
    let (prefixes, uncast) = relational_prefix_reference(2, 0x4000).unwrap();
    let c = *prefixes.last().unwrap();
    assert_eq!(c, 0x3fb5);
    let mut discriminator = None;
    for g in 0x3f80..=0x4000 {
        let expected = relational_final_reference(c, g).unwrap();
        let delayed = cast(arithmetic(Op::Mul, uncast, promoted(g).unwrap()).unwrap()).unwrap();
        if expected != delayed {
            discriminator = Some((g, expected, delayed));
            break;
        }
    }
    let (g, expected, delayed) = discriminator.expect("first BF16 cast must matter");
    assert_ne!(expected, delayed);
    let work = rms_norm_bf16_relational_bound_work(2).unwrap();
    assert_eq!(
        rms_norm_bf16_relational_bound(2, 0x4000, g, work),
        Ok(expected)
    );
}

#[test]
fn relational_bound_covers_signed_rows() {
    for (input, gamma) in [
        (vec![0x3f80, 0x4000], vec![0x3f82, 0x3f80]),
        (
            vec![0xbf80, 0x4040, 0x3e80],
            vec![0x3fc0, 0xbf00, 0x4000],
        ),
        (
            vec![0, 0xc000, 0x8000, 0],
            vec![0xc040, 0xbf80, 1, 0x8000],
        ),
        (
            vec![1, 0x8001, 0, 0x8000],
            vec![0x3f80, 0xbf80, 0xbf80, 0x3f80],
        ),
        (vec![0x8000, 0, 0x8000], vec![0xbf80, 0, 0x3f80]),
    ] {
        let width = input.len();
        let h = input.iter().map(|word| word & 0x7fff).max().unwrap();
        let g = gamma.iter().map(|word| word & 0x7fff).max().unwrap();
        let work = rms_norm_bf16_relational_bound_work(width).unwrap();
        let expected_bound = relational_reference(width, h, g, work).unwrap();
        assert_eq!(
            rms_norm_bf16_relational_bound(width, h, g, work),
            Ok(expected_bound)
        );
        let expected = reference(&input, &gamma);
        let mut output = vec![0x5555; width];
        rms_norm_bf16_row_into(&input, &gamma, &mut output).unwrap();
        assert_eq!(output, expected.output);
        assert!(output.iter().all(|word| (word & 0x7fff) <= expected_bound));
    }
    let trace = reference(&[0x3f80, 0x4000], &[0x3f82, 0x3f80]);
    assert_eq!(trace.output[0], 0x3f25);
    assert_eq!(
        cast(arithmetic(Op::Mul, trace.normalized[0], 0x3f82_0000).unwrap()),
        Ok(0x3f24)
    );
}

#[test]
fn relational_bound_improves_epsilon_envelope() {
    let width = 2;
    let h = 0x4000;
    let g = 0x3f80;
    let work = rms_norm_bf16_relational_bound_work(width).unwrap();
    let old_work = rms_norm_bf16_bound_work(width).unwrap();
    let expected = relational_reference(width, h, g, work).unwrap();
    let old_expected = envelope_reference(width, h, g, old_work).unwrap();
    assert_eq!(expected, 0x3fb5); // Rounded sqrt(2), independently enumerated.
    assert_eq!(old_expected, 0x44fa); // Epsilon-only bound is 2000.
    assert!(expected < old_expected);
    assert_eq!(
        rms_norm_bf16_relational_bound(width, h, g, work),
        Ok(expected)
    );
    assert_eq!(rms_norm_bf16_bound(width, h, g, old_work), Ok(old_expected));
}

#[test]
fn relational_bound_preflight_overflow_and_q8() {
    for width in [1_usize, 3, 128, 2560] {
        let work = rms_norm_bf16_relational_bound_work(width).unwrap();
        assert_eq!(
            work,
            (width + 4 + 4 * 32640) * 8192 + 32640 * 4096 + 1024 * (width + 32640) + 4096
        );
        assert_eq!(work, 9216 * width + 1_236_701_184);
        assert_eq!(
            rms_norm_bf16_relational_bound(width, 0x7fc1, 0xffff, work - 1),
            Err(Error::WorkLimit)
        );
        assert_eq!(
            rms_norm_bf16_relational_bound(width, 0x7fc1, 0xffff, work),
            Err(Error::NonFinite)
        );
        assert_eq!(
            rms_norm_bf16_relational_bound(width, 0x8000, 0x8000, work),
            Ok(0)
        );
    }
    for (width, error) in [
        (0, Error::Geometry),
        (16_777_217, Error::IntegerRange),
        (usize::MAX, Error::IntegerRange),
        (1_usize << (usize::BITS - 1), Error::WorkLimit),
    ] {
        assert_eq!(rms_norm_bf16_relational_bound_work(width), Err(error));
        assert_eq!(
            rms_norm_bf16_relational_bound(width, 0xffff, 0xbf80, usize::MAX),
            Err(error)
        );
        assert_eq!(
            relational_reference(width, 0xffff, 0xbf80, usize::MAX),
            Err(error)
        );
    }
    let work = rms_norm_bf16_relational_bound_work(1).unwrap();
    for invalid in [0x7f80, 0xff80, 0x7f81, 0x7fc1, 0xffff, 0x8001, 0xbf80] {
        let error = if invalid & 0x7f80 == 0x7f80 {
            Error::NonFinite
        } else {
            Error::IntegerRange
        };
        for (h, g) in [(invalid, 0), (0, invalid)] {
            assert_eq!(relational_reference(1, h, g, work), Err(error));
            assert_eq!(rms_norm_bf16_relational_bound(1, h, g, work), Err(error));
        }
    }
    for (h, g, error) in [
        (0xbf80, 0x7fc1, Error::IntegerRange),
        (0x7fc1, 0xbf80, Error::NonFinite),
        (0x7f7f, 0xbf80, Error::IntegerRange),
    ] {
        assert_eq!(relational_reference(1, h, g, work), Err(error));
        assert_eq!(rms_norm_bf16_relational_bound(1, h, g, work), Err(error));
    }
    for (width, h, g) in [
        (1, 0x7f7f, 0),
        (4, 0x5f00, 0),
        (2, 0x4000, 0x7f7f),
        (2, 0x4000, 0x7f35),
    ] {
        let work = rms_norm_bf16_relational_bound_work(width).unwrap();
        assert_eq!(relational_reference(width, h, g, work), Err(Error::NonFinite));
        assert_eq!(
            rms_norm_bf16_relational_bound(width, h, g, work),
            Err(Error::NonFinite)
        );
    }
    // First-cast C=181/128. With G=(181/128)*2^127 the F32 product
    // is finite, but R16 overflows: distinct from RN32 product overflow.
    let product = arithmetic(
        Op::Mul,
        promoted(0x3fb5).unwrap(),
        promoted(0x7f35).unwrap(),
    )
    .unwrap();
    assert_eq!(product, 0x7f7f_f200);
    assert_eq!(cast(product), Err(Error::NonFinite));
    for (g, expected_q8) in [
        (0x5aff, Ok(0x7f80_0000_0000_0000_i64)),
        (0x5b00, Err(Error::IntegerRange)),
    ] {
        let expected = relational_reference(1, 0x3f80, g, work).unwrap();
        assert_eq!(expected, g); // C rounds to exactly one, not a Q8 guarantee.
        assert_eq!(
            rms_norm_bf16_relational_bound(1, 0x3f80, g, work),
            Ok(expected)
        );
        assert_eq!(
            crate::field_float::quantize_q8(promote_bf16(expected).unwrap()),
            expected_q8
        );
    }
}

// Unreduced exact rationals keep the proof independent of all numeric helpers.
type ProofRatio = (BigInt, BigInt);

fn proof_dyadic(numerator: i64, denominator_shift: usize) -> ProofRatio {
    (
        BigInt::from(numerator),
        BigInt::from(1_u8) << denominator_shift,
    )
}

fn proof_add(a: &ProofRatio, b: &ProofRatio) -> ProofRatio {
    (&a.0 * &b.1 + &b.0 * &a.1, &a.1 * &b.1)
}

fn proof_sub(a: &ProofRatio, b: &ProofRatio) -> ProofRatio {
    (&a.0 * &b.1 - &b.0 * &a.1, &a.1 * &b.1)
}

fn proof_mul(a: &ProofRatio, b: &ProofRatio) -> ProofRatio {
    (&a.0 * &b.0, &a.1 * &b.1)
}

fn proof_cmp(a: &ProofRatio, b: &ProofRatio) -> Ordering {
    (&a.0 * &b.1).cmp(&(&b.0 * &a.1))
}

fn q8_integer_reference(word: u16) -> Result<i64, Error> {
    let bits = promoted(word)?;
    // 256*value is the signed grid integer divided by 2^141. Round its
    // magnitude directly, ties to even; no candidate quantizer or F32 multiply.
    let numerator = grid(bits & 0x7fff_ffff);
    let mut quotient = &numerator >> 141_usize;
    let remainder = numerator - (&quotient << 141_usize);
    let half = BigUint::from(1_u8) << 140_usize;
    if remainder > half || (remainder == half && &quotient % 2_u8 != BigUint::from(0_u8)) {
        quotient += 1_u8;
    }
    let signed = BigInt::from_biguint(
        if word & 0x8000 == 0 {
            Sign::Plus
        } else {
            Sign::Minus
        },
        quotient,
    );
    i64::try_from(signed).map_err(|_| Error::IntegerRange)
}

fn gamma_ceiling_reference(word: u16) -> BigUint {
    // Exact BF16 dyadic on the independent universal grid, not candidate decode.
    let numerator = grid(u32::from(word & 0x7fff) << 16) * 392_u32;
    let denominator = BigUint::from(1_u8) << 149_usize;
    let quotient = &numerator / &denominator;
    let remainder = numerator % denominator;
    quotient + BigUint::from(u8::from(remainder != BigUint::from(0_u8)))
}

fn q8_energy_reference(width: usize, g: u16) -> BigUint {
    let k = gamma_ceiling_reference(g) + 1_u8;
    &k * &k * BigUint::from(width)
}

#[test]
fn q8_energy_bound_constant_proof() {
    let one = proof_dyadic(1, 0);
    let two = proof_dyadic(2, 0);
    let half = proof_dyadic(1, 1);
    let u = proof_dyadic(1, 24);
    let a = proof_sub(&one, &u);
    let b = proof_add(&one, &u);
    let eta = proof_dyadic(1, 150);
    let epsilon = (
        BigInt::from_biguint(Sign::Plus, grid(0x3586_37bd)),
        BigInt::from(1_u8) << 149_usize,
    );
    assert_eq!(EPSILON_BITS, 0x3586_37bd);
    assert_eq!(crate::integer_bounds::MAX_COLUMNS, 16384);
    // Underflow debit can be absorbed by epsilon; two rounded mean stages
    // retain eps/2, which itself is normal.
    let underflow_debit = proof_mul(
        &proof_dyadic(4, 0),
        &proof_mul(&proof_mul(&a, &a), &eta),
    );
    assert_ne!(proof_cmp(&underflow_debit, &epsilon), Ordering::Greater);
    assert_ne!(
        proof_cmp(
            &proof_mul(&proof_sub(&a, &half), &epsilon),
            &proof_mul(&proof_add(&a, &one), &eta),
        ),
        Ordering::Less
    );
    assert_eq!(
        proof_cmp(&proof_mul(&epsilon, &half), &proof_dyadic(1, 126)),
        Ordering::Greater
    );
    // Bernoulli induction: (1-n*u)*a - (1-(n+1)*u) = n*u^2 >=0.
    // Positive endpoint 1-(16384+3)*u then dominates b^2/2. No huge powers
    // or width Cartesian enumeration is needed for this exact rational proof.
    for n in [1, 17, 16386] {
        let linear = proof_sub(&one, &proof_mul(&proof_dyadic(n, 0), &u));
        let next = proof_sub(&one, &proof_mul(&proof_dyadic(n + 1, 0), &u));
        let difference = proof_sub(&proof_mul(&linear, &a), &next);
        let expected = proof_mul(&proof_dyadic(n, 0), &proof_mul(&u, &u));
        assert_eq!(
            proof_cmp(&difference, &expected),
            Ordering::Equal
        );
        assert_eq!(proof_cmp(&difference, &proof_dyadic(0, 0)), Ordering::Greater);
    }
    let endpoint = proof_sub(&one, &proof_mul(&proof_dyadic(16387, 0), &u));
    assert_eq!(
        proof_cmp(&proof_mul(&two, &endpoint), &proof_mul(&b, &b)),
        Ordering::Greater
    );
    let one_plus_v = proof_add(&one, &proof_dyadic(1, 8));
    let gain = proof_mul(&one_plus_v, &b);
    let debit = proof_add(&proof_mul(&one_plus_v, &eta), &proof_dyadic(1, 134));
    assert_eq!(proof_cmp(&gain, &two), Ordering::Less);
    assert_eq!(
        proof_cmp(&proof_mul(&gain, &gain), &proof_dyadic(65, 6)),
        Ordering::Less
    );
    assert_eq!(proof_cmp(&debit, &proof_dyadic(1, 133)), Ordering::Less);
    let sqrt_two_upper_squared = proof_mul(&proof_dyadic(3, 1), &proof_dyadic(3, 1));
    assert_eq!(proof_cmp(&sqrt_two_upper_squared, &two), Ordering::Greater);
    let q8_gain = proof_mul(
        &proof_dyadic(256, 0),
        &proof_mul(&proof_dyadic(65, 6), &proof_dyadic(3, 1)),
    );
    assert_eq!(
        proof_cmp(&q8_gain, &proof_dyadic(390, 0)),
        Ordering::Equal
    );
    assert_eq!(
        proof_cmp(
            &proof_mul(&proof_dyadic(256, 0), &proof_dyadic(1, 133)),
            &proof_dyadic(1, 125),
        ),
        Ordering::Equal
    );
    // Compare G coefficients and constant terms separately: valid for EVERY G>=0.
    assert_eq!(
        proof_cmp(
            &proof_add(&proof_dyadic(390, 0), &proof_dyadic(1, 124)),
            &proof_dyadic(392, 0),
        ),
        Ordering::Less
    );
    assert_eq!(
        proof_cmp(&proof_add(&half, &proof_dyadic(1, 125)), &one),
        Ordering::Less
    );
}

#[test]
fn q8_energy_bound_signed_rows() {
    for (input, gamma) in [
        (vec![0x3f80, 0x4000], vec![0x3f82, 0x3f80]),
        (
            vec![0xbf80, 0x4040, 0x3e80],
            vec![0x3fc0, 0xbf00, 0x4000],
        ),
        (
            vec![0, 0xc000, 0x8000, 0],
            vec![0xc040, 0xbf80, 1, 0x8000],
        ),
        (
            vec![1, 0x8001, 0, 0x8000],
            vec![0x3f80, 0xbf80, 0xbf80, 0x3f80],
        ),
        (vec![0x8000, 0, 0x8000], vec![0xbf80, 0, 0x3f80]),
    ] {
        let width = input.len();
        let h = input.iter().map(|word| word & 0x7fff).max().unwrap();
        let g = gamma.iter().map(|word| word & 0x7fff).max().unwrap();
        let work = crate::field_float::rms_norm_bf16_q8_energy_bound_work(width).unwrap();
        let bound = crate::field_float::rms_norm_bf16_q8_energy_bound(width, h, g, work).unwrap();
        assert_eq!(BigUint::from(bound), q8_energy_reference(width, g));
        let expected = reference(&input, &gamma);
        let mut output = vec![0x5555; width];
        rms_norm_bf16_row_into(&input, &gamma, &mut output).unwrap();
        assert_eq!(output, expected.output);
        let mut energy = BigUint::from(0_u8);
        for (&actual, &word) in output.iter().zip(&expected.output) {
            let q = q8_integer_reference(word).unwrap();
            assert_eq!(
                crate::field_float::quantize_q8(promote_bf16(actual).unwrap()),
                Ok(q)
            );
            let magnitude = BigUint::from(q.unsigned_abs());
            energy += &magnitude * &magnitude;
        }
        assert!(energy <= BigUint::from(bound));
    }
    // The independent trace still distinguishes first cast from delayed cast.
    let trace = reference(&[0x3f80, 0x4000], &[0x3f82, 0x3f80]);
    assert_eq!(q8_integer_reference(trace.output[0]), Ok(165));
    let delayed = cast(arithmetic(Op::Mul, trace.normalized[0], 0x3f82_0000).unwrap()).unwrap();
    assert_eq!(q8_integer_reference(delayed), Ok(164));
    for (word, expected) in [
        (0x3b00, Ok(0)),
        (0xbb00, Ok(0)),
        (0x3bc0, Ok(2)),
        (0xbbc0, Ok(-2)),
        (0x5aff, Ok(0x7f80_0000_0000_0000_i64)),
        (0x5b00, Err(Error::IntegerRange)),
        (0xdb00, Ok(i64::MIN)),
    ] {
        assert_eq!(q8_integer_reference(word), expected);
        assert_eq!(
            crate::field_float::quantize_q8(promote_bf16(word).unwrap()),
            expected
        );
    }
}

#[test]
fn q8_energy_bound_dyadic_and_overflow() {
    // Exponent boundaries include all r>=128 cases, subnormal/normal transition,
    // right/exact/left shifts, highest fitting left shift and first overflowing one.
    for g in [
        0, 0x8000, 1, 0x007f, 0x0080, 0x00ff, 0x0100, 0x0180, 0x0200,
        0x0280, 0x0300, 0x037f, 0x0380, 0x3b00, 0x3b80, 0x3f80, 0x427f,
        0x4280, 0x42ff, 0x4300, 0x437f, 0x4380, 0x7aff, 0x7b00, 0x7b7f,
        0x7b80, 0x7f00, 0x7f7f,
    ] {
        let expected = gamma_ceiling_reference(g);
        let actual = ceil_392_gamma(g & 0x7fff).map(BigUint::from);
        if expected > BigUint::from(u128::MAX) {
            assert_eq!(actual, Err(Error::IntegerRange));
        } else {
            assert_eq!(actual, Ok(expected));
        }
        // H=0 permits coordinate Q8 success even for huge G. Coarse energy
        // can nevertheless overflow; compare to unbounded independent arithmetic.
        let expected = q8_energy_reference(1, g);
        let work = rms_norm_bf16_q8_energy_bound_work(1).unwrap();
        let actual = rms_norm_bf16_q8_energy_bound(1, 0, g, work).map(BigUint::from);
        if expected > BigUint::from(u128::MAX) {
            assert_eq!(actual, Err(Error::IntegerRange));
        } else {
            assert_eq!(actual, Ok(expected));
        }
    }
    for (n, shift, expected) in [
        (0, 128, Ok(0)),
        (1, -133, Ok(1)),
        (u128::MAX, -128, Ok(1)),
        (8, -3, Ok(1)),
        (9, -3, Ok(2)),
        (1, 127, Ok(1_u128 << 127)),
        (1, 128, Err(Error::IntegerRange)),
        (1_u128 << 127, 1, Err(Error::IntegerRange)),
        (u128::MAX >> 7, 7, Ok((u128::MAX >> 7) << 7)),
        ((u128::MAX >> 7) + 1, 7, Err(Error::IntegerRange)),
    ] {
        assert_eq!(ceil_positive_dyadic(n, shift), expected);
    }
    // Private substeps expose failures masked by earlier public Q8 checks.
    assert_eq!(q8_coarse_energy(1, u128::MAX), Err(Error::IntegerRange)); // +1
    assert_eq!(q8_coarse_energy(1, 1_u128 << 64), Err(Error::IntegerRange)); // square
    let ceiling = (1_u128 << 64) - 2;
    assert_eq!(
        q8_coarse_energy(1, ceiling).map(BigUint::from),
        Ok(BigUint::from(u64::MAX) * BigUint::from(u64::MAX))
    );
    assert_eq!(q8_coarse_energy(2, ceiling), Err(Error::IntegerRange)); // width
}

#[test]
fn q8_energy_bound_preflight_and_cauchy() {
    for width in [
        0, crate::integer_bounds::MAX_COLUMNS + 1, 16_777_217, usize::MAX,
    ] {
        assert_eq!(rms_norm_bf16_q8_energy_bound_work(width), Err(Error::Geometry));
        assert_eq!(
            rms_norm_bf16_q8_energy_bound(width, 0xffff, 0xbf80, 0),
            Err(Error::Geometry)
        );
    }
    // No quote-overflow test: geometry makes it unreachable on this API's domain.
    for width in [1, 3, crate::integer_bounds::MAX_COLUMNS] {
        let work = rms_norm_bf16_q8_energy_bound_work(width).unwrap();
        let relational = rms_norm_bf16_relational_bound_work(width).unwrap();
        assert_eq!(work, relational + 4096);
        assert_eq!(
            rms_norm_bf16_q8_energy_bound(width, 0x7fc1, 0xffff, work - 1),
            Err(Error::WorkLimit)
        );
        assert_eq!(
            rms_norm_bf16_q8_energy_bound(width, 0x8000, 0x8000, work),
            Ok(width as u128)
        );
    }
    assert!(rms_norm_bf16_relational_bound_work(16385).is_ok()); // old domain intact
    for (width, h, g, error) in [
        (1, 0xbf80, 0x7fc1, Error::IntegerRange),
        (1, 0x7fc1, 0xbf80, Error::NonFinite),
        (1, 0x7f7f, 0xbf80, Error::IntegerRange),
        (1, 0, 0xff80, Error::NonFinite),
        (1, 0, 0x8001, Error::IntegerRange),
        (1, 0x7f7f, 0, Error::NonFinite),
        (4, 0x5f00, 0x8000, Error::NonFinite),
        (2, 0x4000, 0x7f35, Error::NonFinite),
        (1, 0x3f80, 0x5f80, Error::IntegerRange),
        (1, 0, 0x7f7f, Error::IntegerRange),
    ] {
        let work = rms_norm_bf16_q8_energy_bound_work(width).unwrap();
        assert_eq!(rms_norm_bf16_q8_energy_bound(width, h, g, work), Err(error));
    }
    // Q8 must run before a coarse energy overflow (both would be IntegerRange).
    let relational = rms_norm_bf16_relational_bound_work(1).unwrap();
    let n = rms_norm_bf16_relational_bound(1, 0x3f80, 0x5f80, relational).unwrap();
    assert_eq!(q8_integer_reference(n), Err(Error::IntegerRange));
    assert!(q8_energy_reference(1, 0x5f80) > BigUint::from(u128::MAX));
    let n = rms_norm_bf16_relational_bound(1, 0, 0x7f7f, relational).unwrap();
    assert_eq!(q8_integer_reference(n), Ok(0));
    assert!(q8_energy_reference(1, 0x7f7f) > BigUint::from(u128::MAX));

    let input = [0x3f80, 0x4000];
    let gamma = [0x3f82, 0x3f80];
    let trace = reference(&input, &gamma);
    let q: Vec<_> = trace
        .output
        .iter()
        .map(|&word| q8_integer_reference(word).unwrap())
        .collect();
    assert_eq!(q, [165, 324]);
    let work = rms_norm_bf16_q8_energy_bound_work(2).unwrap();
    let energy = rms_norm_bf16_q8_energy_bound(2, 0x4000, 0x3f82, work).unwrap();
    assert_eq!(BigUint::from(energy), q8_energy_reference(2, 0x3f82));
    // Only a complete real scan creates the matrix-side certificate.
    let weights = [3_i64, -4, -2, 5];
    let bytes = crate::integer_bounds::MatrixScan::row_energy_payload_bytes(2, 2).unwrap();
    let mut scan = crate::integer_bounds::MatrixScan::new_row_energy(2, 2, bytes).unwrap();
    scan.append(0, &weights[..3]).unwrap();
    scan.append(3, &weights[3..]).unwrap();
    let matrix = scan.finish_row_energy().unwrap();
    assert_eq!((matrix.rows(), matrix.columns()), (2, q.len())); // separate geometry check
    assert_eq!(matrix.maximum_row_squared_energy(), 29);
    let radius = crate::integer_bounds::dot_radius(
        energy,
        matrix.maximum_row_squared_energy(),
    )
    .unwrap();
    let product = BigUint::from(energy) * 29_u8;
    assert!(BigUint::from(radius) * BigUint::from(radius) >= product);
    assert!(BigUint::from(radius - 1) * BigUint::from(radius - 1) < product);
    for row in weights.chunks(2) {
        let dot: i128 = row
            .iter()
            .zip(&q)
            .map(|(&w, &x)| i128::from(w) * i128::from(x))
            .sum();
        assert!(dot.unsigned_abs() <= radius);
    }
    // Conditional bound does not check centered-field thresholds or geometry.
    let centered = u128::from(crate::coded_linear::CODED_LINEAR_FIELD - 1) / 2;
    assert!(radius < centered); // caller's separate strict no-wrap threshold
    assert!(radius >= (5_u128 - 1) / 2); // same radius would fail field-five check
    let bytes = crate::integer_bounds::MatrixScan::row_energy_payload_bytes(1, 3).unwrap();
    let mut mismatch = crate::integer_bounds::MatrixScan::new_row_energy(1, 3, bytes).unwrap();
    mismatch.append(0, &[1, 0, 0]).unwrap();
    let mismatch = mismatch.finish_row_energy().unwrap();
    assert_ne!(mismatch.columns(), q.len()); // must not compose this geometry
}
