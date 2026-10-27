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
