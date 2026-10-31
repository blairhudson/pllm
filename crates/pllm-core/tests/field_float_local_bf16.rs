//! Independent integer/dyadic oracle; expected values never use production helpers.
use num_bigint::{BigInt, BigUint, Sign};
use pllm_core::field_float::{
    add_bf16_into, add_bf16_work, dot_bf16_rn32, dot_bf16_rn32_work,
    mul_bf16_into, mul_bf16_work, rope_bf16_supplied_into, rope_bf16_supplied_work,
    Error, FINITE_R32_WORK,
};

// Exact magnitude on the universal F32 2^-149 grid, including virtual 2^128.
fn magnitude(bits: u32) -> BigUint {
    if bits == 0x7f80_0000 {
        return BigUint::from(1_u8) << 277_usize;
    }
    let exponent = bits >> 23;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0 {
        BigUint::from(fraction)
    } else {
        BigUint::from(fraction + (1 << 23)) << (exponent - 1) as usize
    }
}

fn integer(bits: u32) -> Result<BigInt, Error> {
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return Err(Error::NonFinite);
    }
    Ok(BigInt::from_biguint(
        if bits >> 31 == 0 { Sign::Plus } else { Sign::Minus },
        magnitude(bits & 0x7fff_ffff),
    ))
}

// Independent monotone-search nearest neighbor: exact integers, no production
// rounding/packing or floating arithmetic. `scale` is 149 (sum) or 298 (product).
fn rounded(value: BigInt, scale: usize, bf16: bool, negative_zero: bool) -> Result<u32, Error> {
    let negative = value.sign() == Sign::Minus
        || (value.sign() == Sign::NoSign && negative_zero);
    let target = value.magnitude();
    let step = if bf16 { 16 } else { 0 };
    let end = 0x7f80_0000_u32 >> step;
    let at = |code: u32| magnitude(code << step) << (scale - 149);
    let (mut low, mut high) = (0_u32, end);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if at(middle) <= *target { low = middle; } else { high = middle - 1; }
    }
    if low == end {
        return Err(Error::NonFinite);
    }
    let lower = at(low);
    let upper = at(low + 1);
    let down = target - &lower;
    let up = &upper - target;
    if down > up || (down == up && low & 1 != 0) {
        low += 1;
    }
    if low == end {
        return Err(Error::NonFinite);
    }
    Ok((u32::from(negative) << 31) | (low << step))
}

fn arithmetic(a: u32, b: u32, multiply: bool) -> Result<u32, Error> {
    let x = integer(a)?;
    let y = integer(b)?;
    if multiply {
        rounded(x * y, 298, false, (a ^ b) >> 31 != 0)
    } else {
        rounded(x + y, 149, false, a >> 31 != 0 && b >> 31 != 0)
    }
}

fn cast(bits: u32) -> Result<u16, Error> {
    Ok((rounded(integer(bits)?, 149, true, bits >> 31 != 0)? >> 16) as u16)
}

fn scalar(a: u16, b: u16, multiply: bool) -> Result<u16, Error> {
    cast(arithmetic(u32::from(a) << 16, u32::from(b) << 16, multiply)?)
}

fn dot(a: &[u16], b: &[u16]) -> Result<u32, Error> {
    let mut accumulator = 0;
    for (&x, &y) in a.iter().zip(b) {
        let product = arithmetic(u32::from(x) << 16, u32::from(y) << 16, true)?;
        accumulator = arithmetic(accumulator, product, false)?;
    }
    Ok(accumulator)
}

#[test]
fn elementwise_integer_oracle_signed_zero_subnormal_ties_and_overflow() {
    let words = [0, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x3b80, 0x3f80,
        0x3f81, 0x3f83, 0xbf80, 0x4000, 0x7f7f, 0xff7f, 0x7f80, 0x7fc1];
    for a in words {
        for b in words {
            for multiply in [false, true] {
                let mut output = [0x1234, 0x4321];
                let expected = scalar(a, b, multiply);
                let result = if multiply {
                    mul_bf16_into(&[a], &[b], &mut output, mul_bf16_work(1).unwrap())
                } else {
                    add_bf16_into(&[a], &[b], &mut output, add_bf16_work(1).unwrap())
                };
                assert_eq!(result, expected.map(|_| ()), "a={a:04x} b={b:04x} mul={multiply}");
                assert_eq!(output[0], expected.unwrap_or(0));
                assert_eq!(output[1], 0x4321);
            }
        }
    }
}

#[test]
fn elementwise_late_failure_clears_entire_active_prefix_not_tail() {
    for multiply in [false, true] {
        for bad in [0x7fc0, 0x7f7f] {
            let left = [0x3f80, bad, 0x3f80];
            let right = [0x3f80, if multiply { 0x4000 } else { bad }, 0x3f80];
            let mut output = [0x5555; 5];
            let result = if multiply {
                mul_bf16_into(&left, &right, &mut output, usize::MAX)
            } else { add_bf16_into(&left, &right, &mut output, usize::MAX) };
            assert_eq!(result, Err(Error::NonFinite));
            assert_eq!(output, [0, 0, 0, 0x5555, 0x5555]);
        }
    }
}

#[test]
fn elementwise_preflight_and_one_short_are_atomic() {
    let mut output = [0x5555; 3];
    assert_eq!(add_bf16_into(&[0x7fc0], &[], &mut output, usize::MAX), Err(Error::Length));
    assert_eq!(mul_bf16_into(&[0x7fc0; 4], &[0; 4], &mut output, usize::MAX), Err(Error::Capacity));
    assert_eq!(add_bf16_into(&[0x7fc0], &[0], &mut output, add_bf16_work(1).unwrap() - 1), Err(Error::WorkLimit));
    assert_eq!(mul_bf16_into(&[0x7fc0], &[0], &mut output, mul_bf16_work(1).unwrap() - 1), Err(Error::WorkLimit));
    assert_eq!(output, [0x5555; 3]);
}

#[test]
fn dot_integer_oracle_cancellation_signed_zero_and_intermediate_overflow() {
    let cases: &[(&[u16], &[u16])] = &[
        (&[], &[]), (&[0x8000], &[0x3f80]),
        (&[0x3f80, 0xbf80], &[0x3f80, 0x3f80]),
        (&[0x4c80, 0x3f80, 0xcc80], &[0x3f80; 3]),
        (&[1, 0x8001, 0x007f], &[0x3780, 0x3700, 0x3f80]),
        (&[0x7f7f, 0x7f7f, 0xff7f], &[0x3f80; 3]),
        (&[0x7f7f, 0xff7f], &[0x4000; 2]),
        (&[0x3f80, 0x7fc0], &[0x3f80; 2]),
    ];
    for &(a, b) in cases {
        assert_eq!(dot_bf16_rn32(a, b, dot_bf16_rn32_work(a.len()).unwrap()).map(f32::to_bits), dot(a, b));
    }
    assert_eq!(dot_bf16_rn32(&[0x8000], &[0x3f80], usize::MAX).unwrap().to_bits(), 0);
}

#[test]
fn dot_nonfused_underflow_counterexample() {
    // First product = 2^-149, second = -2^-150. RN32(second) is -0;
    // separate add retains 2^-149. Fused second iteration rounds exact 2^-150 to +0.
    let a = [1, 1];
    let b = [0x3780, 0xb700];
    assert_eq!(dot(&a, &b), Ok(1));
    assert_eq!(dot_bf16_rn32(&a, &b, usize::MAX).unwrap().to_bits(), 1);
    let fused = integer(1).unwrap() * (BigInt::from(1_u8) << 149_usize)
        + integer(1 << 16).unwrap() * integer(0xb700_0000).unwrap();
    assert_eq!(rounded(fused, 298, false, false), Ok(0));
}

#[test]
fn dot_length_and_one_short_precede_nonfinite_visits() {
    assert_eq!(dot_bf16_rn32(&[0x7fc0], &[], usize::MAX), Err(Error::Length));
    assert_eq!(dot_bf16_rn32(&[0x7fc0], &[0], dot_bf16_rn32_work(1).unwrap() - 1), Err(Error::WorkLimit));
}

fn rotary(input: &[u16], cos: &[u16], sin: &[u16]) -> Result<Vec<u16>, Error> {
    let half = input.len() / 2;
    (0..input.len()).map(|k| {
        let rotated = if k < half { input[k + half] ^ 0x8000 } else { input[k - half] };
        scalar(scalar(input[k], cos[k], true)?, scalar(rotated, sin[k], true)?, false)
    }).collect()
}

#[test]
fn rope_integer_oracle_half_rotation_sign_order_and_product_rounding() {
    let cases = [
        ([0x3f80, 0x4000, 0x4040, 0x4080], [0; 4], [0x3f80; 4]),
        ([0, 0x8000, 1, 0x8001], [0x3f80; 4], [0x8000; 4]),
        ([0x3f81, 0x3f83, 0x3f81, 0x3f83], [0x3f81; 4], [0x3f81; 4]),
        ([0x007f, 0x0080, 0x8080, 0x8001], [0x3f00; 4], [0x3f00; 4]),
    ];
    for (input, cos, sin) in cases {
        let mut output = [0x5555; 6];
        rope_bf16_supplied_into(&input, &cos, &sin, &mut output, rope_bf16_supplied_work(4).unwrap()).unwrap();
        assert_eq!(&output[..4], rotary(&input, &cos, &sin).unwrap());
        assert_eq!(&output[4..], &[0x5555; 2]);
    }
    let mut output = [0; 4];
    rope_bf16_supplied_into(&[0x3f80, 0x4000, 0x4040, 0x4080], &[0; 4], &[0x3f80; 4], &mut output, usize::MAX).unwrap();
    assert_eq!(output, [0xc040, 0xc080, 0x3f80, 0x4000]);
    assert_eq!(scalar(0x3f81, 0x3f81, true), Ok(0x3f82));
}

#[test]
fn rope_independent_product_casts_differ_from_final_only_rounding() {
    // (129/128)^2 - 130/128 = 2^-14 exactly; rounding the first
    // product to BF16 BEFORE adding instead produces exact cancellation (+0).
    let input = [0x3f81, 0x3f80];
    let cos = [0x3f81, 0];
    let sin = [0x3f82, 0];
    let mut output = [0x5555; 3];
    rope_bf16_supplied_into(&input, &cos, &sin, &mut output, usize::MAX).unwrap();
    assert_eq!(&output[..2], rotary(&input, &cos, &sin).unwrap());
    assert_eq!(output, [0, 0, 0x5555]);
    let exact = integer(0x3f81_0000).unwrap() * integer(0x3f81_0000).unwrap()
        - integer(0x3f80_0000).unwrap() * integer(0x3f82_0000).unwrap();
    assert_eq!(rounded(exact, 298, true, false), Ok(0x3880_0000));
}

#[test]
fn rope_failure_clears_active_suffix_and_preserves_tail() {
    for (input, cos, sin) in [
        ([0x3f80, 0x7fc0, 0x3f80, 0x3f80], [0x3f80; 4], [0; 4]),
        ([0x7f7f; 4], [0x4000; 4], [0; 4]),
        ([0x7f00; 4], [0x3f80; 4], [0x3f80; 4]),
    ] {
        let mut output = [0x5555; 6];
        assert_eq!(rope_bf16_supplied_into(&input, &cos, &sin, &mut output, usize::MAX), Err(Error::NonFinite));
        assert_eq!(output, [0, 0, 0, 0, 0x5555, 0x5555]);
    }
}

#[test]
fn rope_geometry_lengths_capacity_and_one_short_are_atomic() {
    let mut output = [0x5555; 3];
    assert_eq!(rope_bf16_supplied_into(&[], &[], &[], &mut output, usize::MAX), Err(Error::Geometry));
    assert_eq!(rope_bf16_supplied_into(&[0x7fc0], &[0], &[0], &mut output, usize::MAX), Err(Error::Geometry));
    assert_eq!(rope_bf16_supplied_into(&[0x7fc0; 2], &[0], &[0; 2], &mut output, usize::MAX), Err(Error::Length));
    assert_eq!(rope_bf16_supplied_into(&[0x7fc0; 2], &[0; 2], &[0], &mut output, usize::MAX), Err(Error::Length));
    assert_eq!(rope_bf16_supplied_into(&[0x7fc0; 4], &[0; 4], &[0; 4], &mut output, usize::MAX), Err(Error::Capacity));
    assert_eq!(rope_bf16_supplied_into(&[0x7fc0; 2], &[0; 2], &[0; 2], &mut output, rope_bf16_supplied_work(2).unwrap() - 1), Err(Error::WorkLimit));
    assert_eq!(output, [0x5555; 3]);
}

#[test]
fn checked_quotes_cover_primitives_and_reject_overflow() {
    for width in [0, 1, 2, 128] {
        assert_eq!(add_bf16_work(width), Ok(width * (FINITE_R32_WORK + 2048) + 4096));
        assert_eq!(mul_bf16_work(width), add_bf16_work(width));
        assert_eq!(dot_bf16_rn32_work(width), Ok(width * (2 * FINITE_R32_WORK + 2048) + 4096));
    }
    assert_eq!(rope_bf16_supplied_work(128), Ok(128 * (3 * FINITE_R32_WORK + 2048) + 4096));
    assert_eq!(add_bf16_work(usize::MAX), Err(Error::WorkLimit));
    assert_eq!(mul_bf16_work(usize::MAX), Err(Error::WorkLimit));
    assert_eq!(dot_bf16_rn32_work(usize::MAX), Err(Error::WorkLimit));
    assert_eq!(rope_bf16_supplied_work(usize::MAX - 1), Err(Error::WorkLimit));
    assert_eq!(rope_bf16_supplied_work(0), Err(Error::Geometry));
    assert_eq!(rope_bf16_supplied_work(3), Err(Error::Geometry));
}

#[test]
fn empty_elementwise_active_region_leaves_spare_tail_untouched() {
    let mut output = [0x5555; 2];
    add_bf16_into(&[], &[], &mut output, add_bf16_work(0).unwrap()).unwrap();
    mul_bf16_into(&[], &[], &mut output, mul_bf16_work(0).unwrap()).unwrap();
    assert_eq!(output, [0x5555; 2]);
    assert_eq!(dot_bf16_rn32(&[], &[], dot_bf16_rn32_work(0).unwrap()).unwrap().to_bits(), 0);
}
