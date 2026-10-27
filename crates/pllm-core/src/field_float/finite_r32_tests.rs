use super::{add_rn32, div_rn32, mul_rn32, Error, FINITE_R32_WORK};
use num_bigint::{BigInt, BigUint, Sign};
use std::cmp::Ordering;

#[derive(Clone, Copy)]
enum Op {
    Add,
    Mul,
    Div,
}

// Independent oracle: universal 2^-149 integer grid and exact bigint rational
// arithmetic. Does not call production decode, magnitude, round, pack or compare;
// no native floating arithmetic/reference is used. The virtual successor of
// maxfinite is the exact real 2^128, used only for its overflow midpoint.
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

fn signed_grid(bits: u32) -> BigInt {
    BigInt::from_biguint(
        if bits >> 31 == 0 { Sign::Plus } else { Sign::Minus },
        grid(bits & 0x7fff_ffff),
    )
}

fn exact(op: Op, a: u32, b: u32) -> Option<(BigInt, BigInt)> {
    if a & 0x7f80_0000 == 0x7f80_0000 || b & 0x7f80_0000 == 0x7f80_0000 {
        return None;
    }
    let a = signed_grid(a);
    let b = signed_grid(b);
    let unit = BigInt::from(1_u8);
    let (mut numerator, mut denominator) = match op {
        Op::Add => (a + b, unit << 149_usize),
        Op::Mul => (a * b, unit << 298_usize),
        Op::Div => {
            if b == BigInt::from(0_u8) {
                return None;
            }
            (a, b)
        }
    };
    if denominator.sign() == Sign::Minus {
        numerator = -numerator;
        denominator = -denominator;
    }
    Some((numerator, denominator))
}

#[derive(Default)]
struct Directions {
    up: bool,
    down: bool,
}

impl Directions {
    fn both(&self) {
        assert!(self.up, "bounded oracle samples must include rounding upward");
        assert!(self.down, "bounded oracle samples must include rounding downward");
    }
}

fn certify(op: Op, a: u32, b: u32, directions: &mut Directions) -> Result<u32, Error> {
    let result = match op {
        Op::Add => add_rn32(f32::from_bits(a), f32::from_bits(b)),
        Op::Mul => mul_rn32(f32::from_bits(a), f32::from_bits(b)),
        Op::Div => div_rn32(f32::from_bits(a), f32::from_bits(b)),
    }
    .map(f32::to_bits);
    let Some((numerator, denominator)) = exact(op, a, b) else {
        assert_eq!(result, Err(Error::NonFinite));
        return result;
    };
    let n = numerator.magnitude();
    let d = denominator.magnitude();
    let scaled = n << 150_usize;
    let overflow = (grid(0x7f7f_ffff) + grid(0x7f80_0000)) * d;
    // Maxfinite has odd significand; equality rounds to the virtual successor,
    // which is overflow, not a finite clamp.
    if scaled >= overflow {
        assert_eq!(result, Err(Error::NonFinite));
        return result;
    }
    let bits = result.expect("oracle rational is inside finite rounding range");
    let magnitude = bits & 0x7fff_ffff;
    assert!(magnitude < 0x7f80_0000);
    if n == &BigUint::from(0_u8) {
        assert_eq!(magnitude, 0);
        return result; // Exact zero signs checked separately below.
    }
    let negative = numerator.sign() == Sign::Minus;
    assert_eq!(bits >> 31 != 0, negative); // Includes signed underflow zero.
    let even = magnitude & 1 == 0;
    if magnitude != 0 {
        let lower = (grid(magnitude - 1) + grid(magnitude)) * d;
        assert!(scaled > lower || (scaled == lower && even));
    }
    let upper = (grid(magnitude) + grid(magnitude + 1)) * d;
    assert!(scaled < upper || (scaled == upper && even));
    let mut order = (grid(magnitude) * d).cmp(&(n << 149_usize));
    if negative {
        order = order.reverse();
    }
    match order {
        Ordering::Greater => directions.up = true,
        Ordering::Less => directions.down = true,
        Ordering::Equal => {} // Exact results do not qualify as either direction.
    }
    result
}

#[test]
fn finite_r32_zero_signs_and_exact_cancellation() {
    let mut directions = Directions::default();
    for a in [0, 0x8000_0000] {
        for b in [0, 0x8000_0000] {
            assert_eq!(certify(Op::Add, a, b, &mut directions), Ok(a & b));
            assert_eq!(certify(Op::Mul, a, b, &mut directions), Ok(a ^ b));
            assert_eq!(certify(Op::Div, a, b, &mut directions), Err(Error::NonFinite));
        }
        for b in [0x3f80_0000, 0xbf80_0000] {
            let signed_zero = Ok((a ^ b) & 0x8000_0000);
            assert_eq!(certify(Op::Mul, a, b, &mut directions), signed_zero);
            assert_eq!(certify(Op::Mul, b, a, &mut directions), signed_zero);
            assert_eq!(certify(Op::Div, a, b, &mut directions), signed_zero);
        }
    }
    for a in [1, 0x0080_0000, 0x3f80_0000, 0x7f7f_ffff] {
        assert_eq!(certify(Op::Add, a, a | 0x8000_0000, &mut directions), Ok(0));
    }
}

#[test]
fn finite_r32_ties_and_underflow_signs() {
    let mut add = Directions::default();
    for (a, b, expected) in [
        (0x3f80_0000, 0x3380_0000, 0x3f80_0000),
        (0x3f80_0001, 0x3380_0000, 0x3f80_0002),
    ] {
        assert_eq!(certify(Op::Add, a, b, &mut add), Ok(expected));
    }
    add.both();
    let mut mul = Directions::default();
    for (a, b, expected) in [
        (0x3f80_0001, 0x3fc0_0000, 0x3fc0_0002),
        (0x3f80_0003, 0x3fc0_0000, 0x3fc0_0004),
        (1, 0x3f00_0000, 0),
        (3, 0x3f00_0000, 2),
        (0x8000_0001, 0x3f00_0000, 0x8000_0000),
        (1, 0xbf00_0000, 0x8000_0000),
    ] {
        assert_eq!(certify(Op::Mul, a, b, &mut mul), Ok(expected));
    }
    mul.both();
    let mut div = Directions::default();
    for (a, b, expected) in [
        (1, 0x4000_0000, 0),
        (3, 0x4000_0000, 2),
        (0x8000_0001, 0x4000_0000, 0x8000_0000),
        (1, 0xc000_0000, 0x8000_0000),
    ] {
        assert_eq!(certify(Op::Div, a, b, &mut div), Ok(expected));
    }
    div.both();
}

#[test]
fn finite_r32_boundaries_gaps_products_and_overflow() {
    let mut directions = Directions::default();
    let values = [
        0, 1, 2, 3, 0x007f_ffff, 0x0080_0000, 0x0080_0001,
        0x3f7f_ffff, 0x3f80_0000, 0x3f80_0001, 0x3fff_ffff,
        0x4000_0000, 0x7f7f_fffe, 0x7f7f_ffff,
    ];
    for a in values {
        for b in values {
            for sign in [0, 0x8000_0000] {
                for op in [Op::Add, Op::Mul, Op::Div] {
                    let _ = certify(op, a, b | sign, &mut directions);
                }
            }
        }
    }
    // Maxfinite + 2^103 is the exact overflow midpoint; immediately below it
    // still rounds to maxfinite. Also exercise subtraction and negative overflow.
    for (a, b, expected) in [
        (0x7f7f_ffff, 0x72ff_ffff, Ok(0x7f7f_ffff)),
        (0x7f7f_ffff, 0x7300_0000, Err(Error::NonFinite)),
        (0xff7f_ffff, 0xf300_0000, Err(Error::NonFinite)),
        (0x0080_0000, 0x807f_ffff, Ok(1)),
        (0x007f_ffff, 1, Ok(0x0080_0000)),
    ] {
        assert_eq!(certify(Op::Add, a, b, &mut directions), expected);
    }
    directions.both();
}

#[test]
fn finite_r32_division_ratios_and_near_half_remainders() {
    let mut directions = Directions::default();
    // 128 and 2560 are exact binary32 operands, not approximate epsilon.
    for (a, b) in [
        (0x4300_0000, 0x4520_0000),
        (0x4520_0000, 0x4300_0000),
        (0x3f80_0000, 0x4040_0000),
        (0x3f80_0000, 0x4120_0000),
    ] {
        let _ = certify(Op::Div, a, b, &mut directions);
    }
    // Subnormal-grid remainders immediately below, at and above half when
    // dividing by two; adjacent divisor encodings perturb those same ties.
    for a in [1, 3, 5, 0x007f_ffff, 0x0080_0001] {
        for b in [0x3fff_ffff, 0x4000_0000, 0x4000_0001] {
            let _ = certify(Op::Div, a, b, &mut directions);
        }
    }
    directions.both();
}

#[test]
fn finite_r32_invalid_classes_and_allowance() {
    let mut directions = Directions::default();
    for invalid in [
        0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xff80_0001,
        0x7fc0_0000, 0xffc0_0000, 0x7fff_ffff, 0xffff_ffff,
    ] {
        for op in [Op::Add, Op::Mul, Op::Div] {
            for valid in [0, 0x8000_0000, 1, 0x3f80_0000, 0xff7f_ffff] {
                assert_eq!(certify(op, invalid, valid, &mut directions), Err(Error::NonFinite));
                assert_eq!(certify(op, valid, invalid, &mut directions), Err(Error::NonFinite));
            }
        }
    }
    for a in [0, 0x8000_0000, 1, 0x3f80_0000, 0xff7f_ffff] {
        for zero in [0, 0x8000_0000] {
            assert_eq!(certify(Op::Div, a, zero, &mut directions), Err(Error::NonFinite));
        }
    }
    assert_eq!(FINITE_R32_WORK, 8192); // Required even on error; no auto-charge.
}
