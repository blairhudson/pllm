use super::{rsqrt_rn32, Error, COMPARISONS, RSQRT_WORK};
use std::cmp::Ordering;

// Independent unsigned fixed-limb oracle. Represent every positive finite f32
// as an integer on the universal 2^-149 grid, not the production (m,e) pair.
// A midpoint is (left_grid + right_grid)*2^-150. Thus a*midpoint^2 is
// compared to one by comparing the integer product with 2^449. No production
// decoder, comparator, search or floating candidate arithmetic is reused.
#[derive(Clone, Copy)]
struct Big([u32; 32]);

impl Big {
    fn grid(bits: u32) -> Self {
        assert!(bits > 0 && bits < 0x7f80_0000);
        let mut out = Self([0; 32]);
        let exponent = bits / 0x0080_0000;
        let coefficient = if exponent == 0 {
            bits
        } else {
            bits % 0x0080_0000 + 0x0080_0000
        };
        let offset = exponent.saturating_sub(1) as usize;
        // Build by setting individual bits, independent of production decoding.
        for bit in 0..24 {
            if coefficient & (1 << bit) != 0 {
                let index = offset + bit;
                out.0[index / 32] |= 1 << (index % 32);
            }
        }
        out
    }

    fn add(self, other: Self) -> Self {
        let mut out = Self([0; 32]);
        let mut carry = 0_u64;
        for (index, slot) in out.0.iter_mut().enumerate() {
            let sum = u64::from(self.0[index]) + u64::from(other.0[index]) + carry;
            *slot = sum as u32;
            carry = sum >> 32;
        }
        assert_eq!(carry, 0);
        out
    }

    fn mul(self, other: Self) -> Self {
        let mut out = Self([0; 32]);
        for i in 0..32 {
            let mut carry = 0_u64;
            for j in 0..32 - i {
                let index = i + j;
                let value = u64::from(self.0[i]) * u64::from(other.0[j])
                    + u64::from(out.0[index])
                    + carry;
                out.0[index] = value as u32;
                carry = value >> 32;
            }
            assert_eq!(carry, 0);
            assert!(other.0[32 - i..].iter().all(|word| *word == 0) || self.0[i] == 0);
        }
        out
    }

    fn cmp(self, other: Self) -> Ordering {
        for index in (0..32).rev() {
            match self.0[index].cmp(&other.0[index]) {
                Ordering::Equal => {}
                order => return order,
            }
        }
        Ordering::Equal
    }
}

fn midpoint_product(a: u32, left: u32, right: u32) -> Ordering {
    let midpoint = Big::grid(left).add(Big::grid(right));
    let product = Big::grid(a).mul(midpoint).mul(midpoint);
    let mut one = Big([0; 32]);
    one.0[449 / 32] = 1 << (449 % 32);
    product.cmp(one)
}

fn observe_rounding_direction(a: u32, result: u32, seen: &mut [bool; 2]) {
    // The midpoint of result with itself is exactly result. Reuse only the
    // independent limb oracle to compare a*result^2 with one: Less means
    // result is below the true reciprocal root (rounding down), Greater above
    // (rounding up). Exact results do not count toward either direction.
    match midpoint_product(a, result, result) {
        Ordering::Less => seen[0] = true,
        Ordering::Greater => seen[1] = true,
        Ordering::Equal => {}
    }
}

fn in_rounding_cell(a: u32, result: u32) -> bool {
    let lower = midpoint_product(a, result - 1, result);
    let upper = midpoint_product(a, result, result + 1);
    let even = result & 1 == 0;
    (lower == Ordering::Less || (lower == Ordering::Equal && even))
        && (upper == Ordering::Greater || (upper == Ordering::Equal && even))
}

fn certify(a: u32) -> u32 {
    COMPARISONS.with(|count| count.set(0));
    let result = rsqrt_rn32(f32::from_bits(a)).unwrap().to_bits();
    let comparisons = COMPARISONS.with(|count| count.get());
    assert!((1..=32).contains(&comparisons), "a={a:08x}");
    assert!(comparisons * 64 + 2048 <= RSQRT_WORK);
    assert!((0x0080_0000..0x7f80_0000).contains(&result));
    assert!(
        in_rounding_cell(a, result),
        "a={a:08x} result={result:08x}"
    );
    // Oracle must reject both immediately adjacent output encodings.
    assert!(!in_rounding_cell(a, result - 1));
    assert!(!in_rounding_cell(a, result + 1));
    result
}

#[test]
fn rsqrt_exact_powers_and_epsilon() {
    for (a, expected) in [
        (0x3f80_0000, 0x3f80_0000), // 1 -> 1
        (0x4080_0000, 0x3f00_0000), // 4 -> 0.5
        (0x3e80_0000, 0x4000_0000), // 0.25 -> 2
        (0x3586_37bd, 0x447a_0000), // specified exact epsilon -> 1000
    ] {
        assert_eq!(certify(a), expected);
    }
    // epsilon = 8796093*2^-43; epsilon*10^6 = 1 - 22208/2^43.
    // Exact reciprocal root is ABOVE 1000, by < 0.00000253 < half ULP = 2^-15.
    // Oracle above proves its rounding cell using integers, not approximate sqrt.
}

#[test]
fn rsqrt_extremes_and_subnormal_boundaries() {
    for a in [
        1,
        2,
        3,
        4,
        0x003f_ffff,
        0x0040_0000,
        0x0040_0001,
        0x007f_fffe,
        0x007f_ffff,
        0x0080_0000,
        0x0080_0001,
        0x0080_0002,
        0x7f7f_fffe,
        0x7f7f_ffff,
    ] {
        certify(a);
    }
}

#[test]
fn rsqrt_exponent_boundaries_and_adjacent_rounding_neighbors() {
    let mut seen = [false; 2]; // [rounding down, rounding up], excluding exact.
    for exponent in 1..255 {
        for fraction in [0, 1, 0x003f_ffff, 0x007f_fffe, 0x007f_ffff] {
            let a = (exponent << 23) | fraction;
            observe_rounding_direction(a, certify(a), &mut seen);
        }
    }
    for center in [
        0x3586_37bd_u32,
        0x3e80_0000,
        0x3f80_0000,
        0x4080_0000,
    ] {
        for a in center - 8..=center + 8 {
            observe_rounding_direction(a, certify(a), &mut seen);
        }
    }
    // Bounded sample coverage, not an exhaustive claim about all f32 inputs.
    assert!(seen[0], "oracle samples must exercise rounding down");
    assert!(seen[1], "oracle samples must exercise rounding up");
}

#[test]
fn rsqrt_rejects_nonpositive_and_nonfinite_before_search() {
    for bits in [0, 0x8000_0000, 0x8000_0001, 0xbf80_0000, 0xff7f_ffff] {
        COMPARISONS.with(|count| count.set(0));
        assert_eq!(rsqrt_rn32(f32::from_bits(bits)), Err(Error::NonPositiveScale));
        assert_eq!(COMPARISONS.with(|count| count.get()), 0);
    }
    for bits in [
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0x7fc0_0000,
        0xff80_0001,
        0xffc0_0000,
        0x7fff_ffff,
        0xffff_ffff,
    ] {
        COMPARISONS.with(|count| count.set(0));
        assert_eq!(rsqrt_rn32(f32::from_bits(bits)), Err(Error::NonFinite));
        assert_eq!(COMPARISONS.with(|count| count.get()), 0);
    }
    assert_eq!(RSQRT_WORK, 4096); // Caller owes this even on rejected classes.
}
