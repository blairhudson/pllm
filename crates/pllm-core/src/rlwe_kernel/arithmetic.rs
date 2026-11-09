//! Secret-value arithmetic with fixed source schedules; not a machine CT claim.
//! Operands canonical under 3<=q<2^61. No secret division or remainder.

pub(super) fn select(a: u64, b: u64, bit: u64) -> u64 {
    let mask = 0_u64.wrapping_sub(bit);
    (a & !mask) | (b & mask)
}

pub(super) fn less(a: u64, b: u64) -> u64 {
    u64::from(a.overflowing_sub(b).1)
}

pub(super) fn add(a: u64, b: u64, q: u64) -> u64 {
    let sum = a.wrapping_add(b);
    let (reduced, borrow) = sum.overflowing_sub(q);
    select(reduced, sum, u64::from(borrow))
}

pub(super) fn sub(a: u64, b: u64, q: u64) -> u64 {
    let (difference, borrow) = a.overflowing_sub(b);
    difference.wrapping_add(q & 0_u64.wrapping_sub(u64::from(borrow)))
}

pub(super) fn mul(mut a: u64, mut b: u64, q: u64) -> u64 {
    let mut product = 0;
    for _ in 0..64 {
        let candidate = add(product, a, q);
        product = select(product, candidate, b & 1);
        a = add(a, a, q);
        b >>= 1;
    }
    product
}

/// Handles i64::MIN without signed abs/negation or secret division.
pub(super) fn magnitude(value: i64) -> u64 {
    let bits = value as u64;
    let mask = 0_u64.wrapping_sub(bits >> 63);
    (bits ^ mask).wrapping_sub(mask)
}

/// Caller has scanned magnitude <= explicit support < q.
pub(super) fn signed(value: i64, q: u64) -> u64 {
    let absolute = magnitude(value);
    select(absolute, sub(0, absolute, q), (value as u64) >> 63)
}

/// Full scan; no .all/.find, secret index, early invalid exit, or coefficient
/// diagnostics. The sole aggregate rejection occurs after ALL input scans.
pub(super) fn invalid_canonical(values: &[u64], modulus: u64) -> u64 {
    let mut invalid = 0;
    for &value in values {
        invalid |= less(value, modulus) ^ 1;
    }
    invalid
}

pub(super) fn invalid_signed(values: &[i64], support: u64) -> u64 {
    let mut invalid = 0;
    for &value in values {
        invalid |= less(support, magnitude(value));
    }
    invalid
}

pub(super) fn convert(input: &[i64], output: &mut [u64], q: u64) {
    for (i, &value) in input.iter().enumerate() {
        output[i] = signed(value, q);
    }
}

/// Schoolbook negacyclic convolution. The wrap branch and all addresses depend
/// only on public indices. Checked 2N/3N geometry precedes this private helper.
pub(super) fn convolution(lhs: &[u64], rhs: &[u64], output: &mut [u64], q: u64) {
    let n = lhs.len();
    output.fill(0);
    for i in 0..n {
        for j in 0..n {
            let product = mul(lhs[i], rhs[j], q);
            let index = i + j;
            if index >= n {
                output[index - n] = sub(output[index - n], product, q);
            } else {
                output[index] = add(output[index], product, q);
            }
        }
    }
}

/// Fixed 128-step restoring unsigned division. Divisor is PUBLIC, nonzero and
/// <=2^62, so remainder doubling fits u128. Numerator may be any u128.
pub(super) fn divide(numerator: u128, divisor: u128) -> u128 {
    let mut remainder = 0_u128;
    let mut quotient = 0_u128;
    for bit in (0..128).rev() {
        remainder = remainder.wrapping_shl(1) | ((numerator >> bit) & 1);
        let (difference, borrow) = remainder.overflowing_sub(divisor);
        let take = u128::from(!borrow);
        let mask = 0_u128.wrapping_sub(take);
        remainder = (difference & mask) | (remainder & !mask);
        quotient |= take << bit;
    }
    quotient
}

/// floor((2*t*phase+q)/(2*q)) mod t, exactly half up. Public bounds imply
/// numerator <2^123 and quotient <=t; one masked subtraction replaces mod t.
pub(super) fn decode(phase: u64, q: u64, t: u64) -> u64 {
    let numerator = u128::from(phase)
        .wrapping_mul(u128::from(t))
        .wrapping_mul(2)
        .wrapping_add(u128::from(q));
    let rounded = divide(numerator, u128::from(q).wrapping_mul(2)) as u64;
    let (reduced, borrow) = rounded.overflowing_sub(t);
    select(reduced, rounded, u64::from(borrow))
}
