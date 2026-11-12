//! Private borrowed little-endian integer mechanisms, not a numeric profile.
//!
//! Supported generic widths are N=5, 8, 16; width/index calculations are bounded
//! for those internal callers, not a promise for arbitrary unbounded N. No heap
//! or by-value wide return is used by these helpers. Exclusive output borrows
//! prevent aliasing inputs. Fundamental-array callers remain responsible for
//! admission and wiping; locals and hidden Rust copies are not promised wiped.
//! Fixed iteration counts do not establish constant time, security, RSS bounds,
//! physical source payment, or funding under FINITE_R32_WORK. New wide division
//! and future SiLU use are unpriced.

use std::cmp::Ordering;

pub(super) type U512 = [u64; 8];
pub(super) type U1024 = [u64; 16];

#[derive(Debug, Eq, PartialEq)]
pub(super) enum WideError {
    Shift,
    Overflow,
    ZeroDivisor,
}

/// Most significant limb first; equal arrays compare equal.
pub(super) fn cmp<const N: usize>(a: &[u64; N], b: &[u64; N]) -> Ordering {
    for (left, right) in a.iter().zip(b).rev() {
        match left.cmp(right) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    Ordering::Equal
}

/// Index of the largest set bit, or None for zero.
pub(super) fn high<const N: usize>(a: &[u64; N]) -> Option<i32> {
    for (i, limb) in a.iter().enumerate().rev() {
        if *limb != 0 {
            return Some((i * 64) as i32 + (63 - limb.leading_zeros()) as i32);
        }
    }
    None
}

/// Overwrite every output limb modulo 2^(64*N); return final carry.
pub(super) fn add_into<const N: usize>(
    a: &[u64; N],
    b: &[u64; N],
    out: &mut [u64; N],
) -> bool {
    let mut carry = false;
    for (i, slot) in out.iter_mut().enumerate() {
        let (sum, first) = a[i].overflowing_add(b[i]);
        let (sum, second) = sum.overflowing_add(u64::from(carry));
        *slot = sum;
        carry = first || second;
    }
    carry
}

/// Overwrite every output limb modulo 2^(64*N); return final borrow.
pub(super) fn sub_into<const N: usize>(
    a: &[u64; N],
    b: &[u64; N],
    out: &mut [u64; N],
) -> bool {
    let mut borrow = false;
    for (i, slot) in out.iter_mut().enumerate() {
        let (difference, first) = a[i].overflowing_sub(b[i]);
        let (difference, second) = difference.overflowing_sub(u64::from(borrow));
        *slot = difference;
        borrow = first || second;
    }
    borrow
}

/// Reject oversized shift FIRST, even for zero, then significant-bit overflow.
/// Both errors preserve all of out. Success overwrites and zero-fills it.
pub(super) fn shl_into<const N: usize>(
    a: &[u64; N],
    shift: u32,
    out: &mut [u64; N],
) -> Result<(), WideError> {
    let width = (64 * N) as u32;
    if shift >= width {
        return Err(WideError::Shift);
    }
    if let Some(top) = high(a) {
        if top as u32 + shift >= width {
            return Err(WideError::Overflow);
        }
    }
    if shift == 0 {
        out.copy_from_slice(a);
        return Ok(());
    }
    out.fill(0);
    let words = (shift / 64) as usize;
    let offset = shift % 64;
    if offset == 0 {
        out[words..].copy_from_slice(&a[..N - words]);
    } else {
        for i in 0..N - words {
            out[i + words] |= a[i] << offset;
            if i + words + 1 < N {
                out[i + words + 1] |= a[i] >> (64 - offset);
            }
        }
    }
    Ok(())
}

/// Full 512x512 -> 1024 product: 64 MACs and eight final carry stores.
pub(super) fn mul_into(a: &U512, b: &U512, out: &mut U1024) {
    out.fill(0);
    for i in 0..8 {
        let mut carry = 0_u128;
        for j in 0..8 {
            // With B=2^64: (B-1)^2+(B-1)+(B-1)=B^2-1.
            // Each partial sum and the whole MAC therefore fit u128.
            let product = u128::from(a[i]) * u128::from(b[j]);
            let mac = product + u128::from(out[i + j]) + carry;
            out[i + j] = mac as u64;
            carry = mac >> 64;
        }
        // Earlier rows reach at most i+7; this limb is still untouched.
        out[i + 8] = carry as u64;
    }
}

/// Restoring division with caller-owned quotient and two nine-limb buffers.
/// ZeroDivisor has priority over Overflow; either preserves ALL three buffers.
/// Overflow iff the upper eight numerator limbs are >=d. After preflight there
/// are exactly 1024 iterations, each shifting, subtracting and selecting all
/// nine limbs. On success n=q*d+rem, rem<d, rem[8]=0, and trial is cleared.
pub(super) fn div_rem_into(
    n: &U1024,
    d: &U512,
    q: &mut U512,
    rem: &mut [u64; 9],
    trial: &mut [u64; 9],
) -> Result<(), WideError> {
    let mut nonzero = 0_u64;
    for limb in d {
        nonzero |= *limb;
    }
    if nonzero == 0 {
        return Err(WideError::ZeroDivisor);
    }
    let mut upper_order = Ordering::Equal;
    for i in (0..8).rev() {
        match n[i + 8].cmp(&d[i]) {
            Ordering::Equal => {}
            order => {
                upper_order = order;
                break;
            }
        }
    }
    // n/d < 2^512 iff n < d*2^512 iff upper(n)<d.
    if upper_order != Ordering::Less {
        return Err(WideError::Overflow);
    }

    q.fill(0);
    rem.fill(0);
    trial.fill(0);
    for bit in (0..1024).rev() {
        let mut carry = (n[bit / 64] >> (bit % 64)) & 1;
        // Previous rem<d, so 2*rem+bit<2*d<2^513: nine limbs suffice.
        for limb in rem.iter_mut() {
            let next = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next;
        }
        let mut borrow = false;
        for i in 0..9 {
            let divisor_limb = if i < 8 { d[i] } else { 0 };
            let (difference, first) = rem[i].overflowing_sub(divisor_limb);
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            trial[i] = difference;
            borrow = first || second;
        }
        let take = u64::from(!borrow);
        let mask = 0_u64.wrapping_sub(take);
        for i in 0..9 {
            rem[i] = (trial[i] & mask) | (rem[i] & !mask);
        }
        // Preflight proves all quotient bits >=512 are zero. Guard BEFORE
        // indexing; no out-of-range quotient index or shift is evaluated.
        if bit < 512 {
            q[bit / 64] |= take << (bit % 64);
        }
    }
    trial.fill(0);
    Ok(())
}

/// In-place modular addition; every limb is visited, including after carry clears.
pub(super) fn add_assign(a: &mut U512, b: &U512) -> bool {
    let mut carry = false;
    for i in 0..8 {
        let (sum, first) = a[i].overflowing_add(b[i]);
        let (sum, second) = sum.overflowing_add(u64::from(carry));
        a[i] = sum;
        carry = first || second;
    }
    carry
}

/// In-place modular increment, with eight visits and final carry.
pub(super) fn increment(a: &mut U512) -> bool {
    let mut carry = true;
    for limb in a.iter_mut() {
        let (sum, next) = limb.overflowing_add(u64::from(carry));
        *limb = sum;
        carry = next;
    }
    carry
}

/// floor(n / (2^(64*drop_words)*d)), with ORIGINAL rational remainder sticky.
/// Shift >16 has priority over zero divisor, then quotient overflow. Errors do
/// not write out. Two fixed 16-step passes use scalar u128 word division only;
/// the separate 16-word scan retains discarded low-word information. No wide
/// preflight backing is needed. Successful second pass defines all eight words.
pub(super) fn shr_words_div_word_into(
    n: &U1024,
    drop_words: usize,
    d: u64,
    out: &mut U512,
) -> Result<bool, WideError> {
    if drop_words > 16 {
        return Err(WideError::Shift);
    }
    if d == 0 {
        return Err(WideError::ZeroDivisor);
    }
    let mut discarded = 0_u64;
    for i in 0..16 {
        if i < drop_words {
            discarded |= n[i];
        }
    }
    let divisor = u128::from(d);
    let mut remainder = 0_u128;
    let mut overflow = false;
    for i in (0..16).rev() {
        let word = if i + drop_words < 16 { n[i + drop_words] } else { 0 };
        let combined = (remainder << 64) | u128::from(word);
        let quotient = combined / divisor;
        remainder = combined % divisor;
        overflow |= i >= 8 && quotient != 0;
    }
    if overflow {
        return Err(WideError::Overflow);
    }
    remainder = 0;
    for i in (0..16).rev() {
        let word = if i + drop_words < 16 { n[i + drop_words] } else { 0 };
        // remainder<d<=u64::MAX, so this full u128 concatenation fits.
        let combined = (remainder << 64) | u128::from(word);
        let quotient = combined / divisor;
        remainder = combined % divisor;
        if i < 8 {
            out[i] = quotient as u64;
        }
    }
    Ok(remainder != 0 || discarded != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    // Test-only allocation. Word-byte conversion is independent of production
    // comparison, bit scan, carry arithmetic, shifts, multiplication/division.
    fn big<const N: usize>(words: &[u64; N]) -> BigUint {
        let mut bytes = Vec::with_capacity(N * 8);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }

    fn put<const N: usize>(value: &BigUint, out: &mut [u64; N]) {
        let bytes = value.to_bytes_le();
        assert!(bytes.len() <= N * 8);
        out.fill(0);
        for (i, chunk) in bytes.chunks(8).enumerate() {
            let mut word = [0_u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            out[i] = u64::from_le_bytes(word);
        }
    }

    // Cases 0..4: zero, max, two alternating patterns. All remaining cases
    // enumerate EVERY single-bit position of the selected supported width.
    fn fixture<const N: usize>(case: usize, out: &mut [u64; N]) {
        out.fill(0);
        match case {
            0 => {}
            1 => out.fill(u64::MAX),
            2 => out.fill(0xaaaa_aaaa_aaaa_aaaa),
            3 => out.fill(0x5555_5555_5555_5555),
            _ => {
                let bit = case - 4;
                assert!(bit < N * 64);
                out[bit / 64] = 1_u64 << (bit % 64);
            }
        }
    }

    fn check_cmp_high<const N: usize>() {
        let mut a = [0_u64; N];
        let mut b = [0_u64; N];
        let cases = 4 + N * 64;
        for case in 0..cases {
            fixture(case, &mut a);
            fixture((case * 73 + 3) % cases, &mut b);
            let av = big(&a);
            let bv = big(&b);
            assert_eq!(cmp(&a, &b), av.cmp(&bv));
            assert_eq!(cmp(&b, &a), bv.cmp(&av));
            assert_eq!(cmp(&a, &a), Ordering::Equal);
            let expected = if av.bits() == 0 { None } else { Some((av.bits() - 1) as i32) };
            assert_eq!(high(&a), expected);
        }
    }

    #[test]
    fn cmp_high_matches_biguint_all_supported_widths() {
        check_cmp_high::<5>();
        check_cmp_high::<8>();
        check_cmp_high::<16>();
    }

    fn check_add_sub_pair<const N: usize>(a: &[u64; N], b: &[u64; N]) {
        let modulus = BigUint::from(1_u8) << (N * 64);
        let av = big(a);
        let bv = big(b);
        let sum = &av + &bv;
        let difference = (&av + &modulus) - &bv;
        let mut out = [0xdead_beef_cafe_1234_u64; N];
        let carry = add_into(a, b, &mut out);
        assert_eq!(big(&out), &sum % &modulus);
        assert_eq!(carry, sum >= modulus);
        out.fill(0x0123_4567_89ab_cdef);
        let borrow = sub_into(a, b, &mut out);
        assert_eq!(big(&out), difference % &modulus);
        assert_eq!(borrow, av < bv);
    }

    fn check_add_sub<const N: usize>() {
        let mut a = [0_u64; N];
        let mut b = [0_u64; N];
        for case in 0..4 + N * 64 {
            fixture(case, &mut a);
            for other in 0..4 {
                fixture(other, &mut b);
                check_add_sub_pair(&a, &b);
                check_add_sub_pair(&b, &a);
            }
        }
        b.fill(0);
        b[0] = 1;
        for bit in 0..N * 64 {
            let power = BigUint::from(1_u8) << bit;
            put(&power, &mut a);
            check_add_sub_pair(&a, &b); // Borrow across all lower limbs.
            put(&(power - 1_u8), &mut a);
            check_add_sub_pair(&a, &b); // Carry across all lower limbs.
        }
    }

    #[test]
    fn add_sub_modular_carry_borrow_matches_biguint() {
        check_add_sub::<5>();
        check_add_sub::<8>();
        check_add_sub::<16>();
    }

    fn check_shift_success<const N: usize>() {
        let width = (N * 64) as u32;
        let mut a = [0_u64; N];
        for case in 0..4 + N * 64 {
            fixture(case, &mut a);
            for shift in [0, 1, 31, 63, 64, 65, 127, 128, 129, width - 65, width - 64, width - 1] {
                let expected = big(&a) << shift as usize;
                if expected.bits() > u64::from(width) { continue; }
                let mut out = [u64::MAX; N];
                assert_eq!(shl_into(&a, shift, &mut out), Ok(()));
                assert_eq!(big(&out), expected, "N={N}, case={case}, shift={shift}");
            }
        }
    }

    #[test]
    fn shift_success_boundaries_and_full_overwrite() {
        check_shift_success::<5>();
        check_shift_success::<8>();
        check_shift_success::<16>();
    }

    fn check_shift_errors<const N: usize>() {
        let width = (N * 64) as u32;
        let mut a = [0_u64; N];
        let sentinel = [0x1937_5bad_c0de_0246_u64; N];
        for case in 0..4 + N * 64 {
            fixture(case, &mut a);
            for shift in [width, width + 1, u32::MAX] {
                let mut out = sentinel;
                assert_eq!(shl_into(&a, shift, &mut out), Err(WideError::Shift));
                assert_eq!(out, sentinel);
            }
            for shift in [1, 63, 64, 65, width - 1] {
                let exact = big(&a) << shift as usize;
                if exact.bits() <= u64::from(width) { continue; }
                let mut out = sentinel;
                assert_eq!(shl_into(&a, shift, &mut out), Err(WideError::Overflow));
                assert_eq!(out, sentinel);
            }
        }
        for bit in 1..N * 64 {
            fixture(4 + bit, &mut a);
            let shift = width - bit as u32;
            assert!((big(&a) << shift as usize).bits() > u64::from(width));
            let mut out = sentinel;
            assert_eq!(shl_into(&a, shift, &mut out), Err(WideError::Overflow));
            assert_eq!(out, sentinel);
        }
    }

    #[test]
    fn shift_error_priority_preserves_output() {
        check_shift_errors::<5>();
        check_shift_errors::<8>();
        check_shift_errors::<16>();
    }

    fn check_product(a: &U512, b: &U512) {
        let expected = big(a) * big(b);
        let mut out = [0xfeed_face_8765_4321_u64; 16];
        mul_into(a, b, &mut out);
        assert_eq!(big(&out), expected);
    }

    #[test]
    fn mul_full_products_match_biguint() {
        let mut a = [0_u64; 8];
        let mut b = [0_u64; 8];
        for case in 0..516 {
            fixture(case, &mut a);
            for other in 0..4 {
                fixture(other, &mut b);
                check_product(&a, &b);
            }
            fixture((case * 73 + 3) % 516, &mut b);
            check_product(&a, &b);
        }
        for bit in 0..512 {
            fixture(4 + bit, &mut a);
            fixture(4 + 511 - bit, &mut b);
            check_product(&a, &b);
        }
    }

    fn check_division(numerator: &BigUint, divisor: &BigUint) {
        assert_ne!(*divisor, BigUint::from(0_u8));
        let expected_q = numerator / divisor;
        let expected_rem = numerator % divisor;
        assert!(expected_q.bits() <= 512);
        let mut n = [0_u64; 16];
        let mut d = [0_u64; 8];
        put(numerator, &mut n);
        put(divisor, &mut d);
        let mut q = [0x1111_2222_3333_4444_u64; 8];
        let mut rem = [0x5555_6666_7777_8888_u64; 9];
        let mut trial = [0x9999_aaaa_bbbb_cccc_u64; 9];
        assert_eq!(div_rem_into(&n, &d, &mut q, &mut rem, &mut trial), Ok(()));
        assert_eq!(big(&q), expected_q);
        assert_eq!(big(&rem), expected_rem);
        assert!(big(&rem) < *divisor);
        assert_eq!(rem[8], 0);
        assert_eq!(trial, [0; 9]);
    }

    fn check_div_error(n: &U1024, d: &U512, expected: WideError) {
        let mut q = [0x1111_2222_3333_4444_u64; 8];
        let mut rem = [0x5555_6666_7777_8888_u64; 9];
        let mut trial = [0x9999_aaaa_bbbb_cccc_u64; 9];
        assert_eq!(div_rem_into(n, d, &mut q, &mut rem, &mut trial), Err(expected));
        assert_eq!(q, [0x1111_2222_3333_4444; 8]);
        assert_eq!(rem, [0x5555_6666_7777_8888; 9]);
        assert_eq!(trial, [0x9999_aaaa_bbbb_cccc; 9]);
    }

    #[test]
    fn div_exact_and_remainder_match_biguint() {
        let mut d = [0_u64; 8];
        let mut q = [0_u64; 8];
        for case in 1..516 {
            fixture(case, &mut d);
            fixture((case * 73 + 3) % 516, &mut q);
            let dv = big(&d);
            let qv = big(&q);
            for remainder in [BigUint::from(0_u8), &dv - 1_u8] {
                // Construct n with the independent bigint product, not mul_into.
                let n = &qv * &dv + &remainder;
                check_division(&n, &dv);
            }
            check_division(&BigUint::from(0_u8), &dv);
        }
        let maximum_divisor = (BigUint::from(1_u8) << 512_usize) - 1_u8;
        for bits in [1_usize, 2, 63, 64, 65, 511, 512, 513, 768, 1023, 1024] {
            let n = BigUint::from(1_u8) << (bits - 1);
            check_division(&n, &maximum_divisor);
        }
    }

    #[test]
    fn div_max_quotient_and_overflow_boundary() {
        let radix = BigUint::from(1_u8) << 512_usize;
        let maximum = &radix - 1_u8;
        let mut d = [0_u64; 8];
        let mut n = [0_u64; 16];
        for case in 1..516 {
            fixture(case, &mut d);
            let dv = big(&d);
            for remainder in [BigUint::from(0_u8), &dv - 1_u8] {
                let exact_n = &maximum * &dv + &remainder;
                check_division(&exact_n, &dv);
            }
            // Immediately below the threshold: q=max, rem=d-1.
            let threshold = &dv * &radix;
            check_division(&(&threshold - 1_u8), &dv);
            put(&threshold, &mut n); // Upper eight limbs == d.
            check_div_error(&n, &d, WideError::Overflow);
            n[..8].fill(u64::MAX);
            check_div_error(&n, &d, WideError::Overflow);
            if dv < maximum {
                let above = (&dv + 1_u8) * &radix;
                put(&above, &mut n); // Upper eight limbs > d.
                check_div_error(&n, &d, WideError::Overflow);
            }
        }
    }

    #[test]
    fn div_zero_priority_preserves_all_buffers() {
        let d = [0_u64; 8];
        let mut n = [0_u64; 16];
        for case in 0..1028 {
            fixture(case, &mut n);
            // For zero d, even zero upper(n) also meets upper(n)>=d:
            // ZeroDivisor must win over the quotient-overflow condition.
            check_div_error(&n, &d, WideError::ZeroDivisor);
        }
    }

    #[test]
    fn add_assign_and_increment_match_biguint() {
        let modulus = BigUint::from(1_u8) << 512_usize;
        let mut a = [0_u64; 8];
        let mut b = [0_u64; 8];
        for case in 0..516 {
            fixture(case, &mut a);
            for other in 0..4 {
                fixture(other, &mut b);
                let expected = big(&a) + big(&b);
                let mut actual = a;
                assert_eq!(add_assign(&mut actual, &b), expected >= modulus);
                assert_eq!(big(&actual), expected % &modulus);
            }
            let expected = big(&a) + 1_u8;
            assert_eq!(increment(&mut a), expected >= modulus);
            assert_eq!(big(&a), expected % &modulus);
        }
        b.fill(0);
        b[0] = 1;
        for bit in 0..=512_usize {
            let chain = (BigUint::from(1_u8) << bit) - 1_u8;
            put(&chain, &mut a);
            let mut second = a;
            let expected = &chain + 1_u8;
            assert_eq!(add_assign(&mut a, &b), expected >= modulus);
            assert_eq!(increment(&mut second), expected >= modulus);
            assert_eq!(big(&a), &expected % &modulus);
            assert_eq!(big(&second), expected % &modulus);
        }
    }

    #[test]
    fn shr_words_div_word_matches_biguint() {
        let mut n = [0_u64; 16];
        for case in 0..1028 {
            fixture(case, &mut n);
            let value = big(&n);
            for drop_words in [0_usize, 1, 4, 8, 15, 16] {
                for d in [1_u64, 2, 3, 64, u64::MAX] {
                    let denominator = BigUint::from(d) << (64 * drop_words);
                    let expected = &value / &denominator;
                    let sticky = (&value % &denominator) != BigUint::from(0_u8);
                    let mut out = [0xdead_beef_0123_4567_u64; 8];
                    if expected.bits() > 512 {
                        assert_eq!(shr_words_div_word_into(&n, drop_words, d, &mut out),
                            Err(WideError::Overflow));
                        assert_eq!(out, [0xdead_beef_0123_4567; 8]);
                    } else {
                        assert_eq!(shr_words_div_word_into(&n, drop_words, d, &mut out), Ok(sticky));
                        assert_eq!(big(&out), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn shr_words_div_word_sticky_and_full_overwrite() {
        let mut n = [0_u64; 16];
        for (low, retained, d, sticky, expected) in [
            (0_u64, 6_u64, 3_u64, false, 2_u64),
            (1, 6, 3, true, 2), // Discarded-only remainder.
            (0, 7, 3, true, 2), // Word-division-only remainder.
            (1, 7, 3, true, 2), // Both components.
            (0, u64::MAX, u64::MAX, false, 1),
        ] {
            n.fill(0);
            n[0] = low;
            n[4] = retained;
            let mut out = [u64::MAX; 8];
            assert_eq!(shr_words_div_word_into(&n, 4, d, &mut out), Ok(sticky));
            assert_eq!(big(&out), BigUint::from(expected));
        }
        for nonzero in [false, true] {
            n.fill(if nonzero { u64::MAX } else { 0 });
            let mut out = [u64::MAX; 8];
            assert_eq!(shr_words_div_word_into(&n, 16, 1, &mut out), Ok(nonzero));
            assert_eq!(out, [0; 8]);
        }
    }

    #[test]
    fn shr_words_div_word_error_priority_preserves_output() {
        let mut n = [0_u64; 16];
        let sentinel = [0x1234_5678_9abc_def0_u64; 8];
        for fill in [0, u64::MAX] {
            n.fill(fill);
            for drop_words in [17, usize::MAX] {
                for d in [0, 1, u64::MAX] {
                    let mut out = sentinel;
                    assert_eq!(shr_words_div_word_into(&n, drop_words, d, &mut out), Err(WideError::Shift));
                    assert_eq!(out, sentinel);
                }
            }
            for drop_words in [0, 1, 4, 8, 15, 16] {
                let mut out = sentinel;
                assert_eq!(shr_words_div_word_into(&n, drop_words, 0, &mut out), Err(WideError::ZeroDivisor));
                assert_eq!(out, sentinel);
            }
        }
        for drop_words in [0, 1, 4] {
            n.fill(0);
            n[drop_words + 8] = 1;
            let mut out = sentinel;
            assert_eq!(shr_words_div_word_into(&n, drop_words, 1, &mut out), Err(WideError::Overflow));
            assert_eq!(out, sentinel);
        }
    }
}
