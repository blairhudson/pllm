//! Source-only finite-rounding experiment, not a production SiLU implementation.
//! Contract: RN16(RN32(exact real SiLU(x))), with gradual underflow and signed zero.
//! Fixed P256/N64 and oracle degree 512 are hypotheses: ambiguity is a failure.
//! Width assertions concern individual integer values, not live allocation/RSS
//! or any production fixed-limb budget. No production numeric helper is used.

use num_bigint::BigUint;

const TINY: u16 = 0x3300; // 2^-25
const TAIL: u16 = 0x4300; // 128
const LAST_FINITE: u16 = 0x7f7f;

struct Width {
    limit: u64,
    max: u64,
}

impl Width {
    fn new(limit: u64) -> Self {
        Self { limit, max: 0 }
    }

    // Every arithmetic BigUint intermediate in the candidate/oracle paths is
    // named through this check, including products before divisions/shifts.
    fn see(&mut self, value: BigUint) -> BigUint {
        self.max = self.max.max(value.bits());
        assert!(
            value.bits() < self.limit,
            "individual width {} >= {}",
            value.bits(), self.limit,
        );
        value
    }
}

fn small(value: &BigUint) -> u32 {
    let digits = value.to_u32_digits();
    assert!(digits.len() <= 1);
    digits.first().copied().unwrap_or(0)
}

// Candidate input decode: coefficient times a power of two, converted to a
// rational. Only finite magnitudes enter; no host floating-point arithmetic.
fn candidate_input(mag: u16, w: &mut Width) -> (BigUint, BigUint) {
    assert!(mag <= LAST_FINITE);
    let exponent = (mag >> 7) & 255;
    let coefficient = if exponent == 0 { mag & 127 } else { 128 + (mag & 127) };
    let power = if exponent == 0 { -133 } else { i32::from(exponent) - 134 };
    let coefficient = w.see(BigUint::from(coefficient));
    if power >= 0 {
        let n = w.see(coefficient << power as usize);
        (n, w.see(BigUint::from(1_u8)))
    } else {
        let d = w.see(BigUint::from(1_u8) << (-power) as usize);
        (coefficient, d)
    }
}

fn candidate_ceil(n: &BigUint, d: &BigUint, w: &mut Width) -> BigUint {
    let q = w.see(n / d);
    let r = w.see(n % d);
    if r == BigUint::from(0_u8) { q } else { w.see(q + 1_u8) }
}

// Quotient/remainder ties-even, independent of the oracle's ordered-grid search.
fn candidate_rn32(n: &BigUint, d: &BigUint, negative: bool, w: &mut Width) -> u32 {
    let sign = u32::from(negative) << 31;
    assert_ne!(*d, BigUint::from(0_u8));
    if *n == BigUint::from(0_u8) {
        return sign;
    }
    let mut e = n.bits() as i32 - d.bits() as i32;
    let below = if e >= 0 {
        let aligned = w.see(d << e as usize);
        n < &aligned
    } else {
        let aligned = w.see(n << (-e) as usize);
        &aligned < d
    };
    if below { e -= 1; }
    let step = if e < -126 { -149 } else { e - 23 };
    let (numerator, denominator) = if step < 0 {
        (w.see(n << (-step) as usize), w.see(d.clone()))
    } else {
        (w.see(n.clone()), w.see(d << step as usize))
    };
    let quotient = w.see(&numerator / &denominator);
    let remainder = w.see(&numerator % &denominator);
    let twice = w.see(&remainder << 1_usize);
    let mut mantissa = small(&quotient);
    if twice > denominator || (twice == denominator && mantissa & 1 != 0) {
        mantissa += 1;
    }
    if e < -126 {
        assert!(mantissa <= 0x0080_0000);
        return sign | mantissa;
    }
    if mantissa == 0x0100_0000 {
        mantissa >>= 1;
        e += 1;
    }
    assert!((-126..=127).contains(&e));
    assert!((0x0080_0000..0x0100_0000).contains(&mantissa));
    sign | (((e + 127) as u32) << 23) | (mantissa - 0x0080_0000)
}

fn candidate_bf16(bits: u32) -> u16 {
    assert!(bits & 0x7f80_0000 != 0x7f80_0000);
    let mut upper = ((bits & 0x7fff_ffff) >> 16) as u16;
    let remainder = bits & 0xffff;
    if remainder > 0x8000 || (remainder == 0x8000 && upper & 1 != 0) {
        upper += 1;
    }
    assert!(upper < 0x7f80);
    ((bits >> 16) as u16 & 0x8000) | upper
}

// Exp(a/128) at scale Q, then seven outward squares. Exp bounds are computed
// once per magnitude, reused for the two different signed SiLU transforms.
fn candidate_exp(mag: u16, w: &mut Width) -> (BigUint, BigUint, BigUint) {
    assert!(mag > TINY && mag < TAIL);
    let q = w.see(BigUint::from(1_u8) << 256_usize);
    let (n, d) = candidate_input(mag, w);
    let scaled = w.see(&q * &n);
    let divisor = w.see(&d << 7_usize);
    let remainder = w.see(&scaled % &divisor);
    assert_eq!(remainder, BigUint::from(0_u8));
    let t = w.see(&scaled / &divisor);
    assert!(t < q);
    let mut lower_term = w.see(q.clone());
    let mut upper_term = w.see(q.clone());
    let mut lower = w.see(q.clone());
    let mut upper = w.see(q.clone());
    for k in 1..=64_u32 {
        let denominator = w.see(&q * k);
        let lower_product = w.see(&lower_term * &t);
        let upper_product = w.see(&upper_term * &t);
        lower_term = w.see(&lower_product / &denominator);
        upper_term = candidate_ceil(&upper_product, &denominator, w);
        lower = w.see(lower + &lower_term);
        upper = w.see(upper + &upper_term);
    }
    // For t<1 the omitted terms are <(66/65)/65!<1/Q. The exact
    // integer inequality establishing this is checked in the analytic test.
    upper = w.see(upper + 1_u8);
    for _ in 0..7 {
        let lower_square = w.see(&lower * &lower);
        let upper_square = w.see(&upper * &upper);
        lower = w.see(&lower_square / &q);
        upper = candidate_ceil(&upper_square, &q, w);
    }
    let a_remainder = w.see(&scaled % &d);
    assert_eq!(a_remainder, BigUint::from(0_u8));
    let a = w.see(&scaled / &d);
    (a, lower, upper)
}

fn candidate_pair(mag: u16, w: &mut Width) -> [u32; 2] {
    let (a, lower_exp, upper_exp) = candidate_exp(mag, w);
    let q = w.see(BigUint::from(1_u8) << 256_usize);
    let lower_denominator = w.see(&q + &lower_exp);
    let upper_denominator = w.see(&q + &upper_exp);
    let positive_lower_product = w.see(&a * &lower_exp);
    let positive_upper_product = w.see(&a * &upper_exp);
    let negative_product = w.see(&a * &q);
    let positive_lower = w.see(&positive_lower_product / &lower_denominator);
    let positive_upper = candidate_ceil(&positive_upper_product, &upper_denominator, w);
    let negative_lower = w.see(&negative_product / &upper_denominator);
    let negative_upper = candidate_ceil(&negative_product, &lower_denominator, w);
    let mut result = [0; 2];
    for (index, (lo, hi)) in [
        (&positive_lower, &positive_upper),
        (&negative_lower, &negative_upper),
    ].into_iter().enumerate() {
        assert!(lo <= hi);
        let low_bits = candidate_rn32(lo, &q, index == 1, w);
        let high_bits = candidate_rn32(hi, &q, index == 1, w);
        assert_eq!(low_bits, high_bits,
            "P256/N64 ambiguous input={:04x}, bounds=[{lo}/{q},{hi}/{q}]",
            mag | if index == 1 { 0x8000 } else { 0 });
        result[index] = low_bits;
    }
    result
}

// Independent oracle input: reduce the promoted BF16 significand directly.
// Interior a=n/d has n<=255 and d<=2^32. No candidate decoder reuse.
fn oracle_input(mag: u16, w: &mut Width) -> (BigUint, BigUint) {
    assert!(mag > TINY && mag < TAIL);
    let mut n = u32::from((mag % 128) + 128);
    let mut denominator_power = 134 - i32::from(mag / 128);
    assert!(denominator_power >= 0);
    while n % 2 == 0 && denominator_power > 0 {
        n /= 2;
        denominator_power -= 1;
    }
    assert!(n <= 255 && denominator_power <= 32);
    let numerator = w.see(BigUint::from(n));
    let denominator = w.see(BigUint::from(1_u8) << denominator_power as usize);
    (numerator, denominator)
}

// Universal positive F32 grid in units 2^-149. The binary search below never
// divides or invokes the candidate quotient/remainder rounder.
fn oracle_grid(word: u32, w: &mut Width) -> BigUint {
    assert!(word <= 0x7f7f_ffff);
    let exponent = word / 0x0080_0000;
    let fraction = word % 0x0080_0000;
    if exponent == 0 {
        w.see(BigUint::from(fraction))
    } else {
        let coefficient = w.see(BigUint::from(fraction + 0x0080_0000));
        w.see(coefficient << (exponent - 1) as usize)
    }
}

fn oracle_rn32(n: &BigUint, d: &BigUint, negative: bool, w: &mut Width) -> u32 {
    assert_ne!(*d, BigUint::from(0_u8));
    let target = w.see(n << 149_usize);
    let maximum = oracle_grid(0x7f7f_ffff, w);
    let maximum_scaled = w.see(&maximum * d);
    assert!(target <= maximum_scaled); // No overflow/clipping fallback.
    let mut lo = 0_u32;
    let mut hi = 0x7f7f_ffff_u32;
    while lo < hi {
        let mid = ((u64::from(lo) + u64::from(hi) + 1) / 2) as u32;
        let grid = oracle_grid(mid, w);
        let scaled = w.see(&grid * d);
        if scaled <= target { lo = mid; } else { hi = mid - 1; }
    }
    let low_grid = oracle_grid(lo, w);
    let exact = w.see(&low_grid * d);
    let magnitude = if exact == target || lo == 0x7f7f_ffff {
        lo
    } else {
        let high_grid = oracle_grid(lo + 1, w);
        let grid_sum = w.see(&low_grid + &high_grid);
        let midpoint_scaled = w.see(&grid_sum * d);
        let twice_target = w.see(&target << 1_usize);
        if twice_target < midpoint_scaled || (twice_target == midpoint_scaled && lo % 2 == 0) {
            lo
        } else {
            lo + 1
        }
    };
    (u32::from(negative) << 31) | magnitude
}

fn oracle_bf16(bits: u32) -> u16 {
    assert!(bits & 0x7f80_0000 != 0x7f80_0000);
    // Widen first: even the largest finite input cannot overflow this addition.
    let magnitude = u64::from(bits & 0x7fff_ffff);
    let biased = magnitude + 0x7fff + ((magnitude >> 16) & 1);
    let rounded = biased >> 16;
    assert!(rounded < 0x7f80);
    ((bits >> 16) as u16 & 0x8000) | rounded as u16
}

// Exact rational Taylor of exp(a), with no reduction or squaring:
// P_k=n^k, D_k=d^k*k!, S_k/D_k=sum_{j=0}^k a^j/j!.
fn oracle_pair(mag: u16, w: &mut Width) -> ([u32; 2], usize) {
    let (n, d) = oracle_input(mag, w);
    let mut p = w.see(BigUint::from(1_u8));
    let mut denominator = w.see(BigUint::from(1_u8));
    let mut sum = w.see(BigUint::from(1_u8));
    let mut result = [None, None];
    for k in 1..=512_u32 {
        p = w.see(&p * &n);
        let dk = w.see(&d * k);
        denominator = w.see(&denominator * &dk);
        let previous_sum_scaled = w.see(&sum * &dk);
        sum = w.see(previous_sum_scaled + &p);
        if ![128, 256, 512].contains(&k) { continue; }
        let d_next = w.see(&d * (k + 2));
        assert!(d_next > n);
        let difference = w.see(d_next - &n);
        let b = w.see(difference * (k + 1));
        // The first omitted term times a geometric bound is
        // P_k*n*(k+2)/(D_k*B), since all later ratios <=a/(k+2)<1.
        let sum_b = w.see(&sum * &b);
        let p_n = w.see(&p * &n);
        let tail_numerator = w.see(p_n * (k + 2));
        let upper_exp_n = w.see(sum_b + tail_numerator);
        let upper_exp_d = w.see(&denominator * &b);
        let lower_sum = w.see(&denominator + &sum);
        let upper_sum = w.see(&upper_exp_d + &upper_exp_n);
        for (index, slot) in result.iter_mut().enumerate() {
            if slot.is_some() { continue; } // Preserve FIRST unique checkpoint.
            let (low_n, low_d, high_n, high_d) = if index == 0 {
                // a*E/(1+E) increases with E.
                let ln = w.see(&n * &sum);
                let ld = w.see(&d * &lower_sum);
                let hn = w.see(&n * &upper_exp_n);
                let hd = w.see(&d * &upper_sum);
                (ln, ld, hn, hd)
            } else {
                // |SiLU(-a)|=a/(1+E) decreases with E.
                let ln = w.see(&n * &upper_exp_d);
                let ld = w.see(&d * &upper_sum);
                let hn = w.see(&n * &denominator);
                let hd = w.see(&d * &lower_sum);
                (ln, ld, hn, hd)
            };
            let low_bits = oracle_rn32(&low_n, &low_d, index == 1, w);
            let high_bits = oracle_rn32(&high_n, &high_d, index == 1, w);
            if low_bits == high_bits {
                *slot = Some(low_bits);
            } else if k == 512 {
                panic!("oracle ambiguous input={:04x}, bounds=[{low_n}/{low_d},{high_n}/{high_d}]",
                    mag | if index == 1 { 0x8000 } else { 0 });
            }
        }
        if let [Some(positive), Some(negative)] = result {
            return ([positive, negative], k as usize);
        }
    }
    panic!("oracle failed to resolve magnitude={mag:04x}");
}

// Analytic branch selection is deliberately NOT evidence for its own accuracy.
// Independent strict cell inequalities are checked separately below.
fn partition(mag: u16) -> usize {
    match mag {
        0 => 0,
        1..=TINY => 1,
        TAIL..=LAST_FINITE => 3,
        0x7f80..=0x7fff => 4,
        _ => 2,
    }
}

fn half_promoted(mag: u16) -> u32 {
    let promoted = u32::from(mag) << 16;
    if promoted < 0x0100_0000 {
        let coefficient = (promoted & 0x007f_ffff)
            | if promoted & 0x0080_0000 != 0 { 0x0080_0000 } else { 0 };
        assert_eq!(coefficient & 1, 0);
        coefficient >> 1
    } else {
        promoted - 0x0080_0000
    }
}

fn analytic_bits(mag: u16, negative: bool) -> u32 {
    let sign = u32::from(negative) << 31;
    match partition(mag) {
        0 => sign,
        1 => sign | half_promoted(mag),
        3 => if negative { sign } else { u32::from(mag) << 16 },
        _ => panic!("not an analytic input: {mag:04x}"),
    }
}

#[test]
fn silu_full_classification_signed_zero_and_partition_counts() {
    let mut counts = [0_usize; 5];
    for word in 0..=u16::MAX {
        let mag = word & 0x7fff;
        let class = partition(mag);
        counts[class] += 1;
        if matches!(class, 0 | 1 | 3) {
            let rounded = analytic_bits(mag, word & 0x8000 != 0);
            assert_eq!(rounded >> 31, u32::from(word >> 15));
            assert_eq!(candidate_bf16(rounded), oracle_bf16(rounded));
        }
    }
    assert_eq!(counts, [2, 26_112, 8_190, 30_976, 256]);
    assert_eq!(counts[..4].iter().sum::<usize>(), 65_280);
    assert_eq!(counts.iter().sum::<usize>(), 65_536);
    assert_eq!(analytic_bits(0, false), 0);
    assert_eq!(analytic_bits(0, true), 0x8000_0000);
    println!("partition counts: {counts:?}");
}

#[test]
fn silu_exact_tail_tiny_cell_inequalities_and_boundary_neighbors() {
    let q = BigUint::from(1_u8) << 256_usize;
    let mut factorial = BigUint::from(1_u8);
    for k in 1..=65_u32 { factorial *= k; }
    assert!((&q * 66_u32) < (&factorial * 65_u32));

    // e^128 > 128^128/128! > 128*2^150. Thus
    // 128/(1+e^128)<2^-150, strictly inside negative-zero's RN32 cell.
    // a*e^-a decreases for a>=128 (derivative e^-a*(1-a)<0),
    // so this SAME strict bound holds for every larger finite BF16 a.
    let mut factorial128 = BigUint::from(1_u8);
    for k in 1..=128_u32 { factorial128 *= k; }
    let exponential_term_numerator = BigUint::from(128_u32).pow(128);
    let threshold = &factorial128 << 157_usize;
    assert!(exponential_term_numerator > threshold);
    let mut proof_width = Width::new(22_000);
    // x/2 is exact F32 throughout the entire finite BF16 domain, not
    // only in the tiny branch. Sign symmetry also establishes negative x.
    for mag in 0..=LAST_FINITE {
        let x_grid = oracle_grid(u32::from(mag) << 16, &mut proof_width);
        let half_grid = oracle_grid(half_promoted(mag), &mut proof_width);
        assert_eq!(&half_grid << 1_usize, x_grid, "exact half {mag:04x}");
    }
    for mag in TAIL..=LAST_FINITE {
        let center = u32::from(mag) << 16;
        let grid = oracle_grid(center, &mut proof_width);
        let previous = oracle_grid(center - 1, &mut proof_width);
        let lower_gap = &grid - &previous;
        // Positive correction a/(1+e^a)<2^-150, whereas every lower
        // midpoint distance here is >=2^-18. Includes binade boundaries.
        assert!(lower_gap >= (BigUint::from(1_u8) << 132_usize));
    }

    // For either sign, delta=SiLU(x)-x/2=(x/2)*tanh(x/2) is positive
    // and STRICTLY less than x^2/4. x/2 is exactly representable: even
    // the least BF16 subnormal halves to 2^-134, on the F32 grid.
    // Check the UPPER signed midpoint, not a symmetric-ulp assumption.
    for mag in 1..=TINY {
        let x_grid = oracle_grid(u32::from(mag) << 16, &mut proof_width);
        let center = half_promoted(mag);
        let y_grid = oracle_grid(center, &mut proof_width);
        assert_eq!(&y_grid << 1_usize, x_grid);
        let positive_next = oracle_grid(center + 1, &mut proof_width);
        let negative_next = oracle_grid(center - 1, &mut proof_width);
        let gaps = [&positive_next - &y_grid, &y_grid - &negative_next];
        // delta < y^2 = y_grid^2*2^-298. Upper midpoint distance is
        // gap*2^-150; equality of bounds would still be strict for delta.
        let square = &y_grid * &y_grid;
        for gap in gaps {
            let midpoint_bound = gap << 148_usize;
            assert!(square <= midpoint_bound, "tiny cell {mag:04x}");
        }
    }
    for (mag, expected) in [
        (0, 0), (1, 1), (TINY - 1, 1), (TINY, 1), (TINY + 1, 2),
        (TAIL - 1, 2), (TAIL, 3), (TAIL + 1, 3),
        (LAST_FINITE, 3), (LAST_FINITE + 1, 4),
    ] {
        for sign in [0, 0x8000] {
            assert_eq!(partition((mag | sign) & 0x7fff), expected);
        }
    }
}

#[test]
fn silu_independent_rounders_midpoints_parity_and_transitions() {
    let mut candidate_width = Width::new(1024);
    let mut oracle_width = Width::new(22_000);
    // Independent dyadic fixtures: adjacent values = listed integer units
    // times 2^power. No candidate decode or oracle grid constructs fixtures.
    for (lo, low_units, high_units, power) in [
        (0, 0_u32, 1_u32, -149_i32),
        (1, 1, 2, -149), (2, 2, 3, -149),
        (0x007f_fffe, 8_388_606, 8_388_607, -149),
        (0x007f_ffff, 8_388_607, 8_388_608, -149),
        (0x0080_0000, 8_388_608, 8_388_609, -149),
        (0x00ff_ffff, 16_777_215, 16_777_216, -149),
        (0x3eff_ffff, 16_777_215, 16_777_216, -25),
        (0x3f00_0000, 8_388_608, 8_388_609, -24),
        (0x3f7f_ffff, 16_777_215, 16_777_216, -24),
        (0x3f80_0000, 8_388_608, 8_388_609, -23),
        (0x42ff_ffff, 16_777_215, 16_777_216, -17),
    ] {
        for offset in [-1_i64, 0, 1] {
            let units = (i64::from(low_units + high_units) * 4 + offset) as u64;
            let n = BigUint::from(units);
            let d = BigUint::from(1_u8) << (3 - power) as usize;
            let expected = if offset < 0 || (offset == 0 && lo & 1 == 0) { lo } else { lo + 1 };
            for negative in [false, true] {
                let expected = expected | (u32::from(negative) << 31);
                assert_eq!(candidate_rn32(&n, &d, negative, &mut candidate_width), expected);
                assert_eq!(oracle_rn32(&n, &d, negative, &mut oracle_width), expected);
            }
        }
    }
    for negative in [false, true] {
        let n = BigUint::from(0_u8);
        let d = BigUint::from(1_u8);
        let expected = u32::from(negative) << 31;
        assert_eq!(candidate_rn32(&n, &d, negative, &mut candidate_width), expected);
        assert_eq!(oracle_rn32(&n, &d, negative, &mut oracle_width), expected);
        for (word, expected) in [
            (0, 0), (0x0000_8000, 0), (0x0000_8001, 1),
            (0x0001_8000, 2), (0x007f_8000, 0x0080),
            (0x3f80_8000, 0x3f80), (0x3f81_8000, 0x3f82),
            (0x3fff_8000, 0x4000),
        ] {
            let bits = word | (u32::from(negative) << 31);
            let expected = expected | (u16::from(negative) << 15);
            assert_eq!(candidate_bf16(bits), expected);
            assert_eq!(oracle_bf16(bits), expected);
        }
    }
}

#[test]
fn silu_exhaustive_signed_middle_rn32_before_bf16() {
    let mut candidate_width = Width::new(1024);
    let mut oracle_width = Width::new(22_000);
    let mut max_checkpoint = 0;
    let mut magnitudes = 0;
    let mut signed_middle = 0;
    let mut analytic = 2; // Both signed zeros.
    let mut nonfinite = 0;
    for mag in 1..=0x7fff_u16 {
        match partition(mag) {
            1 | 3 => {
                for negative in [false, true] {
                    let bits = analytic_bits(mag, negative);
                    assert_eq!(candidate_bf16(bits), oracle_bf16(bits));
                    analytic += 1;
                }
            }
            2 => {
                let candidate = candidate_pair(mag, &mut candidate_width);
                let (oracle, checkpoint) = oracle_pair(mag, &mut oracle_width);
                max_checkpoint = max_checkpoint.max(checkpoint);
                magnitudes += 1;
                for index in 0..2 {
                    let input = mag | if index == 1 { 0x8000 } else { 0 };
                    // Stage F32 is mandatory: BF16 agreement cannot mask it.
                    assert_eq!(candidate[index], oracle[index], "RN32 input={input:04x}");
                    assert_eq!(candidate_bf16(candidate[index]), oracle_bf16(oracle[index]),
                        "RN16 after RN32 input={input:04x}");
                    signed_middle += 1;
                }
            }
            4 => nonfinite += 2,
            _ => panic!("uncovered magnitude {mag:04x}"),
        }
    }
    assert_eq!(magnitudes, 4095);
    assert_eq!(signed_middle, 8190);
    assert_eq!(analytic, 57_090);
    assert_eq!(analytic + signed_middle, 65_280);
    assert_eq!(nonfinite, 256);
    assert_eq!(analytic + signed_middle + nonfinite, 65_536);
    println!("max successful checkpoint degree={max_checkpoint}, candidate individual width={}, oracle individual width={}",
        candidate_width.max, oracle_width.max);
}

#[test]
fn silu_explicit_double_rounding_and_signed_underflow_regressions() {
    // Least BF16 subnormal a=2^-133. Exact SiLU(a)=a/2+delta,
    // 0<delta<a^2/4=2^-268. Therefore direct RN16 is 1, strictly
    // above the BF16 zero/one midpoint, and below the one/two midpoint.
    // RN32 is exactly a/2=2^-134, then RN16 ties to even zero.
    let correction_denominator = BigUint::from(1_u8) << 268_usize;
    let bf16_quantum_denominator = BigUint::from(1_u8) << 133_usize;
    assert!(correction_denominator > bf16_quantum_denominator);
    let staged_positive = analytic_bits(1, false);
    assert_eq!(staged_positive, 1_u32 << 15); // 2^-134 on 2^-149 grid.
    assert_eq!(candidate_bf16(staged_positive), 0);
    assert_eq!(oracle_bf16(staged_positive), 0);
    let direct_real_bf16 = 1_u16; // Derived by the strict inequalities above.
    assert_ne!(direct_real_bf16, candidate_bf16(staged_positive));
    let staged_negative = analytic_bits(1, true);
    assert_eq!(staged_negative, 0x8000_0000 | (1_u32 << 15));
    assert_eq!(candidate_bf16(staged_negative), 0x8000);
    assert_eq!(oracle_bf16(staged_negative), 0x8000);
    // For -a the positive correction puts the real result ABOVE -a/2,
    // so direct RN16 is already negative zero, with sign preserved.

    let mut candidate_width = Width::new(1024);
    let mut oracle_width = Width::new(22_000);
    // +2^-23: center 2^-24, upper half-ulp 2^-48; delta<2^-48.
    // -2^-24: center -2^-25, asymmetric upper half-ulp 2^-50;
    // delta<2^-50. These expected RN32 words follow strict cells, not
    // a guessed libm output or agreement between the two implementations.
    for (mag, index, power, negative) in [
        (0x3400, 0_usize, -24_i32, false),
        (0x3380, 1_usize, -25_i32, true),
    ] {
        let expected = (((power + 127) as u32) << 23) | (u32::from(negative) << 31);
        let center = expected & 0x7fff_ffff;
        let center_grid = oracle_grid(center, &mut oracle_width);
        let neighbor_grid = oracle_grid(
            if negative { center - 1 } else { center + 1 },
            &mut oracle_width,
        );
        let upper_gap = if negative {
            &center_grid - &neighbor_grid
        } else {
            &neighbor_grid - &center_grid
        };
        // Here x/2=center and delta<center^2, so check that this
        // strict bound equals the signed upper midpoint distance.
        let squared_center = &center_grid * &center_grid;
        let upper_midpoint = upper_gap << 148_usize;
        assert_eq!(squared_center, upper_midpoint);
        let candidate = candidate_pair(mag, &mut candidate_width);
        let (oracle, _) = oracle_pair(mag, &mut oracle_width);
        assert_eq!(candidate[index], expected);
        assert_eq!(oracle[index], expected);
        let expected_bf16 = (((power + 127) as u16) << 7) | (u16::from(negative) << 15);
        assert_eq!(candidate_bf16(expected), expected_bf16);
        assert_eq!(oracle_bf16(expected), expected_bf16);
    }
    for mag in [TAIL, TAIL + 1, LAST_FINITE] {
        assert_eq!(analytic_bits(mag, true), 0x8000_0000);
        assert_eq!(candidate_bf16(analytic_bits(mag, true)), 0x8000);
        assert_eq!(oracle_bf16(analytic_bits(mag, true)), 0x8000);
    }
}

#[test]
fn silu_production_matches_candidate_and_oracle_for_all_finite_encodings() {
    let mut candidate_width = Width::new(1024);
    let mut oracle_width = Width::new(22_000);
    let mut scratch = [0xd1a6_502f_37bc_849e_u64; 125];
    let w32 = super::silu_bf16_rn32_work().unwrap();
    let w16 = super::silu_bf16_work().unwrap();
    let s = super::silu_bf16_scratch_words();
    let mut finite = 0;
    let mut middle = 0;
    for mag in 0..=LAST_FINITE {
        let (candidate, oracle) = if partition(mag) == 2 {
            middle += 1;
            let candidate = candidate_pair(mag, &mut candidate_width);
            let (oracle, _) = oracle_pair(mag, &mut oracle_width);
            (candidate, oracle)
        } else {
            // Frozen independent strict-cell proofs qualify all analytic inputs;
            // the direct rational exp oracle is used for EVERY interior input.
            let analytic = [analytic_bits(mag, false), analytic_bits(mag, true)];
            (analytic, analytic)
        };
        for index in 0..2 {
            let word = mag | if index == 1 { 0x8000 } else { 0 };
            assert_eq!(candidate[index], oracle[index], "reference RN32 {word:04x}");
            scratch.fill(0xd1a6_502f_37bc_849e);
            let actual = super::silu_bf16_rn32_with_scratch(word, &mut scratch, w32).unwrap();
            assert_eq!(actual.to_bits(), candidate[index], "candidate {word:04x}");
            assert_eq!(actual.to_bits(), oracle[index], "oracle RN32 {word:04x}");
            assert!(scratch[..s].iter().all(|v| *v == 0));
            assert!(scratch[s..].iter().all(|v| *v == 0xd1a6_502f_37bc_849e));
            scratch[..s].fill(0xd1a6_502f_37bc_849e);
            let actual = super::silu_bf16_with_scratch(word, &mut scratch, w16).unwrap();
            assert_eq!(actual, oracle_bf16(oracle[index]), "RN16 after RN32 {word:04x}");
            assert!(scratch[..s].iter().all(|v| *v == 0));
            assert!(scratch[s..].iter().all(|v| *v == 0xd1a6_502f_37bc_849e));
            finite += 1;
        }
    }
    assert_eq!(middle, 4095);
    assert_eq!(finite, 65_280);
}

#[test]
fn silu_production_rejects_all_nonfinite_encodings_before_preflight() {
    let w32 = super::silu_bf16_rn32_work().unwrap();
    let w16 = super::silu_bf16_work().unwrap();
    let mut nonfinite = 0;
    for mag in 0x7f80..=0x7fff_u16 {
        for sign in [0, 0x8000] {
            let word = mag | sign;
            for (len, work) in [(0, 0), (121, 0), (125, 0), (125, w16)] {
                let mut scratch = vec![0x673a_80c5_2f49_b1ed_u64; len];
                let before = scratch.clone();
                assert_eq!(super::silu_bf16_rn32_with_scratch(word, &mut scratch, work),
                    Err(super::Error::NonFinite));
                assert_eq!(scratch, before);
                assert_eq!(super::silu_bf16_with_scratch(word, &mut scratch, work),
                    Err(super::Error::NonFinite));
                assert_eq!(scratch, before);
            }
            let mut scratch = [0x673a_80c5_2f49_b1ed_u64; 122];
            assert_eq!(super::silu_bf16_rn32_with_scratch(word, &mut scratch, w32),
                Err(super::Error::NonFinite));
            assert_eq!(scratch, [0x673a_80c5_2f49_b1ed; 122]);
            nonfinite += 1;
        }
    }
    assert_eq!(nonfinite, 256);
}

#[test]
fn silu_production_exact_workspace_and_work_boundaries() {
    let s = super::silu_bf16_scratch_words();
    let w32 = super::silu_bf16_rn32_work().unwrap();
    let w16 = super::silu_bf16_work().unwrap();
    assert_eq!(s, 122);
    assert_eq!(w32, 3_888_768);
    assert_eq!(w16, 3_889_024);
    let mut width = Width::new(22_000);
    let (oracle, _) = oracle_pair(0x3f80, &mut width);
    for bf16 in [false, true] {
        let required = if bf16 { w16 } else { w32 };
        for len in [s - 1, s, s + 7] {
            for work in [required - 1, required, required + 1] {
                let mut scratch = vec![0x1c59_8f47_03ad_6be2_u64; len];
                let before = scratch.clone();
                let actual = if bf16 {
                    super::silu_bf16_with_scratch(0x3f80, &mut scratch, work).map(u32::from)
                } else {
                    super::silu_bf16_rn32_with_scratch(0x3f80, &mut scratch, work).map(f32::to_bits)
                };
                if len < s {
                    assert_eq!(actual, Err(super::Error::Capacity));
                    assert_eq!(scratch, before);
                } else if work < required {
                    assert_eq!(actual, Err(super::Error::WorkLimit));
                    assert_eq!(scratch, before);
                } else {
                    assert_eq!(actual, Ok(if bf16 { u32::from(oracle_bf16(oracle[0])) } else { oracle[0] }));
                    assert!(scratch[..s].iter().all(|v| *v == 0));
                    assert_eq!(scratch[s..], before[s..]);
                }
            }
        }
    }
    let mut scratch = [9_u64; 122];
    assert_eq!(super::silu_bf16_with_scratch(0x3f80, &mut scratch, w32), Err(super::Error::WorkLimit));
    assert_eq!(scratch, [9; 122]);
    assert_eq!(super::silu_bf16_rn32_with_scratch(0x3f80, &mut scratch, w16).unwrap().to_bits(), oracle[0]);
    assert_eq!(scratch, [0; 122]);
}

#[test]
fn silu_production_priority_and_refusal_preserve_entire_scratch() {
    let s = super::silu_bf16_scratch_words();
    for bf16 in [false, true] {
        let required = if bf16 { super::silu_bf16_work() } else { super::silu_bf16_rn32_work() }.unwrap();
        for word in [0x7f80_u16, 0xff80, 0x7f81, 0xffff, 0, 0x8000, 0x3f80, 0xbf80] {
            for len in [0, s - 1, s, s + 5] {
                for work in [0, required - 1, required] {
                    let expected = if word & 0x7f80 == 0x7f80 { super::Error::NonFinite }
                        else if len < s { super::Error::Capacity }
                        else if work < required { super::Error::WorkLimit }
                        else { continue }; // Qualified successes are tested separately.
                    let mut scratch = vec![0xb72c_0f89_65a1_d34e_u64; len];
                    let before = scratch.clone();
                    let actual = if bf16 {
                        super::silu_bf16_with_scratch(word, &mut scratch, work).map(u32::from)
                    } else {
                        super::silu_bf16_rn32_with_scratch(word, &mut scratch, work).map(f32::to_bits)
                    };
                    assert_eq!(actual, Err(expected));
                    assert_eq!(scratch, before);
                }
            }
        }
    }
}

#[test]
fn silu_production_wipes_active_prefix_preserves_spare_all_branches() {
    let s = super::silu_bf16_scratch_words();
    let mut width = Width::new(22_000);
    for word in [0_u16, 0x8000, 1, 0x8001, TINY, TINY | 0x8000,
        TINY + 1, (TINY + 1) | 0x8000, 0x3f80, 0xbf80,
        TAIL - 1, (TAIL - 1) | 0x8000, TAIL, TAIL | 0x8000, LAST_FINITE, LAST_FINITE | 0x8000] {
        let mag = word & 0x7fff;
        let negative = word & 0x8000 != 0;
        let expected = if partition(mag) == 2 {
            oracle_pair(mag, &mut width).0[usize::from(negative)]
        } else { analytic_bits(mag, negative) };
        for bf16 in [false, true] {
            let work = if bf16 { super::silu_bf16_work() } else { super::silu_bf16_rn32_work() }.unwrap();
            for spare in [0, 1, 13] {
                let mut scratch = vec![0x81fa_0c53_d29e_764b_u64; s + spare];
                let actual = if bf16 {
                    super::silu_bf16_with_scratch(word, &mut scratch, work).map(u32::from)
                } else {
                    super::silu_bf16_rn32_with_scratch(word, &mut scratch, work).map(f32::to_bits)
                };
                assert_eq!(actual, Ok(if bf16 { u32::from(oracle_bf16(expected)) } else { expected }));
                assert!(scratch[..s].iter().all(|v| *v == 0));
                assert!(scratch[s..].iter().all(|v| *v == 0x81fa_0c53_d29e_764b));
            }
        }
    }
}

#[test]
fn silu_production_local_fault_errors_and_unwind_wipe_active_prefix() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let s = super::silu_bf16_scratch_words();
    for bf16 in [false, true] {
        let work = if bf16 { super::silu_bf16_work() } else { super::silu_bf16_rn32_work() }.unwrap();
        for word in [0_u16, 0x8000, 1, 0x8001, TAIL, TAIL | 0x8000, 0x3f80, 0xbf80] {
            for injection in 1..=6 {
                if !matches!(injection, 1 | 4) && partition(word & 0x7fff) != 2 {
                    continue; // Taylor/endpoint faults require a genuine middle input.
                }
                let mut scratch = vec![0x927c_a064_3b5e_f18d_u64; s + 7];
                let actual = catch_unwind(AssertUnwindSafe(|| {
                    super::silu::with_fault(word, &mut scratch, work, bf16, injection)
                }));
                if injection <= 3 { assert_eq!(actual.unwrap(), Err(super::Error::IntegerRange)); }
                else { assert!(actual.is_err()); }
                assert!(scratch[..s].iter().all(|v| *v == 0));
                assert!(scratch[s..].iter().all(|v| *v == 0x927c_a064_3b5e_f18d));
            }
        }
    }
}

#[test]
fn silu_production_rn32_before_bf16_and_signed_zero_regressions() {
    let w32 = super::silu_bf16_rn32_work().unwrap();
    let w16 = super::silu_bf16_work().unwrap();
    // Expectations derive from exact half, strict tiny/signed-cell inequalities,
    // and tail proofs in the frozen tests, not the production BF16 converter.
    for (word, expected) in [
        (0_u16, 0_u32), (0x8000, 0x8000_0000),
        (1, 1 << 15), (0x8001, 0x8000_0000 | (1 << 15)),
        (0x3400, (127 - 24) << 23),
        (0xb380, 0x8000_0000 | ((127 - 25) << 23)),
        (TAIL, u32::from(TAIL) << 16),
        (TAIL | 0x8000, 0x8000_0000), (LAST_FINITE | 0x8000, 0x8000_0000),
    ] {
        let mut scratch = [0x65a2_08fc_97be_431d_u64; 125];
        let actual = super::silu_bf16_rn32_with_scratch(word, &mut scratch, w32).unwrap();
        assert_eq!(actual.to_bits(), expected);
        assert_eq!(super::silu_bf16_with_scratch(word, &mut scratch, w16), Ok(oracle_bf16(expected)));
        assert_eq!(scratch[..122], [0; 122]);
        assert_eq!(scratch[122..], [0x65a2_08fc_97be_431d; 3]);
    }
    // Positive minimum BF16: real SiLU lies strictly above the BF16 half
    // subnormal, hence direct RN16=1; staged RN32=2^-134 then RN16=+0.
    let mut scratch = [7_u64; 122];
    assert_eq!(super::silu_bf16_with_scratch(1, &mut scratch, w16), Ok(0));
    assert_ne!(super::silu_bf16_with_scratch(1, &mut scratch, w16), Ok(1));
    assert_eq!(scratch, [0; 122]);
}
