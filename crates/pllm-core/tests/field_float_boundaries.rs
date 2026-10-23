//! External conformance samples around every reachable BF16 cell's boundaries.
//! The oracle searches the exact integer/rational BF16 lattice, without F32 or
//! the implementation's significand-rounding helper. This is not exhaustive
//! residue testing or a source, budget, no-wrap, privacy or admission contract.

use std::cmp::Ordering;

use pllm_core::{
    coded_linear::CODED_LINEAR_FIELD,
    field_float::{decode_reply_bf16, f32_to_bf16, Error},
};

// Independently literal BabyBear modulus and the exact positive centred limit.
const MODULUS: u32 = 2_013_265_921;
const MAX_CENTRED: u32 = 1_006_632_960;
const LAST_WORD: u16 = 0x4670; // 15360 = MAX_CENTRED / 65536.

#[derive(Clone, Copy)]
struct Rational {
    numerator: u128,
    denominator: u128,
}

impl Rational {
    fn floor(self) -> u128 {
        self.numerator / self.denominator
    }

    fn is_integer(self) -> bool {
        self.numerator % self.denominator == 0
    }

    fn ceil(self) -> u128 {
        self.floor() + u128::from(!self.is_integer())
    }

    fn compare_integer(self, integer: u128) -> Ordering {
        // Avoid cross multiplication: even the smallest BF16 denominator is
        // supported without multiplying it by a large centred numerator.
        match self.floor().cmp(&integer) {
            Ordering::Equal if !self.is_integer() => Ordering::Greater,
            comparison => comparison,
        }
    }
}

/// A nonnegative BF16 lattice value, measured in 1/65536 integer input ticks.
fn lattice_ticks(word: u16) -> Rational {
    assert!(word <= LAST_WORD + 1);
    let exponent = i32::from(word >> 7);
    let fraction = u128::from(word & 127);
    let (coefficient, power) = if exponent == 0 {
        (fraction, -117)
    } else {
        (128 + fraction, exponent - 118)
    };
    if power >= 0 {
        Rational {
            numerator: coefficient << power,
            denominator: 1,
        }
    } else {
        Rational {
            numerator: coefficient,
            denominator: 1_u128 << -power,
        }
    }
}

fn midpoint(lower: u16) -> Rational {
    let a = lattice_ticks(lower);
    let b = lattice_ticks(lower + 1);
    // Denominators are powers of two, at most 2^117. The sum has a
    // denominator at most 2^118 and a numerator below 2^32 in this range.
    let common = a.denominator.max(b.denominator);
    Rational {
        numerator: a.numerator * (common / a.denominator) + b.numerator * (common / b.denominator),
        denominator: 2 * common,
    }
}

fn positive_oracle(ticks: u128) -> u16 {
    assert!(ticks <= u128::from(MAX_CENTRED));
    if ticks == 0 {
        return 0;
    }
    // Locate adjacent exact lattice points, not the input's high bit or a
    // rounded significand. LAST_WORD + 1 is strictly above every legal input.
    let (mut lower, mut upper) = (0_u16, LAST_WORD + 1);
    while upper - lower > 1 {
        let candidate = lower + (upper - lower) / 2;
        if lattice_ticks(candidate).compare_integer(ticks) == Ordering::Greater {
            upper = candidate;
        } else {
            lower = candidate;
        }
    }
    if lattice_ticks(lower).compare_integer(ticks) == Ordering::Equal {
        return lower;
    }
    match midpoint(lower).compare_integer(ticks) {
        Ordering::Greater => lower,
        Ordering::Less => upper,
        Ordering::Equal if lower & 1 == 0 => lower,
        Ordering::Equal => upper,
    }
}

fn canonical_residue(centred: i64) -> u32 {
    assert!(centred.unsigned_abs() <= u64::from(MAX_CENTRED));
    u32::try_from(if centred < 0 {
        i64::from(MODULUS) + centred
    } else {
        centred
    })
    .unwrap()
}

fn assert_centred(centred: i64) {
    let magnitude = positive_oracle(u128::from(centred.unsigned_abs()));
    let expected = magnitude | if centred < 0 { 0x8000 } else { 0 };
    assert_eq!(
        decode_reply_bf16(canonical_residue(centred)),
        Ok(expected),
        "centred={centred}, expected BF16={expected:04x}"
    );
}

fn check_boundary(boundary: Rational) -> usize {
    let anchors = [boundary.floor(), boundary.ceil()];
    let mut checks = 0;
    for (index, anchor) in anchors.into_iter().enumerate() {
        if index == 1 && anchor == anchors[0] {
            continue;
        }
        // Integral boundaries: exact tie and +/-1 input tick. Sub-tick
        // boundaries: both bounding ticks and their +/-1 neighbours.
        for offset in [-1_i128, 0, 1] {
            let ticks = i128::try_from(anchor).unwrap() + offset;
            if !(0..=i128::from(MAX_CENTRED)).contains(&ticks) {
                continue;
            }
            let ticks = i64::try_from(ticks).unwrap();
            assert_centred(ticks);
            checks += 1;
            if ticks != 0 {
                assert_centred(-ticks);
                checks += 1;
            }
        }
    }
    checks
}

#[test]
fn every_reachable_cell_boundary_has_integer_tick_neighbours_on_both_signs() {
    let mut cells = 0;
    let mut boundaries = 1;
    let mut checks = check_boundary(midpoint(0)); // Reachable zero cell.
    for word in 1..=LAST_WORD {
        let low = midpoint(word - 1);
        let high = midpoint(word);
        let odd = word & 1 != 0;
        // An odd significand loses both exact ties. Determine whether this
        // cell contains any legal positive integer input, without calling SUT.
        let first = (low.ceil() + u128::from(odd && low.is_integer())).max(1);
        let last =
            (high.floor() - u128::from(odd && high.is_integer())).min(u128::from(MAX_CENTRED));
        if first > last {
            continue;
        }
        cells += 1;
        for boundary in [low, high] {
            // The upper wall of the terminal cell is outside the field input
            // range. The actual centred endpoints are checked separately.
            if boundary.compare_integer(u128::from(MAX_CENTRED)) != Ordering::Greater {
                boundaries += 1;
                checks += check_boundary(boundary);
            }
        }
    }
    assert_eq!(cells, 3_056);
    assert_eq!(boundaries, 6_112); // Shared walls intentionally visited twice.
    assert_eq!(checks, 39_734); // Observations, not unique/all field residues.
}

#[test]
fn zero_minimum_ticks_and_centred_endpoints_have_literal_encodings() {
    assert_eq!(CODED_LINEAR_FIELD, MODULUS);
    assert_eq!(
        lattice_ticks(LAST_WORD).compare_integer(u128::from(MAX_CENTRED)),
        Ordering::Equal
    );
    for (residue, expected) in [
        (0, 0x0000),
        (1, 0x3780),
        (MODULUS - 1, 0xb780),
        (MAX_CENTRED - 1, 0x4670),
        (MAX_CENTRED, 0x4670),
        (MAX_CENTRED + 1, 0xc670),
        (MAX_CENTRED + 2, 0xc670),
    ] {
        assert_eq!(decode_reply_bf16(residue), Ok(expected));
    }
}

#[test]
fn literal_258_not_f32_bridge_256_on_both_signs() {
    for (centred, exact, bridge) in [
        (16_842_753_i64, 0x4381, 0x4380),  // +258, NOT +256.
        (-16_842_753_i64, 0xc381, 0xc380), // -258, NOT -256.
    ] {
        assert_eq!(
            decode_reply_bf16(canonical_residue(centred)),
            Ok(exact),
            "direct rational conversion must produce signed 258"
        );
        // Deliberately WRONG bridge: a regression diagnostic, never the oracle
        // used by the boundary sweep above. F32 erases the decisive input tick.
        assert_eq!(
            f32_to_bf16(centred as f32 / 65_536.0),
            Ok(bridge),
            "F32 bridge diagnostic produces signed 256, not signed 258"
        );
    }
}

#[test]
fn foreign_residue_prefix_representatives_preserve_noncanonical_error() {
    for residue in [
        MODULUS,
        MODULUS + 1,
        0x7fff_ffff,
        0x8000_0000,
        u32::MAX - 1,
        u32::MAX,
    ] {
        assert_eq!(decode_reply_bf16(residue), Err(Error::NonCanonical));
    }
    let mut checks = 0;
    // Three representatives from EVERY 16-bit-prefix interval intersecting
    // [p, u32::MAX], including p, the signed-u32 boundary and u32::MAX.
    // This partitions the invalid range; it does not test every member.
    for prefix in MODULUS >> 16..=u32::from(u16::MAX) {
        let first = (prefix << 16).max(MODULUS);
        let last = (prefix << 16) | 0xffff;
        for residue in [first, first + (last - first) / 2, last] {
            assert_eq!(
                decode_reply_bf16(residue),
                Err(Error::NonCanonical),
                "foreign residue={residue} must not reduce or recenter"
            );
            checks += 1;
        }
    }
    assert_eq!(checks, 104_448);
    assert_eq!(Error::NonCanonical.to_string(), "NonCanonical");
}
