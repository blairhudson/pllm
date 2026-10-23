//! TEST ONLY: independent D29 integer equations, not a native implementation.
//!
//! Mount in the owning region with:
//! `#[cfg(test)] #[path = "nexp_integer_oracle.rs"] mod integer_oracle;`
//! No native owner/lease/certificate is constructed here. Clear values and masks
//! are deliberately visible in this synthetic reference; never import it into
//! production. No libm/exp generation, fitted domain or profile choice. Literal
//! fixtures are independent of the verifier's Rust arrays. Separate floor tests
//! must pass before treating those bytes as the selected mathematical tables.
//! Wire/use constants express the approved contract, NOT observed ledger burns
//! or wire validation. Integer correctness proves neither privacy nor fresh RNG.

#![cfg(test)]

const N50: u64 = 1u64 << 50;
const INPUT_LIMIT: i64 = 1i64 << 38;
pub(super) type Tables = [[u64; 256]; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Error {
    Domain,
    Shape,
    Table,
    Overflow,
}

fn modulus(bits: u8) -> Result<u64, Error> {
    if !(1..=50).contains(&bits) {
        return Err(Error::Domain);
    }
    1u64.checked_shl(u32::from(bits)).ok_or(Error::Overflow)
}
fn canonical(word: u64, bits: u8) -> Result<(), Error> {
    if word >= modulus(bits)? {
        return Err(Error::Domain);
    }
    Ok(())
}
fn signed(word: u64, bits: u8) -> Result<i128, Error> {
    canonical(word, bits)?;
    let ring = i128::from(modulus(bits)?);
    Ok(if i128::from(word) < ring / 2 {
        i128::from(word)
    } else {
        i128::from(word) - ring
    })
}
fn reduce(value: i128, bits: u8) -> Result<u64, Error> {
    u64::try_from(value.rem_euclid(i128::from(modulus(bits)?))).map_err(|_| Error::Overflow)
}

/// Clear TEST data, emphatically not CommonMasked/DealerMask native capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Additive {
    pub word: u64,
    pub mask: u64,
    pub bits: u8,
}
impl Additive {
    pub fn new(clear_word: u64, mask: u64, bits: u8) -> Result<Self, Error> {
        canonical(clear_word, bits)?;
        canonical(mask, bits)?;
        Ok(Self {
            word: reduce(i128::from(clear_word) + i128::from(mask), bits)?,
            mask,
            bits,
        })
    }
    pub fn clear(self) -> Result<u64, Error> {
        canonical(self.word, self.bits)?;
        canonical(self.mask, self.bits)?;
        reduce(i128::from(self.word) - i128::from(self.mask), self.bits)
    }
    pub fn project(self, bits: u8) -> Result<Self, Error> {
        self.clear()?;
        if bits > self.bits {
            return Err(Error::Domain);
        }
        let ring = modulus(bits)?;
        Ok(Self {
            word: self.word % ring,
            mask: self.mask % ring,
            bits,
        })
    }
    pub fn add(self, constant: i128) -> Result<Self, Error> {
        self.clear()?;
        Ok(Self {
            word: reduce(
                i128::from(self.word)
                    .checked_add(constant)
                    .ok_or(Error::Overflow)?,
                self.bits,
            )?,
            ..self
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct XorBit {
    pub word: u8,
    pub mask: u8,
}
impl XorBit {
    pub fn new(clear: u8, mask: u8) -> Result<Self, Error> {
        if clear > 1 || mask > 1 {
            return Err(Error::Domain);
        }
        Ok(Self {
            word: clear ^ mask,
            mask,
        })
    }
    pub fn clear(self) -> Result<u8, Error> {
        if self.word > 1 || self.mask > 1 {
            return Err(Error::Domain);
        }
        Ok(self.word ^ self.mask)
    }
    pub fn not(self) -> Result<Self, Error> {
        self.clear()?;
        Ok(Self {
            word: self.word ^ 1,
            ..self
        })
    }
}

/// Select's raw algebra, not selecting directly on the masked predicate word.
pub(super) fn select(
    predicate: XorBit,
    value: Additive,
    output_mask: u64,
) -> Result<Additive, Error> {
    let bit = i128::from(predicate.clear()?);
    value.clear()?;
    canonical(output_mask, value.bits)?;
    let unmasked_term = i128::from(value.word) - i128::from(value.mask);
    Ok(Additive {
        word: reduce(
            bit.checked_mul(unmasked_term)
                .and_then(|v| v.checked_add(i128::from(output_mask)))
                .ok_or(Error::Overflow)?,
            value.bits,
        )?,
        mask: output_mask,
        bits: value.bits,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shift {
    pub output: Additive,
    pub low_borrow: bool,
}

/// Unsigned quotient via HIGH(z)-HIGH(r)-borrow(LOW(z)<LOW(r)). Reduction is in
/// n-shift bits, not signed ARS; high-byte indices 128..255 stay positive.
pub(super) fn tr(value: Additive, shift: u8, output_mask: u64) -> Result<Shift, Error> {
    value.clear()?;
    if shift == 0 || shift >= value.bits {
        return Err(Error::Domain);
    }
    let output_bits = value.bits - shift;
    canonical(output_mask, output_bits)?;
    let unit = 1u64 << shift;
    let low_borrow = value.word % unit < value.mask % unit;
    let quotient = i128::from(value.word / unit)
        - i128::from(value.mask / unit)
        - if low_borrow { 1 } else { 0 };
    Ok(Shift {
        output: Additive {
            word: reduce(quotient + i128::from(output_mask), output_bits)?,
            mask: output_mask,
            bits: output_bits,
        },
        low_borrow,
    })
}

/// Independent quarter-ring GapARS proof: bias clear x by 2^(n-2), apply the
/// unsigned masked borrow equation, then subtract the exactly divisible bias.
/// No nearest/ties-to-even rounding and no sign extension of a masked word.
pub(super) fn gap_ars(value: Additive, shift: u8, output_mask: u64) -> Result<Shift, Error> {
    let clear = signed(value.clear()?, value.bits)?;
    if value.bits < 3 || shift == 0 || shift > value.bits - 2 {
        return Err(Error::Domain);
    }
    let quarter = 1i128 << (value.bits - 2);
    if !(-quarter..quarter).contains(&clear) {
        return Err(Error::Domain);
    }
    canonical(output_mask, value.bits)?;
    let unit = 1u64 << shift;
    let biased = value.add(quarter)?;
    let low_borrow = biased.word % unit < biased.mask % unit;
    let unsigned_q = reduce(
        i128::from(biased.word / unit)
            - i128::from(biased.mask / unit)
            - if low_borrow { 1 } else { 0 },
        value.bits - shift,
    )?;
    let result = i128::from(unsigned_q) - quarter / i128::from(unit);
    Ok(Shift {
        output: Additive {
            word: reduce(result + i128::from(output_mask), value.bits)?,
            mask: output_mask,
            bits: value.bits,
        },
        low_borrow,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LookupChoices {
    pub r: u64,
    pub u: u64,
    pub q: u8,
    pub sigma_positive: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Lookup {
    pub opened_r: u64,
    pub opened_correction: u8,
    pub selected: Additive,
    pub output: Additive,
}

/// D19's corrected signed point sum sigma is +/-1. Its correction bit denotes
/// sigma=+1 (NOT sign(T)); q masks that bit. Retain rho=2u-r, with no remask.
pub(super) fn lookup(payload: u64, choices: LookupChoices) -> Result<Lookup, Error> {
    canonical(payload, 50)?;
    canonical(choices.r, 50)?;
    canonical(choices.u, 50)?;
    let sigma = if choices.sigma_positive { 1i128 } else { -1 };
    let arithmetic = Additive::new(reduce(sigma * i128::from(payload), 50)?, choices.r, 50)?;
    let correction = XorBit::new(u8::from(choices.sigma_positive), choices.q)?;
    let selected = select(correction, arithmetic, choices.u)?;
    let rho = reduce(2 * i128::from(choices.u) - i128::from(choices.r), 50)?;
    let output = Additive {
        word: reduce(
            2 * i128::from(selected.word) - i128::from(arithmetic.word),
            50,
        )?,
        mask: rho,
        bits: 50,
    };
    Ok(Lookup {
        opened_r: arithmetic.word,
        opened_correction: correction.word,
        selected,
        output,
    })
}

/// Algebraic modulo-four count correction, independent truth-table form of the
/// two role formulas. Individual GPU complete-expansion counts need NOT be +/-1.
pub(super) fn count_correction(signed_count: i64, own_q: u8) -> Result<u8, Error> {
    if own_q > 1 {
        return Err(Error::Domain);
    }
    Ok(u8::from(matches!(signed_count.rem_euclid(4), 1 | 2)) ^ own_q)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ProductChoices {
    pub mask: u64,
    pub left_coefficients: [u64; 3],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Product {
    pub coefficients: [[u64; 3]; 2],
    pub contributions: [u64; 2],
    pub output: Additive,
}

/// Full ring50 negative-mask algebra. Inputs are the ACTUAL oracle lookup rho
/// encodings, not freshly remasked roots. Sampling independence is NOT proven by
/// these forced synthetic choices. Exactly one public Uhat0*Uhat1 term: role1.
pub(super) fn product(
    left: Additive,
    right: Additive,
    choices: ProductChoices,
) -> Result<Product, Error> {
    if left.bits != 50 || right.bits != 50 {
        return Err(Error::Domain);
    }
    left.clear()?;
    right.clear()?;
    canonical(choices.mask, 50)?;
    for word in choices.left_coefficients {
        canonical(word, 50)?;
    }
    let totals = [
        reduce(-i128::from(left.mask), 50)?,
        reduce(-i128::from(right.mask), 50)?,
        u64::try_from(
            (u128::from(left.mask) * u128::from(right.mask) + u128::from(choices.mask))
                % u128::from(N50),
        )
        .map_err(|_| Error::Overflow)?,
    ];
    let coefficients = [
        choices.left_coefficients,
        std::array::from_fn(|i| (totals[i] + N50 - choices.left_coefficients[i]) % N50),
    ];
    let mut contributions = [0; 2];
    for role in 0..2 {
        let [a, b, c] = coefficients[role];
        let own = u128::from(a) * u128::from(right.word)
            + u128::from(left.word) * u128::from(b)
            + u128::from(c);
        let public_term = if role == 1 {
            u128::from(left.word) * u128::from(right.word)
        } else {
            0
        };
        contributions[role] =
            u64::try_from((own + public_term) % u128::from(N50)).map_err(|_| Error::Overflow)?;
    }
    let output = Additive {
        word: (contributions[0] + contributions[1]) % N50,
        mask: choices.mask,
        bits: 50,
    };
    Ok(Product {
        coefficients,
        contributions,
        output,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ClearTrace {
    pub x40: u64,
    pub affine40: u64,
    pub h: u8,
    pub b: u8,
    pub w16: u64,
    pub v16: u64,
    pub c16: u64,
    pub j0: u64,
    pub j1: u64,
    pub u0: u64,
    pub u1: u64,
    pub p: u64,
    pub e: u64,
}

pub(super) const BOUNDARIES: [i64; 21] = [
    0,
    1,
    127,
    128,
    254,
    255,
    256,
    257,
    32767,
    32768,
    32769,
    34047,
    34048,
    34303,
    34304,
    65534,
    65535,
    65536,
    65537,
    131071,
    (1i64 << 38) - 1,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Signed,
    Unsigned,
    Predicate,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct UseValue {
    pub word: u64,
    pub bits: u8,
    pub fractional: u8,
    pub kind: Kind,
}
/// Expected CLEAR test value/type at each consumed slot. This does not enforce
/// ownership, burn a native ledger or certify a real private input's range.
pub(super) fn use_values(trace: &ClearTrace) -> [UseValue; 18] {
    let signed = |word, bits, fractional| UseValue {
        word,
        bits,
        fractional,
        kind: Kind::Signed,
    };
    let unsigned = |word, bits, fractional| UseValue {
        word,
        bits,
        fractional,
        kind: Kind::Unsigned,
    };
    let predicate = |word: u8| UseValue {
        word: u64::from(word),
        bits: 1,
        fractional: 0,
        kind: Kind::Predicate,
    };
    [
        signed(trace.x40, 50, 12),
        signed(trace.x40, 50, 12),
        signed(trace.x40, 40, 12),
        signed(trace.affine40, 40, 12),
        predicate(trace.h),
        signed(trace.x40, 40, 12),
        predicate(trace.b),
        unsigned(trace.w16, 16, 12),
        unsigned(trace.v16, 16, 12),
        unsigned(trace.c16, 16, 12),
        unsigned(trace.c16, 16, 12),
        unsigned(trace.j1, 8, 0),
        unsigned(trace.j0, 8, 0),
        unsigned(trace.j1, 8, 0),
        signed(trace.u0, 50, 12),
        signed(trace.u1, 50, 12),
        signed(trace.p, 50, 24),
        signed(trace.e, 50, 12),
    ]
}

pub(super) fn fixture_tables() -> Result<Tables, Error> {
    let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../../paper/reproduction-contracts/fusefss/gate-set/fixtures.json"
    ))
    .map_err(|_| Error::Table)?;
    let mut result = [[0; 256]; 2];
    for (out, name) in result.iter_mut().zip(["exp0", "exp1"]) {
        let row = fixture["tables"][name].as_array().ok_or(Error::Table)?;
        if row.len() != 256 {
            return Err(Error::Table);
        }
        for (value, item) in out.iter_mut().zip(row) {
            *value = item.as_u64().ok_or(Error::Table)?;
            if *value > 4096 {
                return Err(Error::Table);
            }
        }
    }
    Ok(result)
}

pub(super) fn lane(input: i64, tables: &Tables) -> Result<ClearTrace, Error> {
    if !(0..INPUT_LIMIT).contains(&input) {
        return Err(Error::Domain);
    }
    let x40 = input as u64 % (1u64 << 40);
    let affine40 = reduce(i128::from(x40) - 65536, 40)?;
    let h = u8::from(signed(affine40, 40)? >= 0);
    let b = h ^ 1;
    let w16 = reduce(i128::from(x40) - 65535, 16)?;
    let v16 = u64::from(b) * w16;
    let c16 = (v16 + 65535) % 65536;
    let j0 = c16 % 256;
    let j1 = c16 / 256; // Deliberately unsigned, not signed ring8 ARS.
    let u0 = tables[0][j0 as usize];
    let u1 = tables[1][j1 as usize];
    if u0 > 4096 || u1 > 4096 {
        return Err(Error::Table);
    }
    let p = u0.checked_mul(u1).ok_or(Error::Overflow)?;
    let e = u64::try_from(signed(p, 50)?.div_euclid(4096)).map_err(|_| Error::Overflow)?;
    Ok(ClearTrace {
        x40,
        affine40,
        h,
        b,
        w16,
        v16,
        c16,
        j0,
        j1,
        u0,
        u1,
        p,
        e,
    })
}
pub(super) fn lanes(inputs: &[i64], tables: &Tables) -> Result<Vec<ClearTrace>, Error> {
    if !(1..=256).contains(&inputs.len()) {
        return Err(Error::Shape);
    }
    // Preflight WHOLE slice, rather than accept a prefix before invalid tail.
    if inputs.iter().any(|x| !(0..INPUT_LIMIT).contains(x)) {
        return Err(Error::Domain);
    }
    inputs.iter().map(|&x| lane(x, tables)).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Choices {
    pub input_mask: u64,
    pub predicate_mask: u8,
    pub clip_mask: u64,
    pub high_mask: u64,
    pub lookup: [LookupChoices; 2],
    pub product: ProductChoices,
    pub output_mask: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MaskedTrace {
    pub expected: ClearTrace,
    pub input: Additive,
    pub x40: Additive,
    pub affine40: Additive,
    pub h: XorBit,
    pub b: XorBit,
    pub w16: Additive,
    pub v16: Additive,
    pub c16: Additive,
    pub j0: Additive,
    pub j1: Shift,
    pub lookup: [Lookup; 2],
    pub product: Product,
    pub final_shift: Shift,
}
pub(super) fn masked_lane(
    input: i64,
    tables: &Tables,
    choices: Choices,
) -> Result<MaskedTrace, Error> {
    let expected = lane(input, tables)?;
    let input = Additive::new(input as u64, choices.input_mask, 50)?;
    let x40 = input.project(40)?;
    let affine40 = x40.add(-65536)?;
    let h = XorBit::new(
        u8::from(signed(affine40.clear()?, 40)? >= 0),
        choices.predicate_mask,
    )?;
    let b = h.not()?;
    let w16 = x40.project(16)?.add(-65535)?;
    let v16 = select(b, w16, choices.clip_mask)?;
    let c16 = v16.add(65535)?;
    let j0 = c16.project(8)?;
    let j1 = tr(c16, 8, choices.high_mask)?;
    let lookup = [
        lookup(tables[0][j0.clear()? as usize], choices.lookup[0])?,
        lookup(tables[1][j1.output.clear()? as usize], choices.lookup[1])?,
    ];
    let product = product(lookup[0].output, lookup[1].output, choices.product)?;
    let final_shift = gap_ars(product.output, 12, choices.output_mask)?;
    Ok(MaskedTrace {
        expected,
        input,
        x40,
        affine40,
        h,
        b,
        w16,
        v16,
        c16,
        j0,
        j1,
        lookup,
        product,
        final_shift,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Stage {
    pub magic: [u8; 8],
    pub phase: u8,
    pub header: usize,
    pub words_per_lane: usize,
}
pub(super) const STAGES: [Stage; 11] = [
    Stage {
        magic: *b"PLLMSA01",
        phase: 1,
        header: 76,
        words_per_lane: 1,
    },
    Stage {
        magic: *b"PLLMSL01",
        phase: 1,
        header: 76,
        words_per_lane: 1,
    },
    Stage {
        magic: *b"PLLMSA01",
        phase: 0,
        header: 76,
        words_per_lane: 1,
    },
    Stage {
        magic: *b"PLLMSA01",
        phase: 1,
        header: 76,
        words_per_lane: 1,
    },
    Stage {
        magic: *b"PLLMLU01",
        phase: 0,
        header: 80,
        words_per_lane: 2,
    },
    Stage {
        magic: *b"PLLMSL01",
        phase: 1,
        header: 76,
        words_per_lane: 1,
    },
    Stage {
        magic: *b"PLLMLU01",
        phase: 0,
        header: 80,
        words_per_lane: 2,
    },
    Stage {
        magic: *b"PLLMSL01",
        phase: 1,
        header: 76,
        words_per_lane: 1,
    },
    Stage {
        magic: *b"PLLMSP01",
        phase: 1,
        header: 76,
        words_per_lane: 1,
    },
    Stage {
        magic: *b"PLLMSA01",
        phase: 0,
        header: 76,
        words_per_lane: 1,
    },
    Stage {
        magic: *b"PLLMSA01",
        phase: 1,
        header: 76,
        words_per_lane: 1,
    },
];
pub(super) const USES: [(&str, u16); 18] = [
    ("enter-X", 0),
    ("project-X40", 0),
    ("affine-A40", 0),
    ("drelu-H", 0),
    ("xor-complement-B", 0),
    ("project-affine-W16", 0),
    ("select-predicate", 0),
    ("select-value", 1),
    ("affine-C16", 0),
    ("low-coordinate", 0),
    ("tr-C16", 0),
    ("high-coordinate-owner", 0),
    ("lookup-E0", 0),
    ("lookup-E1", 0),
    ("product-U0", 0),
    ("product-U1", 1),
    ("gap-product", 0),
    ("opaque-handoff", 0),
];
/// One direction. Stage ordinal is 1..11, not native phase byte.
pub(super) fn frame_bytes(stage: usize, lanes: usize) -> Result<usize, Error> {
    if !(1..=256).contains(&lanes) {
        return Err(Error::Shape);
    }
    let shape = STAGES
        .get(stage.checked_sub(1).ok_or(Error::Domain)?)
        .ok_or(Error::Domain)?;
    shape
        .words_per_lane
        .checked_mul(lanes)
        .and_then(|v| v.checked_mul(8))
        .and_then(|v| v.checked_add(shape.header))
        .ok_or(Error::Overflow)
}
pub(super) fn payload_bits(stage: usize) -> Result<&'static [u8], Error> {
    match stage {
        1 | 3 | 10 => Ok(&[1]),
        2 => Ok(&[16]),
        4 => Ok(&[8]),
        5 | 7 => Ok(&[50, 1]),
        6 | 8 | 9 | 11 => Ok(&[50]),
        _ => Err(Error::Domain),
    }
}

#[cfg(test)]
#[path = "nexp_integer_oracle_tests.rs"]
mod tests;
