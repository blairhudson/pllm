//! Immutable, method-neutral Q12 activation correction tables.
//!
//! Mathematical target: `paper/reproduction-contracts/shared-profile-resolution.md`
//! section 2.1 (SIGMA Fig. 6/App. F). With `x = i/64`, entries are
//! `floor(4096*x*(1-erf(x/sqrt(2)))/2)` for GELU (256 entries), and
//! `floor(4096*x/(1+exp(x)))` for SiLU (1024 entries).
//!
//! GELU reuses the existing frozen `fusefss/gelu-table.json` integer fixture;
//! SiLU is extracted unchanged from the former FuseFSS composition constants,
//! certified by the independent rational bounds in `fusefss/gate-set/reference.py`.
//! Both match the paper-equation hashes below; the independent project Decimal
//! oracle agreed at 80 and 96 digits. This is not author compiler/libm byte
//! identity. No upstream code, online floating generation, execution machinery,
//! profile identity, correlated material, or range evidence is shared here.

#[path = "fixed_activation_table_values.rs"]
mod values;

/// SHA-256 over successive unsigned u64 little-endian GELU entries.
#[cfg(test)]
pub(crate) const GELU_Q12_SHA256_U64LE: &str =
    "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b";
/// SHA-256 over successive unsigned u64 little-endian SiLU entries.
#[cfg(test)]
pub(crate) const SILU_Q12_SHA256_U64LE: &str =
    "ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f";

pub(crate) fn gelu_q12() -> &'static [u64] {
    &values::GELU
}

pub(crate) fn silu_q12() -> &'static [u64] {
    &values::SILU
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    // Independent frozen equation fixtures, not generated from the Rust arrays.
    const GELU_FIXTURE: &[u8] =
        include_bytes!("../../../paper/reproduction-contracts/fusefss/gelu-table.json");
    const SILU_FIXTURE: &[u8] =
        include_bytes!("../../../paper/reproduction-contracts/fusefss/gate-set/fixtures.json");

    fn hex(bytes: &[u8]) -> String {
        use std::fmt::Write;
        bytes.iter().fold(String::new(), |mut output, byte| {
            write!(output, "{byte:02x}").unwrap();
            output
        })
    }

    fn check(values: &[u64], expected: &Value, expected_hash: &str) {
        let expected = expected.as_array().unwrap();
        assert_eq!(values.len(), expected.len());
        let mut digest = Sha256::new();
        for (index, (&actual, frozen)) in values.iter().zip(expected).enumerate() {
            assert_eq!(Some(actual), frozen.as_u64(), "entry {index}");
            digest.update(actual.to_le_bytes());
        }
        assert_eq!(hex(&digest.finalize()), expected_hash);
    }

    #[test]
    fn gelu_matches_every_independently_frozen_equation_entry_and_hash() {
        assert_eq!(
            hex(&Sha256::digest(GELU_FIXTURE)),
            "4d91281c274275d8b17f6711eb3f454ad25865a197570115fdede8b2db7db69b"
        );
        let fixture: Value = serde_json::from_slice(GELU_FIXTURE).unwrap();
        assert_eq!(gelu_q12().len(), 256);
        assert_eq!(fixture["sha256_u64le"], GELU_Q12_SHA256_U64LE);
        check(gelu_q12(), &fixture["values"], GELU_Q12_SHA256_U64LE);
    }

    #[test]
    fn silu_matches_every_independently_certified_frozen_entry_and_hash() {
        assert_eq!(
            hex(&Sha256::digest(SILU_FIXTURE)),
            "ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87"
        );
        let fixture: Value = serde_json::from_slice(SILU_FIXTURE).unwrap();
        assert_eq!(silu_q12().len(), 1024);
        assert_eq!(fixture["sha256_u64le"]["silu"], SILU_Q12_SHA256_U64LE);
        check(
            silu_q12(),
            &fixture["tables"]["silu"],
            SILU_Q12_SHA256_U64LE,
        );
    }
}
