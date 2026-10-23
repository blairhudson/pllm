//! Selected public nExp literals and fail-closed arithmetic-floor validation.
//!
//! D29 mathematical tables: E0[j]=floor(4096*exp(-j/4096)),
//! E1[j]=floor(4096*exp(-j/16)), j=0..255, ordered unsigned byte coordinates.
//! Values are extracted unchanged from the existing frozen equation fixture;
//! this is not author compiler/libm identity, source admission, a numeric profile,
//! or permission to query private coordinates. B owns immutable charged table
//! ownership and source/invocation binding. Only arithmetic is shared here.
//! Mount `mod exp_floor; mod fixed_exp_tables;` in lib.rs before integration.

use crate::exp_floor::{self, Argument, Workspace};
use std::mem::size_of;

#[path = "fixed_exp_table_values.rs"]
mod values;

pub(crate) const ENTRY_COUNT: usize = 256;
pub(crate) const PAYLOAD_SCALE: u64 = 4096;
pub(crate) const SOURCE_COMMIT: &str = "aa0f2f860266fb79901405ef5e9965b8c9fe2e88";
pub(crate) const PAPER_SHA256: &str =
    "f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Table {
    LowByte,
    HighByte,
}

/// Formula/byte declarations, not authority for any source or private value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Definition {
    pub(crate) entries: usize,
    pub(crate) argument_denominator_bits: u8,
    pub(crate) payload_scale: u64,
    pub(crate) coordinate_bits: u8,
    pub(crate) coordinate_fractional_bits: u8,
    pub(crate) payload_bits: u8,
    pub(crate) payload_fractional_bits: u8,
    pub(crate) sha256_u64le: &'static str,
}

impl Table {
    pub(crate) fn definition(self) -> Definition {
        Definition {
            entries: ENTRY_COUNT,
            argument_denominator_bits: match self {
                Self::LowByte => 12,
                Self::HighByte => 4,
            },
            payload_scale: PAYLOAD_SCALE,
            coordinate_bits: 8,
            coordinate_fractional_bits: 0,
            payload_bits: 50,
            payload_fractional_bits: 12,
            sha256_u64le: match self {
                Self::LowByte => "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b",
                Self::HighByte => {
                    "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9"
                }
            },
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Error {
    Floor(exp_floor::Error),
    Literal {
        table: Table,
        index: usize,
        selected: u64,
        certified: u64,
    },
}

/// Arithmetic-only token for these exact static bytes, with private constructor.
/// No hash/copy/material issuance may precede this check in the owning layer.
#[derive(Debug)]
pub(crate) struct Verified {
    _all_selected_floors_checked: (),
}

impl Verified {
    pub(crate) fn values(&self, table: Table) -> &'static [u64; ENTRY_COUNT] {
        match table {
            Table::LowByte => &values::EXP0,
            Table::HighByte => &values::EXP1,
        }
    }
}

#[allow(dead_code)]
struct ValidationSlots {
    definition: Definition,
    table: Table,
    index: usize,
    selected: u64,
    argument_result: Result<Argument, exp_floor::Error>,
    returned: Result<Verified, Error>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Quote {
    pub(crate) proof: exp_floor::Quote,
    pub(crate) validation_slots_bytes: usize,
    /// Static literal storage is preexisting public backing, not newly allocated.
    /// Owning layer still accounts actual table copies, hashes and retained pins.
    pub(crate) selected_literal_bytes: usize,
}

pub(crate) fn quote() -> Result<Quote, exp_floor::Error> {
    Ok(Quote {
        proof: exp_floor::quote(
            ENTRY_COUNT
                .checked_mul(2)
                .ok_or(exp_floor::Error::Overflow)?,
        )?,
        validation_slots_bytes: size_of::<ValidationSlots>(),
        selected_literal_bytes: size_of::<[u64; ENTRY_COUNT]>()
            .checked_mul(2)
            .ok_or(exp_floor::Error::Overflow)?,
    })
}

fn check(table: Table, selected: &[u64; ENTRY_COUNT], work: &mut Workspace) -> Result<(), Error> {
    let definition = table.definition();
    for (index, &literal) in selected.iter().enumerate() {
        let numerator =
            u32::try_from(index).map_err(|_| Error::Floor(exp_floor::Error::Overflow))?;
        let argument =
            Argument::new(numerator, definition.argument_denominator_bits).map_err(Error::Floor)?;
        let certificate = work
            .certify(argument, definition.payload_scale)
            .map_err(Error::Floor)?;
        if certificate.floor() != literal {
            return Err(Error::Literal {
                table,
                index,
                selected: literal,
                certified: certificate.floor(),
            });
        }
    }
    Ok(())
}

/// Public offline preparation gate, not an online secret-dependent routine.
/// Caller admits quote/return/input overlap before comparison or later hashing.
/// No cache: every successful construction checks all 512 selected literals.
pub(crate) fn verify(work: &mut Workspace) -> Result<Verified, Error> {
    check(Table::LowByte, &values::EXP0, work)?;
    check(Table::HighByte, &values::EXP1, work)?;
    Ok(Verified {
        _all_selected_floors_checked: (),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn all_selected_floors_before_any_digest_and_exact_definitions() {
        let mut work = Workspace::default();
        let checked = verify(&mut work).unwrap();
        for (table, denominator) in [(Table::LowByte, 12), (Table::HighByte, 4)] {
            let d = table.definition();
            assert_eq!(
                (d.entries, d.argument_denominator_bits, d.payload_scale),
                (256, denominator, 4096)
            );
            assert_eq!((d.coordinate_bits, d.coordinate_fractional_bits), (8, 0));
            assert_eq!((d.payload_bits, d.payload_fractional_bits), (50, 12));
            let mut hash = Sha256::new();
            for value in checked.values(table) {
                hash.update(value.to_le_bytes());
            }
            let actual = format!("{:x}", hash.finalize());
            assert_eq!(actual, d.sha256_u64le);
        }
        assert_eq!(checked.values(Table::LowByte)[0], 4096);
        assert_eq!(checked.values(Table::LowByte)[255], 3848);
        assert_eq!(&checked.values(Table::HighByte)[133..=134], &[1, 0]);
        assert_eq!(checked.values(Table::HighByte)[255], 0);
    }

    #[test]
    fn changed_literal_rejects_including_zero_and_zero_tail() {
        for (table, original) in [
            (Table::LowByte, values::EXP0),
            (Table::HighByte, values::EXP1),
        ] {
            for index in [0, 1, 133, 134, 255] {
                let mut changed = original;
                changed[index] += 1;
                assert!(matches!(check(table, &changed, &mut Workspace::default()),
                                 Err(Error::Literal { index: bad, .. }) if bad == index));
            }
        }
        let q = quote().unwrap();
        assert_eq!(q.proof.entries, 512);
        assert_eq!(q.selected_literal_bytes, 4096);
    }
}
