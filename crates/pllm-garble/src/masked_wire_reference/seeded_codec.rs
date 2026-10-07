//! Seeded party-local key representation; full expanded keys remain bounded.
use super::{coins::Tape, *};
use crate::compact_dpf::{BitReader, BitSource, DecodeError};

fn generated(party: u8, field: RandomField) -> bool {
    party == 0 || matches!(field, RandomField::BothParties)
}

pub(super) fn bits(lanes: usize, layout: Layout, party: u8) -> Result<usize, Error> {
    let resource = resources_with_layout(lanes, layout)?;
    if party > 1 {
        return Err(Error::Domain);
    }
    let (n, shift) = layout.reduce_bits();
    let roots: usize = [12, 12, shift, layout.sign_bits() - 1, 10, 12]
        .into_iter()
        .map(|width| {
            if width > 7 {
                128
            } else if party == 0 {
                1 << width
            } else {
                0
            }
        })
        .sum();
    let shares = if party == 0 {
        9 * N as usize + 2 * (n - shift) as usize + 2 + 80 + 9 * N as usize
    } else {
        0
    };
    Ok(resource.party_key_bits - lanes * (roots + shares))
}

pub(super) struct Writer {
    pub body: Bits,
    pub valid: bool,
    pub tape: Tape,
    pub party: u8,
}

impl BitSink for Writer {
    fn put(&mut self, value: u128, width: usize) {
        self.body.put(value, width);
    }
    fn random(&mut self, value: u128, width: usize, field: RandomField) {
        if generated(self.party, field) {
            let mask = u128::MAX >> (128 - width);
            // Verify the omitted field really is derived from this party's tape.
            self.valid &= (self.tape.word() & mask) == (value & mask);
        } else {
            self.put(value, width);
        }
    }
}

pub(super) struct Reader<'a> {
    pub body: BitReader<'a>,
    pub tape: Tape,
    pub party: u8,
}

impl BitSource for Reader<'_> {
    fn take(&mut self, width: usize) -> Result<u128, DecodeError> {
        self.body.take(width)
    }
    fn random(&mut self, width: usize, field: RandomField) -> Result<u128, DecodeError> {
        if !(1..=128).contains(&width) {
            return Err(DecodeError);
        }
        if generated(self.party, field) {
            Ok(self.tape.word() & (u128::MAX >> (128 - width)))
        } else {
            self.take(width)
        }
    }
}
