//! One-use protected normalized coordinate for an immutable Compact Q7 fit.
//!
//! The circuit selects a piece and computes its signed Q20 Chebyshev
//! coordinate with exact public-divisor ties-to-even division. Both the piece
//! index and coordinate remain opaque labels until the trusted client decodes
//! them. The piecewise polynomial itself is not evaluated by this circuit.

use pllm_core::{compact::COMPACT_MAX_PIECES, CompactQ7Profile};

use crate::boolean::{
    BooleanCircuitBuilder, BooleanCircuitClient, BooleanCircuitDecoder, BooleanCircuitInputs,
    BooleanCircuitOutputs, BooleanCircuitProgram, BooleanWire,
};

const INPUT_BITS: usize = 9;
const CENTERED_BITS: usize = 11;
const FRACTIONAL_BITS: usize = 20;
const COORDINATE_BITS: usize = 22;
const INDEX_BITS: usize = 3;
const MAX_CIPHERTEXT_BYTES: u64 = 1_048_576;

pub struct CompactQ7CoordinateClient {
    profile_digest: [u8; 32],
    piece_count: usize,
    client: BooleanCircuitClient,
}
pub struct CompactQ7CoordinateInputs(BooleanCircuitInputs);
pub struct CompactQ7CoordinateOutputs(BooleanCircuitOutputs);
pub struct CompactQ7CoordinateDecoder {
    profile_digest: [u8; 32],
    piece_count: usize,
    decoder: BooleanCircuitDecoder,
}
pub struct CompactQ7CoordinateProgram {
    profile_digest: [u8; 32],
    program: BooleanCircuitProgram,
}

pub(crate) fn piece_coordinate(
    builder: &mut BooleanCircuitBuilder,
    input: &[BooleanWire],
    lower: i16,
    upper: i16,
) -> Result<Vec<BooleanWire>, String> {
    let zero = builder.constant(false)?;
    let mut doubled = Vec::new();
    doubled
        .try_reserve_exact(CENTERED_BITS)
        .map_err(|_| "Compact Q7 coordinate allocation failed")?;
    doubled.push(zero);
    doubled.extend_from_slice(input);
    doubled.push(zero);
    let constant = u128::from((256 + lower + upper) as u16);
    let offset = builder.constant_word(constant, CENTERED_BITS)?;
    let centered = builder.subtract_unsigned(&doubled, &offset)?.0;
    let mut scaled = vec![zero; FRACTIONAL_BITS];
    scaled.extend_from_slice(&centered);
    let span = (upper - lower) as u16;
    let divided = builder.divide_signed_by_public_ties_even(&scaled, span)?;
    Ok(divided[..COORDINATE_BITS].to_vec())
}

/// The same fixed circuit topology runs for all private Q7 values. Public
/// interval boundaries and the committed profile digest remain inspectable.
pub fn prepare_compact_q7_coordinate(
    profile: &CompactQ7Profile,
) -> Result<(CompactQ7CoordinateClient, CompactQ7CoordinateProgram), String> {
    let pieces = profile.pieces();
    if pieces.is_empty() || pieces.len() > usize::from(COMPACT_MAX_PIECES) {
        return Err("Compact Q7 coordinate piece count is invalid".into());
    }
    let mut builder = BooleanCircuitBuilder::new()?;
    let input = builder.input_word(INPUT_BITS)?;
    let last = pieces.len() - 1;
    let mut selected = piece_coordinate(
        &mut builder,
        &input,
        pieces[last].lower(),
        pieces[last].upper(),
    )?;
    let mut selected_index = builder.constant_word(last as u128, INDEX_BITS)?;
    for (index, piece) in pieces[..last].iter().enumerate().rev() {
        let candidate = piece_coordinate(&mut builder, &input, piece.lower(), piece.upper())?;
        let upper = u128::from((piece.upper() + 128) as u16);
        let (less, equal) = builder.compare_unsigned_to_constant(&input, upper)?;
        let at_or_below = builder.xor(less, equal)?;
        selected = builder.select_word(at_or_below, &selected, &candidate)?;
        let candidate_index = builder.constant_word(index as u128, INDEX_BITS)?;
        selected_index = builder.select_word(at_or_below, &selected_index, &candidate_index)?;
    }
    let mut outputs = selected;
    outputs.extend(selected_index);
    let (client, program) = builder.finish(&outputs)?;
    if program.evaluator_ciphertext_bytes()? > MAX_CIPHERTEXT_BYTES {
        return Err("Compact Q7 coordinate circuit exceeds its ciphertext limit".into());
    }
    let profile_digest = profile.digest();
    Ok((
        CompactQ7CoordinateClient {
            profile_digest,
            piece_count: pieces.len(),
            client,
        },
        CompactQ7CoordinateProgram {
            profile_digest,
            program,
        },
    ))
}

impl CompactQ7CoordinateClient {
    pub fn encode(
        self,
        value: i16,
    ) -> Result<(CompactQ7CoordinateInputs, CompactQ7CoordinateDecoder), String> {
        if !(-128..=128).contains(&value) {
            return Err("Compact Q7 coordinate input is outside [-128, 128]".into());
        }
        let offset = (value + 128) as u16;
        let bits = (0..INPUT_BITS)
            .map(|index| offset & (1_u16 << index) != 0)
            .collect::<Vec<_>>();
        let (inputs, decoder) = self.client.encode(&bits)?;
        Ok((
            CompactQ7CoordinateInputs(inputs),
            CompactQ7CoordinateDecoder {
                profile_digest: self.profile_digest,
                piece_count: self.piece_count,
                decoder,
            },
        ))
    }

    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }
}

impl CompactQ7CoordinateProgram {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    pub fn evaluator_ciphertext_bytes(&self) -> Result<u64, String> {
        self.program.evaluator_ciphertext_bytes()
    }

    pub fn evaluate(
        self,
        inputs: CompactQ7CoordinateInputs,
    ) -> Result<CompactQ7CoordinateOutputs, String> {
        self.program
            .evaluate(inputs.0)
            .map(CompactQ7CoordinateOutputs)
    }
}

impl CompactQ7CoordinateDecoder {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    pub fn decode(self, outputs: CompactQ7CoordinateOutputs) -> Result<(usize, i32), String> {
        let bits = self.decoder.decode(outputs.0)?;
        if bits.len() != COORDINATE_BITS + INDEX_BITS {
            return Err("Compact Q7 coordinate output width is invalid".into());
        }
        let unsigned = bits[..COORDINATE_BITS]
            .iter()
            .enumerate()
            .fold(0_i32, |value, (bit, flag)| {
                value | (i32::from(*flag) << bit)
            });
        let coordinate = if unsigned & (1 << (COORDINATE_BITS - 1)) != 0 {
            unsigned - (1 << COORDINATE_BITS)
        } else {
            unsigned
        };
        let index = bits[COORDINATE_BITS..]
            .iter()
            .enumerate()
            .fold(0_usize, |value, (bit, flag)| {
                value | (usize::from(*flag) << bit)
            });
        if index >= self.piece_count || coordinate.unsigned_abs() > (1 << FRACTIONAL_BITS) {
            return Err("Compact Q7 coordinate result is invalid".into());
        }
        Ok((index, coordinate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_core::fit_compact_silu_q7;

    fn rounded_divide(numerator: i64, denominator: i64) -> i32 {
        let absolute = numerator.unsigned_abs();
        let divisor = denominator as u64;
        let quotient = absolute / divisor;
        let remainder = absolute % divisor;
        let rounded = quotient
            + u64::from(remainder * 2 > divisor || (remainder * 2 == divisor && quotient % 2 != 0));
        if numerator < 0 {
            -(rounded as i32)
        } else {
            rounded as i32
        }
    }

    #[test]
    fn private_coordinate_matches_the_committed_piecewise_normalization() {
        let mut counts = [0; 257];
        for count in &mut counts[28..49] {
            *count = 1_000_000;
        }
        for profile in [
            fit_compact_silu_q7(&[0; 257], 4).unwrap(),
            fit_compact_silu_q7(&counts, 8).unwrap(),
        ] {
            let mut values = vec![-128, -127, -1, 0, 1, 127, 128];
            for piece in profile.pieces() {
                values.extend([piece.lower(), piece.upper()]);
            }
            values.sort_unstable();
            values.dedup();
            for value in values {
                let (client, program) = prepare_compact_q7_coordinate(&profile).unwrap();
                assert_eq!(client.profile_digest(), profile.digest());
                assert_eq!(program.profile_digest(), profile.digest());
                assert!(program.evaluator_ciphertext_bytes().unwrap() <= MAX_CIPHERTEXT_BYTES);
                let (inputs, decoder) = client.encode(value).unwrap();
                assert_eq!(decoder.profile_digest(), profile.digest());
                let (index, coordinate) =
                    decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
                let piece = &profile.pieces()[index];
                assert!((piece.lower()..=piece.upper()).contains(&value));
                let centered = i64::from(2 * value - piece.lower() - piece.upper());
                let expected = rounded_divide(
                    centered * (1 << FRACTIONAL_BITS),
                    i64::from(piece.upper() - piece.lower()),
                );
                assert_eq!(coordinate, expected, "{value} in piece {index}");
            }
        }
    }

    #[test]
    fn invalid_input_and_cross_circuit_material_fail_closed() {
        let profile = fit_compact_silu_q7(&[0; 257], 2).unwrap();
        let (client_a, program_a) = prepare_compact_q7_coordinate(&profile).unwrap();
        let (client_b, program_b) = prepare_compact_q7_coordinate(&profile).unwrap();
        let (inputs_a, _) = client_a.encode(-128).unwrap();
        let (inputs_b, _) = client_b.encode(128).unwrap();
        assert!(program_b.evaluate(inputs_a).is_err());
        assert!(program_a.evaluate(inputs_b).is_err());
        let (client, _) = prepare_compact_q7_coordinate(&profile).unwrap();
        assert!(client.encode(-129).is_err());
        let (client, _) = prepare_compact_q7_coordinate(&profile).unwrap();
        assert!(client.encode(129).is_err());
    }
}
