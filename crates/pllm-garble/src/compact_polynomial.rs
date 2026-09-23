//! In-process one-use protected Q7 Chebyshev polynomial reference.
//!
//! This evaluates the committed fixed-point coefficients with exact signed
//! ties-to-even arithmetic inside one half-gates circuit. It is deliberately
//! not transportable, compiler-bound, or a claim about Compact's MPC costs.

use pllm_core::{compact::COMPACT_MAX_PIECES, CompactQ7Piece, CompactQ7Profile};

use crate::{
    boolean::{
        BooleanCircuitBuilder, BooleanCircuitClient, BooleanCircuitDecoder, BooleanCircuitInputs,
        BooleanCircuitOutputs, BooleanCircuitProgram, BooleanWire,
    },
    compact_coordinate::piece_coordinate,
};

const INPUT_BITS: usize = 9;
const OUTPUT_BITS: usize = 13;
const INDEX_BITS: usize = 3;
const FRACTIONAL_BITS: usize = 20;
const MAX_COEFFICIENT_MAGNITUDE: u64 = 1 << 27;
const MAX_CIPHERTEXT_BYTES: u64 = 8 * 1024 * 1024;

pub struct CompactQ7PolynomialClient {
    profile_digest: [u8; 32],
    piece_count: usize,
    client: BooleanCircuitClient,
}
pub struct CompactQ7PolynomialInputs(BooleanCircuitInputs);
pub struct CompactQ7PolynomialOutputs(BooleanCircuitOutputs);
pub struct CompactQ7PolynomialDecoder {
    profile_digest: [u8; 32],
    piece_count: usize,
    decoder: BooleanCircuitDecoder,
}
pub struct CompactQ7PolynomialProgram {
    profile_digest: [u8; 32],
    program: BooleanCircuitProgram,
}

fn polynomial_for_piece(
    builder: &mut BooleanCircuitBuilder,
    input: &[BooleanWire],
    piece: &CompactQ7Piece,
) -> Result<Vec<BooleanWire>, String> {
    let t = piece_coordinate(builder, input, piece.lower(), piece.upper())?;
    let magnitude = builder.absolute_signed(&t)?;
    let square = builder.multiply_unsigned(&magnitude, &magnitude)?;
    let square_q20 = builder.round_shift_ties_even(&square, FRACTIONAL_BITS, 25)?;
    let zero = builder.constant(false)?;
    let mut twice_square = vec![zero];
    twice_square.extend_from_slice(&square_q20[..22]);
    let one_q20 = builder.constant_word(1 << FRACTIONAL_BITS, 23)?;
    let chebyshev_second = builder.subtract_unsigned(&twice_square, &one_q20)?.0;

    let [constant, linear, quadratic] = piece.coefficients_q20();
    let linear_term =
        builder.multiply_signed_by_public_round_ties_even(&t, linear, FRACTIONAL_BITS, 32)?;
    let quadratic_term = builder.multiply_signed_by_public_round_ties_even(
        &chebyshev_second,
        quadratic,
        FRACTIONAL_BITS,
        32,
    )?;
    let both = builder.add_unsigned(&linear_term, &quadratic_term)?.0;
    let constant = i32::try_from(constant)
        .map_err(|_| "Compact Q7 constant coefficient exceeds signed 32-bit")?;
    let constant_word = builder.constant_word(u128::from(constant as u32), 32)?;
    let sum = builder.add_unsigned(&both, &constant_word)?.0;
    let absolute = builder.absolute_signed(&sum)?;
    let rounded = builder.round_shift_ties_even(&absolute, FRACTIONAL_BITS, OUTPUT_BITS)?;
    builder.conditional_negate(&rounded, sum[31])
}

/// Fit outputs are preflighted before generating any material. Selection,
/// normalized coordinate and public-coefficient polynomial evaluation share
/// one circuit and leave no plaintext intermediate at the evaluator.
pub fn prepare_compact_q7_polynomial(
    profile: &CompactQ7Profile,
) -> Result<(CompactQ7PolynomialClient, CompactQ7PolynomialProgram), String> {
    let pieces = profile.pieces();
    if pieces.is_empty() || pieces.len() > usize::from(COMPACT_MAX_PIECES) {
        return Err("Compact Q7 polynomial piece count is invalid".into());
    }
    for piece in pieces {
        if piece
            .coefficients_q20()
            .iter()
            .any(|coefficient| coefficient.unsigned_abs() > MAX_COEFFICIENT_MAGNITUDE)
        {
            return Err("Compact Q7 polynomial coefficient exceeds circuit bound".into());
        }
    }
    for value in -128..=128 {
        let output = profile.evaluate(value).map_err(|error| error.to_string())?;
        if !(-128..=128).contains(&output) {
            return Err("Compact Q7 polynomial output is outside its encoded domain".into());
        }
    }

    let mut builder = BooleanCircuitBuilder::new()?;
    let input = builder.input_word(INPUT_BITS)?;
    let last = pieces.len() - 1;
    let mut selected = polynomial_for_piece(&mut builder, &input, &pieces[last])?;
    let mut selected_index = builder.constant_word(last as u128, INDEX_BITS)?;
    for (index, piece) in pieces[..last].iter().enumerate().rev() {
        let candidate = polynomial_for_piece(&mut builder, &input, piece)?;
        let upper = u128::from((piece.upper() + 128) as u16);
        let (less, equal) = builder.compare_unsigned_to_constant(&input, upper)?;
        let at_or_below = builder.xor(less, equal)?;
        selected = builder.select_word(at_or_below, &selected, &candidate)?;
        let candidate_index = builder.constant_word(index as u128, INDEX_BITS)?;
        selected_index = builder.select_word(at_or_below, &selected_index, &candidate_index)?;
    }
    selected.extend(selected_index);
    let (client, program) = builder.finish(&selected)?;
    if program.evaluator_ciphertext_bytes()? > MAX_CIPHERTEXT_BYTES {
        return Err("Compact Q7 polynomial exceeds its circuit ciphertext limit".into());
    }
    let profile_digest = profile.digest();
    Ok((
        CompactQ7PolynomialClient {
            profile_digest,
            piece_count: pieces.len(),
            client,
        },
        CompactQ7PolynomialProgram {
            profile_digest,
            program,
        },
    ))
}

impl CompactQ7PolynomialClient {
    pub fn encode(
        self,
        value: i16,
    ) -> Result<(CompactQ7PolynomialInputs, CompactQ7PolynomialDecoder), String> {
        if !(-128..=128).contains(&value) {
            return Err("Compact Q7 polynomial input is outside [-128, 128]".into());
        }
        let offset = (value + 128) as u16;
        let bits = (0..INPUT_BITS)
            .map(|index| offset & (1_u16 << index) != 0)
            .collect::<Vec<_>>();
        let (inputs, decoder) = self.client.encode(&bits)?;
        Ok((
            CompactQ7PolynomialInputs(inputs),
            CompactQ7PolynomialDecoder {
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

impl CompactQ7PolynomialProgram {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    pub fn evaluator_ciphertext_bytes(&self) -> Result<u64, String> {
        self.program.evaluator_ciphertext_bytes()
    }

    pub fn evaluate(
        self,
        inputs: CompactQ7PolynomialInputs,
    ) -> Result<CompactQ7PolynomialOutputs, String> {
        self.program
            .evaluate(inputs.0)
            .map(CompactQ7PolynomialOutputs)
    }
}

impl CompactQ7PolynomialDecoder {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    /// Returns a trusted-client-only selected piece and encoded Q7 output.
    pub fn decode(self, outputs: CompactQ7PolynomialOutputs) -> Result<(usize, i16), String> {
        let bits = self.decoder.decode(outputs.0)?;
        if bits.len() != OUTPUT_BITS + INDEX_BITS {
            return Err("Compact Q7 polynomial output width is invalid".into());
        }
        let unsigned = bits[..OUTPUT_BITS]
            .iter()
            .enumerate()
            .fold(0_i16, |value, (bit, flag)| {
                value | (i16::from(*flag) << bit)
            });
        let output = if unsigned & (1 << (OUTPUT_BITS - 1)) != 0 {
            unsigned - (1 << OUTPUT_BITS)
        } else {
            unsigned
        };
        let index = bits[OUTPUT_BITS..]
            .iter()
            .enumerate()
            .fold(0_usize, |value, (bit, flag)| {
                value | (usize::from(*flag) << bit)
            });
        if index >= self.piece_count || !(-128..=128).contains(&output) {
            return Err("Compact Q7 polynomial result is outside its declared domain".into());
        }
        Ok((index, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_core::fit_compact_silu_q7;

    #[test]
    fn private_polynomial_matches_the_exact_locked_numeric_reference() {
        let mut counts = [0; 257];
        for count in &mut counts[28..49] {
            *count = 1_000_000;
        }
        for profile in [
            fit_compact_silu_q7(&[0; 257], 1).unwrap(),
            fit_compact_silu_q7(&[0; 257], 4).unwrap(),
            fit_compact_silu_q7(&counts, 8).unwrap(),
        ] {
            let mut values = vec![-128, -127, -1, 0, 1, 127, 128];
            for piece in profile.pieces() {
                values.extend([
                    piece.lower(),
                    (piece.lower() + piece.upper()) / 2,
                    piece.upper(),
                ]);
            }
            values.sort_unstable();
            values.dedup();
            for value in values {
                let (client, program) = prepare_compact_q7_polynomial(&profile).unwrap();
                assert_eq!(client.profile_digest(), profile.digest());
                assert_eq!(program.profile_digest(), profile.digest());
                assert!(program.evaluator_ciphertext_bytes().unwrap() <= MAX_CIPHERTEXT_BYTES);
                let (inputs, decoder) = client.encode(value).unwrap();
                assert_eq!(decoder.profile_digest(), profile.digest());
                let (index, actual) = decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
                assert!(
                    (profile.pieces()[index].lower()..=profile.pieces()[index].upper())
                        .contains(&value)
                );
                assert_eq!(
                    actual,
                    profile.evaluate(value).unwrap(),
                    "Q7 input {value} piece {index}"
                );
            }
        }
    }

    #[test]
    fn material_and_inputs_fail_closed() {
        let profile = fit_compact_silu_q7(&[0; 257], 2).unwrap();
        let (client_a, program_a) = prepare_compact_q7_polynomial(&profile).unwrap();
        let (client_b, program_b) = prepare_compact_q7_polynomial(&profile).unwrap();
        let (inputs_a, _) = client_a.encode(-128).unwrap();
        let (inputs_b, _) = client_b.encode(128).unwrap();
        assert!(program_a.evaluate(inputs_b).is_err());
        assert!(program_b.evaluate(inputs_a).is_err());
        let (client, _) = prepare_compact_q7_polynomial(&profile).unwrap();
        assert!(client.encode(-129).is_err());
        let (client, _) = prepare_compact_q7_polynomial(&profile).unwrap();
        assert!(client.encode(129).is_err());
    }
}
