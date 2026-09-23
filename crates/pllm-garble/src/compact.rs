//! Bounded, in-process half-gates reference for selecting a Compact Q7 piece.
//!
//! Only the trusted client holds input encodings and the one-use output decoder.
//! The evaluator receives input labels and an opaque, consuming circuit; it
//! cannot decode the interval. This is not a polynomial evaluator, wire format,
//! whole-decoder component, or independently reviewed cryptographic protocol.

use pllm_core::{compact::COMPACT_MAX_PIECES, CompactQ7Profile};

use crate::boolean::{
    BooleanCircuitBuilder, BooleanCircuitClient, BooleanCircuitDecoder, BooleanCircuitInputs,
    BooleanCircuitOutputs, BooleanCircuitProgram,
};

const INPUT_BITS: usize = 9; // offset signed Q7 [-128,128] into [0,256]
const OUTPUT_BITS: usize = 3; // up to eight pieces
const MAX_CIPHERTEXT_BYTES: u64 = 8_192;

pub struct CompactQ7SelectionClient {
    profile_digest: [u8; 32],
    piece_count: usize,
    client: BooleanCircuitClient,
}

pub struct CompactQ7SelectionInputs(BooleanCircuitInputs);
pub struct CompactQ7SelectionOutputs(BooleanCircuitOutputs);

pub struct CompactQ7SelectionDecoder {
    profile_digest: [u8; 32],
    piece_count: usize,
    decoder: BooleanCircuitDecoder,
}

pub struct CompactQ7SelectionProgram {
    profile_digest: [u8; 32],
    program: BooleanCircuitProgram,
}

/// Prepare one circuit from an immutable, public profile. Every input has the
/// same gate topology; private inputs affect only their selected input labels.
pub fn prepare_compact_q7_selection(
    profile: &CompactQ7Profile,
) -> Result<(CompactQ7SelectionClient, CompactQ7SelectionProgram), String> {
    let pieces = profile.pieces();
    if pieces.is_empty() || pieces.len() > usize::from(COMPACT_MAX_PIECES) {
        return Err("Compact Q7 selection piece count is invalid".into());
    }
    let mut builder = BooleanCircuitBuilder::new()?;
    let input = builder.input_word(INPUT_BITS)?;
    let last_index = pieces.len() - 1;
    let mut selected = builder.constant_word(last_index as u128, OUTPUT_BITS)?;
    // Later pieces are the default. Walk the public boundaries backwards so
    // the lowest matching bound wins, without branching on the private value.
    for (index, piece) in pieces[..last_index].iter().enumerate().rev() {
        let inclusive_upper = u128::from((piece.upper() + 128) as u16);
        let (less, equal) = builder.compare_unsigned_to_constant(&input, inclusive_upper)?;
        let at_or_below = builder.xor(less, equal)?;
        let candidate = builder.constant_word(index as u128, OUTPUT_BITS)?;
        selected = builder.select_word(at_or_below, &selected, &candidate)?;
    }
    let (client, program) = builder.finish(&selected)?;
    if program.evaluator_ciphertext_bytes()? > MAX_CIPHERTEXT_BYTES {
        return Err("Compact Q7 selection circuit exceeds its ciphertext limit".into());
    }
    let profile_digest = profile.digest();
    Ok((
        CompactQ7SelectionClient {
            profile_digest,
            piece_count: pieces.len(),
            client,
        },
        CompactQ7SelectionProgram {
            profile_digest,
            program,
        },
    ))
}

impl CompactQ7SelectionClient {
    /// Consumes client material even when an out-of-domain input is rejected.
    pub fn encode(
        self,
        value: i16,
    ) -> Result<(CompactQ7SelectionInputs, CompactQ7SelectionDecoder), String> {
        if !(-128..=128).contains(&value) {
            return Err("Compact Q7 selection input is outside [-128, 128]".into());
        }
        let offset = (value + 128) as u16;
        let bits = (0..INPUT_BITS)
            .map(|index| offset & (1_u16 << index) != 0)
            .collect::<Vec<_>>();
        let (inputs, decoder) = self.client.encode(&bits)?;
        Ok((
            CompactQ7SelectionInputs(inputs),
            CompactQ7SelectionDecoder {
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

impl CompactQ7SelectionProgram {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    pub fn evaluator_ciphertext_bytes(&self) -> Result<u64, String> {
        self.program.evaluator_ciphertext_bytes()
    }

    /// The program and its one-use gate material are consumed on invocation.
    pub fn evaluate(
        self,
        inputs: CompactQ7SelectionInputs,
    ) -> Result<CompactQ7SelectionOutputs, String> {
        self.program
            .evaluate(inputs.0)
            .map(CompactQ7SelectionOutputs)
    }
}

impl CompactQ7SelectionDecoder {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    /// Only the trusted client decodes a piece index. The evaluator sees labels.
    pub fn decode(self, outputs: CompactQ7SelectionOutputs) -> Result<usize, String> {
        let bits = self.decoder.decode(outputs.0)?;
        if bits.len() != OUTPUT_BITS {
            return Err("Compact Q7 selection output width is invalid".into());
        }
        let index = bits.iter().enumerate().fold(0, |value, (shift, bit)| {
            value | (usize::from(*bit) << shift)
        });
        if index >= self.piece_count {
            return Err("Compact Q7 selection returned an invalid piece".into());
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_core::fit_compact_silu_q7;

    #[test]
    fn selected_piece_matches_public_profile_across_the_full_q7_domain() {
        let mut skewed = [0; 257];
        for count in &mut skewed[28..49] {
            *count = 1_000_000;
        }
        let profiles = [1, 2, 4, 8]
            .into_iter()
            .map(|requested| fit_compact_silu_q7(&[0; 257], requested).unwrap())
            .chain([2, 8].map(|requested| fit_compact_silu_q7(&skewed, requested).unwrap()));
        for profile in profiles {
            for value in -128..=128 {
                let (client, program) = prepare_compact_q7_selection(&profile).unwrap();
                assert_eq!(client.profile_digest(), profile.digest());
                assert_eq!(program.profile_digest(), profile.digest());
                assert!(program.evaluator_ciphertext_bytes().unwrap() <= MAX_CIPHERTEXT_BYTES);
                let (inputs, decoder) = client.encode(value).unwrap();
                let outputs = program.evaluate(inputs).unwrap();
                let index = decoder.decode(outputs).unwrap();
                let expected = profile
                    .pieces()
                    .iter()
                    .position(|piece| (piece.lower()..=piece.upper()).contains(&value))
                    .unwrap();
                assert_eq!(
                    index,
                    expected,
                    "piece mismatch at {value} in profile {:?}",
                    profile.digest()
                );
            }
        }
    }

    #[test]
    fn cross_circuit_labels_and_decoders_fail_and_inputs_have_a_closed_domain() {
        let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
        let (client_a, program_a) = prepare_compact_q7_selection(&profile).unwrap();
        let (client_b, program_b) = prepare_compact_q7_selection(&profile).unwrap();
        let (inputs_a, decoder_a) = client_a.encode(-128).unwrap();
        let (inputs_b, decoder_b) = client_b.encode(128).unwrap();
        assert!(program_a.evaluate(inputs_b).is_err());
        let outputs_b = program_b.evaluate(inputs_a);
        assert!(outputs_b.is_err());
        let _ = (decoder_a, decoder_b);
        let (client_c, program_c) = prepare_compact_q7_selection(&profile).unwrap();
        let (client_d, _) = prepare_compact_q7_selection(&profile).unwrap();
        let (inputs_c, _) = client_c.encode(0).unwrap();
        let (_, decoder_d) = client_d.encode(0).unwrap();
        assert!(decoder_d
            .decode(program_c.evaluate(inputs_c).unwrap())
            .is_err());
        let (client, _) = prepare_compact_q7_selection(&profile).unwrap();
        assert!(client.encode(-129).is_err());
        let (client, _) = prepare_compact_q7_selection(&profile).unwrap();
        assert!(client.encode(129).is_err());
    }
}
