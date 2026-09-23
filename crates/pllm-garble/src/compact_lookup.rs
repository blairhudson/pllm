//! Exact, bounded Q7 protected-output oracle for a fitted Compact profile.
//!
//! A fixed Boolean lookup circuit evaluates the *encoded outputs* of the
//! public fitted profile on private client input labels. It is intentionally a
//! correctness/cost reference, not Compact's protected polynomial method.
//! The evaluator receives neither the signed input nor decoded output.

use pllm_core::CompactQ7Profile;

use crate::boolean::{
    BooleanCircuitBuilder, BooleanCircuitClient, BooleanCircuitDecoder, BooleanCircuitInputs,
    BooleanCircuitOutputs, BooleanCircuitProgram,
};

const INPUT_BITS: usize = 9;
const OUTPUT_BITS: usize = 9;
const TABLE_ROWS: usize = 256;
const MAX_CIPHERTEXT_BYTES: u64 = 256 * OUTPUT_BITS as u64 * 32;

pub struct CompactQ7LookupClient {
    profile_digest: [u8; 32],
    client: BooleanCircuitClient,
}

pub struct CompactQ7LookupInputs(BooleanCircuitInputs);
pub struct CompactQ7LookupOutputs(BooleanCircuitOutputs);

pub struct CompactQ7LookupDecoder {
    profile_digest: [u8; 32],
    decoder: BooleanCircuitDecoder,
}

pub struct CompactQ7LookupProgram {
    profile_digest: [u8; 32],
    program: BooleanCircuitProgram,
}

/// Build a one-use lookup over all 257 encoded Q7 inputs. The public profile
/// and all output values are checked before circuit material is generated.
pub fn prepare_compact_q7_lookup(
    profile: &CompactQ7Profile,
) -> Result<(CompactQ7LookupClient, CompactQ7LookupProgram), String> {
    let mut encoded_outputs = [0u16; TABLE_ROWS + 1];
    for (offset, encoded) in (-128..=128).enumerate() {
        let output = profile
            .evaluate(encoded)
            .map_err(|error| error.to_string())?;
        if !(-256..=255).contains(&output) {
            return Err("Compact Q7 lookup output does not fit signed 9-bit".into());
        }
        encoded_outputs[offset] = u16::from_le_bytes(output.to_le_bytes()) & 0x01ff;
    }

    let mut builder = BooleanCircuitBuilder::new()?;
    let input = builder.input_word(INPUT_BITS)?;
    let mut leaves = Vec::new();
    leaves
        .try_reserve_exact(TABLE_ROWS)
        .map_err(|_| "Compact Q7 lookup table allocation failed")?;
    for &output in &encoded_outputs[..TABLE_ROWS] {
        leaves.push(builder.constant_word(u128::from(output), OUTPUT_BITS)?);
    }
    for &bit in &input[..INPUT_BITS - 1] {
        let mut parents = Vec::new();
        parents
            .try_reserve_exact(leaves.len() / 2)
            .map_err(|_| "Compact Q7 lookup selection allocation failed")?;
        for pair in leaves.chunks_exact(2) {
            parents.push(builder.select_word(bit, &pair[0], &pair[1])?);
        }
        leaves = parents;
    }
    let high = builder.constant_word(u128::from(encoded_outputs[TABLE_ROWS]), OUTPUT_BITS)?;
    let output = builder.select_word(input[INPUT_BITS - 1], &leaves[0], &high)?;
    let (client, program) = builder.finish(&output)?;
    if program.evaluator_ciphertext_bytes()? != MAX_CIPHERTEXT_BYTES {
        return Err("Compact Q7 lookup circuit does not match its resource bound".into());
    }
    let profile_digest = profile.digest();
    Ok((
        CompactQ7LookupClient {
            profile_digest,
            client,
        },
        CompactQ7LookupProgram {
            profile_digest,
            program,
        },
    ))
}

impl CompactQ7LookupClient {
    /// Consumes input encodings on success or range failure.
    pub fn encode(
        self,
        value: i16,
    ) -> Result<(CompactQ7LookupInputs, CompactQ7LookupDecoder), String> {
        if !(-128..=128).contains(&value) {
            return Err("Compact Q7 lookup input is outside [-128, 128]".into());
        }
        let offset = (value + 128) as u16;
        let bits = (0..INPUT_BITS)
            .map(|index| offset & (1_u16 << index) != 0)
            .collect::<Vec<_>>();
        let (inputs, decoder) = self.client.encode(&bits)?;
        Ok((
            CompactQ7LookupInputs(inputs),
            CompactQ7LookupDecoder {
                profile_digest: self.profile_digest,
                decoder,
            },
        ))
    }

    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }
}

impl CompactQ7LookupProgram {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    pub fn evaluator_ciphertext_bytes(&self) -> Result<u64, String> {
        self.program.evaluator_ciphertext_bytes()
    }

    /// Consumes all one-use circuit material, including on an input mismatch.
    pub fn evaluate(self, inputs: CompactQ7LookupInputs) -> Result<CompactQ7LookupOutputs, String> {
        self.program.evaluate(inputs.0).map(CompactQ7LookupOutputs)
    }
}

impl CompactQ7LookupDecoder {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    pub fn decode(self, outputs: CompactQ7LookupOutputs) -> Result<i16, String> {
        let bits = self.decoder.decode(outputs.0)?;
        if bits.len() != OUTPUT_BITS {
            return Err("Compact Q7 lookup output width is invalid".into());
        }
        let raw = bits.iter().enumerate().fold(0_i16, |value, (shift, bit)| {
            value | (i16::from(*bit) << shift)
        });
        Ok(if raw & 0x0100 != 0 { raw - 512 } else { raw })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_core::fit_compact_silu_q7;

    #[test]
    fn protected_output_matches_exact_profile_on_every_q7_input() {
        let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
        for input in -128..=128 {
            let (client, program) = prepare_compact_q7_lookup(&profile).unwrap();
            assert_eq!(client.profile_digest(), profile.digest());
            assert_eq!(program.profile_digest(), profile.digest());
            assert_eq!(
                program.evaluator_ciphertext_bytes(),
                Ok(MAX_CIPHERTEXT_BYTES)
            );
            let (inputs, decoder) = client.encode(input).unwrap();
            assert_eq!(decoder.profile_digest(), profile.digest());
            let outputs = program.evaluate(inputs).unwrap();
            assert_eq!(
                decoder.decode(outputs),
                Ok(profile.evaluate(input).unwrap())
            );
        }
    }

    #[test]
    fn public_density_changes_the_locked_oracle_without_changing_its_client_boundary() {
        let mut counts = [0; 257];
        for count in &mut counts[28..49] {
            *count = 1_000_000;
        }
        let profile = fit_compact_silu_q7(&counts, 8).unwrap();
        let uniform = fit_compact_silu_q7(&[0; 257], 8).unwrap();
        assert_ne!(profile.digest(), uniform.digest());
        for input in [-128, -101, -100, -80, -79, 0, 128] {
            let (client, program) = prepare_compact_q7_lookup(&profile).unwrap();
            let (inputs, decoder) = client.encode(input).unwrap();
            assert_eq!(
                decoder.decode(program.evaluate(inputs).unwrap()),
                profile.evaluate(input).map_err(|error| error.to_string())
            );
        }
    }

    #[test]
    fn inputs_and_outputs_are_bound_to_a_single_circuit() {
        let profile = fit_compact_silu_q7(&[0; 257], 2).unwrap();
        let (client_a, program_a) = prepare_compact_q7_lookup(&profile).unwrap();
        let (client_b, program_b) = prepare_compact_q7_lookup(&profile).unwrap();
        let (inputs_a, _) = client_a.encode(-20).unwrap();
        let (inputs_b, decoder_b) = client_b.encode(20).unwrap();
        assert!(program_a.evaluate(inputs_b).is_err());
        let outputs_b = program_b.evaluate(inputs_a);
        assert!(outputs_b.is_err());
        let _ = decoder_b;
        let (client_c, program_c) = prepare_compact_q7_lookup(&profile).unwrap();
        let (client_d, _) = prepare_compact_q7_lookup(&profile).unwrap();
        let (inputs_c, _) = client_c.encode(0).unwrap();
        let (_, decoder_d) = client_d.encode(0).unwrap();
        assert!(decoder_d
            .decode(program_c.evaluate(inputs_c).unwrap())
            .is_err());
        let (client, _) = prepare_compact_q7_lookup(&profile).unwrap();
        assert!(client.encode(-129).is_err());
        let (client, _) = prepare_compact_q7_lookup(&profile).unwrap();
        assert!(client.encode(129).is_err());
    }
}
