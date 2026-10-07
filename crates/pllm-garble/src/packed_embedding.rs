//! One-use packed-share reconstruction, signed extension and fresh resharing.
//!
//! A local trusted issuer encodes both parties' inputs. This independently
//! constructed half-gates boundary is not SiRNN's distributed OT protocol.
use crate::boolean::{
    BooleanCircuitBuilder, BooleanCircuitClient, BooleanCircuitDecoder, BooleanCircuitInputs,
    BooleanCircuitOutputs, BooleanCircuitProgram,
};
use pllm_core::packed_embedding::{Layout, COEFFICIENT_BITS, MODULUS};
use zeroize::Zeroizing;

pub const MAX_ALLOCATION_BYTES: usize = 4 << 20;
const MAX_WIRES: usize = 8192;

pub struct Issuer {
    client: BooleanCircuitClient,
    channels: usize,
}
pub struct Program {
    circuit: BooleanCircuitProgram,
    layout: Layout,
}
pub struct Inputs(BooleanCircuitInputs);
pub struct Outputs(BooleanCircuitOutputs);
pub struct Decoder(BooleanCircuitDecoder);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub issuer_encoding_bytes: usize,
    pub evaluator_material_bytes: usize,
    pub boundary_label_bytes: usize,
    pub public_metadata_bytes_estimate: usize,
    pub and_gates: usize,
    pub wires: usize,
}

pub fn prepare_reference(layout: Layout) -> Result<(Issuer, Program), String> {
    let mut b = BooleanCircuitBuilder::with_wire_limit(MAX_WIRES)?;
    let client_share = b.input_word(COEFFICIENT_BITS)?;
    let provider_share = b.input_word(COEFFICIENT_BITS)?;
    let masks = layout
        .widths()
        .iter()
        .map(|_| b.input_word(32))
        .collect::<Result<Vec<_>, _>>()?;
    // Carries cross packed slots. Reconstruct under opaque labels BEFORE any
    // extraction or sign extension; no intermediate word is decoded.
    let (reconstructed, _) = b.add_unsigned(&client_share, &provider_share)?;
    let mut offset = 0;
    let mut outputs = Vec::new();
    for (&bits, mask) in layout.widths().iter().zip(&masks) {
        let mut signed = reconstructed[offset..offset + bits].to_vec();
        signed.resize(32, reconstructed[offset + bits - 1]);
        outputs.extend(b.subtract_unsigned(&signed, mask)?.0);
        offset += bits + 1;
    }
    let (client, circuit) = b.finish(&outputs)?;
    Ok((
        Issuer {
            client,
            channels: layout.widths().len(),
        },
        Program { circuit, layout },
    ))
}

impl Issuer {
    pub fn encode(
        self,
        client_share: u16,
        provider_share: u16,
        output_masks: &[u32],
    ) -> Result<(Inputs, Decoder), String> {
        if client_share >= MODULUS
            || provider_share >= MODULUS
            || output_masks.len() != self.channels
        {
            return Err("packed-share inputs exceed the committed geometry".into());
        }
        let mut bits = Zeroizing::new(
            [client_share, provider_share]
                .into_iter()
                .flat_map(|v| (0..COEFFICIENT_BITS).map(move |i| v & (1 << i) != 0))
                .collect::<Vec<_>>(),
        );
        for &mask in output_masks {
            bits.extend((0..32).map(|i| mask & (1 << i) != 0));
        }
        let (inputs, decoder) = self.client.encode(&bits)?;
        Ok((Inputs(inputs), Decoder(decoder)))
    }
}

impl Program {
    pub fn resources(&self) -> Result<Resources, String> {
        let boundary_bits = 2 * COEFFICIENT_BITS + 64 * self.layout.widths().len();
        Ok(Resources {
            issuer_encoding_bytes: boundary_bits * 32,
            evaluator_material_bytes: self.circuit.evaluator_ciphertext_bytes()? as usize
                + self.circuit.constant_label_bytes(),
            boundary_label_bytes: boundary_bits * 16,
            public_metadata_bytes_estimate: self.circuit.public_metadata_bytes_estimate(),
            and_gates: self.circuit.and_gate_count(),
            wires: self.circuit.wire_count(),
        })
    }
    pub fn evaluate(self, inputs: Inputs) -> Result<Outputs, String> {
        self.circuit.evaluate(inputs.0).map(Outputs)
    }
}

impl Decoder {
    /// Only the client's freshly masked output share is decoded. The provider
    /// retains its independent u32 mask; neither output alone contains a code.
    pub fn decode(self, output: Outputs) -> Result<Vec<u32>, String> {
        self.0.decode(output.0).map(|bits| {
            bits.chunks_exact(32)
                .map(|word| {
                    word.iter()
                        .enumerate()
                        .fold(0_u32, |v, (i, bit)| v | (u32::from(*bit) << i))
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_wrap_carries_signed_extension_and_output_masks_match_an_independent_oracle() {
        for layout in [Layout::Int8, Layout::Int4, Layout::Mixed433] {
            let mut fixed = None;
            for word_index in 0..32 {
                let codes = layout
                    .widths()
                    .iter()
                    .map(|&width| {
                        let modulus = 1_i32 << width;
                        ((word_index * 17 + 3) % modulus - modulus / 2) as i8
                    })
                    .collect::<Vec<_>>();
                let word = layout.pack(&codes).unwrap();
                for mask in [0, 1, 15, 31, 32, 511, 4095, 8191] {
                    let (issuer, program) = prepare_reference(layout).unwrap();
                    let resources = program.resources().unwrap();
                    assert!(resources.wires * 384 < MAX_ALLOCATION_BYTES);
                    if let Some(expected) = fixed {
                        assert_eq!(resources, expected);
                    }
                    fixed = Some(resources);
                    let output_masks = [0, u32::MAX, 0x8123_ffff][..codes.len()].to_vec();
                    let (inputs, decoder) = issuer
                        .encode(word.wrapping_sub(mask) & (MODULUS - 1), mask, &output_masks)
                        .unwrap();
                    let client = decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
                    for ((actual, mask), code) in client.into_iter().zip(output_masks).zip(&codes) {
                        assert_eq!(actual.wrapping_add(mask), *code as i32 as u32);
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_and_foreign_issuances_consume_their_handles() {
        let layout = Layout::Mixed433;
        let (a, pa) = prepare_reference(layout).unwrap();
        let (b, pb) = prepare_reference(layout).unwrap();
        let (ia, _) = a.encode(0, 0, &[0; 3]).unwrap();
        let (ib, _) = b.encode(0, 0, &[0; 3]).unwrap();
        assert!(pa.evaluate(ib).is_err());
        assert!(pb.evaluate(ia).is_err());
        assert!(prepare_reference(layout)
            .unwrap()
            .0
            .encode(MODULUS, 0, &[0; 3])
            .is_err());
        assert!(prepare_reference(layout)
            .unwrap()
            .0
            .encode(0, 0, &[0; 2])
            .is_err());
    }
}
