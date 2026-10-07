//! Complete bounded Q18 -> Q9 SiLU×up: embedded LogRow lookup and half-gates
//! rescaling/multiplication. Only the trusted client decodes the final result.
//! This is an in-process arithmetic adaptation, not a distributed paper system.
use crate::boolean::{
    BooleanCircuitBuilder, BooleanCircuitClient, BooleanCircuitDecoder, BooleanCircuitInputs,
    BooleanCircuitOutputs, BooleanCircuitProgram,
};
use pllm_core::wavelet_gated_reference::Profile;

const GATE_BITS: usize = 25;
const UP_BITS: usize = 27;
const OUTPUT_BITS: usize = 23;
// The fixed topology has fewer than 32K wires. Charge the same 384-byte growing
// encoding/instruction overlap bound as retained selection, plus LUT workspace.
pub const MAX_ALLOCATION_BYTES: usize = 16 << 20;
const MAX_WIRES: usize = 32_768;

pub struct Client(BooleanCircuitClient);
pub struct Inputs(BooleanCircuitInputs);
pub struct Outputs(BooleanCircuitOutputs);
pub struct Decoder(BooleanCircuitDecoder);
pub struct Program {
    circuit: BooleanCircuitProgram,
    arithmetic_bytes: [usize; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub client_encoding_bytes: usize,
    pub evaluator_half_gate_bytes: usize,
    pub evaluator_lookup_bytes: usize,
    pub evaluator_constant_bytes: usize,
    pub public_metadata_bytes_estimate: usize,
    pub input_output_label_bytes: usize,
    pub allocation_bound_bytes: usize,
    pub and_gates: usize,
    pub wires: usize,
    pub arithmetic_bytes: [usize; 3],
}

pub fn prepare_reference(profile: &Profile) -> Result<(Client, Program), String> {
    if profile.levels() != 0 || profile.table().len() != 4096 {
        return Err("LogRow product requires the uncompressed public Q18/Q9 profile".into());
    }
    let mut builder = BooleanCircuitBuilder::with_wire_limit(MAX_WIRES)?;
    let gate = builder.input_word(GATE_BITS)?;
    let up = builder.input_word(UP_BITS)?;
    // Adding 64 to signed gate Q18 toggles the sign bit in this public range.
    // Dropping 13 low bits gives exact floor and keeps the index hidden.
    let mut index = gate[13..].to_vec();
    index[11] = builder.not(index[11])?;
    let table = profile
        .table()
        .iter()
        .map(|&v| v as i16 as u16)
        .collect::<Vec<_>>();
    let activated = builder.logrow_lookup(&index, 16, &table)?;
    let up_magnitude = builder.absolute_signed(&up)?;
    let rounded_up = builder.round_shift_ties_even(&up_magnitude, 9, 19)?;
    let up_gates = builder.and_gate_count();
    let activation_magnitude = builder.absolute_signed(&activated)?;
    // The public ranges prove |up_Q9| <= 65536 and the complete-table product
    // magnitude < 2^31. These slices remove only publicly proven zero high bits.
    let product = builder.multiply_unsigned_wide(&activation_magnitude, &rounded_up[..17])?;
    let product_gates = builder.and_gate_count();
    let rounded = builder.round_shift_ties_even(&product[..31], 9, OUTPUT_BITS)?;
    let negative = builder.xor(activated[15], up[UP_BITS - 1])?;
    let output = builder.conditional_negate(&rounded, negative)?;
    let (client, program) = builder.finish(&output)?;
    if program.wire_count() > MAX_WIRES {
        return Err("LogRow product topology exceeds its preflight allocation bound".into());
    }
    let total_gates = program.and_gate_count();
    Ok((
        Client(client),
        Program {
            circuit: program,
            arithmetic_bytes: [
                32 * up_gates,
                32 * (product_gates - up_gates),
                32 * (total_gates - product_gates),
            ],
        },
    ))
}

impl Client {
    pub fn encode(self, gate_q18: i32, up_q18: i32) -> Result<(Inputs, Decoder), String> {
        Profile::validate_inputs(gate_q18, up_q18).map_err(|e| format!("{e:?}"))?;
        let bits = [(gate_q18, GATE_BITS), (up_q18, UP_BITS)]
            .into_iter()
            .flat_map(|(value, width)| (0..width).map(move |i| (value as u32) & (1 << i) != 0))
            .collect::<Vec<_>>();
        let (inputs, decoder) = self.0.encode(&bits)?;
        Ok((Inputs(inputs), Decoder(decoder)))
    }
}

impl Program {
    pub fn resources(&self) -> Result<Resources, String> {
        Ok(Resources {
            client_encoding_bytes: (GATE_BITS + UP_BITS + OUTPUT_BITS) * 32,
            evaluator_half_gate_bytes: self.circuit.evaluator_ciphertext_bytes()? as usize,
            evaluator_lookup_bytes: self.circuit.lookup_material_bytes(),
            evaluator_constant_bytes: self.circuit.constant_label_bytes(),
            public_metadata_bytes_estimate: self.circuit.public_metadata_bytes_estimate(),
            input_output_label_bytes: (GATE_BITS + UP_BITS + OUTPUT_BITS) * 16,
            allocation_bound_bytes: MAX_ALLOCATION_BYTES,
            and_gates: self.circuit.and_gate_count(),
            wires: self.circuit.wire_count(),
            arithmetic_bytes: self.arithmetic_bytes,
        })
    }
    pub fn issuance_id(&self) -> [u8; 32] {
        self.circuit.issuance_id()
    }
    pub fn evaluate(self, inputs: Inputs) -> Result<Outputs, String> {
        self.circuit.evaluate(inputs.0).map(Outputs)
    }
}

impl Decoder {
    pub fn decode(self, output: Outputs) -> Result<i32, String> {
        let bits = self.0.decode(output.0)?;
        let raw = bits
            .iter()
            .enumerate()
            .fold(0_i32, |v, (i, b)| v | (i32::from(*b) << i));
        Ok(if bits[OUTPUT_BITS - 1] {
            raw - (1 << OUTPUT_BITS)
        } else {
            raw
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_product_matches_independent_signed_rounding_and_constant_cost() {
        let profile = Profile::new(0).unwrap();
        let mut reference = None;
        for gate in [
            -48 << 18,
            -8193,
            -8192,
            -8191,
            -1,
            0,
            1,
            8191,
            8192,
            8193,
            48 << 18,
        ] {
            for up in [
                -128 << 18,
                -769,
                -768,
                -767,
                -257,
                -256,
                -255,
                255,
                256,
                257,
                128 << 18,
            ] {
                let (client, program) = prepare_reference(&profile).unwrap();
                let resources = program.resources().unwrap();
                assert_eq!(
                    resources.evaluator_lookup_bytes,
                    (23 + 12 * 16) * 16 + 4096 * 2
                );
                assert!(resources.wires * 384 + (2 << 20) < MAX_ALLOCATION_BYTES);
                if let Some(expected) = reference {
                    assert_eq!(resources, expected);
                }
                reference = Some(resources);
                let (inputs, decoder) = client.encode(gate, up).unwrap();
                let actual = decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
                let index = ((i64::from(gate) + (64 << 18)) / 8192) as usize;
                let rounded_up = (f64::from(up) / 512.0).round_ties_even();
                let expected = (f64::from(profile.table()[index]) * rounded_up / 512.0)
                    .round_ties_even() as i32;
                assert_eq!(actual, expected, "gate={gate} up={up}");
                assert_eq!(actual, profile.evaluate(gate, up).unwrap());
            }
        }
    }

    #[test]
    fn invalid_input_and_foreign_one_use_labels_reject() {
        let profile = Profile::new(0).unwrap();
        let (client, _) = prepare_reference(&profile).unwrap();
        assert!(client.encode((48 << 18) + 1, 0).is_err());
        assert!(prepare_reference(&Profile::new(1).unwrap()).is_err());
        let (ca, pa) = prepare_reference(&profile).unwrap();
        let (cb, pb) = prepare_reference(&profile).unwrap();
        assert_ne!(pa.issuance_id(), pb.issuance_id());
        let (ia, _) = ca.encode(-12345, 98765).unwrap();
        let (ib, _) = cb.encode(-12345, 98765).unwrap();
        assert!(pa.evaluate(ib).is_err());
        assert!(pb.evaluate(ia).is_err());
    }
}
