//! Private selection, normalized polynomial evaluation and complete SiLU×up.
//! Fresh serial half-gates material; only the trusted client decodes the result.
use crate::boolean::{
    BooleanCircuitBuilder as Builder, BooleanCircuitClient, BooleanCircuitDecoder,
    BooleanCircuitInputs, BooleanCircuitOutputs, BooleanCircuitProgram, BooleanWire,
};
use pllm_core::compact_product::{Profile, TAIL_Q12};

const GATE_BITS: usize = 25;
const UP_BITS: usize = 27;
const OUTPUT_BITS: usize = 26;
const WORK_BITS: usize = 40;
const MAX_WIRES: usize = 131_072;
pub const MAX_ALLOCATION_BYTES: usize = 64 << 20;

pub struct Client(BooleanCircuitClient);
pub struct Inputs(BooleanCircuitInputs);
pub struct Outputs(BooleanCircuitOutputs);
pub struct Decoder(BooleanCircuitDecoder);
pub struct Program {
    circuit: BooleanCircuitProgram,
    parts: [usize; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub client_encoding_bytes: usize,
    pub evaluator_half_gate_bytes: usize,
    pub evaluator_constant_bytes: usize,
    pub public_metadata_bytes_estimate: usize,
    pub input_output_label_bytes: usize,
    pub allocation_bound_bytes: usize,
    pub and_gates: usize,
    pub wires: usize,
    pub arithmetic_bytes: [usize; 4],
}

fn signed_constant(b: &mut Builder, value: i64, width: usize) -> Result<Vec<BooleanWire>, String> {
    b.constant_word(value as u128 & ((1u128 << width) - 1), width)
}

fn rounded_signed(
    b: &mut Builder,
    input: &[BooleanWire],
    shift: usize,
    width: usize,
) -> Result<Vec<BooleanWire>, String> {
    let magnitude = b.absolute_signed(input)?;
    let rounded = b.round_shift_ties_even(&magnitude, shift, width.max(input.len() - shift + 1))?;
    // Retain the carry before narrowing. Core proves gate/up and polynomial
    // magnitudes fit the target signed word; unselected coordinates never enter
    // arithmetic. The profile bound includes the final rounding half-unit.
    b.conditional_negate(&rounded[..width], input[input.len() - 1])
}

fn product(
    b: &mut Builder,
    left: &[BooleanWire],
    right: &[BooleanWire],
    shift: usize,
    width: usize,
) -> Result<Vec<BooleanWire>, String> {
    let negative = b.xor(left[left.len() - 1], right[right.len() - 1])?;
    let l = b.absolute_signed(left)?;
    let r = b.absolute_signed(right)?;
    let multiplied = b.multiply_unsigned_wide(&l, &r)?;
    let rounded =
        b.round_shift_ties_even(&multiplied, shift, width.max(multiplied.len() - shift + 1))?;
    // Horner's coefficient-sum bound proves signed 40-bit intermediates; the
    // activation bound <2^18 and |up_Q12|≤2^19 prove the signed 26-bit result.
    b.conditional_negate(&rounded[..width], negative)
}

pub fn prepare_reference(profile: &Profile) -> Result<(Client, Program), String> {
    // Core's immutable bounded fitter proves widths/coefficient bounds before
    // any secret allocation. No caller-created unchecked profile is accepted.
    let mut b = Builder::with_wire_limit(MAX_WIRES)?;
    let gate_input = b.input_word(GATE_BITS)?;
    let up_input = b.input_word(UP_BITS)?;
    let gate = rounded_signed(&mut b, &gate_input, 6, 19)?;
    let up = rounded_signed(&mut b, &up_input, 6, 21)?;
    let rescale_end = b.and_gate_count();
    let zero = b.constant(false)?;
    let mut biased = gate.clone();
    biased[18] = b.not(biased[18])?;
    let mut coordinate = vec![zero; 22];
    let mut coefficients = vec![vec![zero; WORK_BITS]; usize::from(profile.degree()) + 1];
    for piece in profile.pieces() {
        let lower = b
            .compare_unsigned_to_constant(&biased, (piece.lower + (1 << 18)) as u128)?
            .0;
        let upper = b
            .compare_unsigned_to_constant(&biased, (piece.upper + (1 << 18)) as u128)?
            .0;
        let not_lower = b.not(lower)?;
        let selected = b.and(not_lower, upper)?;
        let doubled = [vec![zero], gate.clone()].concat();
        let offset = signed_constant(&mut b, -i64::from(piece.lower + piece.upper), 20)?;
        let centered = b.add_unsigned(&doubled, &offset)?.0;
        let shift = 20 - (piece.upper - piece.lower).ilog2() as usize;
        let mut normalized = vec![zero; 22];
        for (out, &bit) in normalized.iter_mut().skip(shift).zip(&centered) {
            *out = bit;
        }
        coordinate = b.select_word(selected, &coordinate, &normalized)?;
        for (index, coefficient) in coefficients.iter_mut().enumerate() {
            let public = signed_constant(&mut b, piece.coefficients[index], WORK_BITS)?;
            *coefficient = b.select_word(selected, coefficient, &public)?;
        }
    }
    let selection_end = b.and_gate_count();
    let mut polynomial = coefficients.pop().expect("degree is at least two");
    for coefficient in coefficients.iter().rev() {
        polynomial = product(&mut b, &polynomial, &coordinate, 20, WORK_BITS)?;
        polynomial = b.add_unsigned(&polynomial, coefficient)?.0;
    }
    let mut activated = rounded_signed(&mut b, &polynomial, 20, 19)?;
    let poly_end = b.and_gate_count();
    let lower_tail = b
        .compare_unsigned_to_constant(&biased, (-TAIL_Q12 + 1 + (1 << 18)) as u128)?
        .0;
    let before_upper_tail = b
        .compare_unsigned_to_constant(&biased, (TAIL_Q12 + (1 << 18)) as u128)?
        .0;
    let upper_tail = b.not(before_upper_tail)?;
    activated = b.select_word(lower_tail, &activated, &[zero; 19])?;
    activated = b.select_word(upper_tail, &activated, &gate)?;
    let output = product(&mut b, &activated, &up, 12, OUTPUT_BITS)?;
    let total = b.and_gate_count();
    let (client, circuit) = b.finish(&output)?;
    Ok((
        Client(client),
        Program {
            circuit,
            parts: [
                rescale_end,
                selection_end - rescale_end,
                poly_end - selection_end,
                total - poly_end,
            ]
            .map(|gates| gates * 32),
        },
    ))
}

impl Client {
    pub fn encode(self, gate: i32, up: i32) -> Result<(Inputs, Decoder), String> {
        Profile::validate_inputs(gate, up)?;
        let bits = [(gate, GATE_BITS), (up, UP_BITS)]
            .into_iter()
            .flat_map(|(v, width)| (0..width).map(move |i| v as u32 & (1 << i) != 0))
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
            evaluator_constant_bytes: self.circuit.constant_label_bytes(),
            public_metadata_bytes_estimate: self.circuit.public_metadata_bytes_estimate(),
            input_output_label_bytes: (GATE_BITS + UP_BITS + OUTPUT_BITS) * 16,
            allocation_bound_bytes: MAX_ALLOCATION_BYTES,
            and_gates: self.circuit.and_gate_count(),
            wires: self.circuit.wire_count(),
            arithmetic_bytes: self.parts,
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
    pub fn decode(self, outputs: Outputs) -> Result<i32, String> {
        let bits = self.0.decode(outputs.0)?;
        let raw = bits
            .iter()
            .enumerate()
            .fold(0, |v, (i, &bit)| v | ((bit as i32) << i));
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
    use pllm_core::compact_product::CALIBRATION_BINS;

    #[test]
    fn exact_public_boundaries_signed_ties_and_every_piece_remain_private() {
        for degree in [2, 3] {
            let profile = Profile::fit(&[0; CALIBRATION_BINS], 8, degree).unwrap();
            let mut gates = vec![-48 << 18, 48 << 18, -33, -32, -31, 31, 32, 33];
            for p in profile.pieces() {
                for edge in [p.lower, p.upper] {
                    gates.extend([-33, -32, 0, 32, 33].map(|d| (edge << 6) + d));
                }
            }
            let mut resources = None;
            for (index, gate) in gates.into_iter().enumerate() {
                let up = [-128 << 18, 128 << 18, -97, -96, -95, 95, 96, 97][index % 8];
                let (client, program) = prepare_reference(&profile).unwrap();
                let r = program.resources().unwrap();
                assert!(r.wires * 384 + (2 << 20) < MAX_ALLOCATION_BYTES);
                if let Some(before) = resources {
                    assert_eq!(before, r);
                }
                resources = Some(r);
                let (input, decoder) = client.encode(gate, up).unwrap();
                assert_eq!(
                    decoder.decode(program.evaluate(input).unwrap()).unwrap(),
                    profile.evaluate(gate, up).unwrap(),
                    "{degree}: {gate} {up}"
                );
            }
        }
    }

    #[test]
    fn foreign_issuance_and_invalid_inputs_consume_handles() {
        let p = Profile::fit(&[0; CALIBRATION_BINS], 1, 2).unwrap();
        let (a, ap) = prepare_reference(&p).unwrap();
        let (b, bp) = prepare_reference(&p).unwrap();
        assert_ne!(ap.issuance_id(), bp.issuance_id());
        let (ai, _) = a.encode(0, 0).unwrap();
        let (bi, _) = b.encode(0, 0).unwrap();
        assert!(ap.evaluate(bi).is_err());
        assert!(bp.evaluate(ai).is_err());
        let (c, _) = prepare_reference(&p).unwrap();
        assert!(c.encode((48 << 18) + 1, 0).is_err());
    }
}
