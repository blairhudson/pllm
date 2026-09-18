//! Experimental Boolean free-XOR and half-gates primitives.
//!
//! This is an independent implementation of Zahur, Rosulek, and Evans (2015),
//! Figure 2. It uses the fixed-key AES Davies-Meyer hash described in their
//! Definition 2. It has not received an independent cryptographic review.

use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use sha2::{Digest as _, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

pub(crate) const LABEL_BYTES: usize = 16;
pub(crate) const CIRCUIT_ID_BYTES: usize = 32;
const MAX_REFERENCE_WIRES: usize = 2_000_000;
const MAX_REFERENCE_AND_GATES: usize = 1_000_000;
const MAX_REFERENCE_INPUTS: usize = 131_072;
const MAX_REFERENCE_OUTPUTS: usize = 131_072;
const CIRCUIT_TWEAK_DOMAIN: &[u8] = b"pllm.garble.half_gates.circuit_tweak.v1";

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct BooleanLabel([u8; LABEL_BYTES]);

impl BooleanLabel {
    fn selection_bit(&self) -> bool {
        self.0[LABEL_BYTES - 1] & 1 == 1
    }
}

impl Drop for BooleanLabel {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

pub(crate) struct BooleanWireEncoding {
    zero: [u8; LABEL_BYTES],
    delta: [u8; LABEL_BYTES],
}

impl BooleanWireEncoding {
    fn encode(&self, value: bool) -> BooleanLabel {
        BooleanLabel(if value {
            xor_block(self.zero, self.delta)
        } else {
            self.zero
        })
    }

    fn decode(&self, label: &BooleanLabel) -> Result<bool, String> {
        if bool::from(label.0.ct_eq(&self.zero)) {
            Ok(false)
        } else if bool::from(label.0.ct_eq(&xor_block(self.zero, self.delta))) {
            Ok(true)
        } else {
            Err("Boolean label does not belong to this wire".into())
        }
    }
}

impl Drop for BooleanWireEncoding {
    fn drop(&mut self) {
        self.zero.zeroize();
        self.delta.zeroize();
    }
}

pub(crate) struct HalfGate {
    circuit_id: [u8; CIRCUIT_ID_BYTES],
    gate_index: u64,
    generator_table: [u8; LABEL_BYTES],
    evaluator_table: [u8; LABEL_BYTES],
}

impl HalfGate {
    fn evaluate(self, left: &BooleanLabel, right: &BooleanLabel) -> BooleanLabel {
        let mut left_hash = fixed_key_hash(&left.0, &self.circuit_id, self.gate_index, 0);
        let mut right_hash = fixed_key_hash(&right.0, &self.circuit_id, self.gate_index, 1);
        let mut generator = if left.selection_bit() {
            xor_block(left_hash, self.generator_table)
        } else {
            left_hash
        };
        let mut evaluator = if right.selection_bit() {
            xor_block(right_hash, xor_block(self.evaluator_table, left.0))
        } else {
            right_hash
        };
        let output = BooleanLabel(xor_block(generator, evaluator));
        left_hash.zeroize();
        right_hash.zeroize();
        generator.zeroize();
        evaluator.zeroize();
        output
    }
}

impl Drop for HalfGate {
    fn drop(&mut self) {
        self.generator_table.zeroize();
        self.evaluator_table.zeroize();
    }
}

struct BooleanGarbler {
    circuit_id: [u8; CIRCUIT_ID_BYTES],
    delta: [u8; LABEL_BYTES],
    next_gate: u64,
}

impl BooleanGarbler {
    fn new() -> Result<Self, String> {
        let mut circuit_id = [0_u8; CIRCUIT_ID_BYTES];
        getrandom::fill(&mut circuit_id).map_err(|error| error.to_string())?;
        let mut delta = [0_u8; LABEL_BYTES];
        getrandom::fill(&mut delta).map_err(|error| error.to_string())?;
        delta[LABEL_BYTES - 1] |= 1;
        Ok(Self {
            circuit_id,
            delta,
            next_gate: 0,
        })
    }

    fn wire(&self) -> Result<BooleanWireEncoding, String> {
        let mut zero = [0_u8; LABEL_BYTES];
        getrandom::fill(&mut zero).map_err(|error| error.to_string())?;
        Ok(BooleanWireEncoding {
            zero,
            delta: self.delta,
        })
    }

    fn xor(
        &self,
        left: &BooleanWireEncoding,
        right: &BooleanWireEncoding,
    ) -> Result<BooleanWireEncoding, String> {
        self.require_compatible(left)?;
        self.require_compatible(right)?;
        Ok(BooleanWireEncoding {
            zero: xor_block(left.zero, right.zero),
            delta: self.delta,
        })
    }

    fn not(&self, input: &BooleanWireEncoding) -> Result<BooleanWireEncoding, String> {
        self.require_compatible(input)?;
        Ok(BooleanWireEncoding {
            zero: xor_block(input.zero, self.delta),
            delta: self.delta,
        })
    }

    fn garble_and(
        &mut self,
        left: &BooleanWireEncoding,
        right: &BooleanWireEncoding,
    ) -> Result<(HalfGate, BooleanWireEncoding), String> {
        self.require_compatible(left)?;
        self.require_compatible(right)?;
        let gate_index = self.next_gate;
        self.next_gate = self
            .next_gate
            .checked_add(1)
            .ok_or("Boolean half-gate index is exhausted")?;

        let mut left_one = xor_block(left.zero, self.delta);
        let mut right_one = xor_block(right.zero, self.delta);
        let left_permute = left.zero[LABEL_BYTES - 1] & 1 == 1;
        let right_permute = right.zero[LABEL_BYTES - 1] & 1 == 1;
        let mut left_zero_hash = fixed_key_hash(&left.zero, &self.circuit_id, gate_index, 0);
        let mut left_one_hash = fixed_key_hash(&left_one, &self.circuit_id, gate_index, 0);
        let mut right_zero_hash = fixed_key_hash(&right.zero, &self.circuit_id, gate_index, 1);
        let mut right_one_hash = fixed_key_hash(&right_one, &self.circuit_id, gate_index, 1);

        let mut generator_table = xor_block(left_zero_hash, left_one_hash);
        if right_permute {
            generator_table = xor_block(generator_table, self.delta);
        }
        let mut generator_zero = if left_permute {
            xor_block(left_zero_hash, generator_table)
        } else {
            left_zero_hash
        };

        let evaluator_table = xor_block(xor_block(right_zero_hash, right_one_hash), left.zero);
        let mut evaluator_zero = if right_permute {
            xor_block(right_zero_hash, xor_block(evaluator_table, left.zero))
        } else {
            right_zero_hash
        };
        let output = BooleanWireEncoding {
            zero: xor_block(generator_zero, evaluator_zero),
            delta: self.delta,
        };
        left_one.zeroize();
        right_one.zeroize();
        left_zero_hash.zeroize();
        left_one_hash.zeroize();
        right_zero_hash.zeroize();
        right_one_hash.zeroize();
        generator_zero.zeroize();
        evaluator_zero.zeroize();
        Ok((
            HalfGate {
                circuit_id: self.circuit_id,
                gate_index,
                generator_table,
                evaluator_table,
            },
            output,
        ))
    }

    fn require_compatible(&self, wire: &BooleanWireEncoding) -> Result<(), String> {
        if wire.delta != self.delta {
            return Err("Boolean wire belongs to a different garbling circuit".into());
        }
        Ok(())
    }
}

impl Drop for BooleanGarbler {
    fn drop(&mut self) {
        self.delta.zeroize();
    }
}

fn evaluate_xor(left: &BooleanLabel, right: &BooleanLabel) -> BooleanLabel {
    BooleanLabel(xor_block(left.0, right.0))
}

fn evaluate_not(input: &BooleanLabel) -> BooleanLabel {
    input.clone()
}

pub(crate) fn fixed_key_hash(
    label: &[u8; LABEL_BYTES],
    circuit_id: &[u8; CIRCUIT_ID_BYTES],
    gate_index: u64,
    half: u8,
) -> [u8; LABEL_BYTES] {
    let mut key = double_gf128(*label);
    let gate_tweak = u128::from(gate_index)
        .checked_mul(2)
        .and_then(|value| value.checked_add(u128::from(half)))
        .expect("u64 gate index and one-bit half fit u128")
        .to_be_bytes();
    let mut tweak = [0_u8; LABEL_BYTES];
    let mut circuit_hasher = Sha256::new();
    circuit_hasher.update(CIRCUIT_TWEAK_DOMAIN);
    circuit_hasher.update(circuit_id);
    let circuit_tweak = circuit_hasher.finalize();
    tweak.copy_from_slice(&circuit_tweak[..LABEL_BYTES]);
    for (byte, gate_byte) in tweak[8..].iter_mut().zip(&gate_tweak[8..]) {
        *byte ^= gate_byte;
    }
    key = xor_block(key, tweak);
    let mut block = key.into();
    Aes128::new(&[0_u8; LABEL_BYTES].into()).encrypt_block(&mut block);
    let mut output: [u8; LABEL_BYTES] = block.into();
    block.zeroize();
    output = xor_block(output, key);
    key.zeroize();
    output
}

fn double_gf128(mut value: [u8; LABEL_BYTES]) -> [u8; LABEL_BYTES] {
    let carry = value[0] >> 7;
    for index in 0..LABEL_BYTES - 1 {
        value[index] = (value[index] << 1) | (value[index + 1] >> 7);
    }
    value[LABEL_BYTES - 1] <<= 1;
    value[LABEL_BYTES - 1] ^= 0x87_u8.wrapping_mul(carry);
    value
}

pub(crate) fn xor_block(left: [u8; LABEL_BYTES], right: [u8; LABEL_BYTES]) -> [u8; LABEL_BYTES] {
    let mut output = [0_u8; LABEL_BYTES];
    for index in 0..LABEL_BYTES {
        output[index] = left[index] ^ right[index];
    }
    output
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanWire {
    index: usize,
    circuit_id: [u8; CIRCUIT_ID_BYTES],
}

enum BooleanInstruction {
    Constant(BooleanLabel),
    Xor {
        left: usize,
        right: usize,
    },
    Not {
        input: usize,
    },
    And {
        left: usize,
        right: usize,
        gate: HalfGate,
    },
}

pub struct BooleanCircuitBuilder {
    garbler: BooleanGarbler,
    encodings: Vec<BooleanWireEncoding>,
    instructions: Vec<BooleanInstruction>,
    input_count: usize,
    and_gate_count: usize,
}

impl BooleanCircuitBuilder {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            garbler: BooleanGarbler::new()?,
            encodings: Vec::new(),
            instructions: Vec::new(),
            input_count: 0,
            and_gate_count: 0,
        })
    }

    pub fn input(&mut self) -> Result<BooleanWire, String> {
        if !self.instructions.is_empty() {
            return Err("Boolean circuit inputs must be declared before gates".into());
        }
        if self.input_count >= MAX_REFERENCE_INPUTS {
            return Err("Boolean reference circuit input limit exceeded".into());
        }
        self.reserve_wire(false)?;
        let wire = BooleanWire {
            index: self.encodings.len(),
            circuit_id: self.garbler.circuit_id,
        };
        self.encodings.push(self.garbler.wire()?);
        self.input_count += 1;
        Ok(wire)
    }

    pub fn input_word(&mut self, width: usize) -> Result<Vec<BooleanWire>, String> {
        if width == 0 || width > 128 {
            return Err("Boolean input word width must be between 1 and 128".into());
        }
        (0..width).map(|_| self.input()).collect()
    }

    pub fn constant(&mut self, value: bool) -> Result<BooleanWire, String> {
        self.reserve_wire(true)?;
        let encoding = self.garbler.wire()?;
        let label = encoding.encode(value);
        let wire = BooleanWire {
            index: self.encodings.len(),
            circuit_id: self.garbler.circuit_id,
        };
        self.encodings.push(encoding);
        self.instructions.push(BooleanInstruction::Constant(label));
        Ok(wire)
    }

    pub fn xor(&mut self, left: BooleanWire, right: BooleanWire) -> Result<BooleanWire, String> {
        let left_encoding = self.encoding(left)?;
        let right_encoding = self.encoding(right)?;
        let output = self.garbler.xor(left_encoding, right_encoding)?;
        self.reserve_wire(true)?;
        let wire = BooleanWire {
            index: self.encodings.len(),
            circuit_id: self.garbler.circuit_id,
        };
        self.encodings.push(output);
        self.instructions.push(BooleanInstruction::Xor {
            left: left.index,
            right: right.index,
        });
        Ok(wire)
    }

    pub fn not(&mut self, input: BooleanWire) -> Result<BooleanWire, String> {
        let output = self.garbler.not(self.encoding(input)?)?;
        self.reserve_wire(true)?;
        let wire = BooleanWire {
            index: self.encodings.len(),
            circuit_id: self.garbler.circuit_id,
        };
        self.encodings.push(output);
        self.instructions
            .push(BooleanInstruction::Not { input: input.index });
        Ok(wire)
    }

    pub fn and(&mut self, left: BooleanWire, right: BooleanWire) -> Result<BooleanWire, String> {
        if self.and_gate_count >= MAX_REFERENCE_AND_GATES {
            return Err("Boolean reference circuit AND-gate limit exceeded".into());
        }
        let left_encoding = clone_encoding(self.encoding(left)?);
        let right_encoding = clone_encoding(self.encoding(right)?);
        let (gate, output) = self.garbler.garble_and(&left_encoding, &right_encoding)?;
        self.reserve_wire(true)?;
        let wire = BooleanWire {
            index: self.encodings.len(),
            circuit_id: self.garbler.circuit_id,
        };
        self.encodings.push(output);
        self.instructions.push(BooleanInstruction::And {
            left: left.index,
            right: right.index,
            gate,
        });
        self.and_gate_count += 1;
        Ok(wire)
    }

    pub fn add_unsigned(
        &mut self,
        left: &[BooleanWire],
        right: &[BooleanWire],
    ) -> Result<(Vec<BooleanWire>, BooleanWire), String> {
        if left.is_empty() || left.len() != right.len() {
            return Err("Boolean unsigned addends must have the same nonzero width".into());
        }
        let mut carry = self.constant(false)?;
        let mut sum = Vec::new();
        sum.try_reserve_exact(left.len())
            .map_err(|_| "Boolean adder output allocation failed")?;
        for (&left_bit, &right_bit) in left.iter().zip(right) {
            let either = self.xor(left_bit, right_bit)?;
            let sum_bit = self.xor(either, carry)?;
            let both = self.and(left_bit, right_bit)?;
            let carry_either = self.and(carry, either)?;
            carry = self.xor(both, carry_either)?;
            sum.push(sum_bit);
        }
        Ok((sum, carry))
    }

    pub fn multiply_unsigned(
        &mut self,
        left: &[BooleanWire],
        right: &[BooleanWire],
    ) -> Result<Vec<BooleanWire>, String> {
        if left.is_empty() || left.len() != right.len() || left.len() > 64 {
            return Err(
                "Boolean unsigned factors must have the same width between 1 and 64".into(),
            );
        }
        self.multiply_unsigned_wide(left, right)
    }

    pub fn multiply_unsigned_wide(
        &mut self,
        left: &[BooleanWire],
        right: &[BooleanWire],
    ) -> Result<Vec<BooleanWire>, String> {
        if left.is_empty() || right.is_empty() {
            return Err("Boolean unsigned factors must be nonempty".into());
        }
        let width = left
            .len()
            .checked_add(right.len())
            .ok_or("Boolean product width overflows usize")?;
        if width > 128 {
            return Err("Boolean product width exceeds 128 bits".into());
        }
        let zero = self.constant(false)?;
        let mut product = vec![zero; width];
        for (shift, &right_bit) in right.iter().enumerate() {
            let mut partial = vec![zero; width];
            for (index, &left_bit) in left.iter().enumerate() {
                partial[index + shift] = self.and(left_bit, right_bit)?;
            }
            product = self.add_unsigned(&product, &partial)?.0;
        }
        Ok(product)
    }

    pub fn constant_word(&mut self, value: u128, width: usize) -> Result<Vec<BooleanWire>, String> {
        if width == 0 || width > 128 || (width < 128 && value >= (1_u128 << width)) {
            return Err("Boolean constant does not fit its declared width".into());
        }
        (0..width)
            .map(|shift| self.constant(value & (1_u128 << shift) != 0))
            .collect()
    }

    pub fn multiply_by_public(
        &mut self,
        input: &[BooleanWire],
        factor: u128,
        output_width: usize,
    ) -> Result<Vec<BooleanWire>, String> {
        if input.is_empty() || output_width == 0 || output_width > 128 {
            return Err("Boolean public multiplication width is invalid".into());
        }
        let zero = self.constant(false)?;
        let mut product = vec![zero; output_width];
        for shift in 0..128 {
            if factor & (1_u128 << shift) == 0 {
                continue;
            }
            let mut partial = vec![zero; output_width];
            for (index, &bit) in input.iter().enumerate() {
                if index + shift < output_width {
                    partial[index + shift] = bit;
                }
            }
            product = self.add_unsigned(&product, &partial)?.0;
        }
        Ok(product)
    }

    pub fn compare_unsigned_to_constant(
        &mut self,
        value: &[BooleanWire],
        constant: u128,
    ) -> Result<(BooleanWire, BooleanWire), String> {
        if value.is_empty() || value.len() > 128 {
            return Err("Boolean comparison width must be between 1 and 128".into());
        }
        if value.len() < 128 && constant >= (1_u128 << value.len()) {
            return Err("Boolean comparison constant does not fit the value width".into());
        }
        let mut less = self.constant(false)?;
        let mut equal = self.constant(true)?;
        for index in (0..value.len()).rev() {
            if constant & (1_u128 << index) != 0 {
                let not_value = self.not(value[index])?;
                let newly_less = self.and(equal, not_value)?;
                less = self.xor(less, newly_less)?;
                equal = self.and(equal, value[index])?;
            } else {
                let not_value = self.not(value[index])?;
                equal = self.and(equal, not_value)?;
            }
        }
        Ok((less, equal))
    }

    pub fn compare_unsigned(
        &mut self,
        left: &[BooleanWire],
        right: &[BooleanWire],
    ) -> Result<(BooleanWire, BooleanWire), String> {
        if left.is_empty() || left.len() != right.len() || left.len() > 128 {
            return Err("Boolean comparison operands must share a width up to 128".into());
        }
        let mut less = self.constant(false)?;
        let mut equal = self.constant(true)?;
        for (&left_bit, &right_bit) in left.iter().zip(right).rev() {
            let not_left = self.not(left_bit)?;
            let left_less = self.and(not_left, right_bit)?;
            let newly_less = self.and(equal, left_less)?;
            less = self.xor(less, newly_less)?;
            let different = self.xor(left_bit, right_bit)?;
            let same = self.not(different)?;
            equal = self.and(equal, same)?;
        }
        Ok((less, equal))
    }

    pub fn subtract_unsigned(
        &mut self,
        left: &[BooleanWire],
        right: &[BooleanWire],
    ) -> Result<(Vec<BooleanWire>, BooleanWire), String> {
        if left.is_empty() || left.len() != right.len() || left.len() > 128 {
            return Err("Boolean subtraction operands must share a width up to 128".into());
        }
        let mut borrow = self.constant(false)?;
        let mut difference = Vec::new();
        difference
            .try_reserve_exact(left.len())
            .map_err(|_| "Boolean subtraction output allocation failed")?;
        for (&left_bit, &right_bit) in left.iter().zip(right) {
            let either = self.xor(left_bit, right_bit)?;
            difference.push(self.xor(either, borrow)?);
            let not_left = self.not(left_bit)?;
            let direct_borrow = self.and(not_left, right_bit)?;
            let same = self.not(either)?;
            let carried_borrow = self.and(borrow, same)?;
            borrow = self.xor(direct_borrow, carried_borrow)?;
        }
        Ok((difference, borrow))
    }

    pub fn select_word(
        &mut self,
        condition: BooleanWire,
        when_false: &[BooleanWire],
        when_true: &[BooleanWire],
    ) -> Result<Vec<BooleanWire>, String> {
        if when_false.is_empty() || when_false.len() != when_true.len() || when_false.len() > 128 {
            return Err("Boolean selected words must share a width up to 128".into());
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(when_false.len())
            .map_err(|_| "Boolean selection output allocation failed")?;
        for (&false_bit, &true_bit) in when_false.iter().zip(when_true) {
            let difference = self.xor(false_bit, true_bit)?;
            let selected_difference = self.and(condition, difference)?;
            output.push(self.xor(false_bit, selected_difference)?);
        }
        Ok(output)
    }

    pub fn divide_public_by_secret(
        &mut self,
        numerator: u128,
        denominator: &[BooleanWire],
    ) -> Result<Vec<BooleanWire>, String> {
        if numerator == 0 || denominator.is_empty() || denominator.len() > 127 {
            return Err("Boolean public/secret division dimensions are invalid".into());
        }
        let numerator_width = usize::try_from(128 - numerator.leading_zeros())
            .map_err(|_| "Boolean division numerator width overflows usize")?;
        let zero = self.constant(false)?;
        let mut extended_denominator = denominator.to_vec();
        extended_denominator.push(zero);
        let mut remainder = vec![zero; extended_denominator.len()];
        let mut quotient = vec![zero; numerator_width];
        for bit in (0..numerator_width).rev() {
            let incoming = self.constant(numerator & (1_u128 << bit) != 0)?;
            let mut shifted = Vec::new();
            shifted
                .try_reserve_exact(remainder.len())
                .map_err(|_| "Boolean division remainder allocation failed")?;
            shifted.push(incoming);
            shifted.extend(remainder.iter().take(remainder.len() - 1).copied());
            let (difference, borrow) = self.subtract_unsigned(&shifted, &extended_denominator)?;
            let greater_or_equal = self.not(borrow)?;
            remainder = self.select_word(greater_or_equal, &shifted, &difference)?;
            quotient[bit] = greater_or_equal;
        }
        Ok(quotient)
    }

    pub(crate) fn reciprocal_sqrt_q(
        &mut self,
        sum_squares: &[BooleanWire],
        row_width: u32,
        fractional_bits: u32,
    ) -> Result<Vec<BooleanWire>, String> {
        if sum_squares.is_empty() || sum_squares.len() > 44 || row_width == 0 || row_width > 8192 {
            return Err("Boolean reciprocal-square-root dimensions are invalid".into());
        }
        if !(5..=30).contains(&fractional_bits) {
            return Err("Boolean reciprocal-square-root precision must be between 5 and 30".into());
        }
        let numerator = u128::from(row_width)
            .checked_mul(1_000_000)
            .and_then(|value| value.checked_shl(2 * fractional_bits))
            .ok_or("Boolean reciprocal-square-root numerator overflows u128")?;
        let epsilon = u128::from(row_width)
            .checked_mul(1_048_576)
            .ok_or("Boolean reciprocal-square-root epsilon overflows u128")?;
        let scaled_sum = self.multiply_by_public(sum_squares, 1_000_000, 64)?;
        let epsilon_word = self.constant_word(epsilon, 64)?;
        let denominator = self.add_unsigned(&scaled_sum, &epsilon_word)?.0;
        let quotient = self.divide_public_by_secret(numerator, &denominator)?;
        let zero = self.constant(false)?;
        let one = self.constant(true)?;
        let output_width = usize::try_from(fractional_bits)
            .map_err(|_| "Boolean reciprocal-square-root width overflows usize")?;
        let mut result = vec![zero; output_width];
        for bit in (0..output_width).rev() {
            let mut trial = result.clone();
            trial[bit] = one;
            let square = self.multiply_unsigned(&trial, &trial)?;
            let mut extended_square = square;
            extended_square.resize(quotient.len(), zero);
            let (less, equal) = self.compare_unsigned(&extended_square, &quotient)?;
            result[bit] = self.xor(less, equal)?;
        }

        let mut twice_plus_one = Vec::new();
        twice_plus_one
            .try_reserve_exact(output_width + 1)
            .map_err(|_| "Boolean reciprocal-square-root rounding allocation failed")?;
        twice_plus_one.push(one);
        twice_plus_one.extend(result.iter().copied());
        let threshold_square = self.multiply_unsigned(&twice_plus_one, &twice_plus_one)?;
        let threshold = self.multiply_unsigned_wide(&threshold_square, &denominator)?;
        let four_numerator = numerator
            .checked_mul(4)
            .ok_or("Boolean reciprocal-square-root rounding threshold overflows u128")?;
        let (below, equal) = self.compare_unsigned_to_constant(&threshold, four_numerator)?;
        let equal_and_odd = self.and(equal, result[0])?;
        let mut carry = self.xor(below, equal_and_odd)?;
        let mut rounded = Vec::new();
        rounded
            .try_reserve_exact(output_width)
            .map_err(|_| "Boolean reciprocal-square-root output allocation failed")?;
        for bit in result {
            rounded.push(self.xor(bit, carry)?);
            carry = self.and(bit, carry)?;
        }
        Ok(rounded)
    }

    pub fn rms_norm_direct_q(
        &mut self,
        rows: &[Vec<BooleanWire>],
        weights: &[i64],
        fractional_bits: u32,
    ) -> Result<Vec<Vec<BooleanWire>>, String> {
        if rows.is_empty() || rows.len() > 8192 || rows.len() != weights.len() {
            return Err("Boolean RMSNorm row and weight counts must match and be nonzero".into());
        }
        let input_width = rows[0].len();
        if !(2..=16).contains(&input_width) || rows.iter().any(|value| value.len() != input_width) {
            return Err("Boolean RMSNorm inputs must share a width between 2 and 16".into());
        }
        let minimum_weight = -(1_i128 << (input_width - 1));
        let maximum_weight = (1_i128 << (input_width - 1)) - 1;
        if weights.iter().any(|weight| {
            i128::from(*weight) < minimum_weight || i128::from(*weight) > maximum_weight
        }) {
            return Err("Boolean RMSNorm weight is outside the input fixed-point domain".into());
        }

        let zero = self.constant(false)?;
        let mut sum_squares = vec![zero; 44];
        for value in rows {
            let square = self.square_signed(value)?;
            let mut extended = square;
            extended.resize(44, zero);
            sum_squares = self.add_unsigned(&sum_squares, &extended)?.0;
        }
        let row_width =
            u32::try_from(rows.len()).map_err(|_| "Boolean RMSNorm row width exceeds u32")?;
        let normalizer = self.reciprocal_sqrt_q(&sum_squares, row_width, fractional_bits)?;
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(rows.len())
            .map_err(|_| "Boolean RMSNorm output allocation failed")?;
        for (input, &weight) in rows.iter().zip(weights) {
            let magnitude = self.absolute_signed(input)?;
            let weighted = self.multiply_by_public(
                &magnitude,
                u128::from(weight.unsigned_abs()),
                input_width * 2,
            )?;
            let scaled = self.multiply_unsigned_wide(&weighted, &normalizer)?;
            let rounded = self.round_shift_ties_even(
                &scaled,
                usize::try_from(fractional_bits)
                    .map_err(|_| "Boolean RMSNorm fractional width exceeds usize")?,
                64,
            )?;
            let sign = if weight.is_negative() {
                self.not(input[input_width - 1])?
            } else {
                input[input_width - 1]
            };
            outputs.push(self.conditional_negate(&rounded, sign)?);
        }
        Ok(outputs)
    }

    pub fn square_signed(&mut self, input: &[BooleanWire]) -> Result<Vec<BooleanWire>, String> {
        if input.len() < 2 || input.len() > 32 {
            return Err("Boolean signed square input width must be between 2 and 32".into());
        }
        let magnitude = self.absolute_signed(input)?;
        self.multiply_unsigned(&magnitude, &magnitude)
    }

    pub fn absolute_signed(&mut self, input: &[BooleanWire]) -> Result<Vec<BooleanWire>, String> {
        if input.len() < 2 || input.len() > 64 {
            return Err("Boolean signed input width must be between 2 and 64".into());
        }
        let sign = input[input.len() - 1];
        let mut inverted = Vec::new();
        inverted
            .try_reserve_exact(input.len())
            .map_err(|_| "Boolean absolute-value allocation failed")?;
        for &bit in input {
            inverted.push(self.xor(bit, sign)?);
        }
        let mut carry = sign;
        let mut magnitude = Vec::new();
        magnitude
            .try_reserve_exact(input.len())
            .map_err(|_| "Boolean absolute-value allocation failed")?;
        for bit in inverted {
            let output = self.xor(bit, carry)?;
            carry = self.and(bit, carry)?;
            magnitude.push(output);
        }
        Ok(magnitude)
    }

    pub fn conditional_negate(
        &mut self,
        magnitude: &[BooleanWire],
        negate: BooleanWire,
    ) -> Result<Vec<BooleanWire>, String> {
        if magnitude.is_empty() || magnitude.len() > 128 {
            return Err("Boolean conditional-negation width is invalid".into());
        }
        let mut carry = negate;
        let mut output = Vec::new();
        output
            .try_reserve_exact(magnitude.len())
            .map_err(|_| "Boolean conditional-negation allocation failed")?;
        for &bit in magnitude {
            let inverted = self.xor(bit, negate)?;
            output.push(self.xor(inverted, carry)?);
            carry = self.and(inverted, carry)?;
        }
        Ok(output)
    }

    pub fn round_shift_ties_even(
        &mut self,
        magnitude: &[BooleanWire],
        shift: usize,
        output_width: usize,
    ) -> Result<Vec<BooleanWire>, String> {
        let minimum_output_width = magnitude.len().saturating_sub(shift).saturating_add(1);
        if shift == 0
            || shift >= magnitude.len()
            || output_width < minimum_output_width
            || output_width > 128
        {
            return Err("Boolean rounded-shift dimensions are invalid".into());
        }
        let remainder = &magnitude[..shift];
        let half = 1_u128 << (shift - 1);
        let (below_half, equal_half) = self.compare_unsigned_to_constant(remainder, half)?;
        let below_or_equal = self.xor(below_half, equal_half)?;
        let above_half = self.not(below_or_equal)?;
        let quotient_odd = magnitude[shift];
        let tie_and_odd = self.and(equal_half, quotient_odd)?;
        let mut carry = self.xor(above_half, tie_and_odd)?;
        let zero = self.constant(false)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_width)
            .map_err(|_| "Boolean rounded-shift output allocation failed")?;
        for index in 0..output_width {
            let bit = magnitude.get(index + shift).copied().unwrap_or(zero);
            output.push(self.xor(bit, carry)?);
            carry = self.and(bit, carry)?;
        }
        Ok(output)
    }

    pub fn finish(
        self,
        outputs: &[BooleanWire],
    ) -> Result<(BooleanCircuitClient, BooleanCircuitProgram), String> {
        if outputs.is_empty() || outputs.len() > MAX_REFERENCE_OUTPUTS {
            return Err("Boolean circuit output count is outside the reference limit".into());
        }
        for output in outputs {
            self.encoding(*output)?;
        }
        let mut input_encodings = Vec::new();
        input_encodings
            .try_reserve_exact(self.input_count)
            .map_err(|_| "Boolean circuit client-input allocation failed")?;
        input_encodings.extend(
            self.encodings
                .iter()
                .take(self.input_count)
                .map(clone_encoding),
        );
        let mut output_encodings = Vec::new();
        output_encodings
            .try_reserve_exact(outputs.len())
            .map_err(|_| "Boolean circuit client-output allocation failed")?;
        output_encodings.extend(
            outputs
                .iter()
                .map(|output| clone_encoding(&self.encodings[output.index])),
        );
        let mut output_wires = Vec::new();
        output_wires
            .try_reserve_exact(outputs.len())
            .map_err(|_| "Boolean circuit evaluator-output allocation failed")?;
        output_wires.extend(outputs.iter().map(|wire| wire.index));
        let circuit_id = self.garbler.circuit_id;
        let wire_count = self.encodings.len();
        Ok((
            BooleanCircuitClient {
                circuit_id,
                input_encodings,
                output_encodings,
            },
            BooleanCircuitProgram {
                circuit_id,
                input_count: self.input_count,
                wire_count,
                output_wires,
                instructions: self.instructions,
            },
        ))
    }

    fn encoding(&self, wire: BooleanWire) -> Result<&BooleanWireEncoding, String> {
        if wire.circuit_id != self.garbler.circuit_id {
            return Err("Boolean wire belongs to a different circuit builder".into());
        }
        self.encodings
            .get(wire.index)
            .ok_or_else(|| "Boolean circuit wire is out of range".into())
    }

    fn reserve_wire(&mut self, instruction: bool) -> Result<(), String> {
        if self.encodings.len() >= MAX_REFERENCE_WIRES {
            return Err("Boolean reference circuit wire limit exceeded".into());
        }
        self.encodings
            .try_reserve(1)
            .map_err(|_| "Boolean circuit wire allocation failed")?;
        if instruction {
            self.instructions
                .try_reserve(1)
                .map_err(|_| "Boolean circuit instruction allocation failed")?;
        }
        Ok(())
    }
}

pub struct BooleanCircuitClient {
    circuit_id: [u8; CIRCUIT_ID_BYTES],
    input_encodings: Vec<BooleanWireEncoding>,
    output_encodings: Vec<BooleanWireEncoding>,
}

impl BooleanCircuitClient {
    pub fn encode(
        self,
        values: &[bool],
    ) -> Result<(BooleanCircuitInputs, BooleanCircuitDecoder), String> {
        if values.len() != self.input_encodings.len() {
            return Err(format!(
                "Boolean circuit requires {} inputs, received {}",
                self.input_encodings.len(),
                values.len()
            ));
        }
        let labels = self
            .input_encodings
            .iter()
            .zip(values)
            .map(|(encoding, value)| encoding.encode(*value))
            .collect();
        Ok((
            BooleanCircuitInputs {
                circuit_id: self.circuit_id,
                labels,
            },
            BooleanCircuitDecoder {
                circuit_id: self.circuit_id,
                output_encodings: self.output_encodings,
            },
        ))
    }
}

pub struct BooleanCircuitInputs {
    circuit_id: [u8; CIRCUIT_ID_BYTES],
    labels: Vec<BooleanLabel>,
}

pub struct BooleanCircuitOutputs {
    circuit_id: [u8; CIRCUIT_ID_BYTES],
    labels: Vec<BooleanLabel>,
}

pub struct BooleanCircuitDecoder {
    circuit_id: [u8; CIRCUIT_ID_BYTES],
    output_encodings: Vec<BooleanWireEncoding>,
}

impl BooleanCircuitDecoder {
    pub fn decode(self, outputs: BooleanCircuitOutputs) -> Result<Vec<bool>, String> {
        if outputs.circuit_id != self.circuit_id {
            return Err("Boolean circuit output belongs to a different circuit".into());
        }
        if outputs.labels.len() != self.output_encodings.len() {
            return Err("Boolean circuit output count is invalid".into());
        }
        self.output_encodings
            .iter()
            .zip(&outputs.labels)
            .map(|(encoding, label)| encoding.decode(label))
            .collect()
    }
}

pub struct BooleanCircuitProgram {
    circuit_id: [u8; CIRCUIT_ID_BYTES],
    input_count: usize,
    wire_count: usize,
    output_wires: Vec<usize>,
    instructions: Vec<BooleanInstruction>,
}

impl BooleanCircuitProgram {
    pub fn and_gate_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|instruction| matches!(instruction, BooleanInstruction::And { .. }))
            .count()
    }

    pub fn evaluator_ciphertext_bytes(&self) -> Result<u64, String> {
        u64::try_from(self.and_gate_count())
            .map_err(|_| "Boolean AND-gate count exceeds u64")?
            .checked_mul(2 * LABEL_BYTES as u64)
            .ok_or_else(|| "Boolean circuit payload size overflows u64".into())
    }

    pub fn wire_count(&self) -> usize {
        self.wire_count
    }

    pub fn evaluate(self, inputs: BooleanCircuitInputs) -> Result<BooleanCircuitOutputs, String> {
        if inputs.circuit_id != self.circuit_id {
            return Err("Boolean circuit input belongs to a different circuit".into());
        }
        if inputs.labels.len() != self.input_count {
            return Err("Boolean circuit input count is invalid".into());
        }
        let mut wires = Vec::new();
        wires
            .try_reserve_exact(self.wire_count)
            .map_err(|_| "Boolean circuit wire allocation failed")?;
        wires.extend(inputs.labels.into_iter().map(Some));
        for instruction in self.instructions {
            let label = match instruction {
                BooleanInstruction::Constant(label) => label,
                BooleanInstruction::Xor { left, right } => {
                    evaluate_xor(require_label(&wires, left)?, require_label(&wires, right)?)
                }
                BooleanInstruction::Not { input } => evaluate_not(require_label(&wires, input)?),
                BooleanInstruction::And { left, right, gate } => {
                    gate.evaluate(require_label(&wires, left)?, require_label(&wires, right)?)
                }
            };
            wires.push(Some(label));
        }
        let labels = self
            .output_wires
            .iter()
            .map(|wire| require_label(&wires, *wire).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BooleanCircuitOutputs {
            circuit_id: self.circuit_id,
            labels,
        })
    }
}

pub struct RmsNormQ10Client {
    width: usize,
    client: BooleanCircuitClient,
}

pub struct RmsNormQ10Inputs(BooleanCircuitInputs);

pub struct RmsNormQ10Decoder {
    width: usize,
    decoder: BooleanCircuitDecoder,
}

pub struct RmsNormQ10Program {
    width: usize,
    program: BooleanCircuitProgram,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RmsNormQ10CircuitEstimate {
    pub and_gate_count: u64,
    pub evaluator_ciphertext_bytes: u64,
}

pub fn estimate_rms_norm_q10_direct(weights: &[i16]) -> Result<RmsNormQ10CircuitEstimate, String> {
    let row_width = u64::try_from(weights.len()).map_err(|_| "RMSNorm width exceeds u64")?;
    if row_width == 0 || row_width > 8192 {
        return Err("protected Q10 RMSNorm width must be between 1 and 8192".into());
    }
    let input_bits = 16_u64;
    let fractional_bits = 30_u64;
    let sum_bits = 44_u64;
    let denominator_bits = 64_u64;
    let numerator = u128::from(row_width)
        .checked_mul(1_000_000)
        .and_then(|value| value.checked_shl(60))
        .ok_or("RMSNorm estimate numerator overflows u128")?;
    let quotient_bits = u64::from(128 - numerator.leading_zeros());

    let square_input = input_bits + 5 * input_bits * input_bits;
    let sum_add = 2 * sum_bits;
    let denominator =
        u64::from(1_000_000_u64.count_ones()) * 2 * denominator_bits + 2 * denominator_bits;
    let division = quotient_bits * 3 * (denominator_bits + 1);
    let root_search = fractional_bits * (5 * fractional_bits * fractional_bits + 3 * quotient_bits);
    let rounded_root = 5 * (fractional_bits + 1) * (fractional_bits + 1)
        + denominator_bits * (6 * fractional_bits + 134)
        + (2 * fractional_bits + 66)
        + u64::from((4 * numerator).count_ones())
        + 1
        + fractional_bits;
    let normalizer = denominator + division + root_search + rounded_root;

    let variable_product = fractional_bits * (6 * input_bits + 2 * fractional_bits);
    let rounded_output = fractional_bits + 1 + 1 + 64 + 64;
    let mut outputs = 0_u64;
    for &weight in weights {
        let public_product = u64::from(weight.unsigned_abs().count_ones()) * 4 * input_bits;
        outputs = outputs
            .checked_add(input_bits + public_product + variable_product + rounded_output)
            .ok_or("RMSNorm estimate gate count overflows u64")?;
    }
    let and_gate_count = row_width
        .checked_mul(square_input + sum_add)
        .and_then(|value| value.checked_add(normalizer))
        .and_then(|value| value.checked_add(outputs))
        .ok_or("RMSNorm estimate gate count overflows u64")?;
    let evaluator_ciphertext_bytes = and_gate_count
        .checked_mul(2 * LABEL_BYTES as u64)
        .ok_or("RMSNorm estimate payload bytes overflow u64")?;
    Ok(RmsNormQ10CircuitEstimate {
        and_gate_count,
        evaluator_ciphertext_bytes,
    })
}

pub fn prepare_rms_norm_q10_direct(
    weights: &[i16],
) -> Result<(RmsNormQ10Client, RmsNormQ10Program), String> {
    if weights.is_empty() || weights.len() > 8192 {
        return Err("protected Q10 RMSNorm width must be between 1 and 8192".into());
    }
    let mut builder = BooleanCircuitBuilder::new()?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(weights.len())
        .map_err(|_| "protected Q10 RMSNorm input allocation failed")?;
    for _ in weights {
        rows.push(builder.input_word(16)?);
    }
    let weight_values = weights.iter().copied().map(i64::from).collect::<Vec<_>>();
    let outputs = builder.rms_norm_direct_q(&rows, &weight_values, 30)?;
    let flattened_outputs = outputs.into_iter().flatten().collect::<Vec<_>>();
    let (client, program) = builder.finish(&flattened_outputs)?;
    let width = weights.len();
    Ok((
        RmsNormQ10Client { width, client },
        RmsNormQ10Program { width, program },
    ))
}

impl RmsNormQ10Client {
    pub fn encode(self, input: &[i16]) -> Result<(RmsNormQ10Inputs, RmsNormQ10Decoder), String> {
        if input.len() != self.width {
            return Err(format!(
                "protected Q10 RMSNorm requires {} values, received {}",
                self.width,
                input.len()
            ));
        }
        let mut bits = Vec::new();
        bits.try_reserve_exact(self.width * 16)
            .map_err(|_| "protected Q10 RMSNorm input-bit allocation failed")?;
        for value in input {
            let encoded = u16::from_le_bytes(value.to_le_bytes());
            bits.extend((0..16).map(|bit| encoded & (1_u16 << bit) != 0));
        }
        let (inputs, decoder) = self.client.encode(&bits)?;
        Ok((
            RmsNormQ10Inputs(inputs),
            RmsNormQ10Decoder {
                width: self.width,
                decoder,
            },
        ))
    }
}

impl RmsNormQ10Program {
    pub fn and_gate_count(&self) -> usize {
        self.program.and_gate_count()
    }

    pub fn evaluator_ciphertext_bytes(&self) -> Result<u64, String> {
        self.program.evaluator_ciphertext_bytes()
    }

    pub fn issuance_id(&self) -> [u8; CIRCUIT_ID_BYTES] {
        self.program.circuit_id
    }

    pub fn evaluate(self, inputs: RmsNormQ10Inputs) -> Result<BooleanCircuitOutputs, String> {
        if self.width == 0 {
            return Err("protected Q10 RMSNorm program width is invalid".into());
        }
        self.program.evaluate(inputs.0)
    }
}

impl RmsNormQ10Decoder {
    pub fn decode(self, outputs: BooleanCircuitOutputs) -> Result<Vec<i32>, String> {
        let bits = self.decoder.decode(outputs)?;
        if bits.len() != self.width * 64 {
            return Err("protected Q10 RMSNorm output width is invalid".into());
        }
        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(self.width)
            .map_err(|_| "protected Q10 RMSNorm output allocation failed")?;
        for value_bits in bits.chunks_exact(64) {
            let mut encoded = 0_u64;
            for (bit, set) in value_bits.iter().enumerate() {
                if *set {
                    encoded |= 1_u64 << bit;
                }
            }
            let signed = i64::from_le_bytes(encoded.to_le_bytes());
            decoded.push(
                i32::try_from(signed).map_err(|_| "protected Q10 RMSNorm output exceeds i32")?,
            );
        }
        Ok(decoded)
    }
}

fn require_label(wires: &[Option<BooleanLabel>], wire: usize) -> Result<&BooleanLabel, String> {
    wires
        .get(wire)
        .and_then(Option::as_ref)
        .ok_or_else(|| "Boolean circuit program references an unavailable wire".into())
}

fn clone_encoding(encoding: &BooleanWireEncoding) -> BooleanWireEncoding {
    BooleanWireEncoding {
        zero: encoding.zero,
        delta: encoding.delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_gate_and_xor_not_cover_all_truth_assignments() {
        for left_value in [false, true] {
            for right_value in [false, true] {
                let mut garbler = BooleanGarbler::new().unwrap();
                let left = garbler.wire().unwrap();
                let right = garbler.wire().unwrap();
                let (and_gate, and_output) = garbler.garble_and(&left, &right).unwrap();
                let xor_output = garbler.xor(&left, &right).unwrap();
                let not_output = garbler.not(&left).unwrap();
                let left_label = left.encode(left_value);
                let right_label = right.encode(right_value);

                let and_label = and_gate.evaluate(&left_label, &right_label);
                assert_eq!(
                    and_output.decode(&and_label).unwrap(),
                    left_value & right_value
                );
                let xor_label = evaluate_xor(&left_label, &right_label);
                assert_eq!(
                    xor_output.decode(&xor_label).unwrap(),
                    left_value ^ right_value
                );
                let not_label = evaluate_not(&left_label);
                assert_eq!(not_output.decode(&not_label).unwrap(), !left_value);
            }
        }
    }

    #[test]
    fn circuit_labels_cannot_cross_garblers() {
        let first = BooleanGarbler::new().unwrap();
        let second = BooleanGarbler::new().unwrap();
        let first_wire = first.wire().unwrap();
        let second_wire = second.wire().unwrap();
        assert!(first.xor(&first_wire, &second_wire).is_err());
        assert!(first_wire.decode(&second_wire.encode(false)).is_err());
    }

    #[test]
    fn every_and_gate_has_exactly_two_ciphertexts() {
        assert_eq!(2 * LABEL_BYTES, 32);
    }

    #[test]
    fn fixed_key_hash_matches_independent_domain_vector() {
        assert_eq!(
            fixed_key_hash(&[0_u8; LABEL_BYTES], &[0_u8; CIRCUIT_ID_BYTES], 0, 0),
            [
                0x59, 0x3b, 0x7b, 0xc9, 0x46, 0xec, 0x06, 0x35, 0x7d, 0x1f, 0xcc, 0x35, 0xf8, 0xf1,
                0x3f, 0xa9,
            ]
        );
        assert_ne!(
            fixed_key_hash(&[0_u8; LABEL_BYTES], &[0_u8; CIRCUIT_ID_BYTES], 0, 0),
            fixed_key_hash(&[0_u8; LABEL_BYTES], &[0_u8; CIRCUIT_ID_BYTES], 0, 1)
        );
        assert_eq!(
            double_gf128([0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x87]
        );
    }

    #[test]
    fn fixed_key_hash_nonzero_vector_locks_doubling_and_tweak_encoding() {
        let mut circuit_id = [0_u8; CIRCUIT_ID_BYTES];
        circuit_id[..8].copy_from_slice(&[0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]);
        let label = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        assert_eq!(
            fixed_key_hash(&label, &circuit_id, 5, 1),
            [
                0x68, 0x44, 0x2d, 0xbb, 0x11, 0xd7, 0x26, 0x3f, 0x0a, 0x6b, 0x12, 0xe8, 0x38, 0xab,
                0xd3, 0x67,
            ]
        );
    }

    #[test]
    fn consuming_circuit_api_binds_inputs_outputs_and_payload_accounting() {
        let mut builder = BooleanCircuitBuilder::new().unwrap();
        let left = builder.input().unwrap();
        let right = builder.input().unwrap();
        let and = builder.and(left, right).unwrap();
        let xor = builder.xor(left, right).unwrap();
        let not = builder.not(left).unwrap();
        let (client, program) = builder.finish(&[and, xor, not]).unwrap();
        assert_eq!(program.and_gate_count(), 1);
        assert_eq!(program.evaluator_ciphertext_bytes().unwrap(), 32);
        let (inputs, decoder) = client.encode(&[true, false]).unwrap();
        let outputs = program.evaluate(inputs).unwrap();
        assert_eq!(decoder.decode(outputs).unwrap(), vec![false, true, false]);
    }

    #[test]
    fn consuming_circuit_api_rejects_cross_circuit_inputs() {
        let mut first = BooleanCircuitBuilder::new().unwrap();
        let first_input = first.input().unwrap();
        let (first_client, first_program) = first.finish(&[first_input]).unwrap();
        let mut second = BooleanCircuitBuilder::new().unwrap();
        let second_input = second.input().unwrap();
        let (second_client, second_program) = second.finish(&[second_input]).unwrap();
        let (first_inputs, _) = first_client.encode(&[true]).unwrap();
        let (_, second_decoder) = second_client.encode(&[true]).unwrap();
        assert!(second_program.evaluate(first_inputs).is_err());

        let mut third = BooleanCircuitBuilder::new().unwrap();
        let third_input = third.input().unwrap();
        let (third_client, third_program) = third.finish(&[third_input]).unwrap();
        let (third_inputs, _) = third_client.encode(&[false]).unwrap();
        let third_outputs = third_program.evaluate(third_inputs).unwrap();
        assert!(second_decoder.decode(third_outputs).is_err());
        drop(first_program);
    }

    #[test]
    fn signed_square_matches_every_i8_input() {
        for value in i8::MIN..=i8::MAX {
            let mut builder = BooleanCircuitBuilder::new().unwrap();
            let input = builder.input_word(8).unwrap();
            let square = builder.square_signed(&input).unwrap();
            let (client, program) = builder.finish(&square).unwrap();
            let encoded = (value as u8) as u128;
            let input_bits = (0..8)
                .map(|shift| encoded & (1_u128 << shift) != 0)
                .collect::<Vec<_>>();
            let (inputs, decoder) = client.encode(&input_bits).unwrap();
            let output_bits = decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
            let decoded = output_bits
                .iter()
                .enumerate()
                .fold(0_u32, |result, (shift, bit)| {
                    result | (u32::from(*bit) << shift)
                });
            assert_eq!(decoded, i32::from(value).unsigned_abs().pow(2));
        }
    }

    #[test]
    fn reciprocal_square_root_matches_small_exact_profile_cases() {
        for sum_squares in [0_u8, 1, 4, 16, 64, 255] {
            let mut builder = BooleanCircuitBuilder::new().unwrap();
            let sum = builder.input_word(8).unwrap();
            let normalizer = builder.reciprocal_sqrt_q(&sum, 1, 5).unwrap();
            let (client, program) = builder.finish(&normalizer).unwrap();
            let bits = (0..8)
                .map(|shift| sum_squares & (1_u8 << shift) != 0)
                .collect::<Vec<_>>();
            let (inputs, decoder) = client.encode(&bits).unwrap();
            let output = decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
            let decoded = output
                .iter()
                .enumerate()
                .fold(0_u32, |value, (shift, bit)| {
                    value | (u32::from(*bit) << shift)
                });
            let denominator = f64::from(sum_squares) * 1_000_000.0 + 1_048_576.0;
            let expected = (32.0 * (1_000_000.0 / denominator).sqrt()).round() as u32;
            assert_eq!(decoded, expected, "sum_squares={sum_squares}");
        }
    }

    #[test]
    fn protected_rms_norm_tiny_row_matches_integer_contract() {
        let values = [3_i64, -4];
        let weights = [3_i64, -2];
        let mut builder = BooleanCircuitBuilder::new().unwrap();
        let rows = (0..values.len())
            .map(|_| builder.input_word(4).unwrap())
            .collect::<Vec<_>>();
        let outputs = builder.rms_norm_direct_q(&rows, &weights, 5).unwrap();
        let flat_outputs = outputs.into_iter().flatten().collect::<Vec<_>>();
        let (client, program) = builder.finish(&flat_outputs).unwrap();
        let input_bits = values
            .iter()
            .flat_map(|value| {
                let encoded = (*value as i8 as u8) & 0x0f;
                (0..4).map(move |shift| encoded & (1_u8 << shift) != 0)
            })
            .collect::<Vec<_>>();
        let (inputs, decoder) = client.encode(&input_bits).unwrap();
        let bits = decoder.decode(program.evaluate(inputs).unwrap()).unwrap();
        let decoded = bits
            .chunks_exact(64)
            .map(|word| {
                word.iter().enumerate().fold(0_u64, |value, (shift, bit)| {
                    value | (u64::from(*bit) << shift)
                }) as i64
            })
            .collect::<Vec<_>>();

        let sum_squares = values.iter().map(|value| value * value).sum::<i64>();
        let denominator = sum_squares * 1_000_000 + 2 * 1_048_576;
        let normalizer = (32.0 * (2_000_000.0 / denominator as f64).sqrt()).round() as i64;
        let expected = values
            .iter()
            .zip(weights)
            .map(|(value, weight)| {
                let product = value * weight * normalizer;
                let magnitude = product.unsigned_abs();
                let quotient = magnitude / 32;
                let remainder = magnitude % 32;
                let rounded =
                    quotient + u64::from(remainder > 16 || (remainder == 16 && quotient & 1 == 1));
                if product.is_negative() {
                    -(rounded as i64)
                } else {
                    rounded as i64
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn protected_q10_rms_norm_matches_core_contract() {
        let input = [1536_i16, -768_i16];
        let weights = [1024_i16, 896_i16];
        let expected = pllm_core::rms_norm::rms_norm_q10_direct(&input, &weights).unwrap();
        let estimate = estimate_rms_norm_q10_direct(&weights).unwrap();
        let (client, program) = prepare_rms_norm_q10_direct(&weights).unwrap();
        assert_eq!(program.and_gate_count(), 196_910);
        assert_eq!(
            estimate.and_gate_count,
            u64::try_from(program.and_gate_count()).unwrap()
        );
        assert_eq!(
            program.evaluator_ciphertext_bytes().unwrap(),
            u64::try_from(program.and_gate_count()).unwrap() * 32
        );
        let (inputs, decoder) = client.encode(&input).unwrap();
        let outputs = program.evaluate(inputs).unwrap();
        assert_eq!(decoder.decode(outputs).unwrap(), expected);
    }

    #[test]
    fn q10_rms_norm_estimate_handles_maximum_width_without_building() {
        let weights = vec![1024_i16; 8192];
        let estimate = estimate_rms_norm_q10_direct(&weights).unwrap();
        assert_eq!(estimate.and_gate_count, 51_829_962);
        assert_eq!(
            estimate.evaluator_ciphertext_bytes,
            estimate.and_gate_count * 32
        );
    }
}
