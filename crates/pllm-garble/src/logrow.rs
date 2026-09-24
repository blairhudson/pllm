//! Bounded, one-use garbled lookup adapted from Heath, Kolesnikov and Ng's
//! LogRow (Eurocrypt 2024), with Heath and Kolesnikov's 2022 one-hot garbling.
//! This in-process reference is not a reviewed or deployed 2PC protocol.
//!
//! The garbler owns a table and zero labels; the evaluator receives only
//! punctured one-hot seeds, a masked table, and one-use active input labels.
//! The masked index is revealed to the evaluator, never the index or mask.

use crate::boolean::fixed_key_hash;
use pllm_core::{logrow_numeric::ScaledSiluQ7Profile, CompactQ7Profile};
use zeroize::{Zeroize, Zeroizing};

type Block = [u8; 16];
const MAX_BITS: usize = 9;
const CIRCUIT_ID_BYTES: usize = 32;
const TREE_DOMAIN: u64 = 1 << 48;
const TREE_ROW_DOMAIN: u64 = 2 << 48;
const MASK_DOMAIN: u64 = 3 << 48;
const MASK_ROW_DOMAIN: u64 = 4 << 48;
/// Evaluator material for the fixed 512-entry, signed-9-bit Compact table.
/// Tree: 17 blocks; mask rows: 81 blocks; masked table: 512 * 9 bits.
pub const COMPACT_Q7_LOGROW_MATERIAL_BYTES: usize = (17 + 81) * 16 + 512 * 9 / 8;

/// Client-only input encodings; consuming encoding burns this reference.
pub struct LogRowClient {
    id: [u8; CIRCUIT_ID_BYTES],
    zero: Vec<Block>,
    delta: Block,
    output_zero: Vec<Block>,
    input_bits: usize,
}

/// Opaque input labels; consuming evaluation burns both labels and material.
pub struct LogRowInputs {
    id: [u8; CIRCUIT_ID_BYTES],
    labels: Vec<Block>,
}

pub struct LogRowDecoder {
    id: [u8; CIRCUIT_ID_BYTES],
    zero: Vec<Block>,
    delta: Block,
}

pub struct LogRowOutputs {
    id: [u8; CIRCUIT_ID_BYTES],
    labels: Vec<Block>,
}

struct MaskLevel {
    depth: usize,
    rows: Vec<Block>,
}

pub struct LogRowProgram {
    id: [u8; CIRCUIT_ID_BYTES],
    input_bits: usize,
    output_bits: usize,
    tree_rows: Vec<[Block; 2]>,
    tree_final: Block,
    mask_levels: Vec<MaskLevel>,
    masked_table: Vec<u8>,
}

/// Client-owned, digest-bound signed-Q7 input / signed-9-bit output table.
pub struct SignedQ7LogRowClient {
    profile_digest: [u8; 32],
    client: LogRowClient,
}

pub struct SignedQ7LogRowProgram {
    profile_digest: [u8; 32],
    program: LogRowProgram,
}

pub struct SignedQ7LogRowDecoder {
    profile_digest: [u8; 32],
    decoder: LogRowDecoder,
}

pub type CompactQ7LogRowClient = SignedQ7LogRowClient;
pub type CompactQ7LogRowProgram = SignedQ7LogRowProgram;
pub type CompactQ7LogRowDecoder = SignedQ7LogRowDecoder;

/// This reference evaluates the **fitted table**, not Compact's polynomial
/// construction. It is in-process, one-use, and not an Experiment component.
pub fn prepare_compact_q7_logrow(
    profile: &CompactQ7Profile,
) -> Result<(CompactQ7LogRowClient, CompactQ7LogRowProgram), String> {
    let table = compact_q7_table(profile)?;
    prepare_compact_q7_logrow_table(profile.digest(), &table)
}

/// Prepare bounded independent elements without fitting/evaluating the table
/// once per element. The complete admitted body is checked by the caller.
pub fn prepare_compact_q7_logrow_elements(
    profile: &CompactQ7Profile,
    count: usize,
) -> Result<Vec<(CompactQ7LogRowClient, CompactQ7LogRowProgram)>, String> {
    let table = compact_q7_table(profile)?;
    prepare_signed_q7_logrow_table_elements(profile.digest(), &table, count)
}

/// A separate public-range SiLU table, with the same one-use LogRow material
/// and signed-Q7 input codes. This does not publish a compiler session.
pub fn prepare_scaled_silu_q7_logrow_elements(
    profile: &ScaledSiluQ7Profile,
    count: usize,
) -> Result<Vec<(SignedQ7LogRowClient, SignedQ7LogRowProgram)>, String> {
    let mut table = Zeroizing::new(vec![0_u16; 512]);
    for (offset, code) in (-128..=128).enumerate() {
        let output = profile
            .encoded_output(code)
            .map_err(|error| error.to_string())?;
        table[offset] = u16::from_le_bytes(output.to_le_bytes()) & 0x01ff;
    }
    prepare_signed_q7_logrow_table_elements(profile.digest(), &table, count)
}

fn prepare_signed_q7_logrow_table_elements(
    profile_digest: [u8; 32],
    table: &[u16],
    count: usize,
) -> Result<Vec<(SignedQ7LogRowClient, SignedQ7LogRowProgram)>, String> {
    if count == 0 || count > (64 * 1024 * 1024) / COMPACT_Q7_LOGROW_MATERIAL_BYTES {
        return Err("signed Q7 LogRow tensor body exceeds 64 MiB".into());
    }
    let mut elements = Vec::new();
    elements
        .try_reserve_exact(count)
        .map_err(|_| "Compact Q7 LogRow element allocation failed")?;
    for _ in 0..count {
        elements.push(prepare_compact_q7_logrow_table(profile_digest, table)?);
    }
    Ok(elements)
}

fn compact_q7_table(profile: &CompactQ7Profile) -> Result<Zeroizing<Vec<u16>>, String> {
    let mut table = Zeroizing::new(vec![0_u16; 512]);
    for (offset, value) in (-128..=128).enumerate() {
        let output = profile.evaluate(value).map_err(|error| error.to_string())?;
        if !(-256..=255).contains(&output) {
            return Err("Compact Q7 LogRow output does not fit signed 9-bit".into());
        }
        table[offset] = u16::from_le_bytes(output.to_le_bytes()) & 0x01ff;
    }
    Ok(table)
}

fn prepare_compact_q7_logrow_table(
    profile_digest: [u8; 32],
    table: &[u16],
) -> Result<(SignedQ7LogRowClient, SignedQ7LogRowProgram), String> {
    let (client, program) = prepare_logrow(9, 9, table)?;
    if program.evaluator_material_bytes() != COMPACT_Q7_LOGROW_MATERIAL_BYTES {
        return Err("Compact Q7 LogRow material layout changed".into());
    }
    Ok((
        SignedQ7LogRowClient {
            profile_digest,
            client,
        },
        SignedQ7LogRowProgram {
            profile_digest,
            program,
        },
    ))
}

impl SignedQ7LogRowClient {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    pub fn encode(self, value: i16) -> Result<(LogRowInputs, CompactQ7LogRowDecoder), String> {
        if !(-128..=128).contains(&value) {
            return Err("signed Q7 LogRow input is outside [-128, 128]".into());
        }
        let (labels, decoder) = self.client.encode((value + 128) as u16)?;
        Ok((
            labels,
            SignedQ7LogRowDecoder {
                profile_digest: self.profile_digest,
                decoder,
            },
        ))
    }
}

impl SignedQ7LogRowProgram {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    pub fn evaluator_material_bytes(&self) -> usize {
        self.program.evaluator_material_bytes()
    }

    pub const fn issuance_id(&self) -> [u8; CIRCUIT_ID_BYTES] {
        self.program.issuance_id()
    }

    pub fn evaluate(self, inputs: LogRowInputs) -> Result<LogRowOutputs, String> {
        self.program.evaluate(inputs)
    }
}

impl SignedQ7LogRowDecoder {
    pub const fn profile_digest(&self) -> [u8; 32] {
        self.profile_digest
    }

    pub fn decode(self, outputs: LogRowOutputs) -> Result<i16, String> {
        let raw = self.decoder.decode(outputs)? as i16;
        Ok(if raw & 0x0100 != 0 { raw - 512 } else { raw })
    }
}

fn xor(left: Block, right: Block) -> Block {
    let mut result = [0; 16];
    for (out, (a, b)) in result.iter_mut().zip(left.iter().zip(right.iter())) {
        *out = a ^ b;
    }
    result
}

fn hash(seed: &Block, id: &[u8; CIRCUIT_ID_BYTES], tweak: u64, half: u8) -> Block {
    fixed_key_hash(seed, id, tweak, half)
}

fn mask_bits(seed: &Block, id: &[u8; CIRCUIT_ID_BYTES], depth: usize, count: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(count);
    for chunk in 0..count.div_ceil(128) {
        let block = hash(
            seed,
            id,
            MASK_DOMAIN | ((depth as u64) << 32) | chunk as u64,
            0,
        );
        for bit in 0..128 {
            if bits.len() == count {
                break;
            }
            bits.push((block[bit / 8] >> (bit % 8)) & 1);
        }
    }
    bits
}

fn folded(hot: &[Block]) -> Vec<Block> {
    let half = hot.len() / 2;
    (0..half).map(|i| xor(hot[i], hot[half + i])).collect()
}

fn mask_sum(hot: &[Block], bits: &[u8], output_bits: usize, bit: usize) -> Block {
    let mut sum = [0; 16];
    for (row, seed) in hot.iter().enumerate() {
        if bits[row * output_bits + bit] != 0 {
            sum = xor(sum, *seed);
        }
    }
    sum
}

/// Prepare a complete n-bit to m-bit private LUT (n,m in 1..=9). All table
/// entries, including padded domain entries, are supplied by the garbler.
pub fn prepare_logrow(
    input_bits: usize,
    output_bits: usize,
    table: &[u16],
) -> Result<(LogRowClient, LogRowProgram), String> {
    if !(1..=MAX_BITS).contains(&input_bits) || !(1..=MAX_BITS).contains(&output_bits) {
        return Err("LogRow input/output widths must be between 1 and 9".into());
    }
    let rows = 1 << input_bits;
    if table.len() != rows
        || table
            .iter()
            .any(|&entry| usize::from(entry) >= 1 << output_bits)
    {
        return Err("LogRow table shape or output width is invalid".into());
    }
    let mut id = [0; CIRCUIT_ID_BYTES];
    getrandom::fill(&mut id).map_err(|error| error.to_string())?;
    let mut delta = [0; 16];
    getrandom::fill(&mut delta).map_err(|error| error.to_string())?;
    delta[15] |= 1;
    let mut zero = Zeroizing::new(Vec::with_capacity(input_bits));
    let mut alpha = 0_usize;
    for bit in 0..input_bits {
        let mut label = [0; 16];
        getrandom::fill(&mut label).map_err(|error| error.to_string())?;
        if label[15] & 1 == 1 {
            alpha |= 1 << bit;
        }
        zero.push(label);
    }
    // Zero labels for the masked input x = a XOR alpha all have color zero.
    let masked_zero = Zeroizing::new(
        zero.iter()
            .enumerate()
            .map(|(bit, &label)| {
                if alpha & (1 << bit) != 0 {
                    xor(label, delta)
                } else {
                    label
                }
            })
            .collect::<Vec<_>>(),
    );

    // One-hot punctured tree: the evaluator learns all leaves except the
    // masked index. The 2022 two-row construction uses 2(n-1)+1 blocks.
    let top = input_bits - 1;
    let mut leaves = Zeroizing::new(vec![xor(masked_zero[top], delta), masked_zero[top]]);
    let mut tree_rows = Vec::with_capacity(input_bits - 1);
    for depth in 1..input_bits {
        let bit = top - depth;
        let mut children = Vec::with_capacity(leaves.len() * 2);
        for (index, seed) in leaves.iter().enumerate() {
            for branch in 0..2 {
                children.push(hash(
                    seed,
                    &id,
                    TREE_DOMAIN | ((depth as u64) << 32) | index as u64,
                    branch,
                ));
            }
        }
        let even = children.iter().step_by(2).copied().fold([0; 16], xor);
        let odd = children
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .fold([0; 16], xor);
        let tag = TREE_ROW_DOMAIN | depth as u64;
        tree_rows.push([
            xor(even, hash(&xor(masked_zero[bit], delta), &id, tag, 0)),
            xor(odd, hash(&masked_zero[bit], &id, tag, 1)),
        ]);
        leaves = Zeroizing::new(children);
    }
    let tree_final = xor(leaves.iter().copied().fold([0; 16], xor), delta);

    let mut hot = leaves;
    let full_leaves = Zeroizing::new((*hot).clone());
    let mut mask_levels = Vec::with_capacity(input_bits);
    let mut random_table = Zeroizing::new(vec![0_u16; rows]);
    let mut output_zero = Zeroizing::new(vec![[0; 16]; output_bits]);
    for depth in (1..=input_bits).rev() {
        let half = 1 << (depth - 1);
        let bit = depth - 1;
        let left = Zeroizing::new(mask_bits(&masked_zero[bit], &id, depth, half * output_bits));
        let right = Zeroizing::new(mask_bits(
            &xor(masked_zero[bit], delta),
            &id,
            depth,
            half * output_bits,
        ));
        let mut message = Vec::with_capacity(output_bits);
        for output in 0..output_bits {
            let left_sum = mask_sum(&hot[..half], &left, output_bits, output);
            let right_sum = mask_sum(&hot[half..], &right, output_bits, output);
            let tag = MASK_ROW_DOMAIN | ((depth as u64) << 32) | output as u64;
            let z = xor(hash(&masked_zero[bit], &id, tag, 0), right_sum);
            message.push(xor(
                xor(hash(&xor(masked_zero[bit], delta), &id, tag, 1), left_sum),
                z,
            ));
            output_zero[output] = xor(output_zero[output], xor(xor(left_sum, right_sum), z));
        }
        for (index, entry) in random_table.iter_mut().enumerate() {
            let offset = index % (half * 2);
            let (bits, row) = if offset < half {
                (&left, offset)
            } else {
                (&right, offset - half)
            };
            for output in 0..output_bits {
                if bits[row * output_bits + output] != 0 {
                    *entry ^= 1 << output;
                }
            }
        }
        mask_levels.push(MaskLevel {
            depth,
            rows: message,
        });
        if depth > 1 {
            hot = Zeroizing::new(folded(&hot));
        }
    }
    let mut base = [0; 2];
    getrandom::fill(&mut base).map_err(|error| error.to_string())?;
    let mut base_value = u16::from_le_bytes(base) & ((1 << output_bits) - 1);
    base.zeroize();
    for (index, entry) in random_table.iter_mut().enumerate() {
        *entry ^= base_value;
        *entry ^= table[index ^ alpha];
    }
    for (output, zero_label) in output_zero.iter_mut().enumerate() {
        if base_value & (1 << output) != 0 {
            *zero_label = xor(*zero_label, delta);
        }
    }
    let mut masked_table = vec![0_u8; (rows * output_bits).div_ceil(8)];
    for (index, &entry) in random_table.iter().enumerate() {
        for output in 0..output_bits {
            if entry & (1 << output) != 0 {
                let offset = index * output_bits + output;
                masked_table[offset / 8] |= 1 << (offset % 8);
            }
        }
    }
    for (index, &entry) in random_table.iter().enumerate() {
        for (output, zero_label) in output_zero.iter_mut().enumerate() {
            if entry & (1 << output) != 0 {
                *zero_label = xor(*zero_label, full_leaves[index]);
            }
        }
    }
    base_value.zeroize();
    Ok((
        LogRowClient {
            id,
            zero: std::mem::take(&mut *zero),
            delta,
            output_zero: std::mem::take(&mut *output_zero),
            input_bits,
        },
        LogRowProgram {
            id,
            input_bits,
            output_bits,
            tree_rows,
            tree_final,
            mask_levels,
            masked_table,
        },
    ))
}

impl LogRowClient {
    pub fn encode(self, index: u16) -> Result<(LogRowInputs, LogRowDecoder), String> {
        if usize::from(index) >= 1 << self.input_bits {
            return Err("LogRow input index is out of range".into());
        }
        let labels = self
            .zero
            .iter()
            .enumerate()
            .map(|(bit, &zero)| {
                if usize::from(index) & (1 << bit) == 0 {
                    zero
                } else {
                    xor(zero, self.delta)
                }
            })
            .collect();
        Ok((
            LogRowInputs {
                id: self.id,
                labels,
            },
            LogRowDecoder {
                id: self.id,
                zero: self.output_zero.clone(),
                delta: self.delta,
            },
        ))
    }
}

impl LogRowProgram {
    /// Fresh circuit identity binds this one-use material to its client input.
    pub const fn issuance_id(&self) -> [u8; CIRCUIT_ID_BYTES] {
        self.id
    }

    /// Includes tree and random-function rows, masked table, and final tree
    /// block; excludes input labels, framing, and local workspace.
    pub fn evaluator_material_bytes(&self) -> usize {
        (self.tree_rows.len() * 2 + 1 + self.input_bits * self.output_bits) * 16
            + self.masked_table.len()
    }

    pub fn evaluate(self, input: LogRowInputs) -> Result<LogRowOutputs, String> {
        if input.id != self.id || input.labels.len() != self.input_bits {
            return Err("LogRow input labels belong to another circuit".into());
        }
        let index = input
            .labels
            .iter()
            .enumerate()
            .fold(0_usize, |index, (bit, label)| {
                index | (usize::from(label[15] & 1) << bit)
            });
        let top = self.input_bits - 1;
        let mut hot = Zeroizing::new(vec![None; 2]);
        hot[1 - (index >> top)] = Some(input.labels[top]);
        for (depth, rows) in self.tree_rows.iter().enumerate() {
            let bit = top - depth - 1;
            let mut expanded = Zeroizing::new(vec![None; hot.len() * 2]);
            for (parent, known) in hot.iter().enumerate() {
                if let Some(seed) = known {
                    for branch in 0..2 {
                        expanded[2 * parent + branch] = Some(hash(
                            seed,
                            &self.id,
                            TREE_DOMAIN | (((depth + 1) as u64) << 32) | parent as u64,
                            branch as u8,
                        ));
                    }
                }
            }
            let missing_offpath = (index >> bit) ^ 1;
            let parity = missing_offpath & 1;
            let sum = expanded
                .iter()
                .enumerate()
                .filter(|(i, _)| i & 1 == parity)
                .filter_map(|(_, seed)| *seed)
                .fold([0; 16], xor);
            let tag = TREE_ROW_DOMAIN | (depth + 1) as u64;
            expanded[missing_offpath] = Some(xor(
                xor(
                    rows[parity],
                    hash(&input.labels[bit], &self.id, tag, parity as u8),
                ),
                sum,
            ));
            hot = expanded;
        }
        let active = index;
        let known = hot.iter().filter_map(|seed| *seed).fold([0; 16], xor);
        hot[active] = Some(xor(self.tree_final, known));
        let mut hot = Zeroizing::new(
            hot.iter()
                .copied()
                .collect::<Option<Vec<_>>>()
                .ok_or("LogRow one-hot tree is incomplete")?,
        );
        let full_leaves = Zeroizing::new((*hot).clone());
        let mut output = Zeroizing::new(vec![[0; 16]; self.output_bits]);
        for level in &self.mask_levels {
            let depth = level.depth;
            let half = 1 << (depth - 1);
            let bit = depth - 1;
            let side = (index >> bit) & 1;
            let bits = Zeroizing::new(mask_bits(
                &input.labels[bit],
                &self.id,
                depth,
                half * self.output_bits,
            ));
            for (column, label) in output.iter_mut().enumerate() {
                let slice = if side == 0 {
                    &hot[..half]
                } else {
                    &hot[half..]
                };
                let sum = mask_sum(slice, &bits, self.output_bits, column);
                let tag = MASK_ROW_DOMAIN | ((depth as u64) << 32) | column as u64;
                let row = if side == 0 {
                    [0; 16]
                } else {
                    level.rows[column]
                };
                *label = xor(
                    *label,
                    xor(
                        xor(hash(&input.labels[bit], &self.id, tag, side as u8), row),
                        sum,
                    ),
                );
            }
            if depth > 1 {
                hot = Zeroizing::new(folded(&hot));
            }
        }
        for (index, &seed) in full_leaves.iter().enumerate() {
            for (column, label) in output.iter_mut().enumerate() {
                let offset = index * self.output_bits + column;
                if self.masked_table[offset / 8] & (1 << (offset % 8)) != 0 {
                    *label = xor(*label, seed);
                }
            }
        }
        Ok(LogRowOutputs {
            id: self.id,
            labels: std::mem::take(&mut *output),
        })
    }
}

impl LogRowDecoder {
    pub fn decode(self, outputs: LogRowOutputs) -> Result<u16, String> {
        if outputs.id != self.id || outputs.labels.len() != self.zero.len() {
            return Err("LogRow outputs belong to another circuit".into());
        }
        let mut result = 0_u16;
        for (bit, (&label, &zero)) in outputs.labels.iter().zip(&self.zero).enumerate() {
            if label == zero {
                continue;
            }
            if label != xor(zero, self.delta) {
                return Err("LogRow output label is invalid".into());
            }
            result |= 1 << bit;
        }
        Ok(result)
    }
}

impl Drop for LogRowClient {
    fn drop(&mut self) {
        self.zero.zeroize();
        self.output_zero.zeroize();
        self.delta.zeroize();
    }
}
impl Drop for LogRowDecoder {
    fn drop(&mut self) {
        self.zero.zeroize();
        self.delta.zeroize();
    }
}
impl Drop for LogRowInputs {
    fn drop(&mut self) {
        self.labels.zeroize();
    }
}
impl Drop for LogRowOutputs {
    fn drop(&mut self) {
        self.labels.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pllm_core::fit_compact_silu_q7;

    #[test]
    fn every_small_input_with_mixed_secret_tables() {
        for input_bits in 1..=4 {
            let rows = 1 << input_bits;
            for output_bits in [1, 3, 9] {
                let table: Vec<u16> = (0..rows)
                    .map(|row| ((row * 73 + 19) & ((1 << output_bits) - 1)) as u16)
                    .collect();
                for input in 0..rows {
                    let (client, program) =
                        prepare_logrow(input_bits, output_bits, &table).unwrap();
                    let (labels, decoder) = client.encode(input as u16).unwrap();
                    assert_eq!(
                        decoder.decode(program.evaluate(labels).unwrap()),
                        Ok(table[input])
                    );
                }
            }
        }
    }

    #[test]
    fn compact_q7_table_uses_less_material_than_binary_tree_oracle() {
        let table: Vec<u16> = (0..512)
            .map(|index| ((index * 313 + 17) & 511) as u16)
            .collect();
        for &index in &[0, 1, 127, 128, 255, 256, 511] {
            let (client, program) = prepare_logrow(9, 9, &table).unwrap();
            // The old 257-row Compact oracle is 256 * 9 * 32 bytes of
            // ciphertext. Count *all* evaluator rows and masked table here.
            assert!(program.evaluator_material_bytes() < 256 * 9 * 32);
            let (labels, decoder) = client.encode(index).unwrap();
            assert_eq!(
                decoder.decode(program.evaluate(labels).unwrap()),
                Ok(table[index as usize])
            );
        }
    }

    #[test]
    fn fitted_compact_profile_is_exact_and_digest_bound() {
        let profile = fit_compact_silu_q7(&[0; 257], 4).unwrap();
        for value in -128..=128 {
            let (client, program) = prepare_compact_q7_logrow(&profile).unwrap();
            assert_eq!(program.profile_digest(), profile.digest());
            assert_eq!(client.profile_digest(), profile.digest());
            assert_eq!(program.evaluator_material_bytes(), 2144);
            let (inputs, decoder) = client.encode(value).unwrap();
            assert_eq!(decoder.profile_digest(), profile.digest());
            assert_eq!(
                decoder.decode(program.evaluate(inputs).unwrap()),
                Ok(profile.evaluate(value).unwrap())
            );
        }
        let (client, _) = prepare_compact_q7_logrow(&profile).unwrap();
        assert!(client.encode(129).is_err());
    }

    #[test]
    fn scaled_silu_uses_independent_public_range_and_signed_table() {
        for max_abs in [1, 4, 16] {
            let profile = ScaledSiluQ7Profile::new(max_abs).unwrap();
            for value in -128..=128 {
                let mut elements = prepare_scaled_silu_q7_logrow_elements(&profile, 1).unwrap();
                let (client, program) = elements.pop().unwrap();
                assert_eq!(client.profile_digest(), profile.digest());
                assert_eq!(program.profile_digest(), profile.digest());
                let (inputs, decoder) = client.encode(value).unwrap();
                assert_eq!(
                    decoder.decode(program.evaluate(inputs).unwrap()),
                    Ok(profile.encoded_output(value).unwrap())
                );
            }
        }
    }

    #[test]
    fn invalid_shapes_and_cross_circuit_material_fail_closed() {
        assert!(prepare_logrow(0, 2, &[0, 1]).is_err());
        assert!(prepare_logrow(10, 2, &[0, 1]).is_err());
        assert!(prepare_logrow(2, 1, &[0, 1]).is_err());
        assert!(prepare_logrow(1, 1, &[0, 2]).is_err());
        let (client_a, program_a) = prepare_logrow(2, 2, &[0, 1, 2, 3]).unwrap();
        let (client_b, program_b) = prepare_logrow(2, 2, &[0, 1, 2, 3]).unwrap();
        let (labels_a, decoder_a) = client_a.encode(3).unwrap();
        let (labels_b, decoder_b) = client_b.encode(1).unwrap();
        assert!(program_a.evaluate(labels_b).is_err());
        assert!(program_b.evaluate(labels_a).is_err());
        let (client_c, program_c) = prepare_logrow(1, 1, &[0, 1]).unwrap();
        let (labels_c, decoder_c) = client_c.encode(0).unwrap();
        let mut result = program_c.evaluate(labels_c).unwrap();
        result.labels[0][0] ^= 1;
        assert!(decoder_c.decode(result).is_err());
        let _ = (decoder_a, decoder_b);
    }

    #[test]
    fn tampered_selected_table_bit_rejects_invalid_output_label() {
        let (client, mut program) = prepare_logrow(2, 3, &[0, 1, 2, 3]).unwrap();
        let (labels, decoder) = client.encode(2).unwrap();
        let masked_index = labels
            .labels
            .iter()
            .enumerate()
            .fold(0_usize, |index, (bit, label)| {
                index | (usize::from(label[15] & 1) << bit)
            });
        let offset = masked_index * 3;
        program.masked_table[offset / 8] ^= 1 << (offset % 8);
        assert!(decoder.decode(program.evaluate(labels).unwrap()).is_err());
    }
}
