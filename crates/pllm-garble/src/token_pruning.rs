//! CipherPrune-inspired threshold/count and stable complete-record compaction.
//!
//! Up to two fresh half-gates circuits replace the paper's OT-based primitives. Only
//! the retained count and compacted degree mask are opened. Inputs, comparisons,
//! scores and moved records otherwise remain shares or opaque labels. The local
//! issuer is trusted and in-process; there is no distributed OT or transport.
use crate::boolean::{
    BooleanCircuitBuilder as Builder, BooleanCircuitClient, BooleanCircuitProgram, BooleanWire,
    MAX_REFERENCE_INPUTS, MAX_REFERENCE_OUTPUTS, MAX_REFERENCE_WIRES,
};
use pllm_core::token_pruning::{self as numeric, Output, Parameters};
use std::time::Instant;
use zeroize::Zeroizing;

pub const MAX_ALLOCATION_BYTES: u64 = 64 << 20;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub input_bits: u64,
    pub output_bits: u64,
    pub constants: u64,
    pub nots: u64,
    pub xors: u64,
    pub ands: u64,
}

impl Counts {
    pub fn wires(self) -> u64 {
        self.input_bits + self.constants + self.nots + self.xors + self.ands
    }
    pub fn evaluator_bytes(self) -> u64 {
        32 * self.ands + 16 * self.constants
    }
    pub fn issuer_bytes(self) -> u64 {
        32 * (self.input_bits + self.output_bits)
    }
    pub fn boundary_bytes(self) -> u64 {
        16 * (self.input_bits + self.output_bits)
    }
    pub fn metadata_bytes(self) -> u64 {
        if self.wires() == 0 {
            return 0;
        }
        64 + 8 * self.output_bits + self.constants + 9 * self.nots + 17 * self.xors + 25 * self.ands
    }
    fn adders(&mut self, count: u64, width: u64) {
        self.constants += count;
        self.xors += count * width * 3;
        self.ands += count * width * 2;
    }
    fn subtractors(&mut self, count: u64, width: u64) {
        self.adders(count, width);
        self.nots += count * width * 2;
    }
    fn greater(&mut self, count: u64, threshold: u32) {
        let ones = u64::from(threshold.count_ones());
        self.constants += 2 * count;
        self.nots += 33 * count;
        self.xors += count * (ones + 1);
        self.ands += count * (32 + ones);
    }
    fn check(self, program: &BooleanCircuitProgram) -> Result<(), String> {
        if program.wire_count() as u64 != self.wires()
            || program.and_gate_count() as u64 != self.ands
            || program.constant_label_bytes() as u64 != 16 * self.constants
            || program.public_metadata_bytes_estimate() as u64 != self.metadata_bytes()
        {
            return Err("pruning circuit differs from its preflight resource contract".into());
        }
        Ok(())
    }
}

fn count_bits(tokens: usize) -> usize {
    (usize::BITS - tokens.leading_zeros()) as usize
}

/// Exact geometry-only counts; allocating a circuit still requires worst-case
/// whole-invocation admission, before the first count is disclosed.
pub fn project(p: Parameters, retained: usize) -> Result<[Counts; 2], String> {
    p.validate()?;
    if retained > p.tokens {
        return Err("retained count exceeds input tokens".into());
    }
    let n = p.tokens as u64;
    let k = retained as u64;
    let width = p.features as u64;
    let count_width = count_bits(p.tokens) as u64;
    let mut first = Counts {
        input_bits: 65 * n,
        output_bits: n + count_width,
        constants: 1,
        xors: n,
        ..Counts::default()
    };
    first.adders(n, 32);
    first.greater(n, p.score_threshold(false));
    first.adders(n, count_width);
    let mut second = Counts::default();
    if retained != 0 && retained != p.tokens {
        second.input_bits = 64 * n * (width + 1) + 2 * n + 32 * k * width;
        second.output_bits = 32 * k * width + if retained < p.tokens { k } else { 0 };
        second.xors = n;
        second.adders(n * (width + 1), 32);
        let swaps = numeric::swaps(p.tokens, retained)? as u64;
        let record_bits = 1 + 32 * (width + 1);
        second.nots += swaps;
        second.xors += 3 * swaps * record_bits;
        second.ands += swaps * record_bits;
        second.subtractors(k * width, 32);
        if retained < p.tokens {
            second.greater(k, p.score_threshold(true));
        }
    }
    Ok([first, second])
}

pub fn allocation_estimate(p: Parameters) -> Result<u64, String> {
    p.validate()?;
    let mut maximum = 0;
    for kept in 0..=p.tokens {
        let phases = project(p, kept)?;
        let wires = phases.iter().map(|v| v.wires()).sum::<u64>();
        let boundaries = phases
            .iter()
            .map(|v| v.input_bits + v.output_bits)
            .sum::<u64>();
        maximum = maximum.max(384 * wires + 64 * boundaries);
    }
    // Both independent share owners, native caller copies, zeroizing temporary
    // score/output buffers and 4 MiB of scratch. External caller owners are extra.
    Ok(maximum + 16 * (p.attention_elements() + p.tokens * p.features) as u64 + (4 << 20))
}

pub fn admit(p: Parameters) -> Result<u64, String> {
    let bytes = allocation_estimate(p)?;
    if bytes > MAX_ALLOCATION_BYTES {
        return Err("pruning exceeds the 64 MiB allocation estimate".into());
    }
    for kept in 0..=p.tokens {
        for c in project(p, kept)? {
            if c.input_bits > MAX_REFERENCE_INPUTS as u64
                || c.output_bits > MAX_REFERENCE_OUTPUTS as u64
                || c.wires() > MAX_REFERENCE_WIRES as u64
            {
                return Err("pruning exceeds the one-use circuit limits".into());
            }
        }
    }
    Ok(bytes)
}

fn greater(b: &mut Builder, word: &[BooleanWire], threshold: u32) -> Result<BooleanWire, String> {
    let (less, equal) = b.compare_unsigned_to_constant(word, u128::from(threshold))?;
    let at_most = b.xor(less, equal)?;
    b.not(at_most)
}

fn count_circuit(p: Parameters) -> Result<(BooleanCircuitClient, BooleanCircuitProgram), String> {
    let counts = project(p, 0)?[0];
    let mut b = Builder::with_wire_limit(counts.wires() as usize)?;
    let mut inputs = Vec::new();
    for _ in 0..p.tokens {
        inputs.push((b.input_word(32)?, b.input_word(32)?, b.input()?));
    }
    let zero = b.constant(false)?;
    let mut count = vec![zero; count_bits(p.tokens)];
    let mut outputs = Vec::new();
    for (left, right, mask) in inputs {
        let score = b.add_unsigned(&left, &right)?.0;
        let keep = greater(&mut b, &score, p.score_threshold(false))?;
        outputs.push(b.xor(keep, mask)?);
        let mut addend = vec![zero; count.len()];
        addend[0] = keep;
        count = b.add_unsigned(&count, &addend)?.0;
    }
    outputs.extend(count);
    let (client, program) = b.finish(&outputs)?;
    counts.check(&program)?;
    Ok((client, program))
}

fn compact_circuit(
    p: Parameters,
    retained: usize,
) -> Result<(BooleanCircuitClient, BooleanCircuitProgram), String> {
    if retained == 0 || retained == p.tokens {
        return Err("empty or unchanged pruning needs no compaction circuit".into());
    }
    let counts = project(p, retained)?[1];
    let mut b = Builder::with_wire_limit(counts.wires() as usize)?;
    let mut inputs = Vec::new();
    for _ in 0..p.tokens {
        let mut words = Vec::new();
        for _ in 0..=p.features {
            words.push((b.input_word(32)?, b.input_word(32)?));
        }
        inputs.push((words, b.input()?, b.input()?));
    }
    let mut masks = Vec::new();
    for _ in 0..retained * p.features {
        masks.push(b.input_word(32)?);
    }
    let mut records = Vec::new();
    for (words, left_keep, right_keep) in inputs {
        // A separate marker wire never overwrites a signed payload bit. The score
        // travels with the complete record for the post-compaction degree test.
        let mut record = vec![b.xor(left_keep, right_keep)?];
        for (left, right) in words {
            record.extend(b.add_unsigned(&left, &right)?.0);
        }
        records.push(record);
    }
    for pass in 0..p.tokens - retained {
        for row in 0..p.tokens - pass - 1 {
            let swap = b.not(records[row][0])?;
            let (left, right) = records.split_at_mut(row + 1);
            for (a, c) in left[row].iter_mut().zip(&mut right[0]) {
                let different = b.xor(*a, *c)?;
                let movement = b.and(swap, different)?;
                *a = b.xor(*a, movement)?;
                *c = b.xor(*c, movement)?;
            }
        }
    }
    let mut outputs = Vec::new();
    for (row, record) in records.iter().take(retained).enumerate() {
        for feature in 0..p.features {
            let offset = 33 + feature * 32;
            outputs.extend(
                b.subtract_unsigned(
                    &record[offset..offset + 32],
                    &masks[row * p.features + feature],
                )?
                .0,
            );
        }
    }
    if retained < p.tokens {
        for record in records.iter().take(retained) {
            outputs.push(greater(&mut b, &record[1..33], p.score_threshold(true))?);
        }
    }
    let (client, program) = b.finish(&outputs)?;
    counts.check(&program)?;
    Ok((client, program))
}

fn random_words(count: usize) -> Result<Zeroizing<Vec<u32>>, String> {
    let mut bytes = Zeroizing::new(vec![0_u8; count * 4]);
    getrandom::fill(&mut bytes).map_err(|e| e.to_string())?;
    Ok(Zeroizing::new(
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().expect("word")))
            .collect(),
    ))
}

fn put(bits: &mut Vec<bool>, word: u32) {
    bits.extend((0..32).map(|i| word & (1 << i) != 0));
}
fn word(bits: &[bool]) -> u32 {
    bits.iter()
        .enumerate()
        .fold(0, |v, (i, b)| v | (u32::from(*b) << i))
}

pub struct Measurement {
    pub output: Output,
    pub phases: [Counts; 2],
    pub allocation_estimate_bytes: u64,
    pub sharing_seconds: f64,
    pub issuance_seconds: [f64; 2],
    pub online_seconds: [f64; 2],
}

/// The trusted test issuer owns supplied plaintexts. Each local party receives
/// independent uniform ring shares; evaluator programs have no output decoder.
/// Count-dependent issuance occurs only after public worst-case preflight.
pub fn run_reference(
    p: Parameters,
    attention: &[u32],
    words: &[u32],
) -> Result<Measurement, String> {
    let allocation_estimate_bytes = admit(p)?;
    let expected = numeric::reference(p, attention, words)?;
    let start = Instant::now();
    let attention_left = random_words(attention.len())?;
    let mut score_left = Zeroizing::new(vec![0_u32; p.tokens]);
    let mut score_right = Zeroizing::new(vec![0_u32; p.tokens]);
    for (i, (&clear, &left)) in attention.iter().zip(attention_left.iter()).enumerate() {
        score_left[i % p.tokens] = score_left[i % p.tokens].wrapping_add(left);
        score_right[i % p.tokens] =
            score_right[i % p.tokens].wrapping_add(clear.wrapping_sub(left));
    }
    drop(attention_left);
    let word_left = random_words(words.len())?;
    let word_right = Zeroizing::new(
        words
            .iter()
            .zip(word_left.iter())
            .map(|(&v, &l)| v.wrapping_sub(l))
            .collect::<Vec<_>>(),
    );
    let keep_left = Zeroizing::new(
        random_words(p.tokens)?
            .iter()
            .map(|v| v & 1 != 0)
            .collect::<Vec<_>>(),
    );
    let sharing_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (client, program) = count_circuit(p)?;
    let first_issuance = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut bits = Zeroizing::new(Vec::new());
    for i in 0..p.tokens {
        put(&mut bits, score_left[i]);
        put(&mut bits, score_right[i]);
        bits.push(keep_left[i]);
    }
    let (inputs, decoder) = client.encode(&bits)?;
    let decoded = Zeroizing::new(decoder.decode(program.evaluate(inputs)?)?);
    let retained = word(&decoded[p.tokens..]) as usize;
    if retained > p.tokens || retained != expected.retained {
        return Err("protected retained count differs from integer reference".into());
    }
    let keep_right = Zeroizing::new(decoded[..p.tokens].to_vec());
    let first_online = start.elapsed().as_secs_f64();
    drop(decoded);
    drop(bits);
    let phases = project(p, retained)?;
    let mut measurement = Measurement {
        output: Output {
            words: Vec::new(),
            retained,
            high_degree: (retained < p.tokens).then(Vec::new),
        },
        phases,
        allocation_estimate_bytes,
        sharing_seconds,
        issuance_seconds: [first_issuance, 0.0],
        online_seconds: [first_online, 0.0],
    };
    if retained == 0 {
        return Ok(measurement);
    }
    if retained == p.tokens {
        // The public count proves that every record stays. Each party keeps its
        // original share: no fresh circuit, output mask or degree disclosure.
        // Only the trusted test client reconstructs for the independent oracle.
        let start = Instant::now();
        measurement.output.words = word_left
            .iter()
            .zip(word_right.iter())
            .map(|(&left, &right)| left.wrapping_add(right))
            .collect();
        measurement.online_seconds[1] = start.elapsed().as_secs_f64();
        if measurement.output != expected {
            return Err("unchanged pruning differs from independent filter".into());
        }
        return Ok(measurement);
    }
    let start = Instant::now();
    let masks = random_words(retained * p.features)?;
    let (client, program) = compact_circuit(p, retained)?;
    measurement.issuance_seconds[1] = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut bits = Zeroizing::new(Vec::new());
    for row in 0..p.tokens {
        put(&mut bits, score_left[row]);
        put(&mut bits, score_right[row]);
        for column in 0..p.features {
            let i = row * p.features + column;
            put(&mut bits, word_left[i]);
            put(&mut bits, word_right[i]);
        }
        bits.extend([keep_left[row], keep_right[row]]);
    }
    for &mask in masks.iter() {
        put(&mut bits, mask);
    }
    let (inputs, decoder) = client.encode(&bits)?;
    let decoded = Zeroizing::new(decoder.decode(program.evaluate(inputs)?)?);
    measurement.output.words = decoded[..32 * masks.len()]
        .chunks_exact(32)
        .zip(masks.iter())
        .map(|(b, &m)| word(b).wrapping_add(m))
        .collect();
    measurement.output.high_degree =
        (retained < p.tokens).then(|| decoded[32 * masks.len()..].to_vec());
    measurement.online_seconds[1] = start.elapsed().as_secs_f64();
    if measurement.output != expected {
        return Err("protected pruning differs from independent filter".into());
    }
    Ok(measurement)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(tokens: usize, features: usize) -> Parameters {
        Parameters {
            tokens,
            features,
            heads: 2,
            threshold_q16: 10,
            reduction_threshold_q16: 20,
        }
    }

    #[test]
    fn protected_count_compaction_and_degree_match_independent_filter() {
        for n in [1, 2, 4, 8] {
            for retained in 0..=n {
                let p = parameters(n, 2);
                let mut row = vec![10; n];
                // Reversed placement exercises movement without exposing indices.
                for (i, value) in row.iter_mut().rev().take(retained).enumerate() {
                    *value = if i % 2 == 0 { 21 } else { 20 };
                }
                let attention = row.repeat(n * p.heads);
                let words = (0..2 * n)
                    .map(|i| [0, u32::MAX, 0x8000_0000, i as u32][i % 4])
                    .collect::<Vec<_>>();
                let measured = run_reference(p, &attention, &words).unwrap();
                assert_eq!(
                    measured.output,
                    numeric::reference(p, &attention, &words).unwrap()
                );
                assert_eq!(measured.output.retained, retained);
                if retained == 0 || retained == n {
                    assert_eq!(measured.phases[1], Counts::default());
                }
                assert!(measured.allocation_estimate_bytes <= MAX_ALLOCATION_BYTES);
            }
        }
    }

    #[test]
    fn all_small_masks_preserve_stable_order_and_same_count_schedule() {
        let p = parameters(4, 2);
        let words = (0..8).collect::<Vec<_>>();
        let mut schedules = std::collections::BTreeMap::new();
        for mask in 0..16_u32 {
            let row = (0..4)
                .map(|i| if mask & (1 << i) != 0 { 21 } else { 0 })
                .collect::<Vec<_>>();
            let attention = row.repeat(8);
            let measured = run_reference(p, &attention, &words).unwrap();
            assert_eq!(
                measured.output,
                numeric::reference(p, &attention, &words).unwrap()
            );
            let kept = mask.count_ones() as usize;
            if kept > 0 && kept < p.tokens {
                let (_, program) = compact_circuit(p, kept).unwrap();
                let digest = program.public_schedule_digest();
                if let Some(previous) = schedules.insert(kept, digest) {
                    assert_eq!(previous, digest);
                }
            }
        }
    }

    #[test]
    fn preflight_rejects_full_width_and_circuit_handles_reject_substitution() {
        assert!(admit(parameters(39, 896)).is_err());
        assert!(project(parameters(4, 2), 5).is_err());
        let p = parameters(2, 1);
        let (a, pa) = count_circuit(p).unwrap();
        let (b, pb) = count_circuit(p).unwrap();
        let (ia, da) = a.encode(&[false; 130]).unwrap();
        let (ib, _) = b.encode(&[false; 130]).unwrap();
        let result = pb.evaluate(ib).unwrap();
        assert!(da.decode(result).is_err());
        drop(ia); // cancellation consumes labels; no reusable handle escapes
        drop(pa);
        let (a, _) = count_circuit(p).unwrap();
        assert!(a.encode(&[false]).is_err());
        let (_, compact_a) = compact_circuit(p, 1).unwrap();
        let (b, _) = compact_circuit(p, 1).unwrap();
        let (foreign, _) = b
            .encode(&vec![false; project(p, 1).unwrap()[1].input_bits as usize])
            .unwrap();
        assert!(compact_a.evaluate(foreign).is_err());
    }
}
