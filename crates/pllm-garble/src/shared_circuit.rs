//! Dealer-aided, one-use Yao conversion between additive ring shares.
//!
//! The garbler sees only its input shares and receives fresh output masks.
//! The evaluator receives one label per input via fresh random-OT correlations
//! and decodes only f(x)-r. Output hash commitments do not expose free-XOR delta.
//! This local trusted dealer is not distributed OT setup or authenticated transport.
use crate::boolean::{
    BooleanCircuitBuilder as Builder, BooleanCircuitProgram, BooleanWire, SharedInputEncodings,
    SharedOutputDecoder,
};
use crate::shared_rescale_reference::{Context, Error};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub const HEADER: usize = 78;
pub const MAX_WIRES: usize = 65_536;
const MAGIC: &[u8; 8] = b"PLLMYC01";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resources {
    pub key_bytes: [usize; 2],
    pub phase_bytes: [[usize; 2]; 2],
    pub ciphertext_bytes: usize,
    pub constant_bytes: usize,
    pub public_metadata_bytes: usize,
    pub allocation_bytes: usize,
}

struct Sender {
    inputs: SharedInputEncodings,
    pads: Zeroizing<Vec<[[u8; 16]; 2]>>,
    masks: Zeroizing<Vec<u64>>,
    labels: Zeroizing<Vec<[u8; 16]>>,
}
struct Receiver {
    program: BooleanCircuitProgram,
    decoder: SharedOutputDecoder,
    choices: Zeroizing<Vec<u8>>,
    pads: Zeroizing<Vec<[u8; 16]>>,
    requested: Zeroizing<Vec<u8>>,
}
enum Material {
    Sender(Sender),
    Receiver(Receiver),
}
pub struct Party {
    id: u8,
    bits: usize,
    words: usize,
    input_widths: Vec<usize>,
    outputs: usize,
    binding: [u8; 32],
    nonce: [u8; 32],
    phase: u8,
    material: Option<Material>,
    resources: Resources,
}
pub enum Progress {
    Open(Vec<u8>),
    Complete(Zeroizing<Vec<u64>>),
}

fn domain(_: String) -> Error {
    Error::Domain
}
pub fn mask(bits: usize) -> u64 {
    u64::MAX >> (64 - bits)
}
pub fn random_word() -> Result<u64, Error> {
    let mut raw = Zeroizing::new([0; 8]);
    getrandom::fill(&mut *raw).map_err(|_| Error::Randomness)?;
    Ok(u64::from_le_bytes(*raw))
}

/// Exact signed ties-even shift using floor quotient and nonnegative remainder.
/// Unlike sharewise shifts, reconstruction, carry and rounding stay garbled.
pub fn rounded(
    b: &mut Builder,
    input: &[BooleanWire],
    shift: usize,
) -> Result<Vec<BooleanWire>, String> {
    if shift == 0 || shift >= input.len() {
        return Err("invalid signed shift".into());
    }
    let sign = *input.last().unwrap();
    let mut quotient = input[shift..].to_vec();
    quotient.resize(input.len(), sign);
    let mut lower = b.constant(false)?;
    for &bit in &input[..shift - 1] {
        lower = or(b, lower, bit)?;
    }
    let condition = or(b, lower, quotient[0])?;
    let carry = b.and(input[shift - 1], condition)?;
    let zero = b.constant(false)?;
    let mut increment = vec![zero; input.len()];
    increment[0] = carry;
    Ok(b.add_unsigned(&quotient, &increment)?.0)
}

fn or(b: &mut Builder, x: BooleanWire, y: BooleanWire) -> Result<BooleanWire, String> {
    let different = b.xor(x, y)?;
    let both = b.and(x, y)?;
    b.xor(different, both)
}

/// The public topology closure receives reconstructed *wires*, never clear
/// inputs. Uniform full-ring output masks are hidden circuit constants.
pub fn issue(
    bits: usize,
    words: usize,
    outputs: usize,
    context: Context,
    build: impl FnOnce(&mut Builder, &[Vec<BooleanWire>]) -> Result<Vec<Vec<BooleanWire>>, String>,
) -> Result<([Party; 2], Resources), Error> {
    issue_bounded(bits, words, outputs, MAX_WIRES, context, build)
}

pub(crate) fn issue_bounded(
    bits: usize,
    words: usize,
    outputs: usize,
    wire_limit: usize,
    context: Context,
    build: impl FnOnce(&mut Builder, &[Vec<BooleanWire>]) -> Result<Vec<Vec<BooleanWire>>, String>,
) -> Result<([Party; 2], Resources), Error> {
    if !(2..=64).contains(&bits) || !(1..=8).contains(&words) || !(1..=8).contains(&outputs) {
        return Err(Error::Resource);
    }
    issue_widths(
        &vec![bits; words],
        bits,
        outputs,
        wire_limit,
        context,
        build,
    )
}

/// Reconstruct each input in its own public ring, while keeping full-width
/// independently masked output shares. Callers must establish input range bounds.
pub(crate) fn issue_widths(
    input_widths: &[usize],
    bits: usize,
    outputs: usize,
    wire_limit: usize,
    context: Context,
    build: impl FnOnce(&mut Builder, &[Vec<BooleanWire>]) -> Result<Vec<Vec<BooleanWire>>, String>,
) -> Result<([Party; 2], Resources), Error> {
    let words = input_widths.len();
    if !(2..=64).contains(&bits)
        || !(1..=8).contains(&words)
        || !(1..=8).contains(&outputs)
        || input_widths.iter().any(|width| !(2..=64).contains(width))
    {
        return Err(Error::Resource);
    }
    if wire_limit > MAX_WIRES {
        return Err(Error::Resource);
    }
    let mut b = Builder::with_wire_limit(wire_limit).map_err(domain)?;
    let a = input_widths
        .iter()
        .map(|&width| b.input_word(width))
        .collect::<Result<Vec<_>, _>>()
        .map_err(domain)?;
    let c = input_widths
        .iter()
        .map(|&width| b.input_word(width))
        .collect::<Result<Vec<_>, _>>()
        .map_err(domain)?;
    let x = a
        .iter()
        .zip(&c)
        .map(|(a, c)| b.add_unsigned(a, c).map(|v| v.0))
        .collect::<Result<Vec<_>, _>>()
        .map_err(domain)?;
    let result = build(&mut b, &x).map_err(domain)?;
    if result.len() != outputs || result.iter().any(|v| v.len() != bits) {
        return Err(Error::Domain);
    }
    let mut masks = Zeroizing::new(Vec::with_capacity(outputs));
    let mut wires = Vec::with_capacity(outputs * bits);
    for value in &result {
        let r = random_word()? & mask(bits);
        masks.push(r);
        let hidden = b.constant_word(u128::from(r), bits).map_err(domain)?;
        wires.extend(b.subtract_unsigned(value, &hidden).map_err(domain)?.0);
    }
    let (client, program) = b.finish(&wires).map_err(domain)?;
    let nonce = program.issuance_id();
    let (inputs, decoder) = client.into_shared_parts();
    debug_assert_eq!(inputs.id(), nonce);
    let count: usize = input_widths.iter().sum();
    let mut sender = Zeroizing::new(Vec::with_capacity(count));
    let mut receiver = Zeroizing::new(Vec::with_capacity(count));
    let mut choices = Zeroizing::new(Vec::with_capacity(count));
    for _ in 0..count {
        let mut pads = Zeroizing::new([[0; 16]; 2]);
        getrandom::fill(&mut pads[0]).map_err(|_| Error::Randomness)?;
        getrandom::fill(&mut pads[1]).map_err(|_| Error::Randomness)?;
        let choice = (random_word()? & 1) as u8;
        sender.push(*pads);
        receiver.push(pads[usize::from(choice)]);
        choices.push(choice);
    }
    let ciphertext = program.evaluator_ciphertext_bytes().map_err(domain)? as usize
        + program.lookup_material_bytes();
    let constants = program.constant_label_bytes();
    let resources = Resources {
        // A: both input encodings + random OT sender pads + output masks.
        // B: one random OT pad/choice per private bit + hash output decoder.
        key_bytes: [
            2 * count * 32 + count * 32 + outputs * 8,
            count * 17 + outputs * bits * 64 + ciphertext + constants,
        ],
        phase_bytes: [[HEADER, HEADER + count], [HEADER + 3 * count * 16, HEADER]],
        ciphertext_bytes: ciphertext,
        constant_bytes: constants,
        public_metadata_bytes: program.public_metadata_bytes_estimate() + 2 * HEADER,
        allocation_bytes: program.wire_count() * 384 + 1024 * (count + outputs * bits) + (1 << 20),
    };
    let mut hash = Sha256::new();
    hash.update(b"pllm.shared_circuit.v1\0");
    hash.update(context.plan);
    hash.update(context.session);
    hash.update(context.operation);
    hash.update(context.first_tensor_index.to_le_bytes());
    hash.update([bits as u8, words as u8, outputs as u8]);
    if input_widths.iter().any(|&width| width != bits) {
        hash.update(b"pllm.shared_circuit.input_widths.v1\0");
        for &width in input_widths {
            hash.update([width as u8]);
        }
    }
    hash.update(program.public_schedule_digest());
    let binding = hash.finalize().into();
    let make = |id, material| Party {
        id,
        bits,
        words,
        input_widths: input_widths.to_vec(),
        outputs,
        binding,
        nonce,
        phase: 0,
        material: Some(material),
        resources: resources.clone(),
    };
    let a = make(
        0,
        Material::Sender(Sender {
            inputs,
            pads: sender,
            masks,
            labels: Zeroizing::new(Vec::new()),
        }),
    );
    let c = make(
        1,
        Material::Receiver(Receiver {
            program,
            decoder,
            choices,
            pads: receiver,
            requested: Zeroizing::new(Vec::new()),
        }),
    );
    Ok(([a, c], resources))
}

impl Party {
    pub fn cancel(&mut self) {
        self.material = None;
    }
    pub fn key_bytes(&self) -> usize {
        if self.material.is_some() {
            self.resources.key_bytes[self.id as usize]
        } else {
            0
        }
    }
    fn frame(&self, phase: u8, body: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(HEADER + body.len());
        frame.extend(MAGIC);
        frame.extend(self.binding);
        frame.extend(self.nonce);
        frame.extend([phase, self.id]);
        frame.extend((self.words as u32).to_le_bytes());
        frame.extend(body);
        frame
    }
    fn peer<'a>(&self, phase: usize, frame: &'a [u8]) -> Result<&'a [u8], Error> {
        if frame.len() != self.resources.phase_bytes[phase][1 - self.id as usize]
            || &frame[..8] != MAGIC
            || frame[8..40] != self.binding
            || frame[40..72] != self.nonce
            || frame[72] != phase as u8
            || frame[73] != 1 - self.id
            || u32::from_le_bytes(frame[74..78].try_into().unwrap()) as usize != self.words
        {
            return Err(Error::Context);
        }
        Ok(&frame[HEADER..])
    }
    pub fn start(&mut self, values: &[u64]) -> Result<Vec<u8>, Error> {
        let mut material = self.material.take().ok_or(Error::Consumed)?;
        if self.phase != 0
            || values.len() != self.words
            || values
                .iter()
                .zip(&self.input_widths)
                .any(|(&v, &bits)| v > mask(bits))
        {
            return Err(Error::Domain);
        }
        let mut body = Zeroizing::new(Vec::new());
        match &mut material {
            Material::Sender(a) => {
                let mut offset = 0;
                for (&value, &bits) in values.iter().zip(&self.input_widths) {
                    for bit in 0..bits {
                        a.labels.push(
                            a.inputs
                                .encode(offset + bit, (value >> bit) & 1 != 0)
                                .map_err(domain)?,
                        );
                    }
                    offset += bits;
                }
            }
            Material::Receiver(b) => {
                let mut offset = 0;
                for (&value, &bits) in values.iter().zip(&self.input_widths) {
                    for bit in 0..bits {
                        let choice = ((value >> bit) & 1) as u8;
                        b.requested.push(choice);
                        body.push(choice ^ b.choices[offset + bit]);
                    }
                    offset += bits;
                }
            }
        }
        let frame = self.frame(0, &body);
        self.phase = 1;
        self.material = Some(material);
        Ok(frame)
    }
    pub fn advance(&mut self, frame: &[u8]) -> Result<Progress, Error> {
        let mut material = self.material.take().ok_or(Error::Consumed)?;
        if !(1..=2).contains(&self.phase) {
            return Err(Error::Context);
        }
        let body = self.peer(self.phase as usize - 1, frame)?;
        if self.phase == 1 {
            let mut response = Zeroizing::new(Vec::new());
            if let Material::Sender(a) = &mut material {
                if body.iter().any(|&v| v > 1) {
                    return Err(Error::Domain);
                }
                for label in a.labels.iter() {
                    response.extend(label);
                }
                let count: usize = self.input_widths.iter().sum();
                for (i, &d) in body.iter().enumerate() {
                    for bit in 0..2 {
                        let label =
                            Zeroizing::new(a.inputs.encode(count + i, bit != 0).map_err(domain)?);
                        for (l, p) in label.iter().zip(a.pads[i][bit ^ d as usize]) {
                            response.push(l ^ p);
                        }
                    }
                }
            }
            let response = self.frame(1, &response);
            self.phase = 2;
            self.material = Some(material);
            return Ok(Progress::Open(response));
        }
        match material {
            Material::Sender(a) => Ok(Progress::Complete(a.masks)),
            Material::Receiver(b) => {
                let count: usize = self.input_widths.iter().sum();
                let mut labels = Zeroizing::new(Vec::with_capacity(2 * count));
                for raw in body[..count * 16].chunks_exact(16) {
                    labels.push(raw.try_into().unwrap());
                }
                for i in 0..count {
                    let start = count * 16 + i * 32 + b.requested[i] as usize * 16;
                    let mut label = [0; 16];
                    for j in 0..16 {
                        label[j] = body[start + j] ^ b.pads[i][j];
                    }
                    labels.push(label);
                }
                let result = Zeroizing::new(
                    b.decoder
                        .decode(
                            b.program
                                .evaluate_shared(self.nonce, labels)
                                .map_err(domain)?,
                        )
                        .map_err(domain)?,
                );
                if result.len() != self.outputs * self.bits {
                    return Err(Error::Context);
                }
                Ok(Progress::Complete(Zeroizing::new(
                    result
                        .chunks_exact(self.bits)
                        .map(|bits| {
                            bits.iter()
                                .enumerate()
                                .fold(0, |v, (i, &bit)| v | (u64::from(bit) << i))
                        })
                        .collect(),
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ctx() -> Context {
        Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: 0,
        }
    }
    #[test]
    fn signed_rounding_random_share_wraps_and_ot_choices_are_exact() {
        for bits in [8, 32, 64] {
            for value in [-129_i64, -128, -127, -5, -3, -1, 0, 1, 3, 5, 127] {
                if bits == 8 && value < -128 {
                    continue;
                }
                let ([mut a, mut b], r) =
                    issue(bits, 1, 1, ctx(), |b, x| Ok(vec![rounded(b, &x[0], 1)?])).unwrap();
                let share = random_word().unwrap() & mask(bits);
                let f = a.start(&[share]).unwrap();
                let g = b
                    .start(&[(value as u64).wrapping_sub(share) & mask(bits)])
                    .unwrap();
                assert_eq!([f.len(), g.len()], r.phase_bytes[0]);
                let (Progress::Open(f), Progress::Open(g)) =
                    (a.advance(&g).unwrap(), b.advance(&f).unwrap())
                else {
                    panic!()
                };
                assert_eq!([f.len(), g.len()], r.phase_bytes[1]);
                let (Progress::Complete(x), Progress::Complete(y)) =
                    (a.advance(&g).unwrap(), b.advance(&f).unwrap())
                else {
                    panic!()
                };
                assert_eq!(
                    x[0].wrapping_add(y[0]) & mask(bits),
                    ((value as f64 / 2.0).round_ties_even() as i64 as u64) & mask(bits)
                );
                assert_eq!(a.key_bytes() + b.key_bytes(), 0);
                assert!(a.advance(&g).is_err());
            }
        }
    }
    #[test]
    fn distinct_input_rings_preserve_carries_and_burn_invalid_shares() {
        let build = |b: &mut Builder, x: &[Vec<BooleanWire>]| {
            let zero = b.constant(false)?;
            let mut word = x[0].clone();
            word.extend(&x[1]);
            word.resize(32, zero);
            Ok(vec![word])
        };
        let ([mut a, mut b], r) = issue_widths(&[2, 9], 32, 1, MAX_WIRES, ctx(), build).unwrap();
        // Reconstruct 1 mod 4 and 300 mod 512, including sharewise wraps.
        let f = a.start(&[3, 500]).unwrap();
        let g = b.start(&[2, 312]).unwrap();
        assert_eq!([f.len(), g.len()], r.phase_bytes[0]);
        let (Progress::Open(f), Progress::Open(g)) =
            (a.advance(&g).unwrap(), b.advance(&f).unwrap())
        else {
            panic!()
        };
        assert_eq!([f.len(), g.len()], r.phase_bytes[1]);
        let (Progress::Complete(x), Progress::Complete(y)) =
            (a.advance(&g).unwrap(), b.advance(&f).unwrap())
        else {
            panic!()
        };
        assert_eq!(x[0].wrapping_add(y[0]) & mask(32), 1 + (300 << 2));
        assert_eq!(a.key_bytes() + b.key_bytes(), 0);
        for bad in [&[4, 0][..], &[0, 512][..], &[0][..]] {
            let ([mut a, mut b], _) =
                issue_widths(&[2, 9], 32, 1, MAX_WIRES, ctx(), build).unwrap();
            assert!(a.start(bad).is_err());
            assert_eq!(a.key_bytes(), 0);
            assert!(a.start(&[0, 0]).is_err());
            b.cancel();
        }
    }

    #[test]
    fn foreign_context_malformed_ot_and_cancellation_burn() {
        for index in [0, 8, 40, 72, 73, 74, 78] {
            let ([mut a, mut b], _) = issue(8, 1, 1, ctx(), |_, x| Ok(x.to_vec())).unwrap();
            a.start(&[1]).unwrap();
            let mut f = b.start(&[2]).unwrap();
            f[index] ^= 128;
            assert!(a.advance(&f).is_err());
            assert_eq!(a.key_bytes(), 0);
            b.cancel();
            assert_eq!(b.key_bytes(), 0);
        }
        let ([mut a, mut b], _) = issue(8, 1, 1, ctx(), |_, x| Ok(x.to_vec())).unwrap();
        assert!(a.start(&[256]).is_err());
        b.cancel();
        assert!(b.start(&[1]).is_err());
    }
}
