//! Complete fixed-profile gated blocks using party-local arithmetic shares.
//! Boolean circuits handle selection, exact rounding and representation changes;
//! fresh 64-bit Beaver triples handle products. No intermediate plaintext decode.
use crate::boolean::{BooleanCircuitBuilder as Builder, BooleanWire};
use crate::shared_rescale_reference::{Context, Error};
use crate::{shared_circuit as gc, shared_lut_reference as lut, shared_wide_product as mul};
use pllm_core::{compact_product as compact, wavelet_gated_reference as wavelet};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use zeroize::Zeroizing;

pub const MAX_ALLOCATION: usize = 64 << 20;
const HEADER: usize = 78;
const RESCALE_WIRES: usize = 4096;
const MAGIC: &[u8; 8] = b"PLLMMX01";

pub enum Profile<'a> {
    Compact(&'a compact::Profile),
    Curl(&'a wavelet::Profile),
    LogRow(&'a wavelet::Profile),
}
impl Profile<'_> {
    fn digest(&self) -> [u8; 32] {
        match self {
            Self::Compact(p) => p.digest(),
            Self::Curl(p) | Self::LogRow(p) => p.digest(),
        }
    }
    fn name(&self) -> &'static [u8] {
        match self {
            Self::Compact(_) => b"compact",
            Self::Curl(_) => b"curl",
            Self::LogRow(_) => b"logrow",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resources {
    pub key_bytes: [usize; 2],
    pub parts: Vec<(&'static str, [usize; 2])>,
    pub phases: Vec<[usize; 2]>,
    pub peer_payload_bits: usize,
    pub ciphertext_bytes: usize,
    pub constant_bytes: usize,
    pub public_metadata_bytes: usize,
    pub public_table_bytes: usize,
    pub allocation_bytes: usize,
}

enum Op {
    Circuit(Box<gc::Party>),
    Product(mul::Party),
    Lookup {
        ready: lut::Party,
        pending: Option<lut::Pending>,
        mask: u16,
    },
}
fn lut_error(_: lut::Error) -> Error {
    Error::Context
}
impl Op {
    fn start(&mut self, values: &[u64]) -> Result<Vec<u8>, Error> {
        match self {
            Self::Circuit(p) => p.start(values),
            Self::Product(p) => p.start(values[0], values[1]),
            Self::Lookup {
                ready,
                pending,
                mask,
            } => {
                let (p, f) = ready.start(values[0] as u16 & *mask).map_err(lut_error)?;
                *pending = Some(p);
                Ok(f.to_vec())
            }
        }
    }
    fn advance(&mut self, frame: &[u8]) -> Result<gc::Progress, Error> {
        match self {
            Self::Circuit(p) => p.advance(frame),
            Self::Product(p) => Ok(gc::Progress::Complete(Zeroizing::new(vec![
                p.finish(frame)?
            ]))),
            Self::Lookup { pending, .. } => {
                Ok(gc::Progress::Complete(Zeroizing::new(vec![u64::from(
                    pending
                        .take()
                        .ok_or(Error::Consumed)?
                        .finish(frame)
                        .map_err(lut_error)?,
                )])))
            }
        }
    }
}
#[derive(Clone, Copy)]
enum After {
    None,
    Copy(usize, usize),
    Add(usize, usize),
}
struct Step {
    op: Op,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    after: After,
    mask: u64,
}
struct Material {
    steps: VecDeque<Step>,
    registers: Zeroizing<[u64; 16]>,
}
pub struct Party {
    id: u8,
    binding: [u8; 32],
    nonce: [u8; 32],
    phase: usize,
    started: bool,
    resources: Resources,
    material: Option<Material>,
}
pub enum Progress {
    Open(Vec<u8>),
    Complete(u64),
}

struct Issuer {
    context: Context,
    bits: usize,
    counter: u64,
    queues: [VecDeque<Step>; 2],
    resources: Resources,
}
impl Issuer {
    fn child(&mut self) -> Context {
        let mut hash = Sha256::new();
        hash.update(self.context.operation);
        hash.update(self.counter.to_le_bytes());
        self.counter += 1;
        Context {
            operation: hash.finalize().into(),
            ..self.context
        }
    }
    fn record(
        &mut self,
        name: &'static str,
        keys: [usize; 2],
        phases: &[[usize; 2]],
        header: usize,
    ) {
        for (total, bytes) in self.resources.key_bytes.iter_mut().zip(keys) {
            *total += bytes;
        }
        self.resources.parts.push((name, keys));
        self.resources.peer_payload_bits +=
            8 * phases.iter().flatten().map(|v| v - header).sum::<usize>();
        self.resources
            .phases
            .extend(phases.iter().map(|f| [f[0] + HEADER, f[1] + HEADER]));
    }
    fn circuit(
        &mut self,
        name: &'static str,
        inputs: &[usize],
        outputs: &[usize],
        limit: usize,
        after: After,
        build: impl FnOnce(&mut Builder, &[Vec<BooleanWire>]) -> Result<Vec<Vec<BooleanWire>>, String>,
    ) -> Result<(), Error> {
        let ctx = self.child();
        let (parties, r) =
            gc::issue_bounded(self.bits, inputs.len(), outputs.len(), limit, ctx, build)?;
        self.record(name, r.key_bytes, &r.phase_bytes, gc::HEADER);
        self.resources.ciphertext_bytes += r.ciphertext_bytes;
        self.resources.constant_bytes += r.constant_bytes;
        self.resources.public_metadata_bytes += r.public_metadata_bytes;
        self.resources.allocation_bytes += r.allocation_bytes;
        for (queue, p) in self.queues.iter_mut().zip(parties) {
            queue.push_back(Step {
                op: Op::Circuit(Box::new(p)),
                inputs: inputs.to_vec(),
                outputs: outputs.to_vec(),
                after,
                mask: gc::mask(self.bits),
            });
        }
        Ok(())
    }
    fn rescale(
        &mut self,
        name: &'static str,
        input: usize,
        output: usize,
        shift: usize,
        after: After,
    ) -> Result<(), Error> {
        self.circuit(name, &[input], &[output], RESCALE_WIRES, after, |b, x| {
            Ok(vec![gc::rounded(b, &x[0], shift)?])
        })
    }
    fn product(&mut self, inputs: [usize; 2], output: usize) -> Result<(), Error> {
        let ctx = self.child();
        let parties = mul::issue(ctx)?;
        self.record(
            "beaver_multiply",
            [mul::KEY_BYTES; 2],
            &[[mul::FRAME_BYTES; 2]],
            mul::HEADER,
        );
        self.resources.allocation_bytes += 4096;
        self.resources.public_metadata_bytes += 2 * mul::HEADER;
        for (queue, p) in self.queues.iter_mut().zip(parties) {
            queue.push_back(Step {
                op: Op::Product(p),
                inputs: inputs.to_vec(),
                outputs: vec![output],
                after: After::None,
                mask: u64::MAX,
            });
        }
        Ok(())
    }
}

fn signed_constant(b: &mut Builder, value: i64) -> Result<Vec<BooleanWire>, String> {
    b.constant_word(value as u64 as u128, 64)
}
fn signed_extend(input: &[BooleanWire], width: usize) -> Vec<BooleanWire> {
    let mut result = input.to_vec();
    result.resize(width, *input.last().unwrap());
    result
}
fn compact_inputs(
    b: &mut Builder,
    x: &[Vec<BooleanWire>],
    profile: &compact::Profile,
) -> Result<Vec<Vec<BooleanWire>>, String> {
    let gate = gc::rounded(b, &x[0], 6)?;
    let up = gc::rounded(b, &x[1], 6)?;
    let zero = b.constant(false)?;
    let mut biased = gate[..19].to_vec();
    biased[18] = b.not(biased[18])?;
    let mut coordinate = vec![zero; 64];
    let mut coefficients = vec![vec![zero; 64]; usize::from(profile.degree()) + 1];
    for piece in profile.pieces() {
        let lower = b
            .compare_unsigned_to_constant(&biased, (piece.lower + (1 << 18)) as u128)?
            .0;
        let upper = b
            .compare_unsigned_to_constant(&biased, (piece.upper + (1 << 18)) as u128)?
            .0;
        let lower = b.not(lower)?;
        let selected = b.and(lower, upper)?;
        let mut doubled = vec![zero];
        doubled.extend(&gate[..63]);
        let offset = signed_constant(b, -i64::from(piece.lower + piece.upper))?;
        let centered = b.add_unsigned(&doubled, &offset)?.0;
        let shift = 20 - (piece.upper - piece.lower).ilog2() as usize;
        let mut normalized = vec![zero; shift];
        normalized.extend(&centered[..64 - shift]);
        coordinate = b.select_word(selected, &coordinate, &normalized)?;
        for (i, c) in coefficients.iter_mut().enumerate() {
            let public = signed_constant(b, piece.coefficients[i])?;
            *c = b.select_word(selected, c, &public)?;
        }
    }
    let before_upper = b
        .compare_unsigned_to_constant(&biased, (compact::TAIL_Q12 + (1 << 18)) as u128)?
        .0;
    let upper = b.not(before_upper)?;
    let below_lower = b
        .compare_unsigned_to_constant(&biased, (-compact::TAIL_Q12 + 1 + (1 << 18)) as u128)?
        .0;
    // Tails use a constant polynomial. Every online lane executes all products.
    let mut scaled_gate = vec![zero; 20];
    scaled_gate.extend(&gate[..44]);
    coefficients[0] = b.select_word(upper, &coefficients[0], &scaled_gate)?;
    for c in &mut coefficients {
        *c = b.select_word(below_lower, c, &[zero; 64])?;
    }
    let mut result = vec![coordinate, up];
    result.extend(coefficients);
    Ok(result)
}

pub fn issue(profile: Profile<'_>, context: Context) -> Result<([Party; 2], Resources), Error> {
    context
        .first_tensor_index
        .checked_add(1)
        .ok_or(Error::Domain)?;
    if matches!(&profile,Profile::LogRow(p) if p.levels()!=0) {
        return Err(Error::Domain);
    }
    let shifts = match &profile {
        Profile::Compact(p) => usize::from(p.degree()) + 2,
        Profile::Curl(_) => 2,
        Profile::LogRow(_) => 1,
    };
    // Full simultaneous secret owner bound BEFORE issuing the first circuit.
    let upper = gc::MAX_WIRES * 384
        + (2 << 20)
        + 1024 * 640
        + shifts * (RESCALE_WIRES * 384 + (2 << 20))
        + (1 << 20);
    if upper > MAX_ALLOCATION {
        return Err(Error::Resource);
    }
    let mut hash = Sha256::new();
    hash.update(b"pllm.mixed_gated.v1\0");
    hash.update(profile.name());
    hash.update(profile.digest());
    hash.update(context.plan);
    hash.update(context.session);
    hash.update(context.operation);
    hash.update(context.first_tensor_index.to_le_bytes());
    let binding: [u8; 32] = hash.finalize().into();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let mut child = Sha256::new();
    child.update(binding);
    child.update(nonce);
    let bits = if matches!(profile, Profile::Compact(_)) {
        64
    } else {
        32
    };
    let mut issuer = Issuer {
        context: Context {
            operation: child.finalize().into(),
            ..context
        },
        bits,
        counter: 0,
        queues: std::array::from_fn(|_| VecDeque::new()),
        resources: Resources {
            key_bytes: [0; 2],
            parts: Vec::new(),
            phases: Vec::new(),
            peer_payload_bits: 0,
            ciphertext_bytes: 0,
            constant_bytes: 0,
            public_metadata_bytes: 2 * HEADER,
            public_table_bytes: 0,
            allocation_bytes: 0,
        },
    };
    match profile {
        Profile::Compact(p) => {
            let outputs = (2..=6 + usize::from(p.degree())).collect::<Vec<_>>();
            // coordinate, up, c0..cd occupy registers 2..(4+degree).
            let outputs = &outputs[..usize::from(p.degree()) + 3];
            issuer.circuit(
                "private_selection_and_input_conversion",
                &[0, 1],
                outputs,
                gc::MAX_WIRES,
                After::Copy(4 + usize::from(p.degree()), 8),
                |b, x| compact_inputs(b, x, p),
            )?;
            for i in (0..usize::from(p.degree())).rev() {
                issuer.product([8, 2], 8)?;
                issuer.rescale("horner_rounding", 8, 8, 20, After::Add(4 + i, 8))?;
            }
            issuer.rescale("activation_rounding", 8, 8, 20, After::None)?;
            issuer.product([8, 3], 8)?;
            issuer.rescale("output_rounding", 8, 8, 12, After::None)?;
        }
        Profile::LogRow(p) => {
            issuer.circuit(
                "logrow_input_conversion",
                &[0, 1],
                &[2, 3],
                gc::MAX_WIRES,
                After::None,
                |b, x| {
                    let mut index = x[0][13..25].to_vec();
                    index[11] = b.not(index[11])?;
                    let table = p
                        .table()
                        .iter()
                        .map(|&v| v as i16 as u16)
                        .collect::<Vec<_>>();
                    let activation = b.logrow_lookup(&index, 16, &table)?;
                    Ok(vec![
                        signed_extend(&activation, 32),
                        gc::rounded(b, &x[1], 9)?,
                    ])
                },
            )?;
            issuer.product([2, 3], 8)?;
            issuer.rescale("output_rounding", 8, 8, 9, After::None)?;
        }
        Profile::Curl(p) => {
            issuer.circuit(
                "index_and_up_conversion",
                &[0, 1],
                &[2, 3],
                gc::MAX_WIRES,
                After::None,
                |b, x| {
                    let zero = b.constant(false)?;
                    let mut index = x[0][13..25].to_vec();
                    index[11] = b.not(index[11])?;
                    let mut index = index[usize::from(p.levels())..].to_vec();
                    index.resize(32, zero);
                    Ok(vec![index, gc::rounded(b, &x[1], 9)?])
                },
            )?;
            let table = lut::Table::new(&p.table().iter().map(|&v| v as u32).collect::<Vec<_>>())
                .map_err(lut_error)?;
            let parties = table.issue_reference().map_err(lut_error)?;
            issuer.record(
                "haar_lookup",
                [2 + 4 * table.rows(); 2],
                &[[lut::OPENING_BYTES; 2]],
                66,
            );
            issuer.resources.public_table_bytes = 4 * table.rows();
            issuer.resources.public_metadata_bytes += 2 * 66;
            issuer.resources.allocation_bytes += 32 * table.rows() + 4096;
            for (queue, p) in issuer.queues.iter_mut().zip(parties) {
                queue.push_back(Step {
                    op: Op::Lookup {
                        ready: p,
                        pending: None,
                        mask: (table.rows() - 1) as u16,
                    },
                    inputs: vec![2],
                    outputs: vec![2],
                    after: After::None,
                    mask: u32::MAX as u64,
                });
            }
            issuer.product([2, 3], 8)?;
            issuer.rescale("output_rounding", 8, 8, 9, After::None)?;
        }
    }
    if issuer.resources.allocation_bytes > upper {
        return Err(Error::Resource);
    }
    issuer.resources.allocation_bytes = upper;
    let resources = issuer.resources;
    let parties = issuer
        .queues
        .into_iter()
        .enumerate()
        .map(|(id, steps)| Party {
            id: id as u8,
            binding,
            nonce,
            phase: 0,
            started: false,
            resources: resources.clone(),
            material: Some(Material {
                steps,
                registers: Zeroizing::new([0; 16]),
            }),
        })
        .collect::<Vec<_>>();
    Ok((parties.try_into().ok().unwrap(), resources))
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
    fn frame(&self, body: &[u8]) -> Result<Vec<u8>, Error> {
        if HEADER + body.len() != self.resources.phases[self.phase][self.id as usize] {
            return Err(Error::Context);
        }
        let mut frame = Vec::with_capacity(HEADER + body.len());
        frame.extend(MAGIC);
        frame.extend(self.binding);
        frame.extend(self.nonce);
        frame.extend([self.phase as u8, self.id]);
        frame.extend(1_u32.to_le_bytes());
        frame.extend(body);
        Ok(frame)
    }
    fn start_step(material: &mut Material) -> Result<Vec<u8>, Error> {
        let step = material.steps.front_mut().ok_or(Error::Context)?;
        let inputs = Zeroizing::new(
            step.inputs
                .iter()
                .map(|&i| material.registers[i] & step.mask)
                .collect::<Vec<_>>(),
        );
        step.op.start(&inputs)
    }
    pub fn start(&mut self, gate: u64, up: u64) -> Result<Vec<u8>, Error> {
        let mut material = self.material.take().ok_or(Error::Consumed)?;
        if self.started {
            return Err(Error::Context);
        }
        material.registers[0] = gate;
        material.registers[1] = up;
        let frame = self.frame(&Self::start_step(&mut material)?)?;
        self.started = true;
        self.material = Some(material);
        Ok(frame)
    }
    pub fn advance(&mut self, frame: &[u8]) -> Result<Progress, Error> {
        let mut material = self.material.take().ok_or(Error::Consumed)?;
        if !self.started
            || self.phase >= self.resources.phases.len()
            || frame.len() != self.resources.phases[self.phase][1 - self.id as usize]
            || &frame[..8] != MAGIC
            || frame[8..40] != self.binding
            || frame[40..72] != self.nonce
            || frame[72] != self.phase as u8
            || frame[73] != 1 - self.id
            || frame[74..78] != 1_u32.to_le_bytes()
        {
            return Err(Error::Context);
        }
        let step = material.steps.front_mut().ok_or(Error::Context)?;
        let result = step.op.advance(&frame[HEADER..])?;
        self.phase += 1;
        let next = match result {
            gc::Progress::Open(frame) => frame,
            gc::Progress::Complete(values) => {
                if values.len() != step.outputs.len() {
                    return Err(Error::Context);
                }
                for (&index, &value) in step.outputs.iter().zip(values.iter()) {
                    material.registers[index] = value;
                }
                match step.after {
                    After::None => (),
                    After::Copy(src, dst) => material.registers[dst] = material.registers[src],
                    After::Add(src, dst) => {
                        material.registers[dst] =
                            material.registers[dst].wrapping_add(material.registers[src])
                    }
                }
                material.steps.pop_front();
                if material.steps.is_empty() {
                    if self.phase != self.resources.phases.len() {
                        return Err(Error::Context);
                    }
                    return Ok(Progress::Complete(material.registers[8]));
                }
                Self::start_step(&mut material)?
            }
        };
        if self.phase >= self.resources.phases.len() {
            return Err(Error::Context);
        }
        let frame = self.frame(&next)?;
        self.material = Some(material);
        Ok(Progress::Open(frame))
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
    fn check(profile: Profile<'_>, gate: i32, up: i32, want: i32) {
        let ([mut a, mut b], r) = issue(profile, ctx()).unwrap();
        let sg = gc::random_word().unwrap();
        let su = gc::random_word().unwrap();
        let mut frames = [
            a.start(sg, su).unwrap(),
            b.start(
                (i64::from(gate) as u64).wrapping_sub(sg),
                (i64::from(up) as u64).wrapping_sub(su),
            )
            .unwrap(),
        ];
        for (i, phase) in r.phases.iter().enumerate() {
            assert_eq!([frames[0].len(), frames[1].len()], *phase);
            match (
                a.advance(&frames[1]).unwrap(),
                b.advance(&frames[0]).unwrap(),
            ) {
                (Progress::Open(x), Progress::Open(y)) => frames = [x, y],
                (Progress::Complete(x), Progress::Complete(y)) => {
                    assert_eq!(i + 1, r.phases.len());
                    assert_eq!(x.wrapping_add(y) as i32, want);
                    assert_eq!(a.key_bytes() + b.key_bytes(), 0);
                    assert!(a.advance(&frames[1]).is_err());
                    return;
                }
                _ => panic!("inconsistent party phases"),
            }
        }
        panic!("missing final result");
    }
    #[test]
    fn complete_profiles_preserve_exact_rounding_tails_and_piece_boundaries() {
        for degree in [2, 3] {
            let profile =
                compact::Profile::fit(&[0; compact::CALIBRATION_BINS], 8, degree).unwrap();
            let mut values = vec![-48 << 18, 48 << 18, -33, -32, -31, 0, 31, 32, 33];
            for p in profile.pieces() {
                for edge in [p.lower, p.upper] {
                    values.extend([-33, -1, 0, 1, 33].map(|d| (edge << 6) + d));
                }
            }
            for (i, gate) in values.into_iter().enumerate() {
                let up = [-128 << 18, 128 << 18, -96, -95, 95, 96, 97][i % 7];
                check(
                    Profile::Compact(&profile),
                    gate,
                    up,
                    profile.evaluate(gate, up).unwrap(),
                );
            }
        }
        for levels in [0, 2, 4] {
            let profile = wavelet::Profile::new(levels).unwrap();
            for (gate, up) in [
                (-48 << 18, 128 << 18),
                (48 << 18, -128 << 18),
                (-8193, -769),
                (-8192, -768),
                (-8191, -767),
                (0, 0),
                (8191, 255),
                (8192, 256),
                (8193, 257),
            ] {
                let want = profile.evaluate(gate, up).unwrap();
                check(Profile::Curl(&profile), gate, up, want);
                if levels == 0 {
                    check(Profile::LogRow(&profile), gate, up, want);
                }
            }
        }
    }
    #[test]
    fn malformed_and_foreign_context_burn_all_remaining_block_material() {
        let p = wavelet::Profile::new(0).unwrap();
        for offset in [0, 8, 40, 72, 73, 74, HEADER, HEADER + 8] {
            let ([mut a, mut b], _) = issue(Profile::LogRow(&p), ctx()).unwrap();
            a.start(0, 0).unwrap();
            let mut f = b.start(0, 0).unwrap();
            f[offset] ^= 128;
            assert!(a.advance(&f).is_err());
            assert_eq!(a.key_bytes(), 0);
            b.cancel();
            assert_eq!(b.key_bytes(), 0);
        }
    }
}
