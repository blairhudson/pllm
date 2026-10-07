//! Complete bounded Curl Fig. 6 Haar lookup composed with exact shared
//! rescaling and a fresh Beaver product. Each evaluator owns only one share.
//! Exact floor/ties-even replace the paper's probabilistic truncation. Trusted
//! input conversion/range checks, in-process issuance and unauthenticated local
//! frames are explicit research boundaries, not a private decoder deployment.
use crate::shared_rescale_reference::{
    self as rs, Backend, Context, Descriptor, Error, Layout, Rounding,
};
use crate::{shared_arithmetic_reference as mul, shared_lut_reference as lut};
use pllm_core::wavelet_gated_reference::{
    Profile, INDEX_FRACTION, INPUT_FRACTION, MAX_LEVELS, OUTPUT_FRACTION, TABLE_OFFSET, TABLE_ROWS,
};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

pub const MAX_LANES: usize = 256;
pub const MAX_ALLOCATION: usize = 64 << 20;
pub const HEADER: usize = 78;
pub const ROUNDS: usize = 4;
const MAGIC: &[u8; 8] = b"PLLMCG01";

pub struct Resources {
    pub party_key_payload_bytes: usize,
    /// Index rescale, up rescale, lookup, multiplication, output rescale.
    pub party_key_bytes_by_operation: [usize; 5],
    pub allocation_estimate_bytes: usize,
    pub phase_frame_bytes_per_party: [usize; ROUNDS],
    pub public_table_bytes_per_party: usize,
}

fn descriptor(shift: u8, rounding: Rounding) -> Descriptor {
    Descriptor::new(32, shift, rounding)
        .unwrap()
        .with_backend(Backend::IntervalDcf)
}

pub fn resources(levels: u8, lanes: usize) -> Result<Resources, Error> {
    if levels > MAX_LEVELS || !(1..=MAX_LANES).contains(&lanes) {
        return Err(Error::Resource);
    }
    let index = descriptor(INPUT_FRACTION - INDEX_FRACTION + levels, Rounding::Floor)
        .resources(Layout::Fused, lanes)?;
    let up = descriptor(INPUT_FRACTION - OUTPUT_FRACTION, Rounding::TiesEven)
        .resources(Layout::Fused, lanes)?;
    let output = descriptor(OUTPUT_FRACTION, Rounding::TiesEven).resources(Layout::Fused, lanes)?;
    let rows = TABLE_ROWS >> levels;
    let lookup = lanes * (2 + 4 * rows);
    let parts = [
        index.party_key_payload_bytes,
        up.party_key_payload_bytes,
        lookup,
        12 * lanes,
        output.party_key_payload_bytes,
    ];
    let phases = [
        HEADER + 2 * (rs::OPENING_HEADER_BYTES + 4 * lanes),
        HEADER + lut::OPENING_BYTES * lanes,
        HEADER + mul::HEADER_BYTES + 8 * lanes,
        HEADER + rs::OPENING_HEADER_BYTES + 4 * lanes,
    ];
    let allocation = index.total_allocation_estimate_bytes
        + up.total_allocation_estimate_bytes
        + output.total_allocation_estimate_bytes
        + 6 * lookup
        + 8192 * lanes
        + 4 * phases.iter().sum::<usize>()
        + 256 * 1024;
    if allocation > MAX_ALLOCATION {
        return Err(Error::Resource);
    }
    Ok(Resources {
        party_key_payload_bytes: parts.iter().sum(),
        party_key_bytes_by_operation: parts,
        allocation_estimate_bytes: allocation,
        phase_frame_bytes_per_party: phases,
        public_table_bytes_per_party: 4 * rows,
    })
}

struct Material {
    index: rs::Party,
    up: rs::Party,
    lookup: Vec<lut::Party>,
    pending: Vec<lut::Pending>,
    product: mul::Party,
    output: rs::Party,
    up_values: Zeroizing<Vec<u32>>,
    step: u8,
}

/// Non-cloneable party-local state. Any malformed frame drops every unused key.
pub struct Party {
    id: u8,
    lanes: usize,
    index_mask: u32,
    binding: [u8; 32],
    nonce: [u8; 32],
    resources: Resources,
    material: Option<Material>,
}
pub enum Progress {
    Open(Vec<u8>),
    Complete(Zeroizing<Vec<u32>>),
}

fn lut_error(error: lut::Error) -> Error {
    match error {
        lut::Error::Domain => Error::Domain,
        lut::Error::Randomness => Error::Randomness,
        lut::Error::Consumed => Error::Consumed,
        lut::Error::Context => Error::Context,
    }
}
fn rescaled(mut values: Vec<rs::OutputShare>) -> Zeroizing<Vec<u32>> {
    let result = Zeroizing::new(values.iter().map(|v| v.rescaled).collect());
    for value in &mut values {
        value.rescaled.zeroize();
        value.nonnegative.zeroize();
    }
    result
}
fn finish_rescale(party: &mut rs::Party, peer: &[u8]) -> Result<Zeroizing<Vec<u32>>, Error> {
    match party.advance(peer)? {
        rs::Progress::Complete(values) => Ok(rescaled(values)),
        _ => Err(Error::Context),
    }
}

pub fn issue_reference(
    profile: &Profile,
    context: Context,
    lanes: usize,
) -> Result<[Party; 2], Error> {
    resources(profile.levels(), lanes)?; // Aggregate preflight before any key prefix.
    context
        .first_tensor_index
        .checked_add(lanes as u64)
        .ok_or(Error::Domain)?;
    let mut hash = Sha256::new();
    hash.update(b"pllm.wavelet_gated_reference.q18_q9.v1\0");
    hash.update(profile.digest());
    hash.update(context.plan);
    hash.update(context.session);
    hash.update(context.operation);
    hash.update(context.first_tensor_index.to_le_bytes());
    hash.update((lanes as u64).to_le_bytes());
    let binding: [u8; 32] = hash.finalize().into();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let child = |step| {
        let mut hash = Sha256::new();
        hash.update(binding);
        hash.update(nonce);
        hash.update([step]);
        Context {
            operation: hash.finalize().into(),
            ..context
        }
    };
    let [ai, bi] = descriptor(profile.index_shift(), Rounding::Floor).issue_reference(
        Layout::Fused,
        child(0),
        lanes,
    )?;
    let [au, bu] = descriptor(INPUT_FRACTION - OUTPUT_FRACTION, Rounding::TiesEven)
        .issue_reference(Layout::Fused, child(1), lanes)?;
    let table = lut::Table::new(
        &profile
            .table()
            .iter()
            .map(|&v| v as u32)
            .collect::<Vec<_>>(),
    )
    .map_err(lut_error)?;
    let mut lookups = [Vec::with_capacity(lanes), Vec::with_capacity(lanes)];
    for _ in 0..lanes {
        let [a, b] = table.issue_reference().map_err(lut_error)?;
        lookups[0].push(a);
        lookups[1].push(b);
    }
    let [ap, bp] = mul::issue_reference(32, lanes, child(2))?;
    let [ao, bo] = descriptor(OUTPUT_FRACTION, Rounding::TiesEven).issue_reference(
        Layout::Fused,
        child(3),
        lanes,
    )?;
    let [al, bl] = lookups;
    let make = |id, index, up, lookup, product, output| -> Result<Party, Error> {
        Ok(Party {
            id,
            lanes,
            index_mask: profile.table().len() as u32 - 1,
            binding,
            nonce,
            resources: resources(profile.levels(), lanes)?,
            material: Some(Material {
                index,
                up,
                lookup,
                pending: Vec::new(),
                product,
                output,
                up_values: Zeroizing::new(Vec::new()),
                step: 0,
            }),
        })
    };
    Ok([make(0, ai, au, al, ap, ao)?, make(1, bi, bu, bl, bp, bo)?])
}

impl Party {
    pub fn cancel(&mut self) {
        self.material = None;
    }
    /// Remaining encoded secret payload; not a measured allocation sample.
    pub fn key_payload_bytes(&self) -> usize {
        self.material.as_ref().map_or(0, |m| {
            m.index.key_payload_bytes()
                + m.up.key_payload_bytes()
                + m.lookup
                    .iter()
                    .map(lut::Party::key_payload_bytes)
                    .sum::<usize>()
                + m.pending
                    .iter()
                    .map(lut::Pending::key_payload_bytes)
                    .sum::<usize>()
                + m.product.key_payload_bytes()
                + m.output.key_payload_bytes()
        })
    }
    fn frame(&self, phase: u8, body: &[u8]) -> Result<Vec<u8>, Error> {
        if phase as usize >= ROUNDS
            || HEADER + body.len() != self.resources.phase_frame_bytes_per_party[phase as usize]
        {
            return Err(Error::Context);
        }
        let mut out = Vec::with_capacity(HEADER + body.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.binding);
        out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&[phase, self.id]);
        out.extend_from_slice(&(self.lanes as u32).to_le_bytes());
        out.extend_from_slice(body);
        Ok(out)
    }
    fn peer<'a>(&self, phase: u8, frame: &'a [u8]) -> Result<&'a [u8], Error> {
        if phase as usize >= ROUNDS
            || frame.len() != self.resources.phase_frame_bytes_per_party[phase as usize]
            || &frame[..8] != MAGIC
            || frame[8..40] != self.binding
            || frame[40..72] != self.nonce
            || frame[72] != phase
            || frame[73] != 1 - self.id
            || u32::from_le_bytes(frame[74..78].try_into().unwrap()) as usize != self.lanes
        {
            return Err(Error::Context);
        }
        Ok(&frame[HEADER..])
    }
    pub fn start(&mut self, gate: &[u32], up: &[u32]) -> Result<Vec<u8>, Error> {
        let mut m = self.material.take().ok_or(Error::Consumed)?;
        if m.step != 0 || gate.len() != self.lanes || up.len() != self.lanes {
            return Err(Error::Domain);
        }
        // The public offset is added to one share only. Secret indices are
        // derived after protected truncation, never by shifting each share.
        let shifted = Zeroizing::new(
            gate.iter()
                .map(|&v| {
                    v.wrapping_add(if self.id == 0 {
                        (TABLE_OFFSET << INPUT_FRACTION) as u32
                    } else {
                        0
                    })
                })
                .collect::<Vec<_>>(),
        );
        let mut body = m.index.start(&shifted)?;
        body.extend(m.up.start(up)?);
        let frame = self.frame(0, &body)?;
        m.step = 1;
        self.material = Some(m);
        Ok(frame)
    }
    pub fn advance(&mut self, frame: &[u8]) -> Result<Progress, Error> {
        let mut m = self.material.take().ok_or(Error::Consumed)?;
        if m.step == 0 {
            return Err(Error::Context);
        }
        let body = self.peer(m.step - 1, frame)?;
        let next = match m.step {
            1 => {
                let split = rs::OPENING_HEADER_BYTES + 4 * self.lanes;
                let indices = finish_rescale(&mut m.index, &body[..split])?;
                m.up_values = finish_rescale(&mut m.up, &body[split..])?;
                let mut out = Vec::with_capacity(self.lanes * lut::OPENING_BYTES);
                for (key, &index) in m.lookup.iter_mut().zip(indices.iter()) {
                    let (pending, opening) = key
                        .start((index & self.index_mask) as u16)
                        .map_err(lut_error)?;
                    m.pending.push(pending);
                    out.extend_from_slice(&opening);
                }
                m.lookup.clear();
                out
            }
            2 => {
                let mut activated = Zeroizing::new(Vec::with_capacity(self.lanes));
                for (pending, opening) in m
                    .pending
                    .drain(..)
                    .zip(body.chunks_exact(lut::OPENING_BYTES))
                {
                    activated.push(pending.finish(opening).map_err(lut_error)?);
                }
                let out = m.product.start(&activated, &m.up_values)?;
                m.up_values.zeroize();
                out
            }
            3 => {
                let product = m.product.finish(body)?;
                m.output.start(&product)?
            }
            4 => return Ok(Progress::Complete(finish_rescale(&mut m.output, body)?)),
            _ => return Err(Error::Context),
        };
        let out = self.frame(m.step, &next)?;
        m.step += 1;
        self.material = Some(m);
        Ok(Progress::Open(out))
    }
}

#[cfg(test)]
mod tests;
