//! Bounded two-party Secret Replication PCF reference.
//!
//! Sections 4.1–4.2: minimal-support lines L_j(t)=t-j, two independent
//! VOLE directions, and z_i=a_i*x_i+y_{other}(x_i)-b_i. Remark 25 supplies
//! the GGM puncturing layout. This is a local dealer, not distributed setup.
//! Only full-field scalar masks admit the ordinary scalar/vector Beaver
//! consumer. One scalar opening belongs to the entire one-use vector.

use crate::shared_rescale_reference::{Context, Error};
use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128,
};
use sha2::{Digest, Sha256};
use std::time::Instant;
use zeroize::Zeroizing;

pub const MAX_LANES: usize = 4096;
pub const MAX_ALLOCATION_BYTES: usize = 16 << 20;
pub const HEADER_BYTES: usize = 78;
const MAGIC: &[u8; 8] = b"PLLMPC01";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Expanded,
    Punctured,
}

#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub field: u32,
    pub scalar_domain: u32,
    pub lanes: usize,
    pub encoding: Encoding,
}

impl Parameters {
    pub fn validate(self) -> Result<(), Error> {
        if ![2, 17, 257].contains(&self.field) || !(2..=self.field).contains(&self.scalar_domain) {
            return Err(Error::Domain);
        }
        if !(1..=MAX_LANES).contains(&self.lanes) {
            return Err(Error::Resource);
        }
        if self.allocation_bytes() > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        Ok(())
    }

    fn allocation_bytes(self) -> usize {
        // Both party key schedules, padded-tree scratch and vector/material/
        // input/output/frame overlap, with explicit small-owner slack. Bounds
        // are checked before this arithmetic; this is an estimate, not RSS.
        4096 + 2 * (2 * self.scalar_domain as usize - 1) * (32 + std::mem::size_of::<Aes128>())
            + 128 * self.scalar_domain.next_power_of_two() as usize
            + 128 * self.lanes
    }

    pub fn allocation_estimate(self) -> Result<usize, Error> {
        self.validate()?;
        Ok(self.allocation_bytes())
    }

    pub fn key_bytes_per_party(self) -> Result<usize, Error> {
        self.validate()?;
        let seeds = match self.encoding {
            Encoding::Expanded => 2 * self.scalar_domain as usize - 1,
            Encoding::Punctured => 1 + self.scalar_domain.next_power_of_two().ilog2() as usize,
        };
        Ok(4 + 16 * seeds)
    }

    /// Exact total-variation witness for an opening with an unknown uniform
    /// mask on [0, scalar_domain), conditioned on the adversary's own share.
    pub fn opening_distance(self) -> Result<(u32, u32, u32), Error> {
        self.validate()?;
        let shift = self.scalar_domain.min(self.field - self.scalar_domain);
        Ok((shift, shift, self.scalar_domain))
    }
}

fn add(a: u32, b: u32, p: u32) -> u32 {
    ((a as u64 + b as u64) % p as u64) as u32
}
fn sub(a: u32, b: u32, p: u32) -> u32 {
    ((a as u64 + p as u64 - b as u64) % p as u64) as u32
}
fn mul(a: u32, b: u32, p: u32) -> u32 {
    ((a as u64 * b as u64) % p as u64) as u32
}

fn random_word(p: u32) -> Result<u32, Error> {
    let cutoff = ((1u64 << 32) / p as u64) * p as u64;
    for _ in 0..128 {
        let mut bytes = Zeroizing::new([0; 4]);
        getrandom::fill(&mut *bytes).map_err(|_| Error::Randomness)?;
        let value = u32::from_le_bytes(*bytes);
        if (value as u64) < cutoff {
            return Ok(value % p);
        }
    }
    Err(Error::Randomness)
}

fn fresh_seed() -> Result<Zeroizing<[u8; 16]>, Error> {
    let mut seed = Zeroizing::new([0; 16]);
    getrandom::fill(&mut *seed).map_err(|_| Error::Randomness)?;
    Ok(seed)
}

/// Independent AES-keyed GGM child expansion; this is not the FSS DPF PRG.
fn children(seed: &[u8; 16]) -> [Zeroizing<[u8; 16]>; 2] {
    let cipher = Aes128::new(seed.into());
    [0, 1].map(|branch| {
        let mut block = aes::cipher::Block::<Aes128>::from(*b"PLLM.PCF.tree.v1");
        block[15] ^= branch;
        cipher.encrypt_block(&mut block);
        let output = Zeroizing::new(block.into());
        block.fill(0);
        output
    })
}

struct Node {
    first: usize,
    size: usize,
    seed: Zeroizing<[u8; 16]>,
}

fn fill_leaves(node: Node, output: &mut [Option<Zeroizing<[u8; 16]>>]) {
    if node.size == 1 {
        if node.first < output.len() {
            output[node.first] = Some(node.seed);
        }
        return;
    }
    let [left, right] = children(&node.seed);
    let size = node.size / 2;
    fill_leaves(
        Node {
            first: node.first,
            size,
            seed: left,
        },
        output,
    );
    fill_leaves(
        Node {
            first: node.first + size,
            size,
            seed: right,
        },
        output,
    );
}

fn puncture(root: &[u8; 16], point: usize, size: usize) -> Vec<Node> {
    let mut current = Node {
        first: 0,
        size,
        seed: Zeroizing::new(*root),
    };
    let mut nodes = Vec::with_capacity(size.ilog2() as usize);
    while current.size > 1 {
        let [left, right] = children(&current.seed);
        let half = current.size / 2;
        let split = current.first + half;
        if point < split {
            nodes.push(Node {
                first: split,
                size: half,
                seed: right,
            });
            current = Node {
                first: current.first,
                size: half,
                seed: left,
            };
        } else {
            nodes.push(Node {
                first: current.first,
                size: half,
                seed: left,
            });
            current = Node {
                first: split,
                size: half,
                seed: right,
            };
        }
    }
    nodes
}

enum Seeds {
    Expanded(Vec<Option<Zeroizing<[u8; 16]>>>),
    Tree(Vec<Node>),
}

impl Seeds {
    fn leaves(self, domain: usize) -> Vec<Option<Zeroizing<[u8; 16]>>> {
        match self {
            Self::Expanded(raw) => raw,
            Self::Tree(nodes) => {
                let mut raw: Vec<_> = (0..domain).map(|_| None).collect();
                for node in nodes {
                    fill_leaves(node, &mut raw);
                }
                raw
            }
        }
    }

    fn expand(self, domain: usize) -> Vec<Option<Aes128>> {
        self.leaves(domain)
            .into_iter()
            .map(|key| key.map(|key| Aes128::new((&*key).into())))
            .collect()
    }
}

fn sender(root: &[u8; 16], p: Parameters) -> Seeds {
    let nodes = vec![Node {
        first: 0,
        size: p.scalar_domain.next_power_of_two() as usize,
        seed: Zeroizing::new(*root),
    }];
    if p.encoding == Encoding::Punctured {
        return Seeds::Tree(nodes);
    }
    let mut raw: Vec<_> = (0..p.scalar_domain).map(|_| None).collect();
    for node in nodes {
        fill_leaves(node, &mut raw);
    }
    Seeds::Expanded(raw)
}

fn receiver(root: &[u8; 16], point: u32, p: Parameters) -> Seeds {
    let nodes = puncture(
        root,
        point as usize,
        p.scalar_domain.next_power_of_two() as usize,
    );
    if p.encoding == Encoding::Punctured {
        return Seeds::Tree(nodes);
    }
    let mut raw: Vec<_> = (0..p.scalar_domain).map(|_| None).collect();
    for node in nodes {
        fill_leaves(node, &mut raw);
    }
    Seeds::Expanded(raw)
}

fn prf(cipher: &Aes128, namespace: &[u8; 8], index: usize, p: u32) -> Result<u32, Error> {
    let cutoff = ((1u64 << 32) / p as u64) * p as u64;
    for attempt in 0..128u32 {
        let mut raw = Zeroizing::new([0u8; 16]);
        raw[..8].copy_from_slice(namespace);
        raw[8..12].copy_from_slice(&(index as u32).to_le_bytes());
        raw[12..].copy_from_slice(&attempt.to_le_bytes());
        let mut block = aes::cipher::Block::<Aes128>::from(*raw);
        cipher.encrypt_block(&mut block);
        let value = u32::from_le_bytes(block[..4].try_into().unwrap());
        block.fill(0);
        if (value as u64) < cutoff {
            return Ok(value % p);
        }
    }
    Err(Error::Randomness)
}

struct Key {
    scalar: Zeroizing<u32>,
    sender: Seeds,
    receiver: Seeds,
}
struct Share {
    scalar: Zeroizing<u32>,
    vectors: Zeroizing<Vec<[u32; 2]>>,
}
enum State {
    Key(Key),
    Prepared(Share),
    Waiting(Share, Zeroizing<Vec<u32>>, Zeroizing<u32>),
}

pub struct Party {
    id: u8,
    parameters: Parameters,
    binding: [u8; 32],
    nonce: [u8; 32],
    state: Option<State>,
}

pub fn issue_reference(p: Parameters, context: Context) -> Result<[Party; 2], Error> {
    p.validate()?;
    context
        .first_tensor_index
        .checked_add(p.lanes as u64)
        .ok_or(Error::Domain)?;
    let mut hash = Sha256::new();
    hash.update(b"pllm.secret_replication.v1\0");
    hash.update(p.field.to_le_bytes());
    hash.update(p.scalar_domain.to_le_bytes());
    hash.update((p.lanes as u64).to_le_bytes());
    hash.update([p.encoding as u8]);
    hash.update(context.plan);
    hash.update(context.session);
    hash.update(context.operation);
    hash.update(context.first_tensor_index.to_le_bytes());
    let binding = hash.finalize().into();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let roots = [fresh_seed()?, fresh_seed()?];
    let scalars = Zeroizing::new([random_word(p.scalar_domain)?, random_word(p.scalar_domain)?]);
    Ok([0, 1].map(|id| Party {
        id,
        parameters: p,
        binding,
        nonce,
        state: Some(State::Key(Key {
            scalar: Zeroizing::new(scalars[id as usize]),
            sender: sender(&roots[id as usize], p),
            receiver: receiver(&roots[1 - id as usize], scalars[id as usize], p),
        })),
    }))
}

impl Party {
    pub fn cancel(&mut self) {
        self.state = None;
    }

    pub fn prepare(&mut self) -> Result<(), Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        let State::Key(key) = state else {
            return Err(Error::Context);
        };
        let p = self.parameters;
        let sender = key.sender.expand(p.scalar_domain as usize);
        let receiver = key.receiver.expand(p.scalar_domain as usize);
        let mut hash = Sha256::new();
        hash.update(b"pllm.secret_replication.prf.v1\0");
        hash.update(self.binding);
        hash.update(self.nonce);
        let namespace: [u8; 8] = hash.finalize()[..8].try_into().unwrap();
        let mut vectors = Zeroizing::new(Vec::with_capacity(p.lanes));
        for index in 0..p.lanes {
            let mut coefficients = Zeroizing::new([0u32; 3]); // a, b, y_other(x_i)
            for (j, cipher) in sender.iter().enumerate() {
                let value = prf(
                    cipher.as_ref().ok_or(Error::Context)?,
                    &namespace,
                    index,
                    p.field,
                )?;
                coefficients[0] = add(coefficients[0], value, p.field);
                coefficients[1] = sub(coefficients[1], mul(j as u32, value, p.field), p.field);
            }
            for (j, cipher) in receiver.iter().enumerate() {
                if let Some(cipher) = cipher {
                    let value = prf(cipher, &namespace, index, p.field)?;
                    coefficients[2] = add(
                        coefficients[2],
                        mul(sub(*key.scalar, j as u32, p.field), value, p.field),
                        p.field,
                    );
                } else if j != *key.scalar as usize {
                    return Err(Error::Context);
                }
            }
            vectors.push([
                coefficients[0],
                sub(
                    add(
                        mul(coefficients[0], *key.scalar, p.field),
                        coefficients[2],
                        p.field,
                    ),
                    coefficients[1],
                    p.field,
                ),
            ]);
        }
        self.state = Some(State::Prepared(Share {
            scalar: key.scalar,
            vectors,
        }));
        Ok(())
    }

    pub fn start(&mut self, scalar: u32, vector: &[u32]) -> Result<Vec<u8>, Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        let State::Prepared(share) = state else {
            return Err(Error::Context);
        };
        let p = self.parameters;
        if p.scalar_domain != p.field
            || scalar >= p.field
            || vector.len() != p.lanes
            || vector.iter().any(|v| *v >= p.field)
        {
            return Err(Error::Domain);
        }
        let scalar_opening = Zeroizing::new(sub(scalar, *share.scalar, p.field));
        let vector_opening = Zeroizing::new(
            vector
                .iter()
                .zip(share.vectors.iter())
                .map(|(v, pair)| sub(*v, pair[0], p.field))
                .collect::<Vec<_>>(),
        );
        let mut frame = Vec::with_capacity(HEADER_BYTES + 4 * (p.lanes + 1));
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&self.binding);
        frame.extend_from_slice(&self.nonce);
        frame.extend_from_slice(&[0, self.id]);
        frame.extend_from_slice(&(p.lanes as u32).to_le_bytes());
        frame.extend_from_slice(&scalar_opening.to_le_bytes());
        for value in vector_opening.iter() {
            frame.extend_from_slice(&value.to_le_bytes());
        }
        self.state = Some(State::Waiting(share, vector_opening, scalar_opening));
        Ok(frame)
    }

    pub fn finish(&mut self, peer: &[u8]) -> Result<Zeroizing<Vec<u32>>, Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        let State::Waiting(share, own_vector, own_scalar) = state else {
            return Err(Error::Context);
        };
        let p = self.parameters;
        if peer.len() != HEADER_BYTES + 4 * (p.lanes + 1)
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != self.nonce
            || peer[72] != 0
            || peer[73] != 1 - self.id
            || u32::from_le_bytes(peer[74..78].try_into().unwrap()) as usize != p.lanes
        {
            return Err(Error::Context);
        }
        let scalar = u32::from_le_bytes(peer[78..82].try_into().unwrap());
        if scalar >= p.field {
            return Err(Error::Domain);
        }
        let scalar = add(scalar, *own_scalar, p.field);
        let mut output = Zeroizing::new(Vec::with_capacity(p.lanes));
        for ((raw, own), pair) in peer[82..]
            .chunks_exact(4)
            .zip(own_vector.iter())
            .zip(share.vectors.iter())
        {
            let vector = u32::from_le_bytes(raw.try_into().unwrap());
            if vector >= p.field {
                return Err(Error::Domain);
            }
            let vector = add(vector, *own, p.field);
            let cross = if self.id == 0 {
                mul(vector, scalar, p.field)
            } else {
                0
            };
            output.push(add(
                add(pair[1], mul(vector, *share.scalar, p.field), p.field),
                add(mul(scalar, pair[0], p.field), cross, p.field),
                p.field,
            ));
        }
        Ok(output)
    }
}

pub struct ReferenceResult {
    pub outputs: Option<Zeroizing<Vec<u32>>>,
    pub checked_correlations: usize,
    pub key_bytes_per_party: usize,
    pub expanded_correlation_bytes_per_party: usize,
    pub prf_field_evaluations_per_party: usize,
    pub peer_bytes_per_party: usize,
    pub issuance_seconds: f64,
    pub expansion_seconds: f64,
    pub online_seconds: f64,
}

/// Local test dealer plus two separate party-local states; no secret state is
/// returned. Inputs use one scalar for the entire vector, not one per element.
pub fn run_reference(
    p: Parameters,
    inputs: Option<(u32, &[u32])>,
    context: Context,
) -> Result<ReferenceResult, Error> {
    p.validate()?;
    if let Some((scalar, vector)) = inputs {
        if p.scalar_domain != p.field
            || scalar >= p.field
            || vector.len() != p.lanes
            || vector.iter().any(|v| *v >= p.field)
        {
            return Err(Error::Domain);
        }
    }
    let start = Instant::now();
    let [mut left, mut right] = issue_reference(p, context)?;
    let issuance_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    left.prepare()?;
    right.prepare()?;
    let expansion_seconds = start.elapsed().as_secs_f64();
    // Test-only observation of both shares checks the defining correlation;
    // production party code above never reads its peer's scalar or vectors.
    let (Some(State::Prepared(a)), Some(State::Prepared(b))) = (&left.state, &right.state) else {
        return Err(Error::Context);
    };
    let scalar = add(*a.scalar, *b.scalar, p.field);
    if a.vectors.iter().zip(b.vectors.iter()).any(|(a, b)| {
        (add(a[0], b[0], p.field) as u64 * scalar as u64 % p.field as u64) as u32
            != add(a[1], b[1], p.field)
    }) {
        return Err(Error::Context);
    }
    let start = Instant::now();
    let (outputs, peer_bytes_per_party) = if let Some((scalar, vector)) = inputs {
        let scalar_share = Zeroizing::new(random_word(p.field)?);
        let vector_share = Zeroizing::new(
            (0..p.lanes)
                .map(|_| random_word(p.field))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let other_vector = Zeroizing::new(
            vector
                .iter()
                .zip(vector_share.iter())
                .map(|(v, s)| sub(*v, *s, p.field))
                .collect::<Vec<_>>(),
        );
        let first = left.start(*scalar_share, &vector_share)?;
        let second = right.start(sub(scalar, *scalar_share, p.field), &other_vector)?;
        let a = left.finish(&second)?;
        let b = right.finish(&first)?;
        (
            Some(Zeroizing::new(
                a.iter()
                    .zip(b.iter())
                    .map(|(a, b)| add(*a, *b, p.field))
                    .collect(),
            )),
            first.len(),
        )
    } else {
        left.cancel();
        right.cancel();
        (None, 0)
    };
    Ok(ReferenceResult {
        outputs,
        checked_correlations: p.lanes,
        key_bytes_per_party: p.key_bytes_per_party()?,
        expanded_correlation_bytes_per_party: 4 + 8 * p.lanes,
        prf_field_evaluations_per_party: p.lanes * (2 * p.scalar_domain as usize - 1),
        peer_bytes_per_party,
        issuance_seconds,
        expansion_seconds,
        online_seconds: start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests;
