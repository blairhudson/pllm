//! Bounded one-use projected polynomial correlations. Research only.
//!
//! Evaluates D[((Gx)^2 + 256 Gx) Ux] modulo 2^k. No intermediate
//! truncation, checkpoint quality, transport authentication or malicious security.
//! Public matrix execution reuses pllm-core's persistent SIMD executor. Material,
//! expansion, packing and consume/burn transitions remain in Rust.

use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes256,
};
use pllm_core::{Executor, Matrix};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
use zeroize::Zeroizing;

const DOMAIN: &[u8] = b"pllm.projected_polynomial_correlations.v1\0";
const HEADER: usize = 41;
const KEY: &[u8; 8] = b"PLLMPC01";
const OPEN: &[u8; 8] = b"PLLMPO01";
const MAX_MATERIAL: usize = 16 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    Dense,
    Derived,
    Contracted,
    Seeded,
}

#[derive(Clone, Copy)]
pub struct Dimensions {
    pub rows: usize,
    pub hidden: usize,
    pub channels: usize,
    pub outputs: usize,
    pub ring_bits: u8,
}

/// Shape-only cost record. Passing this bound does not authorize issuance.
pub struct Cost {
    pub material_bytes: [usize; 2],
    pub opening_bytes: [usize; 2],
    pub dealer_matrix_macs: usize,
    pub online_matrix_macs: usize,
    pub offset_matrix_macs: usize,
}

pub fn estimate(d: Dimensions, layout: Layout) -> Result<Cost, String> {
    if !matches!(d.ring_bits, 24 | 32 | 64)
        || [d.rows, d.hidden, d.channels, d.outputs]
            .iter()
            .any(|v| !(1..=1 << 20).contains(v))
    {
        return Err("projected polynomial cost dimensions or ring are invalid".into());
    }
    let multiply = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| "polynomial cost overflow".to_string())
    };
    let channel_words = match layout {
        Layout::Dense => 5 * d.channels,
        Layout::Derived => 3 * d.channels,
        _ => 2 * d.channels + d.outputs,
    };
    let word = usize::from(d.ring_bits / 8);
    let source_bytes = multiply(multiply(d.rows, d.hidden)?, word)?;
    let coefficient_bytes = multiply(multiply(d.rows, channel_words)?, word)?;
    let expanded = HEADER
        .checked_add(source_bytes)
        .and_then(|v| v.checked_add(coefficient_bytes))
        .ok_or("polynomial cost overflow")?;
    let material_bytes = if layout == Layout::Seeded {
        [
            HEADER + 64,
            (HEADER + 32)
                .checked_add(coefficient_bytes)
                .ok_or("polynomial cost overflow")?,
        ]
    } else {
        [expanded; 2]
    };
    if material_bytes.iter().any(|&v| v as u64 > 1u64 << 40) {
        return Err("polynomial cost exceeds bounded body domain".into());
    }
    let gate_macs = multiply(multiply(d.rows, d.hidden)?, d.channels)?;
    let down_macs = multiply(multiply(d.rows, d.outputs)?, d.channels)?;
    let offset_matrix_macs = multiply(gate_macs, 4)?
        .checked_add(multiply(down_macs, 2)?)
        .ok_or("polynomial cost overflow")?;
    let online_matrix_macs = offset_matrix_macs
        .checked_add(if layout == Layout::Dense {
            0
        } else {
            multiply(gate_macs, 4)?
        })
        .ok_or("polynomial cost overflow")?;
    let dealer_matrix_macs = multiply(gate_macs, 2)?
        .checked_add(if matches!(layout, Layout::Contracted | Layout::Seeded) {
            down_macs
        } else {
            0
        })
        .ok_or("polynomial cost overflow")?;
    Ok(Cost {
        material_bytes,
        opening_bytes: [HEADER + source_bytes; 2],
        dealer_matrix_macs,
        online_matrix_macs,
        offset_matrix_macs,
    })
}

/// Public immutable contract. No secret state and no Python dependency.
pub struct Contract {
    dims: Dimensions,
    layout: Layout,
    binding: [u8; 32],
    digest: [u8; 32],
    weights: [Matrix; 3],
    executor: Executor,
}

impl Contract {
    pub fn new(
        d: Dimensions,
        layout: Layout,
        binding: [u8; 32],
        weights: [&[u8]; 3],
    ) -> Result<Arc<Self>, String> {
        if !(1..=16).contains(&d.rows)
            || !(1..=64).contains(&d.hidden)
            || !(1..=128).contains(&d.channels)
            || !(1..=64).contains(&d.outputs)
            || !matches!(d.ring_bits, 24 | 32 | 64)
        {
            return Err("projected polynomial dimensions or ring exceed reference bounds".into());
        }
        if weights[0].len() != d.channels * d.hidden
            || weights[1].len() != weights[0].len()
            || weights[2].len() != d.outputs * d.channels
        {
            return Err("projected polynomial weights have invalid shape".into());
        }
        let mut hash = Sha256::new();
        hash.update(DOMAIN);
        hash.update(binding);
        hash.update([layout as u8, d.ring_bits]);
        for value in [d.rows, d.hidden, d.channels, d.outputs] {
            hash.update((value as u64).to_le_bytes());
        }
        for weight in weights {
            hash.update(weight);
        }
        Ok(Arc::new(Self {
            dims: d,
            layout,
            binding,
            digest: hash.finalize().into(),
            weights: [
                Matrix::new(weights[0], d.channels, d.hidden)?,
                Matrix::new(weights[1], d.channels, d.hidden)?,
                Matrix::new(weights[2], d.outputs, d.channels)?,
            ],
            executor: Executor::new(1, true)?,
        }))
    }

    fn mask(&self) -> u64 {
        u64::MAX >> (64 - self.dims.ring_bits)
    }
    fn word(&self) -> usize {
        usize::from(self.dims.ring_bits / 8)
    }
    fn coefficient_width(&self) -> usize {
        match self.layout {
            Layout::Dense => 5 * self.dims.channels,
            Layout::Derived => 3 * self.dims.channels,
            _ => 2 * self.dims.channels + self.dims.outputs,
        }
    }
    pub fn material_sizes(&self) -> [usize; 2] {
        estimate(self.dims, self.layout)
            .expect("validated bounded contract")
            .material_bytes
    }
    fn project(&self, which: usize, values: &[u64]) -> Result<Zeroizing<Vec<u64>>, String> {
        let result = if self.dims.ring_bits == 64 {
            self.weights[which].wrap64(&self.executor, values, self.dims.rows)?
        } else {
            let narrow = Zeroizing::new(values.iter().map(|&v| v as u32).collect::<Vec<_>>());
            self.weights[which]
                .wrap32(&self.executor, &narrow, self.dims.rows)?
                .into_iter()
                .map(|v| u64::from(v) & self.mask())
                .collect()
        };
        Ok(Zeroizing::new(result))
    }
    fn valid_input(&self, input: &[u64]) -> bool {
        input.len() == self.dims.rows * self.dims.hidden && input.iter().all(|&v| v <= self.mask())
    }
    fn expand(&self, seed: &[u8], label: u8, count: usize) -> Zeroizing<Vec<u64>> {
        // AES-256 counter expansion with a domain/context/role-derived key.
        // Mask and coefficient streams use independent fresh 256-bit seeds.
        let mut hash = Sha256::new();
        hash.update(DOMAIN);
        hash.update(self.digest);
        hash.update([label]);
        hash.update(seed);
        let key = Zeroizing::new(<[u8; 32]>::from(hash.finalize()));
        let cipher = Aes256::new_from_slice(key.as_ref()).expect("fixed AES key width");
        let mut values = Zeroizing::new(Vec::with_capacity(count));
        for counter in 0..count.div_ceil(2) {
            let mut block = aes::cipher::Block::<Aes256>::from((counter as u128).to_le_bytes());
            cipher.encrypt_block(&mut block);
            for part in block.chunks_exact(8) {
                if values.len() < count {
                    values.push(
                        u64::from_le_bytes(part.try_into().expect("eight bytes")) & self.mask(),
                    );
                }
            }
        }
        values
    }
    fn frame(&self, magic: &[u8; 8], party: u8, body: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(HEADER + body.len());
        result.extend_from_slice(magic);
        result.extend_from_slice(&self.digest);
        result.push(party);
        result.extend_from_slice(body);
        result
    }
    fn check_frame<'a>(
        &self,
        frame: &'a [u8],
        magic: &[u8; 8],
        party: u8,
        length: usize,
    ) -> Result<&'a [u8], String> {
        if frame.len() != HEADER + length
            || &frame[..8] != magic
            || frame[8..40] != self.digest
            || frame[40] != party
        {
            return Err("projected polynomial frame length or commitments differ".into());
        }
        Ok(&frame[HEADER..])
    }
    fn pack(&self, values: &[u64]) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes().into_iter().take(self.word()))
                .collect(),
        )
    }
    fn unpack(&self, bytes: &[u8]) -> Zeroizing<Vec<u64>> {
        Zeroizing::new(
            bytes
                .chunks_exact(self.word())
                .map(|part| {
                    let mut value = [0; 8];
                    value[..part.len()].copy_from_slice(part);
                    u64::from_le_bytes(value)
                })
                .collect(),
        )
    }
}

/// Test-only dealer. Opaque parties contain no other party's seeds or shares.
pub struct Dealer {
    limit: usize,
    pub issued_bytes: usize,
    issued: BTreeSet<[u8; 32]>,
    cancelled: Arc<AtomicBool>,
}

impl Dealer {
    pub fn new(limit: usize) -> Result<Self, String> {
        if !(1..=MAX_MATERIAL).contains(&limit) {
            return Err("projected polynomial budget exceeds reference bound".into());
        }
        Ok(Self {
            limit,
            issued_bytes: 0,
            issued: BTreeSet::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
        })
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    pub fn issue(&mut self, contract: Arc<Contract>) -> Result<[Party; 2], String> {
        if self.cancelled.load(Ordering::Acquire) || self.issued.contains(&contract.binding) {
            return Err("projected polynomial issuance is cancelled or replayed".into());
        }
        let total: usize = contract.material_sizes().iter().sum();
        if total > self.limit - self.issued_bytes {
            return Err("aggregate projected polynomial budget exceeded before issuance".into());
        }
        // Preflight before entropy, secret allocation or any returned prefix.
        self.issued.insert(contract.binding);
        self.issued_bytes += total;
        let mut seeds = Zeroizing::new([0; 96]);
        getrandom::fill(seeds.as_mut()).map_err(|e| e.to_string())?;
        let d = contract.dims;
        let masks = [
            contract.expand(&seeds[..32], 0, d.rows * d.hidden),
            contract.expand(&seeds[32..64], 1, d.rows * d.hidden),
        ];
        let combined = Zeroizing::new(
            masks[0]
                .iter()
                .zip(masks[1].iter())
                .map(|(a, b)| a.wrapping_add(*b) & contract.mask())
                .collect::<Vec<_>>(),
        );
        let a = contract.project(0, &combined)?;
        let b = contract.project(1, &combined)?;
        let mut cu = Zeroizing::new(Vec::with_capacity(a.len()));
        let mut cg = Zeroizing::new(Vec::with_capacity(a.len()));
        let mut constant = Zeroizing::new(Vec::with_capacity(a.len()));
        for (&a, &b) in a.iter().zip(b.iter()) {
            let u = a.wrapping_mul(a).wrapping_sub(256u64.wrapping_mul(a));
            cu.push(u);
            cg.push(a.wrapping_mul(2).wrapping_sub(256).wrapping_mul(b));
            constant.push(0u64.wrapping_sub(u).wrapping_mul(b));
        }
        if matches!(contract.layout, Layout::Contracted | Layout::Seeded) {
            constant = contract.project(2, &constant)?;
        }
        let width = contract.coefficient_width();
        let mut coefficients = Zeroizing::new(Vec::with_capacity(d.rows * width));
        for row in 0..d.rows {
            let range = row * d.channels..(row + 1) * d.channels;
            if contract.layout == Layout::Dense {
                coefficients.extend(
                    a[range.clone()]
                        .iter()
                        .map(|v| 256u64.wrapping_sub(v.wrapping_mul(2))),
                );
            }
            coefficients.extend_from_slice(&cu[range.clone()]);
            if contract.layout == Layout::Dense {
                coefficients.extend(b[range.clone()].iter().map(|v| 0u64.wrapping_sub(*v)));
            }
            coefficients.extend_from_slice(&cg[range]);
            let count = if matches!(contract.layout, Layout::Dense | Layout::Derived) {
                d.channels
            } else {
                d.outputs
            };
            coefficients.extend_from_slice(&constant[row * count..(row + 1) * count]);
        }
        let first = contract.expand(&seeds[64..], 2, coefficients.len());
        let second = Zeroizing::new(
            coefficients
                .iter()
                .zip(first.iter())
                .map(|(c, a)| c.wrapping_sub(*a) & contract.mask())
                .collect::<Vec<_>>(),
        );
        let shares = [first, second];
        let make_party = |party: usize| {
            let mut body = Zeroizing::new(Vec::new());
            if contract.layout == Layout::Seeded {
                body.extend_from_slice(&seeds[party * 32..(party + 1) * 32]);
                if party == 0 {
                    body.extend_from_slice(&seeds[64..]);
                } else {
                    body.extend_from_slice(&contract.pack(&shares[party]));
                }
            } else {
                body.extend_from_slice(&contract.pack(&masks[party]));
                body.extend_from_slice(&contract.pack(&shares[party]));
            }
            Party {
                contract: Arc::clone(&contract),
                party: party as u8,
                material: Some(Zeroizing::new(contract.frame(KEY, party as u8, &body))),
                pending: None,
                cancelled: Arc::clone(&self.cancelled),
            }
        };
        Ok([make_party(0), make_party(1)])
    }
}

struct Pending {
    source: Zeroizing<Vec<u64>>,
    coefficients: Zeroizing<Vec<u64>>,
    opened: Zeroizing<Vec<u64>>,
}

/// No Clone, serialization or key export. Owned buffers zeroize on every drop.
pub struct Party {
    contract: Arc<Contract>,
    party: u8,
    material: Option<Zeroizing<Vec<u8>>>,
    pending: Option<Pending>,
    cancelled: Arc<AtomicBool>,
}

impl Party {
    pub fn cancel(&mut self) {
        self.material = None;
        self.pending = None;
    }
    pub fn begin(&mut self, input: &[u64]) -> Result<Vec<u8>, String> {
        self.pending = None; // A repeated begin burns a pending attempt too.
        let packet = self
            .material
            .take()
            .ok_or("projected polynomial material is spent")?;
        if self.cancelled.load(Ordering::Acquire) || !self.contract.valid_input(input) {
            return Err("projected polynomial input is invalid or cancelled".into());
        }
        let c = &self.contract;
        let count = c.dims.rows * c.dims.hidden;
        let body = c.check_frame(
            &packet,
            KEY,
            self.party,
            c.material_sizes()[usize::from(self.party)] - HEADER,
        )?;
        let (source, coefficients) = if c.layout == Layout::Seeded {
            let source = c.expand(&body[..32], self.party, count);
            let coefficients = if self.party == 0 {
                c.expand(&body[32..], 2, c.dims.rows * c.coefficient_width())
            } else {
                c.unpack(&body[32..])
            };
            (source, coefficients)
        } else {
            (
                c.unpack(&body[..count * c.word()]),
                c.unpack(&body[count * c.word()..]),
            )
        };
        let opened = Zeroizing::new(
            input
                .iter()
                .zip(source.iter())
                .map(|(x, r)| x.wrapping_add(*r) & c.mask())
                .collect::<Vec<_>>(),
        );
        let frame = c.frame(OPEN, self.party, &c.pack(&opened));
        self.pending = Some(Pending {
            source,
            coefficients,
            opened,
        });
        Ok(frame)
    }
    pub fn expanded_array_storage_bytes(&self) -> usize {
        self.pending.as_ref().map_or(0, |p| {
            (p.source.len() + p.coefficients.len() + p.opened.len()) * 8
        })
    }
    pub fn finish(&mut self, peer: &[u8]) -> Result<Zeroizing<Vec<u64>>, String> {
        // Take before peer parsing: all error paths consume and zeroize material.
        self.material = None;
        let p = self
            .pending
            .take()
            .ok_or("projected polynomial material is unavailable or spent")?;
        if self.cancelled.load(Ordering::Acquire) {
            return Err("projected polynomial party is cancelled".into());
        }
        let c = &self.contract;
        let body = c.check_frame(
            peer,
            OPEN,
            1 - self.party,
            c.dims.rows * c.dims.hidden * c.word(),
        )?;
        let peer = c.unpack(body);
        let opened = Zeroizing::new(
            p.opened
                .iter()
                .zip(peer.iter())
                .map(|(a, b)| a.wrapping_add(*b) & c.mask())
                .collect::<Vec<_>>(),
        );
        let gate = c.project(0, &opened)?;
        let up = c.project(1, &opened)?;
        let a = if c.layout == Layout::Dense {
            Zeroizing::new(Vec::new())
        } else {
            c.project(0, &p.source)?
        };
        let b = if c.layout == Layout::Dense {
            Zeroizing::new(Vec::new())
        } else {
            c.project(1, &p.source)?
        };
        let mut values = Zeroizing::new(vec![0; gate.len()]);
        let m = c.dims.channels;
        for (i, value) in values.iter_mut().enumerate() {
            let (row, col) = (i / m, i % m);
            let co =
                &p.coefficients[row * c.coefficient_width()..(row + 1) * c.coefficient_width()];
            let (gu, u, g2, g) = if c.layout == Layout::Dense {
                (co[col], co[m + col], co[2 * m + col], co[3 * m + col])
            } else {
                (
                    (if self.party == 0 { 256u64 } else { 0 })
                        .wrapping_sub(2u64.wrapping_mul(a[i])),
                    co[col],
                    0u64.wrapping_sub(b[i]),
                    co[m + col],
                )
            };
            let (gg, uu) = (gate[i], up[i]);
            let square = gg.wrapping_mul(gg);
            *value = (if self.party == 0 {
                square.wrapping_mul(uu)
            } else {
                0
            })
            .wrapping_add(gu.wrapping_mul(gg).wrapping_mul(uu))
            .wrapping_add(u.wrapping_mul(uu))
            .wrapping_add(g2.wrapping_mul(square))
            .wrapping_add(g.wrapping_mul(gg));
            if matches!(c.layout, Layout::Dense | Layout::Derived) {
                *value = value
                    .wrapping_add(co[(if c.layout == Layout::Dense { 4 } else { 2 }) * m + col]);
            }
        }
        let mut output = c.project(2, &values)?;
        if matches!(c.layout, Layout::Contracted | Layout::Seeded) {
            for (i, value) in output.iter_mut().enumerate() {
                let (row, col) = (i / c.dims.outputs, i % c.dims.outputs);
                *value = value
                    .wrapping_add(p.coefficients[row * c.coefficient_width() + 2 * m + col])
                    & c.mask();
            }
        }
        Ok(output)
    }
}

/// Non-secret measurements and reconstructed public-fixture output only.
pub struct ProbeResult {
    pub output: Vec<u64>,
    pub material_bytes: [usize; 2],
    pub opening_bytes: [usize; 2],
    pub expanded_storage_bytes: [usize; 2],
    pub shared_weight_bytes: usize,
    pub issuance_seconds: f64,
    pub evaluation_seconds: f64,
}

pub fn probe(contract: Arc<Contract>, input: &[u64]) -> Result<ProbeResult, String> {
    if !contract.valid_input(input) {
        return Err("projected polynomial fixture input is invalid".into());
    }
    let mut dealer = Dealer::new(MAX_MATERIAL)?;
    let start = Instant::now();
    let [mut a, mut b] = dealer.issue(Arc::clone(&contract))?;
    let issuance_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut seed = Zeroizing::new([0; 32]);
    getrandom::fill(seed.as_mut()).map_err(|e| e.to_string())?;
    let first = contract.expand(seed.as_ref(), 3, input.len());
    let second = Zeroizing::new(
        input
            .iter()
            .zip(first.iter())
            .map(|(x, a)| x.wrapping_sub(*a) & contract.mask())
            .collect::<Vec<_>>(),
    );
    let messages = [a.begin(&first)?, b.begin(&second)?];
    let expanded_storage_bytes = [
        a.expanded_array_storage_bytes(),
        b.expanded_array_storage_bytes(),
    ];
    let results = [a.finish(&messages[1])?, b.finish(&messages[0])?];
    let output = results[0]
        .iter()
        .zip(results[1].iter())
        .map(|(a, b)| a.wrapping_add(*b) & contract.mask())
        .collect();
    let evaluation_seconds = start.elapsed().as_secs_f64();
    dealer.cancel();
    Ok(ProbeResult {
        output,
        material_bytes: contract.material_sizes(),
        opening_bytes: messages.each_ref().map(|v| v.len()),
        expanded_storage_bytes,
        shared_weight_bytes: contract.weights.iter().map(Matrix::weight_bytes).sum(),
        issuance_seconds,
        evaluation_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(layout: Layout, ring_bits: u8, binding: u8) -> Arc<Contract> {
        Contract::new(
            Dimensions {
                rows: 1,
                hidden: 2,
                channels: 3,
                outputs: 2,
                ring_bits,
            },
            layout,
            [binding; 32],
            [
                &[1, 255, 2, 1, 0, 1],
                &[0, 1, 1, 1, 255, 2],
                &[1, 2, 255, 0, 1, 2],
            ],
        )
        .unwrap()
    }
    #[test]
    fn ablations_match_signed_integer_polynomial() {
        for bits in [24, 32, 64] {
            for layout in [
                Layout::Dense,
                Layout::Derived,
                Layout::Contracted,
                Layout::Seeded,
            ] {
                let c = contract(layout, bits, 1);
                for x in -8i64..=8 {
                    let input = [(x as u64) & c.mask(), 3];
                    let g = [x - 3, 2 * x + 3, 3];
                    let u = [3, x + 3, -x + 6];
                    let f: Vec<i64> = g
                        .iter()
                        .zip(u)
                        .map(|(&g, u)| (g * g + 256 * g) * u)
                        .collect();
                    let expected = [
                        (f[0] + 2 * f[1] - f[2]) as u64 & c.mask(),
                        (f[1] + 2 * f[2]) as u64 & c.mask(),
                    ];
                    assert_eq!(probe(Arc::clone(&c), &input).unwrap().output, expected);
                }
            }
        }
    }
    #[test]
    fn replay_malformed_cancel_and_budget_are_burn_boundaries() {
        let c = contract(Layout::Seeded, 24, 1);
        let mut tiny = Dealer::new(1).unwrap();
        assert!(tiny.issue(Arc::clone(&c)).is_err());
        assert_eq!(tiny.issued_bytes, 0);
        let mut dealer = Dealer::new(MAX_MATERIAL).unwrap();
        let [mut a, mut b] = dealer.issue(Arc::clone(&c)).unwrap();
        assert!(dealer.issue(Arc::clone(&c)).is_err());
        let am = a.begin(&[2, 3]).unwrap();
        let mut bm = b.begin(&[4, 5]).unwrap();
        bm[20] ^= 1;
        assert!(a.finish(&bm).is_err());
        assert!(a.finish(&bm).is_err());
        assert!(a.begin(&[2, 3]).is_err());
        dealer.cancel();
        assert!(b.finish(&am).is_err());
        assert!(dealer.issue(contract(Layout::Derived, 32, 2)).is_err());
        let mut dealer = Dealer::new(MAX_MATERIAL).unwrap();
        let [mut a, mut b] = dealer.issue(c).unwrap();
        assert!(a.begin(&[u64::MAX, 0]).is_err());
        assert!(a.begin(&[0, 0]).is_err());
        b.begin(&[0, 0]).unwrap();
        assert!(b.begin(&[0, 0]).is_err());
        assert!(b.finish(&am).is_err());
    }
}
