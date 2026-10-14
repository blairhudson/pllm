//! Bounded two-server XOR PIR over immutable public pages.
//! Clean-room binary-tree DPF, extending PLLM's point-sharing reference to a
//! streaming native table scan. Semi-honest, non-colluding parties; unreviewed.
use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128Enc,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::Arc;
use zeroize::Zeroize;

type Block = [u8; 16];
const QUERY: &[u8; 8] = b"PLLMPQ01";
const REPLY: &[u8; 8] = b"PLLMPR01";
const PACKED_QUERY: &[u8; 8] = b"PLLMPQ02";
const PACKED_REPLY: &[u8; 8] = b"PLLMPR02";
const HEADER: usize = 65;
const MAX_QUERIES: usize = 4096;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Scalar,
    Packed128,
}
impl Layout {
    fn leaf_bits(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Packed128 => 128,
        }
    }
    fn final_bytes(self) -> usize {
        self.leaf_bits().div_ceil(8)
    }
    fn query_magic(self) -> &'static [u8; 8] {
        match self {
            Self::Scalar => QUERY,
            Self::Packed128 => PACKED_QUERY,
        }
    }
    fn reply_magic(self) -> &'static [u8; 8] {
        match self {
            Self::Scalar => REPLY,
            Self::Packed128 => PACKED_REPLY,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Descriptor {
    pub digest: [u8; 32],
    pub records: usize,
    pub width: usize,
}
impl Descriptor {
    fn validate(self) -> Result<Self, String> {
        if !(1..=262144).contains(&self.records)
            || !(1..=65536).contains(&self.width)
            || self
                .records
                .checked_mul(self.width)
                .is_none_or(|n| n > 512 * 1024 * 1024)
        {
            return Err("private page dimensions exceed bounded table policy".into());
        }
        Ok(self)
    }
    fn depth(self, layout: Layout) -> usize {
        (usize::BITS - (self.records.div_ceil(layout.leaf_bits()) - 1).leading_zeros()) as usize
    }
}
#[derive(Clone)]
struct Correction {
    seed: Block,
    controls: u8,
}
impl Drop for Correction {
    fn drop(&mut self) {
        self.seed.zeroize();
        self.controls.zeroize();
    }
}
struct Key {
    layout: Layout,
    descriptor: Descriptor,
    id: Block,
    party: u8,
    root: Block,
    words: Vec<Correction>,
    final_word: Block,
}
impl Drop for Key {
    fn drop(&mut self) {
        self.root.zeroize();
        self.final_word.zeroize();
    }
}
fn xor(a: Block, b: Block) -> Block {
    std::array::from_fn(|i| a[i] ^ b[i])
}
fn domain(descriptor: Descriptor, id: Block, layout: Layout) -> Block {
    let mut h = Sha256::new();
    h.update(match layout {
        Layout::Scalar => b"pllm/private-page-dpf/aes128/v1\0",
        Layout::Packed128 => b"pllm/private-page-dpf/aes128/v2\0",
    });
    h.update(descriptor.digest);
    h.update(id);
    h.update((descriptor.records as u64).to_le_bytes());
    h.update((descriptor.width as u64).to_le_bytes());
    h.finalize()[..16].try_into().unwrap()
}
fn stretch(seed: Block, domain: Block, level: usize) -> [(Block, u8); 2] {
    let cipher = Aes128Enc::new_from_slice(&seed).unwrap();
    let mut blocks = [aes::cipher::Block::<Aes128Enc>::default(); 3];
    for (i, block) in blocks.iter_mut().enumerate() {
        block.copy_from_slice(&domain);
        block[0] ^= 0xa1;
        block[1] ^= level as u8;
        block[2] ^= i as u8;
    }
    cipher.encrypt_blocks(&mut blocks);
    let result = [
        (blocks[0].into(), blocks[2][0] & 1),
        (blocks[1].into(), (blocks[2][0] >> 1) & 1),
    ];
    for block in &mut blocks {
        block.as_mut_slice().zeroize();
    }
    result
}
fn leaf(seed: Block, domain: Block, layout: Layout) -> Block {
    let cipher = Aes128Enc::new_from_slice(&seed).unwrap();
    let mut block = aes::cipher::Block::<Aes128Enc>::from(domain);
    block[0] ^= 0xb2;
    cipher.encrypt_block(&mut block);
    let mut result: Block = block.into();
    if layout == Layout::Scalar {
        result[0] &= 1;
        result[1..].zeroize();
    }
    block.as_mut_slice().zeroize();
    result
}
fn header(magic: &[u8; 8], d: Descriptor, id: Block, party: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER);
    out.extend_from_slice(magic);
    out.push(party);
    out.extend_from_slice(&(d.records as u32).to_le_bytes());
    out.extend_from_slice(&(d.width as u32).to_le_bytes());
    out.extend_from_slice(&d.digest);
    out.extend_from_slice(&id);
    out
}
impl Key {
    fn encode(&self) -> Vec<u8> {
        let mut out = header(
            self.layout.query_magic(),
            self.descriptor,
            self.id,
            self.party,
        );
        out.extend_from_slice(&self.root);
        for word in &self.words {
            out.extend_from_slice(&word.seed);
            out.push(word.controls);
        }
        out.extend_from_slice(&self.final_word[..self.layout.final_bytes()]);
        out
    }
    fn decode(data: &[u8]) -> Result<Self, String> {
        if data.len() < HEADER + 17 || data[8] > 1 {
            return Err("malformed private page query".into());
        }
        let layout = match &data[..8] {
            value if value == QUERY => Layout::Scalar,
            value if value == PACKED_QUERY => Layout::Packed128,
            _ => return Err("malformed private page query namespace".into()),
        };
        let descriptor = Descriptor {
            records: u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize,
            width: u32::from_le_bytes(data[13..17].try_into().unwrap()) as usize,
            digest: data[17..49].try_into().unwrap(),
        }
        .validate()?;
        let final_bytes = layout.final_bytes();
        if data.len() != HEADER + 16 + final_bytes + 17 * descriptor.depth(layout)
            || (layout == Layout::Scalar && *data.last().unwrap() > 1)
        {
            return Err("private page query size or bit differs".into());
        }
        let mut words = Vec::with_capacity(descriptor.depth(layout));
        for item in data[HEADER + 16..data.len() - final_bytes].chunks_exact(17) {
            if item[16] & !3 != 0 {
                return Err("noncanonical private page control word".into());
            }
            words.push(Correction {
                seed: item[..16].try_into().unwrap(),
                controls: item[16],
            });
        }
        let mut final_word = [0; 16];
        final_word[..final_bytes].copy_from_slice(&data[data.len() - final_bytes..]);
        Ok(Self {
            layout,
            descriptor,
            id: data[49..65].try_into().unwrap(),
            party: data[8],
            root: data[HEADER..HEADER + 16].try_into().unwrap(),
            words,
            final_word,
        })
    }
}
pub struct Decoder {
    layout: Layout,
    descriptor: Descriptor,
    id: Block,
}
impl Decoder {
    /// Ownership is consumed even when response validation fails.
    pub fn decode(self, a: &[u8], b: &[u8]) -> Result<Vec<u8>, String> {
        for (party, reply) in [a, b].iter().enumerate() {
            if reply.len() != HEADER + self.descriptor.width
                || reply[..HEADER]
                    != header(
                        self.layout.reply_magic(),
                        self.descriptor,
                        self.id,
                        party as u8,
                    )
            {
                return Err("private page response binding or size differs".into());
            }
        }
        Ok(a[HEADER..]
            .iter()
            .zip(&b[HEADER..])
            .map(|(&x, &y)| x ^ y)
            .collect())
    }
}
pub fn issue(descriptor: Descriptor, index: usize) -> Result<(Vec<u8>, Vec<u8>, Decoder), String> {
    issue_layout(descriptor, index, Layout::Scalar)
}

/// Packed-output DPF: each terminal PRG block selects 128 original records.
/// This uses a distinct research packet contract; the scalar API is unchanged.
pub fn issue_packed(
    descriptor: Descriptor,
    index: usize,
) -> Result<(Vec<u8>, Vec<u8>, Decoder), String> {
    issue_layout(descriptor, index, Layout::Packed128)
}

fn issue_layout(
    descriptor: Descriptor,
    index: usize,
    layout: Layout,
) -> Result<(Vec<u8>, Vec<u8>, Decoder), String> {
    let descriptor = descriptor.validate()?;
    if index >= descriptor.records {
        return Err("private page index is outside the table".into());
    }
    let mut id = [0; 16];
    let mut roots = [[0; 16]; 2];
    getrandom::fill(&mut id).map_err(|e| e.to_string())?;
    for root in &mut roots {
        getrandom::fill(root).map_err(|e| e.to_string())?;
    }
    let domain = domain(descriptor, id, layout);
    let (mut seeds, mut controls) = (roots, [0_u8, 1]);
    let depth = descriptor.depth(layout);
    let address = index / layout.leaf_bits();
    let mut words = Vec::with_capacity(depth);
    for level in 0..depth {
        let direction = (address >> (depth - level - 1)) & 1;
        let mut children = [
            stretch(seeds[0], domain, level),
            stretch(seeds[1], domain, level),
        ];
        let word = Correction {
            seed: xor(children[0][1 - direction].0, children[1][1 - direction].0),
            controls: (children[0][0].1 ^ children[1][0].1 ^ (1 - direction) as u8)
                | ((children[0][1].1 ^ children[1][1].1 ^ direction as u8) << 1),
        };
        for party in 0..2 {
            let (mut seed, mut control) = children[party][direction];
            if controls[party] != 0 {
                seed = xor(seed, word.seed);
                control ^= (word.controls >> direction) & 1;
            }
            seeds[party] = seed;
            controls[party] = control;
        }
        children.zeroize();
        words.push(word);
    }
    let mut final_word = xor(
        leaf(seeds[0], domain, layout),
        leaf(seeds[1], domain, layout),
    );
    let bit = index % layout.leaf_bits();
    final_word[bit / 8] ^= 1 << (bit % 8);
    let a = Key {
        layout,
        descriptor,
        id,
        party: 0,
        root: roots[0],
        words: words.clone(),
        final_word,
    }
    .encode();
    let b = Key {
        layout,
        descriptor,
        id,
        party: 1,
        root: roots[1],
        words,
        final_word,
    }
    .encode();
    roots.zeroize();
    seeds.zeroize();
    controls.zeroize();
    final_word.zeroize();
    Ok((
        a,
        b,
        Decoder {
            layout,
            descriptor,
            id,
        },
    ))
}
enum Table {
    Records(Vec<u8>),
    // The head and lookup retain the same immutable matrix, rather than a second
    // vocabulary-sized interleaved allocation. Scales keep their original bits.
    Vocabulary {
        matrix: Arc<pllm_core::kernels::Matrix>,
        scales: Vec<u8>,
    },
}
impl Table {
    fn xor_record(&self, index: usize, width: usize, mask: u8, output: &mut [u8]) {
        match self {
            Self::Records(data) => {
                for (dst, &value) in output
                    .iter_mut()
                    .zip(&data[index * width..(index + 1) * width])
                {
                    *dst ^= value & mask;
                }
            }
            Self::Vocabulary { matrix, scales } => {
                let columns = width - 4;
                let weights = &matrix.as_signed_slice()[index * columns..(index + 1) * columns];
                for (dst, &value) in output[..columns].iter_mut().zip(weights) {
                    *dst ^= value as u8 & mask;
                }
                for (dst, &value) in output[columns..]
                    .iter_mut()
                    .zip(&scales[index * 4..index * 4 + 4])
                {
                    *dst ^= value & mask;
                }
            }
        }
    }
}

/// One party's table view and process-local replay ledgers. Context-bound views
/// share immutable public storage but never share a query-consumption ledger.
pub struct Server {
    descriptor: Descriptor,
    data: Arc<Table>,
    party: u8,
    burned: BTreeSet<Block>,
    contexts: BTreeSet<[u8; 32]>,
}
impl Server {
    pub fn new(data: &[u8], width: usize, party: u8) -> Result<Self, String> {
        if width == 0 || data.len() % width != 0 || party > 1 {
            return Err("invalid private page table".into());
        }
        let mut descriptor = Descriptor {
            records: data.len() / width,
            width,
            digest: [0; 32],
        }
        .validate()?;
        let mut hash = Sha256::new();
        hash.update(b"pllm/public-page-table/v1\0");
        hash.update((descriptor.records as u64).to_le_bytes());
        hash.update((width as u64).to_le_bytes());
        hash.update(data);
        descriptor.digest = hash.finalize().into();
        Ok(Self {
            descriptor,
            data: Arc::new(Table::Records(data.to_vec())),
            party,
            burned: BTreeSet::new(),
            contexts: BTreeSet::new(),
        })
    }
    /// Bind the existing signed-i8 head snapshot and per-token float32 scales.
    /// The descriptor is byte-identical to explicitly interleaved records.
    pub fn from_matrix(
        matrix: Arc<pllm_core::kernels::Matrix>,
        scales: &[u8],
        party: u8,
    ) -> Result<Self, String> {
        let (records, columns) = matrix.shape();
        let mut descriptor = Descriptor {
            records,
            width: columns
                .checked_add(4)
                .ok_or("private vocabulary width overflow")?,
            digest: [0; 32],
        }
        .validate()?;
        if party > 1
            || scales.len() != records * 4
            || scales.chunks_exact(4).any(|bytes| {
                let scale = f32::from_le_bytes(bytes.try_into().unwrap());
                !scale.is_finite() || scale <= 0.0
            })
        {
            return Err("private vocabulary party or scales differ".into());
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm/public-page-table/v1\0");
        hash.update((records as u64).to_le_bytes());
        hash.update((descriptor.width as u64).to_le_bytes());
        let mut row = vec![0_u8; columns];
        for (weights, scale) in matrix
            .as_signed_slice()
            .chunks_exact(columns)
            .zip(scales.chunks_exact(4))
        {
            for (byte, &value) in row.iter_mut().zip(weights) {
                *byte = value as u8;
            }
            hash.update(&row);
            hash.update(scale);
        }
        descriptor.digest = hash.finalize().into();
        Ok(Self {
            descriptor,
            data: Arc::new(Table::Vocabulary {
                matrix,
                scales: scales.to_vec(),
            }),
            party,
            burned: BTreeSet::new(),
            contexts: BTreeSet::new(),
        })
    }
    /// Mint at most 4096 unique public contexts per owner. The caller supplies a
    /// source/plan/session/epoch commitment; authenticated transport remains its
    /// responsibility. Repeated contexts reject even after a child retires.
    pub fn bind_context(&mut self, context: [u8; 32]) -> Result<Self, String> {
        if self.contexts.len() >= MAX_QUERIES || !self.contexts.insert(context) {
            return Err("private page context replayed or ledger exhausted".into());
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm/private-page-context/v1\0");
        hash.update(self.descriptor.digest);
        hash.update(context);
        let descriptor = Descriptor {
            digest: hash.finalize().into(),
            ..self.descriptor
        };
        Ok(Self {
            descriptor,
            data: Arc::clone(&self.data),
            party: self.party,
            burned: BTreeSet::new(),
            contexts: BTreeSet::new(),
        })
    }
    pub fn descriptor(&self) -> Descriptor {
        self.descriptor
    }
    fn admit(&mut self, data: &[u8]) -> Result<Key, String> {
        let key = Key::decode(data)?;
        if key.descriptor != self.descriptor || key.party != self.party {
            return Err("private page source or party differs".into());
        }
        if self.burned.len() >= MAX_QUERIES || !self.burned.insert(key.id) {
            return Err("private page query replayed or ledger exhausted".into());
        }
        Ok(key)
    }
    pub fn cancel(&mut self, query: &[u8]) -> Result<(), String> {
        self.admit(query).map(|_| ())
    }
    pub fn evaluate(&mut self, query: &[u8]) -> Result<Vec<u8>, String> {
        let key = self.admit(query)?;
        let mut out = vec![0; self.descriptor.width];
        self.visit(
            &key,
            domain(self.descriptor, key.id, key.layout),
            key.root,
            key.party,
            0,
            0,
            &mut out,
        );
        let mut reply = header(
            key.layout.reply_magic(),
            self.descriptor,
            key.id,
            self.party,
        );
        reply.extend_from_slice(&out);
        out.zeroize();
        Ok(reply)
    }

    /// Fixed public batch size. All keys validate before any are burned; every
    /// admitted key burns before arithmetic. Each party still holds one key share.
    pub fn evaluate_batch(&mut self, queries: &[&[u8]]) -> Result<Vec<Vec<u8>>, String> {
        if !(1..=64).contains(&queries.len()) || self.burned.len() + queries.len() > MAX_QUERIES {
            return Err("private page batch exceeds bounded query ledger".into());
        }
        let mut ids = BTreeSet::new();
        let keys: Vec<Key> = queries
            .iter()
            .map(|q| {
                let key = Key::decode(q)?;
                if key.descriptor != self.descriptor
                    || key.party != self.party
                    || self.burned.contains(&key.id)
                    || !ids.insert(key.id)
                {
                    return Err("private page batch source, party or replay differs".into());
                }
                Ok(key)
            })
            .collect::<Result<_, String>>()?;
        self.burned.extend(ids);
        let count = keys.len();
        let mut bits = zeroize::Zeroizing::new(vec![0_u8; self.descriptor.records * count]);
        for (lane, key) in keys.iter().enumerate() {
            fill_bits(
                key,
                domain(self.descriptor, key.id, key.layout),
                key.root,
                key.party,
                0,
                0,
                lane,
                count,
                &mut bits,
            );
        }
        let mut outputs = zeroize::Zeroizing::new(vec![0_u8; self.descriptor.width * count]);
        // One traversal of the table; all selections are masked XOR, including
        // zero shares. Working masks stay at the worker (at most 16 MiB).
        for (row, selection) in bits.chunks_exact(count).enumerate() {
            for (out, &bit) in outputs.chunks_mut(self.descriptor.width).zip(selection) {
                let mask = 0_u8.wrapping_sub(bit);
                self.data.xor_record(row, self.descriptor.width, mask, out);
            }
        }
        Ok(keys
            .iter()
            .zip(outputs.chunks_exact(self.descriptor.width))
            .map(|(key, out)| {
                let mut reply = header(
                    key.layout.reply_magic(),
                    self.descriptor,
                    key.id,
                    self.party,
                );
                reply.extend_from_slice(out);
                reply
            })
            .collect())
    }
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &self,
        key: &Key,
        domain: Block,
        mut seed: Block,
        control: u8,
        level: usize,
        first: usize,
        out: &mut [u8],
    ) {
        let first_record = first * key.layout.leaf_bits();
        if first_record >= self.descriptor.records {
            seed.zeroize();
            return;
        }
        if level == key.words.len() {
            let mut word = leaf(seed, domain, key.layout);
            for (a, b) in word.iter_mut().zip(key.final_word) {
                *a ^= b & 0_u8.wrapping_sub(control);
            }
            for bit in 0..key
                .layout
                .leaf_bits()
                .min(self.descriptor.records - first_record)
            {
                let mask = 0_u8.wrapping_sub((word[bit / 8] >> (bit % 8)) & 1);
                self.data
                    .xor_record(first_record + bit, self.descriptor.width, mask, out);
            }
            word.zeroize();
        } else {
            let mut children = stretch(seed, domain, level);
            for (direction, child) in children.iter_mut().enumerate() {
                if control != 0 {
                    child.0 = xor(child.0, key.words[level].seed);
                    child.1 ^= (key.words[level].controls >> direction) & 1;
                }
                self.visit(
                    key,
                    domain,
                    child.0,
                    child.1,
                    level + 1,
                    first + (direction << (key.words.len() - level - 1)),
                    out,
                );
                child.0.zeroize();
                child.1.zeroize();
            }
        }
        seed.zeroize();
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_bits(
    key: &Key,
    domain: Block,
    mut seed: Block,
    control: u8,
    level: usize,
    first: usize,
    lane: usize,
    count: usize,
    output: &mut [u8],
) {
    let first_record = first * key.layout.leaf_bits();
    if first_record >= key.descriptor.records {
        seed.zeroize();
        return;
    }
    if level == key.words.len() {
        let mut word = leaf(seed, domain, key.layout);
        for (a, b) in word.iter_mut().zip(key.final_word) {
            *a ^= b & 0_u8.wrapping_sub(control);
        }
        for bit in 0..key
            .layout
            .leaf_bits()
            .min(key.descriptor.records - first_record)
        {
            output[(first_record + bit) * count + lane] = (word[bit / 8] >> (bit % 8)) & 1;
        }
        word.zeroize();
    } else {
        let mut children = stretch(seed, domain, level);
        for (direction, child) in children.iter_mut().enumerate() {
            if control != 0 {
                child.0 = xor(child.0, key.words[level].seed);
                child.1 ^= (key.words[level].controls >> direction) & 1;
            }
            fill_bits(
                key,
                domain,
                child.0,
                child.1,
                level + 1,
                first + (direction << (key.words.len() - level - 1)),
                lane,
                count,
                output,
            );
            child.0.zeroize();
            child.1.zeroize();
        }
    }
    seed.zeroize();
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_vocabulary_snapshot_and_bound_contexts() {
        let (records, columns) = (129, 19);
        let weights: Vec<u8> = (0..records * columns).map(|i| (i * 37) as u8).collect();
        let scales: Vec<u8> = (0..records)
            .flat_map(|i| (0.01 + i as f32 / 10.0).to_le_bytes())
            .collect();
        let matrix = Arc::new(pllm_core::kernels::Matrix::new(&weights, records, columns).unwrap());
        let mut table = Vec::new();
        for (row, scale) in weights.chunks_exact(columns).zip(scales.chunks_exact(4)) {
            table.extend_from_slice(row);
            table.extend_from_slice(scale);
        }
        let raw = Server::new(&table, columns + 4, 0).unwrap();
        let mut a = Server::from_matrix(Arc::clone(&matrix), &scales, 0).unwrap();
        let mut b = Server::from_matrix(Arc::clone(&matrix), &scales, 1).unwrap();
        assert_eq!(Arc::strong_count(&matrix), 3);
        assert!(a.descriptor() == raw.descriptor());
        let mut x = a.bind_context([31; 32]).unwrap();
        let mut y = b.bind_context([31; 32]).unwrap();
        assert_eq!(Arc::strong_count(&a.data), 2);
        assert_eq!(Arc::strong_count(&matrix), 3);
        assert!(a.bind_context([31; 32]).is_err());
        let mut other = a.bind_context([32; 32]).unwrap();
        assert!(x.descriptor() == y.descriptor());
        assert!(x.descriptor() != other.descriptor());
        let mut hash = Sha256::new();
        hash.update(b"pllm/private-page-context/v1\0");
        hash.update(raw.descriptor().digest);
        hash.update([31; 32]);
        assert_eq!(x.descriptor().digest, <[u8; 32]>::from(hash.finalize()));
        // Retiring source wrappers must leave each bound view's immutable owner valid.
        drop(matrix);
        drop(a);
        drop(b);
        let indices = [0, 127, 128];
        let queries: Vec<_> = indices
            .iter()
            .map(|&i| issue_packed(x.descriptor(), i).unwrap())
            .collect();
        assert!(other.evaluate(&queries[0].0).is_err());
        let qa: Vec<_> = queries.iter().map(|q| q.0.as_slice()).collect();
        let qb: Vec<_> = queries.iter().map(|q| q.1.as_slice()).collect();
        let ra = x.evaluate_batch(&qa).unwrap();
        let rb = y.evaluate_batch(&qb).unwrap();
        assert!(x.evaluate_batch(&qa).is_err());
        for ((index, (_, _, decoder)), (a, b)) in
            indices.iter().zip(queries).zip(ra.iter().zip(rb.iter()))
        {
            assert_eq!(
                decoder.decode(a, b).unwrap(),
                table[index * (columns + 4)..(index + 1) * (columns + 4)]
            );
        }
    }

    #[test]
    fn vocabulary_admission_precedes_context_or_scale_allocation() {
        let matrix = Arc::new(pllm_core::kernels::Matrix::new(&[1, 2, 3], 1, 3).unwrap());
        for scale in [f32::NAN, f32::INFINITY, 0.0, -0.0, -1.0] {
            assert!(Server::from_matrix(Arc::clone(&matrix), &scale.to_le_bytes(), 0).is_err());
        }
        assert!(Server::from_matrix(Arc::clone(&matrix), &[0; 3], 0).is_err());
        assert!(Server::from_matrix(Arc::clone(&matrix), &1.0_f32.to_le_bytes(), 2).is_err());
        let mut root = Server::from_matrix(matrix, &1.0_f32.to_le_bytes(), 0).unwrap();
        for i in 0..MAX_QUERIES {
            let mut context = [0; 32];
            context[..8].copy_from_slice(&(i as u64).to_le_bytes());
            drop(root.bind_context(context).unwrap());
        }
        assert!(root.bind_context([255; 32]).is_err());
    }

    #[test]
    fn encryption_only_schedule_preserves_scalar_prg() {
        // The original full cipher remains an independent schedule oracle.
        for seed in [[0; 16], [255; 16], std::array::from_fn(|i| i as u8)] {
            let domain = [17; 16];
            let cipher = aes::Aes128::new_from_slice(&seed).unwrap();
            for level in 0..18 {
                let mut blocks = [aes::cipher::Block::<aes::Aes128>::from(domain); 3];
                for (i, block) in blocks.iter_mut().enumerate() {
                    block[0] ^= 0xa1;
                    block[1] ^= level as u8;
                    block[2] ^= i as u8;
                    cipher.encrypt_block(block);
                }
                assert_eq!(
                    stretch(seed, domain, level),
                    [
                        (blocks[0].into(), blocks[2][0] & 1),
                        (blocks[1].into(), (blocks[2][0] >> 1) & 1),
                    ]
                );
            }
            let mut block = aes::cipher::Block::<aes::Aes128>::from(domain);
            block[0] ^= 0xb2;
            cipher.encrypt_block(&mut block);
            assert_eq!(leaf(seed, domain, Layout::Scalar)[0], block[0] & 1);
            assert_eq!(leaf(seed, domain, Layout::Packed128), <Block>::from(block));
        }
    }

    #[test]
    fn packed_exhaustive_basis_and_incomplete_leaves() {
        // Every output coordinate is visible in the basis oracle, including
        // padded/tail leaves and the 127/128 group boundary.
        for records in [1, 2, 3, 127, 128, 129, 255, 257] {
            let mut data = vec![0; records * records];
            for row in 0..records {
                data[row * records + row] = 1;
            }
            let mut a = Server::new(&data, records, 0).unwrap();
            let mut b = Server::new(&data, records, 1).unwrap();
            for index in 0..records {
                let (ka, kb, decoder) = issue_packed(a.descriptor(), index).unwrap();
                assert_eq!(&ka[..8], PACKED_QUERY);
                assert_eq!(
                    ka.len(),
                    HEADER + 32 + 17 * a.descriptor().depth(Layout::Packed128)
                );
                let ra = a.evaluate(&ka).unwrap();
                let rb = b.evaluate(&kb).unwrap();
                assert_eq!(
                    decoder.decode(&ra, &rb).unwrap(),
                    data[index * records..(index + 1) * records]
                );
                assert!(a.evaluate(&ka).is_err());
            }
        }
    }

    #[test]
    fn mixed_layout_batch_atomicity_and_namespace_binding() {
        let data: Vec<u8> = (0..1009 * 19).map(|i| (i * 131 + i / 19) as u8).collect();
        let mut a = Server::new(&data, 19, 0).unwrap();
        let mut b = Server::new(&data, 19, 1).unwrap();
        let queries: Vec<_> = (0..64)
            .map(|i| {
                let layout = if i % 2 == 0 {
                    Layout::Scalar
                } else {
                    Layout::Packed128
                };
                issue_layout(a.descriptor(), (i * 129) % 1009, layout).unwrap()
            })
            .collect();
        let qa: Vec<_> = queries.iter().map(|q| q.0.as_slice()).collect();
        let qb: Vec<_> = queries.iter().map(|q| q.1.as_slice()).collect();
        assert!(a.evaluate_batch(&[qa[0], qa[1], qa[1]]).is_err());
        assert!(a.evaluate_batch(&[qa[0], b"malformed"]).is_err());
        let ra = a.evaluate_batch(&qa).unwrap();
        let rb = b.evaluate_batch(&qb).unwrap();
        assert!(b.evaluate_batch(&qb).is_err());
        for (i, ((_, _, decoder), (a, b))) in
            queries.into_iter().zip(ra.iter().zip(&rb)).enumerate()
        {
            let index = (i * 129) % 1009;
            assert_eq!(
                decoder.decode(a, b).unwrap(),
                data[index * 19..(index + 1) * 19]
            );
        }
        let (ka, kb, decoder) = issue_packed(a.descriptor(), 128).unwrap();
        for end in 0..ka.len() {
            assert!(a.evaluate(&ka[..end]).is_err());
        }
        let mut bad = ka.clone();
        bad[..8].copy_from_slice(QUERY);
        assert!(a.evaluate(&bad).is_err());
        let mut ra = a.evaluate(&ka).unwrap();
        let rb = b.evaluate(&kb).unwrap();
        ra[..8].copy_from_slice(REPLY);
        assert!(decoder.decode(&ra, &rb).is_err());
        let (ka, kb, _) = issue_packed(a.descriptor(), 1008).unwrap();
        a.cancel(&ka).unwrap();
        b.cancel(&kb).unwrap();
        assert!(a.evaluate_batch(&[&ka]).is_err());
        assert!(b.evaluate(&kb).is_err());
    }

    #[test]
    fn batch_exactness_and_atomic_admission() {
        let data: Vec<u8> = (0..31 * 17).map(|x| x as u8).collect();
        let mut a = Server::new(&data, 17, 0).unwrap();
        let mut b = Server::new(&data, 17, 1).unwrap();
        let queries: Vec<_> = [0, 30, 7, 7]
            .into_iter()
            .map(|i| issue(a.descriptor(), i).unwrap())
            .collect();
        assert!(a.evaluate_batch(&[&queries[0].0, b"invalid"]).is_err());
        assert!(a.evaluate_batch(&[&queries[0].0, &queries[0].0]).is_err());
        let qa: Vec<_> = queries.iter().map(|q| q.0.as_slice()).collect();
        let qb: Vec<_> = queries.iter().map(|q| q.1.as_slice()).collect();
        let ra = a.evaluate_batch(&qa).unwrap();
        let rb = b.evaluate_batch(&qb).unwrap();
        assert!(a.evaluate_batch(&qa).is_err());
        for (((_, _, decoder), (a, b)), index) in queries
            .into_iter()
            .zip(ra.iter().zip(&rb))
            .zip([0, 30, 7, 7])
        {
            assert_eq!(
                decoder.decode(a, b).unwrap(),
                data[index * 17..(index + 1) * 17]
            );
        }
    }
    #[test]
    fn exhaustive_basis_pages_and_one_use_roles() {
        for records in [1, 2, 3, 7, 16, 31] {
            let mut data = vec![0; records * records];
            for i in 0..records {
                data[i * records + i] = 1;
            }
            let mut a = Server::new(&data, records, 0).unwrap();
            let mut b = Server::new(&data, records, 1).unwrap();
            for index in 0..records {
                let (ka, kb, decoder) = issue(a.descriptor(), index).unwrap();
                assert!(a.evaluate(&kb).is_err());
                let ra = a.evaluate(&ka).unwrap();
                let rb = b.evaluate(&kb).unwrap();
                assert_eq!(
                    decoder.decode(&ra, &rb).unwrap(),
                    data[index * records..(index + 1) * records]
                );
                assert!(a.evaluate(&ka).is_err());
                let (ca, cb, _) = issue(a.descriptor(), index).unwrap();
                a.cancel(&ca).unwrap();
                b.cancel(&cb).unwrap();
                assert!(b.evaluate(&cb).is_err());
            }
        }
    }
    #[test]
    fn malformed_context_and_decoder_binding() {
        let mut a = Server::new(&[1; 90], 9, 0).unwrap();
        let mut b = Server::new(&[1; 90], 9, 1).unwrap();
        let (ka, kb, decoder) = issue(a.descriptor(), 9).unwrap();
        for end in 0..ka.len() {
            assert!(a.evaluate(&ka[..end]).is_err());
        }
        let mut bad = ka.clone();
        bad[17] ^= 1;
        assert!(a.evaluate(&bad).is_err());
        let mut bad = ka.clone();
        bad[HEADER + 16 + 16] |= 0x80;
        assert!(a.evaluate(&bad).is_err());
        let ra = a.evaluate(&ka).unwrap();
        let rb = b.evaluate(&kb).unwrap();
        assert!(decoder.decode(&rb, &ra).is_err());
        assert!(issue(a.descriptor(), 10).is_err());
        assert!(Server::new(&[0; 10], 3, 0).is_err());
    }
}
