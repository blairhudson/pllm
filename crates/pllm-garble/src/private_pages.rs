//! Bounded two-server XOR PIR over immutable public pages.
//! Clean-room binary-tree DPF, extending PLLM's point-sharing reference to a
//! streaming native table scan. Semi-honest, non-colluding parties; unreviewed.
use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use zeroize::Zeroize;

type Block = [u8; 16];
const QUERY: &[u8; 8] = b"PLLMPQ01";
const REPLY: &[u8; 8] = b"PLLMPR01";
const HEADER: usize = 65;
const MAX_QUERIES: usize = 4096;

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
    fn depth(self) -> usize {
        (usize::BITS - (self.records - 1).leading_zeros()) as usize
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
    descriptor: Descriptor,
    id: Block,
    party: u8,
    root: Block,
    words: Vec<Correction>,
    final_bit: u8,
}
impl Drop for Key {
    fn drop(&mut self) {
        self.root.zeroize();
        self.final_bit.zeroize();
    }
}
fn xor(a: Block, b: Block) -> Block {
    std::array::from_fn(|i| a[i] ^ b[i])
}
fn domain(descriptor: Descriptor, id: Block) -> Block {
    let mut h = Sha256::new();
    h.update(b"pllm/private-page-dpf/aes128/v1\0");
    h.update(descriptor.digest);
    h.update(id);
    h.update((descriptor.records as u64).to_le_bytes());
    h.update((descriptor.width as u64).to_le_bytes());
    h.finalize()[..16].try_into().unwrap()
}
fn stretch(seed: Block, domain: Block, level: usize) -> [(Block, u8); 2] {
    let cipher = Aes128::new_from_slice(&seed).unwrap();
    let mut blocks = [aes::cipher::Block::<Aes128>::default(); 3];
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
fn leaf(seed: Block, domain: Block) -> u8 {
    let cipher = Aes128::new_from_slice(&seed).unwrap();
    let mut block = aes::cipher::Block::<Aes128>::from(domain);
    block[0] ^= 0xb2;
    cipher.encrypt_block(&mut block);
    let bit = block[0] & 1;
    block.as_mut_slice().zeroize();
    bit
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
        let mut out = header(QUERY, self.descriptor, self.id, self.party);
        out.extend_from_slice(&self.root);
        for word in &self.words {
            out.extend_from_slice(&word.seed);
            out.push(word.controls);
        }
        out.push(self.final_bit);
        out
    }
    fn decode(data: &[u8]) -> Result<Self, String> {
        if data.len() < HEADER + 17 || &data[..8] != QUERY || data[8] > 1 {
            return Err("malformed private page query".into());
        }
        let descriptor = Descriptor {
            records: u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize,
            width: u32::from_le_bytes(data[13..17].try_into().unwrap()) as usize,
            digest: data[17..49].try_into().unwrap(),
        }
        .validate()?;
        if data.len() != HEADER + 17 + 17 * descriptor.depth() || *data.last().unwrap() > 1 {
            return Err("private page query size or bit differs".into());
        }
        let mut words = Vec::with_capacity(descriptor.depth());
        for item in data[HEADER + 16..data.len() - 1].chunks_exact(17) {
            if item[16] & !3 != 0 {
                return Err("noncanonical private page control word".into());
            }
            words.push(Correction {
                seed: item[..16].try_into().unwrap(),
                controls: item[16],
            });
        }
        Ok(Self {
            descriptor,
            id: data[49..65].try_into().unwrap(),
            party: data[8],
            root: data[HEADER..HEADER + 16].try_into().unwrap(),
            words,
            final_bit: *data.last().unwrap(),
        })
    }
}
pub struct Decoder {
    descriptor: Descriptor,
    id: Block,
}
impl Decoder {
    /// Ownership is consumed even when response validation fails.
    pub fn decode(self, a: &[u8], b: &[u8]) -> Result<Vec<u8>, String> {
        for (party, reply) in [a, b].iter().enumerate() {
            if reply.len() != HEADER + self.descriptor.width
                || reply[..HEADER] != header(REPLY, self.descriptor, self.id, party as u8)
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
    let domain = domain(descriptor, id);
    let (mut seeds, mut controls) = (roots, [0_u8, 1]);
    let mut words = Vec::with_capacity(descriptor.depth());
    for level in 0..descriptor.depth() {
        let direction = (index >> (descriptor.depth() - level - 1)) & 1;
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
    let final_bit = leaf(seeds[0], domain) ^ leaf(seeds[1], domain) ^ 1;
    let a = Key {
        descriptor,
        id,
        party: 0,
        root: roots[0],
        words: words.clone(),
        final_bit,
    }
    .encode();
    let b = Key {
        descriptor,
        id,
        party: 1,
        root: roots[1],
        words,
        final_bit,
    }
    .encode();
    roots.zeroize();
    seeds.zeroize();
    controls.zeroize();
    Ok((a, b, Decoder { descriptor, id }))
}
/// One independently owned table and one party's process-local replay ledger.
pub struct Server {
    descriptor: Descriptor,
    data: Vec<u8>,
    party: u8,
    burned: BTreeSet<Block>,
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
            data: data.to_vec(),
            party,
            burned: BTreeSet::new(),
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
            domain(self.descriptor, key.id),
            key.root,
            key.party,
            0,
            0,
            &mut out,
        );
        let mut reply = header(REPLY, self.descriptor, key.id, self.party);
        reply.extend_from_slice(&out);
        out.zeroize();
        Ok(reply)
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
        if first >= self.descriptor.records {
            seed.zeroize();
            return;
        }
        if level == key.words.len() {
            let bit = leaf(seed, domain) ^ (control & key.final_bit);
            let mask = 0_u8.wrapping_sub(bit);
            let row =
                &self.data[first * self.descriptor.width..(first + 1) * self.descriptor.width];
            for (dst, &byte) in out.iter_mut().zip(row) {
                *dst ^= byte & mask;
            }
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
#[cfg(test)]
mod tests {
    use super::*;
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
