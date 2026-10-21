//! Independently implemented SIGMA source-GPU Boolean DPF, CPU execution port.
//!
//! Specification: mpc-msri/EzPC@aa0f2f860266fb79901405ef5e9965b8c9fe2e88,
//! GPU-MPC/fss/gpu_dpf.cu:45-127,308-434 and gpu_aes_shm.cu:167-198.
//! Unlike the unsafe CPU-Llama terminal conversion, the last expansion retains
//! two complete AES outputs and uses two complete terminal correction words.
//! Randomized complementary root controls are retained: logical key size is
//! 129*(n-7)+255, NOT the paper's 254-bit constant. Only widths 8..=63 exist here.
//! Root secret-seed entropy is 127 bits, not 128; no new security claim is made.
//!
//! PLLM codec is an 18-byte header plus the LSB-first logical key and zero tail
//! padding. Upstream non-evalAll GPU batch payload instead uses
//! 16*N*(d+2)+4*ceil(N/32)*d bytes plus headers; it includes an unused control
//! plane. Neither that layout nor its benchmark RNG is imported here.
//!
//! Private key algebra, not a public reusable protocol handle or security proof.
//! Outer runtime must bind session/role, authenticate material, consume one-use
//! inventories and erase input material. This module cannot detect authenticated
//! same-profile payload substitution or enforce that enclosing lifecycle.
//! Encoding consumes its owner. Evaluation borrows internally for table scans.

use super::{BitReader, BitSink, Bits};
use aes::cipher::{Block, BlockEncrypt, KeyInit};
use aes::Aes128;
use zeroize::{Zeroize, Zeroizing};

const MAX_ORDINARY: usize = 55;
const MAX_FULL_BITS: u8 = 10;
const MAGIC: &[u8; 15] = b"PLLM-DPF-GPU127";
const VERSION: u8 = 1;
const HEADER_BYTES: usize = MAGIC.len() + 3; // version, width, party

#[cfg(test)]
std::thread_local! {
    static AES_BLOCKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ERASED_KEYS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Error {
    Domain,
    Encoding,
    Resource,
    Allocation,
    Randomness,
}

/// Checked public geometry and explicit profile identity, never secret material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Profile {
    bits: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Resources {
    /// Fixed inline owners, excluding caller/runtime, stack scratch and AES schedules.
    pub key_owner_bytes: usize,
    pub pair_owner_bytes: usize,
    pub logical_key_bits: usize,
    pub codec_bits: usize,
    pub codec_bytes: usize,
    /// One paired issuance; upstream's duplicate per-party launches each do this work.
    pub paired_issue_aes_blocks: usize,
    pub point_or_less_aes_blocks: usize,
    /// None means full expansion is rejected before allocation or AES work.
    pub full_expansion_bytes: Option<usize>,
    pub full_expansion_aes_blocks: Option<usize>,
}

impl Profile {
    pub const DOMAIN: &'static str = "pllm.compact-dpf.source-gpu127.v1";

    pub fn new(bits: u8) -> Result<Self, Error> {
        if !(8..=63).contains(&bits) {
            return Err(Error::Domain);
        }
        Ok(Self { bits })
    }

    pub fn bits(self) -> u8 {
        self.bits
    }

    fn depth(self) -> usize {
        usize::from(self.bits - 7)
    }

    fn ordinary(self) -> usize {
        self.depth() - 1
    }

    pub fn resources(self) -> Resources {
        let d = self.depth();
        let logical_key_bits = 128 + 129 * self.ordinary() + 256;
        let codec_bits = HEADER_BYTES * 8 + logical_key_bits;
        Resources {
            key_owner_bytes: std::mem::size_of::<GpuKey>(),
            pair_owner_bytes: 2 * std::mem::size_of::<GpuKey>(),
            logical_key_bits,
            codec_bits,
            codec_bytes: codec_bits.div_ceil(8),
            paired_issue_aes_blocks: 4 * d,
            point_or_less_aes_blocks: d,
            full_expansion_bytes: (self.bits <= MAX_FULL_BITS).then(|| 1usize << self.bits),
            full_expansion_aes_blocks: (self.bits <= MAX_FULL_BITS)
                .then(|| 2 * ((1usize << d) - 1)),
        }
    }
}

/// All corrections are inline: issuance and decode never allocate.
/// No Debug/Clone: secret fields erase on success, partial decode and failure.
pub(crate) struct GpuKey {
    profile: Profile,
    party: u8,
    root: Zeroizing<u128>,
    // Packed seed correction (bits 127..1) and left control correction (bit 0).
    words: Zeroizing<[u128; MAX_ORDINARY]>,
    right: Zeroizing<[u8; MAX_ORDINARY]>,
    terminal: Zeroizing<[u128; 2]>,
}

impl Drop for GpuKey {
    fn drop(&mut self) {
        self.root.zeroize();
        self.words.zeroize();
        self.right.zeroize();
        self.terminal.zeroize();
        #[cfg(test)]
        {
            assert_eq!(*self.root, 0);
            assert!(self.words.iter().all(|v| *v == 0));
            assert!(self.right.iter().all(|v| *v == 0));
            assert!(self.terminal.iter().all(|v| *v == 0));
            ERASED_KEYS.with(|count| count.set(count.get() + 1));
        }
    }
}

fn mask(bit: u8) -> u128 {
    0u128.wrapping_sub(u128::from(bit))
}

// AES workspace is erased even on unwinding. Aes128 itself has the workspace's
// existing `zeroize` dependency feature, which erases its expanded key on drop.
struct ExpansionBlocks([Block<Aes128>; 2]);
impl Drop for ExpansionBlocks {
    fn drop(&mut self) {
        for block in &mut self.0 {
            block.as_mut_slice().zeroize();
        }
    }
}

fn cipher(packed: u128) -> Aes128 {
    let packed = Zeroizing::new(packed);
    let bytes = Zeroizing::new((*packed & !1).to_le_bytes());
    Aes128::new_from_slice(&*bytes).expect("AES-128 key is exactly 16 bytes")
}

fn child(packed: u128, branch: usize) -> Zeroizing<u128> {
    let packed = Zeroizing::new(packed);
    debug_assert!(branch < 2);
    #[cfg(test)]
    AES_BLOCKS.with(|count| count.set(count.get() + 1));
    let cipher = cipher(*packed);
    let mut bytes = Zeroizing::new([0u8; 16]);
    bytes[0] = (2 * branch) as u8;
    cipher.encrypt_block(Block::<Aes128>::from_mut_slice(&mut *bytes));
    Zeroizing::new(u128::from_le_bytes(*bytes))
}

fn children(packed: u128) -> Zeroizing<[u128; 2]> {
    let packed = Zeroizing::new(packed);
    #[cfg(test)]
    AES_BLOCKS.with(|count| count.set(count.get() + 2));
    let cipher = cipher(*packed);
    let mut right = [0u8; 16];
    right[0] = 2;
    let mut blocks = ExpansionBlocks([[0u8; 16].into(), right.into()]);
    cipher.encrypt_blocks(&mut blocks.0);
    Zeroizing::new([
        u128::from_le_bytes(blocks.0[0].into()),
        u128::from_le_bytes(blocks.0[1].into()),
    ])
}

impl GpuKey {
    fn empty(profile: Profile, party: u8) -> Self {
        Self {
            profile,
            party,
            root: Zeroizing::new(0),
            words: Zeroizing::new([0; MAX_ORDINARY]),
            right: Zeroizing::new([0; MAX_ORDINARY]),
            terminal: Zeroizing::new([0; 2]),
        }
    }

    pub fn issue(profile: Profile, point: u64) -> Result<[Self; 2], Error> {
        Self::issue_with(profile, point, |bytes| {
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        })
    }

    /// One 32-byte paired draw, after public geometry and point validation.
    /// Injection supports entropy-failure regression tests without a new RNG.
    pub(crate) fn issue_with(
        profile: Profile,
        point: u64,
        mut random: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<[Self; 2], Error> {
        let point = Zeroizing::new(point);
        if *point >= 1u64 << profile.bits {
            return Err(Error::Domain);
        }
        let mut bytes = Zeroizing::new([0u8; 32]);
        random(&mut *bytes)?;
        let mut roots = Zeroizing::new([
            u128::from_le_bytes(bytes[..16].try_into().expect("fixed root width")),
            u128::from_le_bytes(bytes[16..].try_into().expect("fixed root width")),
        ]);
        roots[0] = (roots[0] & !1) | ((roots[1] & 1) ^ 1);
        let mut states = Zeroizing::new(*roots);
        let mut keys = [Self::empty(profile, 0), Self::empty(profile, 1)];
        for party in 0..2 {
            *keys[party].root = roots[party];
        }
        for level in 0..profile.ordinary() {
            let keep = ((*point >> (usize::from(profile.bits) - 1 - level)) & 1) as usize;
            let a = children(states[0]);
            let b = children(states[1]);
            let seed = Zeroizing::new((a[1 - keep] ^ b[1 - keep]) & !1);
            let controls = Zeroizing::new([
                ((a[0] ^ b[0]) & 1) as u8 ^ keep as u8 ^ 1,
                ((a[1] ^ b[1]) & 1) as u8 ^ keep as u8,
            ]);
            for party in 0..2 {
                keys[party].words[level] = *seed | u128::from(controls[0]);
                keys[party].right[level] = controls[1];
                let expansion = if party == 0 { &a } else { &b };
                let correction = Zeroizing::new(*seed | u128::from(controls[keep]));
                states[party] = expansion[keep] ^ (*correction & mask((states[party] & 1) as u8));
            }
        }
        // Final expansion replaces, rather than follows, the last ordinary CW.
        let a = children(states[0]);
        let b = children(states[1]);
        let keep = ((*point >> 7) & 1) as usize;
        let unit = Zeroizing::new(1u128 << (*point & 127));
        for branch in 0..2 {
            let terminal =
                Zeroizing::new(a[branch] ^ b[branch] ^ (*unit & mask((branch == keep) as u8)));
            for key in &mut keys {
                key.terminal[branch] = *terminal;
            }
        }
        Ok(keys)
    }

    fn correction(&self, level: usize, branch: usize) -> Zeroizing<u128> {
        let word = Zeroizing::new(self.words[level]);
        Zeroizing::new(
            (*word & !1)
                | if branch == 0 {
                    *word & 1
                } else {
                    u128::from(self.right[level])
                },
        )
    }

    fn descend(&self, level: usize, state: u128, branch: usize) -> Zeroizing<u128> {
        let state = Zeroizing::new(state);
        let expanded = child(*state, branch);
        let correction = self.correction(level, branch);
        Zeroizing::new(*expanded ^ (*correction & mask((*state & 1) as u8)))
    }

    fn terminal_block(&self, state: u128, branch: usize) -> Zeroizing<u128> {
        let state = Zeroizing::new(state);
        let expanded = child(*state, branch);
        Zeroizing::new(*expanded ^ (self.terminal[branch] & mask((*state & 1) as u8)))
    }

    fn check_input(&self, input: u64) -> Result<(), Error> {
        if input >= 1u64 << self.profile.bits {
            Err(Error::Domain)
        } else {
            Ok(())
        }
    }

    pub fn point(&self, input: u64) -> Result<u8, Error> {
        self.check_input(input)?;
        let mut state = Zeroizing::new(*self.root);
        for level in 0..self.profile.ordinary() {
            let branch = ((input >> (usize::from(self.profile.bits) - 1 - level)) & 1) as usize;
            state = self.descend(level, *state, branch);
        }
        let leaf = self.terminal_block(*state, ((input >> 7) & 1) as usize);
        Ok(((*leaf >> (input & 127)) & 1) as u8)
    }

    /// XOR shares of unsigned input < secret point, as in source doDcf.
    /// Signed fixed-point comparison/range normalization belongs to outer gates;
    /// this Boolean primitive does not silently reinterpret two's-complement bits.
    pub fn less(&self, input: u64) -> Result<u8, Error> {
        self.check_input(input)?;
        let mut state = Zeroizing::new(*self.root);
        let mut parity = Zeroizing::new(self.party);
        let mut previous = 0u8;
        for level in 0..self.profile.ordinary() {
            let branch = ((input >> (usize::from(self.profile.bits) - 1 - level)) & 1) as u8;
            *parity ^= (*state & 1) as u8 & (previous ^ branch);
            state = self.descend(level, *state, usize::from(branch));
            previous = branch;
        }
        let branch = ((input >> 7) & 1) as u8;
        *parity ^= (*state & 1) as u8 & (previous ^ branch);
        let leaf = self.terminal_block(*state, usize::from(branch));
        let offset = input & 127;
        let selected = Zeroizing::new(if branch == 0 {
            *leaf
                & if offset == 127 {
                    u128::MAX
                } else {
                    (1u128 << (offset + 1)) - 1
                }
        } else if offset == 127 {
            0 // Empty suffix. Never shift u128 by 128.
        } else {
            *leaf >> (offset + 1)
        });
        Ok(*parity ^ ((selected.count_ones() & 1) as u8))
    }

    fn walk(&self, depth: usize, state: u128, out: &mut Vec<u8>) {
        let state = Zeroizing::new(state);
        let expanded = children(*state);
        for branch in 0..2 {
            if depth == self.profile.ordinary() {
                let leaf = Zeroizing::new(
                    expanded[branch] ^ (self.terminal[branch] & mask((*state & 1) as u8)),
                );
                out.extend((0..128).map(|offset| ((*leaf >> offset) & 1) as u8));
            } else {
                let correction = self.correction(depth, branch);
                let next =
                    Zeroizing::new(expanded[branch] ^ (*correction & mask((*state & 1) as u8)));
                self.walk(depth + 1, *next, out);
            }
        }
    }

    /// Public-order full scan, at most 1024 output bytes / 14 AES blocks per key.
    /// Geometry and caller's output-byte limit precede callback, allocation, AES.
    pub fn all(&self, allocation_limit: usize) -> Result<Zeroizing<Vec<u8>>, Error> {
        self.all_with_reservation(allocation_limit, |_| Ok(()))
    }

    pub(crate) fn all_with_reservation(
        &self,
        allocation_limit: usize,
        mut before_reserve: impl FnMut(usize) -> Result<(), Error>,
    ) -> Result<Zeroizing<Vec<u8>>, Error> {
        let count = self
            .profile
            .resources()
            .full_expansion_bytes
            .ok_or(Error::Resource)?;
        if count > allocation_limit {
            return Err(Error::Resource);
        }
        before_reserve(count)?;
        let mut out = Zeroizing::new(Vec::new());
        out.try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        self.walk(0, *self.root, &mut out);
        debug_assert_eq!(out.len(), count);
        Ok(out)
    }

    fn write(&self, output: &mut impl BitSink) {
        for byte in MAGIC {
            output.put(u128::from(*byte), 8);
        }
        for byte in [VERSION, self.profile.bits, self.party] {
            output.put(u128::from(byte), 8);
        }
        output.put(*self.root, 128);
        for level in 0..self.profile.ordinary() {
            output.put(self.words[level] >> 1, 127);
            output.put(self.words[level] & 1, 1);
            output.put(u128::from(self.right[level]), 1);
        }
        for word in self.terminal.iter() {
            output.put(*word, 128);
        }
    }

    /// PLLM canonical framing, NOT upstream's GPU batch layout or legacy codec.
    pub fn into_material(self, allocation_limit: usize) -> Result<Zeroizing<Vec<u8>>, Error> {
        self.into_material_with_reservation(allocation_limit, |_| Ok(()))
    }

    pub(crate) fn into_material_with_reservation(
        self,
        allocation_limit: usize,
        mut before_reserve: impl FnMut(usize) -> Result<(), Error>,
    ) -> Result<Zeroizing<Vec<u8>>, Error> {
        let cost = self.profile.resources();
        if cost.codec_bytes > allocation_limit {
            return Err(Error::Resource);
        }
        before_reserve(cost.codec_bytes)?;
        let mut output = Bits::new();
        output
            .data
            .try_reserve_exact(cost.codec_bytes)
            .map_err(|_| Error::Allocation)?;
        self.write(&mut output);
        debug_assert_eq!(output.len, cost.codec_bits);
        // Moving the buffer does not leave a second material owner behind.
        Ok(output.data)
    }

    /// Exact expected width/party and framing/padding checked before key parsing.
    /// Fixed correction arrays mean malformed partial material cannot allocate.
    pub fn decode(profile: Profile, party: u8, bytes: &[u8]) -> Result<Self, Error> {
        if party > 1 {
            return Err(Error::Domain);
        }
        let mut input =
            BitReader::new(bytes, profile.resources().codec_bits).map_err(|_| Error::Encoding)?;
        for byte in MAGIC.iter().copied().chain([VERSION, profile.bits, party]) {
            if input.take(8).map_err(|_| Error::Encoding)? != u128::from(byte) {
                return Err(Error::Encoding);
            }
        }
        let mut key = Self::empty(profile, party);
        *key.root = input.take(128).map_err(|_| Error::Encoding)?;
        for level in 0..profile.ordinary() {
            key.words[level] = input.take(127).map_err(|_| Error::Encoding)? << 1;
            key.words[level] |= input.take(1).map_err(|_| Error::Encoding)?;
            key.right[level] = input.take(1).map_err(|_| Error::Encoding)? as u8;
        }
        for word in key.terminal.iter_mut() {
            *word = input.take(128).map_err(|_| Error::Encoding)?;
        }
        input.finish().map_err(|_| Error::Encoding)?;
        Ok(key)
    }
}

#[cfg(test)]
mod tests;
