//! Exact, one-use two-share signed rescaling and DReLU research reference.
//!
//! Sources: SIGMA §4.2.2–4.2.3 (truncate/reduce then sign extension), and
//! FuseFSS §4.3, §4.5 and Appendix E (mask-aware fused helper evaluation).
//! Both layouts select the SAME comparison backend: the quadratic prefix-DPF
//! control or Figure 1's linear-size DCF from FSS for Mixed-Mode Secure
//! Computation. Neither is a paper-system reproduction. Ties-even extends ARS.
//! Each evaluator holds one share; the in-process dealer is research-only.
//! Framing binds context/phase/issuance but is not transport authentication or
//! malicious-output verification. No Pipeline, native tensor schedule or decoder.

use crate::{compact_dcf, point_fss};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub const MAX_LANES: usize = 1024;
pub const MAX_ALLOCATION_ESTIMATE: usize = 64 << 20;
pub const OPENING_HEADER_BYTES: usize = 78;
const MAGIC: &[u8; 8] = b"PLLMRS01";

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Domain,
    Resource,
    Randomness,
    Consumed,
    Context,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    Floor,
    TiesEven,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Fused,
    Unfused,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    PrefixDpf,
    CompactDcf,
}

impl Backend {
    fn comparison_bytes(self, bits: u8) -> usize {
        match self {
            Self::PrefixDpf => point_fss::Comparison::bytes(bits),
            Self::CompactDcf => compact_dcf::Comparison::bytes(bits),
        }
    }
}

enum Comparison {
    Prefix(point_fss::Comparison),
    Compact(compact_dcf::Comparison),
}

impl Comparison {
    fn issue(backend: Backend, bits: u8, threshold: u32) -> Result<[Self; 2], Error> {
        match backend {
            Backend::PrefixDpf => {
                Ok(point_fss::Comparison::issue(bits, threshold)?.map(Self::Prefix))
            }
            Backend::CompactDcf => {
                Ok(compact_dcf::Comparison::issue(bits, threshold)?.map(Self::Compact))
            }
        }
    }

    fn eval(&self, party: u8, input: u32) -> u32 {
        match self {
            Self::Prefix(key) => key.eval(party, input),
            Self::Compact(key) => key.eval(party, input),
        }
    }

    fn payload_bytes(&self) -> usize {
        match self {
            Self::Prefix(key) => key.payload_bytes(),
            Self::Compact(key) => key.payload_bytes(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Descriptor {
    bits: u8,
    shift: u8,
    rounding: Rounding,
    backend: Backend,
}

/// Public caller binding. The operation digest can bind any semantic operator;
/// it does not imply that a compiler has admitted this reference.
#[derive(Clone, Copy)]
pub struct Context {
    pub plan: [u8; 32],
    pub session: [u8; 32],
    pub operation: [u8; 32],
    pub first_tensor_index: u64,
}

#[derive(Debug)]
pub struct Resources {
    pub lanes: usize,
    pub rounds: usize,
    /// Encoded roots, corrections, final words, mask and constant shares.
    /// No offline transport codec exists; metadata/allocator storage is extra.
    pub party_key_payload_bytes: usize,
    pub party_allocation_estimate_bytes: usize,
    pub total_allocation_estimate_bytes: usize,
    /// Actual online frame lengths, summed across both directions and rounds.
    pub peer_frame_bytes: usize,
    pub comparison_widths: Vec<u8>,
}

fn mask(bits: u8) -> u32 {
    ((1u64 << bits) - 1) as u32
}

fn random_word() -> Result<u32, Error> {
    let mut bytes = Zeroizing::new([0; 4]);
    getrandom::fill(&mut *bytes).map_err(|_| Error::Randomness)?;
    Ok(u32::from_le_bytes(*bytes))
}

fn share(value: u32) -> Result<[u32; 2], Error> {
    let left = random_word()?;
    Ok([left, value.wrapping_sub(left)])
}

impl Descriptor {
    pub fn new(bits: u8, shift: u8, rounding: Rounding) -> Result<Self, Error> {
        if !(2..=32).contains(&bits) || shift == 0 || shift >= bits {
            return Err(Error::Domain);
        }
        Ok(Self {
            bits,
            shift,
            rounding,
            backend: Backend::PrefixDpf,
        })
    }

    /// Select an explicit research backend; the default preserves the control.
    pub fn with_backend(self, backend: Backend) -> Self {
        Self { backend, ..self }
    }

    pub fn bits(self) -> u8 {
        self.bits
    }
    pub fn shift(self) -> u8 {
        self.shift
    }
    pub fn rounding(self) -> Rounding {
        self.rounding
    }

    pub fn resources(self, layout: Layout, lanes: usize) -> Result<Resources, Error> {
        if !(1..=MAX_LANES).contains(&lanes) {
            return Err(Error::Resource);
        }
        let mut widths = match layout {
            Layout::Fused => vec![self.bits, self.shift, self.bits],
            Layout::Unfused => vec![self.shift],
        };
        if self.rounding == Rounding::TiesEven {
            widths.extend([self.shift + 1; 4]);
        }
        if layout == Layout::Unfused {
            widths.extend([self.bits - self.shift; 2]);
        }
        let rounds = if layout == Layout::Fused { 1 } else { 2 };
        // Each phase/lane retains one mask share and three constant shares.
        let payload = lanes
            * (16 * rounds
                + widths
                    .iter()
                    .map(|&bits| self.backend.comparison_bytes(bits))
                    .sum::<usize>());
        // Conservative owner/scratch estimate; not a measured peak or OOM proof.
        let party = 3 * payload + 4096 * lanes;
        let total = 2 * party + 64 * 1024;
        if total > MAX_ALLOCATION_ESTIMATE {
            return Err(Error::Resource);
        }
        Ok(Resources {
            lanes,
            rounds,
            party_key_payload_bytes: payload,
            party_allocation_estimate_bytes: party,
            total_allocation_estimate_bytes: total,
            peer_frame_bytes: 2 * rounds * (OPENING_HEADER_BYTES + 4 * lanes),
            comparison_widths: widths,
        })
    }

    pub fn issue_reference(
        self,
        layout: Layout,
        context: Context,
        lanes: usize,
    ) -> Result<[Party; 2], Error> {
        // Preflight the entire batch before sampling or allocating a key prefix.
        self.resources(layout, lanes)?;
        context
            .first_tensor_index
            .checked_add(lanes as u64)
            .ok_or(Error::Domain)?;
        let mut hash = Sha256::new();
        hash.update(b"pllm.shared_rescale_reference.v1\0");
        hash.update([self.bits, self.shift, self.rounding as u8, layout as u8]);
        hash.update((lanes as u64).to_le_bytes());
        hash.update(context.plan);
        hash.update(context.session);
        hash.update(context.operation);
        hash.update(context.first_tensor_index.to_le_bytes());
        if self.backend == Backend::CompactDcf {
            hash.update(b"compact_dcf.v1\0");
        }
        let binding: [u8; 32] = hash.finalize().into();
        let mut nonce = [0; 32];
        getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
        let mut phases = [Vec::with_capacity(2), Vec::with_capacity(2)];
        for phase in 0..if layout == Layout::Fused { 1 } else { 2 } {
            let mut batch = [Vec::with_capacity(lanes), Vec::with_capacity(lanes)];
            for _ in 0..lanes {
                let [left, right] = issue_lane(self, layout, phase)?;
                batch[0].push(left);
                batch[1].push(right);
            }
            let [left, right] = batch;
            phases[0].push(left);
            phases[1].push(right);
        }
        let make = |id, mut phases: Vec<Vec<Lane>>| {
            let next = if phases.len() == 2 {
                Some(phases.pop().unwrap())
            } else {
                None
            };
            Party {
                id,
                descriptor: self,
                layout,
                lanes,
                binding,
                nonce,
                state: Some(State::Ready(Material {
                    current: phases.pop().unwrap(),
                    next,
                })),
            }
        };
        let [left, right] = phases;
        Ok([make(0, left), make(1, right)])
    }
}

struct Lane {
    mask_share: Zeroizing<u32>,
    constants: Zeroizing<[u32; 3]>,
    comparisons: Vec<Comparison>,
}

impl Lane {
    fn payload_bytes(&self) -> usize {
        16 + self
            .comparisons
            .iter()
            .map(Comparison::payload_bytes)
            .sum::<usize>()
    }
}

fn issue_lane(d: Descriptor, layout: Layout, phase: u8) -> Result<[Lane; 2], Error> {
    let bits = if phase == 1 { d.bits - d.shift } else { d.bits };
    let ring = 1u64 << bits;
    let r = Zeroizing::new(random_word()? & mask(bits));
    let h = ring / 2;
    let theta = ((u64::from(*r) + h) % ring) as u32;
    let carry = u32::from(u64::from(*r) + h >= ring);
    let mut queries = Zeroizing::new(Vec::with_capacity(7));
    let mut constants = Zeroizing::new([0u32; 3]);
    if phase == 1 {
        queries.extend([(bits, theta), (bits, *r)]);
        constants[0] = r
            .wrapping_neg()
            .wrapping_add((ring as u32).wrapping_mul(carry.wrapping_sub(1)));
        constants[1] = carry;
    } else if layout == Layout::Fused {
        queries.extend([(bits, theta), (d.shift, *r & mask(d.shift)), (bits, *r)]);
        constants[0] = (*r >> d.shift)
            .wrapping_neg()
            .wrapping_add((1u32 << (bits - d.shift)).wrapping_mul(carry.wrapping_sub(1)));
        constants[1] = carry;
    } else {
        queries.push((d.shift, *r & mask(d.shift)));
        constants[0] = (*r >> d.shift).wrapping_neg();
    }
    if phase == 0 && d.rounding == Rounding::TiesEven {
        // Increment floor iff the low f+1 bits lie in [2^(f-1)+1,2^f)
        // or [2^f+2^(f-1),2^(f+1)). This includes negative ties correctly.
        let width = d.shift + 1;
        let small_ring = 1u64 << width;
        let low_mask = u64::from(*r & mask(width));
        let unit = 1u64 << d.shift;
        let boundaries = [unit, unit / 2 + 1, unit + unit / 2, 0];
        let carries = boundaries.map(|b| u32::from(low_mask + b >= small_ring));
        for boundary in boundaries {
            queries.push((width, ((low_mask + boundary) % small_ring) as u32));
        }
        constants[2] = 1u32
            .wrapping_add(carries[0])
            .wrapping_sub(carries[1])
            .wrapping_sub(carries[2]);
    }
    let masks = Zeroizing::new(share(*r)?);
    let constant_shares = Zeroizing::new([
        share(constants[0])?,
        share(constants[1])?,
        share(constants[2])?,
    ]);
    let mut comparisons = [
        Vec::with_capacity(queries.len()),
        Vec::with_capacity(queries.len()),
    ];
    for (bits, threshold) in queries.iter().copied() {
        let [left, right] = Comparison::issue(d.backend, bits, threshold)?;
        comparisons[0].push(left);
        comparisons[1].push(right);
    }
    let [left, right] = comparisons;
    Ok([(0, left), (1, right)].map(|(id, comparisons)| Lane {
        mask_share: Zeroizing::new(masks[id] & mask(bits)),
        constants: Zeroizing::new(constant_shares.map(|pair| pair[id])),
        comparisons,
    }))
}

struct Material {
    current: Vec<Lane>,
    next: Option<Vec<Lane>>,
}

enum State {
    Ready(Material),
    Waiting {
        phase: u8,
        material: Material,
        own: Zeroizing<Vec<u32>>,
        rounding: Zeroizing<Vec<u32>>,
    },
}

/// Opaque, non-cloneable party-local one-use ownership. Dropping or cancelling
/// erases the retained masks, constants, seeds and correction words.
pub struct Party {
    id: u8,
    descriptor: Descriptor,
    layout: Layout,
    lanes: usize,
    binding: [u8; 32],
    nonce: [u8; 32],
    state: Option<State>,
}

/// Only the trusted test/probe client reconstructs the two output shares.
pub struct OutputShare {
    pub rescaled: u32,
    pub nonnegative: u32,
}

pub enum Progress {
    Open(Vec<u8>),
    Complete(Vec<OutputShare>),
}

impl Party {
    pub fn cancel(&mut self) {
        self.state = None;
    }

    pub fn key_payload_bytes(&self) -> usize {
        let material = match &self.state {
            Some(State::Ready(material)) | Some(State::Waiting { material, .. }) => material,
            None => return 0,
        };
        material
            .current
            .iter()
            .chain(material.next.iter().flatten())
            .map(Lane::payload_bytes)
            .sum()
    }

    pub fn start(&mut self, input: &[u32]) -> Result<Vec<u8>, Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        let State::Ready(material) = state else {
            return Err(Error::Context);
        };
        if input.len() != self.lanes || input.iter().any(|&x| x > mask(self.descriptor.bits)) {
            return Err(Error::Domain);
        }
        let own = Zeroizing::new(
            input
                .iter()
                .zip(&material.current)
                .map(|(&x, lane)| x.wrapping_add(*lane.mask_share) & mask(self.descriptor.bits))
                .collect::<Vec<_>>(),
        );
        let frame = self.frame(0, &own);
        self.state = Some(State::Waiting {
            phase: 0,
            material,
            own,
            rounding: Zeroizing::new(Vec::new()),
        });
        Ok(frame)
    }

    pub fn advance(&mut self, peer: &[u8]) -> Result<Progress, Error> {
        // Burn all remaining phases BEFORE parsing even the peer's frame header.
        let state = self.state.take().ok_or(Error::Consumed)?;
        let State::Waiting {
            phase,
            mut material,
            own,
            rounding,
        } = state
        else {
            return Err(Error::Context);
        };
        let d = self.descriptor;
        let bits = if phase == 0 { d.bits } else { d.bits - d.shift };
        let peer_words = self.parse(peer, phase, bits)?;
        let mut scaled = Zeroizing::new(Vec::with_capacity(self.lanes));
        let mut signs = Zeroizing::new(Vec::with_capacity(self.lanes));
        let mut increments = Zeroizing::new(Vec::with_capacity(self.lanes));
        for (index, lane) in material.current.iter().enumerate() {
            let opened = own[index].wrapping_add(peer_words[index]) & mask(bits);
            let public = if self.id == 0 { opened } else { 0 };
            let eval =
                |i: usize, width: u8| lane.comparisons[i].eval(self.id, opened & mask(width));
            let (value, nonnegative, round_start) = if phase == 1 {
                let upper = eval(0, bits);
                (
                    public
                        .wrapping_add(lane.constants[0])
                        .wrapping_add((1u32 << bits).wrapping_mul(upper)),
                    lane.constants[1]
                        .wrapping_add(upper)
                        .wrapping_sub(eval(1, bits)),
                    0,
                )
            } else if self.layout == Layout::Fused {
                let upper = eval(0, bits);
                (
                    (public >> d.shift)
                        .wrapping_add(lane.constants[0])
                        .wrapping_add((1u32 << (bits - d.shift)).wrapping_mul(upper))
                        .wrapping_sub(eval(1, d.shift)),
                    lane.constants[1]
                        .wrapping_add(upper)
                        .wrapping_sub(eval(2, bits)),
                    3,
                )
            } else {
                (
                    (public >> d.shift)
                        .wrapping_add(lane.constants[0])
                        .wrapping_sub(eval(0, d.shift)),
                    0,
                    1,
                )
            };
            let increment = if phase == 1 {
                rounding[index]
            } else if d.rounding == Rounding::Floor {
                0
            } else {
                lane.constants[2]
                    .wrapping_add(eval(round_start, d.shift + 1))
                    .wrapping_sub(eval(round_start + 1, d.shift + 1))
                    .wrapping_sub(eval(round_start + 2, d.shift + 1))
                    .wrapping_add(eval(round_start + 3, d.shift + 1))
            };
            scaled.push(value);
            signs.push(nonnegative);
            increments.push(increment);
        }
        if let Some(next) = material.next.take() {
            let own = Zeroizing::new(
                scaled
                    .iter()
                    .zip(&next)
                    .map(|(&x, lane)| x.wrapping_add(*lane.mask_share) & mask(d.bits - d.shift))
                    .collect::<Vec<_>>(),
            );
            let frame = self.frame(1, &own);
            self.state = Some(State::Waiting {
                phase: 1,
                material: Material {
                    current: next,
                    next: None,
                },
                own,
                rounding: increments,
            });
            Ok(Progress::Open(frame))
        } else {
            Ok(Progress::Complete(
                (0..self.lanes)
                    .map(|i| OutputShare {
                        rescaled: scaled[i].wrapping_add(increments[i]) & mask(d.bits),
                        nonnegative: signs[i] & mask(d.bits),
                    })
                    .collect(),
            ))
        }
    }

    fn frame(&self, phase: u8, words: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(OPENING_HEADER_BYTES + 4 * self.lanes);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.binding);
        bytes.extend_from_slice(&self.nonce);
        bytes.extend_from_slice(&[phase, self.id]);
        bytes.extend_from_slice(&(self.lanes as u32).to_le_bytes());
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn parse(&self, bytes: &[u8], phase: u8, bits: u8) -> Result<Zeroizing<Vec<u32>>, Error> {
        if bytes.len() != OPENING_HEADER_BYTES + 4 * self.lanes
            || &bytes[..8] != MAGIC
            || bytes[8..40] != self.binding
            || bytes[40..72] != self.nonce
            || bytes[72] != phase
            || bytes[73] != 1 - self.id
            || u32::from_le_bytes(bytes[74..78].try_into().unwrap()) as usize != self.lanes
        {
            return Err(Error::Context);
        }
        let words = Zeroizing::new(
            bytes[OPENING_HEADER_BYTES..]
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
        );
        if words.iter().any(|&word| word > mask(bits)) {
            return Err(Error::Domain);
        }
        Ok(words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> Context {
        Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: 0,
        }
    }

    fn oracle(d: Descriptor, value: u32) -> [u32; 2] {
        let signed = if value & (1 << (d.bits - 1)) == 0 {
            i64::from(value)
        } else {
            i64::from(value) - (1i64 << d.bits)
        };
        let divisor = 1i64 << d.shift;
        let mut quotient = signed.div_euclid(divisor);
        let remainder = signed.rem_euclid(divisor);
        if d.rounding == Rounding::TiesEven
            && (remainder * 2 > divisor || (remainder * 2 == divisor && quotient % 2 != 0))
        {
            quotient += 1;
        }
        [quotient as u32 & mask(d.bits), u32::from(signed >= 0)]
    }

    fn run(d: Descriptor, layout: Layout, values: &[u32]) {
        let cost = d.resources(layout, values.len()).unwrap();
        let [mut a, mut b] = d.issue_reference(layout, context(), values.len()).unwrap();
        assert_eq!(a.key_payload_bytes(), cost.party_key_payload_bytes);
        assert_eq!(b.key_payload_bytes(), cost.party_key_payload_bytes);
        let left: Vec<_> = values
            .iter()
            .map(|_| random_word().unwrap() & mask(d.bits))
            .collect();
        let right: Vec<_> = values
            .iter()
            .zip(&left)
            .map(|(&x, &share)| x.wrapping_sub(share) & mask(d.bits))
            .collect();
        let mut frames = [a.start(&left).unwrap(), b.start(&right).unwrap()];
        let mut bytes = 0;
        for round in 0..cost.rounds {
            bytes += frames[0].len() + frames[1].len();
            match (
                a.advance(&frames[1]).unwrap(),
                b.advance(&frames[0]).unwrap(),
            ) {
                (Progress::Open(af), Progress::Open(bf)) => {
                    assert_eq!(round, 0);
                    frames = [af, bf];
                }
                (Progress::Complete(ao), Progress::Complete(bo)) => {
                    assert_eq!(round + 1, cost.rounds);
                    for ((aa, bb), &x) in ao.iter().zip(&bo).zip(values) {
                        assert_eq!(
                            [
                                aa.rescaled.wrapping_add(bb.rescaled) & mask(d.bits),
                                aa.nonnegative.wrapping_add(bb.nonnegative) & mask(d.bits)
                            ],
                            oracle(d, x),
                            "{layout:?} {}>>{} x={x}",
                            d.bits,
                            d.shift
                        );
                    }
                }
                _ => panic!("phase mismatch"),
            }
        }
        assert_eq!(bytes, cost.peer_frame_bytes);
        assert_eq!(a.key_payload_bytes(), 0);
        assert!(matches!(a.advance(&frames[1]), Err(Error::Consumed)));
        assert!(matches!(a.start(&left), Err(Error::Consumed)));
    }

    #[test]
    fn exhaustive_signed_domains_and_all_shifts() {
        for bits in 2..=6 {
            let values: Vec<_> = (0..1 << bits).collect();
            for shift in 1..bits {
                for rounding in [Rounding::Floor, Rounding::TiesEven] {
                    for layout in [Layout::Fused, Layout::Unfused] {
                        for backend in [Backend::PrefixDpf, Backend::CompactDcf] {
                            run(
                                Descriptor::new(bits, shift, rounding)
                                    .unwrap()
                                    .with_backend(backend),
                                layout,
                                &values,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn full_width_boundaries_negative_ties_and_random_inputs() {
        for (bits, shift) in [(16, 7), (24, 12), (32, 7), (32, 31)] {
            let m = mask(bits);
            let half = 1u32 << (shift - 1);
            let sign = 1 << (bits - 1);
            let mut values = vec![0, 1, m, sign, sign - 1, sign + 1];
            for center in [
                half,
                half.wrapping_mul(3),
                half.wrapping_neg(),
                half.wrapping_mul(3).wrapping_neg(),
            ] {
                values.extend([
                    center.wrapping_sub(1) & m,
                    center & m,
                    center.wrapping_add(1) & m,
                ]);
            }
            for _ in 0..16 {
                values.push(random_word().unwrap() & m);
            }
            for rounding in [Rounding::Floor, Rounding::TiesEven] {
                for layout in [Layout::Fused, Layout::Unfused] {
                    for backend in [Backend::PrefixDpf, Backend::CompactDcf] {
                        run(
                            Descriptor::new(bits, shift, rounding)
                                .unwrap()
                                .with_backend(backend),
                            layout,
                            &values,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_context_role_phase_domain_and_replay_burn_both_phases() {
        for backend in [Backend::PrefixDpf, Backend::CompactDcf] {
            malformed_lifecycle(
                Descriptor::new(8, 3, Rounding::TiesEven)
                    .unwrap()
                    .with_backend(backend),
            );
        }
    }

    fn malformed_lifecycle(d: Descriptor) {
        for layout in [Layout::Fused, Layout::Unfused] {
            for byte in [0, 8, 40, 72, 73, 74, 81] {
                let [mut a, mut b] = d.issue_reference(layout, context(), 1).unwrap();
                a.start(&[1]).unwrap();
                let mut peer = b.start(&[1]).unwrap();
                peer[byte] ^= 128;
                assert!(a.advance(&peer).is_err());
                assert_eq!(a.key_payload_bytes(), 0);
                assert!(matches!(a.advance(&peer), Err(Error::Consumed)));
            }
            let [mut a, _] = d.issue_reference(layout, context(), 1).unwrap();
            let [_, mut foreign] = d.issue_reference(layout, context(), 1).unwrap();
            a.start(&[1]).unwrap();
            assert!(matches!(
                a.advance(&foreign.start(&[1]).unwrap()),
                Err(Error::Context)
            ));
            let [mut a, _] = d.issue_reference(layout, context(), 1).unwrap();
            assert!(matches!(a.start(&[256]), Err(Error::Domain)));
            assert!(matches!(a.start(&[0]), Err(Error::Consumed)));
            let [mut a, _] = d.issue_reference(layout, context(), 1).unwrap();
            a.cancel();
            assert!(matches!(a.start(&[0]), Err(Error::Consumed)));
            let [mut a, _] = d.issue_reference(layout, context(), 1).unwrap();
            assert!(matches!(a.advance(&[]), Err(Error::Context)));
            assert_eq!(a.key_payload_bytes(), 0);
            let [mut a, _] = d.issue_reference(layout, context(), 1).unwrap();
            a.start(&[0]).unwrap();
            assert!(matches!(a.start(&[0]), Err(Error::Context)));
            assert_eq!(a.key_payload_bytes(), 0);
            for length in [0, OPENING_HEADER_BYTES - 1, OPENING_HEADER_BYTES + 5] {
                let [mut a, _] = d.issue_reference(layout, context(), 1).unwrap();
                a.start(&[0]).unwrap();
                assert!(matches!(a.advance(&vec![0; length]), Err(Error::Context)));
                assert_eq!(a.key_payload_bytes(), 0);
            }
        }
        let [mut a, mut b] = d.issue_reference(Layout::Unfused, context(), 1).unwrap();
        let af = a.start(&[7]).unwrap();
        let bf = b.start(&[0]).unwrap();
        assert!(matches!(a.advance(&bf), Ok(Progress::Open(_))));
        assert!(matches!(b.advance(&af), Ok(Progress::Open(_))));
        assert!(matches!(a.advance(&bf), Err(Error::Context)));
        assert_eq!(a.key_payload_bytes(), 0);
        let [mut a, mut b] = d.issue_reference(Layout::Unfused, context(), 1).unwrap();
        a.start(&[0]).unwrap();
        let frame = b.start(&[0]).unwrap();
        assert!(matches!(a.advance(&frame), Ok(Progress::Open(_))));
        a.cancel();
        assert_eq!(a.key_payload_bytes(), 0);
        assert!(matches!(a.advance(&frame), Err(Error::Consumed)));
    }

    #[test]
    fn backend_binding_is_independent_of_issuance_nonce() {
        let d = Descriptor::new(16, 7, Rounding::TiesEven).unwrap();
        for layout in [Layout::Fused, Layout::Unfused] {
            let [mut prefix, _] = d.issue_reference(layout, context(), 1).unwrap();
            let [_, mut compact] = d
                .with_backend(Backend::CompactDcf)
                .issue_reference(layout, context(), 1)
                .unwrap();
            // Force equal nonces to exercise the backend commitment itself.
            compact.nonce = prefix.nonce;
            let pf = prefix.start(&[0]).unwrap();
            let cf = compact.start(&[0]).unwrap();
            assert!(matches!(prefix.advance(&cf), Err(Error::Context)));
            assert!(matches!(compact.advance(&pf), Err(Error::Context)));
            assert_eq!(prefix.key_payload_bytes(), 0);
            assert_eq!(compact.key_payload_bytes(), 0);
        }
    }

    #[test]
    fn admission_is_bounded_and_layout_shapes_are_public() {
        for (bits, shift) in [(1, 0), (33, 7), (8, 8), (8, 0)] {
            assert!(Descriptor::new(bits, shift, Rounding::Floor).is_err());
        }
        let d = Descriptor::new(32, 31, Rounding::TiesEven).unwrap();
        for lanes in [0, MAX_LANES, MAX_LANES + 1, usize::MAX] {
            assert!(matches!(
                d.issue_reference(Layout::Fused, context(), lanes),
                Err(Error::Resource)
            ));
        }
        let mut c = context();
        c.first_tensor_index = u64::MAX;
        assert!(matches!(
            d.issue_reference(Layout::Fused, c, 1),
            Err(Error::Domain)
        ));
        for layout in [Layout::Fused, Layout::Unfused] {
            for backend in [Backend::PrefixDpf, Backend::CompactDcf] {
                let d = d.with_backend(backend);
                let costs = d.resources(layout, 1).unwrap();
                for _ in 0..4 {
                    let [a, b] = d.issue_reference(layout, context(), 1).unwrap();
                    assert_eq!(a.key_payload_bytes(), costs.party_key_payload_bytes);
                    assert_eq!(a.key_payload_bytes(), b.key_payload_bytes());
                }
                for lanes in [0, MAX_LANES + 1, usize::MAX] {
                    assert!(matches!(
                        d.issue_reference(layout, context(), lanes),
                        Err(Error::Resource)
                    ));
                }
                assert!(matches!(
                    d.issue_reference(layout, c, 1),
                    Err(Error::Domain)
                ));
            }
        }
        let compact = d
            .with_backend(Backend::CompactDcf)
            .resources(Layout::Fused, MAX_LANES)
            .unwrap();
        assert!(compact.total_allocation_estimate_bytes <= MAX_ALLOCATION_ESTIMATE);
    }
}
