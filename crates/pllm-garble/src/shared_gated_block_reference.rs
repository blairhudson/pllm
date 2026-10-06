//! Complete bounded Q14 -> Q7 quadratic SiLU x up -> Q7 shared block.
//! This is a FuseFSS-inspired arithmetic composition with explicit Beaver
//! post-processing, not the paper's coefficient-lookup compiler or Qwen numerics.
//! Only the trusted fixture checks the [-1,1] input bound. Secret range proof,
//! float-to-fixed conversion, distributed dealer and transport remain absent.
use crate::shared_arithmetic_reference as mul;
use crate::shared_rescale_reference::{
    self as rs, Backend, Context, Descriptor, Error, Layout, Rounding,
};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

pub const BITS: u8 = 24;
pub const MASK: u32 = (1 << BITS) - 1;
pub const MAX_LANES: usize = 512;
pub const MAX_ALLOCATION_ESTIMATE: usize = 64 << 20;

#[derive(Debug)]
pub struct Resources {
    pub party_key_payload_bytes: usize,
    pub total_allocation_estimate_bytes: usize,
    pub peer_frame_bytes: usize,
    pub rounds: usize,
}

fn descriptor(backend: Backend, shift: u8) -> Descriptor {
    Descriptor::new(BITS, shift, Rounding::TiesEven)
        .unwrap()
        .with_backend(backend)
}

pub fn resources(backend: Backend, layout: Layout, lanes: usize) -> Result<Resources, Error> {
    if !(1..=MAX_LANES).contains(&lanes) {
        return Err(Error::Resource);
    }
    let input = descriptor(backend, 7).resources(layout, 2 * lanes)?;
    let activation = descriptor(backend, 9).resources(layout, lanes)?;
    let output = descriptor(backend, 7).resources(layout, lanes)?;
    let party_key_payload_bytes = input.party_key_payload_bytes
        + activation.party_key_payload_bytes
        + output.party_key_payload_bytes
        + 24 * lanes;
    let total_allocation_estimate_bytes = input.total_allocation_estimate_bytes
        + activation.total_allocation_estimate_bytes
        + output.total_allocation_estimate_bytes
        + 8192 * lanes
        + 65536;
    if total_allocation_estimate_bytes > MAX_ALLOCATION_ESTIMATE {
        return Err(Error::Resource);
    }
    Ok(Resources {
        party_key_payload_bytes,
        total_allocation_estimate_bytes,
        peer_frame_bytes: input.peer_frame_bytes
            + activation.peer_frame_bytes
            + output.peer_frame_bytes
            + 4 * (mul::HEADER_BYTES + 8 * lanes),
        rounds: input.rounds + activation.rounds + output.rounds + 2,
    })
}

struct Material {
    input: rs::Party,
    square: mul::Party,
    activation: rs::Party,
    product: mul::Party,
    output: rs::Party,
    gate: Zeroizing<Vec<u32>>,
    up: Zeroizing<Vec<u32>>,
    step: u8,
}

pub struct Party {
    lanes: usize,
    material: Option<Material>,
}
pub enum Progress {
    Open(Vec<u8>),
    Complete(Zeroizing<Vec<u32>>),
}

fn take_rescaled(mut values: Vec<rs::OutputShare>) -> Zeroizing<Vec<u32>> {
    let output = Zeroizing::new(values.iter().map(|v| v.rescaled).collect());
    for value in &mut values {
        value.rescaled.zeroize();
        value.nonnegative.zeroize();
    }
    output
}

pub fn issue_reference(
    backend: Backend,
    layout: Layout,
    context: Context,
    lanes: usize,
) -> Result<[Party; 2], Error> {
    resources(backend, layout, lanes)?;
    context
        .first_tensor_index
        .checked_add((2 * lanes) as u64)
        .ok_or(Error::Domain)?;
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let child = |step| {
        let mut hash = Sha256::new();
        hash.update(b"pllm.shared_gated_block_reference.q14_q7.v1\0");
        hash.update(context.operation);
        hash.update(nonce);
        hash.update([step]);
        hash.update([backend as u8, layout as u8]);
        Context {
            operation: hash.finalize().into(),
            ..context
        }
    };
    let [ai, bi] = descriptor(backend, 7).issue_reference(layout, child(0), 2 * lanes)?;
    let [asq, bsq] = mul::issue_reference(BITS, lanes, child(1))?;
    let [aa, ba] = descriptor(backend, 9).issue_reference(layout, child(2), lanes)?;
    let [am, bm] = mul::issue_reference(BITS, lanes, child(3))?;
    let [ao, bo] = descriptor(backend, 7).issue_reference(layout, child(4), lanes)?;
    Ok([(ai, asq, aa, am, ao), (bi, bsq, ba, bm, bo)].map(
        |(input, square, activation, product, output)| Party {
            lanes,
            material: Some(Material {
                input,
                square,
                activation,
                product,
                output,
                gate: Zeroizing::new(Vec::new()),
                up: Zeroizing::new(Vec::new()),
                step: 0,
            }),
        },
    ))
}

impl Party {
    pub fn cancel(&mut self) {
        self.material = None;
    }
    pub fn key_payload_bytes(&self) -> usize {
        self.material.as_ref().map_or(0, |m| {
            m.input.key_payload_bytes()
                + m.square.key_payload_bytes()
                + m.activation.key_payload_bytes()
                + m.product.key_payload_bytes()
                + m.output.key_payload_bytes()
        })
    }
    pub fn start(&mut self, gate: &[u32], up: &[u32]) -> Result<Vec<u8>, Error> {
        let mut m = self.material.take().ok_or(Error::Consumed)?;
        if m.step != 0 || gate.len() != self.lanes || up.len() != self.lanes {
            return Err(Error::Domain);
        }
        let input = Zeroizing::new(gate.iter().chain(up).copied().collect::<Vec<_>>());
        let frame = m.input.start(&input)?;
        m.step = 1;
        self.material = Some(m);
        Ok(frame)
    }
    pub fn advance(&mut self, peer: &[u8]) -> Result<Progress, Error> {
        // Every child and future triple is retired on any failed frame.
        let mut m = self.material.take().ok_or(Error::Consumed)?;
        let frame = match m.step {
            1 => match m.input.advance(peer)? {
                rs::Progress::Open(frame) => frame,
                rs::Progress::Complete(values) => {
                    let values = take_rescaled(values);
                    m.gate.extend_from_slice(&values[..self.lanes]);
                    m.up.extend_from_slice(&values[self.lanes..]);
                    m.step = 2;
                    m.square.start(&m.gate, &m.gate)?
                }
            },
            2 => {
                let square = m.square.finish(peer)?;
                let numerator = Zeroizing::new(
                    square
                        .iter()
                        .zip(m.gate.iter())
                        .map(|(&s, &g)| s.wrapping_add(g.wrapping_mul(256)) & MASK)
                        .collect::<Vec<_>>(),
                );
                m.gate.zeroize();
                m.step = 3;
                m.activation.start(&numerator)?
            }
            3 => match m.activation.advance(peer)? {
                rs::Progress::Open(frame) => frame,
                rs::Progress::Complete(values) => {
                    let activated = take_rescaled(values);
                    m.step = 4;
                    m.product.start(&activated, &m.up)?
                }
            },
            4 => {
                let product = m.product.finish(peer)?;
                m.step = 5;
                m.output.start(&product)?
            }
            5 => match m.output.advance(peer)? {
                rs::Progress::Open(frame) => frame,
                rs::Progress::Complete(values) => {
                    return Ok(Progress::Complete(take_rescaled(values)))
                }
            },
            _ => return Err(Error::Context),
        };
        self.material = Some(m);
        Ok(Progress::Open(frame))
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
    #[test]
    fn full_block_matches_core_with_negative_ties_and_payload_conservation() {
        let inputs = [
            -16384_i32, -16320, -8256, -192, -64, 0, 64, 192, 8256, 16320, 16384,
        ];
        for backend in [Backend::CompactDcf, Backend::IntervalDcf] {
            for layout in [Layout::Fused, Layout::Unfused] {
                let cost = resources(backend, layout, inputs.len()).unwrap();
                let [mut a, mut b] =
                    issue_reference(backend, layout, context(), inputs.len()).unwrap();
                assert_eq!(a.key_payload_bytes(), cost.party_key_payload_bytes);
                let ag = vec![MASK; inputs.len()];
                let au = vec![12345; inputs.len()];
                let bg: Vec<_> = inputs
                    .iter()
                    .map(|&g| (g as u32).wrapping_sub(MASK) & MASK)
                    .collect();
                let bu: Vec<_> = inputs
                    .iter()
                    .rev()
                    .map(|&u| (u as u32).wrapping_sub(12345) & MASK)
                    .collect();
                let mut frames = [a.start(&ag, &au).unwrap(), b.start(&bg, &bu).unwrap()];
                let mut bytes = 0;
                for round in 0..cost.rounds {
                    bytes += frames.iter().map(Vec::len).sum::<usize>();
                    match (
                        a.advance(&frames[1]).unwrap(),
                        b.advance(&frames[0]).unwrap(),
                    ) {
                        (Progress::Open(af), Progress::Open(bf)) => frames = [af, bf],
                        (Progress::Complete(aa), Progress::Complete(bb)) => {
                            assert_eq!(round + 1, cost.rounds);
                            for (i, (&g, &u)) in inputs.iter().zip(inputs.iter().rev()).enumerate()
                            {
                                let expected = pllm_core::gated_multiply_q7(
                                    pllm_core::rescale_q14_to_q7(g).unwrap(),
                                    pllm_core::rescale_q14_to_q7(u).unwrap(),
                                )
                                .unwrap();
                                assert_eq!(
                                    aa[i].wrapping_add(bb[i]) & MASK,
                                    expected as u32 & MASK
                                );
                            }
                        }
                        _ => panic!("phase mismatch"),
                    }
                }
                assert_eq!(bytes, cost.peer_frame_bytes);
                assert_eq!(a.key_payload_bytes(), 0);
                assert!(a.advance(&frames[0]).is_err());
            }
        }
    }
    #[test]
    fn all_future_material_burns_on_each_failed_step() {
        for layout in [Layout::Fused, Layout::Unfused] {
            let rounds = resources(Backend::IntervalDcf, layout, 1).unwrap().rounds;
            for fail_at in 0..rounds {
                let [mut a, mut b] =
                    issue_reference(Backend::IntervalDcf, layout, context(), 1).unwrap();
                let mut frames = [a.start(&[0], &[0]).unwrap(), b.start(&[0], &[0]).unwrap()];
                for _ in 0..fail_at {
                    let (Progress::Open(af), Progress::Open(bf)) = (
                        a.advance(&frames[1]).unwrap(),
                        b.advance(&frames[0]).unwrap(),
                    ) else {
                        panic!("early completion")
                    };
                    frames = [af, bf];
                }
                frames[1][8] ^= 1;
                assert!(a.advance(&frames[1]).is_err());
                assert_eq!(a.key_payload_bytes(), 0);
                b.cancel();
                assert_eq!(b.key_payload_bytes(), 0);
            }
        }
        let [mut a, _] =
            issue_reference(Backend::IntervalDcf, Layout::Fused, context(), 1).unwrap();
        assert!(a.start(&[], &[0]).is_err());
        assert_eq!(a.key_payload_bytes(), 0);
        assert!(resources(Backend::IntervalDcf, Layout::Fused, MAX_LANES + 1).is_err());
    }
}
