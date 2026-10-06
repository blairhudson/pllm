//! Complete shared affine-spline SiLU x up reference. Inputs are signed Q16,
//! caller-admitted in the public profile's domain. The local test dealer is
//! not distributed issuance, and this module supplies no secret range proof.
use crate::{
    shared_affine_lookup_reference as lookup, shared_arithmetic_reference as mul,
    shared_rescale_reference::{self as rs, Backend, Context, Descriptor, Error, Layout, Rounding},
};
use pllm_core::piecewise_gated_reference::{Profile, COEFFICIENT_FRACTION, INPUT_FRACTION};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};

pub const MAX_LANES: usize = 256;
pub const MAX_ALLOCATION_ESTIMATE: usize = 64 << 20;
#[derive(Debug)]
pub struct Resources {
    pub party_key_payload_bytes: usize,
    pub total_allocation_estimate_bytes: usize,
    pub peer_frame_bytes: usize,
    pub rounds: usize,
    pub lookup_key_payload_bytes: usize,
    pub rescale_key_payload_bytes: usize,
}
fn descriptor(shift: u8) -> Descriptor {
    Descriptor::new(32, shift, Rounding::TiesEven)
        .unwrap()
        .with_backend(Backend::IntervalDcf)
}
pub fn resources(
    profile: &Profile,
    layout: lookup::Layout,
    lanes: usize,
) -> Result<Resources, Error> {
    if !(1..=MAX_LANES).contains(&lanes) {
        return Err(Error::Resource);
    }
    let input =
        descriptor(INPUT_FRACTION - profile.fraction()).resources(Layout::Fused, 2 * lanes)?;
    let activation = descriptor(COEFFICIENT_FRACTION).resources(Layout::Fused, lanes)?;
    let output = descriptor(profile.fraction()).resources(Layout::Fused, lanes)?;
    let lookup = lookup::resources(layout, lanes)?;
    let total_allocation_estimate_bytes = input.total_allocation_estimate_bytes
        + activation.total_allocation_estimate_bytes
        + output.total_allocation_estimate_bytes
        + lookup.total_allocation_estimate_bytes
        + 8192 * lanes
        + 65536;
    if total_allocation_estimate_bytes > MAX_ALLOCATION_ESTIMATE {
        return Err(Error::Resource);
    }
    let rescale_key_payload_bytes = input.party_key_payload_bytes
        + activation.party_key_payload_bytes
        + output.party_key_payload_bytes;
    Ok(Resources {
        party_key_payload_bytes: rescale_key_payload_bytes
            + lookup.party_key_payload_bytes
            + 24 * lanes,
        total_allocation_estimate_bytes,
        rounds: 6,
        peer_frame_bytes: input.peer_frame_bytes
            + activation.peer_frame_bytes
            + output.peer_frame_bytes
            + lookup.peer_frame_bytes
            + 4 * (mul::HEADER_BYTES + 8 * lanes),
        lookup_key_payload_bytes: lookup.party_key_payload_bytes,
        rescale_key_payload_bytes,
    })
}
struct Material {
    input: rs::Party,
    lookup: lookup::Party,
    linear: mul::Party,
    activation: rs::Party,
    product: mul::Party,
    output: rs::Party,
    gate: Zeroizing<Vec<u32>>,
    up: Zeroizing<Vec<u32>>,
    intercept: Zeroizing<Vec<u32>>,
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
    profile: Arc<Profile>,
    layout: lookup::Layout,
    context: Context,
    lanes: usize,
) -> Result<[Party; 2], Error> {
    resources(&profile, layout, lanes)?;
    context
        .first_tensor_index
        .checked_add((2 * lanes) as u64)
        .ok_or(Error::Domain)?;
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let child = |step| {
        let mut h = Sha256::new();
        h.update(b"pllm.shared_piecewise_gated.reference.v1\0");
        h.update(context.operation);
        h.update(profile.digest());
        h.update(nonce);
        h.update([layout as u8, step]);
        Context {
            operation: h.finalize().into(),
            ..context
        }
    };
    let [ai, bi] = descriptor(INPUT_FRACTION - profile.fraction()).issue_reference(
        Layout::Fused,
        child(0),
        2 * lanes,
    )?;
    let [ak, bk] = lookup::issue_reference(profile.clone(), layout, child(1), lanes)?;
    let [al, bl] = mul::issue_reference(32, lanes, child(2))?;
    let [aa, ba] =
        descriptor(COEFFICIENT_FRACTION).issue_reference(Layout::Fused, child(3), lanes)?;
    let [am, bm] = mul::issue_reference(32, lanes, child(4))?;
    let [ao, bo] =
        descriptor(profile.fraction()).issue_reference(Layout::Fused, child(5), lanes)?;
    Ok([(ai, ak, al, aa, am, ao), (bi, bk, bl, ba, bm, bo)].map(
        |(input, lookup, linear, activation, product, output)| Party {
            lanes,
            material: Some(Material {
                input,
                lookup,
                linear,
                activation,
                product,
                output,
                gate: Zeroizing::new(Vec::new()),
                up: Zeroizing::new(Vec::new()),
                intercept: Zeroizing::new(Vec::new()),
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
                + m.lookup.key_payload_bytes()
                + m.linear.key_payload_bytes()
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
        // Taking the entire owner burns every future key/triple on any error.
        let mut m = self.material.take().ok_or(Error::Consumed)?;
        let frame = match m.step {
            1 => {
                let rs::Progress::Complete(values) = m.input.advance(peer)? else {
                    return Err(Error::Context);
                };
                let values = take_rescaled(values);
                m.gate.extend_from_slice(&values[..self.lanes]);
                m.up.extend_from_slice(&values[self.lanes..]);
                m.step = 2;
                m.lookup.start(&m.gate)?
            }
            2 => {
                let coefficients = m.lookup.finish(peer)?;
                let slopes = Zeroizing::new(coefficients.iter().map(|v| v[0]).collect::<Vec<_>>());
                m.intercept.extend(coefficients.iter().map(|v| v[1]));
                m.step = 3;
                m.linear.start(&slopes, &m.gate)?
            }
            3 => {
                let product = m.linear.finish(peer)?;
                let affine = Zeroizing::new(
                    product
                        .iter()
                        .zip(m.intercept.iter())
                        .map(|(&p, &b)| p.wrapping_add(b))
                        .collect::<Vec<_>>(),
                );
                m.gate.zeroize();
                m.intercept.zeroize();
                m.step = 4;
                m.activation.start(&affine)?
            }
            4 => {
                let rs::Progress::Complete(values) = m.activation.advance(peer)? else {
                    return Err(Error::Context);
                };
                m.step = 5;
                m.product.start(&take_rescaled(values), &m.up)?
            }
            5 => {
                let product = m.product.finish(peer)?;
                m.up.zeroize();
                m.step = 6;
                m.output.start(&product)?
            }
            6 => {
                let rs::Progress::Complete(values) = m.output.advance(peer)? else {
                    return Err(Error::Context);
                };
                return Ok(Progress::Complete(take_rescaled(values)));
            }
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
    fn complete_block_boundary_ties_tails_and_costs() {
        let input = [
            -48 * 65536,
            -8 * 65536 - 1,
            -8 * 65536,
            -65536,
            -192,
            -64,
            0,
            64,
            192,
            65536,
            8 * 65536,
            48 * 65536,
        ];
        for fraction in [8, 9] {
            for pieces in [16, 64] {
                for layout in [lookup::Layout::Separate, lookup::Layout::Vector] {
                    let p = Arc::new(Profile::new(fraction, pieces).unwrap());
                    let cost = resources(&p, layout, input.len()).unwrap();
                    let [mut a, mut b] =
                        issue_reference(p.clone(), layout, context(), input.len()).unwrap();
                    assert_eq!(a.key_payload_bytes(), cost.party_key_payload_bytes);
                    let mut frames = [
                        a.start(&vec![u32::MAX; input.len()], &vec![12345; input.len()])
                            .unwrap(),
                        b.start(
                            &input
                                .iter()
                                .map(|&g| (g as u32).wrapping_add(1))
                                .collect::<Vec<_>>(),
                            &input
                                .iter()
                                .rev()
                                .map(|&u| (u as u32).wrapping_sub(12345))
                                .collect::<Vec<_>>(),
                        )
                        .unwrap(),
                    ];
                    let mut bytes = 0;
                    for round in 0..cost.rounds {
                        bytes += frames.iter().map(Vec::len).sum::<usize>();
                        match (
                            a.advance(&frames[1]).unwrap(),
                            b.advance(&frames[0]).unwrap(),
                        ) {
                            (Progress::Open(aa), Progress::Open(bb)) => frames = [aa, bb],
                            (Progress::Complete(aa), Progress::Complete(bb)) => {
                                assert_eq!(round + 1, cost.rounds);
                                for (i, (&g, &u)) in
                                    input.iter().zip(input.iter().rev()).enumerate()
                                {
                                    assert_eq!(
                                        aa[i].wrapping_add(bb[i]) as i32,
                                        p.evaluate(g, u).unwrap()
                                    );
                                }
                            }
                            _ => panic!("phase mismatch"),
                        }
                    }
                    assert_eq!(bytes, cost.peer_frame_bytes);
                    assert_eq!(a.key_payload_bytes(), 0);
                    assert!(a.start(&[0], &[0]).is_err());
                }
            }
        }
    }
    #[test]
    fn every_failed_round_retires_all_future_material() {
        for layout in [lookup::Layout::Separate, lookup::Layout::Vector] {
            for fail_at in 0..6 {
                let p = Arc::new(Profile::new(9, 16).unwrap());
                let [mut a, mut b] = issue_reference(p, layout, context(), 1).unwrap();
                let mut frames = [a.start(&[0], &[0]).unwrap(), b.start(&[0], &[0]).unwrap()];
                for _ in 0..fail_at {
                    let (Progress::Open(aa), Progress::Open(bb)) = (
                        a.advance(&frames[1]).unwrap(),
                        b.advance(&frames[0]).unwrap(),
                    ) else {
                        panic!("early complete");
                    };
                    frames = [aa, bb];
                }
                frames[1][8] ^= 1;
                assert!(a.advance(&frames[1]).is_err());
                assert_eq!(a.key_payload_bytes(), 0);
                b.cancel();
                assert_eq!(b.key_payload_bytes(), 0);
                assert!(b.advance(&frames[0]).is_err());
            }
        }
        let p = Arc::new(Profile::new(9, 64).unwrap());
        assert!(resources(&p, lookup::Layout::Vector, MAX_LANES + 1).is_err());
        let [mut a, _] = issue_reference(p, lookup::Layout::Vector, context(), 1).unwrap();
        assert!(a.start(&[], &[0]).is_err());
        assert_eq!(a.key_payload_bytes(), 0);
    }
}
