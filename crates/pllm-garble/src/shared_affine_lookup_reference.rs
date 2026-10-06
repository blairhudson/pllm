//! One-use coefficient selection for a public affine spline. FuseFSS Appendix
//! I.2/N's vector lookup contract, instantiated with the mixed-mode paper's
//! universal public-boundary DCF. No translated boundary or mask is public.
//! This is an independently implemented, unreviewed in-process reference.
use crate::{
    compact_dcf, interval_fss,
    shared_rescale_reference::{Context, Error},
};
use pllm_core::piecewise_gated_reference::Profile;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use zeroize::Zeroizing;

pub const HEADER_BYTES: usize = 78;
pub const MAX_LANES: usize = 256;
const MAGIC: &[u8; 8] = b"PLLMAF01";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Separate,
    Vector,
}
impl Layout {
    fn groups(self) -> usize {
        if self == Self::Vector {
            1
        } else {
            2
        }
    }
}

#[derive(Debug)]
pub struct Resources {
    pub party_key_payload_bytes: usize,
    pub total_allocation_estimate_bytes: usize,
    pub peer_frame_bytes: usize,
}
pub fn resources(layout: Layout, lanes: usize) -> Result<Resources, Error> {
    if !(1..=MAX_LANES).contains(&lanes) {
        return Err(Error::Resource);
    }
    let party_key_payload_bytes =
        lanes * (layout.groups() * (4 + compact_dcf::Comparison::bytes(32)) + 8);
    Ok(Resources {
        party_key_payload_bytes,
        total_allocation_estimate_bytes: 8 * party_key_payload_bytes + 8192 * lanes + 65536,
        peer_frame_bytes: 2 * (HEADER_BYTES + 4 * lanes * layout.groups()),
    })
}
struct Key {
    mask: Zeroizing<u32>,
    constant: Zeroizing<Vec<u32>>,
    comparison: interval_fss::Comparison,
}
enum State {
    Ready(Vec<Key>),
    Waiting(Vec<Key>, Zeroizing<Vec<u32>>),
}
pub struct Party {
    id: u8,
    layout: Layout,
    lanes: usize,
    profile: Arc<Profile>,
    binding: [u8; 32],
    nonce: [u8; 32],
    state: Option<State>,
}
fn random_word() -> Result<u32, Error> {
    let mut bytes = Zeroizing::new([0; 4]);
    getrandom::fill(&mut *bytes).map_err(|_| Error::Randomness)?;
    Ok(u32::from_le_bytes(*bytes))
}
// Dealer-only constant making eval(p)-eval(0) a share of [x < p].
fn correction(mask: u32, p: u32) -> u32 {
    interval_fss::correction(32, mask, p)
        .wrapping_sub(interval_fss::correction(32, mask, 0))
        .wrapping_add(u32::from(u64::from(mask) + u64::from(p) >= 1u64 << 32))
}
fn issue_keys(profile: &Profile, layout: Layout, lanes: usize) -> Result<[Vec<Key>; 2], Error> {
    let mut keys = [Vec::new(), Vec::new()];
    for _ in 0..lanes {
        for group in 0..layout.groups() {
            let mask = Zeroizing::new(random_word()?);
            let ma = random_word()?;
            let indices: &[usize] = if layout == Layout::Vector {
                &[0, 1]
            } else if group == 0 {
                &[0]
            } else {
                &[1]
            };
            let mut constants = [Zeroizing::new(Vec::new()), Zeroizing::new(Vec::new())];
            for &index in indices {
                let intervals = profile.intervals();
                let mut value = Zeroizing::new(intervals.last().unwrap().1[index]);
                for pair in intervals.windows(2) {
                    let delta = pair[0].1[index].wrapping_sub(pair[1].1[index]);
                    *value = value.wrapping_add(delta.wrapping_mul(correction(*mask, pair[1].0)));
                }
                let a = random_word()?;
                constants[0].push(a);
                constants[1].push(value.wrapping_sub(a));
            }
            let [a, b] = interval_fss::Comparison::issue(32, *mask)?;
            let [ca, cb] = constants;
            keys[0].push(Key {
                mask: Zeroizing::new(ma),
                constant: ca,
                comparison: a,
            });
            keys[1].push(Key {
                mask: Zeroizing::new(mask.wrapping_sub(ma)),
                constant: cb,
                comparison: b,
            });
        }
    }
    Ok(keys)
}
pub fn issue_reference(
    profile: Arc<Profile>,
    layout: Layout,
    context: Context,
    lanes: usize,
) -> Result<[Party; 2], Error> {
    resources(layout, lanes)?;
    context
        .first_tensor_index
        .checked_add(lanes as u64)
        .ok_or(Error::Domain)?;
    let mut hash = Sha256::new();
    hash.update(b"pllm.shared_affine_lookup.reference.v1\0");
    hash.update(profile.digest());
    hash.update([layout as u8]);
    hash.update((lanes as u64).to_le_bytes());
    hash.update(context.plan);
    hash.update(context.session);
    hash.update(context.operation);
    hash.update(context.first_tensor_index.to_le_bytes());
    let binding = hash.finalize().into();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let [a, b] = issue_keys(&profile, layout, lanes)?;
    Ok([(0, a), (1, b)].map(|(id, keys)| Party {
        id,
        layout,
        lanes,
        profile: profile.clone(),
        binding,
        nonce,
        state: Some(State::Ready(keys)),
    }))
}
impl Party {
    pub fn cancel(&mut self) {
        self.state = None;
    }
    pub fn key_payload_bytes(&self) -> usize {
        if self.state.is_some() {
            resources(self.layout, self.lanes)
                .unwrap()
                .party_key_payload_bytes
        } else {
            0
        }
    }
    pub fn start(&mut self, values: &[u32]) -> Result<Vec<u8>, Error> {
        let State::Ready(keys) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if values.len() != self.lanes {
            return Err(Error::Domain);
        }
        let own = Zeroizing::new(
            keys.iter()
                .enumerate()
                .map(|(i, k)| values[i / self.layout.groups()].wrapping_add(*k.mask))
                .collect::<Vec<_>>(),
        );
        let mut frame = Vec::with_capacity(HEADER_BYTES + 4 * own.len());
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&self.binding);
        frame.extend_from_slice(&self.nonce);
        frame.extend_from_slice(&[self.layout as u8, self.id]);
        frame.extend_from_slice(&(self.lanes as u32).to_le_bytes());
        for value in own.iter() {
            frame.extend_from_slice(&value.to_le_bytes());
        }
        self.state = Some(State::Waiting(keys, own));
        Ok(frame)
    }
    pub fn finish(&mut self, peer: &[u8]) -> Result<Zeroizing<Vec<[u32; 2]>>, Error> {
        let State::Waiting(keys, own) = self.state.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if peer.len() != HEADER_BYTES + 4 * own.len()
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != self.nonce
            || peer[72] != self.layout as u8
            || peer[73] != 1 - self.id
            || u32::from_le_bytes(peer[74..78].try_into().unwrap()) as usize != self.lanes
        {
            return Err(Error::Context);
        }
        let mut result = Zeroizing::new(vec![[0; 2]; self.lanes]);
        for (i, ((key, own), raw)) in keys
            .iter()
            .zip(own.iter())
            .zip(peer[HEADER_BYTES..].chunks_exact(4))
            .enumerate()
        {
            let opened = own.wrapping_add(u32::from_le_bytes(raw.try_into().unwrap()));
            let base = key.comparison.eval(self.id, opened, 0);
            let mut output = Zeroizing::new(key.constant.to_vec());
            for pair in self.profile.intervals().windows(2) {
                let indicator = key
                    .comparison
                    .eval(self.id, opened, pair[1].0)
                    .wrapping_sub(base);
                for (j, value) in output.iter_mut().enumerate() {
                    let component = if self.layout == Layout::Vector {
                        j
                    } else {
                        i % 2
                    };
                    let delta = pair[0].1[component].wrapping_sub(pair[1].1[component]);
                    *value = value.wrapping_add(delta.wrapping_mul(indicator));
                }
            }
            for (j, value) in output.iter().enumerate() {
                result[i / self.layout.groups()][if self.layout == Layout::Vector {
                    j
                } else {
                    i % 2
                }] = *value;
            }
        }
        Ok(result)
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
    fn coefficient_selection_at_every_boundary_and_ring_endpoint() {
        for fraction in [8, 9] {
            for pieces in [16, 64] {
                let p = Arc::new(Profile::new(fraction, pieces).unwrap());
                let mut inputs = vec![0, u32::MAX, 1 << 31, (1 << 31) - 1];
                for &(start, _) in p.intervals() {
                    inputs.extend([start.wrapping_sub(1), start, start.wrapping_add(1)]);
                }
                for layout in [Layout::Separate, Layout::Vector] {
                    let [mut a, mut b] =
                        issue_reference(p.clone(), layout, context(), inputs.len()).unwrap();
                    let x = a.start(&vec![u32::MAX; inputs.len()]).unwrap();
                    let y = b
                        .start(&inputs.iter().map(|v| v.wrapping_add(1)).collect::<Vec<_>>())
                        .unwrap();
                    let aa = a.finish(&y).unwrap();
                    let bb = b.finish(&x).unwrap();
                    for ((&input, a), b) in inputs.iter().zip(aa.iter()).zip(bb.iter()) {
                        assert_eq!(
                            [a[0].wrapping_add(b[0]), a[1].wrapping_add(b[1])],
                            p.coefficients(input)
                        );
                    }
                    assert_eq!(a.key_payload_bytes(), 0);
                    assert_eq!(a.finish(&y), Err(Error::Consumed));
                }
            }
        }
    }
    #[test]
    fn malformed_cross_profile_layout_and_cancel_burn() {
        let p = Arc::new(Profile::new(9, 64).unwrap());
        for offset in [0, 8, 40, 72, 73, 74] {
            let [mut a, mut b] = issue_reference(p.clone(), Layout::Vector, context(), 1).unwrap();
            a.start(&[0]).unwrap();
            let mut frame = b.start(&[0]).unwrap();
            frame[offset] ^= 1;
            assert!(a.finish(&frame).is_err());
            assert_eq!(a.key_payload_bytes(), 0);
        }
        let [mut a, _] = issue_reference(p, Layout::Vector, context(), 1).unwrap();
        assert!(a.start(&[]).is_err());
        assert_eq!(a.key_payload_bytes(), 0);
        a.cancel();
        assert!(a.start(&[0]).is_err());
        assert!(resources(Layout::Vector, MAX_LANES + 1).is_err());
    }
}
