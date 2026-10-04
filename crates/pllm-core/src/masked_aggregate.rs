//! Bounded one-use output relay reference. Worker A never receives B's mask seed.
//! This codec is not an authenticated transport or an admitted role topology.
use crate::offset_transport::{pack_row_residues, reconstruct_rows, seeded_share};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

const MAGIC: &[u8; 8] = b"PLLMRLY1";
pub const HEADER_BYTES: usize = 41;

#[derive(Clone)]
struct Layout {
    context: [u8; 32],
    widths: Vec<u8>,
    rows: usize,
    bytes: usize,
}

impl Layout {
    fn frame(&self, role: u8, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(HEADER_BYTES + payload.len());
        frame.extend_from_slice(MAGIC);
        frame.push(role);
        frame.extend_from_slice(&self.context);
        frame.extend_from_slice(payload);
        frame
    }

    fn body<'a>(&self, role: u8, frame: &'a [u8]) -> Result<&'a [u8], String> {
        if frame.len() != HEADER_BYTES + self.bytes
            || &frame[..8] != MAGIC
            || frame[8] != role
            || frame[9..HEADER_BYTES] != self.context
        {
            return Err("aggregation frame/context/role differs".into());
        }
        Ok(&frame[HEADER_BYTES..])
    }
}

/// Separate non-cloneable objects for the mask worker, relay, and trusted client.
pub struct OutputMask {
    layout: Layout,
    seed: Zeroizing<[u8; 32]>,
}
pub struct OutputRelay {
    layout: Layout,
}
pub struct OutputReceiver {
    layout: Layout,
    seed: Zeroizing<[u8; 32]>,
}

pub fn issue(
    context: &[u8; 32],
    widths: &[u8],
    rows: usize,
) -> Result<(OutputMask, OutputRelay, OutputReceiver), String> {
    if widths.is_empty()
        || !(1..=4096).contains(&rows)
        || widths.len() > 4_000_000 / rows
        || widths.iter().any(|&w| !(1..=32).contains(&w))
    {
        return Err("invalid aggregation layout".into());
    }
    let mut seed = Zeroizing::new([0; 32]);
    let mut nonce = [0; 32];
    getrandom::fill(seed.as_mut()).map_err(|e| e.to_string())?;
    getrandom::fill(&mut nonce).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    hash.update(b"pllm/output-relay/v1\0");
    hash.update(context);
    hash.update(nonce);
    hash.update((rows as u64).to_le_bytes());
    hash.update(widths);
    let layout = Layout {
        context: hash.finalize().into(),
        widths: widths.to_vec(),
        rows,
        bytes: (widths.iter().map(|&w| usize::from(w)).sum::<usize>() * rows).div_ceil(8),
    };
    Ok((
        OutputMask {
            layout: layout.clone(),
            seed: Zeroizing::new(*seed),
        },
        OutputRelay {
            layout: layout.clone(),
        },
        OutputReceiver { layout, seed },
    ))
}

impl OutputMask {
    pub fn mask(self, value: &[u8]) -> Result<Vec<u8>, String> {
        let n = self.layout.rows * self.layout.widths.len();
        if value.len() != n * 4 {
            return Err("worker output size differs".into());
        }
        let pad = Zeroizing::new(seeded_share(&self.seed, &self.layout.context, n, 32, None)?);
        let raw: Vec<u8> = value
            .chunks_exact(4)
            .zip(pad.iter())
            .flat_map(|(v, pad)| {
                u32::from_le_bytes(v.try_into().unwrap())
                    .wrapping_add(*pad)
                    .to_le_bytes()
            })
            .collect();
        Ok(self.layout.frame(
            1,
            &pack_row_residues(&raw, &self.layout.widths, self.layout.rows)?,
        ))
    }
}

impl OutputRelay {
    pub fn combine(self, own_value: &[u8], masked_peer: &[u8]) -> Result<Vec<u8>, String> {
        let peer = self.layout.body(1, masked_peer)?;
        let own = pack_row_residues(own_value, &self.layout.widths, self.layout.rows)?;
        let combined = reconstruct_rows(&own, peer, &self.layout.widths, self.layout.rows)?;
        let raw: Vec<u8> = combined
            .chunks_exact(8)
            .flat_map(|v| (i64::from_le_bytes(v.try_into().unwrap()) as u32).to_le_bytes())
            .collect();
        Ok(self.layout.frame(
            2,
            &pack_row_residues(&raw, &self.layout.widths, self.layout.rows)?,
        ))
    }
}

impl OutputReceiver {
    pub fn finish(self, frame: &[u8]) -> Result<Vec<u8>, String> {
        let payload = self.layout.body(2, frame)?;
        let pad = Zeroizing::new(seeded_share(
            &self.seed,
            &self.layout.context,
            self.layout.rows * self.layout.widths.len(),
            32,
            None,
        )?);
        let raw = Zeroizing::new(
            pad.iter()
                .flat_map(|v| 0_u32.wrapping_sub(*v).to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let packed = Zeroizing::new(pack_row_residues(
            &raw,
            &self.layout.widths,
            self.layout.rows,
        )?);
        reconstruct_rows(payload, &packed, &self.layout.widths, self.layout.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_is_exact_and_rejects_cross_issuance() {
        let widths = [9, 17, 24, 32];
        let a: Vec<u8> = [u32::MAX, 501, 31337, 12345]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let b: Vec<u8> = [6_u32, 0_u32.wrapping_sub(507), 9, 23]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let (worker, relay, client) = issue(&[9; 32], &widths, 1).unwrap();
        let peer = worker.mask(&b).unwrap();
        let merged = relay.combine(&a, &peer).unwrap();
        let expected = reconstruct_rows(
            &pack_row_residues(&a, &widths, 1).unwrap(),
            &pack_row_residues(&b, &widths, 1).unwrap(),
            &widths,
            1,
        )
        .unwrap();
        assert_eq!(client.finish(&merged).unwrap(), expected);
        let (_, relay, client) = issue(&[9; 32], &widths, 1).unwrap();
        assert!(relay.combine(&a, &peer).is_err());
        assert!(client.finish(&merged).is_err());
        assert!(issue(&[0; 32], &[33], 1).is_err());
    }

    #[test]
    fn ideal_small_ring_view_is_independent_of_the_input() {
        let mut expected = None;
        for x in 0..16_u32 {
            let mut view = [0_u32; 256];
            for a in 0..16_u32 {
                let b = x.wrapping_sub(a) & 15;
                for pad in 0..16_u32 {
                    view[(a * 16 + ((5 * b + pad) & 15)) as usize] += 1;
                }
            }
            if let Some(expected) = expected {
                assert_eq!(view, expected);
            }
            expected = Some(view);
        }
    }
}
