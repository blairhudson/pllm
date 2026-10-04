//! Test-local three-party degree reduction, with an optional public projection
//! before re-sharing. Each Party consumes only its own product share. No decoder
//! numeric/transport contract or production topology is provided by this oracle.
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

pub const MODULUS: u64 = 2_013_265_921; // BabyBear, degree-two interpolation at 1,2,3.
const HEADER: usize = 46;
const MAGIC: &[u8; 8] = b"PLLMDR01";

struct Uniform {
    bytes: Zeroizing<Vec<u8>>,
    at: usize,
}
impl Uniform {
    fn new() -> Self {
        Self {
            bytes: Zeroizing::new(vec![0; 4096]),
            at: 4096,
        }
    }
    fn field(&mut self) -> Result<u32, String> {
        loop {
            if self.at == self.bytes.len() {
                getrandom::fill(&mut self.bytes).map_err(|e| e.to_string())?;
                self.at = 0;
            }
            let value = u32::from_le_bytes(self.bytes[self.at..self.at + 4].try_into().unwrap());
            self.at += 4;
            if u64::from(value) < MODULUS * 2 {
                return Ok((u64::from(value) % MODULUS) as u32);
            }
        }
    }
}

fn project(weights: &[u8], input: &[u32]) -> Vec<u32> {
    weights
        .chunks_exact(input.len())
        .map(|row| {
            let sum: i64 = row
                .iter()
                .zip(input)
                .map(|(&w, &x)| i64::from(w as i8) * i64::from(x))
                .sum();
            sum.rem_euclid(MODULUS as i64) as u32
        })
        .collect()
}

struct Party {
    id: u8,
    context: [u8; 32],
    value: Zeroizing<Vec<u32>>,
}
impl Party {
    fn reshare(self, rng: &mut Uniform) -> Result<[Zeroizing<Vec<u8>>; 3], String> {
        let slopes: Vec<u32> = (0..self.value.len())
            .map(|_| rng.field())
            .collect::<Result<_, _>>()?;
        let slopes = Zeroizing::new(slopes);
        let factor = [3, MODULUS - 3, 1][self.id as usize];
        Ok(std::array::from_fn(|target| {
            let mut frame = Vec::with_capacity(HEADER + self.value.len() * 4);
            frame.extend_from_slice(MAGIC);
            frame.extend_from_slice(&self.context);
            frame.extend_from_slice(&[self.id, target as u8]);
            frame.extend_from_slice(&(self.value.len() as u32).to_le_bytes());
            for (&value, &slope) in self.value.iter().zip(slopes.iter()) {
                let share =
                    (factor * u64::from(value) + (target as u64 + 1) * u64::from(slope)) % MODULUS;
                frame.extend_from_slice(&(share as u32).to_le_bytes());
            }
            Zeroizing::new(frame)
        }))
    }
}

fn receive(
    context: &[u8; 32],
    party: u8,
    width: usize,
    frames: [&[u8]; 3],
) -> Result<Vec<u32>, String> {
    let mut result = vec![0_u32; width];
    for (source, frame) in frames.into_iter().enumerate() {
        if frame.len() != HEADER + width * 4
            || &frame[..8] != MAGIC
            || &frame[8..40] != context
            || frame[40] != source as u8
            || frame[41] != party
            || u32::from_le_bytes(frame[42..46].try_into().unwrap()) as usize != width
        {
            return Err("degree reduction frame binding differs".into());
        }
        for (out, raw) in result.iter_mut().zip(frame[HEADER..].chunks_exact(4)) {
            let value = u32::from_le_bytes(raw.try_into().unwrap());
            if u64::from(value) >= MODULUS {
                return Err("noncanonical field share".into());
            }
            *out = ((u64::from(*out) + u64::from(value)) % MODULUS) as u32;
        }
    }
    Ok(result)
}

pub struct ReferenceResult {
    pub output: Vec<u8>,
    pub peer_bytes: usize,
    pub input_share_bytes: usize,
}

/// All-share fixture coordinator. Input distribution and reconstruction are
/// explicit costs; resident-state use is only a separately labeled projection.
pub fn reference(
    weights: &[u8],
    a: &[u8],
    b: &[u8],
    before: bool,
) -> Result<ReferenceResult, String> {
    let n = a.len() / 4;
    if !(1..=8192).contains(&n)
        || a.len() != n * 4
        || b.len() != a.len()
        || weights.is_empty()
        || weights.len() % n != 0
        || weights.len() > 32 * 1024 * 1024
        || weights.len() / n > 4096
    {
        return Err("projected re-sharing fixture exceeds bounded shape".into());
    }
    let read = |x: &[u8]| -> Result<Vec<u32>, String> {
        x.chunks_exact(4)
            .map(|v| {
                let value = u32::from_le_bytes(v.try_into().unwrap());
                if u64::from(value) < MODULUS {
                    Ok(value)
                } else {
                    Err("fixture outside field".into())
                }
            })
            .collect()
    };
    let a = Zeroizing::new(read(a)?);
    let b = Zeroizing::new(read(b)?);
    let mut rng = Uniform::new();
    let ra = Zeroizing::new((0..n).map(|_| rng.field()).collect::<Result<Vec<_>, _>>()?);
    let rb = Zeroizing::new((0..n).map(|_| rng.field()).collect::<Result<Vec<_>, _>>()?);
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    hash.update(b"pllm/projected-degree-reduction/reference/v1\0");
    hash.update(nonce);
    hash.update(weights);
    hash.update([u8::from(before)]);
    let context: [u8; 32] = hash.finalize().into();
    let width = if before { weights.len() / n } else { n };
    let mut frames = Vec::new();
    for id in 0..3_u8 {
        let point = u64::from(id) + 1;
        let products: Vec<u32> = (0..n)
            .map(|j| {
                let x = (u64::from(a[j]) + point * u64::from(ra[j])) % MODULUS;
                let y = (u64::from(b[j]) + point * u64::from(rb[j])) % MODULUS;
                (x * y % MODULUS) as u32
            })
            .collect();
        let products = Zeroizing::new(products);
        let value = if before {
            project(weights, &products)
        } else {
            products.to_vec()
        };
        frames.push(
            Party {
                id,
                context,
                value: Zeroizing::new(value),
            }
            .reshare(&mut rng)?,
        );
    }
    let mut shares = Vec::new();
    for party in 0..3_u8 {
        let received = Zeroizing::new(receive(
            &context,
            party,
            width,
            std::array::from_fn(|source| frames[source][party as usize].as_slice()),
        )?);
        shares.push(Zeroizing::new(if before {
            received.to_vec()
        } else {
            project(weights, &received)
        }));
    }
    let output: Vec<u8> = shares[0]
        .iter()
        .zip(shares[1].iter())
        .flat_map(|(&a, &b)| {
            (((2 * u64::from(a) + MODULUS - u64::from(b)) % MODULUS) as u32).to_le_bytes()
        })
        .collect();
    Ok(ReferenceResult {
        output,
        peer_bytes: 6 * (HEADER + width * 4),
        input_share_bytes: 3 * 2 * n * 4,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projected_degree_reduction_matches_independent_product_oracle() {
        let a = [7_u32, 811, MODULUS as u32 - 1, 0];
        let b = [19_u32, MODULUS as u32 - 7, 42, 3];
        let w = [1_i8, -3, 127, -128, 0, 3, 7, 1].map(|x| x as u8);
        let aa: Vec<_> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
        let bb: Vec<_> = b.iter().flat_map(|x| x.to_le_bytes()).collect();
        let expected: Vec<u8> = w
            .chunks(4)
            .flat_map(|row| {
                let sum: i128 = row
                    .iter()
                    .zip(a.iter().zip(b))
                    .map(|(&w, (&a, b))| i128::from(w as i8) * i128::from(a) * i128::from(b))
                    .sum();
                (sum.rem_euclid(i128::from(MODULUS)) as u32).to_le_bytes()
            })
            .collect();
        let base = reference(&w, &aa, &bb, false).unwrap();
        let compressed = reference(&w, &aa, &bb, true).unwrap();
        assert_eq!(base.output, expected);
        assert_eq!(compressed.output, expected);
        assert_eq!(base.peer_bytes - compressed.peer_bytes, 6 * 2 * 4);
    }
    #[test]
    fn one_party_source_share_distribution_is_input_independent() {
        for secret in 0..7 {
            for party in 1..=3 {
                let mut counts = [0; 7];
                for mask in 0..7 {
                    counts[(secret + party * mask) % 7] += 1;
                }
                assert_eq!(counts, [1; 7]);
            }
        }
    }
    #[test]
    fn frames_bind_fresh_issuance_and_both_endpoints() {
        let frames = Party {
            id: 0,
            context: [9; 32],
            value: Zeroizing::new(vec![12]),
        }
        .reshare(&mut Uniform::new())
        .unwrap();
        assert!(receive(&[8; 32], 0, 1, [&frames[0], &frames[0], &frames[0]]).is_err());
        assert!(receive(&[9; 32], 1, 1, [&frames[0], &frames[0], &frames[0]]).is_err());
        assert!(receive(&[9; 32], 0, 1, [&frames[0], &frames[0], &frames[0]]).is_err());
    }
}
