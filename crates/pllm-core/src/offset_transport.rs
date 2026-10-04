//! Domain-bound expansion of one worker's additive input share.
use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes256,
};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

/// Expand a fresh seed, or subtract that expansion from a signed-i8 input.
/// The seed belongs only to the mask worker; the complement worker never gets it.
pub fn seeded_share(
    seed: &[u8; 32],
    context: &[u8; 32],
    count: usize,
    bits: u8,
    input: Option<&[u8]>,
) -> Result<Vec<u32>, String> {
    if !matches!(bits, 16 | 24 | 32)
        || !(1..=4_000_000).contains(&count)
        || input.is_some_and(|value| value.len() != count)
    {
        return Err("invalid seeded offset share dimensions or ring".into());
    }
    let mut hash = Sha256::new();
    hash.update(b"pllm/offset-input-share/aes256/v1\0");
    hash.update(seed);
    hash.update(context);
    hash.update((count as u64).to_le_bytes());
    hash.update([bits]);
    let key = Zeroizing::new(<[u8; 32]>::from(hash.finalize()));
    let cipher = Aes256::new_from_slice(key.as_ref()).map_err(|e| e.to_string())?;
    let mask = u32::MAX >> (32 - bits);
    let mut result = Vec::with_capacity(count);
    for first in (0..count.div_ceil(4)).step_by(8) {
        let mut blocks = [aes::cipher::Block::<Aes256>::default(); 8];
        for (lane, block) in blocks.iter_mut().enumerate() {
            block[..8].copy_from_slice(&((first + lane) as u64).to_le_bytes());
        }
        cipher.encrypt_blocks(&mut blocks);
        for block in &mut blocks {
            for bytes in block.chunks_exact(4) {
                if result.len() == count {
                    break;
                }
                let value = u32::from_le_bytes(bytes.try_into().unwrap()) & mask;
                result.push(match input {
                    Some(input) => {
                        (input[result.len()] as i8 as i32 as u32).wrapping_sub(value) & mask
                    }
                    None => value,
                });
            }
            block.as_mut_slice().zeroize();
        }
    }
    Ok(result)
}

/// Minimal signed reconstruction rings from public weights and an absolute input bound.
pub fn row_residue_bits(weights: &[u8], columns: usize, max_input: u8) -> Result<Vec<u8>, String> {
    if columns == 0
        || weights.is_empty()
        || weights.len() % columns != 0
        || weights.len() > 512 * 1024 * 1024
        || !(1..=127).contains(&max_input)
    {
        return Err("invalid row-residue matrix or input bound".into());
    }
    weights
        .chunks_exact(columns)
        .map(|row| {
            let bound = row
                .iter()
                .map(|&x| (x as i8 as i16).unsigned_abs() as u64)
                .sum::<u64>()
                * u64::from(max_input);
            let bits = (65 - bound.leading_zeros()).max(1) as u8;
            if bits > 32 {
                return Err("row bound exceeds signed wrap32".into());
            }
            Ok(bits)
        })
        .collect()
}

fn packed_size(widths: &[u8], rows: usize) -> Result<(usize, usize), String> {
    if widths.is_empty()
        || widths.iter().any(|&n| !(1..=32).contains(&n))
        || !(1..=4096).contains(&rows)
        || widths.len() > 4_000_000 / rows
    {
        return Err("invalid row-residue dimensions".into());
    }
    let bits = widths.iter().map(|&v| usize::from(v)).sum::<usize>() * rows;
    Ok((bits.div_ceil(8), bits))
}

/// LSB-first fixed public widths; values are reduced modulo their row ring.
pub fn pack_row_residues(values: &[u8], widths: &[u8], rows: usize) -> Result<Vec<u8>, String> {
    let (size, _) = packed_size(widths, rows)?;
    if values.len() != rows * widths.len() * 4 {
        return Err("row-residue input size differs".into());
    }
    let mut output = Vec::with_capacity(size);
    let (mut buffer, mut used) = (0_u64, 0_u8);
    for (value, &width) in values.chunks_exact(4).zip(widths.iter().cycle()) {
        let residue =
            u64::from(u32::from_le_bytes(value.try_into().unwrap())) & ((1_u64 << width) - 1);
        buffer |= residue << used;
        used += width;
        while used >= 8 {
            output.push(buffer as u8);
            buffer >>= 8;
            used -= 8;
        }
    }
    if used != 0 {
        output.push(buffer as u8);
    }
    Ok(output)
}

fn validate_packed(data: &[u8], size: usize, bits: usize) -> Result<(), String> {
    if data.len() != size || (bits % 8 != 0 && data[size - 1] >> (bits % 8) != 0) {
        return Err("row-residue length or terminal padding differs".into());
    }
    Ok(())
}

struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
}
impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }
    fn read(&mut self, width: u8) -> u64 {
        let index = self.position / 8;
        let raw = if self.data.len() - index >= 8 {
            u64::from_le_bytes(self.data[index..index + 8].try_into().unwrap())
        } else {
            let mut tail = [0; 8];
            tail[..self.data.len() - index].copy_from_slice(&self.data[index..]);
            u64::from_le_bytes(tail)
        };
        let value = (raw >> (self.position % 8)) & ((1_u64 << width) - 1);
        self.position += usize::from(width);
        value
    }
}

/// Reconstruct directly from both packed shares, avoiding expanded share tensors at the client.
pub fn reconstruct_rows(a: &[u8], b: &[u8], widths: &[u8], rows: usize) -> Result<Vec<u8>, String> {
    let (size, bits) = packed_size(widths, rows)?;
    validate_packed(a, size, bits)?;
    validate_packed(b, size, bits)?;
    let (mut a, mut b) = (BitReader::new(a), BitReader::new(b));
    let mut output = Vec::with_capacity(rows * widths.len() * 8);
    for &width in widths.iter().cycle().take(rows * widths.len()) {
        let modulus = 1_u64 << width;
        let value = (a.read(width) + b.read(width)) & (modulus - 1);
        let centered = if value >= modulus / 2 {
            value as i64 - modulus as i64
        } else {
            value as i64
        };
        output.extend_from_slice(&centered.to_le_bytes());
    }
    Ok(output)
}

/// Preparation's W*r-s, reduced only at the public output coding boundary.
pub fn pack_difference(a: &[u8], b: &[u8], widths: &[u8], rows: usize) -> Result<Vec<u8>, String> {
    packed_size(widths, rows)?;
    if a.len() != rows * widths.len() * 4 || b.len() != a.len() {
        return Err("prepared output operands differ from layout".into());
    }
    let values = zeroize::Zeroizing::new(
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .flat_map(|(a, b)| {
                u32::from_le_bytes(a.try_into().unwrap())
                    .wrapping_sub(u32::from_le_bytes(b.try_into().unwrap()))
                    .to_le_bytes()
            })
            .collect::<Vec<_>>(),
    );
    pack_row_residues(&values, widths, rows)
}

/// Inference adds its integer W*(x-r) to the packed one-use correction.
pub fn add_packed(
    raw: &[u8],
    correction: &[u8],
    widths: &[u8],
    rows: usize,
) -> Result<Vec<u8>, String> {
    let (size, bits) = packed_size(widths, rows)?;
    validate_packed(correction, size, bits)?;
    if raw.len() != rows * widths.len() * 4 {
        return Err("prepared product length differs".into());
    }
    let mut reader = BitReader::new(correction);
    let values: Vec<u8> = raw
        .chunks_exact(4)
        .zip(widths.iter().cycle())
        .flat_map(|(raw, &width)| {
            u32::from_le_bytes(raw.try_into().unwrap())
                .wrapping_add(reader.read(width) as u32)
                .to_le_bytes()
        })
        .collect();
    pack_row_residues(&values, widths, rows)
}

/// Expand validated public output residues without interpreting their sign.
pub fn unpack_rows(payload: &[u8], widths: &[u8], rows: usize) -> Result<Vec<u8>, String> {
    let (size, bits) = packed_size(widths, rows)?;
    validate_packed(payload, size, bits)?;
    let mut reader = BitReader::new(payload);
    let mut result = Vec::with_capacity(rows * widths.len() * 4);
    for &width in widths.iter().cycle().take(rows * widths.len()) {
        result.extend_from_slice(&(reader.read(width) as u32).to_le_bytes());
    }
    Ok(result)
}

/// Trusted-client reconstruction without allocating a second packed mask.
pub fn unmask_packed(
    packed: &[u8],
    mask: &[u8],
    widths: &[u8],
    rows: usize,
) -> Result<Vec<u8>, String> {
    let (size, bits) = packed_size(widths, rows)?;
    validate_packed(packed, size, bits)?;
    if mask.len() != rows * widths.len() * 4 {
        return Err("prepared mask length differs".into());
    }
    let mut reader = BitReader::new(packed);
    let mut output = Vec::with_capacity(rows * widths.len() * 8);
    for (mask, &width) in mask.chunks_exact(4).zip(widths.iter().cycle()) {
        let modulus = 1_u64 << width;
        let value = (reader.read(width) + u64::from(u32::from_le_bytes(mask.try_into().unwrap())))
            & (modulus - 1);
        let centered = if value >= modulus / 2 {
            value as i64 - modulus as i64
        } else {
            value as i64
        };
        output.extend_from_slice(&centered.to_le_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_correction_and_online_residues_preserve_signed_outputs() {
        let widths = [3, 12, 24, 32];
        let expected = [-3_i32, 1441, -151551, i32::MAX];
        let wr = [123_u32, u32::MAX, 800081, 98765];
        let mask = [u32::MAX, 919118, 91091, 333];
        let encode = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let wx: Vec<u32> = expected
            .iter()
            .zip(wr)
            .map(|(&v, r)| (v as u32).wrapping_sub(r))
            .collect();
        let correction = pack_difference(&encode(&wr), &encode(&mask), &widths, 1).unwrap();
        let online = add_packed(&encode(&wx), &correction, &widths, 1).unwrap();
        let decoded = unmask_packed(&online, &encode(&mask), &widths, 1).unwrap();
        assert_eq!(
            decoded,
            expected
                .iter()
                .flat_map(|&v| i64::from(v).to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert!(unmask_packed(&online[..online.len() - 1], &encode(&mask), &widths, 1).is_err());
        let mut bad = online;
        *bad.last_mut().unwrap() |= 0x80;
        assert!(unmask_packed(&bad, &encode(&mask), &widths, 1).is_err());
    }
    #[test]
    fn mixed_rings_reconstruct_and_reject_noncanonical_frames() {
        let widths: Vec<u8> = (1..=32).collect();
        let values: Vec<u32> = (0..64).map(|n| 0xffff_fffe_u32.wrapping_mul(n)).collect();
        let a: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let b: Vec<u8> = values
            .iter()
            .map(|x| 3_u32.wrapping_sub(*x))
            .flat_map(u32::to_le_bytes)
            .collect();
        let pa = pack_row_residues(&a, &widths, 2).unwrap();
        let pb = pack_row_residues(&b, &widths, 2).unwrap();
        let actual = reconstruct_rows(&pa, &pb, &widths, 2).unwrap();
        for (bytes, &width) in actual.chunks_exact(8).zip(widths.iter().cycle()) {
            assert_eq!(
                i64::from_le_bytes(bytes.try_into().unwrap()),
                if width <= 2 { -1 } else { 3 }
            );
        }
        assert!(reconstruct_rows(&pa[..pa.len() - 1], &pb, &widths, 2).is_err());
        assert!(reconstruct_rows(&[0x80], &[0], &[1], 1).is_err());
        assert!(pack_row_residues(&[], &[33], 1).is_err());
        assert_eq!(
            row_residue_bits(&[0, 0, 1, 255, 127, 129], 2, 127).unwrap(),
            [1, 9, 16]
        );
    }
    #[test]
    fn exact_split_and_context_separation() {
        let input = [128, 255, 0, 1, 127, 3, 4];
        for bits in [16, 24, 32] {
            let a = seeded_share(&[3; 32], &[4; 32], input.len(), bits, Some(&input)).unwrap();
            let b = seeded_share(&[3; 32], &[4; 32], input.len(), bits, None).unwrap();
            for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
                assert_eq!(
                    a.wrapping_add(b) & (u32::MAX >> (32 - bits)),
                    (input[i] as i8 as i32 as u32) & (u32::MAX >> (32 - bits))
                );
            }
            assert_ne!(
                b,
                seeded_share(&[3; 32], &[5; 32], input.len(), bits, None).unwrap()
            );
        }
        assert!(seeded_share(&[0; 32], &[0; 32], 4_000_001, 32, None).is_err());
        assert!(seeded_share(&[0; 32], &[0; 32], 2, 32, Some(&input)).is_err());
    }
}
