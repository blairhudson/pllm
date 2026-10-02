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
    index: usize,
    buffer: u64,
    available: u8,
}
impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            index: 0,
            buffer: 0,
            available: 0,
        }
    }
    fn read(&mut self, width: u8) -> u64 {
        while self.available < width {
            self.buffer |= u64::from(self.data[self.index]) << self.available;
            self.index += 1;
            self.available += 8;
        }
        let value = self.buffer & ((1_u64 << width) - 1);
        self.buffer >>= width;
        self.available -= width;
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

#[cfg(test)]
mod tests {
    use super::*;
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
