//! Bounded reversible public-artifact and orthogonal numeric transforms.
const MAX_BYTES: usize = 4 * 1024 * 1024;
const TILE: usize = 65536;

fn transpose(mut x: u64) -> u64 {
    for (shift, mask) in [
        (7, 0x00aa_00aa_00aa_00aa),
        (14, 0x0000_cccc_0000_cccc),
        (28, 0x0000_0000_f0f0_f0f0),
    ] {
        let t = (x ^ (x >> shift)) & mask;
        x ^= t ^ (t << shift);
    }
    x
}

/// Zigzag signed bytes, then transpose each public block into eight bit planes.
/// Length is preserved; a final non-multiple-of-eight suffix is copied verbatim.
pub fn bitplanes(data: &[u8], inverse: bool) -> Result<Vec<u8>, String> {
    if data.is_empty() || data.len() > MAX_BYTES {
        return Err("bit-plane block exceeds 4 MiB".into());
    }
    let mut result = vec![0; data.len()];
    for (src, dst) in data.chunks(TILE).zip(result.chunks_mut(TILE)) {
        let n = src.len() / 8 * 8;
        let stride = n / 8;
        if inverse {
            for (i, out) in dst[..n].chunks_exact_mut(8).enumerate() {
                let mut word = [0_u8; 8];
                for bit in 0..8 {
                    word[bit] = src[bit * stride + i];
                }
                let word = transpose(u64::from_le_bytes(word));
                let decoded =
                    ((word >> 1) & 0x7f7f_7f7f_7f7f_7f7f) ^ ((word & 0x0101_0101_0101_0101) * 255);
                out.copy_from_slice(&decoded.to_le_bytes());
            }
        } else {
            for (i, value) in src[..n].chunks_exact(8).enumerate() {
                let value = u64::from_le_bytes(value.try_into().unwrap());
                let encoded = ((value << 1) & 0xfefe_fefe_fefe_fefe)
                    ^ (((value >> 7) & 0x0101_0101_0101_0101) * 255);
                let bytes = transpose(encoded).to_le_bytes();
                for bit in 0..8 {
                    dst[bit * stride + i] = bytes[bit];
                }
            }
        }
        dst[n..].copy_from_slice(&src[n..]);
    }
    Ok(result)
}

// Public deterministic signs, not a mask generator or a security primitive.
fn sign(seed: u64, column: usize) -> f32 {
    let mut x = seed
        .wrapping_add(column as u64)
        .wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    if (x ^ (x >> 31)) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Apply the same public orthogonal block map to activation and weight rows.
pub fn hadamard(data: &[u8], columns: usize, block: usize, seed: u64) -> Result<Vec<u8>, String> {
    if columns == 0
        || columns > 32768
        || !block.is_power_of_two()
        || !(2..=1024).contains(&block)
        || columns % block != 0
        || data.is_empty()
        || data.len() > 16 * MAX_BYTES
        || data.len() % (columns * 4) != 0
    {
        return Err("invalid bounded Hadamard matrix".into());
    }
    let mut values: Vec<f32> = data
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if values.iter().any(|x| !x.is_finite() || x.abs() > 1e10) {
        return Err("Hadamard input outside finite public numeric bound".into());
    }
    let signs: Vec<f32> = (0..columns).map(|i| sign(seed, i)).collect();
    let scale = (block as f32).sqrt().recip();
    for row in values.chunks_mut(columns) {
        for (x, s) in row.iter_mut().zip(&signs) {
            *x *= s;
        }
        for tile in row.chunks_mut(block) {
            let mut width = 1;
            while width < block {
                for group in tile.chunks_mut(width * 2) {
                    for i in 0..width {
                        let (a, b) = (group[i], group[width + i]);
                        group[i] = a + b;
                        group[width + i] = a - b;
                    }
                }
                width *= 2;
            }
            for x in tile {
                *x *= scale;
            }
        }
    }
    Ok(values.into_iter().flat_map(f32::to_le_bytes).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_transform_inverts_every_byte_and_partial_tile() {
        for n in [1, 7, 8, 127, 256, 1023, 1024, 4097] {
            let data: Vec<u8> = (0..n).map(|i| i as u8).collect();
            assert_eq!(
                bitplanes(&bitplanes(&data, false).unwrap(), true).unwrap(),
                data
            );
        }
        assert!(bitplanes(&[], false).is_err());
    }
    #[test]
    fn orthogonal_map_preserves_inner_products_and_rejects_nan() {
        let a: Vec<f32> = (0..256).map(|i| (i % 31) as f32 / 31.0).collect();
        let b: Vec<f32> = (0..256).map(|i| (i % 23) as f32 / 23.0).collect();
        let transform = |x: &[f32]| {
            let data: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            hadamard(&data, 256, 128, 71)
                .unwrap()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let reference: f64 = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let actual: f64 = transform(&a)
            .iter()
            .zip(transform(&b))
            .map(|(&a, b)| f64::from(a) * f64::from(b))
            .sum();
        assert!((reference - actual).abs() < 1e-4);
        assert!(hadamard(&[0; 12], 3, 2, 0).is_err());
        assert!(hadamard(&f32::NAN.to_le_bytes().repeat(2), 2, 2, 0).is_err());
    }
}
