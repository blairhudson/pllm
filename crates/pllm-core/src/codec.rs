//! Portable little endian buffers. Public functions check lengths before reading.
pub fn u32s(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if bytes.len() % 4 != 0 {
        return Err("uint32 buffer length must be divisible by four".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}
pub fn u64s(bytes: &[u8]) -> Result<Vec<u64>, String> {
    if bytes.len() % 8 != 0 {
        return Err("uint64 buffer length must be divisible by eight".into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}
pub fn f32s(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err("float32 buffer length must be divisible by four".into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite float input".into());
    }
    Ok(values)
}
pub fn pack(values: &[u32], width: usize) -> Result<Vec<u8>, String> {
    if !(2..=4).contains(&width) {
        return Err("wire width must be 2, 3, or 4".into());
    }
    let limit = 1u64 << (8 * width);
    if values.iter().any(|&v| v as u64 >= limit) {
        return Err("value exceeds wire width".into());
    }
    let mut bytes = Vec::with_capacity(
        values
            .len()
            .checked_mul(width)
            .ok_or("wire size overflow")?,
    );
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes()[..width]);
    }
    Ok(bytes)
}
pub fn unpack(bytes: &[u8], width: usize) -> Result<Vec<u32>, String> {
    if !(2..=4).contains(&width) || bytes.len() % width != 0 {
        return Err("invalid wire width or truncated payload".into());
    }
    Ok(bytes
        .chunks_exact(width)
        .map(|c| {
            let mut padded = [0; 4];
            padded[..width].copy_from_slice(c);
            u32::from_le_bytes(padded)
        })
        .collect())
}
pub fn modulus(p: u64) -> Result<(), String> {
    if !(2..=(1u64 << 32)).contains(&p) {
        Err("modulus must be in [2,2^32]".into())
    } else {
        Ok(())
    }
}
pub fn mask(x: &[u8], r: &[u32], p: u64) -> Result<Vec<u32>, String> {
    modulus(p)?;
    if x.len() != r.len() || r.iter().any(|&v| v as u64 >= p) {
        return Err("mask shape or residue mismatch".into());
    }
    Ok(x.iter()
        .zip(r)
        .map(|(&a, &b)| (a as i8 as i64 + b as i64).rem_euclid(p as i64) as u32)
        .collect())
}
pub fn unmask(y: &[u32], r: &[u32], p: u64) -> Result<Vec<i64>, String> {
    modulus(p)?;
    if y.len() != r.len() || y.iter().chain(r).any(|&v| v as u64 >= p) {
        return Err("output mask shape or residue mismatch".into());
    }
    Ok(y.iter()
        .zip(r)
        .map(|(&a, &b)| {
            let value = (a as i64 - b as i64).rem_euclid(p as i64);
            if (value as u64) >= (p + 1) / 2 {
                value - p as i64
            } else {
                value
            }
        })
        .collect())
}
pub fn quantize(
    input: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
    supplied: Option<&[f32]>,
) -> Result<(Vec<u8>, Vec<f32>), String> {
    if cols == 0 || rows.checked_mul(cols) != Some(input.len()) || !(2..=8).contains(&bits) {
        return Err("invalid quantization shape or bits".into());
    }
    if input.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite activation".into());
    }
    if let Some(s) = supplied {
        if s.len() != rows || s.iter().any(|&v| !v.is_finite() || v <= 0.) {
            return Err("invalid row scales".into());
        }
    }
    let qmax = ((1u16 << (bits - 1)) - 1) as f32;
    let mut scales = Vec::with_capacity(rows);
    let mut values = Vec::with_capacity(input.len());
    for (i, row) in input.chunks_exact(cols).enumerate() {
        let max = row.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let scale = supplied.map_or(if max > 0. { max / qmax } else { 1. }, |s| s[i]);
        if scale <= 0. || !scale.is_finite() {
            return Err("row scale underflow or overflow".into());
        }
        scales.push(scale);
        for &v in row {
            values.push((v / scale).round_ties_even().clamp(-qmax, qmax) as i8 as u8);
        }
    }
    Ok((values, scales))
}
pub fn random_residues(p: u64, count: usize) -> Result<Vec<u32>, String> {
    modulus(p)?;
    let limit = ((1u64 << 32) / p) * p;
    let mut output = Vec::new();
    output.try_reserve_exact(count).map_err(|e| e.to_string())?;
    while output.len() < count {
        let n = (count - output.len()).min(16384);
        let mut raw = vec![0u8; n * 4];
        getrandom::fill(&mut raw).map_err(|e| e.to_string())?;
        for bytes in raw.chunks_exact(4) {
            let v = u32::from_le_bytes(bytes.try_into().unwrap()) as u64;
            if v < limit {
                output.push((v % p) as u32);
            }
        }
    }
    Ok(output)
}
