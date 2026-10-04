//! Lossless, bounded four-lane byte rANS for public artifacts, never private tensors.
const LIMIT: usize = 4 << 20;
const BITS: u32 = 12;
const TOTAL: u32 = 1 << BITS;
const LOW: u32 = 1 << 23;
const MAGIC: &[u8; 8] = b"PLLRANS1";
const HEADER: usize = 13;
const CODED_HEADER: usize = HEADER + 512 + 16;

fn raw(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + data.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.push(0);
    out.extend_from_slice(data);
    out
}

/// A frequency model is derived only from bounded, explicitly public bytes.
pub fn encode(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.is_empty() || data.len() > LIMIT {
        return Err("public entropy input exceeds 4 MiB".into());
    }
    if data.len() <= CODED_HEADER {
        return Ok(raw(data));
    }
    let mut counts = [0_u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }
    let n = data.len() as i64;
    let mut frequencies = counts.map(|count| {
        if count == 0 {
            0
        } else {
            ((i64::from(count) * i64::from(TOTAL) / n) as u32).max(1)
        }
    });
    let mut sum: u32 = frequencies.iter().sum();
    while sum != TOTAL {
        let add = sum < TOTAL;
        let index = (0..256)
            .filter(|&i| {
                if add {
                    counts[i] > 0
                } else {
                    frequencies[i] > 1
                }
            })
            .max_by_key(|&i| {
                let deficit =
                    i64::from(counts[i]) * i64::from(TOTAL) - i64::from(frequencies[i]) * n;
                (if add { deficit } else { -deficit }, -(i as i32))
            })
            .ok_or("invalid public entropy frequencies")?;
        if add {
            frequencies[index] += 1;
            sum += 1;
        } else {
            frequencies[index] -= 1;
            sum -= 1;
        }
    }
    let mut cumulative = [0_u32; 256];
    let mut offset = 0;
    for (i, &frequency) in frequencies.iter().enumerate() {
        cumulative[i] = offset;
        offset += frequency;
    }
    let mut states = [LOW; 4];
    let mut coded = Vec::with_capacity(data.len());
    for (index, &symbol) in data.iter().enumerate().rev() {
        let frequency = frequencies[symbol as usize];
        let state = &mut states[index & 3];
        let maximum = ((LOW >> BITS) << 8) * frequency;
        while *state >= maximum {
            coded.push(*state as u8);
            *state >>= 8;
        }
        if coded.len() + CODED_HEADER >= data.len() {
            return Ok(raw(data));
        }
        *state = (*state / frequency) * TOTAL + *state % frequency + cumulative[symbol as usize];
    }
    let mut out = Vec::with_capacity(CODED_HEADER + coded.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.push(1);
    for frequency in frequencies {
        out.extend_from_slice(&(frequency as u16).to_le_bytes());
    }
    for state in states {
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.extend(coded.into_iter().rev());
    Ok(out)
}

/// The enclosing authenticated object still binds the exact raw content hash.
pub fn decode(frame: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let invalid = || "invalid bounded public entropy frame".to_string();
    if expected == 0
        || expected > LIMIT
        || frame.len() < HEADER
        || frame.len() > expected + HEADER
        || &frame[..8] != MAGIC
        || u32::from_le_bytes(frame[8..12].try_into().unwrap()) as usize != expected
    {
        return Err(invalid());
    }
    if frame[12] == 0 {
        if frame.len() != HEADER + expected {
            return Err(invalid());
        }
        return Ok(frame[HEADER..].to_vec());
    }
    if frame[12] != 1 || frame.len() < CODED_HEADER || frame.len() >= expected + HEADER {
        return Err(invalid());
    }
    // (frequency, cumulative, symbol); at most 24 KiB, independent of object size.
    let mut table = [(0_u16, 0_u16, 0_u8); TOTAL as usize];
    let mut cumulative = 0_usize;
    for (symbol, pair) in frame[HEADER..HEADER + 512].chunks_exact(2).enumerate() {
        let frequency = u16::from_le_bytes(pair.try_into().unwrap());
        let end = cumulative + usize::from(frequency);
        if end > TOTAL as usize {
            return Err(invalid());
        }
        table[cumulative..end].fill((frequency, cumulative as u16, symbol as u8));
        cumulative = end;
    }
    if cumulative != TOTAL as usize {
        return Err(invalid());
    }
    let mut states = [0_u32; 4];
    for (state, word) in states
        .iter_mut()
        .zip(frame[HEADER + 512..CODED_HEADER].chunks_exact(4))
    {
        *state = u32::from_le_bytes(word.try_into().unwrap());
        if !(LOW..LOW * 256).contains(state) {
            return Err(invalid());
        }
    }
    let mut cursor = CODED_HEADER;
    let mut output = Vec::with_capacity(expected);
    for index in 0..expected {
        let state = &mut states[index & 3];
        let slot = *state & (TOTAL - 1);
        let (frequency, start, symbol) = table[slot as usize];
        *state = u32::from(frequency) * (*state >> BITS) + slot - u32::from(start);
        while *state < LOW {
            let byte = *frame.get(cursor).ok_or_else(invalid)?;
            cursor += 1;
            *state = (*state << 8) | u32::from(byte);
        }
        output.push(symbol);
    }
    if cursor != frame.len() || states != [LOW; 4] {
        return Err(invalid());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_on_constant_skewed_uniform_and_partial_lanes() {
        for n in [1, 257, 542, 4093, 65536, LIMIT] {
            for mode in 0..3 {
                let data: Vec<u8> = (0..n)
                    .map(|i| match mode {
                        0 => 127,
                        1 => {
                            if i % 31 == 0 {
                                (i / 31) as u8
                            } else {
                                (i % 7) as u8
                            }
                        }
                        _ => i as u8,
                    })
                    .collect();
                let encoded = encode(&data).unwrap();
                assert!(encoded.len() <= n + HEADER);
                assert_eq!(decode(&encoded, n).unwrap(), data);
                if n > CODED_HEADER && mode == 0 {
                    assert!(encoded.len() < n);
                }
            }
        }
    }

    #[test]
    fn rejects_forged_bounds_models_states_and_trailing_bytes() {
        let data = vec![1; 4096];
        let frame = encode(&data).unwrap();
        assert!(decode(&frame, 4095).is_err());
        assert!(decode(&frame, LIMIT + 1).is_err());
        for end in 0..frame.len() {
            assert!(decode(&frame[..end], data.len()).is_err());
        }
        for position in [0, 8, 12, HEADER, HEADER + 2, HEADER + 512] {
            let mut forged = frame.clone();
            forged[position] ^= 0xff;
            assert!(decode(&forged, data.len()).is_err());
        }
        let mut trailing = frame.clone();
        trailing.push(0);
        assert!(decode(&trailing, data.len()).is_err());
        assert!(encode(&[]).is_err());
        assert!(encode(&vec![0; LIMIT + 1]).is_err());
    }
}
