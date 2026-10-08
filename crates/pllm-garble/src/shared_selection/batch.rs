//! Bounded coalescing of independent, individually context-bound one-use frames.
use super::{gc, mul, Error};

pub(super) fn overhead(count: usize) -> usize {
    if count == 1 {
        0
    } else {
        12 + 4 * count
    }
}

pub(super) fn pack(frames: &[Vec<u8>]) -> Result<Vec<u8>, Error> {
    if frames.is_empty() || frames.len() > 16 {
        return Err(Error::Resource);
    }
    if frames.len() == 1 {
        return Ok(frames[0].clone());
    }
    let mut out =
        Vec::with_capacity(overhead(frames.len()) + frames.iter().map(Vec::len).sum::<usize>());
    out.extend(b"PLLMSB01");
    out.extend((frames.len() as u32).to_le_bytes());
    for frame in frames {
        out.extend((frame.len() as u32).to_le_bytes());
    }
    for frame in frames {
        out.extend(frame);
    }
    Ok(out)
}

pub(super) fn unpack<'a>(frame: &'a [u8], sizes: &[usize]) -> Result<Vec<&'a [u8]>, Error> {
    if sizes.is_empty()
        || sizes.len() > 16
        || frame.len() != overhead(sizes.len()) + sizes.iter().sum::<usize>()
    {
        return Err(Error::Context);
    }
    if sizes.len() == 1 {
        return Ok(vec![frame]);
    }
    if &frame[..8] != b"PLLMSB01"
        || u32::from_le_bytes(frame[8..12].try_into().unwrap()) as usize != sizes.len()
    {
        return Err(Error::Context);
    }
    let mut offset = overhead(sizes.len());
    let mut result = Vec::with_capacity(sizes.len());
    for (index, &size) in sizes.iter().enumerate() {
        if u32::from_le_bytes(frame[12 + 4 * index..16 + 4 * index].try_into().unwrap()) as usize
            != size
        {
            return Err(Error::Context);
        }
        result.push(&frame[offset..offset + size]);
        offset += size;
    }
    Ok(result)
}

// An error at any index retires all unused peers, including already issued work.
pub(super) struct Predicates(pub Vec<[gc::Party; 2]>);
impl Drop for Predicates {
    fn drop(&mut self) {
        for pair in &mut self.0 {
            for party in pair {
                party.cancel();
            }
        }
    }
}
pub(super) struct Products(pub Vec<[mul::Party; 2]>);
impl Drop for Products {
    fn drop(&mut self) {
        for pair in &mut self.0 {
            for party in pair {
                party.cancel();
            }
        }
    }
}

/// Public bitonic layers; all pairs in one layer have disjoint record owners.
pub(super) fn sort_layers(rows: usize) -> Vec<Vec<(usize, usize)>> {
    let mut layers = Vec::new();
    let mut group = 2;
    while group <= rows {
        let mut distance = group / 2;
        while distance > 0 {
            layers.push(
                (0..rows)
                    .filter_map(|left| {
                        let right = left ^ distance;
                        (right > left).then_some(if left & group == 0 {
                            (left, right)
                        } else {
                            (right, left)
                        })
                    })
                    .collect(),
            );
            distance /= 2;
        }
        group *= 2;
    }
    layers
}

/// Topological wavefront of the original stable adjacent compaction. A move
/// (pass, row) depends on (pass, row-1) and (pass-1, row+1), both one wave earlier.
pub(super) fn compact_layers(rows: usize, keep: usize) -> Vec<Vec<(usize, usize)>> {
    let passes = rows - keep;
    (0..rows + passes - 2)
        .map(|wave| {
            (0..passes)
                .filter_map(|pass| {
                    wave.checked_sub(2 * pass)
                        .filter(|&row| row < rows - pass - 1)
                        .map(|row| (row, row + 1))
                })
                .collect()
        })
        .collect()
}
