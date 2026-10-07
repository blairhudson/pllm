//! Bounded client-local KV storage with original token positions.
//!
//! Retention is supplied by the trusted caller. This is a storage primitive, not
//! protected selection or a replacement for the compiler's full-KV contract.
//! Key/value rows are flattened [sequence, kv_heads * head_dim]. No view or
//! reference to the original full prefill survives construction.

use zeroize::Zeroize;

pub const MAX_ROWS: usize = 8192;
pub const MAX_WIDTH: usize = 8192;
pub const MAX_POSITION: usize = 1 << 20;
pub const MAX_STORAGE_BYTES: usize = 64 << 20;
pub const MAX_PREFILL_BYTES: usize = 128 << 20;

/// Borrowed active keys, values and original token positions.
pub type RetainedKvView<'a> = (&'a [f32], &'a [f32], &'a [u32]);

/// Private mutable state. No Clone, serialization, or secret-bearing Debug.
pub struct RetainedKvCache {
    keys: Box<[f32]>,
    values: Box<[f32]>,
    positions: Box<[u32]>,
    width: usize,
    capacity: usize,
    length: usize,
    next_position: usize,
    maximum_position: usize,
    closed: bool,
}

pub fn storage_bytes(capacity: usize, width: usize) -> Result<usize, &'static str> {
    if capacity == 0 || capacity > MAX_ROWS || width == 0 || width > MAX_WIDTH {
        return Err("retained KV dimensions exceed their bounds");
    }
    capacity
        .checked_mul(width)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(capacity * 4))
        .filter(|&n| n <= MAX_STORAGE_BYTES)
        .ok_or("retained KV storage exceeds its allocation bound")
}

fn positions_valid(positions: &[usize], limit: usize) -> bool {
    !positions.is_empty()
        && positions.len() <= MAX_ROWS
        && positions.windows(2).all(|p| p[0] < p[1])
        && positions.last().is_some_and(|&p| p < limit)
}

fn zeros<T: Default + Clone>(length: usize) -> Result<Box<[T]>, &'static str> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| "retained KV allocation failed")?;
    values.resize(length, T::default());
    Ok(values.into_boxed_slice())
}

impl RetainedKvCache {
    /// Copy only selected prefill rows into explicitly bounded ownership.
    /// Capacity includes public space for future decode rows. The original
    /// tensors and temporary selection workspace remain caller-owned costs.
    pub fn from_prefill(
        keys: &[f32],
        values: &[f32],
        width: usize,
        retained: &[usize],
        capacity: usize,
        maximum_position: usize,
    ) -> Result<Self, &'static str> {
        storage_bytes(capacity, width)?;
        if keys.is_empty()
            || keys.len() != values.len()
            || keys.len() % width != 0
            || keys.len() > MAX_PREFILL_BYTES / 8
            || maximum_position == 0
            || maximum_position > MAX_POSITION
        {
            return Err("retained KV prefill shape or position bound is invalid");
        }
        let rows = keys.len() / width;
        if rows > MAX_ROWS
            || rows > maximum_position
            || retained.len() > capacity
            || !positions_valid(retained, rows)
            || keys.iter().chain(values).any(|v| !v.is_finite())
        {
            return Err("retained KV prefill values or selection are invalid");
        }
        let mut result = Self::empty(width, capacity, maximum_position)?;
        for (slot, &position) in retained.iter().enumerate() {
            result.keys[slot * width..(slot + 1) * width]
                .copy_from_slice(&keys[position * width..(position + 1) * width]);
            result.values[slot * width..(slot + 1) * width]
                .copy_from_slice(&values[position * width..(position + 1) * width]);
            result.positions[slot] = position as u32;
        }
        result.length = retained.len();
        result.next_position = rows;
        Ok(result)
    }

    fn empty(width: usize, capacity: usize, maximum_position: usize) -> Result<Self, &'static str> {
        storage_bytes(capacity, width)?;
        Ok(Self {
            keys: zeros(capacity * width)?,
            values: zeros(capacity * width)?,
            positions: zeros(capacity)?,
            width,
            capacity,
            length: 0,
            next_position: 0,
            maximum_position,
            closed: false,
        })
    }

    fn live(&self) -> Result<(), &'static str> {
        if self.closed {
            Err("retained KV state is closed")
        } else {
            Ok(())
        }
    }

    /// Append a contiguous logical suffix. All validation precedes mutation;
    /// physical row indices never substitute for absolute rotary positions.
    pub fn append(
        &mut self,
        first_position: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), &'static str> {
        self.live()?;
        if first_position != self.next_position
            || keys.is_empty()
            || keys.len() != values.len()
            || keys.len() % self.width != 0
        {
            return Err("retained KV append shape or logical position differs");
        }
        let rows = keys.len() / self.width;
        if rows > self.capacity - self.length
            || rows > self.maximum_position - self.next_position
            || keys.iter().chain(values).any(|v| !v.is_finite())
        {
            return Err("retained KV append exceeds capacity or has nonfinite values");
        }
        let begin = self.length * self.width;
        let end = begin + keys.len();
        self.keys[begin..end].copy_from_slice(keys);
        self.values[begin..end].copy_from_slice(values);
        for offset in 0..rows {
            self.positions[self.length + offset] = (first_position + offset) as u32;
        }
        self.length += rows;
        self.next_position += rows;
        Ok(())
    }

    /// Resolve original positions locally. Returning slots does not authorize
    /// sending query-dependent addresses to a provider.
    pub fn slots(&self, positions: &[usize]) -> Result<Vec<usize>, &'static str> {
        self.live()?;
        if !positions_valid(positions, self.next_position) {
            return Err("retained KV requested positions are invalid");
        }
        positions
            .iter()
            .map(|&position| {
                self.positions[..self.length]
                    .binary_search(&(position as u32))
                    .map_err(|_| "retained KV requested position was evicted")
            })
            .collect()
    }

    /// Physically compact and release the previous allocation. An explicit new
    /// capacity bounds remaining appends and may only shrink. During success,
    /// old plus new storage coexist; at most 2 * MAX_STORAGE_BYTES. Failure
    /// leaves the complete old state intact.
    pub fn retain(&mut self, positions: &[usize], capacity: usize) -> Result<(), &'static str> {
        let slots = self.slots(positions)?;
        if capacity < positions.len() || capacity > self.capacity {
            return Err("retained KV compaction capacity is invalid");
        }
        let mut next = Self::empty(self.width, capacity, self.maximum_position)?;
        for (destination, &source) in slots.iter().enumerate() {
            let src = source * self.width..(source + 1) * self.width;
            let dst = destination * self.width..(destination + 1) * self.width;
            next.keys[dst.clone()].copy_from_slice(&self.keys[src.clone()]);
            next.values[dst].copy_from_slice(&self.values[src]);
            next.positions[destination] = positions[destination] as u32;
        }
        next.length = positions.len();
        next.next_position = self.next_position;
        *self = next; // Drop erases the old keys, values and indices.
        Ok(())
    }

    pub fn active(&self) -> Result<RetainedKvView<'_>, &'static str> {
        self.live()?;
        Ok((
            &self.keys[..self.length * self.width],
            &self.values[..self.length * self.width],
            &self.positions[..self.length],
        ))
    }

    pub const fn next_position(&self) -> usize {
        self.next_position
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Owned payload allocation, excluding the fixed-size owner and allocator
    /// overhead. Input arrays, caller copies and compaction overlap are extra.
    pub fn allocated_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * 4 + self.positions.len() * 4
    }

    pub fn close(&mut self) {
        self.keys.zeroize();
        self.values.zeroize();
        self.positions.zeroize();
        self.keys = Box::default();
        self.values = Box::default();
        self.positions = Box::default();
        self.length = 0;
        self.capacity = 0;
        self.next_position = 0;
        self.closed = true;
    }
}

impl Drop for RetainedKvCache {
    fn drop(&mut self) {
        self.keys.zeroize();
        self.values.zeroize();
        self.positions.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<f32>, Vec<f32>) {
        let keys = (0..48).map(|n| n as f32 / 8.0 - 2.0).collect();
        let values = (0..48).map(|n| n as f32 / -4.0).collect();
        (keys, values)
    }

    #[test]
    fn owned_retention_preserves_bits_positions_and_shrinks_storage() {
        let (mut keys, mut values) = fixture();
        keys[0] = -0.0;
        let mut cache =
            RetainedKvCache::from_prefill(&keys, &values, 4, &[0, 5, 11], 5, 16).unwrap();
        let (stored_k, stored_v, positions) = cache.active().unwrap();
        for (slot, &position) in positions.iter().enumerate() {
            for col in 0..4 {
                assert_eq!(
                    stored_k[slot * 4 + col].to_bits(),
                    keys[position as usize * 4 + col].to_bits()
                );
                assert_eq!(
                    stored_v[slot * 4 + col].to_bits(),
                    values[position as usize * 4 + col].to_bits()
                );
            }
        }
        keys.fill(99.0);
        values.fill(99.0);
        assert_eq!(cache.active().unwrap().0[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(cache.allocated_bytes(), 5 * (2 * 4 * 4 + 4));
        assert!(cache.allocated_bytes() < (keys.len() + values.len()) * 4);
        cache.append(12, &[7.0; 8], &[8.0; 8]).unwrap();
        assert_eq!(cache.active().unwrap().2, &[0, 5, 11, 12, 13]);
        assert_eq!(cache.next_position(), 14);
        cache.retain(&[5, 13], 3).unwrap();
        assert_eq!(cache.allocated_bytes(), 3 * (2 * 4 * 4 + 4));
        assert_eq!(cache.next_position(), 14);
        cache.append(14, &[9.0; 4], &[10.0; 4]).unwrap();
        assert_eq!(cache.active().unwrap().2, &[5, 13, 14]);
        assert_eq!(cache.slots(&[5, 14]).unwrap(), &[0, 2]);
        assert!(cache.slots(&[12]).is_err());
    }

    #[test]
    fn append_and_compaction_errors_are_transactional() {
        let (keys, values) = fixture();
        let mut cache = RetainedKvCache::from_prefill(&keys, &values, 4, &[3, 11], 4, 14).unwrap();
        let before = cache.active().unwrap().0.to_vec();
        for position in [0, 2, 11, 13, usize::MAX] {
            assert!(cache.append(position, &[1.0; 4], &[2.0; 4]).is_err());
        }
        for (k, v) in [
            (vec![], vec![]),
            (vec![1.0; 3], vec![2.0; 3]),
            (vec![1.0; 4], vec![2.0; 8]),
            (vec![f32::NAN; 4], vec![1.0; 4]),
            (vec![1.0; 12], vec![2.0; 12]),
        ] {
            assert!(cache.append(12, &k, &v).is_err());
        }
        for (positions, capacity) in [
            (vec![], 1),
            (vec![11, 3], 2),
            (vec![3, 3], 2),
            (vec![2], 1),
            (vec![3, 11], 1),
            (vec![3], 5),
            (vec![usize::MAX], 1),
        ] {
            assert!(cache.retain(&positions, capacity).is_err());
        }
        assert_eq!(cache.active().unwrap().0, before);
        assert_eq!(cache.active().unwrap().2, &[3, 11]);
        assert_eq!(cache.next_position(), 12);
        assert_eq!(cache.capacity(), 4);
        cache.append(12, &[1.0; 8], &[2.0; 8]).unwrap();
        assert!(cache.append(14, &[1.0; 4], &[2.0; 4]).is_err());
    }

    #[test]
    fn full_retention_matches_dense_append_and_gather() {
        let (keys, values) = fixture();
        let mut cache =
            RetainedKvCache::from_prefill(&keys, &values, 4, &(0..12).collect::<Vec<_>>(), 16, 16)
                .unwrap();
        let tail = [1.25f32; 12];
        cache.append(12, &tail, &tail).unwrap();
        let mut expected = keys;
        expected.extend(tail);
        assert_eq!(cache.active().unwrap().0, expected);
        assert_eq!(cache.slots(&[1, 7, 14]).unwrap(), &[1, 7, 14]);
        cache.close();
        assert_eq!(cache.allocated_bytes(), 0);
        assert!(cache.active().is_err());
        assert!(cache.slots(&[0]).is_err());
        assert!(cache.append(0, &[0.0; 4], &[0.0; 4]).is_err());
        assert!(cache.retain(&[0], 1).is_err());
        cache.close();
    }

    #[test]
    fn dimensions_allocation_and_initial_position_are_bounded() {
        let (keys, values) = fixture();
        for (width, capacity, maximum) in [
            (0, 4, 16),
            (MAX_WIDTH + 1, 4, 16),
            (4, 0, 16),
            (4, MAX_ROWS + 1, 16),
            (MAX_WIDTH, MAX_ROWS, 16),
            (4, 4, 11),
            (4, 4, MAX_POSITION + 1),
        ] {
            assert!(
                RetainedKvCache::from_prefill(&keys, &values, width, &[0], capacity, maximum)
                    .is_err()
            );
        }
        for indices in [vec![], vec![12], vec![0, 0], vec![11, 0]] {
            assert!(RetainedKvCache::from_prefill(&keys, &values, 4, &indices, 4, 16).is_err());
        }
        assert!(RetainedKvCache::from_prefill(&[f32::INFINITY], &[0.0], 1, &[0], 1, 1).is_err());
    }
}
