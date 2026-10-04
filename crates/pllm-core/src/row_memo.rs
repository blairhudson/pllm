//! Byte-bounded trusted-client memoization of an immutable public integer matrix.
use crate::kernels::{Executor, Matrix};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use zeroize::{Zeroize, Zeroizing};

struct Entry {
    input: Vec<i8>,
    output: Vec<i32>,
    last: u64,
}
impl Drop for Entry {
    fn drop(&mut self) {
        self.input.zeroize();
        self.output.zeroize();
    }
}

pub struct RowMemo {
    matrix: Matrix,
    executor: Executor,
    rows: usize,
    columns: usize,
    maximum: usize,
    charge: usize,
    salt: Zeroizing<[u8; 32]>,
    entries: BTreeMap<[u8; 32], Entry>,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl RowMemo {
    pub fn new(
        weights: &[u8],
        rows: usize,
        columns: usize,
        maximum: usize,
        threads: usize,
    ) -> Result<Self, String> {
        if weights.len() > 64 << 20
            || !(1..=65536).contains(&rows)
            || !(1..=16384).contains(&columns)
            || !(1..=64 << 20).contains(&maximum)
        {
            return Err("row memo exceeds dimension/storage limits".into());
        }
        let mut salt = Zeroizing::new([0; 32]);
        getrandom::fill(salt.as_mut()).map_err(|e| e.to_string())?;
        Ok(Self {
            matrix: Matrix::new(weights, rows, columns)?,
            executor: Executor::new(threads, true)?,
            rows,
            columns,
            maximum,
            charge: columns + rows * 4 + 256,
            salt,
            entries: BTreeMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
        })
    }

    pub fn evaluate(&mut self, input: &[u8], batch: usize) -> Result<Vec<i32>, String> {
        if !(1..=4096).contains(&batch)
            || self.rows > 4_000_000 / batch
            || input.len() != batch * self.columns
            || batch * self.rows * self.columns > 1_000_000_000
        {
            return Err("row memo execution exceeds bounded work".into());
        }
        let mut result = Vec::with_capacity(batch * self.rows);
        for raw in input.chunks_exact(self.columns) {
            if self.clock == u64::MAX {
                self.clear();
                self.clock = 0;
            }
            self.clock += 1;
            let mut hash = Sha256::new();
            hash.update(b"pllm/client-integer-row/v1\0");
            hash.update(self.salt.as_ref());
            hash.update(raw);
            let key: [u8; 32] = hash.finalize().into();
            let input: Vec<i8> = raw.iter().map(|&x| x as i8).collect();
            if let Some(entry) = self.entries.get_mut(&key) {
                // Hash collisions cannot change arithmetic: compare the entire row.
                if entry.input == input {
                    entry.last = self.clock;
                    self.hits += 1;
                    result.extend_from_slice(&entry.output);
                    continue;
                }
            }
            self.misses += 1;
            let output = self.matrix.clear(&self.executor, &input, 1)?;
            result.extend_from_slice(&output);
            if self.charge <= self.maximum {
                while (self.entries.len() + 1) * self.charge > self.maximum {
                    let key = *self
                        .entries
                        .iter()
                        .min_by_key(|(_, entry)| entry.last)
                        .unwrap()
                        .0;
                    self.entries.remove(&key);
                }
                self.entries.insert(
                    key,
                    Entry {
                        input,
                        output,
                        last: self.clock,
                    },
                );
            }
        }
        Ok(result)
    }

    pub fn stats(&self) -> (u64, u64, usize, usize) {
        (
            self.hits,
            self.misses,
            self.entries.len(),
            self.entries.len() * self.charge,
        )
    }
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_hits_eviction_and_clear() {
        let w = [1, 2, 255, 2, 3, 4];
        let mut memo = RowMemo::new(&w, 2, 3, 1000, 1).unwrap();
        let x = [1, 2, 3, 4, 5, 6, 1, 2, 3];
        assert_eq!(memo.evaluate(&x, 3).unwrap(), [2, 20, 8, 47, 2, 20]);
        assert_eq!(memo.stats().0, 1);
        assert_eq!(memo.stats().1, 2);
        memo.clear();
        assert_eq!(memo.stats().3, 0);
        let mut memo = RowMemo::new(&w, 2, 3, 267, 1).unwrap();
        assert_eq!(memo.evaluate(&x, 3).unwrap(), [2, 20, 8, 47, 2, 20]);
        assert_eq!(memo.stats(), (0, 3, 1, 267));
        assert!(memo.evaluate(&x, 4).is_err());
    }
}
