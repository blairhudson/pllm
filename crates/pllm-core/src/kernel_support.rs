//! Shared borrowed-buffer mechanisms, not admission or protected authority.
//!
//! Counter units belong to each caller's documented inventory. Use separate
//! instances for distinct units; there is no implicit conversion, refund,
//! original-owner funding, CPU-time bound, or production tariff.
//! Callers choose when to arm guards. Drop uses zeroize, but makes no claim
//! about aborts, registers, compiler copies, input erasure, or universal OOM.

use zeroize::Zeroize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WorkExhausted;

#[derive(Debug)]
pub(crate) struct KernelWorkLimit {
    remaining: u64,
}

impl KernelWorkLimit {
    pub(crate) fn new(remaining: u64) -> Self {
        Self { remaining }
    }

    pub(crate) fn remaining(&self) -> u64 {
        self.remaining
    }

    pub(crate) fn debit(&mut self, units: u64) -> Result<(), WorkExhausted> {
        let next = self.remaining.checked_sub(units).ok_or(WorkExhausted)?;
        self.remaining = next;
        Ok(())
    }
}

/// Borrow only: no allocation, ownership transfer, callbacks, or retained copy.
/// finish consumes the guard and preserves its destination on ordinary drop.
pub(crate) struct WipeGuard<'a> {
    pub(crate) words: &'a mut [u64],
    armed: bool,
}

impl<'a> WipeGuard<'a> {
    pub(crate) fn new(words: &'a mut [u64]) -> Self {
        Self { words, armed: true }
    }

    pub(crate) fn finish(mut self) {
        self.armed = false;
    }
}

impl Drop for WipeGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.words.zeroize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn exact_insufficient_zero_and_cumulative_debits() {
        let mut work = KernelWorkLimit::new(9);
        work.debit(0).unwrap();
        assert_eq!(work.remaining(), 9);
        assert_eq!(work.debit(10), Err(WorkExhausted));
        assert_eq!(work.remaining(), 9);
        work.debit(4).unwrap();
        assert_eq!(work.remaining(), 5);
        assert_eq!(work.debit(u64::MAX), Err(WorkExhausted));
        assert_eq!(work.remaining(), 5);
        work.debit(5).unwrap();
        assert_eq!(work.remaining(), 0);
        assert_eq!(work.debit(1), Err(WorkExhausted));
        assert_eq!(work.remaining(), 0);
        work.debit(0).unwrap();
        assert_eq!(work.remaining(), 0);
    }

    #[test]
    fn ordinary_drop_wipes_and_finish_preserves() {
        let mut words = [7; 4];
        {
            let guard = WipeGuard::new(&mut words);
            guard.words[3] = 8;
        }
        assert_eq!(words, [0; 4]);
        {
            let guard = WipeGuard::new(&mut words);
            guard.words.fill(9);
            guard.finish();
        }
        assert_eq!(words, [9; 4]);
    }

    #[test]
    fn real_borrowed_guards_wipe_on_bounded_unwind() {
        let mut scratch = [7; 8];
        let mut output = [8; 4];
        let result = catch_unwind(AssertUnwindSafe(|| {
            let scratch = WipeGuard::new(&mut scratch);
            scratch.words[7] = 1;
            let output = WipeGuard::new(&mut output);
            output.words[0] = 2;
            panic!("bounded shared guard unwind");
        }));
        assert!(result.is_err());
        assert_eq!(scratch, [0; 8]);
        assert_eq!(output, [0; 4]);
    }
}
