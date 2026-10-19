//! Borrowed public Boolean projection; no MPC state or source owner is cloned.
use super::*;

impl PartyRuntime {
    /// Extract `count` bits from the row-major (row, low-to-high bit) view.
    /// `width` must equal the source's nominal width, in 1..=64. The source
    /// must already be admitted to this runtime's ledger (e.g. an A2B output).
    /// Raw/foreign imports cannot acquire a lasting lease through this borrow.
    ///
    /// Both source components remain borrowed and unchanged. The returned
    /// one-bit components have independent reservations in the same ledger.
    /// This consumes one local ordinal, no PRSS/AES, and emits no frames.
    /// Repeating a slice is permitted, but gets a fresh operation binding.
    /// Any error retires this runtime; existing exported owners stay charged
    /// until their own wipe/deallocation, including a retained source.
    pub fn boolean_bit_slice(
        &mut self,
        input: &boolean::Share,
        width: u8,
        start: usize,
        count: usize,
    ) -> Result<boolean::Share, Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            validate(self, &input.value)?;
            if !(1..=64).contains(&width) || width != input.bits() {
                return Err(Error::Bounds);
            }
            if !(1..=128).contains(&count) {
                return Err(Error::Bounds);
            }
            self.source.limits.core.admit(count)?;
            let rows = input.value.operation.lanes;
            let width_usize = usize::from(width);
            let total = rows.checked_mul(width_usize).ok_or(Error::Bounds)?;
            let end = start.checked_add(count).ok_or(Error::Bounds)?;
            if end > total {
                return Err(Error::Bounds);
            }
            let usage = self.payload_usage();
            let source_bytes = rows.checked_mul(8).ok_or(Error::Bounds)?;
            if input.value.first.charged_bytes(&usage) != source_bytes
                || input.value.second.charged_bytes(&usage) != source_bytes
            {
                return Err(Error::Binding);
            }
            reserve(self, count.checked_mul(16).ok_or(Error::Bounds)?)?;
            let mut operation = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                count,
                1,
                self.source.limits.core,
            )?;
            let mut hash = Sha256::new();
            hash.update(BOOLEAN_BIT_SLICE_PROFILE.as_bytes());
            hash.update(self.source.setup);
            hash.update(input.binding());
            hash.update([input.bits(), width]);
            for geometry in [rows, start, count] {
                hash.update((geometry as u64).to_le_bytes());
            }
            operation.consumer = Some(hash.finalize().into());
            self.source.runtime.consume_ordinal(operation)?;
            let mut first = usage.zeros(count)?;
            let mut second = usage.zeros(count)?;
            for i in 0..count {
                let flat = start + i;
                let row = flat / width_usize;
                let bit = flat % width_usize;
                first[i] = (input.first()[row] >> bit) & 1;
                second[i] = (input.second()[row] >> bit) & 1;
            }
            Ok(boolean::Share {
                value: ReplicatedShare {
                    operation,
                    party: self.party(),
                    first,
                    second,
                },
                bits: 1,
            })
        })();
        self.checked(result)
    }
}

#[cfg(test)]
#[path = "boolean_bit_slice_tests.rs"]
mod tests;
