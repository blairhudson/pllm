//! Fixed outer ledger: 13k+1, separate from the child's original eighteen uses.
use super::*;
use pllm_core::edge_contract::ConsumerUse;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Read {
    Entry,
    Max { update: u8, operand: u8 },
    Difference { lane: u8, operand: u8 },
    ChildInput,
    ChildOutput,
    Sum { lane: u8 },
    Project,
    Truncate,
    Coordinate,
    Inverse,
    Product { lane: u8, operand: u8 },
    Gap,
    Output,
}
pub(super) fn template(k: usize) -> Result<([Read; spec::MAX_USES], usize), Error> {
    if !(1..=3).contains(&k) {
        return Err(Error::Domain);
    }
    let mut reads = [Read::Output; spec::MAX_USES];
    let mut n = 0;
    let mut append = |read| {
        reads[n] = read;
        n += 1;
    };
    append(Read::Entry);
    if k == 1 {
        append(Read::Output);
        return Ok((reads, n));
    }
    for update in 0..k - 1 {
        for operand in 0..8 {
            append(Read::Max {
                update: update as u8,
                operand,
            });
        }
    }
    for lane in 0..k {
        for operand in 0..2 {
            append(Read::Difference {
                lane: lane as u8,
                operand,
            });
        }
    }
    append(Read::ChildInput);
    append(Read::ChildOutput);
    for lane in 0..k {
        append(Read::Sum { lane: lane as u8 });
    }
    append(Read::Project);
    append(Read::Truncate);
    append(Read::Coordinate);
    append(Read::Inverse);
    for lane in 0..k {
        for operand in 0..2 {
            append(Read::Product {
                lane: lane as u8,
                operand,
            });
        }
    }
    append(Read::Gap);
    append(Read::Output);
    if n != 13 * k + 1 {
        return Err(Error::Context);
    }
    Ok((reads, n))
}

pub(super) struct Ledger {
    native: internal_use::Ledger<{ spec::MAX_USES }>,
    reads: [Read; spec::MAX_USES],
    length: usize,
}
impl Ledger {
    pub(super) fn new(
        k: usize,
        role: u8,
        uses: [ConsumerUse; spec::MAX_USES],
        bindings: [Binding; spec::MAX_USES],
    ) -> Result<Self, Error> {
        if role > 2 {
            return Err(Error::Domain);
        }
        let (reads, length) = template(k)?;
        Ok(Self {
            native: internal_use::Ledger {
                uses,
                bindings,
                role,
                next: 0,
            },
            reads,
            length,
        })
    }
    /// Burn even on a wrong next-read request. Never retry or restore a suffix.
    pub(super) fn burn(&mut self, read: Read, binding: Binding, role: u8) -> Result<(), Error> {
        let slot = usize::from(self.native.next);
        if slot >= self.length {
            self.native.next = spec::MAX_USES as u8;
            return Err(Error::Consumed);
        }
        let expected = self.reads[slot];
        // Construction-only append from the actual original native owner. No
        // issued handle can access this builder; consumed prefix never changes.
        // Role ledgers are created only from its completed immutable postimage.
        if role == 2 && self.native.role == 2 {
            self.native.bindings[slot] = binding;
        }
        let result = self.native.burn(slot as u8, binding, role);
        if result.is_err() || expected != read {
            self.native.next = spec::MAX_USES as u8;
            return Err(Error::Consumed);
        }
        Ok(())
    }
    pub(super) fn complete(&self) -> bool {
        usize::from(self.native.next) == self.length
    }
    pub(super) fn frozen(&self) -> Result<[Binding; spec::MAX_USES], Error> {
        if !self.complete() {
            return Err(Error::Consumed);
        }
        Ok(self.native.bindings)
    }
}
