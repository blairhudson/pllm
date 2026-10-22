//! Private fixed-template ordered operand uses, not a graph or fanout permit.
//!
//! `ConsumerUse` describes a use; only an owning region's private ledger binds
//! that declaration to original invocation owners. Failed attempts burn slots.

use super::{digest, Binding, Error};
use pllm_core::edge_contract::ConsumerUse;
use sha2::{Digest, Sha256};

/// Fixed templates use at most 255 slots. Fields stay inside the typed boundary;
/// regions construct these only after validating their complete fixed template.
pub(super) struct Ledger<const N: usize> {
    pub(super) uses: [ConsumerUse; N],
    pub(super) bindings: [Binding; N],
    pub(super) role: u8,
    pub(super) next: u8,
}

impl<const N: usize> Ledger<N> {
    /// Hash slots in order, with caller-owned protocol domains and the original
    /// length-prefixed digest framing. No extra slot-count field is added.
    pub(super) fn digest_with_domains(&self, slot_domain: &[u8], ledger_domain: &[u8]) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update((ledger_domain.len() as u64).to_le_bytes());
        hash.update(ledger_domain);
        for index in 0..N {
            hash.update(32u64.to_le_bytes());
            hash.update(digest(&[
                slot_domain,
                &self.uses[index].digest(),
                &self.bindings[index].digest(),
                &[self.role],
            ]));
        }
        hash.finalize().into()
    }

    /// Burn before checking: declaration or failed attempt never grants retry.
    pub(super) fn burn(&mut self, expected: u8, binding: Binding, role: u8) -> Result<(), Error> {
        let slot = self.next;
        self.next = self.next.saturating_add(1);
        if N > usize::from(u8::MAX) || slot != expected || usize::from(slot) >= N {
            return Err(Error::Consumed);
        }
        if self.bindings[usize::from(slot)] != binding
            || self.role != role
            || self.uses[usize::from(slot)].edge() != binding.descriptor
        {
            return Err(Error::Context);
        }
        Ok(())
    }
}
