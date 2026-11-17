//! Authenticated complete physical-row metadata, not consumer admission.
//!
//! No transpose, guard, private operand, fee exemption or execution authority is
//! supplied here. Future adoption must bind the original operation and schedule
//! and admit this metadata/check_registry work under its original payer.

use super::public_dot_operand_cap;
use crate::replicated_decoder::source_registry::SourceRegistry;
use crate::replicated_decoder::{Error, PublicSource, SourceTensor};

/// Candidate cap for dots against complete stored rows of one public file.
/// Both actual owners and the registry-owned key/record stay borrowed. Equal
/// bindings do not turn these borrows into request-epoch or operation authority.
pub(in crate::replicated_decoder::program) struct StoredRowDotCap<'owner, 'records> {
    pub(in crate::replicated_decoder::program) source: &'owner PublicSource,
    pub(in crate::replicated_decoder::program) registry: &'owner SourceRegistry<'records>,
    key: &'records String,
    pub(in crate::replicated_decoder::program) record: &'records SourceTensor,
    requested: i64,
    cap: i64,
}

impl<'owner, 'records> StoredRowDotCap<'owner, 'records> {
    /// Sole constructor. New-leaf refusal order: negative request; existing
    /// registry/file authentication; registry key lookup; rank/nonzero/checked
    /// geometry; non-inline file under that same key; optional L1; scalar helper.
    /// Missing L1 is None, never an inferred zero or an interval/norm estimate.
    pub(in crate::replicated_decoder::program) fn new(
        source: &'owner PublicSource,
        registry: &'owner SourceRegistry<'records>,
        key: &str,
        requested: i64,
    ) -> Result<Option<Self>, Error> {
        if requested < 0 {
            return Err(Error::Bounds);
        }
        source.check_registry(registry)?;
        let (key, record) = registry
            .source_records()
            .get_key_value(key)
            .ok_or(Error::Binding)?;
        let [rows, columns] = record.shape.as_slice() else {
            return Err(Error::Shape);
        };
        if *rows == 0 || *columns == 0 {
            return Err(Error::Shape);
        }
        let lanes = rows.checked_mul(*columns).ok_or(Error::Bounds)?;
        let bytes = u64::try_from(lanes)
            .ok()
            .and_then(|n| n.checked_mul(8))
            .ok_or(Error::Bounds)?;
        if record.public_q18.is_some() {
            return Err(Error::Binding);
        }
        let file = source.file(key)?;
        if file.size() != bytes || file.digest() != record.digest {
            return Err(Error::Binding);
        }
        let Some(row_l1) = record.row_l1 else {
            return Ok(None);
        };
        let cap = public_dot_operand_cap(requested, row_l1)?;
        Ok(Some(Self {
            source,
            registry,
            key,
            record,
            requested,
            cap,
        }))
    }

    pub(super) fn key(&self) -> &str {
        self.key
    }

    /// Physical [row count, row width], never transposed consumer dimensions.
    pub(super) fn stored_shape(&self) -> &[usize] {
        &self.record.shape
    }

    pub(super) fn stored_row_l1(&self) -> u64 {
        // Construction requires Some; immutable record remains borrowed.
        self.record.row_l1.unwrap()
    }

    pub(super) fn requested(&self) -> i64 {
        self.requested
    }

    pub(in crate::replicated_decoder::program) fn candidate_cap(&self) -> i64 {
        self.cap
    }
}

#[cfg(test)]
#[path = "public_dot_metadata_tests.rs"]
mod tests;
