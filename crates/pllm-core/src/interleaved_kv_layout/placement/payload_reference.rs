//! Borrowed clear-integer recipe arithmetic only. No publication or funding.

use super::{LayoutError, PlacementContribution, PlacementDestination, PlacementMask};

/// Exact successful traversal, excluding metadata guards and quote arithmetic.
/// Both scans evaluate both masks at every slot, including zero contributions.
/// Not `ideal_payload_work`, a CPU tariff, admission, or physical executor work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct I64ReferenceWork {
    pub validation_slots: usize,
    pub write_slots: usize,
    pub mask_evaluations: usize,
    pub parent_additions: usize,
}

impl I64ReferenceWork {
    pub(crate) fn checked_repeated(self, calls: usize) -> Result<Self, LayoutError> {
        Ok(Self {
            validation_slots: self.validation_slots.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            write_slots: self.write_slots.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            mask_evaluations: self.mask_evaluations.checked_mul(calls).ok_or(LayoutError::Overflow)?,
            parent_additions: self.parent_additions.checked_mul(calls).ok_or(LayoutError::Overflow)?,
        })
    }
}

// Infallible internal mask evaluation, after geometry/lane/feature validation.
// Unlike selected(), this has no Result or test instrumentation callback: the
// write pass must not introduce a refusal or callback after its first write.
fn bit(mask: PlacementMask, slot: usize) -> i64 {
    let g = mask.geometry;
    let block = slot / g.tokens_per_key_tile();
    i64::from(slot % g.tokens_per_key_tile() == mask.lane
        && mask.feature.is_none_or(|f| block / g.heads() == f))
}

fn delta(op: &PlacementContribution, source: &[i64], slot: usize, shift: usize) -> i64 {
    let from = (slot + shift) % source.len();
    // Evaluate BOTH masks, even when the first is zero. Binary products cannot
    // overflow, including source values i64::MIN and i64::MAX.
    let source_bit = bit(op.source_mask, from);
    let destination_bit = bit(op.destination_mask, slot);
    destination_bit * source_bit * source[from]
}

impl PlacementContribution {
    fn reference_geometry(&self) -> Result<usize, LayoutError> {
        let g = self.source_mask.geometry;
        // Compare the COMPLETE geometry, not merely S or the flattened width.
        if g != self.destination_mask.geometry {
            return Err(LayoutError::Geometry);
        }
        for mask in [self.source_mask, self.destination_mask] {
            if mask.lane >= g.tokens_per_key_tile()
                || mask.feature.is_some_and(|f| f >= g.head_dim())
            {
                return Err(LayoutError::Geometry);
            }
        }
        if self.source_mask.feature != self.destination_mask.feature {
            return Err(LayoutError::Geometry);
        }
        match self.destination {
            PlacementDestination::Key { tile } => {
                if self.source_mask.feature.is_some() {
                    return Err(LayoutError::Geometry);
                }
                if tile >= g.capacity().div_ceil(g.tokens_per_key_tile()) {
                    return Err(LayoutError::Index);
                }
            }
            PlacementDestination::Value { group, ciphertext } => {
                if self.source_mask.feature.is_none() {
                    return Err(LayoutError::Geometry);
                }
                if group >= g.capacity().div_ceil(g.tokens_per_value_group())
                    || ciphertext / g.head_dim() != group
                {
                    return Err(LayoutError::Index);
                }
            }
        }
        // Rotation is arithmetic, not authentication of an append descriptor.
        // Every i32 shift is legal, including deliberate wrong-sign controls.
        Ok(g.slots())
    }

    /// Exact bounded successful two-pass traversal; allocates/debits nothing.
    /// Refusal may stop before completing validation; no writes then occur.
    pub(crate) fn i64_reference_work(&self) -> Result<I64ReferenceWork, LayoutError> {
        let slots = self.reference_geometry()?;
        Ok(I64ReferenceWork {
            validation_slots: slots,
            write_slots: slots,
            mask_evaluations: slots.checked_mul(4).ok_or(LayoutError::Overflow)?,
            parent_additions: if self.retain_parent {
                slots.checked_mul(2).ok_or(LayoutError::Overflow)?
            } else { 0 },
        })
    }

    /// Apply exact clear-slot recipe arithmetic into exclusive caller scratch.
    ///
    /// For S slots, `out[j] = dst[j] * src_mask[(j+s) mod S] *
    /// source[(j+s) mod S]`, plus the ORIGINAL parent iff `retain_parent`.
    /// Buffers must have exactly S elements; parent presence must match the flag.
    /// Public recipe fields are checked before either scan. All overflow checks
    /// precede every write; the second scan has no fallible call or callback.
    /// No allocation, scratch, ciphertext, input-range profile, cache mutation,
    /// authority, lifecycle, or funded executor is provided. Borrowed originals
    /// remain unchanged on success and refusal; this function invokes no user code.
    ///
    /// Safe borrows forbid source/output overlap:
    /// ```compile_fail
    /// use pllm_core::interleaved_kv_layout::PlacementContribution;
    /// fn overlap(op: PlacementContribution, data: &mut [i64]) {
    ///     op.apply_i64_reference(data, None, data).unwrap();
    /// }
    /// ```
    /// Safe borrows also forbid parent/output overlap:
    /// ```compile_fail
    /// use pllm_core::interleaved_kv_layout::PlacementContribution;
    /// fn overlap(op: PlacementContribution, source: &[i64], data: &mut [i64]) {
    ///     op.apply_i64_reference(source, Some(data), data).unwrap();
    /// }
    /// ```
    pub fn apply_i64_reference(
        &self,
        source: &[i64],
        parent: Option<&[i64]>,
        output: &mut [i64],
    ) -> Result<(), LayoutError> {
        let slots = self.reference_geometry()?;
        if source.len() != slots || output.len() != slots
            || parent.is_some_and(|p| p.len() != slots)
        {
            return Err(LayoutError::BufferLength);
        }
        if parent.is_some() != self.retain_parent {
            return Err(LayoutError::Geometry);
        }
        // Geometry bounds S <= 32768. This handles i32::MIN without negation;
        // slot+shift is < 2*S and cannot overflow usize on supported targets.
        let shift = i64::from(self.rotation).rem_euclid(slots as i64) as usize;
        for slot in 0..slots {
            let contribution = delta(self, source, slot, shift);
            if let Some(original) = parent {
                original[slot].checked_add(contribution).ok_or(LayoutError::Overflow)?;
            }
        }
        // Identical immutable operands and indices; every sum was proved above.
        // No Result, allocation, callback, instrumentation, or other refusal here.
        for (slot, value) in output.iter_mut().enumerate() {
            let contribution = delta(self, source, slot, shift);
            *value = if let Some(original) = parent { original[slot] + contribution }
                else { contribution };
        }
        Ok(())
    }
}
