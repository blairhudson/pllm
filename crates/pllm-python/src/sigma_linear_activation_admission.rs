//! Typed Rust boundary for the UNREGISTERED local bridge. Incoming allocations
//! must already have been made under their original real charge. This census is
//! not admission of their construction. Charged Rust plan bytes can use the
//! shared compiler importer; Python exporter/packaging gates remain shut.

use crate::sigma_linear_activation::{BridgeError, MAX_VALUES, SELECTED_SOURCE_BYTE_CAPACITY};
use pllm_compiler::sigma_linear_activation::{
    CheckpointTensorSelection, IntegerInterval, OwnedSigmaOperation, SigmaSourceBytes, SourceDtype,
    SourceEndian,
};
use pllm_core::edge_contract::{ConsumerUse, EdgeDescriptor, SourceIdentity};
use pllm_garble::shared_allocation::{OwnedBudget, OwnedCharged};
use pllm_models::DecoderMode;
use std::mem::size_of;

/// Private typed owner. No Python parser, borrowed certificate or plain DTO can
/// enter execution: the bridge takes only an ORIGINAL same-counter OwnedCharged.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct ExecutionData {
    pub(crate) selection: CheckpointTensorSelection,
    pub(crate) artifact_name: String,
    pub(crate) dtype: SourceDtype,
    pub(crate) endian: SourceEndian,
    pub(crate) tensor_offset: u64,
    pub(crate) offsets: Vec<u64>,
    pub(crate) bytes: Vec<u8>,
    pub(crate) coefficients: Vec<f64>,
    pub(crate) encoded: Vec<i64>,
    pub(crate) input: Vec<i64>,
    pub(crate) mode: DecoderMode,
    pub(crate) linear: String,
    pub(crate) activation: String,
    pub(crate) next: String,
    pub(crate) source: SourceIdentity,
    pub(crate) input_edge: EdgeDescriptor,
    pub(crate) private_weights: EdgeDescriptor,
    pub(crate) complete_sum: EdgeDescriptor,
    pub(crate) rescaled: EdgeDescriptor,
    pub(crate) activation_output: EdgeDescriptor,
    pub(crate) rescale_operation: [u8; 32],
    pub(crate) next_use: ConsumerUse,
    pub(crate) input_domain: IntegerInterval,
}

impl ExecutionData {
    pub(crate) fn backing_bytes(&self) -> Result<usize, BridgeError> {
        let mut total = self
            .selection
            .checked_metadata_backing_bytes()
            .map_err(|_| BridgeError::Resource)?;
        for bytes in [
            self.artifact_name.capacity(),
            self.linear.capacity(),
            self.activation.capacity(),
            self.next.capacity(),
            self.bytes.capacity(),
        ] {
            total = total.checked_add(bytes).ok_or(BridgeError::Resource)?;
        }
        for (capacity, element) in [
            (self.offsets.capacity(), size_of::<u64>()),
            (self.coefficients.capacity(), size_of::<f64>()),
            (self.encoded.capacity(), size_of::<i64>()),
            (self.input.capacity(), size_of::<i64>()),
        ] {
            total = total
                .checked_add(capacity.checked_mul(element).ok_or(BridgeError::Resource)?)
                .ok_or(BridgeError::Resource)?;
        }
        Ok(total)
    }

    pub(crate) fn operation(&self) -> OwnedSigmaOperation<'_> {
        OwnedSigmaOperation {
            mode: self.mode,
            linear_operation: &self.linear,
            activation_operation: &self.activation,
            next_operation: &self.next,
            source: self.source,
            input: self.input_edge,
            private_weights: self.private_weights,
            complete_sum: self.complete_sum,
            rescaled: self.rescaled,
            activation_output: self.activation_output,
            rescale_operation: self.rescale_operation,
            next_use: self.next_use,
            input_domain: self.input_domain,
            source_coefficients: &self.coefficients,
            encoded_private_coefficients: &self.encoded,
        }
    }

    pub(crate) fn source(&self) -> SigmaSourceBytes<'_> {
        SigmaSourceBytes {
            artifact_name: &self.artifact_name,
            selection: &self.selection,
            dtype: self.dtype,
            endian: self.endian,
            tensor_offset: self.tensor_offset,
            scalar_artifact_offsets: &self.offsets,
            bytes: &self.bytes,
        }
    }
}

/// Called under the real bridge reservation, before any compiler/native factory.
pub(crate) fn check_original_data(
    data: &OwnedCharged<ExecutionData>,
    root: &OwnedBudget,
) -> Result<(), BridgeError> {
    if !std::ptr::eq(data.budget(), root.budget()) {
        return Err(BridgeError::ForeignInput);
    }
    let actual = data
        .payload()
        .backing_bytes()?
        .checked_add(size_of::<OwnedCharged<ExecutionData>>())
        .ok_or(BridgeError::Resource)?;
    if data.charged_bytes() < actual {
        return Err(BridgeError::UnadmittedInput);
    }
    let value = data.payload();
    if value.bytes.is_empty()
        || value.bytes.len() > SELECTED_SOURCE_BYTE_CAPACITY
        || value.offsets.len() > MAX_VALUES
        || value.coefficients.len() > MAX_VALUES
        || value.encoded.len() > MAX_VALUES
        || value.input.len() > MAX_VALUES
    {
        return Err(BridgeError::Resource);
    }
    Ok(())
}

/// No allocation is permitted behind this gate. Rust charged-byte import does
/// not admit Python exporter acquisition/capacity/immutability/lifetime;
/// CPython GC/PyCell/type-init/exception/tuple/object backing and allocator peak.
/// Rust `size_of::<Py<T>>()` is NOT a CPython allocation quote.
pub(crate) fn require_python_allocation_contract() -> Result<(), BridgeError> {
    Err(BridgeError::UnadmittedPython)
}

#[cfg(test)]
#[path = "sigma_linear_activation_admission_tests.rs"]
mod tests;
