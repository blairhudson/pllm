//! Borrowed, admitted tensor invocation. No program, cache, material, or register
//! owner lives here. The admitting runtime seals register provenance and retains
//! authority over invocation coordinates, accounting, transport and retirement.
use pllm_garble::{
    fusefss_backend::{composition::{cache::SourceTableCache, gates, Plan}, Error},
    shared_allocation::AdmissionFailure,
};
use zeroize::Zeroizing;

/// Implementations are private runtime adapters, never caller-supplied proofs or
/// public trait objects. Admission supplies effective signed-38 A50/Q12 bounds;
/// the frozen SiLU constructor remains authoritative for the numeric contract.
pub(crate) mod sealed {
    use super::*;

    pub(crate) trait AdmittedSiluRegisters<'a> {
        fn into_parts(self) -> (gates::InputRange, &'a [u64], &'a mut [u64]);
    }

    /// Primitive invocation mechanics, not a whole-program execution callback.
    /// Plans borrow the existing external Budget/cache; gathers/results erase in
    /// the runtime's existing cached_kernel/lanes implementations.
    pub(crate) trait CachedTensorPort<'b> {
        fn uses_lanes(&self) -> bool;
        fn cached_kernel<F>(
            &mut self, input: &[u64], outputs: usize, build: F,
        ) -> Result<Zeroizing<Vec<u64>>, AdmissionFailure<Error>>
        where F: FnOnce(&SourceTableCache<'b>) -> Result<Plan<'b>, Error>;

        fn lanes<F, I, O>(
            &mut self, count: usize, build: F, gather: I, store: O,
        ) -> Result<(), AdmissionFailure<Error>>
        where
            F: FnMut(&SourceTableCache<'b>) -> Result<Plan<'b>, Error>,
            I: FnMut(usize, &mut Vec<u64>) -> Result<(), Error>,
            O: FnMut(usize, &[u64]) -> Result<(), Error>;
    }

    /// Only a port borrowing an installed native Driver implements this marker.
    pub(crate) trait SiluExecutionPort<'b>: CachedTensorPort<'b> {}
    /// Construction/material traversal is deliberately not execution authority.
    pub(crate) trait SiluDiagnosticPort<'b>: CachedTensorPort<'b> {}
}

/// Opaque, move-only, request-local proof. Contains only the admitting runtime's
/// borrowed registers; cannot be built from lengths, IDs, or a caller range.
pub(crate) struct BoundSilu<R> {
    registers: R,
}

pub(crate) fn bind_silu<'a, R: sealed::AdmittedSiluRegisters<'a>>(registers: R) -> BoundSilu<R> {
    BoundSilu { registers }
}

pub(crate) fn execute_silu<'a, 'b, R: sealed::AdmittedSiluRegisters<'a>>(
    binding: BoundSilu<R>, port: &mut impl sealed::SiluExecutionPort<'b>,
) -> Result<(), AdmissionFailure<Error>> {
    invoke_silu(binding, port)
}

/// Same public Plan walk and charges, with driverless zero-output semantics.
/// This path cannot be supplied to execute_silu.
pub(crate) fn diagnose_silu<'a, 'b, R: sealed::AdmittedSiluRegisters<'a>>(
    binding: BoundSilu<R>, port: &mut impl sealed::SiluDiagnosticPort<'b>,
) -> Result<(), AdmissionFailure<Error>> {
    invoke_silu(binding, port)
}

fn invoke_silu<'a, 'b, R: sealed::AdmittedSiluRegisters<'a>>(
    binding: BoundSilu<R>, port: &mut impl sealed::CachedTensorPort<'b>,
) -> Result<(), AdmissionFailure<Error>> {
    let (range, input, output) = binding.registers.into_parts();
    if port.uses_lanes() {
        return port.lanes(
            input.len(),
            |tables| Ok(gates::silu_cached(range, tables)?.0),
            |i, values| { values.push(input[i]); Ok(()) },
            |i, values| { output[i] = values[0]; Ok(()) },
        );
    }
    for i in 0..input.len() {
        output[i] = port.cached_kernel(&input[i..i + 1], 1, |tables| {
            Ok(gates::silu_cached(range, tables)?.0)
        })?[0];
    }
    Ok(())
}
