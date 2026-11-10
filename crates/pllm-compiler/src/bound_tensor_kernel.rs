//! Borrowed, admitted tensor invocation. No program, cache, material, or register
//! owner lives here. The admitting runtime seals register provenance and retains
//! authority over invocation coordinates, accounting, transport and retirement.
use pllm_garble::{
    fusefss_backend::{composition::{cache::SourceTableCache, gates, Plan}, Error},
    shared_allocation::AdmissionFailure,
};
/// Crate-internal operation boundary, not a hostile-Rust capability proof.
/// Whitelist implementations name concrete owner-minted invocation types only;
/// adding another owner requires an explicit issuance implementation and review.
/// No trait exposes registers, a detachable port, or a whole-program callback.
pub(crate) mod sealed {
    use super::*;

    pub(crate) mod whitelist {
        pub(crate) trait FssSilu {}
        pub(crate) trait NativeFssSilu: FssSilu {}
        pub(crate) trait DiagnosticFssSilu: FssSilu {}
    }

    /// One admitted operation, with inseparable schedule/register borrows.
    /// Plans still borrow the owner's external Budget/cache. Neither invocation
    /// method may detach, replace, or install a Driver.
    pub(crate) trait SiluInvocation<'b>: whitelist::FssSilu {
        fn range(&self) -> gates::InputRange;
        fn len(&self) -> usize;
        fn uses_lanes(&self) -> bool;
        fn scalar<F>(&mut self, index: usize, build: F) -> Result<(), AdmissionFailure<Error>>
        where F: FnOnce(&SourceTableCache<'b>) -> Result<Plan<'b>, Error>;

        fn lanes<F>(&mut self, build: F) -> Result<(), AdmissionFailure<Error>>
        where F: FnMut(&SourceTableCache<'b>) -> Result<Plan<'b>, Error>;
    }

    pub(crate) trait NativeSiluInvocation<'b>:
        SiluInvocation<'b> + whitelist::NativeFssSilu {}
    pub(crate) trait DiagnosticSiluInvocation<'b>:
        SiluInvocation<'b> + whitelist::DiagnosticFssSilu {}
}

pub(crate) fn execute_silu<'b>(
    invocation: impl sealed::NativeSiluInvocation<'b>,
) -> Result<(), AdmissionFailure<Error>> {
    invoke_silu(invocation)
}

/// Same public Plan walk and charges, with driverless zero-output semantics.
/// A diagnostic invocation cannot satisfy the native whitelist.
pub(crate) fn diagnose_silu<'b>(
    invocation: impl sealed::DiagnosticSiluInvocation<'b>,
) -> Result<(), AdmissionFailure<Error>> {
    invoke_silu(invocation)
}

fn invoke_silu<'b>(
    mut invocation: impl sealed::SiluInvocation<'b>,
) -> Result<(), AdmissionFailure<Error>> {
    let range = invocation.range();
    if invocation.uses_lanes() {
        return invocation.lanes(|tables| Ok(gates::silu_cached(range, tables)?.0));
    }
    for i in 0..invocation.len() {
        invocation.scalar(i, |tables| {
            Ok(gates::silu_cached(range, tables)?.0)
        })?;
    }
    Ok(())
}
