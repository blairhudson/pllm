//! Admitted public deterministic coefficient algebra, NOT HE or provenance.
//!
//! ClearInputBuilder must construct the actual immutable u64 source AND public
//! oracle tensor. Its original private ledger pays construction before copying.
//! Every prepay burns a shared-owner attempt, including failure/cancel/unwind;
//! dropping outputs or prepayments never refills work or resets attempt limits.
//! Descriptors/attributes are PUBLIC semantic IR under the shared owner contract.
//! Logical accounting/zeroization is not physical erasure or a security claim.
use crate::clear_input_owner::{self as owner, BorrowedOutput, ClearInputs, Phase};
use pllm_core::coefficient_extraction as core;
pub use pllm_core::coefficient_extraction::Case;
use std::mem::size_of;
use zeroize::Zeroize;

const PREFLIGHT: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Descriptor,
    Length,
    NonCanonical,
    Resource,
    Work,
    Burned,
    Source,
}
impl From<core::Error> for Error {
    fn from(error: core::Error) -> Self {
        match error {
            core::Error::Descriptor => Self::Descriptor,
            core::Error::Length => Self::Length,
            core::Error::NonCanonical => Self::NonCanonical,
            core::Error::Resource => Self::Resource,
        }
    }
}
impl From<owner::Error> for Error {
    fn from(error: owner::Error) -> Self {
        match error {
            owner::Error::Resource => Self::Resource,
            owner::Error::Work => Self::Work,
            owner::Error::Burned => Self::Burned,
            owner::Error::Shape | owner::Error::NonFinite => Self::Source,
        }
    }
}

/// Pure algebra quote, NOT compiler admission or a promise owner ceilings fit.
pub fn quote(cases: &[Case]) -> Result<core::Quote, Error> {
    core::quote(cases).map_err(Into::into)
}

// Both numeric capacities are initialized completely, then wiped before freeing.
// The core Extraction borrows scratch and wipes it on drop; retained words are a
// separate paid copy, avoiding self-references or changes to the core algebra.
struct Buffers {
    scratch: Vec<u64>,
    words: Vec<u64>,
    relation: [u64; 2],
}
impl Zeroize for Buffers {
    fn zeroize(&mut self) {
        self.scratch.as_mut_slice().zeroize();
        self.words.as_mut_slice().zeroize();
        self.relation.zeroize();
    }
}
impl Drop for Buffers {
    fn drop(&mut self) {
        self.zeroize();
        #[cfg(test)]
        CLEANUP_PROBE.with(|probe| {
            if let Some(check) = &*probe.borrow() {
                check(self);
            }
        });
    }
}
#[cfg(test)]
type CleanupProbe = Box<dyn Fn(&Buffers)>;
#[cfg(test)]
thread_local! {
    static CLEANUP_PROBE: std::cell::RefCell<Option<CleanupProbe>> =
        std::cell::RefCell::new(None);
}

fn initialize(v: &mut Vec<u64>, n: usize) -> Result<(), Error> {
    v.try_reserve_exact(n).map_err(|_| Error::Resource)?;
    if v.capacity() != n {
        return Err(Error::Resource);
    }
    v.resize(n, 0);
    Ok(())
}

/// Consuming reservation, bound to the actual original owner, not a Vec header.
/// No callback, caller ledger, guard, output factory, or input adoption route.
/// Valid public usage (imports also control the compile-fail examples):
/// ```
/// use pllm_compiler::{clear_input_owner::{Caps, ClearInputBuilder, ClearInputs},
///     coefficient_extraction_reference::{prepay, Case, Output}};
/// use pllm_garble::shared_allocation::OwnedBudget;
/// fn construct(budget: OwnedBudget) -> ClearInputs {
///     let caps = Caps { f64_values: 0, u64_values: 3, tensors: 2, operations: 0 };
///     let mut b = ClearInputBuilder::new(budget, caps, 65536).unwrap();
///     b.add_u64_tensor("source", &[2], &[1, 2]).unwrap();
///     b.add_u64_tensor("public_oracle", &[1], &[2]).unwrap();
///     b.finish().unwrap()
/// }
/// fn run(input: &ClearInputs, case: Case) -> Output<'_> {
///     prepay(input, case, 0, 1).unwrap().extract().unwrap()
/// }
/// ```
/// Old signature fails arity/types, not unresolved imports.
/// ```compile_fail,E0061
/// use pllm_compiler::coefficient_extraction_reference::{prepay, Case};
/// use pllm_core::row_fold::WorkLedger;
/// use pllm_garble::shared_allocation::{Budget, ExternalCharge};
/// fn old(cases: &[Case], ledger: &mut WorkLedger, budget: &Budget,
///     guard: &ExternalCharge<'_>, input: &Vec<u64>) {
///     let _ = prepay(cases, ledger, budget, guard, input);
/// }
/// ```
/// Fresh ledger cannot reset the retained owner.
/// ```compile_fail,E0061
/// use pllm_compiler::{clear_input_owner::ClearInputs,
///     coefficient_extraction_reference::{prepay, Case}};
/// use pllm_core::row_fold::WorkLedger;
/// fn reset(input: &ClearInputs, case: Case) {
///     let mut fresh = WorkLedger::new(65536, 0);
///     let _ = prepay(input, case, 0, 1, &mut fresh);
/// }
/// ```
/// Reservation itself retains original backing even before execution.
/// ```compile_fail,E0505
/// use pllm_compiler::{clear_input_owner::ClearInputs,
///     coefficient_extraction_reference::{prepay, Case}};
/// fn retire(input: ClearInputs, case: Case) {
///     let p = prepay(&input, case, 0, 1).unwrap();
///     drop(input);
///     std::hint::black_box(p.extract());
/// }
/// ```
/// Cancelled or executed tickets cannot be replayed.
/// ```compile_fail,E0382
/// use pllm_compiler::coefficient_extraction_reference::Prepayment;
/// fn replay(p: Prepayment<'_>) {
///     drop(p.extract());
///     drop(p.extract());
/// }
/// ```
pub struct Prepayment<'a> {
    input: &'a ClearInputs,
    case: Case,
    source: usize,
    oracle: usize,
    quote: core::Quote,
    phase: Phase<'a, Buffers>,
}

/// One bounded case per consuming attempt. Source/oracle select immutable owned
/// u64 tensors, never caller Vecs. Different newly paid owners may start fresh.
/// Trusted owner-attempt seam obligations are local: debit bounded preflight
/// BEFORE descriptor validation, debit all traversal/arithmetic/init/copy/cleanup
/// BEFORE tensor views or factories, reserve capacities BEFORE allocation, and
/// never return/replace the original ledger. Callback does no allocation or
/// numeric traversal. Core quote covers canonical scans and independent oracles;
/// supplement covers both buffer copies/initialization and redundant RAII wipes.
pub fn prepay(
    input: &ClearInputs,
    case: Case,
    source: usize,
    oracle: usize,
) -> Result<Prepayment<'_>, Error> {
    let ticket = input.begin_attempt()?;
    let (quoted, prepared) = ticket.with_ledger(|ledger| {
        ledger.debit(PREFLIGHT, 0).map_err(|_| owner::Error::Work)?;
        // Nested Result preserves descriptor refusals without exporting authority.
        let q = match quote(&[case]) {
            Ok(q) => q,
            Err(e) => return Ok(Err(e)),
        };
        ledger
            .debit(q.word_ops + 16 * q.output_words + 32, 0)
            .map_err(|_| owner::Error::Work)?;
        Ok(Ok(q))
    })?;
    let q = quoted?;
    // Core's 32768-byte control envelope covers inline descriptors, views, moves,
    // wrappers and scalar temporaries. Source/oracle capacities stay charged by
    // the original owner; two full output capacities are charged by this phase.
    let bytes = q.owner_bytes - 8 * (q.input_words + q.oracle_words)
        + 8 * q.output_words
        + size_of::<Buffers>();
    let phase = prepared.reserve::<Buffers>(bytes)?;
    Ok(Prepayment {
        input,
        case,
        source,
        oracle,
        quote: q,
        phase,
    })
}

impl<'a> Prepayment<'a> {
    pub fn quote(&self) -> core::Quote {
        self.quote
    }

    /// Consumed before ANY validation or allocation. Failure/unwind/cancellation
    /// cleans buffers before the phase charge; no repeat-oracle execution route.
    pub fn extract(self) -> Result<Output<'a>, Error> {
        let admitted = self;
        if admitted.source == admitted.oracle {
            return Err(Error::Source);
        }
        let input = admitted.input.u64_tensor(admitted.source)?;
        let secret = admitted.input.u64_tensor(admitted.oracle)?;
        if input.len() != admitted.quote.input_words || secret.len() != admitted.quote.oracle_words
        {
            return Err(Error::Length);
        }
        // Cleanup owner exists from the FIRST allocation and is declared AFTER
        // admission, so every failure frees/zeroizes before phase release.
        let mut buffers = Buffers {
            scratch: Vec::new(),
            words: Vec::new(),
            relation: [0; 2],
        };
        initialize(&mut buffers.scratch, admitted.quote.output_words)?;
        initialize(&mut buffers.words, admitted.quote.output_words)?;
        let extraction = core::extract(admitted.case, &input, &mut buffers.scratch)?;
        let (direct, dot) = extraction.check_public_relation(&secret)?;
        buffers.words.copy_from_slice(extraction.words());
        buffers.relation = [direct, dot];
        drop(extraction); // core borrowed scratch cleanup, original numeric order
        Ok(Output {
            inner: admitted.phase.finish(buffers),
            descriptor: admitted.case.destination,
        })
    }
}

/// Read-only result retains actual original source, oracle and their charges.
/// Numeric payload drops (wipe, then free) BEFORE shared phase releases charge.
/// ```compile_fail,E0505
/// use pllm_compiler::{clear_input_owner::ClearInputs,
///     coefficient_extraction_reference::{prepay, Case}};
/// fn retire(input: ClearInputs, case: Case) {
///     let out = prepay(&input, case, 0, 1).unwrap().extract().unwrap();
///     drop(input);
///     std::hint::black_box(out.words());
/// }
/// ```
/// ```compile_fail,E0506
/// use pllm_compiler::{clear_input_owner::ClearInputs,
///     coefficient_extraction_reference::{prepay, Case}};
/// fn replace(input: &mut ClearInputs, replacement: ClearInputs, case: Case) {
///     let out = prepay(input, case, 0, 1).unwrap().extract().unwrap();
///     *input = replacement;
///     std::hint::black_box(out.words());
/// }
/// ```
pub struct Output<'a> {
    inner: BorrowedOutput<'a, Buffers>,
    descriptor: core::Destination,
}
impl Output<'_> {
    pub fn descriptor(&self) -> core::Destination {
        self.descriptor
    }
    pub fn words(&self) -> &[u64] {
        &self.inner.values().words
    }
    /// Reads the one prepaid oracle result; does not execute more arithmetic.
    pub fn public_relation(&self) -> (u64, u64) {
        let r = self.inner.values().relation;
        (r[0], r[1])
    }
    pub fn original(&self) -> &ClearInputs {
        self.inner.original()
    }
}

#[cfg(test)]
#[path = "coefficient_extraction_reference_tests.rs"]
mod tests;
