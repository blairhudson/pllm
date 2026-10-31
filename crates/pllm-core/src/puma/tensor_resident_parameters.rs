use super::*;

enum ParameterStore<'a> {
    Borrowed(FactoryParameters<'a>),
    Rotary {
        input: Interval,
        cosine: Vec<i64>,
        sine: Vec<i64>,
    },
}

struct FactoryBody<'a> {
    construction: Construction,
    parameters: ParameterStore<'a>,
}

// Temporary only: scalar values are copied, slices borrow the ORIGINAL retained
// fields. Never stored inside their owner; no self-reference or lifetime cast.
pub(super) struct EphemeralBoundLens<'a> {
    pub(super) construction: &'a Construction,
    pub(super) parameters: FactoryParameters<'a>,
}

impl<'a> EphemeralBoundLens<'a> {
    pub(super) fn borrowed(
        construction: &'a Construction,
        parameters: &'a FactoryParameters<'_>,
    ) -> Self {
        Self { construction, parameters: parameters.borrowed() }
    }
}

impl FactoryParameters<'_> {
    pub(super) fn borrowed(&self) -> FactoryParameters<'_> {
        use FactoryParameters as P;
        match self {
            P::GuardedSilu { input } => P::GuardedSilu { input: *input },
            P::Negative { input } => P::Negative { input: *input },
            P::EqualZero { input } => P::EqualZero { input: *input },
            P::Rescale { input, shift } => P::Rescale { input: *input, shift: *shift },
            P::Scale18 { input, coefficient } => P::Scale18 { input: *input, coefficient: *coefficient },
            P::CheckedRange { incoming, accepted, fallback } => P::CheckedRange {
                incoming: *incoming, accepted: *accepted, fallback: *fallback,
            },
            P::AllValid => P::AllValid,
            P::RmsNormCertified { input, gamma } => P::RmsNormCertified { input: *input, gamma },
            P::RmsNormWithInputNorm { input, gamma, squared_norm } => P::RmsNormWithInputNorm {
                input: *input, gamma, squared_norm: *squared_norm,
            },
            P::SoftmaxCertified { input } => P::SoftmaxCertified { input: *input },
            P::RotaryHalf { input, cosine, sine } => P::RotaryHalf { input: *input, cosine, sine },
        }
    }
}

impl FactoryBody<'_> {
    fn lens(&self) -> EphemeralBoundLens<'_> {
        let parameters = match &self.parameters {
            ParameterStore::Borrowed(parameters) => parameters.borrowed(),
            ParameterStore::Rotary { input, cosine, sine } => FactoryParameters::RotaryHalf {
                input: *input, cosine, sine,
            },
        };
        EphemeralBoundLens { construction: &self.construction, parameters }
    }
}

/// Complete caller-owned source, not yet bound, priced or admitted. Populate
/// `Some(source)` BEFORE entering the custody proof. First Vec acquisition and
/// argument evaluation/assembly before that population are outside this proof.
/// No Clone, setters, owned RMS gamma or replacement-parameter build method.
pub struct ResidentFactorySource<'a> {
    body: FactoryBody<'a>,
}

impl Construction {
    /// Infallibly capture existing borrowed parameters without binding them.
    /// RMS gamma remains borrowed for the resident schedule's entire lifetime.
    ///
    /// ```compile_fail
    /// use pllm_core::puma::multiply::{Interval, tensor::{Construction, FactoryParameters, WorkPreflightWorkspace}};
    /// let mut gamma = vec![1_i64; 2];
    /// let mut source = Some(Construction::RmsNormCertified { rows: 1, width: 2, gamma_len: 2 }
    ///     .resident_parameters(FactoryParameters::RmsNormCertified {
    ///         input: Interval::new(0, 0).unwrap(), gamma: &gamma,
    ///     }));
    /// let mut resident = WorkPreflightWorkspace::resident_schedule();
    /// resident.install(&mut source, None, 1).unwrap();
    /// gamma[0] = 2;
    /// resident.bind().unwrap(); // original immutable borrow is still used
    /// ```
    pub fn resident_parameters(self, parameters: FactoryParameters<'_>) -> ResidentFactorySource<'_> {
        ResidentFactorySource {
            body: FactoryBody { construction: self, parameters: ParameterStore::Borrowed(parameters) },
        }
    }

    /// Infallibly capture BOTH original rotary Vecs and scalars. Construct and
    /// store this complete source outside the fallible/unwinding operation.
    /// Capacity is retained, not credited or replaced by requested length.
    pub fn resident_rotary_parameters<'a>(
        self,
        input: Interval,
        cosine: Vec<i64>,
        sine: Vec<i64>,
    ) -> ResidentFactorySource<'a> {
        ResidentFactorySource {
            body: FactoryBody { construction: self, parameters: ParameterStore::Rotary { input, cosine, sine } },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Empty,
    Installed,
    BindingEntered,
    Bound,
    PreflightEntered,
    Priced,
    AdmissionEntered,
    Admitted,
    Burned,
    Complete,
}

/// Opaque retained parameter, RMS recipe and rotary-finish custody. Keep slots
/// resident outside unwind frames. Entered/failed/burned stages cannot retry,
/// including after scratch reset. Numeric summaries cannot grant permission.
/// Rotary finish custody starts ONLY after recipe success and installation in
/// the private slot. RMS installs its empty-header Builder before recipe entry;
/// recipe/pending/partial-finish backing stays resident on error/unwind. Rotary
/// prepare/recipe, other native custody, first acquisition, compiler publication
/// and funding remain open. Full/tail inline slots enlarge this owner: layout,
/// temporary/result overlap and original native parameter backing require
/// prepayment review before composition.
/// No funded capacity or cap-fit claim follows from this storage.
/// Abort, OOM termination and double panic are not protected.
pub struct ResidentNativeSchedule<'a> {
    full: Option<FactoryBody<'a>>,
    tail: Option<FactoryBody<'a>>,
    count: u64,
    phase: Phase,
    priced: Option<WorkSummary>,
    full_finish: NativeFinishSlot,
    tail_finish: NativeFinishSlot,
}

// Independent resident storage, never a local catch-frame holder. Do not replace
// an installed Builder or rebuild a Program. The schedule burns before dispatch.
// The sole production caller has already refused occupied output slots and can
// dispatch each slot only once; private helpers do not create another permission.
struct NativeFinishSlot {
    builder: Option<ResidentBuilder>,
    program: Option<Program>,
    rms: rms_resident::RmsResidentSlot,
}

impl NativeFinishSlot {
    const fn empty() -> Self {
        Self { builder: None, program: None, rms: rms_resident::RmsResidentSlot::empty() }
    }

    fn build_into(&mut self, lens: EphemeralBoundLens<'_>, out: &mut Option<FactoryOutput>) -> Result<(), Error> {
        match (*lens.construction, &lens.parameters) {
            (Construction::RmsNormCertified { rows, width, .. }, FactoryParameters::RmsNormCertified { input, gamma }) => {
                #[cfg(test)]
                test_hook(TestStage::Build);
                self.rms.build_into(&rms_resident::RmsArgs {
                    rows, width, input: *input, gamma, input_norm: None,
                }, out)
            }
            (Construction::RmsNormWithInputNorm { rows, width, .. }, FactoryParameters::RmsNormWithInputNorm { input, gamma, squared_norm }) => {
                #[cfg(test)]
                test_hook(TestStage::Build);
                self.rms.build_into(&rms_resident::RmsArgs {
                    rows, width, input: *input, gamma, input_norm: Some(*squared_norm),
                }, out)
            }
            (Construction::RotaryHalf { rows, width, .. }, FactoryParameters::RotaryHalf { input, cosine, sine }) => {
                #[cfg(test)]
                test_hook(TestStage::Build);
                let (builder, output) = numeric::prepare_rotary_half(rows, width, *input, cosine, sine)?;
                // Safe inline move immediately after recipe return, BEFORE finish.
                self.builder = Some(ResidentBuilder::from_builder(builder));
                #[cfg(test)]
                rotary_finish_regressions::installed(self.builder.as_ref().unwrap());
                self.finish_into(output, out)
            }
            _ => {
                *out = Some(build_core(lens.construction, &lens.parameters)?);
                Ok(())
            }
        }
    }

    fn finish_into(&mut self, output: Value, out: &mut Option<FactoryOutput>) -> Result<(), Error> {
        let builder = self.builder.as_mut().ok_or(Error::Phase)?;
        builder.finish(output)?;
        builder.transfer_complete(&mut self.program)?;
        match self.program.take() {
            Some(program) => {
                // Infallible commit gap: no hooks, checks, allocation or callbacks.
                *out = Some(FactoryOutput { program, certificate: ProducedCertificate::None });
                Ok(())
            }
            None => Err(Error::Phase),
        }
    }
}

impl WorkPreflightWorkspace {
    /// Prepare an empty resident slot BEFORE acquiring/installing sources.
    /// This does not price or fund the wrapper, workspace or result slots.
    pub const fn resident_schedule<'a>() -> ResidentNativeSchedule<'a> {
        ResidentNativeSchedule {
            full: None, tail: None, count: 0, phase: Phase::Empty, priced: None,
            full_finish: NativeFinishSlot::empty(), tail_finish: NativeFinishSlot::empty(),
        }
    }
}

impl<'a> ResidentNativeSchedule<'a> {
    /// Move complete full/tail sources only after ALL refusal checks. No bind,
    /// allocation, callback or fallible assembly occurs between first take and
    /// final installation. Refusal leaves both source slots and resident intact.
    pub fn install(
        &mut self,
        full: &mut Option<ResidentFactorySource<'a>>,
        tail: Option<&mut Option<ResidentFactorySource<'a>>>,
        count: u64,
    ) -> Result<(), WorkPreflightError> {
        if self.phase != Phase::Empty || self.full.is_some() || self.tail.is_some() || self.priced.is_some() {
            return Err(Error::Phase.into());
        }
        if full.is_none() || tail.as_ref().is_some_and(|source| source.is_none()) {
            return Err(Error::Phase.into());
        }
        self.full = full.take().map(|source| source.body);
        self.tail = tail.and_then(|source| source.take()).map(|source| source.body);
        self.count = count;
        self.phase = Phase::Installed;
        Ok(())
    }

    pub fn bind(&mut self) -> Result<(), WorkPreflightError> {
        if self.phase != Phase::Installed {
            return Err(Error::Phase.into());
        }
        self.phase = Phase::BindingEntered;
        #[cfg(test)]
        test_hook(TestStage::Bind);
        let full = self.full.as_ref().ok_or(Error::Phase)?.lens();
        bind_core(full.construction, &full.parameters)?;
        if let Some(tail) = &self.tail {
            let tail = tail.lens();
            bind_core(tail.construction, &tail.parameters)?;
        }
        self.phase = Phase::Bound;
        Ok(())
    }

    pub fn preflight(&mut self, workspace: &mut WorkPreflightWorkspace) -> Result<(), WorkPreflightError> {
        if self.phase != Phase::Bound {
            return Err(Error::Phase.into());
        }
        self.phase = Phase::PreflightEntered;
        let full = self.full.as_ref().ok_or(Error::Phase)?.lens();
        let summary = preflight_core(full, self.count, self.tail.as_ref().map(FactoryBody::lens), workspace)?;
        self.priced = Some(summary);
        self.phase = Phase::Priced;
        Ok(())
    }

    /// Read-only numbers, never an adoptable ticket or authority source.
    pub fn summary(&self) -> Option<WorkSummary> {
        self.priced
    }

    pub fn admit(&mut self, ledger: &mut CallerWorkLedger<'_>) -> Result<(), WorkPreflightError> {
        if self.phase != Phase::Priced {
            return Err(Error::Phase.into());
        }
        self.phase = Phase::AdmissionEntered;
        #[cfg(test)]
        test_hook(TestStage::Admit);
        let work = self.priced.as_ref().ok_or(Error::Phase)?.work;
        admit_core(ledger, work)?;
        // No fallible operation/callback/allocation between debit and permission.
        // This private phase binds the same immutable bodies/count just queried.
        self.phase = Phase::Admitted;
        Ok(())
    }

    /// One construction per admitted full/tail TEMPLATE, as in the legacy API.
    /// Counts debit scheduled work; they are not online execution permissions.
    /// Burn before native fallibility. Build borrows parameters, NEVER consumes
    /// their retained owners. Caller must keep outputs resident through later
    /// failures, and retire associated backing before any enclosing charge.
    /// Full success is installed before attempting tail. RMS errors keep native
    /// Builder/Program custody; this does not repair other factories or compiler
    /// postchecks/publication. No all-or-none result or refund is claimed.
    pub fn build_into(
        &mut self,
        full_out: &mut Option<FactoryOutput>,
        tail_out: &mut Option<FactoryOutput>,
    ) -> Result<(), Error> {
        if self.phase != Phase::Admitted {
            return Err(Error::Phase);
        }
        self.phase = Phase::Burned;
        if full_out.is_some() || tail_out.is_some() {
            return Err(Error::Binding);
        }
        let full = self.full.as_ref().ok_or(Error::Phase)?.lens();
        self.full_finish.build_into(full, full_out)?;
        if let Some(tail) = &self.tail {
            let tail = tail.lens();
            self.tail_finish.build_into(tail, tail_out)?;
        }
        self.phase = Phase::Complete;
        Ok(())
    }
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TestStage { Bind, Price, Admit, Build }

#[cfg(test)]
thread_local! {
    static PANIC_STAGE: std::cell::Cell<Option<(TestStage, usize)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) fn test_hook(stage: TestStage) {
    PANIC_STAGE.with(|pending| {
        if let Some((wanted, remaining)) = pending.get() {
            if wanted == stage {
                if remaining == 1 {
                    pending.set(None);
                    panic!("test resident parameter stage unwind");
                }
                pending.set(Some((wanted, remaining - 1)));
            }
        }
    });
}

#[cfg(test)]
pub(super) fn arm_build_panic(invocation: usize) {
    assert!(invocation > 0);
    PANIC_STAGE.with(|pending| assert!(pending.replace(Some((TestStage::Build, invocation))).is_none()));
}

#[cfg(test)]
#[path = "tensor_resident_parameters_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tensor_rotary_finish_regressions.rs"]
mod rotary_finish_regressions;
