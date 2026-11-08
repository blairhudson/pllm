//! Opt-in closed native construction/work binding. Numeric quotes/debits are
//! not Budget, source/range certificates, runtime permits or request admission.
use super::*;

#[path = "tensor_resident_parameters.rs"]
mod resident_parameters;
use resident_parameters::EphemeralBoundLens;
pub use resident_parameters::{ResidentFactorySource, ResidentNativeSchedule};

#[cfg(test)]
pub(super) fn arm_resident_build_panic(invocation: usize) {
    resident_parameters::arm_build_panic(invocation);
}

#[cfg(test)]
#[path = "tensor_work_preflight_silu_tests.rs"]
mod silu_tests;

#[cfg(test)]
thread_local! { static ROW_ENTRIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
#[cfg(test)]
pub(super) fn observed_row_entry() {
    ROW_ENTRIES.with(|n| n.set(n.get() + 1));
}
#[cfg(test)]
pub(super) fn row_entries() -> usize {
    ROW_ENTRIES.with(|n| n.get())
}

/// Actual arguments, not a numeric profile or an extensible Builder callback.
/// Gamma and rotary rows stay immutably borrowed through construction. Input-norm is exactly
/// the existing general-input-row obligation, NOT an attention proof conversion.
#[derive(Debug)]
pub enum FactoryParameters<'a> {
    GuardedSilu {
        input: Interval,
    },
    Negative {
        input: Interval,
    },
    EqualZero {
        input: Interval,
    },
    Rescale {
        input: Interval,
        shift: u8,
    },
    Scale18 {
        input: Interval,
        coefficient: i64,
    },
    CheckedRange {
        incoming: Interval,
        accepted: Interval,
        fallback: i64,
    },
    AllValid,
    RmsNormCertified {
        input: Interval,
        gamma: &'a [i64],
    },
    RmsNormWithInputNorm {
        input: Interval,
        gamma: &'a [i64],
        squared_norm: u128,
    },
    SoftmaxCertified {
        input: Interval,
    },
    RotaryHalf {
        input: Interval,
        cosine: &'a [i64],
        sine: &'a [i64],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkPreflightError {
    Unsupported,
    ParameterBinding,
    EmptySchedule,
    WorkspacePoisoned,
    Native(Error),
    Quote(ConstructionQuoteError),
}
impl From<Error> for WorkPreflightError {
    fn from(value: Error) -> Self {
        Self::Native(value)
    }
}

/// Owns scalar construction parameters and borrows the original public rows, with no
/// setters/Clone or build method. Only ledger-admitted parts can construct.
///
/// ```compile_fail
/// use pllm_core::puma::multiply::{Interval,tensor::{Construction,FactoryParameters,WorkPreflightWorkspace}};
/// let mut gamma=vec![1_i64;2];
/// let factory=Construction::RmsNormCertified{rows:1,width:2,gamma_len:2}.bind(
///     FactoryParameters::RmsNormCertified{input:Interval::new(0,0).unwrap(),gamma:&gamma}).unwrap();
/// let mut workspace=WorkPreflightWorkspace::empty();
/// let ticket=factory.preflight(1,&mut workspace).unwrap();
/// gamma[0]=2; // ticket still owns the original immutable borrow
/// let _=ticket.summary();
/// ```
///
/// ```compile_fail
/// use pllm_core::puma::multiply::{Interval,tensor::{Construction,FactoryParameters,WorkPreflightWorkspace}};
/// let gamma=vec![1_i64;2];
/// let factory=Construction::RmsNormCertified{rows:1,width:2,gamma_len:2}.bind(
///     FactoryParameters::RmsNormCertified{input:Interval::new(0,0).unwrap(),gamma:&gamma}).unwrap();
/// let mut workspace=WorkPreflightWorkspace::empty();
/// let ticket=factory.preflight(1,&mut workspace).unwrap();
/// drop(gamma);
/// let _=ticket.summary();
/// ```
#[derive(Debug)]
pub struct ClosedFactory<'a> {
    construction: Construction,
    parameters: FactoryParameters<'a>,
}

impl Construction {
    /// Bind actual arguments once. This neither prices/adopts numeric work nor
    /// admits resource bytes. Builder accepts only the closed Rescale and Scale18
    /// recipes, not a generic instruction builder or an admission callback.
    /// Uncovered constructors have a typed refusal, not a guessed work formula.
    pub fn bind(
        self,
        parameters: FactoryParameters<'_>,
    ) -> Result<ClosedFactory<'_>, WorkPreflightError> {
        bind_core(&self, &parameters)?;
        Ok(ClosedFactory {
            construction: self,
            parameters,
        })
    }
}

fn bind_core(
    construction: &Construction,
    parameters: &FactoryParameters<'_>,
) -> Result<(), WorkPreflightError> {
        use FactoryParameters as P;
        let compatible = match (construction, parameters) {
            (Construction::Negative { .. }, P::Negative { .. })
            | (Construction::GuardedSilu { .. }, P::GuardedSilu { .. })
            | (Construction::EqualZero { .. }, P::EqualZero { .. })
            | (Construction::Builder { .. }, P::Rescale { .. })
            | (Construction::Builder { .. }, P::Scale18 { .. })
            | (Construction::CheckedRange { .. }, P::CheckedRange { .. })
            | (Construction::AllValid { .. }, P::AllValid)
            | (Construction::SoftmaxCertified { .. }, P::SoftmaxCertified { .. }) => true,
            (Construction::RmsNormCertified { gamma_len, .. }, P::RmsNormCertified { gamma, .. })
            | (
                Construction::RmsNormWithInputNorm { gamma_len, .. },
                P::RmsNormWithInputNorm { gamma, .. },
            ) => *gamma_len == gamma.len(),
            (
                Construction::RotaryHalf {
                    cosine_len,
                    sine_len,
                    ..
                },
                P::RotaryHalf { cosine, sine, .. },
            ) => *cosine_len == cosine.len() && *sine_len == sine.len(),
            (
                Construction::Negative { .. }
                | Construction::EqualZero { .. }
                | Construction::Builder { .. }
                | Construction::CheckedRange { .. }
                | Construction::AllValid { .. }
                | Construction::SoftmaxCertified { .. }
                | Construction::RmsNormCertified { .. }
                | Construction::RmsNormWithInputNorm { .. }
                | Construction::RotaryHalf { .. },
                _,
            ) => false,
            _ => return Err(WorkPreflightError::Unsupported),
        };
        if !compatible {
            return Err(WorkPreflightError::ParameterBinding);
        }
        Ok(())
}

/// Known retained inline slots ONLY. No whole phase/callframe/allocator/RSS
/// claim; factory quotes below remain per-construction owner/buffer envelopes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkPreflightResources {
    pub workspace_bytes: usize,
    pub factory_bytes: usize,
    pub ticket_bytes: usize,
    pub admitted_schedule_bytes: usize,
    pub admitted_part_bytes: usize,
    pub summary_bytes: usize,
    pub ledger_bytes: usize,
    pub ticket_result_bytes: usize,
    pub build_result_bytes: usize,
}

/// Caller-owned fixed work scratch. Quote resources() BEFORE empty(). Three
/// RMS, softmax and activation fixed native stores; no growing buffer or Program storage.
/// Success reuses storage; any error poisons until reset. Reset cannot refund a
/// caller ledger. Public proof CPU/private callframes and caller input backing
/// remain independently unpriced, not zero.
///
/// Resident source and schedule types are publicly nameable without binding native parameters.
///
/// ```
/// use pllm_core::puma::multiply::tensor::{ResidentFactorySource, ResidentNativeSchedule, WorkPreflightWorkspace};
/// let _slots: (Option<ResidentFactorySource<'static>>, ResidentNativeSchedule<'static>) =
///     (None, WorkPreflightWorkspace::resident_schedule());
/// ```
pub struct WorkPreflightWorkspace {
    rms: RmsWorkWorkspace,
    softmax: SoftmaxWorkWorkspace,
    activation: ActivationWorkWorkspace,
    poisoned: bool,
}
impl WorkPreflightWorkspace {
    pub const fn resources() -> WorkPreflightResources {
        WorkPreflightResources {
            workspace_bytes: std::mem::size_of::<Self>(),
            factory_bytes: std::mem::size_of::<ClosedFactory<'static>>(),
            ticket_bytes: std::mem::size_of::<WorkPreflight<'static>>(),
            admitted_schedule_bytes: std::mem::size_of::<AdmittedWorkSchedule<'static>>(),
            admitted_part_bytes: std::mem::size_of::<AdmittedFactory<'static>>(),
            summary_bytes: std::mem::size_of::<WorkSummary>(),
            ledger_bytes: std::mem::size_of::<CallerWorkLedger<'static>>(),
            ticket_result_bytes: std::mem::size_of::<
                Result<WorkPreflight<'static>, WorkPreflightError>,
            >(),
            build_result_bytes: std::mem::size_of::<Result<FactoryOutput, Error>>(),
        }
    }
    pub const fn empty() -> Self {
        Self {
            rms: RmsWorkWorkspace::empty(),
            softmax: SoftmaxWorkWorkspace::empty(),
            activation: ActivationWorkWorkspace::empty(),
            poisoned: false,
        }
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub fn reset(&mut self) {
        self.rms.reset();
        self.softmax.clear();
        self.activation.reset();
        self.poisoned = false;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactoryWorkSummary {
    pub invocations: u64,
    pub unit_work: WorkBudget,
    /// Per-factory only, not an aggregate full+tail/concurrent peak. Borrowed
    /// slice bytes do not bound the caller's original Vec/owner capacities.
    pub construction: ConstructionQuote,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkSummary {
    pub work: WorkBudget,
    pub full: FactoryWorkSummary,
    pub tail: Option<FactoryWorkSummary>,
}

/// Proposed, UNADOPTED source-level event vector. Not CPU time, RSS, authority or
/// an alternate ledger. Bounds are prepaid; cleanup/refusal never refunds them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogicalEvents {
    pub writes: u64,
    pub validation: u64,
    pub transitions: u64,
    pub allocation_attempts: u64,
    pub cleanup_dispatches: u64,
}
impl LogicalEvents {
    /// Category-preserving arithmetic only: no budget, permission or refund.
    pub fn checked_add(self, other: Self) -> Result<Self, Error> {
        Ok(Self {
            writes: self.writes.checked_add(other.writes).ok_or(Error::Bounds)?,
            validation: self.validation.checked_add(other.validation).ok_or(Error::Bounds)?,
            transitions: self.transitions.checked_add(other.transitions).ok_or(Error::Bounds)?,
            allocation_attempts: self.allocation_attempts.checked_add(other.allocation_attempts).ok_or(Error::Bounds)?,
            cleanup_dispatches: self.cleanup_dispatches.checked_add(other.cleanup_dispatches).ok_or(Error::Bounds)?,
        })
    }
    pub fn checked_mul(self, count: u64) -> Result<Self, Error> {
        Ok(Self {
            writes: self.writes.checked_mul(count).ok_or(Error::Bounds)?,
            validation: self.validation.checked_mul(count).ok_or(Error::Bounds)?,
            transitions: self.transitions.checked_mul(count).ok_or(Error::Bounds)?,
            allocation_attempts: self.allocation_attempts.checked_mul(count).ok_or(Error::Bounds)?,
            cleanup_dispatches: self.cleanup_dispatches.checked_mul(count).ok_or(Error::Bounds)?,
        })
    }
    /// Hypothetical unit-weight evaluation, not approved pricing or source funds.
    pub fn total(self) -> Result<u64, Error> {
        [self.writes, self.validation, self.transitions, self.allocation_attempts, self.cleanup_dispatches]
            .into_iter().try_fold(0u64, |n, x| n.checked_add(x).ok_or(Error::Bounds))
    }
}
impl Construction {
    /// One native invocation, including the exclusive failure/cleanup path.
    /// Existing query/lookup/AES debits are not subtracted from these bounds.
    pub fn logical_events(self) -> Result<LogicalEvents, Error> {
        self.quote().map_err(|_| Error::Bounds)?;
        let (lanes, tables) = match self {
            Self::Builder { lanes } | Self::Negative { lanes } | Self::EqualZero { lanes }
            | Self::AllValid { lanes } | Self::CheckedRange { lanes } | Self::GuardedSilu { lanes } => (lanes, 0),
            Self::RmsNormCertified { rows, width, .. } | Self::RmsNormWithInputNorm { rows, width, .. } =>
                (rows.checked_mul(width).ok_or(Error::Bounds)?, 1),
            Self::RotaryHalf { rows, width, .. } => (rows.checked_mul(width).ok_or(Error::Bounds)?, 2),
            Self::SoftmaxCertified { rows, width } => (rows.checked_mul(width).ok_or(Error::Bounds)?, 0),
            Self::RotaryCoefficients { .. } => return Err(Error::Binding),
        };
        let lanes = u64::try_from(lanes).map_err(|_| Error::Bounds)?;
        let nodes = MAX_NODES as u64;
        let scratch = match self {
            Self::RmsNormCertified { .. } | Self::RmsNormWithInputNorm { .. } => 2 * MAX_ROW_LANES,
            Self::RotaryHalf { .. } => 2 * MAX_TILE_LANES,
            _ => 0,
        } as u64;
        // This inventory covers the live RMS/rotary/validity constructors only.
        // Other recipes retain their original numeric/work contracts, but have
        // no adoptable logical-event quote here until separately inventoried.
        if matches!(self, Self::GuardedSilu { .. } | Self::SoftmaxCertified { .. }) {
            return Err(Error::Binding);
        }
        let entries = lanes.checked_mul(tables).ok_or(Error::Bounds)?;
        // Program::new: four full arrays. finish_in_place: available_at is a
        // FIFTH full array; last_use <=2 writes/node, slots/register/available
        // <=3, work accumulator <=1. push: node/width/len <=3 writes/node.
        // Sequential rows add (5+2+3+1+3), not max. No scalar-drop loop.
        let base = LogicalEvents {
            writes: nodes.checked_mul(5 + 2 + 3 + 1 + 3).ok_or(Error::Bounds)?,
            // finish's available_at.position is an independent N*N search.
            // Each candidate: load, predicate, iterator transition <=3. Node
            // emission, digest matching, instruction_work and operand visiting
            // have independent 64-event blocks (no AES-work overlap credit).
            validation: nodes.checked_mul(nodes).and_then(|n| n.checked_mul(3))
                .and_then(|n| n.checked_add(nodes * (64 + 64 + 64 + 64))).ok_or(Error::Bounds)?,
            ..LogicalEvents::default()
        };
        // Full/tail scratch array tails are written even for short outputs.
        // Each Affine has three field initializations and one array destination.
        let scratch_events = LogicalEvents { writes: scratch.checked_mul(4).ok_or(Error::Bounds)?, ..LogicalEvents::default() };
        // affine_lanes_with_pending: destination AND prefix counter; iterator
        // cursor and coefficient fields/scratch are separate. output_bounds,
        // finish digest's two coefficient fields, and geometry/status checks
        // receive distinct per-entry blocks. Prefix is also an owner transition.
        let coefficient_events = LogicalEvents {
            writes: entries.checked_mul(2 + 1 + 3).ok_or(Error::Bounds)?,
            validation: entries.checked_mul(32 + 4 + 4).ok_or(Error::Bounds)?,
            transitions: entries,
            ..LogicalEvents::default()
        };
        // Empty->Building->Ready->transferred (three changes/table). RmsSlot
        // enter/complete/refuse <=3, ResidentBuilder mutation/finish <=4.
        // Allocation attempts are native TABLES only; caller prices Arc new.
        // Cleanup: resident/Builder/Program dispatch <=3, plus each table's
        // rich Box-or-ExactBuffer dispatch AND backing release. Ready/Building
        // are exclusive alternatives, never two releases for the same payload.
        let controls = LogicalEvents {
            writes: 8 + 4 + 2 + 3 + 4 + 3 * tables,
            validation: 8 + 4 + 2 + 3 + 4,
            transitions: 3 + 4 + 3 * tables,
            allocation_attempts: tables,
            cleanup_dispatches: 3 + 2 * tables,
        };
        let mut events = base.checked_add(scratch_events)?.checked_add(coefficient_events)?.checked_add(controls)?;
        if matches!(self, Self::RmsNormCertified { .. } | Self::RmsNormWithInputNorm { .. }) {
            // Ranged product is the exclusive maximum: two initialized arrays,
            // two cell overwrites, four corners, three source/range locals, two
            // extrema accumulators and three result/return destinations =16.
            // Input/tableau repair loops add four non-emitting domain sweeps.
            let cells = nodes.checked_add(4).and_then(|n| n.checked_mul(36)).ok_or(Error::Bounds)?;
            events = events.checked_add(LogicalEvents {
                writes: cells.checked_mul(2 + 2 + 4 + 3 + 2 + 3).ok_or(Error::Bounds)?,
                validation: cells.checked_mul(64).ok_or(Error::Bounds)?,
                ..LogicalEvents::default()
            })?;
            // Partitioned rsqrt_polynomial_bounds: <=36 domains, U in
            // [ONE/4,ONE/2], inclusive step_by(256): <=257 public cells.
            // Nine interval primitives/cell (3 products, 5 affines, 1 rescale)
            // plus scalar/three-term-loop control. Per primitive blocks:
            // writes 4 corners+4 conversions+8 results+4 fields+8 operands+4 args;
            // validation 8 each: reads,mul/conversion,endpoints,extrema,adds,
            // Result handling,rescale bounds,iteration; transitions 4 each:
            // branch entry,result,fold return,borrow retirement. This is
            // SEQUENTIAL to Ranged/hull work and the preserved proof debit.
            let polynomial = LogicalEvents {
                writes: 9 * (4 + 4 + 8 + 4 + 8 + 4) + 32,
                validation: 9 * (8 + 8 + 8 + 8 + 8 + 8 + 8 + 8) + 16,
                transitions: 9 * (4 + 4 + 4 + 4) + 16,
                ..LogicalEvents::default()
            }.checked_mul(36 * 257)?;
            events = events.checked_add(polynomial)?;
        }
        Ok(events)
    }
}

#[derive(Debug)]
struct Part<'a> {
    factory: ClosedFactory<'a>,
    summary: FactoryWorkSummary,
}
#[derive(Debug)]
pub struct WorkPreflight<'a> {
    full: Part<'a>,
    tail: Option<Part<'a>>,
    summary: WorkSummary,
}

impl<'a> ClosedFactory<'a> {
    pub fn preflight(
        self,
        invocations: u64,
        workspace: &mut WorkPreflightWorkspace,
    ) -> Result<WorkPreflight<'a>, WorkPreflightError> {
        preflight(self, invocations, None, workspace)
    }
    /// Bind separately specified full/tail arguments, not a merged fill. Counts
    /// describe actual native invocations; a tail is one invocation. Use single
    /// for tail-only cases; zero full counts are refused, never online permits.
    pub fn preflight_full_tail(
        self,
        full_invocations: u64,
        tail: ClosedFactory<'a>,
        workspace: &mut WorkPreflightWorkspace,
    ) -> Result<WorkPreflight<'a>, WorkPreflightError> {
        preflight(self, full_invocations, Some(tail), workspace)
    }
    fn build(self) -> Result<FactoryOutput, Error> {
        build_core(&self.construction, &self.parameters)
    }
}

fn price_core(
    construction: &Construction,
    parameters: &FactoryParameters<'_>,
    workspace: &mut WorkPreflightWorkspace,
) -> Result<(WorkBudget, ConstructionQuote), WorkPreflightError> {
        use FactoryParameters as P;
        let work = match (construction, parameters) {
            (Construction::GuardedSilu { lanes }, P::GuardedSilu { input }) => {
                Program::activation_guarded_work(
                    Activation::Silu,
                    *lanes,
                    *input,
                    &mut workspace.activation,
                )?
            }
            (Construction::Negative { lanes }, P::Negative { input }) => {
                Program::negative_work(*lanes, *input)?
            }
            (Construction::EqualZero { lanes }, P::EqualZero { input }) => {
                Program::equal_zero_work(*lanes, *input)?
            }
            (Construction::Builder { lanes }, P::Rescale { input, shift }) => {
                validate_tile_lanes(*lanes)?;
                rescaled_bounds(*input, *shift)?;
                WorkBudget::rescale(*lanes)?
            }
            (Construction::Builder { lanes }, P::Scale18 { input, coefficient }) => {
                Program::scale18_work(*lanes, *input, *coefficient, &mut workspace.softmax)?
            }
            (
                Construction::CheckedRange { lanes },
                P::CheckedRange {
                    incoming,
                    accepted,
                    fallback,
                },
            ) => Program::checked_range_work(
                *lanes,
                *incoming,
                *accepted,
                *fallback,
                &mut workspace.rms,
            )?,
            (Construction::AllValid { lanes }, P::AllValid) => {
                Program::all_valid_work(*lanes, &mut workspace.rms)?
            }
            (
                Construction::RmsNormCertified { rows, width, .. },
                P::RmsNormCertified { input, gamma },
            ) => {
                Program::rms_norm_certified_work(*rows, *width, *input, gamma, &mut workspace.rms)?
            }
            (
                Construction::RmsNormWithInputNorm { rows, width, .. },
                P::RmsNormWithInputNorm {
                    input,
                    gamma,
                    squared_norm,
                },
            ) => Program::rms_norm_with_input_norm_work(
                *rows,
                *width,
                *input,
                gamma,
                *squared_norm,
                &mut workspace.rms,
            )?,
            (Construction::SoftmaxCertified { rows, width }, P::SoftmaxCertified { input }) => {
                Program::softmax_certified_work(*rows, *width, *input, &mut workspace.softmax)?
            }
            (
                Construction::RotaryHalf { rows, width, .. },
                P::RotaryHalf {
                    input,
                    cosine,
                    sine,
                },
            ) => Program::rotary_half_work(
                *rows,
                *width,
                *input,
                cosine,
                sine,
                &mut workspace.softmax,
            )?,
            _ => return Err(WorkPreflightError::ParameterBinding),
        };
        // Numeric validation FIRST: geometry-only quoting cannot change the
        // original constructor's Shape/Bounds/domain winner.
        #[cfg(test)]
        resident_parameters::test_hook(resident_parameters::TestStage::Price);
        Ok((
            work,
            construction
                .quote()
                .map_err(WorkPreflightError::Quote)?,
        ))
}

fn build_core(
    construction: &Construction,
    parameters: &FactoryParameters<'_>,
) -> Result<FactoryOutput, Error> {
        #[cfg(test)]
        resident_parameters::test_hook(resident_parameters::TestStage::Build);
        use FactoryParameters as P;
        // The ephemeral enum copies scalars and borrows original parameter rows;
        // no retained owner moves into a consuming native factory.
        let (program, certificate) = match (*construction, parameters.borrowed()) {
            (Construction::GuardedSilu { lanes }, P::GuardedSilu { input }) => (
                Program::activation_guarded(Activation::Silu, lanes, input)?,
                ProducedCertificate::None,
            ),
            (Construction::Negative { lanes }, P::Negative { input }) => {
                (Program::negative(lanes, input)?, ProducedCertificate::None)
            }
            (Construction::EqualZero { lanes }, P::EqualZero { input }) => (
                Program::equal_zero(lanes, input)?,
                ProducedCertificate::None,
            ),
            (Construction::Builder { lanes }, P::Rescale { input, shift }) => {
                let mut b = Builder::new(lanes, input)?;
                let value = b.input();
                let out = b.rescale(value, shift)?;
                (b.finish(out)?, ProducedCertificate::None)
            }
            (Construction::Builder { lanes }, P::Scale18 { input, coefficient }) => (
                Program::scale18(lanes, input, coefficient)?,
                ProducedCertificate::None,
            ),
            (
                Construction::CheckedRange { lanes },
                P::CheckedRange {
                    incoming,
                    accepted,
                    fallback,
                },
            ) => (
                Program::checked_range(lanes, incoming, accepted, fallback)?,
                ProducedCertificate::None,
            ),
            (Construction::AllValid { lanes }, P::AllValid) => {
                (Program::all_valid(lanes)?, ProducedCertificate::None)
            }
            (
                Construction::RmsNormCertified { rows, width, .. },
                P::RmsNormCertified { input, gamma },
            ) => {
                let (p, c) = Program::rms_norm_certified(rows, width, input, gamma)?;
                (p, ProducedCertificate::GeneralOutputRowSquared(c))
            }
            (
                Construction::RmsNormWithInputNorm { rows, width, .. },
                P::RmsNormWithInputNorm {
                    input,
                    gamma,
                    squared_norm,
                },
            ) => {
                let (p, c) =
                    Program::rms_norm_with_input_norm(rows, width, input, gamma, squared_norm)?;
                (p, ProducedCertificate::GeneralOutputRowSquared(c))
            }
            (Construction::SoftmaxCertified { rows, width }, P::SoftmaxCertified { input }) => {
                let (p, c) = Program::softmax_certified(rows, width, input)?;
                (p, ProducedCertificate::AttentionMass(c))
            }
            (
                Construction::RotaryHalf { rows, width, .. },
                P::RotaryHalf {
                    input,
                    cosine,
                    sine,
                },
            ) => (
                Program::rotary_half(rows, width, input, cosine, sine)?,
                ProducedCertificate::None,
            ),
            _ => return Err(Error::Binding),
        };
        Ok(FactoryOutput {
            program,
            certificate,
        })
}

fn preflight<'a>(
    factory: ClosedFactory<'a>,
    count: u64,
    tail: Option<ClosedFactory<'a>>,
    workspace: &mut WorkPreflightWorkspace,
) -> Result<WorkPreflight<'a>, WorkPreflightError> {
    let summary = preflight_core(
        EphemeralBoundLens::borrowed(&factory.construction, &factory.parameters),
        count,
        tail.as_ref().map(|factory| {
            EphemeralBoundLens::borrowed(&factory.construction, &factory.parameters)
        }),
        workspace,
    )?;
    Ok(WorkPreflight {
        full: Part {
            factory,
            summary: summary.full,
        },
        tail: tail
            .zip(summary.tail)
            .map(|(factory, summary)| Part { factory, summary }),
        summary,
    })
}

fn preflight_core(
    factory: EphemeralBoundLens<'_>,
    count: u64,
    tail: Option<EphemeralBoundLens<'_>>,
    workspace: &mut WorkPreflightWorkspace,
) -> Result<WorkSummary, WorkPreflightError> {
    if workspace.poisoned {
        return Err(WorkPreflightError::WorkspacePoisoned);
    }
    let result = (|| {
        let (unit, quote) = price_core(factory.construction, &factory.parameters, workspace)?;
        if count == 0 {
            return Err(WorkPreflightError::EmptySchedule);
        }
        let full = FactoryWorkSummary {
            invocations: count,
            unit_work: unit,
            construction: quote,
        };
        let mut work = unit.checked_mul(count)?;
        let tail = if let Some(factory) = tail {
            let (unit, quote) = price_core(factory.construction, &factory.parameters, workspace)?;
            work = work.checked_add(unit)?;
            Some(FactoryWorkSummary {
                invocations: 1,
                unit_work: unit,
                construction: quote,
            })
        } else {
            None
        };
        let summary = WorkSummary {
            work,
            full,
            tail,
        };
        Ok(summary)
    })();
    if result.is_err() {
        workspace.poisoned = true;
    }
    result
}

impl WorkPreflight<'_> {
    pub fn summary(&self) -> WorkSummary {
        self.summary
    }
}

/// Caller-owned numeric ledger over the ACTUAL mutable aggregate. Caps are
/// caller-supplied numeric ceilings, not Budget/PrssLimits/source authority.
/// Admit atomically before any factory. Debit is not refunded on later native
/// allocation/factory failure; do not hide prior or other scheduled work.
pub struct CallerWorkLedger<'a> {
    total: &'a mut WorkBudget,
    max_operations: u64,
    max_aes: u64,
}
impl<'a> CallerWorkLedger<'a> {
    pub fn new(
        total: &'a mut WorkBudget,
        max_operations: u64,
        max_aes: u64,
    ) -> Result<Self, WorkPreflightError> {
        total.admit(
            0,
            crate::puma::prss::Cursors::default(),
            max_operations,
            max_aes,
        )?;
        Ok(Self {
            total,
            max_operations,
            max_aes,
        })
    }
    pub fn total(&self) -> WorkBudget {
        *self.total
    }
    pub fn admit<'p>(
        &mut self,
        ticket: WorkPreflight<'p>,
    ) -> Result<AdmittedWorkSchedule<'p>, WorkPreflightError> {
        admit_core(self, ticket.summary.work)?;
        Ok(AdmittedWorkSchedule {
            full: AdmittedFactory(ticket.full),
            tail: ticket.tail.map(AdmittedFactory),
        })
    }
}

fn admit_core(ledger: &mut CallerWorkLedger<'_>, work: WorkBudget) -> Result<(), WorkPreflightError> {
    work.admit(
        ledger.total.operations(),
        ledger.total.cursors(),
        ledger.max_operations,
        ledger.max_aes,
    )?;
    let combined = ledger.total.checked_add(work)?;
    *ledger.total = combined;
    Ok(())
}

/// No Clone/Copy or parameter-taking build. Numeric debit covers scheduled
/// invocations; resulting Program is NOT an online execution-count permit.
///
/// Byte-only quotes have no construction method:
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::Construction;
/// let quote=Construction::EqualZero{lanes:3}.quote().unwrap();
/// let _=quote.build();
/// ```
#[derive(Debug)]
pub struct AdmittedWorkSchedule<'a> {
    full: AdmittedFactory<'a>,
    tail: Option<AdmittedFactory<'a>>,
}
impl<'a> AdmittedWorkSchedule<'a> {
    /// Split to preserve caller control of native construction/retention overlap.
    /// Factory quotes do not themselves admit that aggregate lifetime peak.
    pub fn into_parts(self) -> (AdmittedFactory<'a>, Option<AdmittedFactory<'a>>) {
        (self.full, self.tail)
    }
}
/// Construction cannot replace even equally sized foreign gamma:
/// ```compile_fail
/// use pllm_core::puma::{WorkBudget,multiply::{Interval,tensor::{Construction,FactoryParameters,WorkPreflightWorkspace,CallerWorkLedger}}};
/// let gamma=[1_i64;2]; let foreign=[2_i64;2];
/// let factory=Construction::RmsNormCertified{rows:1,width:2,gamma_len:2}.bind(
///     FactoryParameters::RmsNormCertified{input:Interval::new(0,0).unwrap(),gamma:&gamma}).unwrap();
/// let mut workspace=WorkPreflightWorkspace::empty();
/// let ticket=factory.preflight(1,&mut workspace).unwrap();
/// let mut total=WorkBudget::default();
/// let admitted=CallerWorkLedger::new(&mut total,u64::MAX,u64::MAX).unwrap().admit(ticket).unwrap();
/// let (part,_)=admitted.into_parts();
/// let _=part.build(&foreign); // build accepts NO replacement arguments
/// ```
#[derive(Debug)]
pub struct AdmittedFactory<'a>(Part<'a>);
impl<'a> AdmittedFactory<'a> {
    /// Bind theorem only to the original admitted RMS arguments. The raw energy
    /// is conditional, never a compiler input certificate. No replacement args.
    pub fn prepare_rms_energy(
        self,
        workspace: &mut RmsEnergyWorkspace,
    ) -> Result<PreparedRmsEnergy<'a>, Error> {
        let (rows, width, input, gamma, energy) =
            match (&self.0.factory.construction, &self.0.factory.parameters) {
                (
                    Construction::RmsNormWithInputNorm { rows, width, .. },
                    FactoryParameters::RmsNormWithInputNorm {
                        input,
                        gamma,
                        squared_norm,
                    },
                ) => (*rows, *width, *input, *gamma, *squared_norm),
                _ => return Err(Error::Binding),
            };
        let parameters = RmsEnergyParameters::new(rows, width, input, gamma, energy)?;
        rms_output_energy::prepare(self, parameters, workspace)
    }
    pub fn summary(&self) -> FactoryWorkSummary {
        self.0.summary
    }
    /// Consumes the exact captured factory. No replacement gamma/domain/lanes
    /// accepted. Actual native constructor may still fail Allocation or another
    /// native limit; no resource/request admission is granted by this ticket.
    pub fn build(self) -> Result<FactoryOutput, Error> {
        self.0.factory.build()
    }
}

/// Original native output labels, never a conversion between proof families.
/// All existing public range/source obligations still apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProducedCertificate {
    None,
    GeneralOutputRowSquared(u128),
    AttentionMass(u64),
}
pub struct FactoryOutput {
    program: Program,
    certificate: ProducedCertificate,
}

// All refusals precede ownership transfer. The map/assignment gap cannot fail,
// allocate or invoke a hook; the original affine backing moves exactly once.
pub(super) fn commit_rms_output(
    program: &mut Option<Program>,
    squared: u128,
    out: &mut Option<FactoryOutput>,
) -> Result<(), Error> {
    if out.is_some() {
        return Err(Error::Binding);
    }
    if program.is_none() {
        return Err(Error::Phase);
    }
    *out = program.take().map(|program| FactoryOutput {
        program,
        certificate: ProducedCertificate::GeneralOutputRowSquared(squared),
    });
    Ok(())
}

impl FactoryOutput {
    pub fn program(&self) -> &Program {
        &self.program
    }
    pub fn certificate(&self) -> ProducedCertificate {
        self.certificate
    }
    pub fn into_parts(self) -> (Program, ProducedCertificate) {
        (self.program, self.certificate)
    }
}
