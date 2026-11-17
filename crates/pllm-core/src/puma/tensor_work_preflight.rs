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
    CheckedRangeRows {
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
            | (Construction::CheckedRangeRows { .. }, P::CheckedRangeRows { .. })
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
                | Construction::CheckedRangeRows { .. }
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

// UNADOPTED source controls, not native work or permission to spend. The literal
// census is recorded beside the new helpers; loops remain in the old N/N^2 rows.
const fn checked_rows_control_events() -> LogicalEvents {
    LogicalEvents { writes: 64, validation: 64, transitions: 8,
        allocation_attempts: 0, cleanup_dispatches: 1 }
}

// Literal new source sites, nested callees receive separate rows. Geometry has
// five scalar comparisons, two short-circuit groups, one checked_mul/ok_or/?
// chain, one local and one Ok return; <=4 writes,32 reads/checks/results,8
// transitions including exclusive refusal. One K covers that loop-free body.
// Each try_from/map_err/? chain gets its OWN K, not guessed spare transitions
// inside its caller's K. Identity validation has TWO such chains and geometry;
// its remaining body has two profile/validity predicates, one lane comparison,
// their short-circuit branches and Ok return (<=8 writes,32 validation,8 changes).
// Builder: geometry + two conversion chains + private entry + identity install
// =5 K. Finish: geometry + two conversion chains + validator + hash selection
// =5 K. Ordinary closed producer assembly =1 K: total 11, failure reserve 22.
// Resident: same builder 5 + occupancy/install/recipe dispatch 3 + finish 5 +
// finish/transfer/pre-take/commit 4 + pre-take geometry 1 + validator conversions
// 2 + requested-geometry conversions 2 + SECOND validator BODY 1 =23 K,
// exclusive failure reserve 46. Pre-take dispatch prices the validator CALL
// only; its lane/validity/profile tests, branches and return execute separately.
// Occupancy =3 presence tests/one branch; install =one assignment/Some/from_builder
// call; recipe dispatch =one sink constructor/input/read/call; finish and transfer
// =one call/? each. Pre-take dispatch =occupancy, installed borrow, identity match,
// validator call, two geometry comparisons, one OR/branch; conversions separate.
// Commit =take, Some match, FactoryOutput+Some construction, destination write,
// return. No fallible call, hook or allocation after take. Each of these small
// bodies is <=16 writes,48 reads/checks/results,8 transitions, excluding nested
// rows. No bounded recipe or schedule loop is hidden inside K.
// Query entry/output/exit =3 K + geometry =4; bind selection =1; quote selection
// +geometry =2; price selection+query+quote =7; logical quote+selection =3;
// borrowed copy/lens =2; installed quote's additional selection =1; unsupported
// classifier =1. Copies have three scalar fields; all selections are one match.
// Resources =geometry+validator+two conversions =4 K, exclusive failure =8.
// shared checked_recipe, checked_output, prepare, row initialization, finish
// arrays/liveness/work and ordinary resource loops retain their full inventories.
// This is conservative source-call accounting, not exact CPU counts or SHA rounds.

pub(super) const fn checked_rows_constructor_control_bytes() -> usize {
    std::mem::size_of::<resident_parameters::CheckedRowsArgs>()
        + std::mem::size_of::<CheckedRowsIdentity>() // construction temporary
        // Builder::checked_range_rows: rows and width usize->u64 sites.
        + 2 * checked_rows_conversion_control_bytes::<usize, u64>()
        + std::mem::size_of::<Result<usize, Error>>()
        + std::mem::size_of::<usize>() // lanes binding
        + 5 * std::mem::size_of::<usize>() // builder/geometry formals and geometry local
        + std::mem::size_of::<Option<usize>>() // checked product return
        + 2 * std::mem::size_of::<Interval>() // constructor input/accepted formals
        + std::mem::size_of::<i64>() // fallback formal
        + std::mem::size_of::<Builder>() // private entry local, distinct from caller
        + std::mem::size_of::<Result<Builder, Error>>() // private entry return
        + std::mem::size_of::<CheckedValue>()
        + std::mem::size_of::<Result<CheckedValue, Error>>()
        + std::mem::size_of::<Value>()
        + std::mem::size_of::<ProducedCertificate>()
        + std::mem::size_of::<ResourceProducer>()
        + std::mem::size_of::<FactoryOutput>() // construction temporary
        + std::mem::size_of::<Option<FactoryOutput>>() // installed destination
        + 2 * std::mem::size_of::<&'static mut Builder>()
        + std::mem::size_of::<Result<Program, Error>>() // checked finish return
        + std::mem::size_of::<Result<(), Error>>()
}

pub(super) const fn checked_rows_resident_control_bytes() -> usize {
    resident_parameters::checked_rows_resident_control_bytes()
}

// The public fallback still performs the complete real calculation. Rows adds
// only its real geometry/identity controls; never substitute indexed scratch.
pub(super) const fn checked_rows_resource_control_bytes() -> usize {
    resource_calculation_control_bytes() + checked_rows_validation_control_bytes()
        + std::mem::size_of::<Result<Resources, Error>>()
}

pub(super) fn checked_rows_resource_events() -> Result<LogicalEvents, Error> {
    LogicalEvents { validation: Program::full_resource_validation_visits()?,
        ..LogicalEvents::default() }.checked_add(checked_rows_control_events().checked_mul(8)?)
}

impl Construction {
    /// One native invocation, including the exclusive failure/cleanup path.
    /// Existing query/lookup/AES debits are not subtracted from these bounds.
    pub fn logical_events(self) -> Result<LogicalEvents, Error> {
        self.quote().map_err(|_| Error::Bounds)?;
        let (lanes, tables) = match self {
            Self::Builder { lanes } | Self::Negative { lanes } | Self::EqualZero { lanes }
            | Self::AllValid { lanes } | Self::CheckedRange { lanes } | Self::GuardedSilu { lanes } => (lanes, 0),
            Self::CheckedRangeRows { rows, width } => (rows.checked_mul(width).ok_or(Error::Bounds)?, 0),
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
        match self {
        Self::CheckedRangeRows { .. } => {
            // Ordinary constructor <=11 K; resident <=23 K. Reserve resident's
            // exclusive failure envelope (46), plus quote/selection (3) and
            // installed-quote selection (1), each with exclusive retirement.
            // Shared N/N^2 recipe/finish work stays in base, not in K or H.
            events = events.checked_add(checked_rows_control_events().checked_mul(54)?)?
                .checked_add(LogicalEvents { writes: 3468, validation: 17_728,
                    transitions: 277, allocation_attempts: 0, cleanup_dispatches: 1 })?;
        },
        Self::RmsNormCertified { .. } | Self::RmsNormWithInputNorm { .. } => {
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
        },
        _ => {},
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
            (Construction::CheckedRangeRows { rows, width }, P::CheckedRangeRows { incoming, accepted, fallback }) =>
                Program::checked_range_rows_work(*rows, *width, *incoming, *accepted, *fallback, &mut workspace.rms)?,
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
            (Construction::CheckedRangeRows { rows, width }, P::CheckedRangeRows { incoming, accepted, fallback }) => (
                Program::checked_range_rows(rows, width, incoming, accepted, fallback)?, ProducedCertificate::None,
            ),
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
            producer: match parameters {
                P::GuardedSilu { .. } => ResourceProducer::GuardedSilu,
                P::Negative { .. } => ResourceProducer::Negative,
                P::EqualZero { .. } => ResourceProducer::EqualZero,
                P::Rescale { .. } => ResourceProducer::Rescale,
                P::Scale18 { .. } => ResourceProducer::Scale18,
                P::CheckedRange { .. } => ResourceProducer::CheckedRange,
                P::CheckedRangeRows { .. } => ResourceProducer::CheckedRangeRows,
                P::AllValid => ResourceProducer::AllValid,
                P::RmsNormCertified { .. } => ResourceProducer::RmsCertified,
                P::RmsNormWithInputNorm { .. } => ResourceProducer::RmsWithInputNorm,
                P::SoftmaxCertified { .. } => ResourceProducer::Softmax,
                P::RotaryHalf { .. } => ResourceProducer::Rotary,
            },
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
/// Closed native product. Public resource views borrow only genuine closed
/// outputs; no construction from arbitrary programs or SOURCE funding authority.
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::ProofFreeOutput;
/// let view = ProofFreeOutput { program: todo!() };
/// ```
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::rms_resident::RmsOutputHandoff;
/// let foreign = RmsOutputHandoff { program: &mut None, squared: 0, origin: todo!() };
/// ```
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::{FactoryOutput, Program, ProducedCertificate};
/// fn certify_foreign(program: Program) -> FactoryOutput {
///     FactoryOutput { program, certificate: ProducedCertificate::None, producer: todo!() }
/// }
/// ```
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::{FactoryOutput, Program};
/// fn replace(output: &FactoryOutput, foreign: &Program) {
///     output.resources_proof_free(foreign); // no replacement-program API
/// }
/// ```
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::{FactoryOutput, Program};
/// fn escape(output: FactoryOutput) -> &'static Program {
///     output.program() // immutable endpoint cannot escape original output
/// }
/// ```
pub struct FactoryOutput {
    program: Program,
    certificate: ProducedCertificate,
    // Resource provenance only; never digest, WorkBudget or certification input.
    producer: ResourceProducer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResourceProducer {
    GuardedSilu, Negative, EqualZero, Rescale, Scale18, CheckedRange, AllValid,
    CheckedRangeRows,
    RmsCertified, RmsWithInputNorm, Softmax, Rotary,
}

// Named commit scalar slots, separate from handoff, take and output temporaries.
pub(super) struct RmsCommitControls {
    producer: ResourceProducer,
    certificate: ProducedCertificate,
}

pub(super) const fn rms_commit_control_bytes() -> usize {
    std::mem::size_of::<RmsCommitControls>()
        + std::mem::size_of::<ResourceProducer>() // origin selection return
        + std::mem::size_of::<u128>() // squared scalar return
        + std::mem::size_of::<ProducedCertificate>() // certificate construction
        + std::mem::size_of::<&'static mut Option<FactoryOutput>>() // destination argument
        + 2 * std::mem::size_of::<Option<&'static Program>>()
        + std::mem::size_of::<Result<&'static Program, Error>>()
        + 2 * std::mem::size_of::<Option<Program>>() // take return and named taken
        + std::mem::size_of::<Program>() // match binding, before construction
        + std::mem::size_of::<FactoryOutput>() // named construction temporary
        + 2 * std::mem::size_of::<Option<FactoryOutput>>() // Some temporary, resident destination
        + std::mem::size_of::<Result<(), Error>>() // distinct commit return
}

// All refusals precede ownership transfer. The take/assignment gap cannot fail,
// allocate or invoke a hook; the original affine backing moves exactly once.
pub(super) fn commit_rms_output(
    handoff: rms_resident::RmsOutputHandoff<'_>,
    out: &mut Option<FactoryOutput>,
) -> Result<(), Error> {
    if out.is_some() {
        return Err(Error::Binding);
    }
    if handoff.program().is_none() {
        return Err(Error::Phase);
    }
    if !handoff.program().ok_or(Error::Phase)?.extension.absent() {
        return Err(Error::Binding);
    }
    let controls = RmsCommitControls { producer: handoff.producer(),
        certificate: ProducedCertificate::GeneralOutputRowSquared(handoff.squared()) };
    // No closure environment, allocation, check or hook between take/install.
    let taken = handoff.take();
    let program = match taken { Some(program) => program, None => unreachable!() };
    let output = FactoryOutput { program, certificate: controls.certificate, producer: controls.producer };
    *out = Some(output);
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

// Opt-in postconstruction leaf. Constants are logical event bounds, NOT
// funding, permits or a replacement for the public universal resource quote.
pub const RESOURCE_VIEW_QUERY: u64 = 256;
pub const RESOURCE_VIEW_BODY: u64 = 1_198_656;
/// QUERY + BODY only; accessor, initialization and caller controls are separate.
pub const RESOURCE_VIEW_TOTAL: u64 = 1_198_912;
const PROOF_FREE_CONSTRUCTION_MAINTENANCE: u64 = 37_568;
pub const PROOF_FREE_CONSTRUCTION_EVENTS: LogicalEvents = LogicalEvents {
    writes: 1, validation: PROOF_FREE_CONSTRUCTION_MAINTENANCE,
    transitions: 0, allocation_attempts: 0, cleanup_dispatches: 1,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResourcePhase { Ready, Burned, Complete }

/// Opaque original-output borrow. No public constructor, Clone, mutable endpoint
/// or conversion from a foreign Program.
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::{Program, ResourceViewSlots};
/// fn foreign(program: &Program) { let _ = ResourceViewSlots::new(program); }
/// ```
pub struct ProofFreeOutput<'p> { program: &'p Program }
struct ResourceViewQuery<'p> {
    output: &'p FactoryOutput,
    producer: ResourceProducer,
    phase: ResourcePhase,
}
/// One-attempt query controls retaining the original output borrow.
/// Query/body work is intrinsic Core work, not SOURCE authentication. Compiler
/// integration still needs Stage T's genuine original-root loan. Admit controls
/// and pay before each operation; initialization and caller frames are separate.
/// Keep output/backing and its covering charge through postcheck, error and
/// unwind. Retire views/results before output take or capacity shrink.
///
/// ```
/// use pllm_core::puma::multiply::tensor::{Construction, FactoryParameters,
///     WorkPreflightWorkspace, CallerWorkLedger, ResourceViewSlots,
///     ResourceBodySlots, query_resource_view, resources_proof_free};
/// let factory = Construction::AllValid { lanes: 1 }.bind(FactoryParameters::AllValid).unwrap();
/// let ticket = factory.preflight(1, &mut WorkPreflightWorkspace::empty()).unwrap();
/// let mut work = Default::default();
/// let admitted = CallerWorkLedger::new(&mut work, u64::MAX, u64::MAX).unwrap()
///     .admit(ticket).unwrap();
/// let (part, _) = admitted.into_parts();
/// let output = part.build().unwrap();
/// {
///     let mut query = ResourceViewSlots::new(&output);
///     query_resource_view(&mut query).unwrap();
///     let view = query.view().unwrap();
///     let mut body = ResourceBodySlots::empty();
///     let resources = resources_proof_free(view, &mut body).unwrap();
///     assert_eq!(resources, output.program().resources().unwrap()); // postcheck before take
/// }
/// let (_program, _certificate) = output.into_parts();
/// ```
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::ResourceViewSlots;
/// fn private(slots: &ResourceViewSlots<'_>) { let _ = &slots.view; }
/// ```
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::{ResourceBodySlots, ProofFreeOutput};
/// fn private(body: &ResourceBodySlots, view: &ProofFreeOutput<'_>) {
///     let _ = &body.resources;
///     let _ = view.program;
/// }
/// ```
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::{FactoryOutput, ProofFreeOutput,
///     ResourceViewSlots, query_resource_view};
/// fn escape(output: &FactoryOutput) -> &ProofFreeOutput<'_> {
///     let mut slots = ResourceViewSlots::new(output);
///     query_resource_view(&mut slots).unwrap();
///     slots.view().unwrap()
/// }
/// ```
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::{FactoryOutput, ResourceViewSlots,
///     ResourceBodySlots, query_resource_view, resources_proof_free};
/// fn take(output: FactoryOutput) {
///     let mut slots = ResourceViewSlots::new(&output);
///     query_resource_view(&mut slots).unwrap();
///     let view = slots.view().unwrap();
///     let _ = output.into_parts();
///     let _ = resources_proof_free(view, &mut ResourceBodySlots::empty());
/// }
/// ```
pub struct ResourceViewSlots<'p> {
    query: ResourceViewQuery<'p>,
    view: Option<ProofFreeOutput<'p>>,
    result: Result<(), WorkPreflightError>,
}
/// One-attempt body controls. No output identity or funding authority; callers
/// pair the sealed view with their original paid output and retain results through postcheck.
pub struct ResourceBodySlots {
    resources: Option<Resources>,
    result: Result<(), Error>,
    phase: ResourcePhase,
}

impl<'p> ResourceViewSlots<'p> {
    /// Borrow the original output; no classification or descriptor access.
    pub fn new(output: &'p FactoryOutput) -> Self {
        Self { query: ResourceViewQuery { output, producer: output.producer, phase: ResourcePhase::Ready },
            view: None, result: Err(WorkPreflightError::WorkspacePoisoned) }
    }
    /// Borrow only a completed query. Each call has its own separate accessor
    /// tariff/control inventory; it does not execute QUERY or BODY again.
    pub fn view<'s>(&'s self) -> Option<&'s ProofFreeOutput<'p>> {
        if self.query.phase != ResourcePhase::Complete { return None; }
        match &self.view { Some(view) => Some(view), None => None }
    }
}
impl ResourceBodySlots {
    pub const fn empty() -> Self {
        Self { resources: None, result: Err(Error::Phase), phase: ResourcePhase::Ready }
    }
}

pub fn query_resource_view<'p>(slots: &mut ResourceViewSlots<'p>) -> Result<(), WorkPreflightError> {
    // Block 1: one attempt, burned before classifier, fallibility or unwind.
    if slots.query.phase != ResourcePhase::Ready { return Err(WorkPreflightError::WorkspacePoisoned); }
    slots.query.phase = ResourcePhase::Burned;
    #[cfg(test)]
    resource_leaf_hook(ResourceLeafFault::QueryBurn);
    #[cfg(test)]
    resource_leaf_hook(ResourceLeafFault::QueryInstall);
    // Blocks 2/3 have a distinct typed callee return; only immutable closed
    // query fields cross this call. No callback, bind, quote or descriptor walk.
    let returned = query_proof_free_output(&slots.query);
    // Block 4: resident result installation and exclusive failure retirement.
    // No fallible hook between helper return and resident installation.
    match returned {
        Ok(view) => { slots.view = Some(view); slots.result = Ok(()); }
        Err(error) => { slots.result = Err(error); return Err(error); }
    }
    #[cfg(test)]
    resource_leaf_hook(ResourceLeafFault::QueryInstalled);
    slots.query.phase = ResourcePhase::Complete;
    slots.result
}

fn query_proof_free_output<'p>(query: &ResourceViewQuery<'p>)
    -> Result<ProofFreeOutput<'p>, WorkPreflightError> {
    // Block 2: closed producer selection plus call/return transitions.
    let selected = match query.producer {
        ResourceProducer::Negative | ResourceProducer::EqualZero | ResourceProducer::Rescale
        | ResourceProducer::Scale18 | ResourceProducer::CheckedRange | ResourceProducer::AllValid
        | ResourceProducer::RmsCertified | ResourceProducer::RmsWithInputNorm | ResourceProducer::Rotary => true,
        ResourceProducer::GuardedSilu | ResourceProducer::Softmax | ResourceProducer::CheckedRangeRows => false,
    };
    // Block 3: structural absence only; no descriptor, rule or proof access.
    if !selected || !query.output.program.extension.absent() {
        Err(WorkPreflightError::Unsupported)
    } else { Ok(ProofFreeOutput { program: &query.output.program }) }
}

pub fn resources_proof_free(view: &ProofFreeOutput<'_>, slots: &mut ResourceBodySlots) -> Result<Resources, Error> {
    // BODY's first 64: phase entry/burn. This is not a debit or funding proof.
    if slots.phase != ResourcePhase::Ready { return Err(Error::Phase); }
    slots.phase = ResourcePhase::Burned;
    #[cfg(test)]
    resource_leaf_hook(ResourceLeafFault::BodyBurn);
    // B + four maintenance blocks: immutable endpoint entry, validator bypass,
    // table backing selection, checker scratch selection/max. Shared numerical
    // calculation, not a second resource algorithm and no optional checker.
    #[cfg(test)]
    resource_leaf_hook(ResourceLeafFault::BodyInstall);
    let returned = view.program.resource_calculation();
    // BODY's last 64: resident result write and exclusive failure retirement.
    // No fallible hook or callback between returned value and installation.
    match returned {
        Ok(resources) => { slots.resources = Some(resources); slots.result = Ok(()); }
        Err(error) => { slots.result = Err(error); return Err(error); }
    }
    #[cfg(test)]
    resource_leaf_hook(ResourceLeafFault::BodyInstalled);
    slots.phase = ResourcePhase::Complete;
    returned
}

// Two independent 64-event rows, each 32 normal + 32 exclusive refusal.
// Phase gate: receiver/phase read; both operand and formal borrows; Complete
// operand; derived PartialEq::ne/equality-negation typed logical primitive;
// bool transport/branch; early None construction/return and trivial retirement.
// Extraction: Option borrow/discriminant/branch, borrowed match binding,
// Some/None construction/return and trivial retirement. Exclusive constructors
// are counted conservatively. No allocation, loop, descriptor or classifier.
pub const RESOURCE_VIEW_ACCESSOR: u64 = 128;

// Nominal source slots, not ABI stack/RSS. Count receiver, Complete operand,
// comparison operand/formal borrows, equality/ne/branch bool transport, Option
// borrow, match binding, and all three exclusive constructors plus final return.
const fn resource_view_accessor_control_bytes() -> usize {
    std::mem::size_of::<&'static ResourceViewSlots<'static>>()
        + std::mem::size_of::<ResourcePhase>()
        + 4 * std::mem::size_of::<&'static ResourcePhase>()
        + 3 * std::mem::size_of::<bool>()
        + std::mem::size_of::<&'static Option<ProofFreeOutput<'static>>>()
        + std::mem::size_of::<&'static ProofFreeOutput<'static>>()
        + 4 * std::mem::size_of::<Option<&'static ProofFreeOutput<'static>>>()
}

/// Separate accessor controls only; caller-held Option/match, admission and
/// refusal controls remain external, as do retained output/query owners.
pub const RESOURCE_VIEW_ACCESSOR_CONTROL_BYTES: usize = resource_view_accessor_control_bytes();
/// Existing QUERY/BODY control inventory, excluding the separate accessor.
pub const RESOURCE_VIEW_CONTROL_BYTES: usize = resource_view_control_bytes();

// Exact named logical slots, not stack ABI/RSS. Endpoints live inside records;
// Program/FactoryOutput/affine backing/source guards stay in the OLD envelope.
// Helpers' distinct returns coexist with resident query/body/results.
const fn resource_view_control_bytes() -> usize {
    std::mem::size_of::<ResourceViewSlots<'static>>()
        + 2 * std::mem::size_of::<Result<ProofFreeOutput<'static>, WorkPreflightError>>() // helper return, named returned
        + std::mem::size_of::<ProofFreeOutput<'static>>() // match binding
        + std::mem::size_of::<Result<(), WorkPreflightError>>()
        + std::mem::size_of::<WorkPreflightError>() // query failure match binding
        + std::mem::size_of::<Error>() // body failure match binding
        + std::mem::size_of::<Resources>() // body success match binding
        + std::mem::size_of::<bool>()
        + std::mem::size_of::<&'static mut ResourceViewSlots<'static>>()
        + std::mem::size_of::<&'static ResourceViewQuery<'static>>()
        + std::mem::size_of::<&'static ProofFreeOutput<'static>>()
        + std::mem::size_of::<&'static mut ResourceBodySlots>()
        + std::mem::size_of::<ResourceBodySlots>()
        + 3 * std::mem::size_of::<Result<Resources, Error>>() // calculation return, named returned, body return
        + resource_calculation_control_bytes()
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum ResourceLeafFault { QueryBurn, QueryInstall, QueryInstalled, BodyBurn, BodyInstall, BodyInstalled }
#[cfg(test)]
thread_local! {
    static RESOURCE_LEAF_FAULT: std::cell::Cell<Option<ResourceLeafFault>> = const { std::cell::Cell::new(None) };
    static RESOURCE_QUERY_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static RESOURCE_BODY_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
fn resource_leaf_hook(stage: ResourceLeafFault) {
    if stage == ResourceLeafFault::QueryBurn { RESOURCE_QUERY_ENTRIES.with(|n| n.set(n.get() + 1)); }
    if stage == ResourceLeafFault::BodyBurn { RESOURCE_BODY_ENTRIES.with(|n| n.set(n.get() + 1)); }
    RESOURCE_LEAF_FAULT.with(|fault| {
        if fault.get() == Some(stage) { fault.set(None); panic!("injected resource leaf unwind"); }
    });
}

#[cfg(test)]
#[path = "tensor_resource_view_tests.rs"]
mod resource_view_tests;
