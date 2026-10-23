//! Opt-in closed native construction/work binding. Numeric quotes/debits are
//! not Budget, source/range certificates, runtime permits or request admission.
use super::*;

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
    /// admits resource bytes. Builder is accepted ONLY for closed single-rescale.
    /// Uncovered constructors have a typed refusal, not a guessed work formula.
    pub fn bind(
        self,
        parameters: FactoryParameters<'_>,
    ) -> Result<ClosedFactory<'_>, WorkPreflightError> {
        use FactoryParameters as P;
        let compatible = match (&self, &parameters) {
            (Self::Negative { .. }, P::Negative { .. })
            | (Self::EqualZero { .. }, P::EqualZero { .. })
            | (Self::Builder { .. }, P::Rescale { .. })
            | (Self::CheckedRange { .. }, P::CheckedRange { .. })
            | (Self::AllValid { .. }, P::AllValid)
            | (Self::SoftmaxCertified { .. }, P::SoftmaxCertified { .. }) => true,
            (Self::RmsNormCertified { gamma_len, .. }, P::RmsNormCertified { gamma, .. })
            | (
                Self::RmsNormWithInputNorm { gamma_len, .. },
                P::RmsNormWithInputNorm { gamma, .. },
            ) => *gamma_len == gamma.len(),
            (
                Self::RotaryHalf {
                    cosine_len,
                    sine_len,
                    ..
                },
                P::RotaryHalf { cosine, sine, .. },
            ) => *cosine_len == cosine.len() && *sine_len == sine.len(),
            (
                Self::Negative { .. }
                | Self::EqualZero { .. }
                | Self::Builder { .. }
                | Self::CheckedRange { .. }
                | Self::AllValid { .. }
                | Self::SoftmaxCertified { .. }
                | Self::RmsNormCertified { .. }
                | Self::RmsNormWithInputNorm { .. }
                | Self::RotaryHalf { .. },
                _,
            ) => false,
            _ => return Err(WorkPreflightError::Unsupported),
        };
        if !compatible {
            return Err(WorkPreflightError::ParameterBinding);
        }
        Ok(ClosedFactory {
            construction: self,
            parameters,
        })
    }
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

/// Caller-owned fixed work scratch. Quote resources() BEFORE empty(). Two
/// explicit fixed native metadata stores, no growing buffer or Program storage.
/// Success reuses storage; any error poisons until reset. Reset cannot refund a
/// caller ledger. Public proof CPU/private callframes and caller input backing
/// remain independently unpriced, not zero.
pub struct WorkPreflightWorkspace {
    rms: RmsWorkWorkspace,
    softmax: SoftmaxWorkWorkspace,
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
            poisoned: false,
        }
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub fn reset(&mut self) {
        self.rms.reset();
        self.softmax.clear();
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
    fn price(
        &self,
        workspace: &mut WorkPreflightWorkspace,
    ) -> Result<(WorkBudget, ConstructionQuote), WorkPreflightError> {
        use FactoryParameters as P;
        let work = match (&self.construction, &self.parameters) {
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
        Ok((
            work,
            self.construction
                .quote()
                .map_err(WorkPreflightError::Quote)?,
        ))
    }
    fn build(self) -> Result<FactoryOutput, Error> {
        use FactoryParameters as P;
        let (program, certificate) = match (self.construction, self.parameters) {
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
}

fn preflight<'a>(
    factory: ClosedFactory<'a>,
    count: u64,
    tail: Option<ClosedFactory<'a>>,
    workspace: &mut WorkPreflightWorkspace,
) -> Result<WorkPreflight<'a>, WorkPreflightError> {
    if workspace.poisoned {
        return Err(WorkPreflightError::WorkspacePoisoned);
    }
    let result = (|| {
        let (unit, quote) = factory.price(workspace)?;
        if count == 0 {
            return Err(WorkPreflightError::EmptySchedule);
        }
        let full = Part {
            factory,
            summary: FactoryWorkSummary {
                invocations: count,
                unit_work: unit,
                construction: quote,
            },
        };
        let mut work = unit.checked_mul(count)?;
        let tail = if let Some(factory) = tail {
            let (unit, quote) = factory.price(workspace)?;
            work = work.checked_add(unit)?;
            Some(Part {
                factory,
                summary: FactoryWorkSummary {
                    invocations: 1,
                    unit_work: unit,
                    construction: quote,
                },
            })
        } else {
            None
        };
        let summary = WorkSummary {
            work,
            full: full.summary,
            tail: tail.as_ref().map(|p| p.summary),
        };
        Ok(WorkPreflight {
            full,
            tail,
            summary,
        })
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
        ticket.summary.work.admit(
            self.total.operations(),
            self.total.cursors(),
            self.max_operations,
            self.max_aes,
        )?;
        let combined = self.total.checked_add(ticket.summary.work)?;
        *self.total = combined;
        Ok(AdmittedWorkSchedule {
            full: AdmittedFactory(ticket.full),
            tail: ticket.tail.map(AdmittedFactory),
        })
    }
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
impl AdmittedFactory<'_> {
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
