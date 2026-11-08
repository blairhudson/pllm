//! Public unsigned coefficient-digit embedding and three-row RNS contraction.
//!
//! No ciphertext/producer authority, keys,
//! backend activation, parameter adoption, or encrypted orchestration. Inputs
//! may represent public c = sigma(c1). The optional Galois composition computes
//! sigma on public buffers; synthetic rows are NOT authenticated evaluation keys.
//! Variable-time clear algebra only. Existing legacy degree gates are unchanged;
//! degree 8192 requires explicitly Public8192-bounded NTT descriptors.
//!
//! Source spectra have shape [3][N], rows [3][2][4][N], embedded digits
//! [3][4][N], and result [2][4][N], all flattened in that order. Components
//! are (b,a); targets are ordered [q0,q1,q2,p]. Source inverse NTT precedes
//! coefficient-canonical UNSIGNED lifting into every target and target forward
//! NTT. Copying source NTT ordinates across moduli is not this operation.

use crate::coefficient_layout::modular::mul;
use crate::coefficient_layout::{LayoutError, RoundedLimbDrop};
use crate::coefficient_ntt::{LogicalOperations, NegacyclicNtt, Operation};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    ContextMismatch,
    BufferLength,
    NonCanonical,
    ArithmeticCapacity,
    WorkLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Evaluation {
    Embed,
    Contract,
}

/// Componentwise conservative logical event bounds, not scalar credits, a
/// resource ledger, physical memory, time, or a complete runtime price. The NTT
/// owner's nested-category semantics apply. Reductions count unsigned coefficient
/// `% target_modulus` events separately from modular multiplications. Checks
/// include preflight, repeated NTT validation, and the finite comparison census
/// documented on `RnsContraction::work`. These are logical predicates, not CPU
/// instructions: loop/iterator termination, index/bounds machinery, boolean
/// short-circuit/branch instructions, unchecked arithmetic, and library-internal
/// implementation details are excluded. Capacity adapters and Result tags are
/// explicitly included below rather than treating them as excluded control.
/// Descriptor construction/primality/root setup is outside this operation quote.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Work {
    pub events: LogicalOperations,
    pub coefficient_reductions: u64,
}

macro_rules! event_fields {
    ($action:ident, $left:expr, $right:expr) => {
        $action!($left, $right;
            checks, residue_checks, moves, swaps, modular_additions,
            modular_subtractions, modular_multiplications, exponentiations,
            exponent_steps, primality_rounds)
    };
}

impl Work {
    fn add_scaled_ntt(&mut self, mut other: LogicalOperations, count: u64) -> Result<(), Error> {
        macro_rules! scale {
            ($left:expr, $right:expr; $($field:ident),*) => {
                $( $left.$field = $left.$field.checked_mul($right)
                    .ok_or(Error::ArithmeticCapacity)?; )*
            };
        }
        event_fields!(scale, other, count);
        self.add_ntt(other)
    }

    fn add_ntt(&mut self, other: LogicalOperations) -> Result<(), Error> {
        macro_rules! add {
            ($left:expr, $right:expr; $($field:ident),*) => {
                $( $left.$field = $left.$field.checked_add($right.$field)
                    .ok_or(Error::ArithmeticCapacity)?; )*
            };
        }
        event_fields!(add, self.events, other);
        Ok(())
    }

    /// No scalarization: each event category must fit its own caller ceiling.
    /// The operation quote includes its ONE preflight call here (10+1 fields),
    /// not arbitrary caller invocations of this method.
    pub fn fits(self, ceiling: Self) -> bool {
        macro_rules! fits {
            ($left:expr, $right:expr; $($field:ident),*) => {
                true $( && $left.$field <= $right.$field )*
            };
        }
        event_fields!(fits, self.events, ceiling.events)
            && self.coefficient_reductions <= ceiling.coefficient_reductions
    }
}

/// Required simultaneously-live u64 payload prefixes, not physical peak memory.
/// Caller-provided excess scratch tails and descriptor storage are excluded.
/// No owned buffers, allocations, retained data, or hidden transaction copies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Storage {
    pub source_words: usize,
    pub row_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    pub total_words: usize,
    pub total_bytes: usize,
}

/// Composition failures preserve every output and scratch word. Constructor
/// errors retain the exact LayoutError, including Overflow, and target index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionError {
    Contraction(Error),
    ArithmeticCapacity,
    BufferLength,
    WorkLimit,
    DropConstruction { limb: usize, error: LayoutError },
}

/// Simultaneously live logical u64 payloads only: 3N source, 24N rows,
/// 6N output, 19N scratch; total 52N words / 416N bytes. Not minimal memory,
/// physical peak, descriptor storage, caller tails, or control funding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionStorage {
    pub source_words: usize,
    pub row_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    pub total_words: usize,
    pub total_bytes: usize,
    spectra_words: usize,
    coefficients_end: usize,
    endpoints: [usize; 9],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DropWork {
    pub checks: u64,
    pub residue_checks: u64,
    pub moves: u64,
    pub modular_subtractions: u64,
    pub modular_multiplications: u64,
    pub unsigned_reductions: u64,
    pub ordinary_subtractions: u64,
    pub ordinary_divisions: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DropConstructionWork {
    pub checks: u64,
    pub descriptor_stores: u64,
    pub euclid_iterations: u64,
    pub signed_divisions: u64,
    pub signed_multiplications: u64,
    pub signed_subtractions: u64,
    pub signed_remainders: u64,
    pub unsigned_remainders: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompositionControlWork {
    pub capacity_outcomes: u64,
    pub adapters: u64,
    pub result_inspections: u64,
    pub ceiling_comparisons: u64,
    pub length_comparisons: u64,
    pub dispatches: u64,
    pub selected_loop_polls: u64,
    pub slice_partitions: u64,
    pub selected_phase_calls: u64,
}

/// Extra logical work beyond the unchanged contraction Work. Nested NTT
/// categories retain their owner's semantics and control exclusions. No scalar
/// tariff, complete CPU inventory, allocation claim, or admission funding.
/// Moves count primitive payload stores, not compiler/control-structure moves.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompositionExtraWork {
    pub inverse_ntts: LogicalOperations,
    pub drop: DropWork,
    pub construction: DropConstructionWork,
    pub control: CompositionControlWork,
}

impl CompositionExtraWork {
    /// Exactly 35 componentwise comparisons on a successful fit. Query work
    /// before operation entry is unpriced; this counts the one internal fit.
    pub fn fits(self, ceiling: Self) -> bool {
        macro_rules! fits {
            ($left:expr, $right:expr; $($field:ident),*) => {
                true $( && $left.$field <= $right.$field )*
            };
        }
        event_fields!(fits, self.inverse_ntts, ceiling.inverse_ntts)
            && fits!(self.drop, ceiling.drop; checks, residue_checks, moves,
                modular_subtractions, modular_multiplications, unsigned_reductions,
                ordinary_subtractions, ordinary_divisions)
            && fits!(self.construction, ceiling.construction; checks,
                descriptor_stores, euclid_iterations, signed_divisions,
                signed_multiplications, signed_subtractions, signed_remainders,
                unsigned_remainders)
            && fits!(self.control, ceiling.control; capacity_outcomes, adapters,
                result_inspections, ceiling_comparisons, length_comparisons,
                dispatches, selected_loop_polls, slice_partitions,
                selected_phase_calls)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompositionWork {
    pub contraction: Work,
    pub extra: CompositionExtraWork,
}

/// Public spectra [2][3][N], component order (c0,c1). Asserted descriptor
/// identity is checked, including roots; this is not ciphertext provenance.
pub struct ComponentSpectra<'a> {
    pub contexts: &'a [NegacyclicNtt; 3],
    pub values: &'a [u64],
}

impl<'a> ComponentSpectra<'a> {
    // Only called after exact full length/capacity checks. Two borrowed helper
    // visits, no copying or implied ownership/provenance transition.
    fn component(&self, component: usize, words: usize) -> SourceSpectra<'_> {
        SourceSpectra {
            contexts: self.contexts,
            values: &self.values[component * words..(component + 1) * words],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GaloisError {
    Composition(CompositionError),
    ContextMismatch,
    InvalidAutomorphism,
    BufferLength,
    NonCanonical,
    ArithmeticCapacity,
    WorkLimit,
}

/// Payload prefixes: input 6N, rows 24N, output 6N, scratch 29N = 65N
/// words / 520N bytes. Scratch: substituted c1 3N, old composition 19N,
/// staged result 6N, original-c0 inverse temporary N. Not physical memory,
/// descriptor/control storage, caller tails, or a minimal-storage claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GaloisStorage {
    pub source_words: usize,
    pub row_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    pub total_words: usize,
    pub total_bytes: usize,
    component_words: usize,
    composition_end: usize,
    result_end: usize,
}

/// Separate logical validation/index/control envelopes. Not scalar funding.
/// These deliberately conservative bounds include wrapper query work and
/// preflight; nested composition and NTT execution retain their own quotes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GaloisControlWork {
    pub validation_checks: u64,
    pub residue_checks: u64,
    pub capacity_outcomes: u64,
    pub adapters: u64,
    pub result_inspections: u64,
    pub ceiling_comparisons: u64,
    pub borrowed_component_visits: u64,
    pub selected_loop_polls: u64,
    pub slice_partitions: u64,
    pub phase_calls: u64,
    pub index_products: u64,
    pub index_additions: u64,
    pub index_remainders: u64,
    pub index_comparisons: u64,
    pub index_subtractions: u64,
    pub index_divisions: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GaloisExtraWork {
    pub automorphisms: LogicalOperations,
    pub inverse_ntts: LogicalOperations,
    /// 3N signed modular additions, each bundling optional negation and
    /// canonical sum correction; 6N final output copies counted separately.
    pub signed_modular_additions: u64,
    pub final_copies: u64,
    /// Duplicate preflight constructors, before any mutable write. The old
    /// composition's own three constructors remain in its unchanged quote.
    pub construction: DropConstructionWork,
    pub control: GaloisControlWork,
}

impl GaloisExtraWork {
    pub fn fits(self, ceiling: Self) -> bool {
        macro_rules! fits {
            ($left:expr, $right:expr; $($field:ident),*) => {
                true $( && $left.$field <= $right.$field )*
            };
        }
        event_fields!(fits, self.automorphisms, ceiling.automorphisms)
            && event_fields!(fits, self.inverse_ntts, ceiling.inverse_ntts)
            && self.signed_modular_additions <= ceiling.signed_modular_additions
            && self.final_copies <= ceiling.final_copies
            && fits!(self.construction, ceiling.construction; checks,
                descriptor_stores, euclid_iterations, signed_divisions,
                signed_multiplications, signed_subtractions, signed_remainders,
                unsigned_remainders)
            && fits!(self.control, ceiling.control; validation_checks,
                residue_checks, capacity_outcomes, adapters, result_inspections,
                ceiling_comparisons, borrowed_component_visits,
                selected_loop_polls, slice_partitions, phase_calls,
                index_products, index_additions, index_remainders,
                index_comparisons, index_subtractions, index_divisions)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GaloisWork {
    pub composition: CompositionWork,
    pub extra: GaloisExtraWork,
}

/// Public source spectra plus their asserted ordered transform contexts.
/// This binding checks descriptor identity, not truth/provenance of supplied data.
pub struct SourceSpectra<'a> {
    pub contexts: &'a [NegacyclicNtt; 3],
    pub values: &'a [u64],
}

/// Public synthetic row spectra, component order (b,a), target order [q0,q1,q2,p].
pub struct RowSpectra<'a> {
    pub contexts: &'a [NegacyclicNtt; 4],
    pub values: &'a [u64],
}

/// Borrowed immutable validated NTT plans. Source order is caller-defined, not
/// numerical sorting. All four target primes must be distinct and same-degree;
/// first three target descriptors must equal corresponding source descriptors,
/// INCLUDING root identity, not merely ring modulus/degree.
pub struct RnsContraction<'a> {
    sources: &'a [NegacyclicNtt; 3],
    targets: &'a [NegacyclicNtt; 4],
}

impl<'a> RnsContraction<'a> {
    pub fn new(
        sources: &'a [NegacyclicNtt; 3],
        targets: &'a [NegacyclicNtt; 4],
    ) -> Result<Self, Error> {
        let degree = sources[0].ring().degree();
        for i in 0..4 {
            if targets[i].ring().degree() != degree
                || (i < 3 && sources[i] != targets[i])
            {
                return Err(Error::ContextMismatch);
            }
            for j in 0..i {
                if targets[i].ring().modulus() == targets[j].ring().modulus() {
                    return Err(Error::ContextMismatch);
                }
            }
        }
        Ok(Self { sources, targets })
    }

    pub fn degree(&self) -> usize {
        self.sources[0].ring().degree()
    }

    pub fn galois_storage(&self) -> Result<GaloisStorage, GaloisError> {
        let old = self.composition_storage().map_err(GaloisError::Composition)?;
        let capacity = GaloisError::ArithmeticCapacity;
        let source_words = old.source_words.checked_mul(2).ok_or(capacity)?;
        let composition_end = old.source_words.checked_add(old.scratch_words).ok_or(capacity)?;
        let result_end = composition_end.checked_add(old.output_words).ok_or(capacity)?;
        let scratch_words = result_end.checked_add(self.degree()).ok_or(capacity)?;
        let total_words = source_words.checked_add(old.row_words).ok_or(capacity)?
            .checked_add(old.output_words).ok_or(capacity)?
            .checked_add(scratch_words).ok_or(capacity)?;
        let total_bytes = total_words.checked_mul(core::mem::size_of::<u64>()).ok_or(capacity)?;
        Ok(GaloisStorage {
            source_words, row_words: old.row_words, output_words: old.output_words,
            scratch_words, total_words, total_bytes, component_words: old.source_words,
            composition_end, result_end,
        })
    }

    /// Old composition plus three spectral automorphisms, three original-c0
    /// inverse NTTs, 3N signed modular additions and 6N final stores. No inverse
    /// power-of-two extraction. Quote construction before entry is unpriced.
    ///
    /// Wrapper control is a selected logical inventory, not all Rust calls,
    /// instructions, a scalar tariff, runtime funding, or physical memory.
    /// Own scans visit 30N residues. Descriptor equality uses <=217 scalar
    /// checks; g has three predicates, lengths four, and the three fits compare
    /// 11+35+46=92 fields. These fit 30N+1024 validation checks. Capacity and
    /// Result handling remain separate: additional storage/query work and
    /// duplicate-constructor handling fit 1024 capacity outcomes, 1024 adapters
    /// and 2048 Result inspections. Two component helper visits are selected.
    ///
    /// Selected phase calls are invocation markers, NOT inclusive body prices.
    /// Complete list, counted only in this wrapper and its earlier queries:
    /// galois_storage and galois_work (2); storage's composition_storage (1);
    /// galois_work's work and composition_extra_work (2); work's scaled calls
    /// (5), operations calls (15), and add_ntt calls (15); extra query's eight
    /// expanded operations calls (8); Galois query's operations calls (6) and
    /// scaled calls (11); the three fits (3); component helpers (2); duplicate
    /// RoundedLimbDrop::new calls and their modular::inverse calls (3+3);
    /// automorphism_into and its validate calls (3+3); contract_drop_into entry
    /// (1); original-c0 inverse_into, transform and validate calls (3+3+3);
    /// final copy_from_slice invocation (1). Total 93 <=128.
    ///
    /// Selected iterator polls count yielded items plus exhaustion if reached.
    /// Complete list: component loop 3, its limb loops 8, input scan loops
    /// 6(N+1), row loop 25, row scans 24(N+1), duplicate-constructor loop 4,
    /// its three Euclid loops <=387, gather limb loop 4, inverse/scatter limb
    /// loop 4, scatter loops 3(N+1), earlier work query's source/target loops
    /// 4+15, and Galois quote context loop 4. Total <=33N+491 <=40N+1024 for
    /// 1<=N<=8192. Fit-field comparisons are NOT polls; macros generate no
    /// traversal loops. NTT-internal loops are excluded from this selection.
    ///
    /// Selected partitions are explicit range views and split calls only:
    /// component helper ranges 2, scan ranges 6+24, scratch prefix and splits
    /// 1+3, gather argument ranges 6, inverse input ranges 3: 45 <=256.
    /// Element indexing/borrowing is not a range partition. Whole-slice
    /// arguments and validation input arrays add no selected partitions.
    ///
    /// Selected index events count explicit address arithmetic only, with NO
    /// implicit range/iterator/enumerate increments. Import spectral gather's
    /// excluded address expression: 6N products and 3N each additions,
    /// remainders, subtractions and divisions. Scatter adds 6N products, 3N
    /// additions/remainders/comparisons and <=3N index subtractions. Signed
    /// value negation/correction stays in signed_modular_additions.
    /// Fixed address arithmetic: two component ranges contribute 4 products
    /// and 2 additions; wrapper 2*N contributes 1 product; six input ranges
    /// contribute 12 products and 6 additions; row target selection contributes
    /// 24 remainders; 24 row ranges contribute 48 products and 24 additions;
    /// six gather ranges contribute 12 products and 6 additions; three inverse
    /// ranges contribute 6 products and 3 additions; three gather 2*N values
    /// contribute 3 products. Thus totals are 12N+86 products, 6N+41 additions,
    /// 6N+24 remainders, 3N comparisons, <=6N subtractions and 3N divisions.
    /// These fit respectively 12N+256, 12N+256, 6N+256, 3N+256, 6N+256 and
    /// 3N+256 for every legal degree. Other validation predicates are not
    /// imported as index comparisons. Query arithmetic is not address work.
    ///
    /// This finite selection excludes getter/accessor calls, casts, local or
    /// default initialization, iterator adapters, implicit counter advances,
    /// bounds-check/compiler machinery and library internals. Only the listed
    /// final copy invocation is selected, not its library body. Checked
    /// arithmetic, conversions, Option/Result adapters, their callbacks and
    /// Result inspections keep their existing categories, not phase-call
    /// markers. Listed scaled closure invocations are selected; macro
    /// invocations are not calls. Categories may overlap and are not summed.
    ///
    /// Repeated earlier queries are real work distinct from nested execution.
    /// Nested composition execution retains its quote once; its internal
    /// queries, constructors and control are not re-imported here. NTT numeric
    /// checks/stores retain their owner; NTT internal index/control exclusions
    /// remain except the explicitly imported gather arithmetic. Descriptor
    /// construction outside operation entry is excluded. Unselected work is
    /// not physically free: complete caller/control/native funding is open.
    /// Preflight runs before ceiling acceptance; this logical inventory creates
    /// neither a prepaid caller nor a ledger/refill or physical-fit guarantee.
    pub fn galois_work(&self) -> Result<GaloisWork, GaloisError> {
        let composition = CompositionWork {
            contraction: self.work(Evaluation::Contract)
                .map_err(|error| GaloisError::Composition(CompositionError::Contraction(error)))?,
            extra: self.composition_extra_work().map_err(GaloisError::Composition)?,
        };
        let n = u64::try_from(self.degree()).map_err(|_| GaloisError::ArithmeticCapacity)?;
        let scaled = |factor: u64, constant: u64| {
            n.checked_mul(factor).and_then(|v| v.checked_add(constant))
                .ok_or(GaloisError::ArithmeticCapacity)
        };
        let mut automorphisms = LogicalOperations::default();
        let mut inverse_ntts = LogicalOperations::default();
        macro_rules! add_fields {
            ($left:expr, $right:expr; $($field:ident),*) => {
                $( $left.$field = $left.$field.checked_add($right.$field)
                    .ok_or(GaloisError::ArithmeticCapacity)?; )*
            };
        }
        for context in self.sources {
            let spectral = context.operations(Operation::Automorphism);
            event_fields!(add_fields, automorphisms, spectral);
            let inverse = context.operations(Operation::Inverse);
            event_fields!(add_fields, inverse_ntts, inverse);
        }
        let control = GaloisControlWork {
            validation_checks: scaled(30, 1024)?,
            residue_checks: scaled(30, 0)?,
            capacity_outcomes: 1024,
            adapters: 1024,
            result_inspections: 2048,
            ceiling_comparisons: 92,
            borrowed_component_visits: 2,
            selected_loop_polls: scaled(40, 1024)?,
            slice_partitions: 256,
            phase_calls: 128,
            index_products: scaled(12, 256)?,
            index_additions: scaled(12, 256)?,
            index_remainders: scaled(6, 256)?,
            index_comparisons: scaled(3, 256)?,
            index_subtractions: scaled(6, 256)?,
            index_divisions: scaled(3, 256)?,
        };
        let extra = GaloisExtraWork {
            automorphisms, inverse_ntts,
            signed_modular_additions: scaled(3, 0)?,
            final_copies: scaled(6, 0)?,
            construction: composition.extra.construction,
            control,
        };
        Ok(GaloisWork { composition, extra })
    }

    /// Public-buffer sigma/key-switch/drop composition, NOT authenticated key
    /// evaluation or encrypted adoption. Input [c0,c1][q0,q1,q2][N], synthetic
    /// row spectra [3][b,a][q0,q1,q2,p][N], coefficient output [b,a][3][N].
    /// Computes (Dp(sum unsigned_digits(sigma(c1))*b_i)+sigma(c0),
    ///           Dp(sum unsigned_digits(sigma(c1))*a_i)).
    /// c0 is added ONLY AFTER rounded drop. Accepted g is odd, 1 <= g < 2N.
    /// Exact input/rows/output lengths; scratch >=29N, excess tail untouched.
    /// All Result refusals preserve complete output AND scratch, including tails.
    /// All nested refusal conditions, capacities and ceilings are discharged
    /// before first store. Expects below express validated invariants, not panic
    /// rollback. Safe borrowing requires mutable buffers disjoint from inputs.
    pub fn galois_contract_drop_into(
        &self,
        source: ComponentSpectra<'_>,
        rows: RowSpectra<'_>,
        g: u64,
        out: &mut [u64],
        scratch: &mut [u64],
        ceiling: GaloisWork,
    ) -> Result<(), GaloisError> {
        let storage = self.galois_storage()?;
        let work = self.galois_work()?;
        let n = self.degree();
        let two_n = 2 * n as u64;
        // Validated descriptor bound N<=8192 proves these public index products
        // fit u64/usize; storage above checks all payload products/endpoints.
        if source.contexts != self.sources || rows.contexts != self.targets {
            return Err(GaloisError::ContextMismatch);
        }
        if g == 0 || g >= two_n || g & 1 == 0 {
            return Err(GaloisError::InvalidAutomorphism);
        }
        if source.values.len() != storage.source_words || rows.values.len() != storage.row_words
            || out.len() != storage.output_words || scratch.len() < storage.scratch_words
        {
            return Err(GaloisError::BufferLength);
        }
        if !work.composition.contraction.fits(ceiling.composition.contraction)
            || !work.composition.extra.fits(ceiling.composition.extra)
            || !work.extra.fits(ceiling.extra)
        {
            return Err(GaloisError::WorkLimit);
        }
        let c0 = source.component(0, storage.component_words);
        let c1 = source.component(1, storage.component_words);
        let mut invalid = false;
        for component in [&c0, &c1] {
            for (limb, context) in self.sources.iter().enumerate() {
                for &value in &component.values[limb * n..(limb + 1) * n] {
                    invalid |= value >= context.ring().modulus();
                }
            }
        }
        for polynomial in 0..24 {
            let q = self.targets[polynomial % 4].ring().modulus();
            for &value in &rows.values[polynomial * n..(polynomial + 1) * n] {
                invalid |= value >= q;
            }
        }
        if invalid { return Err(GaloisError::NonCanonical); }
        // Discharge deterministic constructor failures before mutating scratch.
        // Nested composition repeats these same constructors under same contexts.
        for limb in 0..3 {
            RoundedLimbDrop::new(self.targets[limb].ring(), self.targets[3].ring())
                .map_err(|error| GaloisError::Composition(
                    CompositionError::DropConstruction { limb, error }))?;
        }

        let (prefix, c0_coefficients) = scratch[..storage.scratch_words]
            .split_at_mut(storage.result_end);
        let (prefix, result) = prefix.split_at_mut(storage.composition_end);
        let (substituted, composition_scratch) = prefix.split_at_mut(storage.component_words);
        for limb in 0..3 {
            self.sources[limb].automorphism_into(
                &c1.values[limb * n..(limb + 1) * n], g,
                &mut substituted[limb * n..(limb + 1) * n],
            ).expect("whole-call preflight validated g, contexts, residues and exact buffers");
        }
        self.contract_drop_into(
            SourceSpectra { contexts: self.sources, values: substituted }, rows,
            result, composition_scratch, ceiling.composition,
        ).expect("whole-call preflight discharged every nested composition refusal");
        for limb in 0..3 {
            let context = &self.sources[limb];
            let q = context.ring().modulus();
            context.inverse_into(&c0.values[limb * n..(limb + 1) * n], c0_coefficients)
                .expect("whole-call preflight validated original c0 spectrum and exact temporary");
            for (j, &value) in c0_coefficients.iter().enumerate() {
                let exponent = (j as u64 * g) % two_n;
                let negative = exponent >= n as u64;
                let at = if negative { exponent as usize - n } else { exponent as usize };
                let signed = if negative && value != 0 { q - value } else { value };
                let destination = &mut result[limb * n + at];
                let sum = *destination + signed;
                *destination = if sum >= q { sum - q } else { sum };
            }
        }
        out.copy_from_slice(result);
        Ok(())
    }

    /// Reported logical payload storage only; see CompositionStorage. Checks
    /// every product, total, byte count and reused partition endpoint before
    /// execution. Six word products + three sums + bytes + nine endpoints =
    /// 19 checked capacity outcomes, adapters and Result inspections.
    pub fn composition_storage(&self) -> Result<CompositionStorage, CompositionError> {
        let n = self.degree();
        let source_words = n.checked_mul(3).ok_or(CompositionError::ArithmeticCapacity)?;
        let row_words = n.checked_mul(24).ok_or(CompositionError::ArithmeticCapacity)?;
        let output_words = n.checked_mul(6).ok_or(CompositionError::ArithmeticCapacity)?;
        let spectra_words = n.checked_mul(8).ok_or(CompositionError::ArithmeticCapacity)?;
        let coefficients_end = n.checked_mul(16).ok_or(CompositionError::ArithmeticCapacity)?;
        let scratch_words = n.checked_mul(19).ok_or(CompositionError::ArithmeticCapacity)?;
        let total_words = source_words.checked_add(row_words)
            .ok_or(CompositionError::ArithmeticCapacity)?;
        let total_words = total_words.checked_add(output_words)
            .ok_or(CompositionError::ArithmeticCapacity)?;
        let total_words = total_words.checked_add(scratch_words)
            .ok_or(CompositionError::ArithmeticCapacity)?;
        let total_bytes = total_words.checked_mul(core::mem::size_of::<u64>())
            .ok_or(CompositionError::ArithmeticCapacity)?;
        let endpoints = [
            n.checked_mul(0).ok_or(CompositionError::ArithmeticCapacity)?,
            n.checked_mul(1).ok_or(CompositionError::ArithmeticCapacity)?,
            n.checked_mul(2).ok_or(CompositionError::ArithmeticCapacity)?,
            n.checked_mul(3).ok_or(CompositionError::ArithmeticCapacity)?,
            n.checked_mul(4).ok_or(CompositionError::ArithmeticCapacity)?,
            n.checked_mul(5).ok_or(CompositionError::ArithmeticCapacity)?,
            n.checked_mul(6).ok_or(CompositionError::ArithmeticCapacity)?,
            n.checked_mul(7).ok_or(CompositionError::ArithmeticCapacity)?,
            n.checked_mul(8).ok_or(CompositionError::ArithmeticCapacity)?,
        ];
        Ok(CompositionStorage {
            source_words, row_words, output_words, scratch_words, total_words,
            total_bytes, spectra_words, coefficients_end, endpoints,
        })
    }

    /// Extra logical-only bound, excluding the existing contraction quote.
    /// Eight inverse quotes are summed fieldwise; their owner's control
    /// exclusions remain in force. For B=(N/2)*log2(N), their sums are checks
    /// 8N+16, residues 8N, moves 16N+16B, swaps 8*floor(N/2), additions and
    /// subtractions 8B, products (N==1 ? 0 : 16B+24N).
    ///
    /// Actual additional control census (no wrapper traversal loops):
    /// storage 19 capacities; quote 1 conversion + 80 field additions +
    /// 7 drop products + 2 drop sums + 1 poll product + 1 poll sum = 92.
    /// Thus 111 capacity outcomes, 110 ok_or + one conversion map_err;
    /// add three internal inverse maps, three constructor maps, one contraction
    /// map = 118 adapters. Result inspections: 111 capacity + two outer query
    /// calls + six constructor internal/outer + sixteen inverse validation/
    /// expect + six drop expects = 141. Twelve canonical Result inspections
    /// are already in drop checks, never counted again here.
    /// Dispatches: 24 NTT quote selections + two outer refusal predicates +
    /// three inverse-error matches = 29. Extra fit compares 10+8+8+9 fields.
    /// Selected polls: 18N+405. Slices: two immutable preflight views, one
    /// mutable prefix, two splits, sixteen inverse and eighteen drop = 39.
    /// Selected phase calls: storage + quote + eight NTT quotes + three
    /// constructors + validated contraction + eight inverses + six drops +
    /// three constructor inverses + twelve canonical scans = 43.
    /// The old preflight caller's Result tag remains in its unchanged census.
    /// No panic rollback guarantee; quote computation before entry is unpriced.
    pub fn composition_extra_work(&self) -> Result<CompositionExtraWork, CompositionError> {
        let n = u64::try_from(self.degree())
            .map_err(|_| CompositionError::ArithmeticCapacity)?;
        let mut inverse_ntts = LogicalOperations::default();
        // Macros expand direct statements, not extra helper calls or Result tags.
        macro_rules! add_fields {
            ($left:expr, $right:expr; $($field:ident),*) => {
                $( $left.$field = $left.$field.checked_add($right.$field)
                    .ok_or(CompositionError::ArithmeticCapacity)?; )*
            };
        }
        macro_rules! inverse_quote {
            ($limb:expr) => {{
                let quote = self.targets[$limb].operations(Operation::Inverse);
                event_fields!(add_fields, inverse_ntts, quote);
            }};
        }
        inverse_quote!(0);
        inverse_quote!(1);
        inverse_quote!(2);
        inverse_quote!(3);
        inverse_quote!(0);
        inverse_quote!(1);
        inverse_quote!(2);
        inverse_quote!(3);
        let drop = DropWork {
            checks: n.checked_mul(18).ok_or(CompositionError::ArithmeticCapacity)?
                .checked_add(30).ok_or(CompositionError::ArithmeticCapacity)?,
            residue_checks: n.checked_mul(12).ok_or(CompositionError::ArithmeticCapacity)?,
            moves: n.checked_mul(6).ok_or(CompositionError::ArithmeticCapacity)?,
            modular_subtractions: n.checked_mul(12).ok_or(CompositionError::ArithmeticCapacity)?,
            modular_multiplications: n.checked_mul(6).ok_or(CompositionError::ArithmeticCapacity)?,
            unsigned_reductions: n.checked_mul(6).ok_or(CompositionError::ArithmeticCapacity)?,
            ordinary_subtractions: n.checked_mul(6).ok_or(CompositionError::ArithmeticCapacity)?
                .checked_add(6).ok_or(CompositionError::ArithmeticCapacity)?,
            ordinary_divisions: 6,
        };
        let construction = DropConstructionWork {
            checks: 390,
            descriptor_stores: 15,
            euclid_iterations: 384,
            signed_divisions: 384,
            signed_multiplications: 768,
            signed_subtractions: 768,
            signed_remainders: 3,
            unsigned_remainders: 3,
        };
        let control = CompositionControlWork {
            capacity_outcomes: 111,
            adapters: 118,
            result_inspections: 141,
            ceiling_comparisons: 35,
            length_comparisons: 2,
            dispatches: 29,
            selected_loop_polls: n.checked_mul(18)
                .ok_or(CompositionError::ArithmeticCapacity)?
                .checked_add(405).ok_or(CompositionError::ArithmeticCapacity)?,
            slice_partitions: 39,
            selected_phase_calls: 43,
        };
        Ok(CompositionExtraWork { inverse_ntts, drop, construction, control })
    }

    pub fn storage(&self, evaluation: Evaluation) -> Result<Storage, Error> {
        let n = self.degree();
        let words = |count: usize| n.checked_mul(count).ok_or(Error::ArithmeticCapacity);
        let source_words = words(3)?;
        let (row_words, output_words, scratch_words) = match evaluation {
            Evaluation::Embed => (0, words(12)?, words(2)?),
            Evaluation::Contract => (words(24)?, words(8)?, words(3)?),
        };
        let total_words = source_words.checked_add(row_words)
            .and_then(|v| v.checked_add(output_words))
            .and_then(|v| v.checked_add(scratch_words))
            .ok_or(Error::ArithmeticCapacity)?;
        let total_bytes = total_words.checked_mul(core::mem::size_of::<u64>())
            .ok_or(Error::ArithmeticCapacity)?;
        Ok(Storage {
            source_words, row_words, output_words, scratch_words, total_words, total_bytes,
        })
    }

    /// Per embed/contract call, including the single storage/work computation
    /// inside preflight, not earlier caller queries or `Self::new` setup.
    /// The unchanged 1024 non-residue check envelope has this finite census;
    /// contract maximizes every term (embed and early refusal are bounded by it):
    ///
    /// | Predicate category | Maximum |
    /// | --- | ---: |
    /// | Descriptor scalar equality: 7 * (2 ring + 3 scalar + 2*13 roots) | 217 |
    /// | Input/row/output/scratch length comparisons | 4 |
    /// | One preflight Work::fits: 10 event fields + reductions | 11 |
    /// | Checked capacity outcomes: storage 8 + work 157 | 165 |
    /// | Capacity Option adapters / conversion map_err: 8 + 157 | 165 |
    /// | Result tags: storage 6 + work 157 + helper 15 + preflight 3 + NTT 15 | 196 |
    /// | Aggregate context/length/work/noncanonical refusal predicates | 4 |
    /// | Row Option shape dispatches 4 + storage/work Evaluation selections 2 | 6 |
    /// | NTT quote selections: 15 * (operation match + N==1 + inverse selection) | 45 |
    /// | Total (211 unused checks remain below 1024) | 813 |
    ///
    /// Capacity census from actual calls: storage has four word multiplications,
    /// three total-word additions, one byte multiplication = 8. Its four word
    /// ok_or adapters, two and_then adapters, total ok_or and byte ok_or = 8;
    /// six internal `?` tags propagate four word Results plus total and bytes.
    /// Work has one usize->u64 conversion, FIVE scaled multiplications (residues,
    /// moves, additions, products, reductions), one check-envelope addition, and
    /// 15 add_ntt calls * TEN checked field additions = 157 capacity outcomes.
    /// These have 156 ok_or adapters plus one conversion map_err, and 157
    /// internal `?` tags. The helper calls add 15 Result tags; preflight adds
    /// storage?, work?, then its caller's preflight? = 3. Fifteen NTT expect
    /// tags are counted separately; validated inputs make their failure arms
    /// unreachable. No panic rollback guarantee is provided.
    ///
    /// Own residue comparisons (3N / 27N) and repeated NTT validation
    /// 15*(N+2) are separately added, NEVER charged again to this 813 census.
    /// NTT execution retains its owner's bundled check/correction semantics.
    /// Per-row sum corrections are included in modular_additions, not checks.
    /// RnsContraction::new is not called here; its checks are not operation work.
    pub fn work(&self, evaluation: Evaluation) -> Result<Work, Error> {
        let n = u64::try_from(self.degree()).map_err(|_| Error::ArithmeticCapacity)?;
        let scaled = |count: u64| n.checked_mul(count).ok_or(Error::ArithmeticCapacity);
        let contract = evaluation == Evaluation::Contract;
        let residues = scaled(if contract { 27 } else { 3 })?;
        let mut work = Work {
            events: LogicalOperations {
                // Proven 813-predicate census above fits unchanged 1024 envelope.
                // This is not a scalar tariff or a complete CPU-work inventory.
                checks: residues.checked_add(1024).ok_or(Error::ArithmeticCapacity)?,
                residue_checks: residues,
                // Lift stores, plus result initialization and 24N accumulations.
                moves: scaled(if contract { 44 } else { 12 })?,
                modular_additions: if contract { scaled(24)? } else { 0 },
                modular_multiplications: if contract { scaled(24)? } else { 0 },
                ..LogicalOperations::default()
            },
            coefficient_reductions: scaled(12)?,
        };
        for source in self.sources {
            work.add_ntt(source.operations(Operation::Inverse))?;
            for target in self.targets {
                work.add_ntt(target.operations(Operation::Forward))?;
            }
        }
        Ok(work)
    }

    fn preflight(
        &self,
        source: &SourceSpectra<'_>,
        rows: Option<&RowSpectra<'_>>,
        output: &[u64],
        scratch: &[u64],
        ceiling: Work,
    ) -> Result<Storage, Error> {
        let evaluation = if rows.is_some() { Evaluation::Contract } else { Evaluation::Embed };
        let storage = self.storage(evaluation)?;
        let work = self.work(evaluation)?;
        if source.contexts != self.sources
            || rows.is_some_and(|rows| rows.contexts != self.targets)
        {
            return Err(Error::ContextMismatch);
        }
        if source.values.len() != storage.source_words
            || rows.is_some_and(|rows| rows.values.len() != storage.row_words)
            || output.len() != storage.output_words
            || scratch.len() < storage.scratch_words
        {
            return Err(Error::BufferLength);
        }
        if !work.fits(ceiling) {
            return Err(Error::WorkLimit);
        }
        let n = self.degree();
        let mut invalid = false;
        for (i, context) in self.sources.iter().enumerate() {
            for &value in &source.values[i * n..(i + 1) * n] {
                invalid |= value >= context.ring().modulus();
            }
        }
        if let Some(rows) = rows {
            // Never stop early: include final row, component, limb, coefficient.
            for row in 0..3 {
                for component in 0..2 {
                    for (limb, context) in self.targets.iter().enumerate() {
                        let start = ((row * 2 + component) * 4 + limb) * n;
                        for &value in &rows.values[start..start + n] {
                            invalid |= value >= context.ring().modulus();
                        }
                    }
                }
            }
        }
        if invalid { Err(Error::NonCanonical) } else { Ok(storage) }
    }

    /// d_i is the canonical unsigned coefficient vector of inverse_NTT_qi(c_i).
    /// Output[i][limb] = NTT_target(d_i mod target), not centered or CRT lift.
    /// Exact input/output lengths; scratch >=2N, unused tail unchanged. Every
    /// Result refusal leaves ALL scratch/output words untouched (also tails).
    pub fn embed_into(
        &self,
        source: SourceSpectra<'_>,
        output: &mut [u64],
        scratch: &mut [u64],
        ceiling: Work,
    ) -> Result<(), Error> {
        self.preflight(&source, None, output, scratch, ceiling)?;
        let n = self.degree();
        let (coefficients, lifted) = scratch[..2 * n].split_at_mut(n);
        for i in 0..3 {
            self.sources[i].inverse_into(&source.values[i * n..(i + 1) * n], coefficients)
                .expect("preflight validated source spectrum and exact buffers");
            for (limb, target) in self.targets.iter().enumerate() {
                for (value, &digit) in lifted.iter_mut().zip(coefficients.iter()) {
                    *value = digit % target.ring().modulus();
                }
                let start = (i * 4 + limb) * n;
                target.forward_into(lifted, &mut output[start..start + n])
                    .expect("lift is canonical and buffers are exact");
            }
        }
        Ok(())
    }

    /// (v0,v1) = sum_i d_i*(b_i,a_i) in each target negacyclic ring; output
    /// remains natural-order NTT spectra. Source inverse is performed once per
    /// row, target forward once per row/limb, reused for both components.
    /// Accumulate modulo each target after EACH row. Every product <2^122;
    /// every canonical sum <2^62. No machine-word reconstruction of p*q or Q.
    /// Exact lengths; scratch >=3N, unused tail unchanged. Safe exclusive mutable
    /// borrows require scratch/output disjoint from each other and immutable
    /// inputs. All Result refusals precede ANY scratch/output write; validated
    /// primitive calls cannot subsequently refuse. No rollback allocation.
    pub fn contract_into(
        &self,
        source: SourceSpectra<'_>,
        rows: RowSpectra<'_>,
        output: &mut [u64],
        scratch: &mut [u64],
        ceiling: Work,
    ) -> Result<(), Error> {
        self.preflight(&source, Some(&rows), output, scratch, ceiling)?;
        self.contract_validated_into(source, rows, output, scratch);
        Ok(())
    }

    // Exact former contract_into execution body: no additional validation,
    // arithmetic changes, or fallible return after the caller's preflight.
    fn contract_validated_into(
        &self,
        source: SourceSpectra<'_>,
        rows: RowSpectra<'_>,
        output: &mut [u64],
        scratch: &mut [u64],
    ) {
        let n = self.degree();
        let (coefficients, rest) = scratch[..3 * n].split_at_mut(n);
        let (lifted, spectrum) = rest.split_at_mut(n);
        output.fill(0);
        for i in 0..3 {
            self.sources[i].inverse_into(&source.values[i * n..(i + 1) * n], coefficients)
                .expect("preflight validated source spectrum and exact buffers");
            for (limb, target) in self.targets.iter().enumerate() {
                let q = target.ring().modulus();
                for (value, &digit) in lifted.iter_mut().zip(coefficients.iter()) {
                    *value = digit % q;
                }
                target.forward_into(lifted, spectrum)
                    .expect("lift is canonical and buffers are exact");
                for component in 0..2 {
                    let row_start = ((i * 2 + component) * 4 + limb) * n;
                    let out_start = (component * 4 + limb) * n;
                    for k in 0..n {
                        let product = mul(spectrum[k], rows.values[row_start + k], q);
                        let sum = output[out_start + k] + product;
                        output[out_start + k] = if sum >= q { sum - q } else { sum };
                    }
                }
            }
        }
    }

    /// Contract public spectra, inverse-transform both components under all
    /// four targets, then rounded-drop p into coefficient output [2][3][N].
    /// Components are (b,a), retained targets [q0,q1,q2]; this output is NOT NTT
    /// spectra. The same p coefficient vector is reused for each retained limb.
    /// Source/row views and all authority/profile/degree gates are unchanged.
    ///
    /// Exact 6N output and at least 19N scratch. Scratch is [S:8N,C:8N,W:3N];
    /// all excess tail words remain untouched. Every Result refusal precedes
    /// ANY mutable write, including all three fallible drop constructors in
    /// q0,q1,q2 order. No public prepared token or mutation gap is exposed.
    /// Canonical arithmetic and checked exact slices justify subsequent expects;
    /// this is not a panic-rollback or physical peak-memory guarantee.
    pub fn contract_drop_into(
        &self,
        source: SourceSpectra<'_>,
        rows: RowSpectra<'_>,
        out: &mut [u64],
        scratch: &mut [u64],
        ceiling: CompositionWork,
    ) -> Result<(), CompositionError> {
        // Whole-call preflight: storage/endpoints, quote, lengths, extra fit,
        // exactly one existing preflight, then all constructors, then writes.
        let storage = self.composition_storage()?;
        let extra = self.composition_extra_work()?;
        if out.len() != storage.output_words || scratch.len() < storage.scratch_words {
            return Err(CompositionError::BufferLength);
        }
        if !extra.fits(ceiling.extra) {
            return Err(CompositionError::WorkLimit);
        }
        self.preflight(
            &source,
            Some(&rows),
            &scratch[..storage.spectra_words],
            &scratch[storage.coefficients_end..storage.scratch_words],
            ceiling.contraction,
        ).map_err(CompositionError::Contraction)?;
        let drop0 = RoundedLimbDrop::new(self.targets[0].ring(), self.targets[3].ring())
            .map_err(|error| CompositionError::DropConstruction { limb: 0, error })?;
        let drop1 = RoundedLimbDrop::new(self.targets[1].ring(), self.targets[3].ring())
            .map_err(|error| CompositionError::DropConstruction { limb: 1, error })?;
        let drop2 = RoundedLimbDrop::new(self.targets[2].ring(), self.targets[3].ring())
            .map_err(|error| CompositionError::DropConstruction { limb: 2, error })?;

        let e = storage.endpoints;
        let (spectra_and_coefficients, work) = scratch[..storage.scratch_words]
            .split_at_mut(storage.coefficients_end);
        let (spectra, coefficients) = spectra_and_coefficients
            .split_at_mut(storage.spectra_words);
        self.contract_validated_into(source, rows, spectra, work);
        macro_rules! inverse {
            ($limb:expr, $at:expr) => {
                self.targets[$limb].inverse_into(
                    &spectra[e[$at]..e[$at + 1]],
                    &mut coefficients[e[$at]..e[$at + 1]],
                ).expect("validated contraction produces canonical spectra; buffers are exact");
            };
        }
        inverse!(0, 0);
        inverse!(1, 1);
        inverse!(2, 2);
        inverse!(3, 3);
        inverse!(0, 4);
        inverse!(1, 5);
        inverse!(2, 6);
        inverse!(3, 7);
        macro_rules! project {
            ($drop:expr, $retained:expr, $removed:expr, $output:expr) => {
                $drop.apply_into(
                    &coefficients[e[$retained]..e[$retained + 1]],
                    &coefficients[e[$removed]..e[$removed + 1]],
                    &mut out[e[$output]..e[$output + 1]],
                ).expect("inverse outputs are canonical in matching rings; buffers are exact");
            };
        }
        project!(drop0, 0, 3, 0);
        project!(drop1, 1, 3, 1);
        project!(drop2, 2, 3, 2);
        project!(drop0, 4, 7, 3);
        project!(drop1, 5, 7, 4);
        project!(drop2, 6, 7, 5);
        Ok(())
    }
}

#[cfg(test)]
#[path = "coefficient_rns_contraction/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "coefficient_rns_contraction/galois_tests.rs"]
mod galois_tests;

// Shared unadopted public trace arithmetic; no protected/backend activation.
#[allow(dead_code)]
#[path = "coefficient_rns_contraction/olt_trace.rs"]
mod olt_trace;

#[path = "coefficient_rns_contraction/olt.rs"]
mod olt;
pub use olt::{Olt, OltCensus, OltLhsSpectra, OltStorage, OltWork};
pub use olt_trace::{SwitchBinding as OltSwitchBinding, TraceError as OltError,
    TraceWork as OltTraceWork};

#[cfg(test)]
#[path = "coefficient_rns_contraction/olt_trace_tests.rs"]
mod olt_trace_tests;
