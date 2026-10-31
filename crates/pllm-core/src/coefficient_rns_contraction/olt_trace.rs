//! Bounded PUBLIC variable-time two-component trace/merge, not an OLT executor.
//! Synthetic rows assert no key provenance. Coefficients enter and leave as
//! coefficients; every switch receives explicitly forward-transformed spectra.
//! No allocation, authority, resource owner, parameter default, or backend gate.
//!
//! For k inputs, p=next_power_of_two(k), s=stride, d=log2(s/p),
//! C=p*d+p-1 switches (including padded lanes), M=p-1 merges:
//! caller payloads are 6kN input, 24CN row references, 6N result, and
//! (6p+47)N scratch = lanes 6pN + difference 6N + spectra 6N + switched
//! 6N + Galois scratch 29N. Total bound N*(6k+24C+6p+53) words.
//! Row references may alias; this is a conservative referenced-payload inventory,
//! NOT unique backing/capacity accounting or physical/encrypted-region admission.
//! Original NTT descriptors (three source and four target), C binding records,
//! plan/cursor/work/control values and caller tails are additionally live, NOT
//! included in word totals. No descriptor/owner/control admission is claimed.
//!
//! Six-lane N8192/stride64 fixture: p=8,C=31, scratch=95N words;
//! total referenced payload <=881N words =57,737,216 bytes. This does not
//! establish either 52N or 32MiB as an encrypted-region budget.
use super::*;
use crate::coefficient_layout::schedule::{InterleaveSchedule, InterleaveStep};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TraceError {
    Geometry(LayoutError),
    Galois(GaloisError),
    Context,
    Exponent,
    Length,
    NonCanonical,
    Capacity,
    WorkLimit,
}

/// One immutable binding for EACH scheduled call, in schedule order. Repeated
/// exponents may borrow the same original row slice. No owned or mutable token.
pub(super) struct SwitchBinding<'a> {
    pub exponent: u64,
    pub rows: RowSpectra<'a>,
}

pub(super) struct Coefficients<'a> {
    pub contexts: &'a [NegacyclicNtt; 3],
    /// [lane][c0,c1][q0,q1,q2][coefficient], NOT spectra.
    pub values: &'a [u64],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TraceStorage {
    pub input_words: usize,
    pub row_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    pub total_words: usize,
    pub total_bytes: usize,
    pub switches: usize,
    pub padded: usize,
    lanes: usize,
}

/// Fieldwise nested quote: Galois fields retain EXACT parent order/semantics;
/// forward fields retain LogicalOperations order. No scalarized work allowance.
/// `wrapper` counts coefficient arithmetic/stores/canonical scans only.
/// `control` bounds wrapper query/preflight, descriptor comparisons, fit visits,
/// checked arithmetic/adapters, dispatches, loop polls, slices and Result tags;
/// `indices` separately bounds wrapper indexing (not NTT-internal indexing).
/// Both are logical conservative envelopes, not instruction/time tariffs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TraceWork {
    pub switches: u64,
    pub galois: [u64; 92],
    pub forwards: [u64; 10],
    pub wrapper: [u64; 10],
    /// One duplicate three-drop preflight, outside the C nested switches.
    pub preflight_construction: [u64; 8],
    pub control: u64,
    pub indices: u64,
}

impl TraceWork {
    pub fn fits(&self, ceiling: &Self) -> bool {
        self.switches <= ceiling.switches
            && self.galois.iter().zip(ceiling.galois).all(|(&a, b)| a <= b)
            && self.forwards.iter().zip(ceiling.forwards).all(|(&a, b)| a <= b)
            && self.wrapper.iter().zip(ceiling.wrapper).all(|(&a, b)| a <= b)
            && self.preflight_construction.iter().zip(ceiling.preflight_construction)
                .all(|(&a, b)| a <= b)
            && self.control <= ceiling.control && self.indices <= ceiling.indices
    }
}

fn ntt_fields(v: LogicalOperations) -> [u64; 10] {
    [v.checks, v.residue_checks, v.moves, v.swaps, v.modular_additions,
        v.modular_subtractions, v.modular_multiplications, v.exponentiations,
        v.exponent_steps, v.primality_rounds]
}

fn galois_fields(v: GaloisWork) -> [u64; 92] {
    let mut out = [0; 92];
    let mut at = 0;
    macro_rules! append {
        ($v:expr; $($field:ident),+) => { $(out[at] = $v.$field; at += 1;)+ };
    }
    macro_rules! ntt {
        ($v:expr) => { for value in ntt_fields($v) { out[at] = value; at += 1; } };
    }
    macro_rules! construction {
        ($v:expr) => { append!($v; checks, descriptor_stores, euclid_iterations,
            signed_divisions, signed_multiplications, signed_subtractions,
            signed_remainders, unsigned_remainders); };
    }
    ntt!(v.composition.contraction.events);
    append!(v.composition.contraction; coefficient_reductions);
    ntt!(v.composition.extra.inverse_ntts);
    append!(v.composition.extra.drop; checks, residue_checks, moves,
        modular_subtractions, modular_multiplications, unsigned_reductions,
        ordinary_subtractions, ordinary_divisions);
    construction!(v.composition.extra.construction);
    append!(v.composition.extra.control; capacity_outcomes, adapters,
        result_inspections, ceiling_comparisons, length_comparisons, dispatches,
        selected_loop_polls, slice_partitions, selected_phase_calls);
    ntt!(v.extra.automorphisms);
    ntt!(v.extra.inverse_ntts);
    append!(v.extra; signed_modular_additions, final_copies);
    construction!(v.extra.construction);
    append!(v.extra.control; validation_checks, residue_checks, capacity_outcomes,
        adapters, result_inspections, ceiling_comparisons, borrowed_component_visits,
        selected_loop_polls, slice_partitions, phase_calls, index_products,
        index_additions, index_remainders, index_comparisons, index_subtractions,
        index_divisions);
    debug_assert_eq!(at, out.len());
    out
}

pub(super) fn add(a: u64, b: u64) -> Result<u64, TraceError> {
    a.checked_add(b).ok_or(TraceError::Capacity)
}
pub(super) fn times(a: u64, b: u64) -> Result<u64, TraceError> {
    a.checked_mul(b).ok_or(TraceError::Capacity)
}
pub(super) fn usize_words(a: usize, b: usize) -> Result<usize, TraceError> {
    a.checked_mul(b).ok_or(TraceError::Capacity)
}
pub(super) fn usize_sum(a: usize, b: usize) -> Result<usize, TraceError> {
    a.checked_add(b).ok_or(TraceError::Capacity)
}

pub(super) struct TracePlan<'a> {
    contraction: &'a RnsContraction<'a>,
    schedule: InterleaveSchedule,
    count: usize,
    stride: usize,
}

impl<'a> TracePlan<'a> {
    pub fn new(contraction: &'a RnsContraction<'a>, stride: usize, count: usize)
        -> Result<Self, TraceError>
    {
        let schedule = InterleaveSchedule::bounded(contraction.degree(), stride, count, 8192)
            .map_err(TraceError::Geometry)?;
        Ok(Self { contraction, schedule, count, stride })
    }

    pub fn storage(&self) -> Result<TraceStorage, TraceError> {
        let n = self.contraction.degree();
        let padded = self.schedule.padded();
        let depth = (self.stride / padded).trailing_zeros() as usize;
        let switches = usize_sum(usize_words(padded, depth)?, padded - 1)?;
        let input_words = usize_words(usize_words(6, self.count)?, n)?;
        let row_words = usize_words(usize_words(24, switches)?, n)?;
        let output_words = usize_words(6, n)?;
        let lanes = usize_words(usize_words(6, padded)?, n)?;
        let scratch_words = usize_sum(lanes, usize_words(47, n)?)?;
        let total_words = usize_sum(usize_sum(input_words, row_words)?,
            usize_sum(output_words, scratch_words)?)?;
        let total_bytes = usize_words(total_words, core::mem::size_of::<u64>())?;
        Ok(TraceStorage { input_words, row_words, output_words, scratch_words,
            total_words, total_bytes, switches, padded, lanes })
    }

    /// Q = C*GaloisWork + C*2*sum_q Forward(q) + wrapper + one three-drop
    /// preflight construction quote. All products and
    /// fieldwise sums checked. No caller ceiling is refreshed by an inner call.
    /// Query before entry is outside operation; entry's repeated queries fit
    /// wrapper control below. Nested scans/constructors remain in C*GaloisWork.
    /// Control <=65536*(C+p+k+1)+256*N*(C+k+p+1): each switch has <2048
    /// wrapper query/fit/descriptor/dispatch events and <128 per-coefficient
    /// events; remaining factors cover normalization, full preflight scans,
    /// iteration/index/slice machinery and quote arithmetic. Generous margins
    /// deliberately avoid a claim of minimal or exact hardware work.
    pub fn work(&self) -> Result<TraceWork, TraceError> {
        let st = self.storage()?;
        let c = u64::try_from(st.switches).map_err(|_| TraceError::Capacity)?;
        let p = u64::try_from(st.padded).map_err(|_| TraceError::Capacity)?;
        let k = u64::try_from(self.count).map_err(|_| TraceError::Capacity)?;
        let n = u64::try_from(self.contraction.degree()).map_err(|_| TraceError::Capacity)?;
        let g = self.contraction.galois_work().map_err(TraceError::Galois)?;
        let mut galois = galois_fields(g);
        for field in &mut galois { *field = times(*field, c)?; }
        let mut forwards = [0; 10];
        for context in self.contraction.sources {
            for (total, field) in forwards.iter_mut().zip(ntt_fields(context.operations(Operation::Forward))) {
                *total = add(*total, times(field, times(2, c)?)?)?;
            }
        }
        let residues = times(n, add(times(6, k)?, times(24, c)?)?)?;
        let m = p - 1;
        let wrapper = ntt_fields(LogicalOperations {
            checks: residues,
            residue_checks: residues,
            moves: times(n, add(add(times(6, p)?, times(6, k)?)?, add(times(18, c)?, 6)?)?)?,
            modular_additions: times(times(6, n)?, add(c, m)?)?,
            modular_subtractions: times(times(12, n)?, m)?,
            modular_multiplications: add(times(times(6, n)?, k)?,
                times(3, u64::from(self.stride.trailing_zeros()))?)?,
            ..LogicalOperations::default()
        });
        let units = add(add(c, p)?, add(k, 1)?)?;
        let control = add(times(65536, units)?, times(times(256, n)?, units)?)?;
        let indices = times(times(256, n)?, units)?;
        let construction = g.composition.extra.construction;
        let preflight_construction = [construction.checks, construction.descriptor_stores,
            construction.euclid_iterations, construction.signed_divisions,
            construction.signed_multiplications, construction.signed_subtractions,
            construction.signed_remainders, construction.unsigned_remainders];
        Ok(TraceWork { switches: c, galois, forwards, wrapper, preflight_construction,
            control, indices })
    }

    /// Exact input/binding count; output and scratch accept excess untouched tails.
    /// Every Result refusal precedes ALL writes. Safe borrows keep original contexts,
    /// exponents and rows immutable/disjoint from caller mutable workspaces.
    /// No panic rollback promise. NTT descriptor construction is outside quote;
    /// repeated rounded-drop construction is explicitly included.
    pub fn apply(&self, input: Coefficients<'_>, bindings: &[SwitchBinding<'_>],
        output: &mut [u64], scratch: &mut [u64], ceiling: TraceWork)
        -> Result<(), TraceError>
    {
        let st = self.storage()?;
        let work = self.work()?;
        let plan = self.contraction;
        let gwork = plan.galois_work().map_err(TraceError::Galois)?;
        let gst = plan.galois_storage().map_err(TraceError::Galois)?;
        if input.contexts != plan.sources { return Err(TraceError::Context); }
        if input.values.len() != st.input_words || bindings.len() != st.switches
            || output.len() < st.output_words || scratch.len() < st.scratch_words
        { return Err(TraceError::Length); }
        if !work.fits(&ceiling) { return Err(TraceError::WorkLimit); }
        let n = plan.degree();
        let mut invalid = false;
        for (poly, coefficients) in input.values.chunks_exact(n).enumerate() {
            let q = plan.sources[poly % 3].ring().modulus();
            for &value in coefficients { invalid |= value >= q; }
        }
        let mut call = 0;
        for step in self.schedule.steps() {
            let exponent = match step {
                InterleaveStep::Trace { element, .. } | InterleaveStep::Merge { element, .. } => element,
                _ => continue,
            };
            let binding = &bindings[call];
            if binding.exponent != exponent as u64 { return Err(TraceError::Exponent); }
            if binding.rows.contexts != plan.targets { return Err(TraceError::Context); }
            if binding.rows.values.len() != gst.row_words { return Err(TraceError::Length); }
            for (poly, coefficients) in binding.rows.values.chunks_exact(n).enumerate() {
                let q = plan.targets[poly % 4].ring().modulus();
                for &value in coefficients { invalid |= value >= q; }
            }
            call += 1;
        }
        if invalid { return Err(TraceError::NonCanonical); }
        // Discharge the nested deterministic drop-constructor refusals once,
        // before normalization writes. The same immutable contexts are retained.
        for limb in 0..3 {
            RoundedLimbDrop::new(plan.targets[limb].ring(), plan.targets[3].ring())
                .map_err(|error| TraceError::Galois(GaloisError::Composition(
                    CompositionError::DropConstruction { limb, error })))?;
        }
        let width = st.output_words;
        let (lanes, temporary) = scratch[..st.scratch_words].split_at_mut(st.lanes);
        let (difference, temporary) = temporary.split_at_mut(width);
        let (spectra, temporary) = temporary.split_at_mut(width);
        let (switched, switch_scratch) = temporary.split_at_mut(width);
        let mut call = 0;
        for step in self.schedule.steps() {
            let lane = match step {
                InterleaveStep::Normalize { divisor, .. } => {
                    lanes.fill(0);
                    for limb in 0..3 {
                        let q = plan.sources[limb].ring().modulus();
                        let mut inverse = 1;
                        for _ in 0..divisor.trailing_zeros() { inverse = mul(inverse, q.div_ceil(2), q); }
                        for lane in 0..self.count {
                            for component in 0..2 {
                                let at = lane * width + (component * 3 + limb) * n;
                                for j in 0..n { lanes[at + j] = mul(input.values[at + j], inverse, q); }
                            }
                        }
                    }
                    continue;
                }
                InterleaveStep::Trace { lane, .. } => {
                    difference.copy_from_slice(&lanes[lane * width..(lane + 1) * width]);
                    lane
                }
                InterleaveStep::Merge { left, right, shift, .. } => {
                    // Both old coefficients read before left is overwritten.
                    for poly in 0..6 {
                        let q = plan.sources[poly % 3].ring().modulus();
                        for j in 0..n {
                            let b = lanes[right * width + poly * n + (j + n - shift) % n];
                            let b = if j < shift && b != 0 { q - b } else { b };
                            let at = left * width + poly * n + j;
                            let a = lanes[at];
                            difference[poly * n + j] = if a >= b { a - b } else { q - (b - a) };
                            let sum = a + b;
                            lanes[at] = if sum >= q { sum - q } else { sum };
                        }
                    }
                    left
                }
                InterleaveStep::Finish => {
                    output[..width].copy_from_slice(&lanes[..width]);
                    continue;
                }
            };
            for poly in 0..6 {
                plan.sources[poly % 3].forward_into(&difference[poly * n..(poly + 1) * n],
                    &mut spectra[poly * n..(poly + 1) * n])
                    .expect("whole trace preflight and canonical transitions validate forward NTT");
            }
            let binding = &bindings[call];
            // Local quote is a fixed subquote of the admitted C-fold total,
            // never a refreshed caller allowance. Exactly C calls execute.
            plan.galois_contract_drop_into(ComponentSpectra { contexts: plan.sources, values: spectra },
                RowSpectra { contexts: binding.rows.contexts, values: binding.rows.values },
                binding.exponent, switched, switch_scratch, gwork)
                .expect("whole trace preflight discharged every nested refusal");
            for poly in 0..6 {
                let q = plan.sources[poly % 3].ring().modulus();
                for j in 0..n {
                    let at = lane * width + poly * n + j;
                    let sum = lanes[at] + switched[poly * n + j];
                    lanes[at] = if sum >= q { sum - q } else { sum };
                }
            }
            call += 1;
        }
        Ok(())
    }
}
