//! One bounded PUBLIC coefficient OLT interleave group, with borrowed spectra.
//! Not encryption/key birth, issuer provenance, security/parameter adoption,
//! funding, protected activation, or a client-phase interface. Variable time.
//! RHS is unscaled ring-2^64 words: sign (if present) wraps BEFORE reduction.
//! All coefficient output is retained, including synthetic switch-row errors.
//!
//! Scratch is exactly (6P+51)N: existing trace lanes and 47N workspace plus
//! packed RHS, reduced RHS, RHS spectrum, and product/inverse temporary (4N).
//! Products are accumulated directly into trace lanes in the spectral domain;
//! only after all inner blocks are those lanes inverse transformed in place
//! through the temporary. No product Down, hidden staging, allocation or cache.
//! Original borrowed NTT descriptor construction/root/primality setup is outside
//! the operation quote. No default primes, roots, or security profile are chosen.
use super::*;
use super::olt_trace::{add, bump, times, usize_sum, usize_words, SwitchBinding,
    TraceError, TracePlan, TraceWork};
use crate::coefficient_layout::coordinates::Coordinates;

/// Exact [rowBlock][innerBlock][c0,c1][q0,q1,q2][N] spectral payload.
pub struct OltLhsSpectra<'a> {
    pub contexts: &'a [NegacyclicNtt; 3],
    pub values: &'a [u64],
}

/// Accessed prefixes. Row payload counts UNIQUE EXPONENT bindings, not scheduled
/// references; aliasing among immutable slices can only reduce unique backing.
/// Descriptor/control storage is excluded; this is not physical memory admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OltStorage {
    pub lhs_words: usize,
    pub rhs_words: usize,
    pub unique_row_words: usize,
    pub scheduled_row_reference_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    pub total_words: usize,
    pub total_bytes: usize,
    pub padded: usize,
    pub switches: usize,
    pub unique_exponents: usize,
}

/// Full supplied slice lengths, distinct from accessed prefixes. Immutable row
/// aliases are conservatively counted per unique-exponent binding, NOT deduced
/// as disjoint backing. Includes output/scratch excess tails, never accesses them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OltCensus {
    pub lhs_words: usize,
    pub rhs_words: usize,
    pub row_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    pub total_words: usize,
    pub total_bytes: usize,
}

/// Fieldwise owner composition. `trace` retains all existing trace/Galois array
/// category order and semantics, including normalization/merge/copy work and
/// three rounded-drop constructor preflights. `outer` reuses Work/NTT categories.
/// Explicit extra categories are conservative logical event bounds, not CPU
/// counts or scalar fees. Invocation counters do not replace nested body work.
/// `control`/`indices` cover incremental outer work ONLY; `trace` retains its
/// own mode-independent quote. Included: entry queries, full census, field fits,
/// descriptor checks, coordinate walks, checked offsets/bytes, adapters, Result
/// tags, slicing and loop polls. Geometry/constructor work outside apply is
/// excluded. NTT
/// internal control retains its owner's exclusions; no physical timing proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OltWork {
    pub trace: TraceWork,
    pub outer: Work,
    /// All wrapper forwards, including the six forwards before each switch;
    /// excludes forwards nested inside Galois (already in trace.galois).
    pub forward_ntts: u64,
    pub rhs_forward_ntts: u64,
    /// Product-lane inverses only; Galois inverse work stays in trace.galois.
    pub inverse_ntts: u64,
    /// N-word polynomial product passes, not coefficient multiplication count.
    pub pointwise_products: u64,
    pub packing_stores: u64,
    pub ring_negations: u64,
    pub coordinate_visits: u64,
    pub lane_copies: u64,
    pub spectral_accumulations: u64,
    pub control: u64,
    pub indices: u64,
}

impl OltWork {
    pub fn fits(self, ceiling: Self) -> bool {
        self.trace.fits(&ceiling.trace) && self.outer.fits(ceiling.outer)
            && self.forward_ntts <= ceiling.forward_ntts
            && self.rhs_forward_ntts <= ceiling.rhs_forward_ntts
            && self.inverse_ntts <= ceiling.inverse_ntts
            && self.pointwise_products <= ceiling.pointwise_products
            && self.packing_stores <= ceiling.packing_stores
            && self.ring_negations <= ceiling.ring_negations
            && self.coordinate_visits <= ceiling.coordinate_visits
            && self.lane_copies <= ceiling.lane_copies
            && self.spectral_accumulations <= ceiling.spectral_accumulations
            && self.control <= ceiling.control && self.indices <= ceiling.indices
    }
}

// Same SOURCE boundary and joint evaluator skeleton as olt_trace::trace_bounds;
// its local/reached-owner table includes OLT. B=K*innerBlocks,
// A=rowBlocks*innerBlocks, V=rowBlocks*rhs_words; N,K,E as in trace.
// Outer entry 1541+Sany(6A,N), query extra 3837+7E,
// execution excluding packing 12+65K+97B+51BN, RHS union 21B+27V
// give Base=5397+7E+72A+30AN+65K+118B+51BN+27V.
// RHS finite unit 17B+23V minus caller loop 4B+3V gives constructor/body
// 13B+20V; add complete execution packing 8B+7V. Execution owns its exclusive
// operation/expect. No RHS double charge. Payload indices
// 18BN+65B+38K+6A+7V plus query 91+E give outer indices.
// One extra trace evaluator occurrence belongs here (314), not inside the
// nested trace quote: OLT reaches two, trace owns one. Fold into 5397+314=5711.
// Outer self evaluation has 8 control+7 index ordinary terms, three checked
// products A,AN,BN (9 events): 13*15+18+9=222; final constant bump is CONTROL
// ONLY. No loops, term arrays, extra helpers/products or arbitrary headroom.
fn outer_bounds(n: u64, k: u64, b: u64, v: u64, e: u128,
    rows: u128, inner: u128) -> Result<(u64, u64), TraceError>
{
    let (n, k, b, v) = (n as u128, k as u128, b as u128, v as u128);
    let a = rows.checked_mul(inner).ok_or(TraceError::Capacity)?;
    let an = a.checked_mul(n).ok_or(TraceError::Capacity)?;
    let bn = b.checked_mul(n).ok_or(TraceError::Capacity)?;
    let mut control = 0;
    bump(&mut control, 1, 5711)?;
    bump(&mut control, e, 7)?;
    bump(&mut control, a, 72)?;
    bump(&mut control, an, 30)?;
    bump(&mut control, k, 65)?;
    bump(&mut control, b, 118)?;
    bump(&mut control, bn, 51)?;
    bump(&mut control, v, 27)?;
    let mut indices = 0;
    bump(&mut indices, 1, 91)?;
    bump(&mut indices, e, 1)?;
    bump(&mut indices, a, 6)?;
    bump(&mut indices, k, 38)?;
    bump(&mut indices, b, 65)?;
    bump(&mut indices, bn, 18)?;
    bump(&mut indices, v, 7)?;
    bump(&mut control, 1, 222)?;
    let control = u64::try_from(control).map_err(|_| TraceError::Capacity)?;
    let indices = u64::try_from(indices).map_err(|_| TraceError::Capacity)?;
    Ok((control, indices))
}

pub struct Olt<'a> {
    contraction: &'a RnsContraction<'a>,
    coordinates: Coordinates,
    shape: [usize; 3],
    blocks: [usize; 3],
    count: usize,
    trace: TracePlan<'a>,
}

impl<'a> Olt<'a> {
    /// Geometry is parameterized; exactly ONE group, rowBlocks*columnBlocks<=mw.
    /// Context prefix identity/order/distinctness is inherited from the borrowed
    /// RnsContraction constructor, not inferred from data or scalar moduli.
    pub fn new(contraction: &'a RnsContraction<'a>, shape: [usize; 3], window: [usize; 3])
        -> Result<Self, TraceError>
    {
        let coordinates = Coordinates::bounded(contraction.degree(), shape, window, 8192)
            .map_err(TraceError::Geometry)?;
        let blocks = core::array::from_fn(|i| shape[i].div_ceil(window[i]));
        let count = usize_words(blocks[0], blocks[2])?;
        let trace = TracePlan::new(contraction, window[1], count)?;
        let result = Self { contraction, coordinates, shape, blocks, count, trace };
        result.storage()?;
        result.work()?;
        Ok(result)
    }

    pub fn required_exponent_count(&self) -> usize {
        self.trace.unique_exponent_count()
    }

    /// Each level in 0..log2(mw) names N/2^level+1. For mw=1 there
    /// are no valid levels and rows MUST be empty. No shift/divide by zero.
    pub fn required_exponent(&self, level: usize) -> Result<u64, TraceError> {
        self.trace.unique_exponent(level)
    }

    pub fn storage(&self) -> Result<OltStorage, TraceError> {
        let old = self.trace.storage()?;
        let n = self.contraction.degree();
        let lhs_words = usize_words(usize_words(self.blocks[0], self.blocks[1])?, old.output_words)?;
        let rhs_words = usize_words(self.shape[1], self.shape[2])?;
        let unique_exponents = self.required_exponent_count();
        let unique_row_words = usize_words(usize_words(24, unique_exponents)?, n)?;
        let scratch_words = usize_sum(old.scratch_words, usize_words(4, n)?)?;
        let total_words = usize_sum(usize_sum(lhs_words, rhs_words)?,
            usize_sum(unique_row_words, usize_sum(old.output_words, scratch_words)?)?)?;
        let total_bytes = usize_words(total_words, core::mem::size_of::<u64>())?;
        Ok(OltStorage { lhs_words, rhs_words, unique_row_words,
            scheduled_row_reference_words: old.row_words, output_words: old.output_words,
            scratch_words, total_words, total_bytes, padded: old.padded,
            switches: old.switches, unique_exponents })
    }

    pub fn census(&self, lhs: &OltLhsSpectra<'_>, rhs: &[u64], rows: &[SwitchBinding<'_>],
        output: &[u64], scratch: &[u64]) -> Result<OltCensus, TraceError>
    {
        let mut row_words = 0;
        for row in rows { row_words = usize_sum(row_words, row.rows.values.len())?; }
        let total_words = usize_sum(usize_sum(lhs.values.len(), rhs.len())?,
            usize_sum(row_words, usize_sum(output.len(), scratch.len())?)?)?;
        Ok(OltCensus { lhs_words: lhs.values.len(), rhs_words: rhs.len(), row_words,
            output_words: output.len(), scratch_words: scratch.len(), total_words,
            total_bytes: usize_words(total_words, core::mem::size_of::<u64>())? })
    }

    pub fn work(&self) -> Result<OltWork, TraceError> {
        let st = self.storage()?;
        let convert = |value| u64::try_from(value).map_err(|_| TraceError::Capacity);
        let n = convert(self.contraction.degree())?;
        let k = convert(self.count)?;
        let inner = convert(self.blocks[1])?;
        let pairs = times(k, inner)?;
        let rhs_forward_ntts = times(3, pairs)?;
        let inverse_ntts = times(6, k)?;
        let pointwise_products = times(6, pairs)?;
        let mut outer = Work::default();
        for context in self.contraction.sources {
            outer.add_scaled_ntt(context.operations(Operation::Forward), pairs)
                .map_err(|_| TraceError::Capacity)?;
            outer.add_scaled_ntt(context.operations(Operation::Inverse), times(2, k)?)
                .map_err(|_| TraceError::Capacity)?;
        }
        // Each output lane repacks its column block, even across row blocks.
        let coordinate_visits = times(convert(self.blocks[0])?, convert(st.rhs_words)?)?;
        let packing_stores = add(times(n, pairs)?, coordinate_visits)?;
        let lane_copies = times(times(6, k)?, n)?;
        let spectral_accumulations = times(pointwise_products, n)?;
        let reductions = times(times(3, pairs)?, n)?;
        let stores = add(add(convert(usize_words(usize_words(6, st.padded)?, self.contraction.degree())?)?,
            packing_stores)?, add(reductions, add(times(2, spectral_accumulations)?, lane_copies)?)?)?;
        outer.add_ntt(LogicalOperations {
            checks: convert(st.lhs_words)?, residue_checks: convert(st.lhs_words)?,
            moves: stores, modular_multiplications: spectral_accumulations,
            modular_additions: spectral_accumulations, ..LogicalOperations::default()
        }).map_err(|_| TraceError::Capacity)?;
        outer.coefficient_reductions = reductions;
        let trace = self.trace.work()?;
        let forward_ntts = add(rhs_forward_ntts, times(6, trace.switches)?)?;
        let (control, indices) = outer_bounds(n, k, pairs, coordinate_visits,
            st.unique_exponents as u128, self.blocks[0] as u128, self.blocks[1] as u128)?;
        Ok(OltWork { trace, outer, forward_ntts, rhs_forward_ntts, inverse_ntts, pointwise_products,
            packing_stores, ring_negations: coordinate_visits, coordinate_visits,
            lane_copies, spectral_accumulations, control, indices })
    }

    /// All returned refusals preserve FULL output/scratch (including tails).
    /// Immutable borrowed inputs cannot change between preflight and execution.
    /// No panic rollback promise. RHS accepts every u64, including MAX; neither
    /// qi-based rejection, centering, signed casts nor plaintext-U lift applies.
    pub fn apply(&self, lhs: OltLhsSpectra<'_>, rhs: &[u64], rows: &[SwitchBinding<'_>],
        output: &mut [u64], scratch: &mut [u64], ceiling: OltWork) -> Result<(), TraceError>
    {
        let st = self.storage()?;
        let work = self.work()?;
        if lhs.contexts != self.contraction.sources { return Err(TraceError::Context); }
        if lhs.values.len() != st.lhs_words || rhs.len() != st.rhs_words
            || rows.len() != st.unique_exponents
            || output.len() < st.output_words || scratch.len() < st.scratch_words
        { return Err(TraceError::Length); }
        self.census(&lhs, rhs, rows, output, scratch)?;
        if !work.fits(ceiling) { return Err(TraceError::WorkLimit); }
        let n = self.contraction.degree();
        for (poly, values) in lhs.values.chunks_exact(n).enumerate() {
            let q = self.contraction.sources[poly % 3].ring().modulus();
            if values.iter().any(|&value| value >= q) { return Err(TraceError::NonCanonical); }
        }
        // Exact unique inventory and all canonical row words, descriptor roots
        // and target order, nested quote/offset capacity, and all THREE drop
        // constructors are checked before even clearing a scratch lane.
        let old = self.trace.storage()?;
        self.trace.preflight(None, rows, true, output.len(), old.scratch_words, work.trace)?;
        // Checked aggregate products above dominate every subsequent offset:
        // source=((rb*innerBlocks+ib)*6+poly)*N+j < lhs_words;
        // destination=(lane*6+poly)*N+j < 6*K*N <= old.lanes.
        // Coordinates checked all shape products and window volume<=N, so its
        // matrix/coefficient offsets fit rhs_words/N. Each residue is <q<2^61;
        // additions of two canonical residues cannot overflow u64. No fallible
        // returned path remains after these borrows and the first store below.
        let (trace_scratch, extra) = scratch[..st.scratch_words].split_at_mut(old.scratch_words);
        let (packed, extra) = extra.split_at_mut(n);
        let (reduced, extra) = extra.split_at_mut(n);
        let (spectrum, temporary) = extra.split_at_mut(n);
        let lanes = &mut trace_scratch[..old.lanes];
        lanes.fill(0);
        for lane in 0..self.count {
            let row_block = lane / self.blocks[2];
            let column_block = lane % self.blocks[2];
            for inner_block in 0..self.blocks[1] {
                packed.fill(0);
                for coordinate in self.coordinates.rhs(inner_block, column_block)
                    .expect("bounded geometry and checked block loop validate RHS coordinates")
                {
                    let value = rhs[coordinate.matrix];
                    packed[coordinate.coefficient] = if coordinate.negative {
                        value.wrapping_neg()
                    } else { value };
                }
                for limb in 0..3 {
                    let context = &self.contraction.sources[limb];
                    let q = context.ring().modulus();
                    for j in 0..n { reduced[j] = packed[j] % q; }
                    context.forward_into(reduced, spectrum)
                        .expect("preflight exact buffers; unsigned reductions canonicalize RHS");
                    for component in 0..2 {
                        let poly = component * 3 + limb;
                        let source = ((row_block * self.blocks[1] + inner_block) * 6 + poly) * n;
                        let destination = (lane * 6 + poly) * n;
                        for j in 0..n {
                            temporary[j] = mul(lhs.values[source + j], spectrum[j], q);
                            let sum = lanes[destination + j] + temporary[j];
                            lanes[destination + j] = if sum >= q { sum - q } else { sum };
                        }
                    }
                }
            }
            // Do NOT apply rounded limb drop to products. All inner products
            // have accumulated spectrally before this coefficient transition.
            for poly in 0..6 {
                let at = (lane * 6 + poly) * n;
                self.contraction.sources[poly % 3].inverse_into(&lanes[at..at + n], temporary)
                    .expect("preflight exact buffers and canonical modular spectral accumulation");
                lanes[at..at + n].copy_from_slice(temporary);
            }
        }
        self.trace.execute_populated(None, rows, true, output, trace_scratch);
        Ok(())
    }
}
