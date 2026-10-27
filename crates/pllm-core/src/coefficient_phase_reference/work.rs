//! Checked static inventory for THIS fixed clear qualification campaign.
//!
//! Unit: one scalar u64/u128 arithmetic/bit operation, comparison (including
//! an overflow flag or bounds check), or logical payload/control-state write.
//! A loop visit pays advance+termination comparison (2). Pure reads, casts,
//! immutable SSA bindings, borrow/pointer formation and branches are not units.
//! This is NOT instruction/latency/allocator/RSS/atomic-retry accounting.
//! Fixed-wide calls expand below; none is priced as one arbitrary-wide operation.
//! Observer terms cover one full all-cell phase/dense replay, not a second region.
use super::{
    CELLS, CONVERTED_WORDS, DEGREE, Error, MATRIX_WORDS, PHASE_WORDS, SCRATCH_WORDS, SOURCE_WORDS,
};

const LOOP: u64 = 2;
const COPY: u64 = LOOP + 1;
const CMP: u64 = 4 * (2 * LOOP + 1) + 1;
// Four zeroed output words, carry/borrow, four two-overflow operations,
// bool combination, destination/carry writes, loop and final capacity check.
const ADD: u64 = 4 + 1 + 4 * (2 * 2 + 1 + 1 + 1 + LOOP) + 1 + 1;
const SUB: u64 = ADD;
const MUL: u64 = 4 + 1 + 4 * (1 + 1 + 1 + 1 + 1 + LOOP) + 1 + 1;
// checked_shl: four initial writes, four prefix checks/arithmetic, then each
// source word: index/overflow tests, both shifts/ORs/destination writes and
// bounds checks (18 includes both mutually exclusive overflow destinations).
const SHIFT: u64 = 4 + 4 + 4 * (18 + LOOP) + 1;
const HALF: u64 = 4 * (2 + 1 + 1 + 1 + 1 + LOOP) + 1;
const REM: u64 = 1 + 4 * (1 + 1 + 1 + 1 + LOOP) + 1;
const ENCODE: u64 = MUL + 4 + 1;
const ADD_MOD: u64 = 3 * CMP + ADD + SUB + 1;
const INVERSE: u64 = 4 + 128 * (1 + 1 + 2 + 2 + 4 + LOOP) + 4;
const CRT: u64 = 3 * (LOOP + 1) + 1 + 3 + 2 + 2 * MUL + 2 * ADD + REM + 3 + 2 + 8 + 4;
// Four shifts (x*t, q*t, normalized q, normalized numerator), HALF/ADD,
// canonical/wrap tests, top-word u128 estimate, product, two corrections and
// exact final remainder test. Correction iterations price their worst path.
const DOWN: u64 = 4 * SHIFT
    + HALF
    + ADD
    + 2 * CMP
    + 1
    + 2
    + 1
    + 1
    + 1
    + MUL
    + 2 * (LOOP + CMP + 2 + SUB)
    + 2 * CMP
    + SUB
    + 12
    + 3 * (4 + 1);
// Per convolution pair after PUBLIC index-plane construction: cursor, four
// slice bounds, sum/remainder index, wrap check, u128 mul/rem, worst modular
// accumulation (compare/two subtracts), destination write, two control writes.
const PAIR: u64 = LOOP + 4 + 2 + 1 + 2 + 3 + 1 + 2;
// Shared trace source: cursor, i*g/(2N), N doubling, sign comparison/sub,
// output index, modular add, write, three bounds, two control writes.
const AUTOMORPHISM: u64 = LOOP + 2 + 1 + 1 + 3 + 1 + 2 + 1 + 3 + 2;
// Merge source before automorphism: cursor, shift branch/index/negation,
// modular subtract and sum, two writes, four bounds and two control writes.
const MERGE: u64 = LOOP + 1 + 3 + 3 + 2 + 2 + 4 + 2;
// Coordinate iterators, including public min/edge arithmetic, nested cursors,
// fields and destination/index-plane writes. No iterator visit inside PAIR.
const COORDINATE: u64 = 2 * LOOP + 12 + 3 + 3 + 4 + 4;
// Shared InterleaveSchedule: <=256 units for repeated degree/geometry gates,
// power-of-two padding, length arithmetic, descriptor/cursor initialization and
// scratch split checks. Each next/dispatch/slice setup costs <=128 units:
// <=16 arithmetic/comparison operations, <=8 cursor writes, <=6 returned enum
// words, <=8 slice arithmetic/bounds checks, loop/dispatch and terminal paths.
// These are conservative scalar envelopes, not one-unit iterator calls.
// Fixed (8192,64,6): Normalize + 24 Trace + 7 Merge + Finish + terminal next.
const SCHEDULE_SETUP: u64 = 256;
const SCHEDULE_NEXT: u64 = 128;
const SCHEDULE_CALL: u64 = SCHEDULE_SETUP + 34 * SCHEDULE_NEXT;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Term {
    name: &'static str,
    visits: u64,
    units_per_visit: u64,
}
impl Term {
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn visits(&self) -> u64 {
        self.visits
    }
    pub fn units_per_visit(&self) -> u64 {
        self.units_per_visit
    }
    pub fn units(&self) -> Result<u64, Error> {
        self.visits
            .checked_mul(self.units_per_visit)
            .ok_or(Error::Overflow)
    }
}
const TERMS: usize = 40;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Inventory {
    terms: [Term; TERMS],
    len: usize,
    total: u64,
}
fn times(values: &[u64]) -> Result<u64, Error> {
    values
        .iter()
        .try_fold(1_u64, |a, &b| a.checked_mul(b).ok_or(Error::Overflow))
}
fn sum(values: &[u64]) -> Result<u64, Error> {
    values
        .iter()
        .try_fold(0_u64, |a, &b| a.checked_add(b).ok_or(Error::Overflow))
}
impl Inventory {
    fn push(&mut self, name: &'static str, visits: u64, units_per_visit: u64) -> Result<(), Error> {
        let term = Term {
            name,
            visits,
            units_per_visit,
        };
        self.total = self
            .total
            .checked_add(term.units()?)
            .ok_or(Error::Overflow)?;
        *self.terms.get_mut(self.len).ok_or(Error::Capacity)? = term;
        self.len = self.len.checked_add(1).ok_or(Error::Overflow)?;
        Ok(())
    }
    pub fn terms(&self) -> &[Term] {
        &self.terms[..self.len]
    }
    pub fn total(&self) -> u64 {
        self.total
    }
    pub(super) fn convolution_pairs(&self) -> u64 {
        self.terms[16].visits
    }
    pub(super) fn qualification() -> Result<Self, Error> {
        let mut out = Self {
            terms: [Term {
                name: "",
                visits: 0,
                units_per_visit: 0,
            }; TERMS],
            len: 0,
            total: 0,
        };
        let n = DEGREE as u64;
        let cells = CELLS as u64;
        let planes = times(&[2, 3])?; // directions * limbs
        let blocks = times(&[planes, 3, 2, 2])?;
        let lhs_visits = times(&[planes, 3, 2, 65])?;
        let rhs_visits = times(&[planes, 3, 65, 129])?;
        let pairs = times(&[
            planes,
            3,
            sum(&[times(&[64, 64, 128])?, times(&[64, 64])?, 128, 1])?,
        ])?;
        // Thirteen quote constructions in the retained-output/refusal observer:
        // first evaluate(3), output quote(1), late failed evaluate(3), short
        // quote(1), short evaluate(2), invalid-mask evaluation(3). A push pays two checked arithmetics,
        // bounds + four field-word writes, checked cursor/write and Result (13).
        // Four initializer words per Term. Constructor expressions have at most
        // fourteen times/sum calls per term (helper observer is maximal), each
        // with <=16 scalar arguments. Each fold visit pays checked arithmetic
        // (2), cursor (2), and state write (1). One extra term covers preamble.
        // The final 128 are Quote's field products/sums and <=TERMS find visits.
        out.push(
            "quote/predebit construction",
            3 + 1 + 3 + 1 + 2 + 3,
            sum(&[
                times(&[TERMS as u64, 13 + 4])?,
                times(&[TERMS as u64 + 1, 14, 16, 2 + LOOP + 1])?,
                128,
            ])?,
        )?;
        // Native admission/state: six source capacity checks, four Vec factories
        // (allocation hook + exact-capacity check + resize), two budget reserves,
        // one payload-first retirement, one Work checked debit, six wrapper
        // states, four Result transitions. Each state has <=8 scalar checks/writes.
        out.push(
            "native counter/owner state",
            sum(&[6, times(&[4, 3])?, 2, 1, 1, 6, 4])?,
            8,
        )?;
        // One execution Rns3 and one observer Rns3; each has two bounded Euclid
        // inverses, two wide products and one wide word-remainder.
        out.push(
            "Rns3 constructors",
            2,
            sum(&[
                times(&[2, INVERSE])?,
                times(&[2, MUL])?,
                REM,
                3 * (LOOP + 4) + 4 + 2 + 4 + 1 + 2 + 13 + 1,
            ])?,
        )?;
        // q bound; coordinate gates/shape loops/ceil blocks/field writes;
        // exact tuple equality and admitted descriptor/control transition.
        out.push(
            "geometry constructor/checks",
            1,
            4 + SHIFT
                + CMP
                + (4 + 1
                    + 1
                    + 2 * 3 * (LOOP + 1)
                    + 4
                    + 1
                    + 2 * 2
                    + 1
                    + 3 * (2 + LOOP)
                    + 3 * (4 + 1)
                    + 2
                    + 10
                    + 1)
                + (4 * (LOOP + 1) + 1)
                + 8,
        )?;
        out.push("length/error repeated validations", 4, 6 * 4 + 4)?;
        out.push(
            "full mask repeated validation",
            times(&[2, 2, n])?,
            CMP + LOOP,
        )?;
        out.push("buffer capacity checks", 4, 4)?;
        out.push(
            "workspace and output initialization",
            sum(&[
                SCRATCH_WORDS as u64,
                PHASE_WORDS as u64,
                CONVERTED_WORDS as u64,
                MATRIX_WORDS as u64,
            ])?,
            COPY,
        )?;
        out.push(
            "supplied source fixture construction",
            SOURCE_WORDS as u64,
            LOOP + 7,
        )?;
        out.push(
            "supplied mask fixture construction",
            times(&[2, n])?,
            ENCODE + SUB + HALF + 4 + LOOP + 8,
        )?;
        out.push("group clears", times(&[planes, 6, n])?, COPY)?;
        out.push(
            "two packed polynomial clears",
            times(&[blocks, 2, n])?,
            COPY,
        )?;
        out.push(
            "lhs coordinates/U/index plane",
            lhs_visits,
            COORDINATE + ENCODE + REM + 1,
        )?;
        out.push(
            "rhs coordinates/residue/index plane",
            rhs_visits,
            COORDINATE + 1,
        )?;
        out.push(
            "coordinate iterator setup/outer cursor",
            times(&[blocks, 2])?,
            2 * LOOP + 12,
        )?;
        out.push("convolution lhs cursor/value", lhs_visits, LOOP + 4)?;
        out.push("convolution pairs", pairs, PAIR)?;
        out.push("trace validation", times(&[planes, 6, n])?, LOOP + 1)?;
        out.push(
            "trace inverse-power-two",
            times(&[planes, 6])?,
            LOOP + 2 + 4 + 1,
        )?;
        out.push("trace zero-padded scratch", times(&[planes, 8, n])?, COPY)?;
        out.push(
            "trace inverse scaling",
            times(&[planes, 6, n])?,
            LOOP + 2 + 1 + 2,
        )?;
        out.push("partial trace copies", times(&[planes, 8, 3, n])?, COPY)?;
        out.push(
            "all 31 automorphisms",
            times(&[planes, 31, n])?,
            AUTOMORPHISM,
        )?;
        out.push("seven merge prepasses", times(&[planes, 7, n])?, MERGE)?;
        out.push("trace final copy", times(&[planes, n])?, COPY)?;
        out.push(
            "trace schedule construction/iterator/control",
            planes,
            SCHEDULE_CALL,
        )?;
        out.push(
            "public visible/tail coordinate predicate",
            times(&[2, n])?,
            24 + LOOP,
        )?;
        out.push("global visible CRT", times(&[2, cells])?, CRT + 3 * 4)?;
        out.push(
            "all masks plus visible result Down",
            times(&[2, sum(&[n, cells])?])?,
            DOWN,
        )?;
        out.push("visible phase+mask", times(&[2, cells])?, ADD_MOD)?;
        out.push("conversion/output/index writes", times(&[2, n])?, 14)?;
        out.push(
            "ParseMat addresses and cross writes",
            times(&[2, cells])?,
            24 + 2 * 8,
        )?;
        out.push(
            "local complete dots/final cross adds",
            times(&[2, cells])?,
            65 * (LOOP + 10) + 2 * 10 + 8,
        )?;
        // Observer unchanged: independent dense cells + whole-phase replay and
        // residual identity. Three RNS replay terms per h use u128, not BigInt.
        let replay = sum(&[
            LOOP,
            2 + 6 + 2 + 1,
            MUL,
            4,
            MUL,
            SUB,
            ADD_MOD,
            times(&[3, REM + 2 + 3 + 2 + 3 + LOOP + 1])?,
            8,
        ])?;
        out.push(
            "observer dense/residual/phase replay",
            times(&[2, cells, 65])?,
            replay,
        )?;
        out.push(
            "observer cell CRT/identity/assertions",
            times(&[2, cells])?,
            CRT + SHIFT + MUL + ADD_MOD + 3 * 8 + 24,
        )?;
        out.push(
            "observer full padding/phase scans",
            PHASE_WORDS as u64,
            LOOP + 3 + 5 + 2 * (4 + 1),
        )?;
        out.push(
            "observer final independent dense cells",
            cells,
            65 * (LOOP + 12) + 12,
        )?;
        // Cursor/setup checks in the finite loops not charged per coefficient:
        // execution (72 inner blocks, 36 output blocks, 6 planes, 2 directions),
        // observer (774 cells), and source/mask construction (6 vectors).
        out.push(
            "outer loop/control state",
            sum(&[blocks, 36, planes, 2, 2 * cells, 6, TERMS as u64])?,
            24,
        )?;
        // Registered helper tests: three small degrees, four strides and
        // sum(1,2,4,8)=15 partial counts. Bound every group by padded=8,
        // depth=3, merge=7; add two legacy n32 comparisons, 256 divider replay
        // cases, ten constructor/boundary cases and a full visible-map scan.
        let small_trace = sum(&[
            times(&[8 + 16 + 32, 15, 31, AUTOMORPHISM])?,
            times(&[8 + 16 + 32, 15, 7, MERGE])?,
            times(&[8 + 16 + 32, 15, 8 + 6 + 24 + 1, COPY])?,
            times(&[2, 32, 31, AUTOMORPHISM + MERGE + COPY])?,
            // Existing 45 tiny calls plus two legacy/extracted comparisons.
            // Their schedules are shorter than the fixed 34-next envelope.
            times(&[3 * 15 + 2, SCHEDULE_CALL])?,
        ])?;
        let slow_down = sum(&[
            times(&[65, CMP + SUB + HALF + LOOP + 2])?,
            times(&[2, SHIFT])?,
            ADD,
            HALF,
            CMP,
        ])?;
        out.push(
            "registered finite helper observers",
            1,
            sum(&[
                small_trace,
                times(&[256, DOWN + slow_down + HALF + ADD + 12])?,
                times(&[10, 2 * INVERSE + 2 * MUL + REM + 64])?,
                times(&[n, 40])?,
                times(&[3 * 5 * 7 + 3 * 32, COORDINATE + 16])?,
            ])?,
        )?;
        // The second native test never runs a second region: one unknown-error
        // early refusal, then invalid last mask before workspace factories.
        // Price its supplied-source/mask initialization, scan, two Rns3
        // constructors and original counter/Result checks nonetheless.
        out.push(
            "native early-refusal observer",
            1,
            sum(&[
                times(&[sum(&[SOURCE_WORDS as u64, times(&[2, n, 4])?])?, COPY])?,
                times(&[n, CMP + LOOP])?,
                times(&[
                    2,
                    2 * INVERSE + 2 * MUL + REM + (3 * (LOOP + 4) + 4 + 2 + 4 + 1 + 2 + 13 + 1),
                ])?,
                2 * 6 * 4 + 16 * 8,
            ])?,
        )?;
        Ok(out)
    }
}
