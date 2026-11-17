//! Independent checked-u128 literal source oracle. Production rows are read only
//! as actual assertion operands, never to construct expected subtotals/layouts.
//! Traces below simulate logical primitives, not intercepted std execution.
use crate::owned_backing::retirement as owner;
use super::super::{ChildRow, Error, LogicalRetirementQuote,
    LogicalRetirementWork};
use std::{alloc::Layout, array::IntoIter, convert::Infallible, mem::{align_of, align_of_val, size_of, size_of_val},
    num::TryFromIntError, ops::ControlFlow, slice::Iter};

type E = Error;
type X = TryFromIntError;
type W = LogicalRetirementWork;
type Q = LogicalRetirementQuote;
type C = ChildRow;
type P = (String, Vec<usize>);
type Seq = Vec<P>;
type I = Iter<'static, P>;
type Z = ();
type WR = &'static mut W;

fn plus(a: u128, b: u128) -> u128 { a.checked_add(b).expect("oracle addition") }
fn product(a: u128, b: u128) -> u128 { a.checked_mul(b).expect("oracle multiplication") }
fn sz<T>() -> u128 { size_of::<T>() as u128 }
// Each ledger slot records its real type's size AND alignment independently.
fn ty<T>() -> (u128, usize) { (size_of::<T>() as u128, align_of::<T>()) }
fn total(c: [u128; 3]) -> u128 { plus(plus(c[0], c[1]), c[2]) }

#[derive(Clone, Debug)]
struct Audit {
    counts: [u128; 3],
    bytes: u128,
    max: u128,
    max_align: usize,
    // Each leaf is a call or a complete typed logical control family.
    trace: Vec<[u128; 3]>,
}
fn aggregate(parts: impl IntoIterator<Item = Audit>) -> Audit {
    let mut out = Audit { counts: [0; 3], bytes: 0, max: 0, max_align: 1, trace: vec![] };
    for part in parts {
        for (sum, value) in out.counts.iter_mut().zip(part.counts) { *sum = plus(*sum, value); }
        out.bytes = plus(out.bytes, part.bytes);
        out.max = out.max.max(part.max);
        out.max_align = out.max_align.max(part.max_align);
        out.trace.extend(part.trace);
    }
    out
}
fn literal(part: Audit, n: usize) -> Audit {
    let mut out = aggregate((0..n).map(|_| part.clone()));
    out.max = part.max; out.max_align = part.max_align;
    out
}
fn iteration(part: Audit, n: usize) -> Audit {
    let bytes = part.bytes;
    let max = part.max;
    let max_align = part.max_align;
    let mut out = literal(part, n);
    out.bytes = bytes; out.max = max; out.max_align = max_align;
    out
}
fn typed(types: &[(u128, usize)]) -> (u128, u128, usize) {
    (types.iter().map(|t| t.0).fold(0, plus), types.iter().map(|t| t.0).max().unwrap_or(0),
        types.iter().map(|t| t.1).max().unwrap_or(1))
}
fn invocation(types: &[(u128, usize)]) -> Audit {
    let (pool, max, max_align) = typed(types);
    let counts = [1, 0, product(2, pool)];
    Audit { counts, bytes: pool, max, max_align, trace: vec![counts] }
}
fn controls(types: &[(u128, usize)], events: u128) -> Audit {
    let (pool, max, max_align) = typed(types);
    let counts = [0, events, product(product(2, pool), plus(events, 1))];
    Audit { counts, bytes: pool, max, max_align, trace: vec![counts] }
}
fn primitive<A, T>() -> Audit {
    aggregate([invocation(&[ty::<A>(), ty::<A>(), ty::<T>(), ty::<T>()]),
        controls(&[ty::<A>(), ty::<T>()], 2)])
}
fn error_types<T, F>() -> [(u128, usize); 5] {
    [ty::<T>(), ty::<Option<T>>(), ty::<Result<T, F>>(),
        ty::<ControlFlow<Result<Infallible, F>, T>>(), ty::<Result<Infallible, F>>()]
}
fn interface<A, T, F>() -> Audit {
    aggregate([invocation(&[ty::<A>(), ty::<A>(), ty::<Result<T, F>>(), ty::<Result<T, F>>()]),
        controls(&error_types::<T, F>(), 5)])
}
fn write<T>(n: usize) -> Audit { literal(controls(&[ty::<T>()], 1), n) }
fn returned<T>() -> Audit { controls(&[ty::<T>(), ty::<Result<T, E>>()], 2) }
fn branch() -> Audit { controls(&[ty::<bool>()], 1) }
fn inline_operation<T>() -> Audit {
    aggregate([primitive::<(T, T), Option<T>>(), write::<E>(1),
        primitive::<(Option<T>, E), Result<T, E>>()])
}
fn add64() -> Audit {
    aggregate([interface::<(u64, u64), u64, E>(), inline_operation::<u64>(), returned::<u64>()])
}
fn multiply64() -> Audit {
    aggregate([interface::<(u64, u64), u64, E>(), inline_operation::<u64>(), returned::<u64>()])
}
fn add_bytes() -> Audit {
    aggregate([interface::<(usize, usize), usize, E>(), inline_operation::<usize>(), returned::<usize>()])
}
fn array_bytes<T>() -> Audit {
    let _element_size = size_of::<T>();
    aggregate([interface::<(usize,), usize, E>(), inline_operation::<usize>(), returned::<usize>()])
}
fn convert_count() -> Audit {
    aggregate([
        interface::<(usize,), u64, E>(), interface::<(usize,), u64, X>(),
        primitive::<(Result<u64, X>, Z), Result<u64, E>>(), write::<Z>(1),
        primitive::<(X,), E>(), write::<E>(1), returned::<u64>(),
    ])
}
fn allocation(nonzero: bool) -> Audit {
    let mut rows = vec![interface::<(usize, usize), (usize, usize), E>(),
        literal(primitive::<(usize, usize), bool>(), 2), primitive::<(bool, bool), bool>(), branch()];
    if nonzero {
        rows.push(inline_operation::<usize>());
        rows.push(controls(&error_types::<usize, E>(), 5));
    }
    rows.extend([write::<(usize, usize)>(1), returned::<(usize, usize)>()]);
    aggregate(rows)
}
fn inspection() -> Audit {
    aggregate([interface::<(W,), u64, E>(), literal(add64(), 2), returned::<u64>()])
}
fn fold<const N: usize>() -> Audit {
    aggregate([
        interface::<([u64; N],), u64, E>(), write::<[u64; N]>(1),
        primitive::<([u64; N],), IntoIter<u64, N>>(), write::<IntoIter<u64, N>>(1),
        write::<u64>(1), write::<Z>(1),
        interface::<(&'static mut IntoIter<u64, N>, u64, Z), u64, E>(),
        iteration(controls(&[ty::<IntoIter<u64, N>>(), ty::<bool>()], 2), N.checked_add(1).unwrap()),
        iteration(aggregate([
            primitive::<(&'static mut Z, u64, u64), Result<u64, E>>(), add64(),
            controls(&error_types::<u64, E>(), 5), write::<u64>(1),
        ]), N), returned::<u64>(),
    ])
}
fn work_call() -> Audit {
    aggregate([interface::<(WR, u64, u64), (), E>(), literal(add64(), 3), literal(multiply64(), 2),
        primitive::<(u64, u64), bool>(), branch(), write::<u64>(3), returned::<()>()])
}
fn work_control() -> Audit {
    aggregate([interface::<(WR, u64, u64, u64), (), E>(), literal(add64(), 4), literal(multiply64(), 4),
        primitive::<(u64, u64), bool>(), branch(), write::<u64>(3), returned::<()>()])
}
fn pool<A, T>() -> Audit {
    let _actual_generic_sizes = (size_of::<A>(), size_of::<T>());
    aggregate([interface::<(), u64, E>(), literal(convert_count(), 2), add64(), multiply64(), returned::<u64>()])
}
fn child_control() -> Audit {
    aggregate([interface::<(), u64, E>(), literal(convert_count(), 4), multiply64(), fold::<4>(), returned::<u64>()])
}
fn string_row() -> Audit {
    aggregate([interface::<(), C, E>(), pool::<&'static mut String, ()>(), child_control(), write::<C>(1), returned::<C>()])
}
fn vec_row() -> Audit {
    aggregate([interface::<(), C, E>(), pool::<&'static mut Vec<usize>, ()>(), child_control(), write::<C>(1), returned::<C>()])
}
fn append() -> Audit {
    aggregate([interface::<(C, WR, u64), (), E>(), work_call(), work_control(), convert_count(),
        multiply64(), add64(), write::<u64>(1), returned::<()>()])
}
fn release() -> Audit {
    aggregate([interface::<(WR, u64), (), E>(), pool::<(*mut u8, Layout), ()>(), work_call(),
        literal(convert_count(), 3), fold::<3>(), work_control(), returned::<()>()])
}
fn default_work() -> Audit { aggregate([primitive::<(), W>(), write::<u64>(4), write::<W>(1)]) }
fn child_quote() -> Audit {
    aggregate([interface::<(C, usize, usize), Q, E>(), default_work(), append(), convert_count(),
        release(), inspection(), add_bytes(), write::<W>(1), write::<Q>(1), returned::<Q>()])
}
fn fixed_work() -> Audit {
    aggregate([
        interface::<(usize, usize), W, E>(), literal(convert_count(), 8), inline_operation::<usize>(),
        controls(&error_types::<usize, E>(), 5), default_work(),
        pool::<&'static mut Seq, ()>(), pool::<(&'static mut [P], usize), Option<&'static mut P>>(),
        pool::<&'static mut P, ()>(), literal(work_call(), 3), literal(work_control(), 3),
        literal(fold::<3>(), 2), literal(multiply64(), 2), literal(add64(), 2),
        string_row(), vec_row(), literal(append(), 2), release(), inspection(),
        write::<(u64, u64, u64, u64, W)>(1), write::<u64>(1), returned::<W>(),
    ])
}
fn entry(nonzero: bool) -> Audit {
    aggregate([interface::<(&'static Seq,), Q, E>(), literal(primitive::<(&'static Seq,), usize>(), 2),
        array_bytes::<P>(), allocation(nonzero), primitive::<(&'static Seq,), I>(),
        write::<I>(1), write::<(usize, usize)>(1)])
}
fn next() -> Audit {
    aggregate([primitive::<(&'static mut I,), Option<&'static P>>(),
        controls(&[ty::<Option<&'static P>>(), ty::<bool>()], 2)])
}
fn pair(name_nonzero: bool, dims_nonzero: bool) -> Audit {
    aggregate([
        write::<(&'static String, &'static Vec<usize>)>(1), primitive::<(&'static String,), usize>(),
        literal(primitive::<(&'static Vec<usize>,), usize>(), 2),
        allocation(name_nonzero), allocation(dims_nonzero), array_bytes::<usize>(),
        literal(add_bytes(), 4), write::<(usize, usize, usize, usize)>(1), write::<usize>(2),
    ])
}
fn tail() -> Audit {
    aggregate([literal(primitive::<(&'static Seq,), usize>(), 2), add_bytes(), write::<Q>(1), returned::<Q>()])
}
fn bound_entry() -> Audit {
    aggregate([
        interface::<(usize,), W, E>(), primitive::<(usize, usize), Option<usize>>(), write::<Z>(1),
        primitive::<(Option<usize>, Z), Option<usize>>(), primitive::<(usize,), Option<usize>>(),
        primitive::<(usize, usize), Option<usize>>(), write::<E>(1),
        primitive::<(Option<usize>, E), Result<usize, E>>(),
        controls(&error_types::<usize, E>(), 5), write::<usize>(1),
    ])
}
fn bound() -> Audit { aggregate([bound_entry(), fixed_work(), returned::<W>()]) }
// Positive independent body ledger. Keep bound_entry above expanded separately
// so legacy parity does not depend on production's extracted body recipe.
fn bound_entry_body() -> Audit {
    aggregate([
        primitive::<(usize, usize), Option<usize>>(), write::<Z>(1),
        primitive::<(Option<usize>, Z), Option<usize>>(), primitive::<(usize,), Option<usize>>(),
        primitive::<(usize, usize), Option<usize>>(), write::<E>(1),
        primitive::<(Option<usize>, E), Result<usize, E>>(),
        controls(&error_types::<usize, E>(), 5), write::<usize>(1),
    ])
}
fn bound_raw_return() -> Audit {
    aggregate([primitive::<(usize,), Result<W, E>>(), bound_entry_body(), fixed_work(), returned::<W>()])
}
fn string() -> Audit {
    aggregate([interface::<(&'static String,), Q, E>(), primitive::<(&'static String,), usize>(), allocation(true),
        string_row(), child_quote(), write::<(usize, usize)>(1), returned::<Q>()])
}
fn vector() -> Audit {
    aggregate([interface::<(&'static Vec<usize>,), Q, E>(), literal(primitive::<(&'static Vec<usize>,), usize>(), 2),
        array_bytes::<usize>(), allocation(true), vec_row(), child_quote(), write::<(usize, usize)>(1), returned::<Q>()])
}
fn sequence(m: usize, outer: bool, name: bool, dims: bool) -> Audit {
    aggregate([entry(outer), fixed_work(), iteration(next(), m.checked_add(1).unwrap()),
        iteration(pair(name, dims), m), tail()])
}

fn compare(name: &str, actual: super::Row, expected: Audit) {
    assert_eq!([actual.k as u128, actual.v as u128, actual.p as u128], expected.counts, "{name}");
    assert_eq!(actual.layout.profile.bytes as u128, expected.bytes, "{name} lexical union");
    assert_eq!(actual.layout.profile.max as u128, expected.max, "{name} constituent maximum");
    assert_eq!(actual.layout.max_align, expected.max_align, "{name} typed alignment maximum");
    assert_eq!(actual.work() as u128, total(expected.counts), "{name} total");
    assert!(u64::try_from(total(expected.counts)).is_ok());
    assert!(usize::try_from(expected.bytes).is_ok());
    eprintln!("QUERY {name}: K={} V={} P={} WORK={} CONTROL_BYTES={} LAYOUT_MAX={} ALIGN_MAX={}",
        expected.counts[0], expected.counts[1], expected.counts[2], total(expected.counts), expected.bytes, expected.max, expected.max_align);
}

#[test]
fn independent_literal_rows_and_supported_abi_records() {
    compare("ADD", super::ADD, add64());
    compare("MUL", super::MUL, multiply64());
    compare("BADD", super::BADD, add_bytes());
    compare("ARRAYpair", super::array::<P>(), array_bytes::<P>());
    compare("ARRAYusize", super::array::<usize>(), array_bytes::<usize>());
    compare("COUNT raw X + mapped E", super::COUNT, convert_count());
    compare("FACTS", super::FACTS, allocation(true));
    compare("TOTAL", super::TOTAL, inspection());
    compare("SUM3", super::sum::<3>(), fold::<3>());
    compare("SUM4", super::sum::<4>(), fold::<4>());
    compare("WORK_CALL 3ADD 2MUL", super::WORK_CALL, work_call());
    compare("WORK_CONTROL 4ADD 4MUL", super::WORK_CONTROL, work_control());
    compare("CALL_POOL string", super::call_pool::<&mut String, ()>(), pool::<&mut String, ()>());
    compare("CALL_POOL vec", super::call_pool::<&mut Vec<usize>, ()>(), pool::<&mut Vec<usize>, ()>());
    compare("CALL_POOL release", super::call_pool::<(*mut u8, Layout), ()>(), pool::<(*mut u8, Layout), ()>());
    compare("CHILD_CONTROL", super::CHILD_CONTROL, child_control());
    compare("ROW_STRING", super::ROW_STRING, string_row());
    compare("ROW_VEC", super::ROW_VEC, vec_row());
    compare("APPEND", super::APPEND, append());
    compare("RELEASE", super::RELEASE, release());
    compare("DEFAULT", super::DEFAULT, default_work());
    compare("CHILD_QUOTE", super::CHILD_QUOTE, child_quote());
    compare("SEQUENCE_WORK eight direct COUNT", super::SEQUENCE_WORK, fixed_work());
    compare("ENTRY", super::ENTRY, entry(true));
    compare("FIXED", super::FIXED, fixed_work());
    compare("NEXT", super::NEXT, next());
    compare("PAIR", super::PAIR, pair(true, true));
    compare("TAIL", super::TAIL, tail());
    compare("BOUND_ENTRY", super::BOUND_ENTRY, bound_entry());
    compare("BOUND_ENTRY_BODY", super::BOUND_ENTRY_BODY, bound_entry_body());
    compare("BOUND", super::BOUND, bound());
    compare("BOUND_RAW_RETURN", super::BOUND_RAW_RETURN, bound_raw_return());
    compare("STRING", super::STRING, string());
    compare("USIZE_VEC", super::VEC, vector());
    compare("SEQUENCE lexical union", super::SEQUENCE_LEXICAL,
        aggregate([entry(true), fixed_work(), next(), pair(true, true), tail()]));

    use super::*;
    for (expected, actual, work, capacity) in [
        (bound(), [RETIREMENT_QUERY_BOUND_K, RETIREMENT_QUERY_BOUND_V, RETIREMENT_QUERY_BOUND_P],
            RETIREMENT_QUERY_BOUND_WORK, RETIREMENT_QUERY_BOUND_CONTROL_BYTES),
        (string(), [RETIREMENT_QUERY_STRING_K, RETIREMENT_QUERY_STRING_V, RETIREMENT_QUERY_STRING_P],
            RETIREMENT_QUERY_STRING_WORK, RETIREMENT_QUERY_STRING_CONTROL_BYTES),
        (vector(), [RETIREMENT_QUERY_USIZE_VEC_K, RETIREMENT_QUERY_USIZE_VEC_V, RETIREMENT_QUERY_USIZE_VEC_P],
            RETIREMENT_QUERY_USIZE_VEC_WORK, RETIREMENT_QUERY_USIZE_VEC_CONTROL_BYTES),
        (inspection(), [RETIREMENT_QUERY_TOTAL_K, RETIREMENT_QUERY_TOTAL_V, RETIREMENT_QUERY_TOTAL_P],
            RETIREMENT_QUERY_TOTAL_WORK, RETIREMENT_QUERY_TOTAL_CONTROL_BYTES),
    ] {
        assert_eq!(actual.map(u128::from), expected.counts);
        assert_eq!(work as u128, total(expected.counts));
        assert_eq!(capacity as u128, expected.bytes);
    }
    let seq = sequence(0, true, true, true);
    let maximum = [bound().max, string().max, vector().max, inspection().max, seq.max].into_iter().max().unwrap();
    assert_eq!(RETIREMENT_QUERY_LAYOUT_MAX as u128, maximum);
    assert_eq!(RETIREMENT_QUERY_SEQUENCE_CONTROL_BYTES as u128, seq.bytes);
}

#[test]
fn positive_raw_return_boundary_and_public_export_contract() {
    fn same(a: Audit, b: Audit) {
        assert_eq!(a.counts, b.counts);
        assert_eq!((a.bytes, a.max, a.max_align), (b.bytes, b.max, b.max_align));
        assert_eq!(a.trace, b.trace);
    }
    // Legacy source still owns direct-edge residuals, followed by unchanged body.
    same(bound_entry(), aggregate([interface::<(usize,), W, E>(), bound_entry_body()]));
    same(bound(), aggregate([bound_entry(), fixed_work(), returned::<W>()]));

    // Enumerate raw boundary literally: complete padded argument/Result tuples,
    // endpoint call, then plain two-event control. No outer raw-M Try family.
    let raw_boundary = aggregate([
        invocation(&[ty::<(usize,)>(), ty::<(usize,)>(), ty::<Result<W, E>>(), ty::<Result<W, E>>()]),
        controls(&[ty::<(usize,)>(), ty::<Result<W, E>>()], 2),
    ]);
    assert_eq!(raw_boundary.counts[0..2], [1, 2]);
    let expected = aggregate([raw_boundary.clone(), bound_entry_body(), fixed_work(), returned::<W>()]);
    same(bound_raw_return(), expected.clone());
    compare("RAW_RETURN positive literal boundary", super::BOUND_RAW_RETURN, expected.clone());

    // Omission/duplicate/direct-residual alternatives cannot satisfy this ledger.
    let no_boundary = aggregate([bound_entry_body(), fixed_work(), returned::<W>()]);
    let duplicate = aggregate([raw_boundary, bound_raw_return()]);
    let extra_raw_residual = aggregate([bound_raw_return(), controls(&error_types::<W, E>(), 5)]);
    for wrong in [no_boundary, duplicate, bound(), extra_raw_residual] {
        assert_ne!(wrong.counts, expected.counts);
        assert_ne!(wrong.bytes, expected.bytes);
    }

    // Resolve via actual public retirement owner, including wildcard re-export.
    assert_eq!([
        owner::RETIREMENT_QUERY_BOUND_RAW_RETURN_K,
        owner::RETIREMENT_QUERY_BOUND_RAW_RETURN_V,
        owner::RETIREMENT_QUERY_BOUND_RAW_RETURN_P,
    ].map(u128::from), expected.counts);
    assert_eq!(owner::RETIREMENT_QUERY_BOUND_RAW_RETURN_WORK as u128, total(expected.counts));
    assert_eq!(owner::RETIREMENT_QUERY_BOUND_RAW_RETURN_CONTROL_BYTES as u128, expected.bytes);
    assert_eq!(owner::RETIREMENT_QUERY_BOUND_RAW_RETURN_LAYOUT_MAX as u128, expected.max);
    assert_eq!(owner::RETIREMENT_QUERY_BOUND_RAW_RETURN_ALIGN_MAX, expected.max_align);
    // Explicit legacy public-owner parity, independently reconstructed above.
    assert_eq!([
        owner::RETIREMENT_QUERY_BOUND_K, owner::RETIREMENT_QUERY_BOUND_V, owner::RETIREMENT_QUERY_BOUND_P,
    ].map(u128::from), bound().counts);
    assert_eq!(owner::RETIREMENT_QUERY_BOUND_WORK as u128, total(bound().counts));
    assert_eq!(owner::RETIREMENT_QUERY_BOUND_CONTROL_BYTES as u128, bound().bytes);
}

#[test]
fn independent_alignment_ledger_retains_internal_and_zero_repeat_slots() {
    #[repr(align(64))]
    struct AlignedZst;
    let zst = write::<AlignedZst>(1);
    assert_eq!((zst.bytes, zst.max, zst.max_align), (0, 0, 64));
    let empty = aggregate([]);
    assert_eq!((empty.bytes, empty.max, empty.max_align), (0, 0, 1));
    let internal = aggregate([
        fold::<3>(), fold::<4>(), work_call(), work_control(),
        write::<(u64, u64, u64, u64, W)>(1),
    ]);
    for n in [0, 1, 4] {
        let repeated = iteration(internal.clone(), n);
        assert_eq!((repeated.bytes, repeated.max, repeated.max_align),
            (internal.bytes, internal.max, internal.max_align));
        let separate = literal(internal.clone(), n);
        assert_eq!((separate.bytes, separate.max, separate.max_align),
            (product(internal.bytes, n as u128), internal.max, internal.max_align));
    }
    for m in [0, 1, 4] {
        assert_eq!(sequence(m, true, true, true).max_align, sequence(0, true, true, true).max_align);
    }
    // Every helper comparison in independent_literal_rows also checks alignment;
    // this witness expressly includes internal tuples beyond public endpoints.
    assert!(bound_raw_return().max_align >= internal.max_align);
}

// Public-only composition: additional caller arithmetic, not a production API.
fn coefficients() -> [[u64; 3]; 5] {
    use super::*;
    [
        [RETIREMENT_QUERY_SEQUENCE_ENTRY_K, RETIREMENT_QUERY_SEQUENCE_ENTRY_V, RETIREMENT_QUERY_SEQUENCE_ENTRY_P],
        [RETIREMENT_QUERY_SEQUENCE_FIXED_K, RETIREMENT_QUERY_SEQUENCE_FIXED_V, RETIREMENT_QUERY_SEQUENCE_FIXED_P],
        [RETIREMENT_QUERY_SEQUENCE_NEXT_K, RETIREMENT_QUERY_SEQUENCE_NEXT_V, RETIREMENT_QUERY_SEQUENCE_NEXT_P],
        [RETIREMENT_QUERY_SEQUENCE_PAIR_K, RETIREMENT_QUERY_SEQUENCE_PAIR_V, RETIREMENT_QUERY_SEQUENCE_PAIR_P],
        [RETIREMENT_QUERY_SEQUENCE_TAIL_K, RETIREMENT_QUERY_SEQUENCE_TAIL_V, RETIREMENT_QUERY_SEQUENCE_TAIL_P],
    ]
}
fn composed(m: usize, terms: [[u64; 3]; 5]) -> Option<([u64; 3], u64)> {
    let successor = u64::try_from(m.checked_add(1)?).ok()?;
    let count = u64::try_from(m).ok()?;
    let mut out = [0u64; 3];
    for component in 0..3 {
        out[component] = terms[0][component].checked_add(terms[1][component])?
            .checked_add(terms[2][component].checked_mul(successor)?)?
            .checked_add(terms[3][component].checked_mul(count)?)?
            .checked_add(terms[4][component])?;
    }
    Some((out, out[0].checked_add(out[1])?.checked_add(out[2])?))
}
fn wide_composition(m: usize) -> Option<([u128; 3], u64)> {
    let successor = m.checked_add(1)? as u128;
    let terms = [entry(true), fixed_work(), next(), pair(true, true), tail()];
    let mut out = [0u128; 3];
    for component in 0..3 {
        out[component] = plus(plus(plus(plus(terms[0].counts[component], terms[1].counts[component]),
            product(terms[2].counts[component], successor)),
            product(terms[3].counts[component], m as u128)), terms[4].counts[component]);
        u64::try_from(out[component]).ok()?;
    }
    Some((out, u64::try_from(total(out)).ok()?))
}
fn dominated(lower: [u128; 3], upper: [u128; 3]) {
    for (a, b) in lower.into_iter().zip(upper) { assert!(a <= b, "{lower:?} not <= {upper:?}"); }
}
fn all_logical_prefixes(trace: &Audit, upper: [u128; 3]) {
    let mut prefix = [0u128; 3];
    dominated(prefix, upper);
    for leaf in &trace.trace {
        // A control leaf reserves both success/error forms. Any stopped event
        // prefix retains no more than this whole leaf's endpoint potential.
        if leaf[1] != 0 {
            for events in 0..=leaf[1] {
                dominated([prefix[0], plus(prefix[1], events), plus(prefix[2], leaf[2])], upper);
            }
        }
        for i in 0..3 { prefix[i] = plus(prefix[i], leaf[i]); }
        dominated(prefix, upper);
    }
    assert_eq!(prefix, trace.counts);
}

#[test]
fn expanded_success_zero_and_error_prefixes_fit_checked_sequence_coefficients() {
    for m in [0, 1, 2, 7, 19] {
        let (wide, work) = wide_composition(m).unwrap();
        let (actual, actual_work) = composed(m, coefficients()).unwrap();
        assert_eq!(actual.map(u128::from), wide);
        assert_eq!(actual_work, work);
        for initialized in 0..=m {
            for outer in [false, true] {
                for name in [false, true] {
                    for dims in [false, true] {
                        let simulated = sequence(initialized, outer, name, dims);
                        all_logical_prefixes(&simulated, wide);
                    }
                }
            }
        }
        eprintln!("QUERY sequence M={m}: KVP={actual:?} WORK={actual_work} CONTROL_BYTES={}",
            super::RETIREMENT_QUERY_SEQUENCE_CONTROL_BYTES);
    }
    // Large values checked algebraically, without constructing dynamic traces.
    for m in [usize::MAX / 100_000_000, usize::MAX / 100_000, usize::MAX / 128, usize::MAX] {
        assert_eq!(composed(m, coefficients()).map(|(c, w)| (c.map(u128::from), w)), wide_composition(m));
    }
    let intercept = total(sequence(0, true, true, true).counts);
    let slope = total(sequence(1, true, true, true).counts).checked_sub(intercept).unwrap();
    let work_limited = (u64::MAX as u128).checked_sub(intercept).unwrap().checked_div(slope).unwrap();
    let large = usize::try_from(work_limited.min((usize::MAX - 1) as u128)).unwrap();
    let composed_large = composed(large, coefficients()).unwrap();
    assert_eq!((composed_large.0.map(u128::from), composed_large.1), wide_composition(large).unwrap());
    if (large as u128) == work_limited {
        assert!(composed(large.checked_add(1).unwrap(), coefficients()).is_none());
    }
    eprintln!("QUERY checked-large M={large}: KVP={:?} WORK={}", composed_large.0, composed_large.1);
    for (cap, element) in [(0, 0), (0, 7), (usize::MAX, 0), (3, 7), (usize::MAX, 2)] {
        let zero = cap == 0 || element == 0;
        all_logical_prefixes(&allocation(!zero), allocation(true).counts);
        let expected = if zero { Ok((0, 0)) } else {
            cap.checked_mul(element).map(|n| (n, 1)).ok_or(Error::Overflow)
        };
        assert_eq!(owner::allocation_facts(cap, element), expected);
    }
    // ARRAY and FACTS are two separate edges even when both compute extent.
    let separate = aggregate([array_bytes::<usize>(), allocation(true)]);
    assert_eq!(separate.bytes, plus(array_bytes::<usize>().bytes, allocation(true).bytes));
    assert!(separate.counts[0] > allocation(true).counts[0]);
    all_logical_prefixes(&bound(), bound().counts);
    all_logical_prefixes(&string(), string().counts);
    all_logical_prefixes(&vector(), vector().counts);
    all_logical_prefixes(&inspection(), inspection().counts);
}

#[test]
fn numeric_exact_one_short_and_overflow_boundary_only() {
    // Numeric comparison only: no fake payer/token/Scope or pre-read claim.
    fn accepts(work: u128, capacity: u128, needed: &Audit) -> bool {
        work >= total(needed.counts) && capacity >= needed.bytes
    }
    for required in [bound(), string(), vector(), inspection(), sequence(7, true, true, true)] {
        let work = total(required.counts);
        assert!(accepts(work, required.bytes, &required));
        assert!(!accepts(work.checked_sub(1).unwrap(), required.bytes, &required));
        assert!(!accepts(work, required.bytes.checked_sub(1).unwrap(), &required));
    }
    assert!(composed(usize::MAX, coefficients()).is_none()); // M+1
    let mut scale = [[0; 3]; 5]; scale[3][0] = u64::MAX;
    assert!(composed(2, scale).is_none()); // component multiplication
    let mut addition = [[0; 3]; 5]; addition[0][0] = u64::MAX; addition[1][0] = 1;
    assert!(composed(0, addition).is_none()); // component addition
    let mut sum = [[0; 3]; 5]; sum[0] = [u64::MAX, 1, 0];
    assert!(composed(0, sum).is_none()); // K+V+P
}

#[test]
fn actual_callback_and_closure_layouts_retain_borrowed_fnmut_receivers() {
    fn add_item<F: FnMut(u64, u64) -> Result<u64, E>>(mut f: F) {
        assert_eq!(size_of::<F>(), size_of::<Z>());
        assert_eq!(align_of::<F>(), align_of::<Z>());
        assert_eq!(size_of::<(&mut F, u64, u64)>(), size_of::<(&mut Z, u64, u64)>());
        assert_eq!(align_of::<(&mut F, u64, u64)>(), align_of::<(&mut Z, u64, u64)>());
        assert_eq!(f(2, 3), Ok(5));
        assert_eq!(f(u64::MAX, 1), Err(Error::Overflow));
    }
    let item = owner::add; // Actual function item, not coerced function pointer.
    assert_eq!(size_of_val(&item), size_of::<Z>());
    assert_eq!(align_of_val(&item), align_of::<Z>());
    add_item(item);
    // Exact noncapturing bound-successor closure expression. No fn pointer.
    let successor = |n: usize| n.checked_add(1);
    fn bound_closure<F: FnOnce(usize) -> Option<usize>>(f: F) {
        assert_eq!(size_of::<F>(), size_of::<Z>());
        assert_eq!(align_of::<F>(), align_of::<Z>());
        assert_eq!(size_of::<(Option<usize>, F)>(), size_of::<(Option<usize>, Z)>());
        assert_eq!(align_of::<(Option<usize>, F)>(), align_of::<(Option<usize>, Z)>());
        assert_eq!(f(7), Some(8));
    }
    assert_eq!(size_of_val(&successor), size_of::<Z>());
    assert_eq!(align_of_val(&successor), align_of::<Z>());
    bound_closure(successor);
    fn count_mapper<F: FnOnce(X) -> E>(mapper: F) {
        assert_eq!(size_of_val(&mapper), size_of::<Z>());
        assert_eq!(align_of::<F>(), align_of::<Z>());
        assert_eq!(size_of::<(Result<u64, X>, F)>(), size_of::<(Result<u64, X>, Z)>());
        assert_eq!(align_of::<(Result<u64, X>, F)>(), align_of::<(Result<u64, X>, Z)>());
    }
    count_mapper(|_| Error::Overflow);
    assert!(size_of::<&mut Z>() > size_of::<Z>());
    assert_eq!(sz::<C>(), size_of::<owner::ChildRow>() as u128);
    // Both raw X and mapped E endpoints, including both residual families.
    for sizes in [error_types::<u64, X>(), error_types::<u64, E>()] {
        assert_eq!(controls(&sizes, 5).bytes, sizes.into_iter().map(|t| t.0).fold(0, plus));
    }
    assert_eq!(fold::<3>().max, super::sum::<3>().layout.profile.max as u128);
    assert_eq!(fold::<4>().max, super::sum::<4>().layout.profile.max as u128);
    eprintln!("QUERY actual ABI C={} E={} X={} AddItem={} BoundSuccessor={} FnMutReceiver={} Ai3={} Ai4={}",
        sz::<C>(), sz::<E>(), sz::<X>(), size_of_val(&item), size_of_val(&successor),
        sz::<&mut Z>(), sz::<IntoIter<u64, 3>>(), sz::<IntoIter<u64, 4>>());
}

#[test]
fn lexical_frames_reuse_dynamic_iterations_not_literal_or_nested_slots() {
    let base = sequence(0, true, true, true);
    for m in [1, 2, 7, 19] {
        let repeated = sequence(m, true, true, true);
        assert_eq!(repeated.bytes, base.bytes);
        assert_eq!(repeated.max, base.max);
        assert!(total(repeated.counts) > total(base.counts));
    }
    let callback = aggregate([primitive::<(&'static mut Z, u64, u64), Result<u64, E>>(), add64(),
        controls(&error_types::<u64, E>(), 5), write::<u64>(1)]);
    assert_eq!(iteration(callback.clone(), 4).bytes, callback.bytes);
    assert_eq!(literal(callback.clone(), 4).bytes, product(4, callback.bytes));
    let progress = controls(&[ty::<IntoIter<u64, 3>>(), ty::<bool>()], 2);
    assert_eq!(iteration(progress.clone(), 4).bytes, progress.bytes);
    assert_eq!(iteration(progress.clone(), 4).counts[1], product(4, 2));
    assert_eq!(iteration(callback.clone(), 3).counts[0], product(3, callback.counts[0]));
    // Outer/inner mul and outer add interfaces stay distinct while nested.
    let nested = aggregate([add64(), multiply64(), multiply64()]);
    assert_eq!(nested.bytes, plus(add64().bytes, product(2, multiply64().bytes)));
    assert!(nested.bytes > nested.max);
    let distinct = aggregate([pool::<&mut Seq, ()>(),
        pool::<(&mut [P], usize), Option<&mut P>>(), pool::<&mut P, ()>()]);
    assert_eq!(distinct.bytes, product(3, pool::<&mut Seq, ()>().bytes));
    assert_eq!(inspection().bytes, plus(interface::<(W,), u64, E>().bytes,
        plus(product(2, add64().bytes), returned::<u64>().bytes)));
    // Quote.work copy is an external extra cell, not free repeated inspection.
    assert!(aggregate([write::<W>(1), inspection()]).bytes > inspection().bytes);
}

#[test]
fn source_overflows_are_paid_failure_prefixes_not_composition_success() {
    assert_eq!(owner::checked_array_bytes::<usize>(usize::MAX), Err(Error::Overflow));
    assert_eq!(owner::allocation_facts(usize::MAX, 2), Err(Error::Overflow));
    assert_eq!(owner::bytes_add(usize::MAX, 1), Err(Error::Overflow));
    assert_eq!(owner::add(u64::MAX, 1), Err(Error::Overflow));
    assert_eq!(owner::mul(u64::MAX, 2), Err(Error::Overflow));
    assert_eq!(W { k: u64::MAX, v: 1, p: 0, s: 0 }.checked_total(), Err(Error::Overflow));
    assert_eq!(owner::string_usize_sequence_retirement_bound_v1(usize::MAX), Err(Error::Overflow));
    assert_eq!(owner::sequence_work(usize::MAX, 0), Err(Error::Overflow));
    // COUNT overflow only where representable; never assume a future ABI width.
    if usize::BITS > u64::BITS {
        let value = usize::try_from(plus(u64::MAX as u128, 1)).unwrap();
        assert_eq!(owner::count(value), Err(Error::Overflow));
    } else {
        assert_eq!(owner::count(usize::MAX), u64::try_from(usize::MAX).map_err(|_| Error::Overflow));
    }
    assert!(u64::try_from(plus(u64::MAX as u128, 1)).is_err()); // layout/count export
    assert!(usize::try_from(plus(usize::MAX as u128, 1)).is_err());
    // QUERY_BOUND composition is fixed and fits while evaluator may fail.
    assert!(u64::try_from(total(bound().counts)).is_ok());
    all_logical_prefixes(&bound(), bound().counts);
    // Standalone query fits independently of impossible extent input.
    all_logical_prefixes(&array_bytes::<usize>(), array_bytes::<usize>().counts);
    all_logical_prefixes(&allocation(true), allocation(true).counts);
    all_logical_prefixes(&add_bytes(), add_bytes().counts);
    all_logical_prefixes(&convert_count(), convert_count().counts);
}

fn census(value: &Vec<P>) -> (usize, usize) {
    let mut releases = usize::from(value.capacity() != 0);
    let mut h = size_of::<Seq>().checked_add(value.capacity().checked_mul(size_of::<P>()).unwrap()).unwrap();
    for (s, v) in value {
        releases = releases.checked_add(usize::from(s.capacity() != 0)).unwrap()
            .checked_add(usize::from(v.capacity() != 0)).unwrap();
        h = h.checked_add(s.capacity()).unwrap().checked_add(v.capacity().checked_mul(size_of::<usize>()).unwrap()).unwrap();
    }
    (releases, h)
}

#[test]
fn observed_capacity_payload_and_copy_quotes_leave_query_tariff_unchanged() {
    let mut initialized_empty = Vec::with_capacity(4);
    initialized_empty.push((String::new(), Vec::new()));
    let mut allocated_empty = Vec::with_capacity(5);
    allocated_empty.push((String::with_capacity(17), Vec::with_capacity(9)));
    let mut mixed = Vec::with_capacity(12);
    mixed.push((String::from("rank-one"), vec![2]));
    mixed.push((String::new(), vec![3, 4]));
    mixed.push((String::with_capacity(31), Vec::with_capacity(7)));
    for value in [Vec::new(), Vec::with_capacity(8), initialized_empty, allocated_empty, mixed] {
        let before = (value.as_ptr(), value.len(), value.capacity(), value.clone());
        let q = owner::string_usize_sequence_retirement_v1(&value).unwrap();
        let (a, h) = census(&value);
        assert_eq!((q.initialized_pairs, q.allocation_releases, q.h_owner), (value.len(), a, h));
        let coefficients = composed(value.len(), coefficients()).unwrap();
        assert_eq!(coefficients.1 as u128, total(sequence(value.len(), true, true, true).counts));
        let successors = value.iter().map(|_| 1usize).sum::<usize>().checked_add(1).unwrap();
        assert_eq!(successors, value.len().checked_add(1).unwrap());
        let copy: Q = q; // Type's real Copy implementation; no custody transfer.
        assert_eq!(copy, q);
        assert_eq!((value.as_ptr(), value.len(), value.capacity()), (before.0, before.1, before.2));
        assert_eq!(value, before.3);
        eprintln!("QUERY fixture m={} next={} outer_capacity={} releases={} H={} KVP={:?}",
            value.len(), successors, value.capacity(), a, h, coefficients.0);
    }
    let mut value = Vec::with_capacity(4);
    value.push((String::with_capacity(8), Vec::with_capacity(8)));
    let before = owner::string_usize_sequence_retirement_v1(&value).unwrap();
    let tariff = composed(value.len(), coefficients()).unwrap();
    value[0].0.push_str("abc"); value[0].1.extend_from_slice(&[usize::MAX, 0, 7]);
    assert_eq!(owner::string_usize_sequence_retirement_v1(&value).unwrap(), before);
    let capacities = (value.capacity(), value[0].0.capacity(), value[0].1.capacity());
    value.reserve_exact(capacities.0.checked_add(11).unwrap());
    value[0].0.reserve_exact(capacities.1.checked_add(17).unwrap());
    value[0].1.reserve_exact(capacities.2.checked_add(13).unwrap());
    assert!(value.capacity() > capacities.0);
    assert!(value[0].0.capacity() > capacities.1);
    assert!(value[0].1.capacity() > capacities.2);
    let after = owner::string_usize_sequence_retirement_v1(&value).unwrap();
    assert_eq!(after.work, before.work);
    assert_eq!(after.allocation_releases, before.allocation_releases);
    assert!(after.h_owner > before.h_owner);
    assert_eq!(after.h_owner, census(&value).1);
    assert_eq!(composed(value.len(), coefficients()).unwrap(), tariff);
    assert_eq!(sequence(value.len(), true, true, true).bytes, super::RETIREMENT_QUERY_SEQUENCE_CONTROL_BYTES as u128);
    // No claim reserve capacity equals requested amount; census uses observations.
}
