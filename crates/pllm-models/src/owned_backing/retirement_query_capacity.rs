//! Inactive, source-expanded QUERY metadata; not sponsorship or retirement W.
//!
//! Checked operations/conversions, scalar comparisons, Option/Result combinators,
//! borrowed metadata getters and iterator transport are typed logical primitives,
//! NOT CPU bounds or pricing of private std/allocator/Drop implementations.
//! Every source-owned helper body is expanded. Fold adapters count N callbacks
//! and N+1 progress decisions; their borrowed FnMut receiver is not a ZST.
//! CONTROL_BYTES is a conservative whole lexical union, not a stack promise.
//! Literal source occurrences retain distinct frames, including nested arithmetic;
//! only iteration repeats reuse frames. Owner backing/future retirement S and
//! caller receipt/inspection/payment pools are additional. All errors are retained.
//!
//! Sequence work is entry + fixed + (M+1)*next + M*pair + tail, componentwise
//! checked, then checked K+V+P. M+1 must be checked too. Runtime composition and
//! acquiring/certifying M require additional, previously sponsored caller work.
//! These constants do not obtain M, install a payer, or authorize Compiler wiring.

use super::{ChildRow, Error, LogicalRetirementQuote, LogicalRetirementWork, Pair, Sequence};
use crate::source_budget::quote_layout::{add, maximum, mul, aligned_union as union, AlignedProfile as Profile};
use std::{alloc::Layout, array::IntoIter, convert::Infallible, mem::size_of,
    num::TryFromIntError, ops::ControlFlow, slice::Iter};

type W = LogicalRetirementWork;
type Q = LogicalRetirementQuote;
type C = ChildRow;
type E = Error;
type X = TryFromIntError;
type Z = (); // Actual function-item/noncapturing-closure witness; never fn pointer.
type R<T> = Result<T, E>;
type It = Iter<'static, Pair>;
type WR = &'static mut W;

// Shared neutral usize algebra checks every layout and coefficient operation.
// Export conversion is checked separately; unsupported targets fail const eval.
const fn number(n: usize) -> u64 {
    assert!((n as u128) <= u64::MAX as u128, "retirement query count overflow");
    n as u64
}
const fn slot<T>() -> Profile { Profile::slot::<T>() }
const fn endpoints<A, T>() -> Profile { union([slot::<A>().times(2), slot::<T>().times(2)]) }
const fn errors<T, F>() -> Profile {
    union([slot::<T>(), slot::<Option<T>>(), slot::<Result<T, F>>(),
        slot::<ControlFlow<Result<Infallible, F>, T>>(), slot::<Result<Infallible, F>>()])
}

#[derive(Clone, Copy)]
struct Row { k: usize, v: usize, p: usize, layout: Profile }
impl Row {
    // Literal separate source occurrences: multiply lexical slots as well.
    const fn times(self, n: usize) -> Self {
        Self { k: mul(self.k, n), v: mul(self.v, n), p: mul(self.p, n),
            layout: self.layout.times(n) }
    }
    // Iteration only: keep exactly one active lexical frame, even at zero.
    const fn iterations(self, n: usize) -> Self {
        Self { k: mul(self.k, n), v: mul(self.v, n), p: mul(self.p, n), layout: self.layout }
    }
    const fn work(self) -> u64 { number(add(add(self.k, self.v), self.p)) }
}
const fn rows<const N: usize>(terms: [Row; N]) -> Row {
    let mut out = Row { k: 0, v: 0, p: 0, layout: union([]) };
    let mut i = 0;
    while i < N {
        let r = terms[i];
        out.k = add(out.k, r.k); out.v = add(out.v, r.v); out.p = add(out.p, r.p);
        out.layout = union([out.layout, r.layout]);
        i += 1;
    }
    out
}
const fn call(pool: Profile) -> Row {
    Row { k: 1, v: 0, p: mul(2, pool.profile.bytes), layout: pool }
}
const fn control(pool: Profile, events: usize) -> Row {
    Row { k: 0, v: events, p: mul(mul(2, pool.profile.bytes), add(events, 1)), layout: pool }
}
const fn plain<A, T>() -> Row {
    rows([call(endpoints::<A, T>()), control(union([slot::<A>(), slot::<T>()]), 2)])
}
const fn edge<A, T, F>() -> Row {
    rows([call(endpoints::<A, Result<T, F>>()), control(errors::<T, F>(), 5)])
}
const fn cell<T>(n: usize) -> Row { control(slot::<T>(), 1).times(n) }
const fn finish<T>() -> Row { control(union([slot::<T>(), slot::<R<T>>()]), 2) }
const BRANCH: Row = control(slot::<bool>(), 1);

// Positive inline recipe: no subtraction of a Finish or implicit helper edge.
const fn inline<T>() -> Row {
    rows([plain::<(T, T), Option<T>>(), cell::<E>(1), plain::<(Option<T>, E), R<T>>()])
}
const fn checked<T>() -> Row { rows([inline::<T>(), finish::<T>()]) }
const ADD: Row = rows([edge::<(u64, u64), u64, E>(), checked::<u64>()]);
const MUL: Row = rows([edge::<(u64, u64), u64, E>(), checked::<u64>()]);
const BADD: Row = rows([edge::<(usize, usize), usize, E>(), checked::<usize>()]);
// T selects the source size constant; runtime signature remains (usize,).
const fn array<T>() -> Row {
    let _ = size_of::<T>();
    rows([edge::<(usize,), usize, E>(), checked::<usize>()])
}
const COUNT: Row = rows([
    edge::<(usize,), u64, E>(), edge::<(usize,), u64, X>(),
    plain::<(Result<u64, X>, Z), R<u64>>(), cell::<Z>(1),
    plain::<(X,), E>(), cell::<E>(1), finish::<u64>(),
]);
const FACTS: Row = rows([
    edge::<(usize, usize), (usize, usize), E>(),
    plain::<(usize, usize), bool>().times(2), plain::<(bool, bool), bool>(), BRANCH,
    inline::<usize>(), control(errors::<usize, E>(), 5),
    cell::<(usize, usize)>(1), finish::<(usize, usize)>(),
]);
const TOTAL: Row = rows([edge::<(W,), u64, E>(), ADD.times(2), finish::<u64>()]);
const fn sum<const N: usize>() -> Row {
    rows([
        edge::<([u64; N],), u64, E>(), cell::<[u64; N]>(1),
        plain::<([u64; N],), IntoIter<u64, N>>(), cell::<IntoIter<u64, N>>(1),
        cell::<u64>(1), cell::<Z>(1),
        edge::<(&'static mut IntoIter<u64, N>, u64, Z), u64, E>(),
        control(union([slot::<IntoIter<u64, N>>(), slot::<bool>()]), 2).iterations(add(N, 1)),
        rows([plain::<(&'static mut Z, u64, u64), R<u64>>(), ADD,
            control(errors::<u64, E>(), 5), cell::<u64>(1)]).iterations(N),
        finish::<u64>(),
    ])
}
const WORK_CALL: Row = rows([
    edge::<(WR, u64, u64), (), E>(), ADD.times(3), MUL.times(2),
    plain::<(u64, u64), bool>(), BRANCH, cell::<u64>(3), finish::<()>(),
]);
const WORK_CONTROL: Row = rows([
    edge::<(WR, u64, u64, u64), (), E>(), ADD.times(4), MUL.times(4),
    plain::<(u64, u64), bool>(), BRANCH, cell::<u64>(3), finish::<()>(),
]);
const fn call_pool<A, T>() -> Row {
    // Actual generic sizes pass through COUNT; they are not free runtime calls.
    let _ = (size_of::<A>(), size_of::<T>());
    rows([edge::<(), u64, E>(), COUNT.times(2), ADD, MUL, finish::<u64>()])
}
const CHILD_CONTROL: Row = rows([
    edge::<(), u64, E>(), COUNT.times(4), MUL, sum::<4>(), finish::<u64>(),
]);
const ROW_STRING: Row = rows([
    edge::<(), C, E>(), call_pool::<&'static mut String, ()>(),
    CHILD_CONTROL, cell::<C>(1), finish::<C>(),
]);
const ROW_VEC: Row = rows([
    edge::<(), C, E>(), call_pool::<&'static mut Vec<usize>, ()>(),
    CHILD_CONTROL, cell::<C>(1), finish::<C>(),
]);
const APPEND: Row = rows([
    edge::<(C, WR, u64), (), E>(), WORK_CALL, WORK_CONTROL, COUNT, MUL, ADD,
    cell::<u64>(1), finish::<()>(),
]);
const RELEASE: Row = rows([
    edge::<(WR, u64), (), E>(), call_pool::<(*mut u8, Layout), ()>(),
    WORK_CALL, COUNT.times(3), sum::<3>(), WORK_CONTROL, finish::<()>(),
]);
const DEFAULT: Row = rows([plain::<(), W>(), cell::<u64>(4), cell::<W>(1)]);
const CHILD_QUOTE: Row = rows([
    edge::<(C, usize, usize), Q, E>(), DEFAULT, APPEND, COUNT, RELEASE, TOTAL, BADD,
    cell::<W>(1), cell::<Q>(1), finish::<Q>(),
]);
const SEQUENCE_WORK: Row = rows([
    edge::<(usize, usize), W, E>(), COUNT.times(8), inline::<usize>(),
    control(errors::<usize, E>(), 5), DEFAULT,
    call_pool::<&'static mut Sequence, ()>(),
    call_pool::<(&'static mut [Pair], usize), Option<&'static mut Pair>>(),
    call_pool::<&'static mut Pair, ()>(), WORK_CALL.times(3), WORK_CONTROL.times(3),
    sum::<3>().times(2), MUL.times(2), ADD.times(2), ROW_STRING, ROW_VEC,
    APPEND.times(2), RELEASE, TOTAL, cell::<(u64, u64, u64, u64, W)>(1),
    cell::<u64>(1), finish::<W>(),
]);
const ENTRY: Row = rows([
    edge::<(&'static Sequence,), Q, E>(), plain::<(&'static Sequence,), usize>().times(2),
    array::<Pair>(), FACTS, plain::<(&'static Sequence,), It>(), cell::<It>(1),
    cell::<(usize, usize)>(1),
]);
const FIXED: Row = SEQUENCE_WORK;
const NEXT: Row = rows([
    plain::<(&'static mut It,), Option<&'static Pair>>(),
    control(union([slot::<Option<&'static Pair>>(), slot::<bool>()]), 2),
]);
const PAIR: Row = rows([
    cell::<(&'static String, &'static Vec<usize>)>(1), plain::<(&'static String,), usize>(),
    plain::<(&'static Vec<usize>,), usize>().times(2), FACTS.times(2), array::<usize>(),
    BADD.times(4), cell::<(usize, usize, usize, usize)>(1), cell::<usize>(2),
]);
const TAIL: Row = rows([
    plain::<(&'static Sequence,), usize>().times(2), BADD, cell::<Q>(1), finish::<Q>(),
]);
const BOUND_ENTRY_BODY: Row = rows([
    plain::<(usize, usize), Option<usize>>(), cell::<Z>(1),
    plain::<(Option<usize>, Z), Option<usize>>(), plain::<(usize,), Option<usize>>(),
    plain::<(usize, usize), Option<usize>>(), cell::<E>(1),
    plain::<(Option<usize>, E), R<usize>>(), control(errors::<usize, E>(), 5), cell::<usize>(1),
]);
const BOUND_ENTRY: Row = rows([edge::<(usize,), W, E>(), BOUND_ENTRY_BODY]);
const BOUND: Row = rows([BOUND_ENTRY, SEQUENCE_WORK, finish::<W>()]);
const BOUND_RAW_RETURN: Row = rows([
    plain::<(usize,), Result<W, E>>(), BOUND_ENTRY_BODY, SEQUENCE_WORK, finish::<W>(),
]);
const STRING: Row = rows([
    edge::<(&'static String,), Q, E>(), plain::<(&'static String,), usize>(), FACTS,
    ROW_STRING, CHILD_QUOTE, cell::<(usize, usize)>(1), finish::<Q>(),
]);
const VEC: Row = rows([
    edge::<(&'static Vec<usize>,), Q, E>(), plain::<(&'static Vec<usize>,), usize>().times(2),
    array::<usize>(), FACTS, ROW_VEC, CHILD_QUOTE, cell::<(usize, usize)>(1), finish::<Q>(),
]);
// Only capacity/max are read: no dynamic-M repeats in this lexical union.
const SEQUENCE_LEXICAL: Row = rows([ENTRY, FIXED, NEXT, PAIR, TAIL]);

/// Full pure scalar-bound evaluator; independent of its numeric argument.
pub const RETIREMENT_QUERY_BOUND_K: u64 = number(BOUND.k);
pub const RETIREMENT_QUERY_BOUND_V: u64 = number(BOUND.v);
pub const RETIREMENT_QUERY_BOUND_P: u64 = number(BOUND.p);
pub const RETIREMENT_QUERY_BOUND_WORK: u64 = BOUND.work();
/// Alternate evaluator boundary: exactly one raw Result<W, Error> call/receipt,
/// the shared owned body/helper inventory and W finish. No outer raw-Error `?`.
/// Caller owns argument read, mapping, mapped residual and local installation.
/// Metadata only: no sponsorship, admission or production invocation is added.
pub const RETIREMENT_QUERY_BOUND_RAW_RETURN_K: u64 = number(BOUND_RAW_RETURN.k);
pub const RETIREMENT_QUERY_BOUND_RAW_RETURN_V: u64 = number(BOUND_RAW_RETURN.v);
pub const RETIREMENT_QUERY_BOUND_RAW_RETURN_P: u64 = number(BOUND_RAW_RETURN.p);
pub const RETIREMENT_QUERY_BOUND_RAW_RETURN_WORK: u64 = BOUND_RAW_RETURN.work();
pub const RETIREMENT_QUERY_BOUND_RAW_RETURN_CONTROL_BYTES: usize = BOUND_RAW_RETURN.layout.profile.bytes;
/// Maximum nominal constituent size within this raw-return contract only.
pub const RETIREMENT_QUERY_BOUND_RAW_RETURN_LAYOUT_MAX: usize = BOUND_RAW_RETURN.layout.profile.max;
/// Maximum actual typed alignment across all internal and boundary slots.
pub const RETIREMENT_QUERY_BOUND_RAW_RETURN_ALIGN_MAX: usize = BOUND_RAW_RETURN.layout.max_align;
pub const RETIREMENT_QUERY_STRING_K: u64 = number(STRING.k);
pub const RETIREMENT_QUERY_STRING_V: u64 = number(STRING.v);
pub const RETIREMENT_QUERY_STRING_P: u64 = number(STRING.p);
pub const RETIREMENT_QUERY_STRING_WORK: u64 = STRING.work();
pub const RETIREMENT_QUERY_USIZE_VEC_K: u64 = number(VEC.k);
pub const RETIREMENT_QUERY_USIZE_VEC_V: u64 = number(VEC.v);
pub const RETIREMENT_QUERY_USIZE_VEC_P: u64 = number(VEC.p);
pub const RETIREMENT_QUERY_USIZE_VEC_WORK: u64 = VEC.work();
pub const RETIREMENT_QUERY_SEQUENCE_ENTRY_K: u64 = number(ENTRY.k);
pub const RETIREMENT_QUERY_SEQUENCE_ENTRY_V: u64 = number(ENTRY.v);
pub const RETIREMENT_QUERY_SEQUENCE_ENTRY_P: u64 = number(ENTRY.p);
pub const RETIREMENT_QUERY_SEQUENCE_FIXED_K: u64 = number(FIXED.k);
pub const RETIREMENT_QUERY_SEQUENCE_FIXED_V: u64 = number(FIXED.v);
pub const RETIREMENT_QUERY_SEQUENCE_FIXED_P: u64 = number(FIXED.p);
pub const RETIREMENT_QUERY_SEQUENCE_NEXT_K: u64 = number(NEXT.k);
pub const RETIREMENT_QUERY_SEQUENCE_NEXT_V: u64 = number(NEXT.v);
pub const RETIREMENT_QUERY_SEQUENCE_NEXT_P: u64 = number(NEXT.p);
pub const RETIREMENT_QUERY_SEQUENCE_PAIR_K: u64 = number(PAIR.k);
pub const RETIREMENT_QUERY_SEQUENCE_PAIR_V: u64 = number(PAIR.v);
pub const RETIREMENT_QUERY_SEQUENCE_PAIR_P: u64 = number(PAIR.p);
pub const RETIREMENT_QUERY_SEQUENCE_TAIL_K: u64 = number(TAIL.k);
pub const RETIREMENT_QUERY_SEQUENCE_TAIL_V: u64 = number(TAIL.v);
pub const RETIREMENT_QUERY_SEQUENCE_TAIL_P: u64 = number(TAIL.p);
/// Repeated caller inspection costs this again; internal checks do not prepay it.
pub const RETIREMENT_QUERY_TOTAL_K: u64 = number(TOTAL.k);
pub const RETIREMENT_QUERY_TOTAL_V: u64 = number(TOTAL.v);
pub const RETIREMENT_QUERY_TOTAL_P: u64 = number(TOTAL.p);
pub const RETIREMENT_QUERY_TOTAL_WORK: u64 = TOTAL.work();
pub const RETIREMENT_QUERY_BOUND_CONTROL_BYTES: usize = BOUND.layout.profile.bytes;
pub const RETIREMENT_QUERY_STRING_CONTROL_BYTES: usize = STRING.layout.profile.bytes;
pub const RETIREMENT_QUERY_USIZE_VEC_CONTROL_BYTES: usize = VEC.layout.profile.bytes;
pub const RETIREMENT_QUERY_SEQUENCE_CONTROL_BYTES: usize = SEQUENCE_LEXICAL.layout.profile.bytes;
pub const RETIREMENT_QUERY_TOTAL_CONTROL_BYTES: usize = TOTAL.layout.profile.bytes;
/// Maximum single constituent type, not a sum, square or physical stack bound.
pub const RETIREMENT_QUERY_LAYOUT_MAX: usize = maximum(maximum(BOUND.layout.profile.max, STRING.layout.profile.max),
    maximum(maximum(VEC.layout.profile.max, SEQUENCE_LEXICAL.layout.profile.max), TOTAL.layout.profile.max));

#[cfg(test)]
#[path = "retirement_query_capacity_tests.rs"]
mod tests;
