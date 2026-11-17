//! Compile-time inner inventory for SourceStorage::from_params_bounded only.
//!
//! Replaces the retired partial SelectorBytes at future composition, not an
//! extra reservation. No caller currently consumes these numbers. Outer selector
//! input/error-map arguments, raw/mapped Result/residual families, enum locals
//! and enclosing caller bools remain outside this leaf. No String adapter,
//! borrowed JSON/string backing, work fee or admission authority is included.
use std::mem::size_of;

// Static lifetimes describe layouts, not ownership or static runtime data.
type J = &'static serde_json::Value;
type Str = &'static str;
type OJ = Option<J>;
type OS = Option<Str>;
type OU = Option<u64>;
type R = std::ops::RangeInclusive<u64>;
type GetArgs = (J, Str);
type IndexArgs = (J, Str);
type StrArgs = (J,);
type U64Args = (J,);
type IsSomeArgs = (&'static OJ,);
type CaptureLayout = ();
type SomeAndArgs = (OU, CaptureLayout);
type ClosureArgs = (u64,);
type ContainsArgs = (&'static R, &'static u64);
type CompareArgs = (&'static OS, &'static OS);
type OrArgs = (bool, bool);
type NotArgs = (bool,);
type MatchArgs = (OJ,);
type RangeBounds = (u64, u64);

const fn checked_add(a: usize, b: usize) -> usize {
    match a.checked_add(b) {
        Some(value) => value,
        None => panic!("SDK selector layout overflow"),
    }
}

const fn checked_mul(a: usize, b: usize) -> usize {
    match a.checked_mul(b) {
        Some(value) => value,
        None => panic!("SDK selector layout overflow"),
    }
}

const fn checked_sum<const N: usize>(parts: [usize; N]) -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < N {
        total = checked_add(total, parts[i]);
        i += 1;
    }
    total
}

const fn checked_max<const N: usize>(parts: [usize; N]) -> usize {
    let mut maximum = 0;
    let mut i = 0;
    while i < N {
        if parts[i] > maximum {
            maximum = parts[i];
        }
        i += 1;
    }
    maximum
}

// Actual/formal pair; producer/TX/RX/return family; their checked sum.
const fn a<T>() -> usize {
    checked_mul(2, size_of::<T>())
}

const fn v<T>() -> usize {
    checked_mul(4, size_of::<T>())
}

const fn b<Args, Output>() -> usize {
    checked_add(a::<Args>(), v::<Output>())
}

// All inactive short-circuit branches remain reserved. Fifteen bool families,
// four J families and six OS families; no extra final-guard bool or match E.
pub(crate) const SDK_SELECTOR_BYTES: usize = checked_sum([
    checked_mul(4, b::<GetArgs, OJ>()),
    checked_mul(3, b::<IndexArgs, J>()),
    checked_mul(3, b::<StrArgs, OS>()),
    b::<U64Args, OU>(),
    checked_mul(3, b::<IsSomeArgs, bool>()),
    b::<SomeAndArgs, bool>(),
    checked_mul(3, b::<CompareArgs, bool>()),
    checked_mul(5, b::<OrArgs, bool>()),
    b::<NotArgs, bool>(),
    a::<MatchArgs>(),
    v::<J>(), // Some(value) pattern binding
    b::<ClosureArgs, bool>(),
    v::<RangeBounds>(),
    v::<R>(),
    b::<ContainsArgs, bool>(),
    checked_mul(3, v::<OS>()), // Some("compact"), Some("public"), Some("checked")
]);

// Nominal maximum is independent of additive capacity, not a substitute pool.
// Every actual argument/output/Option/range layout participates, including the
// zero-sized noncapturing closure layout (never a function pointer).
pub(crate) const SDK_SELECTOR_NOMINAL_MAX: usize = checked_max([
    size_of::<J>(),
    size_of::<Str>(),
    size_of::<OJ>(),
    size_of::<OS>(),
    size_of::<OU>(),
    size_of::<u64>(),
    size_of::<bool>(),
    size_of::<R>(),
    size_of::<CaptureLayout>(),
    size_of::<GetArgs>(),
    size_of::<IndexArgs>(),
    size_of::<StrArgs>(),
    size_of::<U64Args>(),
    size_of::<IsSomeArgs>(),
    size_of::<SomeAndArgs>(),
    size_of::<ClosureArgs>(),
    size_of::<ContainsArgs>(),
    size_of::<CompareArgs>(),
    size_of::<OrArgs>(),
    size_of::<NotArgs>(),
    size_of::<MatchArgs>(),
    size_of::<RangeBounds>(),
]);

#[cfg(test)]
#[path = "sdk_source_storage_capacity_tests.rs"]
mod tests;
