//! Layout and selector regressions only. No Geometry funding/custody claim.
use super::*;
use super::super::{source_storage_selector_cases, SourceStorage, SourceStorageError};
use std::mem::size_of_val;
use std::ops::RangeInclusive;

// Independent test arithmetic: do not reuse the production const helpers.
fn reconstruct(terms: &[(usize, usize)]) -> usize {
    terms.iter().try_fold(0usize, |total, &(count, bytes)| {
        total.checked_add(count.checked_mul(bytes)?)
    }).expect("independent selector reconstruction overflow")
}

#[test]
fn fully_expanded_abi_reconstructs_selector_bytes() {
    // Direct ABI types, not production aliases or B/A/V helpers. Arguments
    // expand to 2*site-count; outputs expand to 4*family-count.
    let expected = reconstruct(&[
        (8, size_of::<(&serde_json::Value, &str)>()), // four get
        (6, size_of::<(&serde_json::Value, &str)>()), // three index
        (6, size_of::<(&serde_json::Value,)>()), // three as_str
        (2, size_of::<(&serde_json::Value,)>()), // as_u64
        (6, size_of::<(&Option<&serde_json::Value>,)>()), // three is_some
        (2, size_of::<(Option<u64>, ())>()), // is_some_and receiver/capture
        (6, size_of::<(&Option<&str>, &Option<&str>)>()), // comparisons
        (10, size_of::<(bool, bool)>()), // five OR
        (2, size_of::<(bool,)>()), // NOT
        (2, size_of::<(Option<&serde_json::Value>,)>()), // match
        (2, size_of::<(u64,)>()), // invoked closure
        (2, size_of::<(&RangeInclusive<u64>, &u64)>()), // contains
        (16, size_of::<Option<&serde_json::Value>>()), // four get outputs
        (16, size_of::<&serde_json::Value>()), // three index + Some binding
        (24, size_of::<Option<&str>>()), // three as_str + three literal Some
        (4, size_of::<Option<u64>>()),
        (60, size_of::<bool>()), // fifteen distinct bool families
        (4, size_of::<(u64, u64)>()), // range construction inputs
        (4, size_of::<RangeInclusive<u64>>()),
    ]);
    assert_eq!(SDK_SELECTOR_BYTES, expected);
    assert_eq!(super::super::SDK_SELECTOR_BYTES, expected);
}

#[test]
fn literal_sites_independently_account_for_every_family() {
    let get = (size_of::<(&serde_json::Value, &str)>(),
        size_of::<Option<&serde_json::Value>>());
    let index = (size_of::<(&serde_json::Value, &str)>(),
        size_of::<&serde_json::Value>());
    let string = (size_of::<(&serde_json::Value,)>(), size_of::<Option<&str>>());
    let is_some = (size_of::<(&Option<&serde_json::Value>,)>(), size_of::<bool>());
    let compare = (size_of::<(&Option<&str>, &Option<&str>)>(), size_of::<bool>());
    let or = (size_of::<(bool, bool)>(), size_of::<bool>());
    // Each row is one call/control family, even when ABI layouts coincide.
    let sites = [
        ("get storage", get),
        ("get lookup", get),
        ("get selection", get),
        ("get peer_transport", get),
        ("index weights", index),
        ("index range_policy", index),
        ("index max_abs_activation", index),
        ("as_str storage", string),
        ("as_str weights", string),
        ("as_str range_policy", string),
        ("as_u64 activation", (size_of::<(&serde_json::Value,)>(), size_of::<Option<u64>>())),
        ("is_some lookup", is_some),
        ("is_some selection", is_some),
        ("is_some peer_transport", is_some),
        ("is_some_and activation", (size_of::<(Option<u64>, ())>(), size_of::<bool>())),
        ("compare compact", compare),
        ("compare public", compare),
        ("compare checked", compare),
        ("OR weights/range", or),
        ("OR activation", or),
        ("OR lookup", or),
        ("OR selection", or),
        ("OR peer_transport", or),
        ("NOT activation", (size_of::<(bool,)>(), size_of::<bool>())),
        ("closure invocation", (size_of::<(u64,)>(), size_of::<bool>())),
        ("contains", (size_of::<(&RangeInclusive<u64>, &u64)>(), size_of::<bool>())),
    ];
    let mut total = 0usize;
    for (_, (args, output)) in sites {
        total = total.checked_add(reconstruct(&[(2, args), (4, output)])).unwrap();
    }
    // Explicit non-call sites, including each literal independently.
    let values = [
        ("Some compact", size_of::<Option<&str>>()),
        ("Some public", size_of::<Option<&str>>()),
        ("Some checked", size_of::<Option<&str>>()),
        ("Some value binding", size_of::<&serde_json::Value>()),
        ("range bounds", size_of::<(u64, u64)>()),
        ("inclusive range", size_of::<RangeInclusive<u64>>()),
    ];
    for (_, bytes) in values {
        total = total.checked_add(bytes.checked_mul(4).unwrap()).unwrap();
    }
    total = total.checked_add(reconstruct(&[
        (2, size_of::<(Option<&serde_json::Value>,)>()),
    ])).unwrap();
    assert_eq!(SDK_SELECTOR_BYTES, total);

    let count = |prefix: &str| sites.iter().filter(|(name, _)| name.starts_with(prefix)).count();
    assert_eq!((count("get "), count("index "), count("as_str "), count("as_u64 ")),
        (4, 3, 3, 1));
    assert_eq!((count("is_some "), count("is_some_and "), count("compare ")),
        (3, 1, 3));
    assert_eq!((count("OR "), count("NOT "), count("closure "), count("contains")),
        (5, 1, 1, 1));
    assert_eq!(count("is_some ") + count("is_some_and ") + count("compare ")
        + count("OR ") + count("NOT ") + count("closure ") + count("contains"), 15);
    assert_eq!(count("index ") + 1, 4); // J, including Some binding
    assert_eq!(count("as_str ") + values.iter()
        .filter(|(name, _)| matches!(*name, "Some compact" | "Some public" | "Some checked"))
        .count(), 6); // OS, including all literal Some values
}

#[test]
fn nominal_max_includes_every_reviewed_alias_layout() {
    // Separate direct-type reconstruction verifies both aliases and max inputs.
    let layouts = [
        (size_of::<J>(), size_of::<&serde_json::Value>()),
        (size_of::<Str>(), size_of::<&str>()),
        (size_of::<OJ>(), size_of::<Option<&serde_json::Value>>()),
        (size_of::<OS>(), size_of::<Option<&str>>()),
        (size_of::<OU>(), size_of::<Option<u64>>()),
        (size_of::<u64>(), size_of::<u64>()),
        (size_of::<bool>(), size_of::<bool>()),
        (size_of::<R>(), size_of::<RangeInclusive<u64>>()),
        (size_of::<CaptureLayout>(), size_of::<()>()),
        (size_of::<GetArgs>(), size_of::<(&serde_json::Value, &str)>()),
        (size_of::<IndexArgs>(), size_of::<(&serde_json::Value, &str)>()),
        (size_of::<StrArgs>(), size_of::<(&serde_json::Value,)>()),
        (size_of::<U64Args>(), size_of::<(&serde_json::Value,)>()),
        (size_of::<IsSomeArgs>(), size_of::<(&Option<&serde_json::Value>,)>()),
        (size_of::<SomeAndArgs>(), size_of::<(Option<u64>, ())>()),
        (size_of::<ClosureArgs>(), size_of::<(u64,)>()),
        (size_of::<ContainsArgs>(), size_of::<(&RangeInclusive<u64>, &u64)>()),
        (size_of::<CompareArgs>(), size_of::<(&Option<&str>, &Option<&str>)>()),
        (size_of::<OrArgs>(), size_of::<(bool, bool)>()),
        (size_of::<NotArgs>(), size_of::<(bool,)>()),
        (size_of::<MatchArgs>(), size_of::<(Option<&serde_json::Value>,)>()),
        (size_of::<RangeBounds>(), size_of::<(u64, u64)>()),
    ];
    for (alias, direct) in layouts {
        assert_eq!(alias, direct);
        assert!(SDK_SELECTOR_NOMINAL_MAX >= direct);
    }
    let expected = layouts.iter().map(|(_, direct)| *direct).max().unwrap();
    assert_eq!(SDK_SELECTOR_NOMINAL_MAX, expected);
    assert_eq!(super::super::SDK_SELECTOR_NOMINAL_MAX, expected);
    assert!(SDK_SELECTOR_BYTES > SDK_SELECTOR_NOMINAL_MAX);
}

#[test]
fn zero_capture_retains_invocation_option_and_range_layouts() {
    let closure = |v: u64| (1..=1024).contains(&v);
    assert_eq!(size_of_val(&closure), 0);
    assert_eq!(size_of::<CaptureLayout>(), size_of_val(&closure));
    assert!(size_of::<fn(u64) -> bool>() > size_of::<CaptureLayout>());
    assert_eq!(size_of_val(&(Some(1u64), closure)), size_of::<SomeAndArgs>());
    assert_eq!(size_of::<SomeAndArgs>(), size_of::<OU>());
    assert_eq!(size_of::<ClosureArgs>(), size_of_val(&(1u64,)));
    let bounds = (1u64, 1024u64);
    let range = bounds.0..=bounds.1;
    let value = 1u64;
    assert_eq!(size_of::<RangeBounds>(), size_of_val(&bounds));
    assert_eq!(size_of::<R>(), size_of_val(&range));
    assert_eq!(size_of::<ContainsArgs>(), size_of_val(&(&range, &value)));
    assert!(!closure(0));
    assert!(closure(1));
    assert!(closure(1024));
    assert!(!closure(1025));
}

#[test]
fn shared_selector_cases_keep_bounded_and_legacy_semantics() {
    for (case, params, expected) in source_storage_selector_cases() {
        assert_eq!(SourceStorage::from_params_bounded(&params), expected, "{case}: {params}");
        let legacy = match expected {
            Ok(storage) => Ok(storage),
            Err(SourceStorageError::UnsupportedCompactComposition) =>
                Err("unsupported compact replicated storage composition".to_owned()),
            Err(SourceStorageError::NoncanonicalChoice) =>
                Err("noncanonical replicated storage choice".to_owned()),
        };
        assert_eq!(SourceStorage::from_params(&params), legacy, "{case}: {params}");
    }
}

// Exact/one-short full Geometry admission and live-owner failed-tail custody
// remain unimplemented: this inactive numeric leaf cannot establish either.
