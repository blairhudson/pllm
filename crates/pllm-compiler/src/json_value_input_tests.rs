use super::*;
use crate::plan_input::{FACTORIES, NUMBER_CALLS};
use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};

fn limits() -> Limits {
    Limits {
        max_source_bytes: 1 << 20,
        max_nodes: 100_000,
        max_depth: 32,
        max_string_bytes: 1 << 16,
        max_number_bytes: 128,
        max_number_work: 1 << 26,
        max_work: 1 << 30,
        max_payload_bytes: MAX_ALLOCATION_BYTES,
    }
}
fn incoming(root: &OwnedBudget, bytes: &[u8], spare: usize) -> OwnedCharged<Vec<u8>> {
    let capacity = bytes.len() + spare;
    root.charge_external(capacity)
        .unwrap()
        .try_build(|_| {
            let mut data = Vec::with_capacity(capacity);
            data.extend_from_slice(bytes);
            assert_eq!(data.capacity(), capacity);
            Ok(data)
        })
        .unwrap()
}
fn working(root: &OwnedBudget) -> OwnedExternalCharge {
    root.charge_external(bootstrap_bytes().unwrap()).unwrap()
}
fn reset() {
    FACTORIES.with(|s| s.set((0, None, None)));
    NUMBER_CALLS.with(|s| s.set(0));
}
fn factories() -> usize {
    FACTORIES.with(|s| s.get().0)
}
fn import(root: &OwnedBudget, raw: &[u8]) -> OwnedCharged<ImportedValue> {
    quote(incoming(root, raw, 19), working(root), limits())
        .unwrap()
        .construct()
        .unwrap()
}
const FIXTURE: &[u8] = br#"{"model_type":"qwen2","hidden_size":64,"epsilon":0.000001,"theta":1000000.0,"layers":[{"id":"a","scale":0.125},{"id":"b","scale":-0.0}],"empty":{},"unused":[null,true,false]}"#;

#[test]
fn all_json_roots_and_exact_serialized_values_match_normal_serde() {
    for raw in [
        b"null".as_slice(),
        b"true",
        b"false",
        b"0",
        b"-0",
        b"1.25",
        br#""a\n\u03bb\ud83e\udd80""#,
        b"[]",
        b"{}",
        br#"[null,{},[],{"a":[1,2,"x"]}]"#,
        FIXTURE,
    ] {
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let output = import(&root, raw);
        let expected: Value = serde_json::from_slice(raw).unwrap();
        assert_eq!(output.payload().value(), &expected);
        assert_eq!(
            serde_json::to_vec(output.payload().value()).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
        assert_eq!(output.payload().raw(), raw);
    }
}

#[test]
fn numbers_keep_ordinary_serde_rounding_kind_bits_and_integer_boundaries() {
    for text in [
        "0",
        "-0",
        "0.0",
        "-0.0",
        "1e-6",
        "1.000000000000000111022302462515654",
        "1000000.0",
        "1e-400",
        "-5e-324",
        "2.2250738585072014e-308",
        "1.7976931348623157e308",
        "18446744073709551615",
        "18446744073709551616",
        "-9223372036854775808",
        "-9223372036854775809",
    ] {
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let output = import(&root, text.as_bytes());
        let expected: Value = serde_json::from_str(text).unwrap();
        let actual = output.payload().value();
        assert_eq!(actual.is_i64(), expected.is_i64());
        assert_eq!(actual.is_u64(), expected.is_u64());
        assert_eq!(actual.is_f64(), expected.is_f64());
        assert_eq!(
            actual.as_f64().unwrap().to_bits(),
            expected.as_f64().unwrap().to_bits()
        );
    }
}

#[test]
fn duplicate_and_escaped_keys_retire_overwritten_subtrees_without_changing_serde_semantics() {
    let raw = br#"{"a":{"discard":[1,2,3]},"\u0061":["last",{"\ud83e\udd80":"\u03bb"}],"":{},"prefix":0,"prefix-long":1}"#;
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let output = import(&root, raw);
    assert_eq!(
        output.payload().value(),
        &serde_json::from_slice::<Value>(raw).unwrap()
    );
    assert_eq!(output.payload().counts().objects, 4);
    assert_eq!(output.payload().raw(), raw);
}

#[test]
fn original_bytes_pointer_spare_capacity_and_both_original_charges_survive_root_drop() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let raw = incoming(&root, FIXTURE, 4096);
    let pointer = raw.payload().as_ptr();
    let capacity = raw.payload().capacity();
    let incoming_charge = raw.charged_bytes();
    let counter = std::ptr::from_ref(raw.budget());
    let quoted = quote(raw, working(&root), limits()).unwrap();
    let resources = quoted.resources();
    let output = quoted.construct().unwrap();
    assert_eq!(output.charged_bytes(), resources.working_charge_bytes);
    drop(root);
    assert_eq!(std::ptr::from_ref(output.budget()), counter);
    assert_eq!(std::ptr::from_ref(output.payload().budget()), counter);
    assert_eq!(output.payload().raw().as_ptr(), pointer);
    assert_eq!(output.payload().raw_owner().payload().capacity(), capacity);
    assert_eq!(
        output.payload().raw_owner().charged_bytes(),
        incoming_charge
    );
    assert_eq!(resources.incoming_owner_bytes, incoming_charge);
    assert_eq!(
        resources.original_owners_peak_bytes,
        output.charged_bytes() + incoming_charge
    );
    assert!(output.budget().used_bytes() >= resources.original_owners_peak_bytes);
}

#[test]
fn wrong_counter_and_actual_undercharged_capacity_reject_before_factory() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let other = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    reset();
    assert!(matches!(
        quote(incoming(&root, FIXTURE, 0), working(&other), limits()),
        Err(Error::Counter)
    ));
    let invalid = root
        .charge_external(0)
        .unwrap()
        .try_build(|_| Ok(FIXTURE.to_vec()))
        .unwrap();
    assert!(matches!(
        quote(invalid, working(&root), limits()),
        Err(Error::OriginalOwner)
    ));
    assert_eq!(factories(), 0);
}

#[test]
fn bootstrap_zero_one_short_and_exact_are_original_real_reservations() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let before = root.budget().used_bytes();
    reset();
    for bytes in [0, bootstrap_bytes().unwrap() - 1] {
        assert!(matches!(
            quote(
                incoming(&root, FIXTURE, 0),
                root.charge_external(bytes).unwrap(),
                limits()
            ),
            Err(Error::Bootstrap)
        ));
        assert_eq!(root.budget().used_bytes(), before);
    }
    assert_eq!(factories(), 0);
    let q = quote(incoming(&root, FIXTURE, 0), working(&root), limits()).unwrap();
    assert!(q.resources().required_working_payload > bootstrap_bytes().unwrap());
    drop(q);
    assert_eq!(root.budget().used_bytes(), before);
}

#[test]
fn complete_peak_exact_one_short_and_zero_refuse_before_factory() {
    let exact = {
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = root.budget().used_bytes();
        let q = quote(incoming(&root, FIXTURE, 83), working(&root), limits()).unwrap();
        baseline + q.resources().original_owners_peak_bytes
    };
    for (limit, success) in [(exact, true), (exact - 1, false)] {
        let root = OwnedBudget::new(limit).unwrap();
        reset();
        let q = quote(incoming(&root, FIXTURE, 83), working(&root), limits()).unwrap();
        let output = q.construct();
        assert_eq!(output.is_ok(), success);
        if !success {
            assert_eq!(factories(), 0);
        }
    }
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    reset();
    assert!(matches!(
        quote(
            incoming(&root, FIXTURE, 0),
            working(&root),
            Limits {
                max_payload_bytes: 0,
                ..limits()
            }
        ),
        Err(Error::Capacity)
    ));
    assert_eq!(factories(), 0);
}

#[test]
fn whole_work_slices_are_disjoint_prepaid_exact_and_one_short_before_factories() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let q = quote(incoming(&root, FIXTURE, 0), working(&root), limits()).unwrap();
    let work = q.resources().work;
    assert_eq!(
        work.total,
        work.quotation + work.construction + work.numeric + work.census
    );
    assert!(work.quotation > FIXTURE.len() as u128);
    assert!(work.construction > work.quotation);
    assert!(work.numeric > 0 && work.census > 0);
    drop(q);
    reset();
    assert!(matches!(
        quote(
            incoming(&root, FIXTURE, 0),
            working(&root),
            Limits {
                max_work: work.total - 1,
                ..limits()
            }
        ),
        Err(Error::WorkLimit)
    ));
    assert_eq!(factories(), 0);
    let result = quote(
        incoming(&root, FIXTURE, 0),
        working(&root),
        Limits {
            max_work: work.total,
            ..limits()
        },
    )
    .unwrap()
    .construct()
    .unwrap();
    assert_eq!(result.payload().resources().work, work);
}

#[test]
fn numeric_lexeme_and_numeric_work_ceilings_refuse_before_helper_factories() {
    let raw = b"[1.2345678901234567e-200,18446744073709551616]";
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let q = quote(incoming(&root, raw, 0), working(&root), limits()).unwrap();
    let numeric = q.resources().work.numeric;
    drop(q);
    reset();
    assert!(matches!(
        quote(
            incoming(&root, raw, 0),
            working(&root),
            Limits {
                max_number_work: numeric - 1,
                ..limits()
            }
        ),
        Err(Error::NumberWorkLimit)
    ));
    assert!(matches!(
        quote(
            incoming(&root, raw, 0),
            working(&root),
            Limits {
                max_number_bytes: 3,
                ..limits()
            }
        ),
        Err(Error::NumberLimit)
    ));
    assert_eq!(factories(), 0);
    assert_eq!(NUMBER_CALLS.with(|s| s.get()), 0);
}

#[test]
fn syntax_unicode_trailing_input_and_nonfinite_fail_with_fixed_codes_under_guards() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    for raw in [
        b"".as_slice(),
        b"{",
        b"[1,]",
        b"01",
        b"1e+",
        b"null null",
        br#""\ud800""#,
        br#""\udc00""#,
        &[b'"', 0xff, b'"'],
    ] {
        reset();
        assert!(serde_json::from_slice::<Value>(raw).is_err());
        assert!(matches!(
            quote(incoming(&root, raw, 0), working(&root), limits()),
            Err(Error::Syntax)
        ));
        assert_eq!(factories(), 0);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
    for raw in [b"1e9999".as_slice(), b"-1.7976931348623159e308"] {
        reset();
        assert!(serde_json::from_slice::<Value>(raw).is_err());
        let q = quote(incoming(&root, raw, 0), working(&root), limits()).unwrap();
        assert_eq!(factories(), 0);
        assert!(matches!(q.construct(), Err(Error::InvalidNumber)));
        assert_eq!(NUMBER_CALLS.with(|s| s.get()), 1);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn depth_node_source_and_string_gates_are_opt_in_and_preconstruction() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let deep = format!("{}0{}", "[".repeat(17), "]".repeat(17));
    reset();
    assert!(serde_json::from_str::<Value>(&deep).is_ok());
    assert!(matches!(
        quote(
            incoming(&root, deep.as_bytes(), 0),
            working(&root),
            limits()
        ),
        Err(Error::DepthLimit)
    ));
    for changed in [
        Limits {
            max_work: 0,
            ..limits()
        },
        Limits {
            max_nodes: 1,
            ..limits()
        },
        Limits {
            max_depth: 1,
            ..limits()
        },
        Limits {
            max_source_bytes: 1,
            ..limits()
        },
        Limits {
            max_string_bytes: 1,
            ..limits()
        },
    ] {
        assert!(quote(incoming(&root, FIXTURE, 0), working(&root), changed).is_err());
    }
    assert_eq!(factories(), 0);
    let at_limit = format!("{}0{}", "[".repeat(16), "]".repeat(16));
    assert_eq!(
        import(&root, at_limit.as_bytes()).payload().value(),
        &serde_json::from_str::<Value>(&at_limit).unwrap()
    );
}

#[test]
fn post_quote_counter_pressure_refuses_without_factory_and_retires_both_inputs() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    let q = quote(incoming(&root, FIXTURE, 0), working(&root), limits()).unwrap();
    let free = MAX_ALLOCATION_BYTES - root.budget().used_bytes();
    let pressure = root
        .charge_external(free - size_of::<OwnedExternalCharge>())
        .unwrap();
    reset();
    assert!(matches!(
        q.construct(),
        Err(Error::Native(NativeError::Resource))
    ));
    assert_eq!(factories(), 0);
    drop(pressure);
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn every_shared_factory_failure_and_unwind_retires_partials_before_guard_refund() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    reset();
    drop(import(&root, FIXTURE));
    let sites = factories();
    assert!(sites > 10);
    for site in 1..=sites {
        FACTORIES.with(|s| s.set((0, Some(site), None)));
        assert!(matches!(
            quote(incoming(&root, FIXTURE, 0), working(&root), limits())
                .unwrap()
                .construct(),
            Err(Error::Factory)
        ));
        assert_eq!(root.budget().used_bytes(), baseline);
        FACTORIES.with(|s| s.set((0, None, Some(site))));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = quote(incoming(&root, FIXTURE, 0), working(&root), limits())
                .unwrap()
                .construct();
        }))
        .is_err());
        assert_eq!(root.budget().used_bytes(), baseline);
    }
    reset();
}

#[test]
fn oversized_original_work_reservation_is_never_shrunk_or_replaced() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let working = root.charge_external(1 << 20).unwrap();
    let q = quote(incoming(&root, b"null", 7), working, limits()).unwrap();
    let resources = q.resources();
    assert!(resources.required_working_payload < 1 << 20);
    assert_eq!(
        resources.working_charge_bytes,
        (1 << 20) + size_of::<OwnedCharged<ImportedValue>>()
    );
    let output = q.construct().unwrap();
    assert_eq!(output.charged_bytes(), resources.working_charge_bytes);
}

#[test]
fn wide_arrays_and_overwritten_maps_are_width_priced_not_all_siblings_active() {
    let raw = serde_json::to_vec(&serde_json::json!({"config": (0..1024).map(|i| serde_json::json!({"id":i,"scale":0.125})).collect::<Vec<_>>()})).unwrap();
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let q = quote(incoming(&root, &raw, 0), working(&root), limits()).unwrap();
    let counts = q.counts();
    assert_eq!(counts.depth, 3);
    assert_eq!(counts.array_children, 1024);
    assert!(q.resources().required_working_payload < MAX_ALLOCATION_BYTES);
    assert_eq!(
        q.construct().unwrap().payload().value(),
        &serde_json::from_slice::<Value>(&raw).unwrap()
    );
}

#[test]
fn arithmetic_overflow_refuses_without_numeric_or_heap_factories() {
    let huge = SourceCounts {
        spans: usize::MAX,
        ..SourceCounts::default()
    };
    reset();
    assert_eq!(payload(huge), Err(Error::Overflow));
    assert_eq!(
        super::super::shared_build_work(
            SourceCounts {
                map_squares: u128::MAX,
                ..SourceCounts::default()
            },
            0,
            3
        ),
        Err(Error::Overflow)
    );
    assert_eq!(factories(), 0);
}

#[test]
fn original_plan_quote_work_and_payload_formulas_stay_identical_after_shared_extraction() {
    use crate::plan_input as plan;
    use pllm_models::{AppliedMethod, ModelOperation, StateTensor};
    let counts = SourceCounts {
        spans: 100,
        objects: 12,
        array_children: 17,
        map_entries: 33,
        decoded_string_bytes: 127,
        largest_string: 23,
        depth: 4,
        numbers: 3,
        number_bytes: 12,
        map_squares: 63,
        scan_work: 567,
    };
    let arrays = size_of::<String>()
        + size_of::<u64>()
        + size_of::<ModelOperation>()
        + size_of::<StateTensor>()
        + size_of::<AppliedMethod>()
        + size_of::<Value>();
    // Independent original concrete list, not the extracted helper itself.
    let frames: usize = [
        size_of::<plan::PartialPlan>(),
        size_of::<plan::PartialGraph>(),
        size_of::<plan::PartialOperation>(),
        size_of::<plan::PartialState>(),
        size_of::<plan::PartialMethod>(),
        size_of::<ExactBuffer<String>>(),
        size_of::<ExactBuffer<u64>>(),
        size_of::<ExactBuffer<ModelOperation>>(),
        size_of::<ExactBuffer<StateTensor>>(),
        size_of::<ExactBuffer<AppliedMethod>>(),
        size_of::<ExactBuffer<Value>>(),
        size_of::<ExactBuffer<u8>>(),
        size_of::<String>(),
        size_of::<serde_json::Map<String, Value>>(),
        size_of::<Value>(),
        size_of::<Result<String, Error>>(),
        size_of::<Result<Value, Error>>(),
        size_of::<Result<pllm_models::DecoderPlan, Error>>(),
        size_of::<Result<pllm_models::DecoderGraph, Error>>(),
        size_of::<Result<ModelOperation, Error>>(),
        size_of::<Result<StateTensor, Error>>(),
        size_of::<Result<AppliedMethod, Error>>(),
        size_of::<Result<Vec<String>, Error>>(),
        size_of::<Result<Vec<u64>, Error>>(),
        size_of::<Result<Vec<Value>, Error>>(),
        size_of::<Result<Vec<ModelOperation>, Error>>(),
        size_of::<Result<Vec<StateTensor>, Error>>(),
        size_of::<Result<Vec<AppliedMethod>, Error>>(),
        size_of::<std::string::FromUtf8Error>(),
        size_of::<serde::de::value::StrDeserializer<'_, Error>>(),
        size_of::<serde::de::value::StringDeserializer<Error>>(),
        size_of::<std::str::Bytes<'_>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<(&Builder<'_>, usize)>(),
        size_of::<(usize, Span, serde_json::Number)>(),
        size_of::<[u8; 4]>(),
    ]
    .into_iter()
    .sum();
    assert_eq!(plan::partial_frame_slots().unwrap(), frames);
    let expected = plan::bootstrap_bytes().unwrap()
        + size_of::<Span>() * counts.spans
        + arrays * counts.array_children
        + counts.decoded_string_bytes
        + (counts.objects + counts.map_entries)
            * pllm_models::owned_backing::checked_tree_node_bytes::<(String, Value)>().unwrap()
        + 2 * (counts.depth + 2) * frames
        + number_resources().unwrap().total().unwrap();
    assert_eq!(plan::payload(counts).unwrap(), expected);
    let fields = 8u128 + 8 + 7 + 5 + 5;
    let expected = counts.scan_work
        + counts.decoded_string_bytes as u128 * fields
        + 1000 * 3
        + counts.spans as u128 * (fields + 2 + 2 + 1)
        + counts.array_children as u128 * 2
        + counts.map_squares * (counts.largest_string as u128 + 1)
        + counts.map_squares * 2
        + (2 * plan::DEPTH + 2 * CENSUS_SLOTS) as u128;
    assert_eq!(plan::build_work(counts, 1000).unwrap(), expected);
}
