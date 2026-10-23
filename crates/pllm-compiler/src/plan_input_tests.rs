use super::*;
use pllm_garble::shared_allocation::OwnedBudget;
const TEST_LIMIT: usize = pllm_garble::shared_allocation::MAX_ALLOCATION_BYTES;

fn limits() -> Limits {
    Limits {
        max_source_bytes: 1 << 20,
        max_nodes: 100_000,
        max_depth: DEPTH,
        max_string_bytes: 1 << 16,
        max_number_bytes: 128,
        max_number_work: 1 << 26,
        max_work: 1 << 30,
        max_payload_bytes: 64 << 20,
    }
}
fn fixture() -> Vec<u8> {
    let operation = serde_json::json!({"id":"λ", "operator":"linear", "layer":null,
        "state_kind":null,"inputs":["x"],"output_shape":[1,2],
        "attributes":{"weight_id":"w","weight_shape":[2,2],"zeros":[-0.0,0.0],"u":"🦀"}});
    let graph = |mode: &str| {
        serde_json::json!({"mode":mode,"batch":1,"query_sequence":1,
        "maximum_key_sequence":8,"operations":[operation.clone()],"state_inputs":[],
        "state_outputs":[],"output":"λ"})
    };
    serde_json::to_vec(&serde_json::json!({"schema_version":"pllm.decoder_plan.v1",
        "model_family":"test","adapter":"original","config_digest":"a".repeat(64),
        "transformations":[],"prefill":graph("prefill"),"decode":graph("decode"),
        "token_feedback":false}))
    .unwrap()
}
fn incoming(root: &OwnedBudget, raw: &[u8], spare: usize) -> OwnedCharged<Vec<u8>> {
    let capacity = raw.len() + spare;
    root.charge_external(capacity)
        .unwrap()
        .try_build(|_| {
            let mut bytes = Vec::with_capacity(capacity);
            bytes.extend_from_slice(raw);
            assert_eq!(bytes.capacity(), capacity);
            Ok(bytes)
        })
        .unwrap()
}
fn work(root: &OwnedBudget) -> OwnedExternalCharge {
    root.charge_external(bootstrap_bytes().unwrap()).unwrap()
}
fn reset() {
    FACTORIES.with(|s| s.set((0, None, None)));
    NUMBER_CALLS.with(|s| s.set(0));
}
fn factories() -> usize {
    FACTORIES.with(|s| s.get().0)
}
fn import(root: &OwnedBudget, bytes: &OwnedCharged<Vec<u8>>) -> OwnedCharged<DecoderPlan> {
    quote(bytes, work(root), limits())
        .unwrap()
        .construct()
        .unwrap()
}

#[test]
fn original_serde_bytes_digest_and_signed_zero_parity() {
    reset();
    let raw = fixture();
    let expected: DecoderPlan = serde_json::from_slice(&raw).unwrap();
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &raw, 91);
    let admitted = import(&root, &source);
    assert_eq!(admitted.payload(), &expected);
    assert_eq!(
        serde_json::to_vec(admitted.payload()).unwrap(),
        serde_json::to_vec(&expected).unwrap()
    );
    assert_eq!(admitted.payload().digest(), expected.digest());
    let zero = admitted.payload().prefill.operations[0].attributes["zeros"][0]
        .as_f64()
        .unwrap();
    assert!(zero.is_sign_negative());
    assert!(std::ptr::eq(admitted.budget(), source.budget()));
}

#[test]
fn wide_siblings_keep_depth_bounded_partials_and_width_priced_backings_under_real_cap() {
    let mut value: Value = serde_json::from_slice(&fixture()).unwrap();
    let operation = value["prefill"]["operations"][0].clone();
    for mode in ["prefill", "decode"] {
        value[mode]["operations"] = Value::Array(
            (0..512)
                .map(|i| {
                    let mut op = operation.clone();
                    op["id"] = Value::String(format!("operation.{i}"));
                    op
                })
                .collect(),
        );
    }
    // Parser promises genuine schema DTOs, not graph validation/source trust.
    let bytes = serde_json::to_vec(&value).unwrap();
    let expected: DecoderPlan = serde_json::from_slice(&bytes).unwrap();
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &bytes, 73);
    let q = quote(&source, work(&root), limits()).unwrap();
    let counts = q.counts();
    assert_eq!(
        partial_frames(counts).unwrap(),
        2 * (counts.depth + 2) * partial_frame_slots().unwrap()
    );
    assert!(
        counts.spans * partial_frame_slots().unwrap() > TEST_LIMIT,
        "old all-siblings-active quotation must exceed the real hard cap"
    );
    assert!(q.required_working_payload() < TEST_LIMIT);
    let one_span = SourceCounts {
        spans: counts.spans - 1,
        ..counts
    };
    assert_eq!(
        payload(counts).unwrap() - payload(one_span).unwrap(),
        size_of::<Span>()
    );
    let one_child = SourceCounts {
        array_children: counts.array_children - 1,
        ..counts
    };
    assert!(payload(counts).unwrap() > payload(one_child).unwrap());
    assert_eq!(q.construct().unwrap().payload(), &expected);
}

#[test]
fn escaped_strings_surrogates_keys_and_external_unit_enums_match_serde() {
    reset();
    let raw = String::from_utf8(fixture())
        .unwrap()
        .replace("λ", "\\u03bb")
        .replace("🦀", "\\ud83e\\udd80");
    let mut value: Value = serde_json::from_str(&raw).unwrap();
    value["prefill"]["mode"] = serde_json::json!({"prefill":null});
    value["prefill"]["operations"][0]["operator"] = serde_json::json!({"linear":null});
    let raw = serde_json::to_vec(&value).unwrap();
    // Keep escape spellings in the actual source, not just the oracle's Value.
    let raw = String::from_utf8(raw)
        .unwrap()
        .replace("λ", "\\u03bb")
        .replace("🦀", "\\ud83e\\udd80");
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, raw.as_bytes(), 0);
    assert_eq!(
        import(&root, &source).payload(),
        &serde_json::from_str::<DecoderPlan>(&raw).unwrap()
    );
}

#[test]
fn json_duplicate_keys_last_wins_but_dto_duplicates_refuse_in_source_order() {
    reset();
    let raw = String::from_utf8(fixture()).unwrap().replace(
        "\"weight_id\":\"w\"",
        "\"weight_id\":\"old\",\"weight_id\":\"w\"",
    );
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, raw.as_bytes(), 0);
    assert_eq!(
        import(&root, &source).payload(),
        &serde_json::from_str::<DecoderPlan>(&raw).unwrap()
    );
    let raw = raw.replace(
        "\"adapter\":\"original\"",
        "\"adapter\":\"original\",\"adapter\":\"again\"",
    );
    assert!(serde_json::from_str::<DecoderPlan>(&raw)
        .unwrap_err()
        .to_string()
        .contains("duplicate field"));
    let source = incoming(&root, raw.as_bytes(), 0);
    assert!(matches!(
        quote(&source, work(&root), limits()).unwrap().construct(),
        Err(Error::DuplicateField)
    ));
}

#[test]
fn ordinary_missing_unknown_type_digest_and_optional_field_semantics() {
    let digest = "a".repeat(64);
    let cases: [(&str, &str, Error); 4] = [
        ("\"adapter\":\"original\",", "", Error::MissingField),
        (
            "\"adapter\":\"original\"",
            "\"alien\":\"original\"",
            Error::UnknownField,
        ),
        ("\"batch\":1", "\"batch\":true", Error::Schema),
        (&digest, "invalid", Error::Schema),
    ];
    for (from, to, error) in cases {
        reset();
        let raw = String::from_utf8(fixture()).unwrap().replace(from, to);
        assert!(serde_json::from_str::<DecoderPlan>(&raw).is_err());
        let root = OwnedBudget::new(TEST_LIMIT).unwrap();
        let source = incoming(&root, raw.as_bytes(), 0);
        assert!(
            matches!(quote(&source,work(&root),limits()).unwrap().construct(),Err(e) if e == error)
        );
    }
    let raw = String::from_utf8(fixture())
        .unwrap()
        .replace("\"layer\":null,", "")
        .replace("\"state_kind\":null,", "");
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, raw.as_bytes(), 0);
    assert_eq!(
        import(&root, &source).payload(),
        &serde_json::from_str::<DecoderPlan>(&raw).unwrap()
    );
}

#[test]
fn typed_dimensions_reject_noninteger_numbers_without_coercion() {
    for number in [
        "1.0",
        "1.25",
        "1e3",
        "-0",
        "0.0",
        "18446744073709551616",
        "-9223372036854775809",
    ] {
        reset();
        let raw = String::from_utf8(fixture())
            .unwrap()
            .replace("\"batch\":1", &format!("\"batch\":{number}"));
        let root = OwnedBudget::new(TEST_LIMIT).unwrap();
        let source = incoming(&root, raw.as_bytes(), 0);
        assert!(serde_json::from_str::<DecoderPlan>(&raw).is_err());
        assert!(matches!(
            quote(&source, work(&root), limits()).unwrap().construct(),
            Err(Error::Schema)
        ));
        assert!(NUMBER_CALLS.with(|s| s.get()) > 0);
    }
}

#[test]
fn malformed_unicode_grammar_and_utf8_match_normal_json_rejection() {
    for raw in [
        b"[1,]".as_slice(),
        b"{\"a\":}",
        b"{\"a\":0,}",
        b"01",
        b"[true false]",
        b"\"\\ud800\"",
        b"\"\\udc00\"",
        b"\"\\uZZZZ\"",
        b"\"\xff\"",
        b"true false",
    ] {
        reset();
        assert!(serde_json::from_slice::<Value>(raw).is_err());
        let root = OwnedBudget::new(TEST_LIMIT).unwrap();
        let source = incoming(&root, raw, 0);
        assert!(matches!(
            quote(&source, work(&root), limits()),
            Err(Error::Syntax)
        ));
        assert_eq!(factories(), 0);
    }
}

#[test]
fn depth_source_nodes_strings_and_zero_work_fail_without_factory() {
    let raw = fixture();
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &raw, 0);
    for changed in [
        Limits {
            max_work: 0,
            ..limits()
        },
        Limits {
            max_depth: 1,
            ..limits()
        },
        Limits {
            max_nodes: 1,
            ..limits()
        },
        Limits {
            max_string_bytes: 1,
            ..limits()
        },
        Limits {
            max_source_bytes: raw.len() - 1,
            ..limits()
        },
    ] {
        reset();
        assert!(quote(&source, work(&root), changed).is_err());
        assert_eq!(factories(), 0);
    }
}

#[test]
fn bootstrap_zero_exact_and_one_short_are_real_reservations() {
    reset();
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &fixture(), 0);
    for bytes in [0, bootstrap_bytes().unwrap() - 1] {
        let charge = root.charge_external(bytes).unwrap();
        assert!(matches!(
            quote(&source, charge, limits()),
            Err(Error::Bootstrap)
        ));
    }
    assert_eq!(factories(), 0);
    let quoted = quote(&source, work(&root), limits()).unwrap();
    assert!(quoted.required_working_payload() > bootstrap_bytes().unwrap());
}

#[test]
fn full_boundary_exact_and_one_short_before_any_factory() {
    let raw = fixture();
    let (exact, required) = {
        let root = OwnedBudget::new(TEST_LIMIT).unwrap();
        let source = incoming(&root, &raw, 0);
        let q = quote(&source, work(&root), limits()).unwrap();
        let required = q.required_working_payload();
        let inline = size_of::<OwnedCharged<DecoderPlan>>() - size_of::<OwnedExternalCharge>();
        (
            root.budget().used_bytes() + required - bootstrap_bytes().unwrap() + inline,
            required,
        )
    };
    for (total, success) in [(exact, true), (exact - 1, false)] {
        reset();
        let root = OwnedBudget::new(total).unwrap();
        let source = incoming(&root, &raw, 0);
        let q = quote(&source, work(&root), limits()).unwrap();
        assert_eq!(q.required_working_payload(), required);
        let result = q.construct();
        assert_eq!(result.is_ok(), success);
        if !success {
            assert_eq!(factories(), 0);
        }
    }
}

#[test]
fn full_work_exact_and_one_short_reject_before_factories() {
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &fixture(), 0);
    let bound = quote(&source, work(&root), limits())
        .unwrap()
        .total_work_bound()
        .unwrap();
    reset();
    assert!(matches!(
        quote(
            &source,
            work(&root),
            Limits {
                max_work: bound - 1,
                ..limits()
            }
        ),
        Err(Error::WorkLimit)
    ));
    assert_eq!(factories(), 0);
    let result = quote(
        &source,
        work(&root),
        Limits {
            max_work: bound,
            ..limits()
        },
    )
    .unwrap()
    .construct()
    .unwrap();
    assert!(std::ptr::eq(result.budget(), source.budget()));
}

#[test]
fn actual_spare_capacity_wrong_counter_and_undercharged_owner() {
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let other = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &fixture(), 1000);
    reset();
    assert!(matches!(
        quote(&source, work(&other), limits()),
        Err(Error::Counter)
    ));
    let raw = fixture();
    let invalid = root
        .charge_external(0)
        .unwrap()
        .try_build(|_| Ok(raw))
        .unwrap();
    assert!(matches!(
        quote(&invalid, work(&root), limits()),
        Err(Error::OriginalOwner)
    ));
    assert_eq!(factories(), 0);
    let quoted = quote(&source, work(&root), limits()).unwrap();
    assert_eq!(
        quoted.original_capacity_bytes(),
        source.payload().capacity()
    );
    assert!(quoted.original_capacity_bytes() > source.payload().len());
    let expected_charge = quoted.working_charge_bytes().unwrap();
    let owned_peak = quoted.original_owners_peak_bytes().unwrap();
    let output = quoted.construct().unwrap();
    assert_eq!(output.charged_bytes(), expected_charge);
    assert_eq!(owned_peak, output.charged_bytes() + source.charged_bytes());
}

#[test]
fn pressure_after_quote_refuses_atomically_without_factory() {
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &fixture(), 0);
    let before = root.budget().used_bytes();
    let q = quote(&source, work(&root), limits()).unwrap();
    let free = TEST_LIMIT - root.budget().used_bytes();
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
    assert_eq!(root.budget().used_bytes(), before);
}

#[test]
fn every_external_factory_failure_and_panic_refunds_after_cleanup() {
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &fixture(), 0);
    reset();
    drop(import(&root, &source));
    let sites = factories();
    assert!(sites > 10);
    let before = root.budget().used_bytes();
    for site in 1..=sites {
        FACTORIES.with(|s| s.set((0, Some(site), None)));
        assert!(matches!(
            quote(&source, work(&root), limits()).unwrap().construct(),
            Err(Error::Factory)
        ));
        assert_eq!(root.budget().used_bytes(), before);
        FACTORIES.with(|s| s.set((0, None, Some(site))));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = quote(&source, work(&root), limits()).unwrap().construct();
        }))
        .is_err());
        assert_eq!(root.budget().used_bytes(), before);
    }
    reset();
}

#[test]
fn output_outlives_raw_and_root_without_counter_recreation() {
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &fixture(), 0);
    reset();
    let output = import(&root, &source);
    let counter = std::ptr::from_ref(output.budget());
    let charged = output.charged_bytes();
    drop(source);
    drop(root);
    assert_eq!(counter, std::ptr::from_ref(output.budget()));
    assert!(output.budget().used_bytes() >= charged);
    assert_eq!(output.payload().model_family, "test");
}

#[test]
fn arithmetic_refusal_and_meter_overflow_are_atomic() {
    assert_eq!(mul(usize::MAX, 2), Err(Error::Overflow));
    let mut meter = Meter {
        used: u128::MAX,
        limit: u128::MAX,
    };
    assert_eq!(meter.pay(1), Err(Error::Overflow));
    assert_eq!(meter.used, u128::MAX);
    assert_eq!(
        layout_bytes::<Span>(usize::MAX).map_err(|_| Error::Overflow),
        Err(Error::Overflow)
    );
}

#[test]
fn source_length_work_cannot_be_hidden_by_whitespace_or_short_dto() {
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let mut raw = fixture();
    raw.extend(std::iter::repeat_n(b' ', 10000));
    let source = incoming(&root, &raw, 0);
    reset();
    assert!(matches!(
        quote(
            &source,
            work(&root),
            Limits {
                max_work: raw.len() as u128 - 1,
                ..limits()
            }
        ),
        Err(Error::WorkLimit)
    ));
    assert_eq!(factories(), 0);
}

#[test]
fn every_dto_array_has_exact_original_schema_backing() {
    let mut value: Value = serde_json::from_slice(&fixture()).unwrap();
    value["transformations"] = serde_json::json!([{"component":"shared", "implementation":"original",
        "method_id":"method", "input_digest":"a".repeat(64), "configuration_digest":"b".repeat(64)}]);
    let state =
        serde_json::json!({"id":"cache","layer":1,"kind":"key","shape":[1,2],"maximum_sequence":8});
    value["prefill"]["state_outputs"] = serde_json::json!([state.clone()]);
    value["decode"]["state_inputs"] = serde_json::json!([state]);
    value["prefill"]["operations"][0]["attributes"]["bounds"] =
        serde_json::json!([u64::MAX, i64::MIN]);
    let raw = serde_json::to_vec(&value).unwrap();
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &raw, 0);
    reset();
    let output = import(&root, &source);
    let plan = output.payload();
    assert_eq!(plan, &serde_json::from_slice::<DecoderPlan>(&raw).unwrap());
    assert_eq!(plan.transformations.capacity(), plan.transformations.len());
    for graph in [&plan.prefill, &plan.decode] {
        assert_eq!(graph.operations.capacity(), graph.operations.len());
        assert_eq!(graph.state_inputs.capacity(), graph.state_inputs.len());
        assert_eq!(graph.state_outputs.capacity(), graph.state_outputs.len());
        for op in &graph.operations {
            assert_eq!(op.inputs.capacity(), op.inputs.len());
            assert_eq!(op.output_shape.capacity(), op.output_shape.len());
        }
    }
}

#[test]
fn integer_negative_zero_keeps_serde_float_representation() {
    let raw = String::from_utf8(fixture())
        .unwrap()
        .replace("[-0.0,0.0]", "[-0,0.0]");
    assert!(raw.contains("[-0,0.0]"));
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, raw.as_bytes(), 0);
    reset();
    let output = import(&root, &source);
    assert_eq!(
        output.payload(),
        &serde_json::from_str::<DecoderPlan>(&raw).unwrap()
    );
    assert!(output.payload().prefill.operations[0].attributes["zeros"][0].is_f64());
}

fn numeric_fixture(spelling: &str) -> Vec<u8> {
    String::from_utf8(fixture())
        .unwrap()
        .replace(
            "\"weight_id\":\"w\"",
            &format!("\"number\":{spelling},\"weight_id\":\"w\""),
        )
        .into_bytes()
}

#[test]
fn serde_controls_all_finite_rounding_kinds_and_boundaries() {
    for spelling in [
        "0.000001",
        "1e-6",
        "10000.0",
        "1.4142135623730951",
        "0.1",
        "1.23456789012345678901234567890123456789",
        "1e+20",
        "1e-300",
        "5e-324",
        "4.9406564584124654e-324",
        "2.2250738585072014e-308",
        "1.7976931348623157e308",
        "-1.7976931348623157e308",
        "1e-99999999",
        "0e999999999999999999999",
        "0",
        "-0",
        "0.0",
        "-0.0",
        "-0e10",
        "18446744073709551615",
        "18446744073709551616",
        "-9223372036854775808",
        "-9223372036854775809",
        "9007199254740993.0",
    ] {
        reset();
        let raw = numeric_fixture(spelling);
        let ordinary: DecoderPlan = serde_json::from_slice(&raw).unwrap();
        let root = OwnedBudget::new(TEST_LIMIT).unwrap();
        let source = incoming(&root, &raw, 7);
        let quoted = quote(&source, work(&root), limits()).unwrap();
        assert_eq!(
            NUMBER_CALLS.with(|calls| calls.get()),
            0,
            "scan parsed {spelling}"
        );
        let plan = quoted.construct().unwrap();
        assert_eq!(
            serde_json::to_vec(plan.payload()).unwrap(),
            serde_json::to_vec(&ordinary).unwrap(),
            "{spelling}"
        );
        assert_eq!(plan.payload().digest(), ordinary.digest(), "{spelling}");
        let expected = &ordinary.prefill.operations[0].attributes["number"];
        let actual = &plan.payload().prefill.operations[0].attributes["number"];
        assert_eq!(actual.is_f64(), expected.is_f64(), "kind {spelling}");
        assert_eq!(
            actual.as_f64().map(f64::to_bits),
            expected.as_f64().map(f64::to_bits),
            "bits {spelling}"
        );
        assert!(NUMBER_CALLS.with(|calls| calls.get()) > 0);
    }
}

#[test]
fn nonfinite_and_overflow_errors_retire_under_original_guard_without_formatting() {
    for spelling in ["1e309", "-1e309", "1e99999999999999999999"] {
        reset();
        let raw = numeric_fixture(spelling);
        assert!(serde_json::from_slice::<DecoderPlan>(&raw).is_err());
        let root = OwnedBudget::new(TEST_LIMIT).unwrap();
        let source = incoming(&root, &raw, 0);
        let baseline = root.budget().used_bytes();
        let quoted = quote(&source, work(&root), limits()).unwrap(); // grammar only
        assert_eq!(NUMBER_CALLS.with(|calls| calls.get()), 0);
        assert!(matches!(quoted.construct(), Err(Error::InvalidNumber)));
        assert!(NUMBER_CALLS.with(|calls| calls.get()) > 0);
        assert_eq!(root.budget().used_bytes(), baseline);
    }
}

#[test]
fn number_grammar_rejects_malformed_spelling_without_numeric_helper() {
    for spelling in ["1.", "1e", "1e+", "+1", "01", "--1", "NaN", "Infinity"] {
        reset();
        let raw = numeric_fixture(spelling);
        assert!(serde_json::from_slice::<DecoderPlan>(&raw).is_err());
        let root = OwnedBudget::new(TEST_LIMIT).unwrap();
        let source = incoming(&root, &raw, 0);
        assert!(matches!(
            quote(&source, work(&root), limits()),
            Err(Error::Syntax)
        ));
        assert_eq!(factories(), 0);
        assert_eq!(NUMBER_CALLS.with(|calls| calls.get()), 0);
    }
}

#[test]
fn numeric_length_and_prepaid_work_exact_short_before_factory() {
    let spelling =
        "1.234567890123456789012345678901234567890123456789012345678901234567890123456789";
    let raw = numeric_fixture(spelling);
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &raw, 0);
    reset();
    assert!(matches!(
        quote(
            &source,
            work(&root),
            Limits {
                max_number_bytes: spelling.len() - 1,
                ..limits()
            }
        ),
        Err(Error::NumberLimit)
    ));
    assert_eq!(factories(), 0);
    assert_eq!(NUMBER_CALLS.with(|calls| calls.get()), 0);
    let quoted = quote(
        &source,
        work(&root),
        Limits {
            max_number_bytes: spelling.len(),
            ..limits()
        },
    )
    .unwrap();
    let numeric = numeric_work(quoted.counts(), limits()).unwrap();
    drop(quoted);
    assert!(matches!(
        quote(
            &source,
            work(&root),
            Limits {
                max_number_work: numeric - 1,
                ..limits()
            }
        ),
        Err(Error::NumberWorkLimit)
    ));
    assert_eq!(factories(), 0);
    let result = quote(
        &source,
        work(&root),
        Limits {
            max_number_work: numeric,
            ..limits()
        },
    )
    .unwrap()
    .construct()
    .unwrap();
    assert_eq!(
        result.payload(),
        &serde_json::from_slice::<DecoderPlan>(&raw).unwrap()
    );
}

#[test]
fn numeric_envelope_is_reserved_before_parser_entry_and_factory_failure() {
    let raw = numeric_fixture("0.000001");
    let root = OwnedBudget::new(TEST_LIMIT).unwrap();
    let source = incoming(&root, &raw, 0);
    reset();
    let quoted = quote(&source, work(&root), limits()).unwrap();
    let required = quoted.required_working_payload();
    drop(quoted);
    assert!(matches!(
        quote(
            &source,
            work(&root),
            Limits {
                max_payload_bytes: required - 1,
                ..limits()
            }
        ),
        Err(Error::Capacity)
    ));
    assert_eq!(factories(), 0);
    assert_eq!(NUMBER_CALLS.with(|calls| calls.get()), 0);
    let profile = number_resources().unwrap();
    assert_eq!(profile.scratch_backing, 0); // audited default/std, NOT other features
    assert!(profile.error_backing_policy >= size_of::<(Box<str>, std::io::Error, usize, usize)>());
    assert!(profile.total().unwrap() > profile.error_backing_policy);
    // Local prepaid number allowance cannot borrow unused construction credit.
    let mut local = Meter {
        used: 0,
        limit: number_work(8, 1).unwrap() - 1,
    };
    assert_eq!(local.pay(number_work(8, 1).unwrap()), Err(Error::WorkLimit));
    assert_eq!(local.used, 0);

    let exact_total = {
        let probe = OwnedBudget::new(TEST_LIMIT).unwrap();
        let bytes = incoming(&probe, &raw, 0);
        let quoted = quote(&bytes, work(&probe), limits()).unwrap();
        probe.budget().used_bytes() + quoted.required_working_payload() - bootstrap_bytes().unwrap()
            + size_of::<OwnedCharged<DecoderPlan>>()
            - size_of::<OwnedExternalCharge>()
    };
    for (cap, succeeds) in [(exact_total, true), (exact_total - 1, false)] {
        let bounded = OwnedBudget::new(cap).unwrap();
        let bytes = incoming(&bounded, &raw, 0);
        reset();
        let result = quote(&bytes, work(&bounded), limits()).unwrap().construct();
        assert_eq!(result.is_ok(), succeeds);
        if !succeeds {
            assert_eq!(factories(), 0);
            assert_eq!(NUMBER_CALLS.with(|calls| calls.get()), 0);
        }
    }
}
