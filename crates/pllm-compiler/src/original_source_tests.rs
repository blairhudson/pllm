use super::*;
use crate::plan_input::json;
use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};

const RAW: &[u8] = br#"{"model_type":"qwen2","hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":7,"max_position_embeddings":8,"hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true,"use_sliding_window":false}"#;
const BAD_EPSILON: &[u8] = br#"{"model_type":"qwen2","hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":7,"max_position_embeddings":8,"hidden_act":"silu","rms_norm_eps":-1,"rope_theta":10000,"tie_word_embeddings":true,"use_sliding_window":false}"#;

fn limits() -> Limits {
    Limits {
        expansion: DecoderExpansionBudget {
            max_layers: 8,
            max_feature_width: 128,
            max_attention_heads: 8,
            max_vocabulary: 128,
            max_tensor_elements: 32768,
        },
        factory: ConfigLowererLimits {
            max_operations_per_phase: 256,
            max_string_bytes: 1024,
            max_attribute_nodes: 32768,
            max_retained_strings: 1 << 20,
            max_factory_bytes: 64 << 20,
        },
        max_work: 1_000_000_000_000,
        max_working_payload: 128 << 20,
    }
}

fn workload(tokens: u64) -> DecoderWorkload {
    DecoderWorkload {
        batch: 1,
        max_input_tokens: tokens,
        max_new_tokens: 2,
    }
}

fn imported(root: &OwnedBudget, raw: &[u8]) -> OwnedCharged<ImportedValue> {
    // Original byte factory itself runs under the current genuine counter; no
    // detached pre-existing Vec or serialized DecoderPlan is blessed afterward.
    let original = root
        .charge_external(raw.len())
        .unwrap()
        .try_build(|_| Ok(raw.to_vec()))
        .unwrap();
    imported_owned(root, original)
}

fn imported_owned(
    root: &OwnedBudget,
    original: OwnedCharged<Vec<u8>>,
) -> OwnedCharged<ImportedValue> {
    let working = root
        .charge_external(json::bootstrap_bytes().unwrap())
        .unwrap();
    json::quote(
        original,
        working,
        json::Limits {
            max_source_bytes: 32768,
            max_nodes: 512,
            max_depth: 16,
            max_string_bytes: 16384,
            max_number_bytes: 64,
            max_number_work: 1_000_000,
            max_work: 100_000_000,
            max_payload_bytes: 4 << 20,
        },
    )
    .unwrap()
    .construct()
    .unwrap()
}

fn prepared(root: &OwnedBudget, raw: &[u8], tokens: u64, limits: Limits) -> SourceQuote {
    let source = imported(root, raw);
    let bytes = bootstrap_bytes(&source, limits).unwrap();
    quote(
        source,
        root.charge_external(bytes).unwrap(),
        workload(tokens),
        limits,
    )
    .unwrap()
}

fn attempts() -> usize {
    FACTORIES.with(|n| n.get())
}

#[test]
fn genuine_original_factory_root_and_growing_plan_match_independent_byte_oracle() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    let before = first.resources().prepaid_work;
    let original4 =
        pllm_models::lower_model_json_with_budget(RAW, workload(4), limits().expansion).unwrap();
    assert_eq!(first.plan(), &original4);
    assert_eq!(first.raw_configuration(), RAW);
    assert_eq!(first.plan().prefill.query_sequence, 4);
    let control = bootstrap_bytes(first.configuration(), limits()).unwrap();
    let second = first
        .lower(
            root.charge_external(control).unwrap(),
            workload(5),
            limits(),
        )
        .unwrap();
    let original5 =
        pllm_models::lower_model_json_with_budget(RAW, workload(5), limits().expansion).unwrap();
    assert_eq!(second.plan(), &original5);
    assert_eq!(second.plan().prefill.query_sequence, 5);
    assert!(first.same_origin(&second));
    assert!(!first.same_plan(&second));
    assert!(std::ptr::eq(first.budget(), second.budget()));
    assert!(second.resources().prepaid_work > before);
    // This is NOT four tokens plus accepted feedback. No request epoch,
    // executable import or TokenFeedback issuer was produced by this operation.
}

#[test]
fn foreign_equal_raw_digest_and_shapes_never_share_origin() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    let foreign = prepared(&root, RAW, 4, limits()).construct().unwrap();
    assert_eq!(first.plan(), foreign.plan());
    assert_eq!(first.plan().digest(), foreign.plan().digest());
    assert_eq!(first.raw_configuration(), foreign.raw_configuration());
    assert!(!first.same_origin(&foreign));
    assert!(!first.same_plan(&foreign));
}

#[test]
fn exact_and_one_short_work_are_checked_before_real_factory() {
    let probe = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let quoted = prepared(&probe, RAW, 4, limits());
    let total = quoted.resources().prepaid_work;
    assert!(total > 0);
    drop(quoted);
    let mut exact = limits();
    exact.max_work = total;
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let plan = prepared(&root, RAW, 4, exact).construct().unwrap();
    assert_eq!(plan.resources().prepaid_work, total);

    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = imported(&root, RAW);
    let bytes = bootstrap_bytes(&source, limits()).unwrap();
    let mut short = exact;
    short.max_work -= 1;
    let before = attempts();
    assert!(quote(
        source,
        root.charge_external(bytes).unwrap(),
        workload(4),
        short
    )
    .is_err());
    assert_eq!(attempts(), before);
}

#[test]
fn exact_and_one_short_actual_working_envelope_before_factory() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let quoted = prepared(&root, RAW, 4, limits());
    let required = quoted.resources().required_working_payload;
    drop(quoted);
    let mut exact = limits();
    exact.max_working_payload = required;
    let plan = prepared(&root, RAW, 4, exact).construct().unwrap();
    assert!(plan.resources().working_charge_bytes >= required);
    drop(plan);

    let source = imported(&root, RAW);
    let control = bootstrap_bytes(&source, exact).unwrap();
    let mut short = exact;
    short.max_working_payload -= 1;
    let before = attempts();
    assert!(quote(
        source,
        root.charge_external(control).unwrap(),
        workload(4),
        short
    )
    .is_err());
    assert_eq!(attempts(), before);
}

#[test]
fn current_work_and_source_siblings_cannot_reset_or_replay_construction_quota() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    let control = bootstrap_bytes(first.configuration(), limits()).unwrap();
    let second = first
        .lower(
            root.charge_external(control).unwrap(),
            workload(5),
            limits(),
        )
        .unwrap();
    let before = attempts();
    let mut exhausted = limits();
    exhausted.max_work = second.resources().prepaid_work;
    // Old prefix handle must see the later prefix's actual prepaid work.
    assert!(first
        .lower(
            root.charge_external(control).unwrap(),
            workload(5),
            exhausted
        )
        .is_err());
    assert_eq!(attempts(), before);
    assert!(matches!(
        second.lower(
            root.charge_external(control).unwrap(),
            workload(5),
            limits()
        ),
        Err(Error::Burned)
    ));
    assert_eq!(attempts(), before);
}

#[test]
fn foreign_counter_rejected_before_quotation_or_factory() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = imported(&root, RAW);
    let control = bootstrap_bytes(&source, limits()).unwrap();
    let before = attempts();
    assert!(matches!(
        quote(
            source,
            foreign.charge_external(control).unwrap(),
            workload(4),
            limits()
        ),
        Err(Error::Counter)
    ));
    assert_eq!(attempts(), before);
}

#[test]
fn original_factory_failure_issues_no_owner_and_burns_source_derivation() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    let quoted = prepared(&root, BAD_EPSILON, 4, limits());
    let before = attempts();
    assert!(matches!(quoted.construct(), Err(Error::Factory)));
    assert_eq!(attempts(), before + 1);
    assert_eq!(root.budget().used_bytes(), baseline);

    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    let control = bootstrap_bytes(first.configuration(), limits()).unwrap();
    let before = attempts();
    // Genuine original max_position_embeddings=8 rejects total positions 9.
    assert!(matches!(
        first.lower(
            root.charge_external(control).unwrap(),
            workload(7),
            limits()
        ),
        Err(Error::Factory)
    ));
    assert_eq!(attempts(), before + 1);
    assert!(matches!(
        first.lower(
            root.charge_external(control).unwrap(),
            workload(5),
            limits()
        ),
        Err(Error::Burned)
    ));
    assert_eq!(attempts(), before + 1);
}

#[test]
fn actual_owned_source_and_initial_plan_survive_root_handle_and_parent_plan_drop() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let monitor = root.try_clone().unwrap();
    let baseline = monitor.budget().used_bytes();
    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    let control = bootstrap_bytes(first.configuration(), limits()).unwrap();
    let second = first
        .lower(
            root.charge_external(control).unwrap(),
            workload(5),
            limits(),
        )
        .unwrap();
    assert_eq!(Arc::strong_count(second.root()), 2);
    drop(first);
    drop(root);
    assert_eq!(Arc::strong_count(second.root()), 1);
    assert_eq!(second.raw_configuration(), RAW);
    assert_eq!(second.plan().prefill.query_sequence, 5);
    assert!(monitor.budget().used_bytes() > baseline);
    drop(second);
    assert_eq!(
        monitor.budget().used_bytes(),
        baseline - size_of::<OwnedBudget>()
    );
}

#[test]
fn cancellation_and_unwind_retire_payload_before_original_guards() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    let pending = prepared(&root, RAW, 4, limits());
    let before = attempts();
    drop(pending);
    assert_eq!(attempts(), before);
    assert_eq!(root.budget().used_bytes(), baseline);

    let pending = prepared(&root, RAW, 4, limits());
    PANIC_FACTORY.with(|p| p.set(true));
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pending.construct()));
    assert!(failed.is_err());
    assert_eq!(attempts(), before + 1);
    assert_eq!(root.budget().used_bytes(), baseline);

    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    let control = bootstrap_bytes(first.configuration(), limits()).unwrap();
    let charged = root.budget().used_bytes();
    let before = attempts();
    let working = root.charge_external(control).unwrap();
    PANIC_FACTORY.with(|p| p.set(true));
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        first.lower(working, workload(5), limits())
    }));
    assert!(failed.is_err());
    assert_eq!(attempts(), before + 1);
    assert_eq!(root.budget().used_bytes(), charged);
    assert!(matches!(
        first.lower(
            root.charge_external(control).unwrap(),
            workload(5),
            limits()
        ),
        Err(Error::Burned)
    ));
    assert_eq!(attempts(), before + 1);
    drop(first);
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn original_long_string_payload_is_not_a_cursor_work_allowance() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let suffix = br#", "ignored_training_metadata":""#;
    let bytes = RAW.len() - 1 + suffix.len() + 8192 + 2;
    let original = root
        .charge_external(bytes)
        .unwrap()
        .try_build(|_| {
            let mut raw = Vec::with_capacity(bytes);
            raw.extend_from_slice(&RAW[..RAW.len() - 1]);
            raw.extend_from_slice(suffix);
            raw.resize(raw.len() + 8192, b'x');
            raw.extend_from_slice(b"\"}");
            assert_eq!(raw.len(), bytes);
            assert_eq!(raw.capacity(), bytes);
            Ok(raw)
        })
        .unwrap();
    let source = imported_owned(&root, original);
    let value = source.payload();
    assert!(value.counts().decoded_string_bytes as u128 > value.resources().work.census);
    assert!(value.traversal_payload_bound().unwrap() > value.resources().work.census as usize);
    let mut policy = limits();
    policy.factory.max_string_bytes = 16384;
    let control = bootstrap_bytes(&source, policy).unwrap();
    let quoted = quote(
        source,
        root.charge_external(control).unwrap(),
        workload(4),
        policy,
    )
    .unwrap();
    let plan = quoted.construct().unwrap();
    let original = pllm_models::lower_model_json_with_budget(
        plan.raw_configuration(),
        workload(4),
        policy.expansion,
    )
    .unwrap();
    assert_eq!(plan.plan(), &original);
    assert!(plan.raw_configuration().len() > 8192);
}

#[test]
fn captured_workload_and_bootstrap_query_preserve_actual_factory_origin() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    assert_eq!(first.workload(), workload(4));
    let charged = root.budget().used_bytes();
    let before = attempts();
    let work = first
        .root()
        .payload()
        .ledger
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .used;
    let bytes = first.lower_bootstrap_bytes(limits()).unwrap();
    assert_eq!(root.budget().used_bytes(), charged);
    assert_eq!(attempts(), before);
    assert_eq!(
        first
            .root()
            .payload()
            .ledger
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .used,
        work
    );
    let next = DecoderWorkload {
        batch: 1,
        max_input_tokens: 5,
        max_new_tokens: 1,
    };
    let second = first
        .lower(root.charge_external(bytes).unwrap(), next, limits())
        .unwrap();
    assert_eq!(second.workload(), next);
    assert_eq!(first.workload(), workload(4));
    assert!(first.same_origin(&second));
    assert!(!first.same_plan(&second));
    assert_eq!(second.plan().prefill.query_sequence, 5);
    assert_eq!(attempts(), before + 1);
}

#[test]
fn bootstrap_uses_actual_remaining_shared_work_not_older_handle_summary() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    let bytes = first.lower_bootstrap_bytes(limits()).unwrap();
    let second = first
        .lower(root.charge_external(bytes).unwrap(), workload(5), limits())
        .unwrap();
    assert!(second.resources().prepaid_work > first.resources().prepaid_work);
    let mut exhausted = limits();
    exhausted.max_work = second.resources().prepaid_work;
    let before = attempts();
    assert!(matches!(
        first.lower_bootstrap_bytes(exhausted),
        Err(Error::Work)
    ));
    assert!(matches!(
        second.lower_bootstrap_bytes(exhausted),
        Err(Error::Work)
    ));
    assert_eq!(attempts(), before);
    assert!(
        first.lower_bootstrap_bytes(limits()).is_ok(),
        "pure failed query must not burn source"
    );
}

#[test]
fn returned_bootstrap_one_short_fails_before_original_factory_and_burns_transition() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    let bytes = first.lower_bootstrap_bytes(limits()).unwrap();
    let before = attempts();
    assert!(matches!(
        first.lower(
            root.charge_external(bytes - 1).unwrap(),
            workload(5),
            limits()
        ),
        Err(Error::Capacity)
    ));
    assert_eq!(attempts(), before);
    assert!(matches!(
        first.lower_bootstrap_bytes(limits()),
        Err(Error::Burned)
    ));
}
