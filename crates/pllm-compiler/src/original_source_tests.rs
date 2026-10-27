use super::*;
use crate::plan_input::json;
use pllm_garble::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};

const RAW: &[u8] = br#"{"model_type":"qwen2","hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":7,"max_position_embeddings":8,"hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true,"use_sliding_window":false}"#;
const BAD_EPSILON: &[u8] = br#"{"model_type":"qwen2","hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":7,"max_position_embeddings":8,"hidden_act":"silu","rms_norm_eps":-1,"rope_theta":10000,"tie_word_embeddings":true,"use_sliding_window":false}"#;

#[test]
fn compiler_metadata_uses_actual_shared_quota_and_prequoted_transition_slots() {
    let root = OwnedBudget::new(pllm_garble::shared_allocation::MAX_ALLOCATION_BYTES).unwrap();
    let first = prepared(&root, RAW, 4, limits()).construct().unwrap();
    let control = OriginalSourcePlan::compiler_metadata_control_bytes().unwrap();
    type G<'a> = std::sync::MutexGuard<'a, Ledger>;
    let literal = size_of::<G<'_>>()
        + size_of::<Result<G<'_>, std::sync::PoisonError<G<'_>>>>()
        + size_of::<&OwnedExternalCharge>()
        + size_of::<&Arc<OwnedCharged<Context>>>()
        + size_of::<Result<u64, Error>>()
        + size_of::<Option<u64>>()
        + 3 * size_of::<u64>();
    assert_eq!(control, literal);
    let before = first
        .root()
        .payload()
        .ledger
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .used;
    assert_eq!(
        first.prepay_compiler_metadata(&root.charge_external(control - 1).unwrap(), 1),
        Err(Error::Capacity)
    );
    let foreign = OwnedBudget::new(pllm_garble::shared_allocation::MAX_ALLOCATION_BYTES).unwrap();
    assert_eq!(
        first.prepay_compiler_metadata(&foreign.charge_external(control).unwrap(), 1),
        Err(Error::Counter)
    );
    let guard = root.charge_external(control).unwrap();
    first.prepay_compiler_metadata(&guard, 7).unwrap();
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
        before + 7
    );
    let remaining = limits().max_work - before - 7;
    first.prepay_compiler_metadata(&guard, remaining).unwrap();
    assert_eq!(first.prepay_compiler_metadata(&guard, 1), Err(Error::Work));
    assert_eq!(first.lower_bootstrap_bytes(limits()), Err(Error::Burned));
}

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

// Ordinary independent structural oracle. It never constructs a production
// Header or blesses a digest/parsed plan as source custody.
fn assert_actual_header(source: &OriginalSourcePlan) {
    fn json_counts(value: &serde_json::Value, totals: &mut [usize; 5]) {
        totals[4] += 1;
        match value {
            serde_json::Value::Array(values) => {
                totals[0] += 1;
                totals[1] += values.len();
                for value in values { json_counts(value, totals); }
            }
            serde_json::Value::Object(values) => {
                totals[0] += 1;
                totals[1] += values.len();
                for (key, value) in values {
                    totals[3] += key.len();
                    json_counts(value, totals);
                }
            }
            serde_json::Value::String(text) => totals[2] += text.len(),
            _ => {}
        }
    }
    let plan = source.plan();
    let header = source.header();
    let mut operations = 0;
    let mut states = 0;
    let mut inputs = 0;
    let mut dimensions = 0;
    let mut containers = 1;
    let mut text = plan.schema_version.len() + plan.model_family.len()
        + plan.adapter.len() + plan.config_digest.as_str().len();
    for method in &plan.transformations {
        text += method.component.len() + method.implementation.len() + method.method_id.len()
            + method.input_digest.as_str().len() + method.configuration_digest.as_str().len();
    }
    let mut json = [0; 5];
    for graph in [&plan.prefill, &plan.decode] {
        operations += graph.operations.len();
        states += graph.state_inputs.len() + graph.state_outputs.len();
        containers += 3;
        text += graph.output.len();
        for operation in &graph.operations {
            containers += 1;
            inputs += operation.inputs.len();
            dimensions += operation.output_shape.len();
            text += operation.id.len() + operation.inputs.iter().map(String::len).sum::<usize>();
            json_counts(&operation.attributes, &mut json);
        }
        for state in graph.state_inputs.iter().chain(&graph.state_outputs) {
            text += state.id.len();
            dimensions += state.shape.len();
        }
    }
    assert_eq!(header.graphs(), 2);
    assert_eq!(header.methods(), plan.transformations.len());
    assert_eq!(header.operations(), operations);
    assert_eq!(header.states(), states);
    assert_eq!(header.input_refs(), inputs);
    assert_eq!(header.dimensions(), dimensions);
    assert_eq!(header.containers(), containers + json[0]);
    assert_eq!(header.entries(), json[1]);
    assert_eq!(header.text_bytes(), text + json[2]);
    assert_eq!(header.key_bytes(), json[3]);
    assert_eq!(header.json_values(), json[4]);
}

#[test]
fn constructor_headers_retain_original_root_and_one_cumulative_ledger() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let first = prepared(&budget, RAW, 4, limits()).construct().unwrap();
    assert_actual_header(&first);
    let used = first.resources().prepaid_work;
    let second = first.lower(
        budget.charge_external(first.lower_bootstrap_bytes(limits()).unwrap()).unwrap(),
        workload(5), limits()).unwrap();
    assert_actual_header(&second);
    assert!(Arc::ptr_eq(first.root(), second.root()));
    assert!(!Arc::ptr_eq(&first.inner, &second.inner));
    assert!(second.inner.payload().ledger.is_none());
    assert_eq!(first.root().payload().ledger.as_ref().unwrap().lock().unwrap().used,
        second.resources().prepaid_work);
    assert!(second.resources().prepaid_work > used);
    assert!(std::ptr::eq(first.header(), &second.root().payload().header));
    assert!(!std::ptr::eq(first.header(), second.header()));
    let transition = OriginalSourcePlan::compiler_metadata_control_bytes().unwrap();
    let control = budget.charge_external(transition).unwrap();
    second.prepay_compiler_metadata(&control, 7).unwrap();
    first.prepay_compiler_metadata(&control, 5).unwrap();
    assert_eq!(second.root().payload().ledger.as_ref().unwrap().lock().unwrap().used,
        second.resources().prepaid_work + 12);
    // Retiring the public root handle cannot retire its original plan/header,
    // imported configuration, or the root's single genuine WorkLedger.
    drop(first);
    assert_eq!(second.raw_configuration(), RAW);
    let third = second.lower(
        budget.charge_external(second.lower_bootstrap_bytes(limits()).unwrap()).unwrap(),
        workload(6), limits()).unwrap();
    assert!(second.same_origin(&third));
    assert_actual_header(&third);
    assert!(third.resources().prepaid_work > second.resources().prepaid_work + 12);
    assert_eq!(third.root().payload().ledger.as_ref().unwrap().lock().unwrap().used,
        third.resources().prepaid_work);
}

#[test]
fn enlarged_context_wrapper_and_arc_layout_are_admitted_before_factory() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let quoted = prepared(&budget, RAW, 4, limits());
    let attempts = attempts();
    let required = working_payload(quoted.prepared).unwrap();
    assert_eq!(quoted.resources().required_working_payload, required);
    let (layout, offset) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<OwnedCharged<Context>>()).unwrap();
    let arc = layout.pad_to_align().size() - size_of::<OwnedCharged<Context>>();
    assert!(offset >= size_of::<[AtomicUsize; 2]>());
    assert_eq!(arc_header_bytes::<OwnedCharged<Context>>().unwrap(), arc);
    assert_eq!(working_charge_bytes(required).unwrap(),
        required + size_of::<OwnedCharged<Configured>>());
    assert!(control_bytes().unwrap() >= size_of::<Context>()
        + size_of::<Result<Context, Error>>()
        + size_of::<Result<OwnedCharged<Context>, NativeError>>()
        + 3 * size_of::<(DecoderPlan, census::Header)>()
        + 6 * size_of::<Result<(DecoderPlan, census::Header), Error>>());
    assert_eq!(self::attempts(), attempts);
    let source = quoted.construct().unwrap();
    assert_eq!(source.inner.payload().configured.get().unwrap().charged_bytes(), source.resources().working_charge_bytes);
    assert_actual_header(&source);
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

#[test]
fn primitive_attempt_control_quote_covers_owned_guard_arc_and_transition_moves() {
    type G<'a> = MutexGuard<'a, Ledger>;
    let slots = [
        4 * size_of::<PrimitiveAttempt>(),
        3 * size_of::<Result<PrimitiveAttempt, Error>>(),
        4 * size_of::<OwnedExternalCharge>(),
        2 * size_of::<Option<OwnedExternalCharge>>(),
        2 * size_of::<Result<OwnedExternalCharge, Error>>(),
        3 * size_of::<Arc<OwnedCharged<Context>>>(),
        3 * size_of::<&Arc<OwnedCharged<Context>>>(),
        2 * size_of::<&OriginalSourcePlan>(),
        2 * size_of::<&Mutex<Ledger>>(),
        size_of::<Option<&Mutex<Ledger>>>(),
        2 * size_of::<G<'_>>(),
        size_of::<LockResult<G<'_>>>(),
        size_of::<std::sync::PoisonError<G<'_>>>(),
        size_of::<Result<(), Error>>(),
        3 * size_of::<Result<u64, Error>>(),
        2 * size_of::<Option<u64>>(),
        4 * size_of::<u64>(),
        4 * size_of::<usize>(),
        2 * size_of::<Result<usize, Error>>(),
        2 * size_of::<Option<usize>>(),
        2 * size_of::<bool>(),
    ];
    let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<OwnedCharged<Context>>()).unwrap();
    let header = layout.pad_to_align().size() - size_of::<OwnedCharged<Context>>();
    let expected = slots.iter().sum::<usize>() + std::mem::size_of_val(&slots)
        + std::mem::size_of_val(&slots.into_iter()) + header + 3 * size_of::<AttemptDebit>();
    let before = attempts();
    assert_eq!(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap(), expected);
    assert_eq!(attempts(), before);
}

#[test]
fn primitive_attempt_foreign_or_short_controller_never_enters_or_burns_siblings() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let first = prepared(&budget, RAW, 4, limits()).construct().unwrap();
    let sibling = first.lower(
        budget.charge_external(first.lower_bootstrap_bytes(limits()).unwrap()).unwrap(),
        workload(5), limits()).unwrap();
    let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
    let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let before = attempts();
    let used = first.root().payload().ledger.as_ref().unwrap().lock().unwrap().used;
    let strong = Arc::strong_count(first.root());
    assert!(matches!(first.begin_primitive_attempt(
        foreign.charge_external(bytes).unwrap(), 0), Err(Error::Counter)));
    assert!(matches!(sibling.begin_primitive_attempt(
        budget.charge_external(bytes - 1).unwrap(), 0), Err(Error::Capacity)));
    assert_eq!(Arc::strong_count(first.root()), strong);
    assert_eq!(first.root().payload().ledger.as_ref().unwrap().lock().unwrap().used, used);
    assert_eq!(attempts(), before);
    assert!(first.lower_bootstrap_bytes(limits()).is_ok());
    assert!(sibling.lower_bootstrap_bytes(limits()).is_ok());
    let attempt = sibling.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0).unwrap();
    assert!(Arc::ptr_eq(&attempt.root, first.root()));
    assert!(!Arc::ptr_eq(&attempt.root, &sibling.inner));
    assert!(std::ptr::eq(attempt.controller.as_ref().unwrap().budget(), first.budget()));
    drop(attempt.commit().unwrap());
    let next = first.lower(
        budget.charge_external(first.lower_bootstrap_bytes(limits()).unwrap()).unwrap(),
        workload(6), limits()).unwrap();
    assert!(next.same_origin(&sibling));
    assert_eq!(attempts(), before + 1);
}

#[test]
fn primitive_attempt_debits_exact_visits_plus_one_and_keeps_original_controller() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let source = prepared(&budget, RAW, 4, limits()).construct().unwrap();
    let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap() + 37;
    let controller = budget.charge_external(bytes).unwrap();
    let charged = controller.charged_bytes();
    let used_bytes = budget.budget().used_bytes();
    let before = attempts();
    let used = source.root().payload().ledger.as_ref().unwrap().lock().unwrap().used;
    let attempt = source.begin_primitive_attempt(controller, 7).unwrap();
    assert!(Arc::ptr_eq(&attempt.root, source.root()));
    // Caller work can acquire the mutex: begin does not retain a lock.
    let ledger = source.root().payload().ledger.as_ref().unwrap().try_lock().unwrap();
    assert_eq!(ledger.used, used + 8);
    assert!(!ledger.burned);
    drop(ledger);
    let controller = attempt.commit().unwrap();
    assert_eq!(controller.payload_bytes(), bytes);
    assert_eq!(controller.charged_bytes(), charged);
    assert!(std::ptr::eq(controller.budget(), source.budget()));
    assert_eq!(budget.budget().used_bytes(), used_bytes);
    drop(controller);
    // No fresh ledger across the publication/control gap; zero work costs one.
    let control = source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0)
        .unwrap().commit().unwrap();
    assert_eq!(source.root().payload().ledger.as_ref().unwrap().lock().unwrap().used, used + 9);
    assert_eq!(attempts(), before);
    drop(control);
    drop(source);
    // Extra source/attempt/control charges retire; live OwnedBudget control stays.
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn primitive_attempt_zero_work_cannot_renew_the_original_work_ceiling() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = prepared(&budget, RAW, 4, limits()).construct().unwrap();
    let used = source.root().payload().ledger.as_ref().unwrap().lock().unwrap().used;
    let metadata = budget.charge_external(OriginalSourcePlan::compiler_metadata_control_bytes().unwrap()).unwrap();
    source.prepay_compiler_metadata(&metadata, limits().max_work - used - 2).unwrap();
    let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
    for _ in 0..2 {
        drop(source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0)
            .unwrap().commit().unwrap());
    }
    assert_eq!(source.root().payload().ledger.as_ref().unwrap().lock().unwrap().used, limits().max_work);
    let before = attempts();
    assert!(matches!(source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0), Err(Error::Work)));
    assert_eq!(source.lower_bootstrap_bytes(limits()), Err(Error::Burned));
    assert_eq!(attempts(), before);
}

#[test]
fn primitive_attempt_fee_overflow_cumulative_overflow_and_exhaustion_burn_before_factory() {
    for case in 0..3 {
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let source = prepared(&budget, RAW, 4, limits()).construct().unwrap();
        let used = source.root().payload().ledger.as_ref().unwrap().lock().unwrap().used;
        let visits = match case {
            0 => u64::MAX,
            1 => u64::MAX - 1,
            _ => limits().max_work - used,
        };
        let before = attempts();
        let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
        assert!(matches!(source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), visits), Err(Error::Work)));
        assert_eq!(source.root().payload().ledger.as_ref().unwrap().lock().unwrap().used, used);
        assert_eq!(source.lower_bootstrap_bytes(limits()), Err(Error::Burned));
        assert!(matches!(source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0), Err(Error::Burned)));
        assert_eq!(attempts(), before);
    }
}

#[test]
fn primitive_attempt_abandonment_error_and_unwind_burn_original_root_through_sibling() {
    for case in 0..3 {
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let source = prepared(&budget, RAW, 4, limits()).construct().unwrap();
        let sibling = source.lower(
            budget.charge_external(source.lower_bootstrap_bytes(limits()).unwrap()).unwrap(),
            workload(5), limits()).unwrap();
        let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
        let before = attempts();
        let used = source.root().payload().ledger.as_ref().unwrap().lock().unwrap().used;
        let controller = budget.charge_external(bytes).unwrap();
        match case {
            0 => drop(sibling.begin_primitive_attempt(controller, 3).unwrap()),
            1 => {
                let result: Result<(), Error> = (|| {
                    let _attempt = sibling.begin_primitive_attempt(controller, 3)?;
                    Err(Error::Factory) // Caller failure, not a numeric factory.
                })();
                assert_eq!(result, Err(Error::Factory));
            }
            _ => {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _attempt = sibling.begin_primitive_attempt(controller, 3).unwrap();
                    panic!("primitive caller unwind");
                }));
                assert!(result.is_err());
            }
        }
        assert_eq!(source.root().payload().ledger.as_ref().unwrap().lock().unwrap().used, used + 4);
        assert_eq!(source.lower_bootstrap_bytes(limits()), Err(Error::Burned));
        assert_eq!(sibling.lower_bootstrap_bytes(limits()), Err(Error::Burned));
        assert!(matches!(source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0), Err(Error::Burned)));
        assert_eq!(sibling.plan().prefill.query_sequence, 5);
        assert_eq!(attempts(), before);
    }
}

#[test]
fn primitive_attempt_commit_cannot_clear_independent_sibling_lowering_failure() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = prepared(&budget, RAW, 4, limits()).construct().unwrap();
    let sibling = source.lower(
        budget.charge_external(source.lower_bootstrap_bytes(limits()).unwrap()).unwrap(),
        workload(5), limits()).unwrap();
    let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
    let pending = source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 2).unwrap();
    let short = sibling.lower_bootstrap_bytes(limits()).unwrap() - 1;
    let before = attempts();
    assert!(matches!(sibling.lower(budget.charge_external(short).unwrap(), workload(6), limits()), Err(Error::Capacity)));
    assert!(matches!(pending.commit(), Err(Error::Burned)));
    assert_eq!(source.lower_bootstrap_bytes(limits()), Err(Error::Burned));
    assert_eq!(sibling.lower_bootstrap_bytes(limits()), Err(Error::Burned));
    assert_eq!(attempts(), before);
}

#[test]
fn primitive_attempt_pending_sibling_failure_cannot_be_cleared_by_commit() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = prepared(&budget, RAW, 4, limits()).construct().unwrap();
    let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
    let first = source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0).unwrap();
    let second = source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0).unwrap();
    drop(second);
    assert!(matches!(first.commit(), Err(Error::Burned)));
    assert_eq!(source.lower_bootstrap_bytes(limits()), Err(Error::Burned));
}

#[test]
fn primitive_attempt_retains_actual_original_backing_without_public_source_handles() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let source = prepared(&budget, RAW, 4, limits()).construct().unwrap();
    let sibling = source.lower(
        budget.charge_external(source.lower_bootstrap_bytes(limits()).unwrap()).unwrap(),
        workload(5), limits()).unwrap();
    let weak = Arc::downgrade(source.root());
    let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
    let attempt = sibling.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0).unwrap();
    assert!(Arc::ptr_eq(&attempt.root, source.root()));
    drop(sibling);
    drop(source);
    assert_eq!(attempt.root.payload().plan.prefill.query_sequence, 4);
    match &attempt.root.payload().configuration {
        Configuration::Original(value) => {
            assert_eq!(value.payload().raw(), RAW);
            assert!(std::ptr::eq(value.budget(), attempt.controller.as_ref().unwrap().budget()));
        }
        Configuration::OriginalContext(_) => panic!("attempt retained derived context"),
    }
    assert_eq!(weak.strong_count(), 1);
    let controller = attempt.commit().unwrap();
    assert_eq!(weak.strong_count(), 0);
    assert_eq!(budget.budget().used_bytes(), baseline + controller.charged_bytes());
    drop(controller);
    // Extra source/attempt/control charges retire; live OwnedBudget control stays.
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn primitive_attempt_committed_control_and_immutable_plan_survive_later_failure() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let source = prepared(&budget, RAW, 4, limits()).construct().unwrap();
    let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
    let attempt = source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0).unwrap();
    // Ordinary already-prepared immutable test output; NOT the future bridge or
    // an escaped output owner retaining source backing. No fallible prep follows.
    let output = [b'o', b'k'];
    let committed = attempt.commit().unwrap();
    drop(source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0).unwrap());
    assert_eq!(source.lower_bootstrap_bytes(limits()), Err(Error::Burned));
    assert_eq!(source.raw_configuration(), RAW);
    assert_eq!(source.plan().prefill.query_sequence, 4);
    assert_eq!(output, *b"ok");
    assert!(std::ptr::eq(committed.budget(), source.budget()));
    assert_eq!(committed.payload_bytes(), bytes);
    drop(source);
    assert_eq!(budget.budget().used_bytes(), baseline + committed.charged_bytes());
    assert_eq!(output, *b"ok");
    drop(committed);
    // Extra source/attempt/control charges retire; live OwnedBudget control stays.
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn primitive_attempt_poisoned_ledger_drop_never_panics_or_clears_burn() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = prepared(&budget, RAW, 4, limits()).construct().unwrap();
    let bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
    let attempt = source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0).unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ledger = source.root().payload().ledger.as_ref().unwrap().lock().unwrap();
        panic!("poison actual original ledger");
    })).is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(attempt))).is_ok());
    let poisoned = source.root().payload().ledger.as_ref().unwrap().lock().err().unwrap().into_inner();
    assert!(poisoned.burned);
    drop(poisoned);
    assert_eq!(source.lower_bootstrap_bytes(limits()), Err(Error::Burned));
    assert!(matches!(source.begin_primitive_attempt(budget.charge_external(bytes).unwrap(), 0), Err(Error::Burned)));
}

fn native_admission(budget: &OwnedBudget) -> SourceAdmission {
    SourceAdmission::new(budget.charge_external(SourceAdmission::root_control_bytes().unwrap()).unwrap(), workload(4)).unwrap()
}

fn native_prepared(
    budget: &OwnedBudget, admission: &mut SourceAdmission, raw: &[u8], visits: u64,
) -> PrivatePreparedConfiguration {
    let phase = admission.reserve_bootstrap_phase(
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), visits,
    ).unwrap();
    let raw = budget.charge_external(raw.len()).unwrap().try_build(|_| Ok(raw.to_vec())).unwrap();
    phase.prepare_configuration(raw, budget.charge_external(json::bootstrap_bytes().unwrap()).unwrap()).unwrap()
}

#[test]
fn native_bootstrap_root_and_phase_exact_short_foreign_preserve_unentered_ledger() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let before = attempts();
    let bytes = SourceAdmission::root_control_bytes().unwrap();
    assert!(matches!(SourceAdmission::new(budget.charge_external(bytes - 1).unwrap(), workload(4)), Err(Error::Capacity)));
    let mut admission = native_admission(&budget);
    assert!(admission.root.payload().configured.get().is_none());
    let control = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();
    assert!(matches!(admission.reserve_bootstrap_phase(foreign.charge_external(control).unwrap(), 3), Err(Error::Counter)));
    assert!(matches!(admission.reserve_bootstrap_phase(budget.charge_external(control - 1).unwrap(), 3), Err(Error::Capacity)));
    let ledger = admission.root.payload().ledger.as_ref().unwrap().lock().unwrap();
    assert_eq!(ledger.used, 0);
    assert_eq!(ledger.limit, BOOTSTRAP_SOURCE_WORK);
    assert!(!ledger.burned);
    drop(ledger);
    let phase = admission.reserve_bootstrap_phase(budget.charge_external(control).unwrap(), 3).unwrap();
    assert_eq!(admission.root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 4);
    assert!(matches!(admission.reserve_bootstrap_phase(budget.charge_external(control).unwrap(), 0), Err(Error::Burned)));
    assert!(!admission.root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    assert_eq!(attempts(), before);
    drop(phase);
    assert!(admission.root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
}

#[test]
fn native_bootstrap_carries_exact_preconfig_debits_into_configured_and_lowered_siblings() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let probe = prepared(&budget, RAW, 4, limits());
    let model_work = probe.prepared.work.total_work();
    drop(probe);
    let probe = prepared(&budget, RAW, 5, limits());
    let lower_work = probe.prepared.work.total_work();
    drop(probe);
    let mut admission = native_admission(&budget);
    let actual_root = Arc::clone(&admission.root);
    let prepared = native_prepared(&budget, &mut admission, RAW, 37);
    let raw_backing = prepared.source.payload().raw().as_ptr();
    prepared.prepay_policy(13).unwrap();
    let preconfig = 38 + u64::try_from(bootstrap_json_limits().max_work).unwrap() + 13;
    assert_eq!(actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().used, preconfig);
    let bytes = prepared.bootstrap_bytes(limits()).unwrap();
    let before = attempts();
    let source = admission.configure(prepared, budget.charge_external(bytes).unwrap(), limits()).unwrap();
    assert_eq!(attempts(), before + 1);
    assert!(Arc::ptr_eq(source.root(), &actual_root));
    // Observation of the retained allocation, NOT a source-authority predicate.
    assert_eq!(source.raw_configuration().as_ptr(), raw_backing);
    assert_eq!(source.resources().prepaid_work, preconfig + model_work);
    assert_eq!(actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().used, preconfig + model_work);
    let sibling = source.lower(budget.charge_external(source.lower_bootstrap_bytes(limits()).unwrap()).unwrap(), workload(5), limits()).unwrap();
    assert!(source.same_origin(&sibling));
    assert_eq!(actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().used, sibling.resources().prepaid_work);
    assert_eq!(sibling.resources().prepaid_work, preconfig + model_work + lower_work);
    assert_eq!(attempts(), before + 2);
    assert!(std::ptr::eq(sibling.budget(), budget.budget()));
    drop(source);
    drop(actual_root);
    assert_eq!(sibling.raw_configuration(), RAW);
    assert_eq!(sibling.raw_configuration().as_ptr(), raw_backing);
    drop(sibling);
    // Extra attempt/config/root owners retire; live OwnedBudget control remains.
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn native_bootstrap_same_budget_foreign_admission_cannot_adopt_prepared_configuration() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut first = native_admission(&budget);
    let second = native_admission(&budget);
    let actual_first = Arc::clone(&first.root);
    let actual_second = Arc::clone(&second.root);
    let prepared = native_prepared(&budget, &mut first, RAW, 0);
    assert!(std::ptr::eq(actual_first.budget(), actual_second.budget()));
    assert!(!Arc::ptr_eq(&actual_first, &actual_second));
    let bytes = prepared.bootstrap_bytes(limits()).unwrap();
    let before = attempts();
    assert!(matches!(second.configure(prepared, budget.charge_external(bytes).unwrap(), limits()), Err(Error::Counter)));
    assert_eq!(attempts(), before);
    assert!(actual_first.payload().configured.get().is_none());
    assert!(actual_second.payload().configured.get().is_none());
    assert!(actual_first.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    assert!(actual_second.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
}

#[test]
fn native_bootstrap_work_exhaustion_and_overflow_burn_before_factory() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    for visits in [BOOTSTRAP_SOURCE_WORK, u64::MAX] {
        let mut admission = native_admission(&budget);
        let before = attempts();
        assert!(matches!(admission.reserve_bootstrap_phase(
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), visits), Err(Error::Work)));
        let ledger = admission.root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert_eq!(ledger.used, 0);
        assert!(ledger.burned);
        assert!(admission.root.payload().configured.get().is_none());
        assert_eq!(attempts(), before);
    }
}

#[test]
fn native_bootstrap_drop_error_and_unwind_never_install_configured_authority() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    for failure in 0..3 {
        let mut admission = native_admission(&budget);
        let actual_root = Arc::clone(&admission.root);
        let prepared = native_prepared(&budget, &mut admission, if failure == 1 { BAD_EPSILON } else { RAW }, 0);
        let before = attempts();
        if failure == 0 {
            drop(prepared);
            drop(admission);
            assert_eq!(attempts(), before);
        } else {
            let bytes = prepared.bootstrap_bytes(limits()).unwrap();
            if failure == 1 {
                assert!(matches!(admission.configure(prepared, budget.charge_external(bytes).unwrap(), limits()), Err(Error::Factory)));
            } else {
                PANIC_FACTORY.with(|flag| flag.set(true));
                let working = budget.charge_external(bytes).unwrap();
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| admission.configure(prepared, working, limits()))).is_err());
            }
            assert_eq!(attempts(), before + 1);
        }
        assert!(actual_root.payload().configured.get().is_none());
        let ledger = actual_root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert!(ledger.burned);
        assert!(ledger.used >= 1 + u64::try_from(bootstrap_json_limits().max_work).unwrap());
        drop(ledger);
        drop(actual_root);
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn native_bootstrap_short_config_control_refuses_before_factory_without_install() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut admission = native_admission(&budget);
    let prepared = native_prepared(&budget, &mut admission, RAW, 0);
    let actual_root = Arc::clone(&admission.root);
    let used = actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
    let bytes = prepared.bootstrap_bytes(limits()).unwrap();
    let before = attempts();
    assert!(matches!(admission.configure(prepared, budget.charge_external(bytes - 1).unwrap(), limits()), Err(Error::Capacity)));
    assert_eq!(attempts(), before);
    let ledger = actual_root.payload().ledger.as_ref().unwrap().lock().unwrap();
    assert_eq!(ledger.used, used);
    assert!(ledger.burned);
    assert!(actual_root.payload().configured.get().is_none());
}

#[test]
fn native_bootstrap_prior_sibling_failure_cannot_be_cleared_by_configure() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut admission = native_admission(&budget);
    let prepared = native_prepared(&budget, &mut admission, RAW, 0);
    let actual_root = Arc::clone(&admission.root);
    let bytes = prepared.bootstrap_bytes(limits()).unwrap();
    let sibling = begin_attempt(&actual_root,
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
    drop(sibling);
    let before = attempts();
    assert!(matches!(admission.configure(prepared, budget.charge_external(bytes).unwrap(), limits()), Err(Error::Burned)));
    assert_eq!(attempts(), before);
    assert!(actual_root.payload().configured.get().is_none());
    assert!(actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
}

#[test]
fn native_bootstrap_quote_admitted_before_failure_rechecks_before_factory() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut admission = native_admission(&budget);
    let prepared = native_prepared(&budget, &mut admission, RAW, 0);
    let bytes = prepared.bootstrap_bytes(limits()).unwrap();
    let PrivatePreparedConfiguration { source, attempt, workload } = prepared;
    let quoted = quote_on_root(source, budget.charge_external(bytes).unwrap(), workload, limits(), attempt, JsonPayment::Prepaid).unwrap();
    let sibling = begin_attempt(&admission.root,
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
    drop(sibling);
    let before = attempts();
    assert!(matches!(quoted.construct(), Err(Error::Burned)));
    assert_eq!(attempts(), before);
    assert!(admission.root.payload().configured.get().is_none());
    assert!(admission.root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
}

#[test]
fn native_bootstrap_configured_payload_retires_before_root_lease() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    RETIREMENT_BUDGET.with(|slot| *slot.borrow_mut() = Some(budget.try_clone().unwrap()));
    let baseline = budget.budget().used_bytes();
    let mut admission = native_admission(&budget);
    let root_charge = admission.root.charged_bytes();
    let prepared = native_prepared(&budget, &mut admission, RAW, 0);
    let bytes = prepared.bootstrap_bytes(limits()).unwrap();
    let source = admission.configure(prepared, budget.charge_external(bytes).unwrap(), limits()).unwrap();
    RETIRED_BYTES.with(|value| value.set(None));
    drop(source);
    // Production Context::drop retired Configured, imported Value, and raw owner
    // before this observation; enclosing root lease is still charged here.
    assert_eq!(RETIRED_BYTES.with(|value| value.take()), Some(baseline + root_charge));
    assert_eq!(budget.budget().used_bytes(), baseline);
    RETIREMENT_BUDGET.with(|slot| drop(slot.borrow_mut().take()));
    assert_eq!(budget.budget().used_bytes(), baseline - size_of::<OwnedBudget>());
}

#[test]
fn native_bootstrap_c9_raw_cap_is_private_and_legacy_native_domain_unchanged() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut admission = native_admission(&budget);
    let actual_root = Arc::clone(&admission.root);
    let phase = admission.reserve_bootstrap_phase(
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0,
    ).unwrap();
    let bytes = BOOTSTRAP_CONFIG_BYTES + 1;
    let raw = budget.charge_external(bytes).unwrap().try_build(|_| Ok(vec![b' '; bytes])).unwrap();
    let before = attempts();
    assert!(matches!(phase.prepare_configuration(raw, budget.charge_external(json::bootstrap_bytes().unwrap()).unwrap()), Err(Error::Capacity)));
    assert_eq!(attempts(), before);
    assert_eq!(actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 1);
    assert!(actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    assert!(actual_root.payload().configured.get().is_none());
    drop(admission);
    drop(actual_root);
    // Existing native C9 quote route keeps its caller's original raw-byte cap.
    // This is NOT an oracle for ordinary-serde/Python numeric or duplicate keys.
    let bytes = BOOTSTRAP_CONFIG_BYTES + 1;
    let raw = budget.charge_external(bytes).unwrap().try_build(|_| {
        let mut raw = Vec::with_capacity(bytes);
        raw.extend_from_slice(RAW);
        raw.resize(bytes, b' ');
        Ok(raw)
    }).unwrap();
    let imported = imported_owned(&budget, raw);
    let bytes = bootstrap_bytes(&imported, limits()).unwrap();
    let source = quote(imported, budget.charge_external(bytes).unwrap(), workload(4), limits()).unwrap().construct().unwrap();
    assert_eq!(source.raw_configuration().len(), BOOTSTRAP_CONFIG_BYTES + 1);
    assert_eq!(attempts(), before + 1);
}
