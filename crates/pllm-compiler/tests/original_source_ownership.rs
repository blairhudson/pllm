//! External consumers of the real original-config factory and bounded JSON input.
//!
//! Local configuration bytes only: no private test helpers, factory counters,
//! parsed-plan promotion, serialization roundtrip or executable/source credential.
//! `same_origin` compares custody; no public expected-origin authorization API is
//! invented here. Budget assertions cover named logical owners, not stack/RSS,
//! opaque dependency allocations or a complete request/CPU-work bound.
//! Factory-unwind injection and temporary frame/drop counters are private;
//! these tests assert public error cleanup/residency, not those observations.

use pllm_compiler::{original_source as source_owner, plan_input::json};
use pllm_garble::shared_allocation::{OwnedBudget, OwnedCharged, MAX_ALLOCATION_BYTES};
use pllm_models::{ConfigLowererLimits, DecoderExpansionBudget, DecoderWorkload};
use std::{mem::size_of, ptr};

// Same tiny config data as the internal source-owner fixtures; no internal code.
const CONFIG: &[u8] = br#"{"model_type":"qwen2","hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":7,"max_position_embeddings":8,"hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true,"use_sliding_window":false}"#;
const BAD_EPSILON: &[u8] = br#"{"model_type":"qwen2","hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":7,"max_position_embeddings":8,"hidden_act":"silu","rms_norm_eps":-1,"rope_theta":10000,"tie_word_embeddings":true,"use_sliding_window":false}"#;
const SPARE: usize = 73;

fn limits() -> source_owner::Limits {
    source_owner::Limits {
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

fn json_limits() -> json::Limits {
    json::Limits {
        max_source_bytes: 32768,
        max_nodes: 512,
        max_depth: 16,
        max_string_bytes: 16384,
        max_number_bytes: 64,
        max_number_work: 1_000_000,
        max_work: 100_000_000,
        max_payload_bytes: 4 << 20,
    }
}

fn workload(tokens: u64) -> DecoderWorkload {
    DecoderWorkload {
        batch: 1,
        max_input_tokens: tokens,
        max_new_tokens: 2,
    }
}

fn raw(root: &OwnedBudget, bytes: &[u8]) -> OwnedCharged<Vec<u8>> {
    let capacity = bytes.len() + SPARE;
    let before = root.budget().used_bytes();
    root.charge_external(capacity)
        .unwrap()
        .try_build(|counter| {
            assert!(ptr::eq(counter, root.budget()));
            // Both original backing and final inline wrapper are admitted before
            // the actual allocating byte factory, not blessed after allocation.
            assert_eq!(
                counter.used_bytes(),
                before + capacity + size_of::<OwnedCharged<Vec<u8>>>()
            );
            let mut bytes_owner = Vec::with_capacity(capacity);
            bytes_owner.extend_from_slice(bytes);
            assert_eq!(bytes_owner.capacity(), capacity);
            Ok(bytes_owner)
        })
        .unwrap()
}

fn import(
    root: &OwnedBudget,
    original: OwnedCharged<Vec<u8>>,
) -> OwnedCharged<json::ImportedValue> {
    json::quote(
        original,
        root.charge_external(json::bootstrap_bytes().unwrap())
            .unwrap(),
        json_limits(),
    )
    .unwrap()
    .construct()
    .unwrap()
}

fn prepared(root: &OwnedBudget, config: &[u8]) -> source_owner::SourceQuote {
    let source = import(root, raw(root, config));
    let bootstrap = source_owner::bootstrap_bytes(&source, limits()).unwrap();
    source_owner::quote(
        source,
        root.charge_external(bootstrap).unwrap(),
        workload(4),
        limits(),
    )
    .unwrap()
}

#[test]
fn equal_config_roots_are_distinct_and_foreign_counter_errors_do_not_burn_them() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    let foreign_baseline = foreign.budget().used_bytes();
    let first = prepared(&root, CONFIG).construct().unwrap();
    let independent = prepared(&root, CONFIG).construct().unwrap();
    assert_eq!(first.plan(), independent.plan());
    assert_eq!(first.raw_configuration(), independent.raw_configuration());
    assert!(ptr::eq(first.budget(), independent.budget()));
    assert!(!first.same_origin(&independent));
    assert!(!first.same_plan(&independent));
    let resident = root.budget().used_bytes();
    let source = import(&root, raw(&root, CONFIG));
    let control = source_owner::bootstrap_bytes(&source, limits()).unwrap();
    assert!(matches!(
        source_owner::quote(
            source,
            foreign.charge_external(control).unwrap(),
            workload(4),
            limits()
        ),
        Err(source_owner::Error::Counter)
    ));
    assert_eq!(root.budget().used_bytes(), resident);
    assert_eq!(foreign.budget().used_bytes(), foreign_baseline);
    let control = first.lower_bootstrap_bytes(limits()).unwrap();
    assert!(matches!(
        first.lower(
            foreign.charge_external(control).unwrap(),
            workload(5),
            limits()
        ),
        Err(source_owner::Error::Counter)
    ));
    assert_eq!(root.budget().used_bytes(), resident);
    assert_eq!(foreign.budget().used_bytes(), foreign_baseline);
    let second = first
        .lower(
            root.charge_external(control).unwrap(),
            workload(5),
            limits(),
        )
        .unwrap();
    assert!(first.same_origin(&second));
    assert!(!independent.same_origin(&second));
    drop((first, independent, second));
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn spare_capacity_bootstrap_cancellation_and_real_factory_errors_release_owners() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    // Deliberately dishonest public declaration: len paid, spare capacity not.
    // The bounded Value entry must refuse this before issuing an ImportedValue.
    let undercharged = root
        .charge_external(CONFIG.len())
        .unwrap()
        .try_build(|_| {
            let mut bytes = Vec::with_capacity(CONFIG.len() + SPARE);
            bytes.extend_from_slice(CONFIG);
            assert_eq!(bytes.capacity(), CONFIG.len() + SPARE);
            Ok(bytes)
        })
        .unwrap();
    assert!(matches!(
        json::quote(
            undercharged,
            root.charge_external(json::bootstrap_bytes().unwrap())
                .unwrap(),
            json_limits()
        ),
        Err(json::Error::OriginalOwner)
    ));
    assert_eq!(root.budget().used_bytes(), baseline);
    assert!(matches!(
        json::quote(
            raw(&root, CONFIG),
            root.charge_external(json::bootstrap_bytes().unwrap() - 1)
                .unwrap(),
            json_limits()
        ),
        Err(json::Error::Bootstrap)
    ));
    assert_eq!(root.budget().used_bytes(), baseline);
    let source = import(&root, raw(&root, CONFIG));
    let control = source_owner::bootstrap_bytes(&source, limits()).unwrap();
    assert!(matches!(
        source_owner::quote(
            source,
            root.charge_external(control - 1).unwrap(),
            workload(4),
            limits()
        ),
        Err(source_owner::Error::Capacity)
    ));
    assert_eq!(root.budget().used_bytes(), baseline);
    let pending = prepared(&root, CONFIG);
    assert!(root.budget().used_bytes() > baseline);
    drop(pending); // cancellation before the original model factory
    assert_eq!(root.budget().used_bytes(), baseline);
    assert!(matches!(
        prepared(&root, BAD_EPSILON).construct(),
        Err(source_owner::Error::Factory)
    ));
    assert_eq!(root.budget().used_bytes(), baseline);
}

#[test]
fn dropped_siblings_cannot_refund_shared_work_and_admitted_failure_burns_all() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = root.budget().used_bytes();
    let first = prepared(&root, CONFIG).construct().unwrap();
    let initial_work = first.resources().prepaid_work;
    let first_resident = root.budget().used_bytes();
    let control = first.lower_bootstrap_bytes(limits()).unwrap();
    let second = first
        .lower(
            root.charge_external(control).unwrap(),
            workload(5),
            limits(),
        )
        .unwrap();
    let spent = second.resources().prepaid_work;
    assert!(spent > initial_work);
    drop(second);
    assert_eq!(root.budget().used_bytes(), first_resident);
    assert_eq!(
        first.resources().prepaid_work,
        initial_work,
        "old handle exposes a historical summary, not refunded work"
    );
    let mut exhausted = limits();
    exhausted.max_work = spent;
    let resident = root.budget().used_bytes();
    assert_eq!(
        first.lower_bootstrap_bytes(exhausted),
        Err(source_owner::Error::Work)
    );
    assert_eq!(root.budget().used_bytes(), resident);
    // A failed pure query neither resets/refunds work nor burns the source.
    let control = first.lower_bootstrap_bytes(limits()).unwrap();
    let third = first
        .lower(
            root.charge_external(control).unwrap(),
            workload(6),
            limits(),
        )
        .unwrap();
    assert!(third.resources().prepaid_work > spent);
    exhausted.max_work = third.resources().prepaid_work;
    let resident = root.budget().used_bytes();
    assert!(first
        .lower(
            root.charge_external(control).unwrap(),
            workload(5),
            exhausted
        )
        .is_err());
    assert_eq!(root.budget().used_bytes(), resident);
    for owner in [&first, &third] {
        assert_eq!(
            owner.lower_bootstrap_bytes(limits()),
            Err(source_owner::Error::Burned)
        );
        assert!(matches!(
            owner.lower(
                root.charge_external(control).unwrap(),
                workload(5),
                limits()
            ),
            Err(source_owner::Error::Burned)
        ));
        assert_eq!(root.budget().used_bytes(), resident);
        assert_eq!(owner.raw_configuration(), CONFIG);
    }
    drop((first, third));
    assert_eq!(root.budget().used_bytes(), baseline);
}
