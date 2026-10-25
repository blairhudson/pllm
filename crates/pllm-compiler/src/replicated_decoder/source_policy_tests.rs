use super::*;
use pllm_garble::shared_allocation::OwnedBudget;
use serde_json::json;

fn experiment() -> ExperimentDocument {
    serde_json::from_value(json!({"schema":"pllm.experiment.v1","name":"server-policy",
        "pipeline":{"model":{"source":"server-owned-config"},"components":{}},
        "deployment":{"kind":"local","root":"."},
        "budget":{"requests":1,"max_input_tokens":32,"max_new_tokens":8}}))
    .unwrap()
}
fn execution() -> ExecutionPolicy {
    ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        ..ExecutionPolicy::default()
    }
}
fn workload() -> DecoderWorkload {
    DecoderWorkload {
        batch: 1,
        max_input_tokens: 4,
        max_new_tokens: 2,
    }
}
fn raw() -> Vec<u8> {
    serde_json::to_vec(&json!({"model_type":"qwen2","hidden_size":4,
        "intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,
        "num_key_value_heads":1,"vocab_size":7,"max_position_embeddings":64,
        "hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000,
        "tie_word_embeddings":true,"use_sliding_window":false}))
    .unwrap()
}

fn server_raw() -> Vec<u8> {
    let mut server = serde_json::to_value(experiment()).unwrap();
    server["pipeline"]["components"]["linear"] = json!({
        "component":"pllm/puma-three-party-decoder/v1", "params":{
            "weights":"public", "range_policy":"checked", "max_abs_activation":1}});
    serde_json::to_vec(&server).unwrap()
}

fn client_raw(config: &[u8]) -> serde_json::Value {
    json!({"config":config,"rows":4,"outputs":2,"weights":"public",
        "execution":{"range_policy":"checked","max_abs_activation":1},"tensors":{}})
}
fn import(root: &OwnedBudget, raw: Vec<u8>) -> OwnedCharged<json::ImportedValue> {
    let source = root
        .charge_external(raw.capacity())
        .unwrap()
        .try_build(|_| Ok(raw))
        .unwrap();
    json::quote(
        source,
        root.charge_external(json::bootstrap_bytes().unwrap())
            .unwrap(),
        json::Limits {
            max_source_bytes: 1 << 20,
            max_nodes: 65536,
            max_depth: 16,
            max_string_bytes: 1 << 20,
            max_number_bytes: 64,
            max_number_work: 100_000_000,
            max_work: 1_000_000_000,
            max_payload_bytes: 8 << 20,
        },
    )
    .unwrap()
    .construct()
    .unwrap()
}

#[test]
fn checked_public_permissions_come_from_exact_server_owner_not_client_caps() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let config = import(&root, raw());
    let server = experiment();
    let policy = ServerSourcePolicy::checked_public(
        &server,
        &config,
        workload(),
        WeightMode::Public,
        execution(),
    )
    .unwrap();
    let limits = policy.limits_for(&config, workload()).unwrap();
    assert_eq!(
        (
            limits.expansion.max_layers,
            limits.expansion.max_feature_width,
            limits.expansion.max_attention_heads,
            limits.expansion.max_vocabulary
        ),
        (1, 8, 2, 7)
    );
    assert_eq!(limits.max_work, SOURCE_WORK);
    assert_eq!(limits.factory.max_attribute_nodes, 32768);
    assert_eq!(limits.factory.max_operations_per_phase, 34);
    assert_eq!(limits.expansion.max_tensor_elements, 50);
    let mut short = limits.expansion;
    short.max_tensor_elements -= 1;
    assert!(pllm_models::ConfigLowererQuote::new(
        config.payload().value(),
        DecoderWorkload {
            max_input_tokens: 5,
            max_new_tokens: 1,
            ..workload()
        },
        short,
        limits.factory
    )
    .is_err());
    let mut short = limits.factory;
    short.max_operations_per_phase -= 1;
    assert!(pllm_models::ConfigLowererQuote::new(
        config.payload().value(),
        workload(),
        limits.expansion,
        short
    )
    .is_err());
    assert_eq!(limits.factory.max_factory_bytes, MAX_ALLOCATION_BYTES);
    let client = json!({"source_policy":{"max_work":u64::MAX,"max_layers":u64::MAX}});
    assert!(client["source_policy"]["max_work"].as_u64().unwrap() > limits.max_work);
    // There is deliberately no client-policy argument or serde implementation.
    assert_eq!(
        policy
            .limits_for(&config, workload())
            .unwrap()
            .expansion
            .max_layers,
        1
    );
    let foreign = import(&root, raw());
    assert!(policy.limits_for(&foreign, workload()).is_err());
    // Real entry rejects foreign SERVER configuration/client policy BEFORE the
    // actual config factory. No token/history/source permit is invented here.
    use super::super::program::source_entry::{PolicyFactoryTrap, SourceRequestImport};
    let actual = raw();
    let mut client = client_raw(&actual);
    let _trap = PolicyFactoryTrap::arm();
    client["source_policy"] = json!({"max_layers":u64::MAX});
    assert!(matches!(
        SourceRequestImport::new_checked_public(
            &serde_json::to_vec(&client).unwrap(),
            &server_raw(),
            &actual
        ),
        Err(Error::Binding)
    ));
    client.as_object_mut().unwrap().remove("source_policy");
    client["source_limits"] = json!({"max_attribute_nodes":65536});
    assert!(matches!(
        SourceRequestImport::new_checked_public(
            &serde_json::to_vec(&client).unwrap(),
            &server_raw(),
            &actual
        ),
        Err(Error::Binding)
    ));
    client.as_object_mut().unwrap().remove("source_limits");
    let mut foreign_config = actual.clone();
    foreign_config.push(b' ');
    assert!(matches!(
        SourceRequestImport::new_checked_public(
            &serde_json::to_vec(&client).unwrap(),
            &server_raw(),
            &foreign_config
        ),
        Err(Error::Binding)
    ));
    let foreign_root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let foreign = import(&foreign_root, raw());
    assert!(policy.limits_for(&foreign, workload()).is_err());
}

#[test]
fn workload_and_mode_gates_exact_and_short() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let config = import(&root, raw());
    let server = experiment();
    let maximum = DecoderWorkload {
        batch: 1,
        max_input_tokens: 32,
        max_new_tokens: 8,
    };
    let policy = ServerSourcePolicy::checked_public(
        &server,
        &config,
        maximum,
        WeightMode::Public,
        execution(),
    )
    .unwrap();
    assert!(policy.limits_for(&config, workload()).is_err());
    for bad in [
        DecoderWorkload {
            max_input_tokens: 33,
            ..maximum
        },
        DecoderWorkload {
            max_new_tokens: 9,
            ..maximum
        },
        DecoderWorkload {
            batch: 2,
            ..maximum
        },
    ] {
        assert!(ServerSourcePolicy::checked_public(
            &server,
            &config,
            bad,
            WeightMode::Public,
            execution()
        )
        .is_err());
    }
    assert!(ServerSourcePolicy::checked_public(
        &server,
        &config,
        maximum,
        WeightMode::Private,
        execution()
    )
    .is_err());
    assert!(ServerSourcePolicy::checked_public(
        &server,
        &config,
        maximum,
        WeightMode::Public,
        ExecutionPolicy::default()
    )
    .is_err());
}

#[test]
fn configuration_bytes_exact_and_one_over_no_model_factory() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let server = experiment();
    for (size, accept) in [(CONFIG_BYTES, true), (CONFIG_BYTES + 1, false)] {
        let mut bytes = raw();
        bytes.resize(size, b' ');
        let config = import(&root, bytes);
        assert_eq!(
            ServerSourcePolicy::checked_public(
                &server,
                &config,
                workload(),
                WeightMode::Public,
                execution()
            )
            .is_ok(),
            accept
        );
    }
}

#[test]
fn original_quote_exact_short_and_prior_json_work_no_construct() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let server = experiment();
    let config = import(&root, raw());
    let limits = ServerSourcePolicy::checked_public(
        &server,
        &config,
        workload(),
        WeightMode::Public,
        execution(),
    )
    .unwrap()
    .limits_for(&config, workload())
    .unwrap();
    let actual_server = import(&root, server_raw());
    let deducted = admitted_limits(&actual_server, &config, None, workload()).unwrap();
    assert!(deducted.max_work < SOURCE_WORK);
    assert_eq!(
        SOURCE_WORK - deducted.max_work,
        u64::try_from(actual_server.payload().resources().work.total * 2).unwrap()
            + config.payload().raw().len() as u64
            + workload().max_new_tokens * pllm_models::SOURCE_SCALAR_SUMMARY_WORK
    );
    let foreign_root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let foreign_server = import(&foreign_root, server_raw());
    assert!(admitted_limits(&foreign_server, &config, None, workload()).is_err());
    let boot = original_source::bootstrap_bytes(&config, limits).unwrap();
    let quote = original_source::quote(
        config,
        root.charge_external(boot).unwrap(),
        workload(),
        limits,
    )
    .unwrap();
    let required = quote.resources().prepaid_work;
    assert!(required > 0);
    drop(quote);
    for (ceiling, accept) in [(required, true), (required - 1, false)] {
        let config = import(&root, raw());
        let bounded = original_source::Limits {
            max_work: ceiling,
            ..limits
        };
        let quote = original_source::bootstrap_bytes(&config, bounded).and_then(|boot| {
            original_source::quote(
                config,
                root.charge_external(boot).unwrap(),
                workload(),
                bounded,
            )
        });
        assert_eq!(quote.is_ok(), accept);
        // NEVER construct, compile, read tensors or launch a child.
    }
}

#[test]
fn cached_qwen_config_scalar_policy_and_original_metadata_quote_only() {
    // Actual repository-pinned cached configuration, never checkpoint payloads.
    let raw = include_bytes!("../../../../tests/fixtures/qwen25_0_5b_config.json");
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let config = import(&root, raw.to_vec());
    let server = experiment();
    let policy = ServerSourcePolicy::checked_public(
        &server,
        &config,
        workload(),
        WeightMode::Public,
        execution(),
    )
    .unwrap();
    let limits = policy.limits_for(&config, workload()).unwrap();
    assert_eq!(
        (
            limits.expansion.max_layers,
            limits.expansion.max_feature_width,
            limits.expansion.max_attention_heads,
            limits.expansion.max_vocabulary
        ),
        (24, 4864, 14, 151936)
    );
    assert_eq!(limits.factory.max_attribute_nodes, 32768);
    assert_eq!(limits.factory.max_operations_per_phase, 678);
    assert_eq!(
        pllm_models::source_scalar_summary(config.payload().value(), workload())
            .unwrap()
            .semantic_elements(),
        607744
    );
    assert_eq!(limits.expansion.max_tensor_elements, 5 * 151936);
    assert_eq!(limits.expansion.max_tensor_elements, 759680);
    assert!(limits.expansion.max_tensor_elements < ORDINARY_LANES as u64);
    // The SAME registered, nonconstructing quote owns operation/shape formulas.
    // This does not invoke ConfigLowererQuote::construct, source file import,
    // Program/Manifest compile, tensor factories, Worker or any process launcher.
    let quote = pllm_models::ConfigLowererQuote::new(
        config.payload().value(),
        workload(),
        limits.expansion,
        limits.factory,
    )
    .unwrap();
    assert!(quote.metadata().operations_per_phase > 256);
    assert!(quote.metadata().operations_per_phase <= DESCRIPTORS);
    assert!(quote.factory_bytes() <= MAX_ALLOCATION_BYTES);
    use super::super::program::source_entry::{PolicyFactoryTrap, SourceRequestImport};
    let _trap = PolicyFactoryTrap::arm();
    match SourceRequestImport::quote_checked_public(&server_raw(), raw, 4, 2) {
        Ok(resources) => {
            eprintln!("pinned-config original source metadata quote ACCEPTED: {resources:?}");
            assert!(resources.model_factory_bytes <= MAX_ALLOCATION_BYTES);
            assert!(resources.prepaid_work <= SOURCE_WORK);
        }
        // A metadata quote is NOT permission to raise the physical/phase cap.
        Err(Error::Bounds) => {
            eprintln!("pinned-config original source metadata quote REFUSED: Bounds (unchanged finite phase/memory/work guards)");
        }
        other => panic!("unexpected metadata-only quotation: {other:?}"),
    }
    // Original conservative full-row logits, NEVER a last-token-only fit.
    assert!(matches!(
        SourceRequestImport::quote_checked_public(&server_raw(), raw, 8, 2),
        Err(Error::Bounds)
    ));
}

#[test]
fn cached_qwen_compact_metadata_quote_only() {
    use super::super::program::source_entry::{PolicyFactoryTrap, SourceRequestImport};
    use super::super::sdk::SourceStorage;
    use super::super::{LookupPolicy, PeerTransport, SelectionPolicy};
    use sha2::Digest as _;
    use std::io::Read as _;

    // Portable default is CONFIG metadata, not a claim of cached-byte identity.
    // An explicit override is mandatory for the actual-cache qualification run:
    // missing/oversize input fails, never silently falls back or downloads.
    let (raw, origin) = match std::env::var_os("PLLM_TEST_QWEN25_CONFIG_PATH") {
        Some(path) => {
            let mut raw = Vec::new();
            std::fs::File::open(&path)
                .expect("explicit config-only path must exist")
                .take((CONFIG_BYTES + 1) as u64)
                .read_to_end(&mut raw)
                .expect("bounded config-only read");
            (
                raw,
                format!("actual override {}", std::path::Path::new(&path).display()),
            )
        }
        None => (
            include_bytes!("../../../../tests/fixtures/qwen25_0_5b_config.json").to_vec(),
            "repository config fixture (not actual cached bytes)".to_owned(),
        ),
    };
    assert!(!raw.is_empty() && raw.len() <= CONFIG_BYTES);
    eprintln!(
        "compact metadata CONFIG {origin}; bytes={}; sha256={:x}",
        raw.len(),
        sha2::Sha256::digest(&raw)
    );

    let _trap = PolicyFactoryTrap::arm();
    let execution = execution();
    assert_eq!(execution.lookup, LookupPolicy::PerToken);
    assert_eq!(execution.selection, SelectionPolicy::Expanded);
    assert_eq!(execution.peer_transport, PeerTransport::Http);
    let mut params = execution.component_params(WeightMode::Public);
    params["storage"] = json!("compact");
    assert_eq!(
        SourceStorage::from_params(&params).unwrap(),
        SourceStorage::Compact
    );
    let mut server: serde_json::Value = serde_json::from_slice(&server_raw()).unwrap();
    server["pipeline"]["components"]["linear"]["params"] = params;
    server["budget"] = json!({"requests":1,"max_input_tokens":4,"max_new_tokens":2});
    let server: ExperimentDocument = serde_json::from_value(server).unwrap();
    let server = pllm_types::canonical_bytes(&server);

    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let config = import(&root, raw.clone());
    let admitted_server = import(&root, server.clone());
    let limits = admitted_limits(&admitted_server, &config, None, workload()).unwrap();
    // The unchanged original policy covers both request epochs, 4 -> 5 rows.
    let summary = pllm_models::source_scalar_summary(config.payload().value(), workload()).unwrap();
    assert_eq!(summary.semantic_elements(), 4 * 151936);
    assert_eq!(limits.expansion.max_tensor_elements, 5 * 151936);
    assert_eq!(limits.expansion.max_layers, 24);
    assert_eq!(limits.expansion.max_feature_width, 4864);
    assert_eq!(limits.expansion.max_attention_heads, 14);
    assert_eq!(limits.expansion.max_vocabulary, 151936);
    assert_eq!(limits.factory.max_operations_per_phase, 678);
    assert_eq!(DESCRIPTORS, 65536);
    assert_eq!(MAX_ALLOCATION_BYTES, 64 << 20);
    assert!(limits.max_work <= SOURCE_WORK);
    assert_eq!(limits.factory.max_factory_bytes, MAX_ALLOCATION_BYTES);

    // Never construct any model/source/payload factory, publisher, scanner,
    // seal, Program, request history or arithmetic. The armed trap panics if a
    // guarded factory is entered. Bounds refusal remains a legitimate outcome.
    match SourceRequestImport::quote_checked_public(&server, &raw, 4, 2) {
        Ok(resources) => {
            assert!(resources.model_factory_bytes <= MAX_ALLOCATION_BYTES);
            assert!(resources.prepaid_work <= SOURCE_WORK);
            eprintln!("compact original-source metadata quote ACCEPTED: {resources:?}; factory trap remained armed/unentered; NOT whole request/import/header/history/finalizer/runtime fit");
        }
        Err(Error::Bounds) => eprintln!("compact original-source metadata quote REFUSED: Bounds; unchanged guards, factory trap remained armed/unentered; NOT whole-request fit"),
        other => panic!("unexpected compact metadata-only quotation: {other:?}"),
    }
}

#[test]
fn registered_wide_heads_and_defaults_share_the_original_validator() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let server = experiment();
    let mut doc: serde_json::Value = serde_json::from_slice(&raw()).unwrap();
    doc["model_type"] = json!("qwen3");
    doc["head_dim"] = json!(8);
    doc["num_attention_heads"] = json!(4);
    doc["num_key_value_heads"] = json!(2);
    let config = import(&root, serde_json::to_vec(&doc).unwrap());
    let policy = ServerSourcePolicy::checked_public(
        &server,
        &config,
        workload(),
        WeightMode::Public,
        execution(),
    )
    .unwrap();
    let limits = policy.limits_for(&config, workload()).unwrap();
    assert_eq!(limits.expansion.max_feature_width, 32); // A*D, not H/I/D.
    assert_eq!(limits.factory.max_operations_per_phase, 36);
    assert!(pllm_models::ConfigLowererQuote::new(
        config.payload().value(),
        workload(),
        limits.expansion,
        limits.factory
    )
    .is_ok());
    doc["num_key_value_heads"] = json!(5);
    let summary = pllm_models::source_scalar_summary(&doc, workload()).unwrap();
    assert_eq!(summary.feature_width(), 40); // KV*D also retained, no semantic proof.
    doc["model_type"] = json!("qwen2");
    doc.as_object_mut().unwrap().remove("num_key_value_heads");
    doc["head_dim"] = json!(999); // Qwen2 EXISTING HiddenPerHead ignores this field.
    let summary = pllm_models::source_scalar_summary(&doc, workload()).unwrap();
    assert_eq!(summary.head_dim, 1);
    assert_eq!(summary.kv_heads, 4);
    assert_eq!(summary.feature_width(), 8);
}

#[test]
fn shared_summary_overflow_and_original_attribute_ceiling_refuse() {
    let mut doc: serde_json::Value = serde_json::from_slice(&raw()).unwrap();
    doc["model_type"] = json!("qwen3");
    doc["head_dim"] = json!(2);
    doc["num_attention_heads"] = json!(u64::MAX);
    assert!(pllm_models::source_scalar_summary(&doc, workload()).is_err());
    doc["num_attention_heads"] = json!(2);
    doc["num_hidden_layers"] = json!(u64::MAX);
    assert!(pllm_models::source_scalar_summary(&doc, workload()).is_err());
    doc["num_hidden_layers"] = json!(50); // 50*384*2 exceeds ORIGINAL 32768 attrs.
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let server = experiment();
    let config = import(&root, serde_json::to_vec(&doc).unwrap());
    let limits = ServerSourcePolicy::checked_public(
        &server,
        &config,
        workload(),
        WeightMode::Public,
        execution(),
    )
    .unwrap()
    .limits_for(&config, workload())
    .unwrap();
    assert_eq!(limits.factory.max_attribute_nodes, 32768);
    assert!(pllm_models::ConfigLowererQuote::new(
        config.payload().value(),
        workload(),
        limits.expansion,
        limits.factory
    )
    .is_err());
}

#[test]
fn prefix_summary_prepayment_exact_short_invalid_and_overflow() {
    let work = 2 * pllm_models::SOURCE_SCALAR_SUMMARY_WORK;
    let prior = u128::from(SOURCE_WORK - work);
    assert_eq!(prepay_summaries(prior, 2).unwrap(), 0);
    assert!(prepay_summaries(prior + 1, 2).is_err());
    assert!(prepay_summaries(u128::MAX, 2).is_err());
    for outputs in [0, 9, u64::MAX] {
        assert!(prepay_summaries(0, outputs).is_err());
    }
    assert_eq!(
        prepay_summaries(17, 1).unwrap(),
        SOURCE_WORK - 17 - pllm_models::SOURCE_SCALAR_SUMMARY_WORK
    );
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let config = import(&root, raw());
    for bad in [
        DecoderWorkload {
            max_input_tokens: u64::MAX,
            ..workload()
        },
        DecoderWorkload {
            max_new_tokens: 0,
            ..workload()
        },
        DecoderWorkload {
            max_new_tokens: 9,
            ..workload()
        },
    ] {
        assert!(ServerSourcePolicy::checked_public(
            &experiment(),
            &config,
            bad,
            WeightMode::Public,
            execution()
        )
        .is_err());
    }
}

#[test]
fn actual_prefix_plan_growth_retains_origin_and_cumulative_source_debit() {
    // Metadata/SSA construction ONLY. No tensor bytes, Program, arithmetic,
    // Worker, token feedback issuer, or SDK request is constructed here.
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut doc: serde_json::Value = serde_json::from_slice(&server_raw()).unwrap();
    doc["budget"]["max_input_tokens"] = json!(4);
    let server = import(&root, serde_json::to_vec(&doc).unwrap());
    let config = import(&root, raw());
    let limits = admitted_limits(&server, &config, None, workload()).unwrap();
    assert_eq!(limits.expansion.max_tensor_elements, 50);
    let before = SOURCE_WORK - limits.max_work;
    assert_eq!(
        before,
        u64::try_from(server.payload().resources().work.total * 2).unwrap()
            + config.payload().raw().len() as u64
            + 2 * pllm_models::SOURCE_SCALAR_SUMMARY_WORK
    );
    let boot = original_source::bootstrap_bytes(&config, limits).unwrap();
    let first = original_source::quote(
        config,
        root.charge_external(boot).unwrap(),
        workload(),
        limits,
    )
    .unwrap()
    .construct()
    .unwrap();
    let initial_work = first.resources().prepaid_work;
    let next = DecoderWorkload {
        max_input_tokens: 5,
        max_new_tokens: 1,
        ..workload()
    };
    let boot = first.lower_bootstrap_bytes(limits).unwrap();
    let second = first
        .lower(root.charge_external(boot).unwrap(), next, limits)
        .unwrap();
    assert_eq!(first.plan().prefill.query_sequence, 4);
    assert_eq!(second.plan().prefill.query_sequence, 5);
    assert!(first.same_origin(&second));
    assert!(!first.same_plan(&second));
    assert!(std::ptr::eq(first.budget(), second.budget()));
    assert!(second.resources().prepaid_work > initial_work);
    assert_eq!(limits.max_work, SOURCE_WORK - before); // Policy debit never refunded.
    let boot = second.lower_bootstrap_bytes(limits).unwrap();
    let exhausted = original_source::Limits {
        max_work: second.resources().prepaid_work,
        ..limits
    };
    assert!(first
        .lower(root.charge_external(boot).unwrap(), next, exhausted)
        .is_err());
    assert!(second.lower_bootstrap_bytes(limits).is_err()); // Shared ledger burned, no sibling reset.
}

#[test]
fn one_output_permissions_unchanged_and_no_server_maximum_row_substitution() {
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let config = import(&root, raw());
    let once = DecoderWorkload {
        max_new_tokens: 1,
        ..workload()
    };
    let single = ServerSourcePolicy::checked_public(
        &experiment(),
        &config,
        once,
        WeightMode::Public,
        execution(),
    )
    .unwrap()
    .limits_for(&config, once)
    .unwrap();
    assert_eq!(single.expansion.max_tensor_elements, 32);
    for max_input in [4, 8, 32] {
        let mut server = experiment();
        server.budget.max_input_tokens = max_input;
        let policy = ServerSourcePolicy::checked_public(
            &server,
            &config,
            workload(),
            WeightMode::Public,
            execution(),
        )
        .unwrap();
        assert_eq!(
            policy
                .limits_for(&config, workload())
                .unwrap()
                .expansion
                .max_tensor_elements,
            50
        );
    }
}
