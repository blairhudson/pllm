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

#[test]
fn original_raw_counter_and_borrows_survive_four_to_five_and_parent_drop() {
    use std::ptr;

    const SPARE: usize = 73;
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let monitor = root.try_clone().unwrap();
    let baseline = root.budget().used_bytes();
    let capacity = RAW.len() + SPARE;
    let before = root.budget().used_bytes();
    let original = root
        .charge_external(capacity)
        .unwrap()
        .try_build(|counter| {
            assert!(ptr::eq(counter, root.budget()));
            // Admit backing and inline wrapper before the allocating factory.
            assert_eq!(
                counter.used_bytes(),
                before + capacity + size_of::<OwnedCharged<Vec<u8>>>()
            );
            let mut bytes_owner = Vec::with_capacity(capacity);
            bytes_owner.extend_from_slice(RAW);
            assert_eq!(bytes_owner.capacity(), capacity);
            Ok(bytes_owner)
        })
        .unwrap();
    let raw_pointer = original.payload().as_ptr();
    let raw_charge = original.charged_bytes();
    let source = imported_owned(&root, original);
    assert!(ptr::eq(source.budget(), source.payload().budget()));
    assert_eq!(source.payload().raw().as_ptr(), raw_pointer);
    assert_eq!(
        source.payload().raw_owner().payload().capacity(),
        RAW.len() + SPARE
    );
    assert_eq!(source.payload().raw_owner().charged_bytes(), raw_charge);
    let control = bootstrap_bytes(&source, limits()).unwrap();
    let quoted = quote(
        source,
        root.charge_external(control).unwrap(),
        workload(4),
        limits(),
    )
    .unwrap();
    // Observe independent held guards, not resource-reported owner totals.
    // Scalars only: no extra owners/references survive construction or drops.
    let source_charge = quoted.source.charged_bytes() + raw_charge;
    let root_charge = quoted.attempt.root.charged_bytes();
    let attempt_charge = quoted.attempt.controller.as_ref().unwrap().charged_bytes();
    let quote_charge = quoted.working.charged_bytes();
    let predicted_working = quoted.resources().working_charge_bytes;
    let predicted_peak = quoted.resources().original_owners_peak_bytes;
    assert_eq!(
        root.budget().used_bytes(),
        baseline + source_charge + root_charge + attempt_charge + quote_charge
    );
    assert_eq!(
        quoted.resources().original_owner_bytes,
        source_charge + root_charge + attempt_charge
    );
    let first = quoted.construct().unwrap();
    let working4 = first.inner.payload().configured.get().unwrap().charged_bytes();
    assert_eq!(predicted_working, working4);
    assert_eq!(predicted_peak, source_charge + root_charge + attempt_charge + working4);
    assert_eq!(
        first.resources().original_owner_bytes,
        source_charge + root_charge + attempt_charge
    );
    assert_eq!(first.resources().working_charge_bytes, working4);
    assert_eq!(first.resources().original_owners_peak_bytes, predicted_peak);
    assert_eq!(
        root.budget().used_bytes(),
        baseline + source_charge + root_charge + working4
    );
    assert_eq!(root.budget().used_bytes(), baseline + predicted_peak - attempt_charge);
    assert_eq!(
        (
            first.plan().prefill.query_sequence,
            first.plan().decode.query_sequence,
            first.plan().decode.maximum_key_sequence
        ),
        (4, 1, 5)
    );
    // Independent normal-model oracle is test-only, outside the named Budget;
    // it consumes original config bytes, never a serialized/reconstructed plan.
    assert_eq!(
        first.plan(),
        &pllm_models::lower_model_json_with_budget(RAW, workload(4), limits().expansion)
            .unwrap()
    );
    let borrowed_plan = first.plan();
    let borrowed_config = first.raw_configuration();
    let resident = root.budget().used_bytes();
    let control = first.lower_bootstrap_bytes(limits()).unwrap();
    assert_eq!(
        root.budget().used_bytes(),
        resident,
        "pure bootstrap must not reserve/drop owner bytes"
    );
    let second = first
        .lower(
            root.charge_external(control).unwrap(),
            workload(5),
            limits(),
        )
        .unwrap();
    assert_eq!(borrowed_plan.prefill.query_sequence, 4);
    assert_eq!(borrowed_config.as_ptr(), raw_pointer);
    assert_eq!(
        second.plan(),
        &pllm_models::lower_model_json_with_budget(RAW, workload(5), limits().expansion)
            .unwrap()
    );
    assert_eq!(
        (
            second.plan().prefill.query_sequence,
            second.plan().decode.query_sequence,
            second.plan().decode.maximum_key_sequence
        ),
        (5, 1, 6)
    );
    assert_eq!(second.workload(), workload(5));
    assert!(first.same_origin(&second));
    assert!(!first.same_plan(&second));
    assert!(ptr::eq(first.budget(), second.budget()));
    let root5 = second.inner.charged_bytes();
    let working5 = second.inner.payload().configured.get().unwrap().charged_bytes();
    // quote and public lower split the same primitive_attempt_control_bytes;
    // neither resizes that controller. Its captured actual charge applies here.
    let incoming5 = source_charge + root_charge + working4 + root5 + attempt_charge;
    assert_eq!(second.resources().original_owner_bytes, incoming5);
    assert_eq!(second.resources().working_charge_bytes, working5);
    assert_eq!(second.resources().original_owners_peak_bytes, incoming5 + working5);
    let resident = baseline + source_charge + root_charge + working4 + root5 + working5;
    assert_eq!(monitor.budget().used_bytes(), resident);
    assert_eq!(
        monitor.budget().used_bytes(),
        baseline + second.resources().original_owners_peak_bytes - attempt_charge
    );
    drop(first); // second retains the ORIGINAL plan/config root, not a copy.
    assert_eq!(monitor.budget().used_bytes(), resident);
    drop(root);
    assert_eq!(
        monitor.budget().used_bytes(),
        resident - size_of::<OwnedBudget>()
    );
    assert!(ptr::eq(second.budget(), monitor.budget()));
    assert_eq!(second.raw_configuration(), RAW);
    assert_eq!(second.raw_configuration().as_ptr(), raw_pointer);
    drop(second);
    assert_eq!(
        monitor.budget().used_bytes(),
        baseline - size_of::<OwnedBudget>()
    );
    // This is source-factory workload growth, NOT accepted token feedback.
}

fn attempts() -> usize {
    FACTORIES.with(|n| n.get())
}

// Regression fixtures for authenticated, untransformed registered output;
// not a universal proof.
#[derive(Default)]
struct AttributeCounts {
    values: usize,
    keys: usize,
    key_bytes: usize,
    string_bytes: usize,
    epsilons: usize,
    depth: usize,
}

fn count_registered_attributes(
    value: &serde_json::Value,
    epsilon: bool,
    depth: usize,
    e: usize,
) -> AttributeCounts {
    use serde_json::Value;
    assert!(depth <= 2);
    let mut count = AttributeCounts {
        values: 1,
        depth,
        ..AttributeCounts::default()
    };
    match value {
        Value::Array(_) => panic!("registered attributes must not contain arrays"),
        Value::Object(fields) => {
            if depth == 0 {
                assert!(fields.len() <= 9);
            }
            for (key, child) in fields {
                assert!(key.len() <= 32);
                count.keys += 1;
                count.key_bytes += key.len();
                let child = count_registered_attributes(child, key == "epsilon", depth + 1, e);
                count.values += child.values;
                count.keys += child.keys;
                count.key_bytes += child.key_bytes;
                count.string_bytes += child.string_bytes;
                count.epsilons += child.epsilons;
                count.depth = count.depth.max(child.depth);
            }
        }
        Value::String(text) => {
            assert!(text.len() <= if epsilon { e } else { 64 });
            count.string_bytes += text.len();
            count.epsilons += usize::from(epsilon);
        }
        _ => {}
    }
    count
}

fn registered_decoder_fixture(family: &str, layers: usize) -> serde_json::Value {
    let mut config: serde_json::Value = serde_json::from_slice(RAW).unwrap();
    config["model_type"] = serde_json::json!(family);
    config["num_hidden_layers"] = serde_json::json!(layers);
    config["tie_word_embeddings"] = serde_json::json!(layers == 1);
    config["attention_bias"] = serde_json::json!(false);
    if family == "qwen3" {
        config["rms_norm_eps"] = serde_json::json!("0.000001");
        config["head_dim"] = serde_json::json!(2);
        config["attention_dropout"] = serde_json::json!("0");
        config["use_cache"] = serde_json::json!(true);
        config["max_window_layers"] = serde_json::json!(layers);
    } else if family == "llama" {
        config.as_object_mut().unwrap().remove("rms_norm_eps");
        if layers == 2 {
            config["rope_scaling"] = serde_json::json!({
                "rope_type": "llama3", "factor": 8.0,
                "original_max_position_embeddings": 4,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0
            });
        }
    }
    config
}

fn assert_registered_attribute_bounds(source: &OriginalSourcePlan, layers: usize) {
    let plan = source.plan();
    assert!(plan.transformations.is_empty());
    assert!(matches!(
        plan.model_family.as_str(),
        "qwen2" | "qwen3" | "llama"
    ));
    assert!((1..=2).contains(&layers)); // Small fixtures keep arithmetic bounded.
    let o_bound = 2 * (30 * layers + 6);
    let e = source.raw_configuration().len().max(1024);
    let mut total = AttributeCounts::default();
    let mut operations = 0;
    for graph in [&plan.prefill, &plan.decode] {
        assert!(!graph.operations.is_empty());
        let per_layer = if plan.model_family == "qwen3" {
            30
        } else {
            28
        };
        assert_eq!(graph.operations.len(), per_layer * layers + 6);
        let mut phase_epsilons = 0;
        let mut empty_attributes = 0;
        for operation in &graph.operations {
            assert!(operation.attributes.is_object());
            let count = count_registered_attributes(&operation.attributes, false, 0, e);
            assert!(count.values <= 15);
            assert!(count.keys <= 14);
            empty_attributes += usize::from(count.keys == 0);
            phase_epsilons += count.epsilons;
            total.values += count.values;
            total.keys += count.keys;
            total.key_bytes += count.key_bytes;
            total.string_bytes += count.string_bytes;
            total.epsilons += count.epsilons;
            operations += 1;
        }
        assert!(empty_attributes > 0);
        let norms_per_layer = if plan.model_family == "qwen3" {
            4
        } else {
            2
        };
        assert_eq!(phase_epsilons, norms_per_layer * layers + 1);
    }
    assert!(operations > 0 && operations <= o_bound);
    assert!(total.values <= 15 * o_bound);
    assert!(total.keys <= 14 * o_bound);
    assert!(total.key_bytes <= 448 * o_bound);
    assert!(total.string_bytes <= 896 * o_bound + (8 * layers + 2) * e);
    assert!(total.epsilons > 0 && total.epsilons <= 8 * layers + 2);
    assert!(total.values + total.keys <= 29 * o_bound);
}

#[test]
fn registered_output_attribute_bounds_and_scaled_rope() {
    for family in ["qwen2", "qwen3", "llama"] {
        for layers in [1, 2] {
            let config = registered_decoder_fixture(family, layers);
            let raw = serde_json::to_vec(&config).unwrap();
            let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let source = prepared(&root, &raw, 4, limits()).construct().unwrap();
            assert_registered_attribute_bounds(&source, layers);
            for graph in [&source.plan().prefill, &source.plan().decode] {
                let attributes = |id: &str| {
                    &graph
                        .operations
                        .iter()
                        .find(|op| op.id == id)
                        .unwrap()
                        .attributes
                };
                let head = attributes("output_head");
                assert_eq!(head["tied"], layers == 1);
                assert_eq!(
                    head["weight"],
                    if layers == 1 {
                        "model.embed_tokens.weight"
                    } else {
                        "lm_head.weight"
                    }
                );
                let bias = &attributes("layer.0.q_linear")["bias"];
                if family == "qwen2" {
                    assert_eq!(bias, "model.layers.0.self_attn.q_proj.bias");
                } else {
                    assert!(bias.is_null());
                }
                assert_eq!(
                    attributes("layer.0.input_norm")["epsilon"].as_str(),
                    Some(if family == "qwen2" {
                        "1e-6"
                    } else {
                        "0.000001"
                    })
                );
                for layer in 0..layers {
                    for suffix in ["rope_q", "rope_k"] {
                        let rope = attributes(&format!("layer.{layer}.{suffix}"));
                        if family == "llama" && layers == 2 {
                            let count = count_registered_attributes(
                                rope,
                                false,
                                0,
                                raw.len().max(1024),
                            );
                            assert_eq!(rope.as_object().unwrap().len(), 9);
                            assert_eq!((count.keys, count.values, count.depth), (14, 15, 2));
                            let scaling = &rope["frequency_scaling"];
                            assert_eq!(
                                scaling,
                                &serde_json::json!({
                                    "kind": "wavelength_transition", "factor": 8.0,
                                    "original_max_position_embeddings": 4,
                                    "low_freq_factor": 1.0, "high_freq_factor": 4.0
                                })
                            );
                            let fields = scaling.as_object().unwrap();
                            assert_eq!(fields.len(), 5);
                            assert!(fields.values().all(|v| !v.is_object() && !v.is_array()));
                        } else {
                            assert!(rope.get("frequency_scaling").is_none());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn long_epsilon_uses_configuration_derived_allowance() {
    // Trailing fractional zeroes preserve positive finite 1e-6, not an underflow
    // literal. Existing parsing/semantic validation remains unchanged.
    let epsilon = format!("0.000001{}", "0".repeat(1100));
    assert!(epsilon.len() > 1024);
    assert_eq!(epsilon.parse::<f64>().unwrap(), 0.000001);
    let mut config = registered_decoder_fixture("qwen3", 2);
    config["rms_norm_eps"] = serde_json::json!(epsilon);
    let raw = serde_json::to_vec(&config).unwrap();
    let mut policy = limits();
    policy.factory.max_string_bytes = epsilon.len(); // Only test-local metadata allowance.
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let source = prepared(&root, &raw, 4, policy).construct().unwrap();
    source.plan().validate().unwrap();
    assert!(raw.len() > epsilon.len());
    assert_registered_attribute_bounds(&source, 2);
    for graph in [&source.plan().prefill, &source.plan().decode] {
        let norms: Vec<_> = graph
            .operations
            .iter()
            .filter_map(|op| op.attributes.get("epsilon"))
            .collect();
        assert_eq!(norms.len(), 9);
        assert!(norms
            .iter()
            .all(|value| value.as_str() == Some(epsilon.as_str())));
    }
}

#[test]
fn nested_rotary_scalar_and_unsupported_mapping_refusals() {
    let mut config = registered_decoder_fixture("llama", 2);
    config["rope_scaling"]["factor"] = serde_json::json!({"nested": 8.0});
    let scaling = config["rope_scaling"].as_object().unwrap();
    assert_eq!(scaling.len(), 5);
    for key in [
        "rope_type",
        "factor",
        "original_max_position_embeddings",
        "low_freq_factor",
        "high_freq_factor",
    ] {
        assert!(scaling.contains_key(key));
    }
    // Five required fields pass source_descriptor's field gate; genuine factory
    // clones nested factor into descriptor before scalar validation refuses it.
    let raw = serde_json::to_vec(&config).unwrap();
    let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let pending = prepared(&root, &raw, 4, limits());
    let before = attempts();
    assert!(matches!(pending.construct(), Err(Error::Factory)));
    assert_eq!(attempts(), before + 1);

    config["model_type"] = serde_json::json!("not_a_registered_mapping");
    let raw = serde_json::to_vec(&config).unwrap();
    let source = imported(&root, &raw);
    let control = bootstrap_bytes(&source, limits()).unwrap();
    let before = attempts();
    assert!(matches!(
        quote(
            source,
            root.charge_external(control).unwrap(),
            workload(4),
            limits()
        ),
        Err(Error::Quote)
    ));
    assert_eq!(attempts(), before);
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

fn decode_limits() -> crate::json_bytes_codec::Limits {
    crate::json_bytes_codec::Limits {
        max_source_bytes: BOOTSTRAP_CONFIG_BYTES, max_points: BOOTSTRAP_CONFIG_BYTES,
        max_work: 1_000_000, max_working_payload: 8 << 20,
    }
}

fn decode_raw(budget: &OwnedBudget, bytes: &[u8]) -> OwnedCharged<Vec<u8>> {
    budget.charge_external(bytes.len()).unwrap().try_build(|_| Ok(bytes.to_vec())).unwrap()
}

#[test]
fn native_reserved_decode_same_root_uncommitted_payment_drop_burn_and_committed_survival() {
    use crate::json_bytes_codec::reserved_codec_control_bytes;
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let mut admission = native_admission(&budget);
    let root = Arc::clone(&admission.root);
    let committed = begin_attempt(&root,
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 3)
        .unwrap().commit().unwrap();
    let retained = budget.budget().used_bytes();
    let phase = admission.reserve_bootstrap_phase(
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 7).unwrap();
    let before = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
    assert_eq!(before, 4 + 8); // genuine earlier sibling + phase visits/fees
    let raw = decode_raw(&budget, b"a");
    let ptr = raw.payload().as_ptr();
    let q = phase.quote_decoding(raw,
        budget.charge_external(reserved_codec_control_bytes().unwrap()).unwrap(), decode_limits()).unwrap();
    assert_eq!(q.resources().prepaid_work, 131); // literal INCREMENT, no second fee
    assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, before + 131);
    let pending = q.construct().unwrap();
    assert!(root.payload().configured.get().is_none());
    assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, before + 131);
    assert!(!root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    assert_eq!(pending.payload().decoded().original_owner().payload().as_ptr(), ptr);
    assert!(std::ptr::eq(pending.budget(), root.budget()));
    drop(pending);
    assert_eq!(budget.budget().used_bytes(), retained);
    let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
    assert!(ledger.burned);
    assert_eq!(ledger.used, before + 131);
    drop(ledger);
    assert!(matches!(begin_attempt(&root,
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0), Err(Error::Burned)));
    assert!(committed.is_for(budget.budget()));
    assert_eq!(budget.budget().used_bytes(), retained);
    drop(admission);
    drop(committed);
    drop(root);
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn native_reserved_decode_literal_cumulative_exact_and_one_short_preserve_paid_baseline() {
    use crate::json_bytes_codec::{self as codec, reserved_codec_control_bytes};
    for (bytes, increment) in [(&b""[..], 124u64), (&b"a"[..], 131), (&b"abc"[..], 145)] {
        for short in [false, true] {
            let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut admission = native_admission(&budget);
            let root = Arc::clone(&admission.root);
            let phase = admission.reserve_bootstrap_phase(
                budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 19).unwrap();
            let paid = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
            assert_eq!(paid, 20);
            let remaining = increment - u64::from(short);
            phase.attempt.debit(BOOTSTRAP_SOURCE_WORK - paid - remaining).unwrap();
            let before = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
            assert_eq!(before, BOOTSTRAP_SOURCE_WORK - remaining);
            let q = phase.quote_decoding(decode_raw(&budget, bytes),
                budget.charge_external(reserved_codec_control_bytes().unwrap()).unwrap(), decode_limits());
            if short {
                assert_eq!(q.err(), Some(codec::Error::Source(Error::Work)));
                let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
                assert!(ledger.burned);
                assert_eq!(ledger.used, before); // failed addition did not refund/change baseline
            } else {
                let q = q.unwrap();
                assert_eq!(q.resources().prepaid_work, increment);
                let pending = q.construct().unwrap();
                {
                    let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
                    assert_eq!(ledger.used, BOOTSTRAP_SOURCE_WORK);
                    assert!(!ledger.burned);
                }
                drop(pending);
                assert!(root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
            }
            assert!(root.payload().configured.get().is_none());
            assert!(matches!(begin_attempt(&root,
                budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0), Err(Error::Burned)));
        }
    }
}

#[test]
fn native_reserved_decode_quote_abandonment_and_foreign_refusals_burn_only_actual_root() {
    use crate::json_bytes_codec::{self as codec, reserved_codec_control_bytes};
    for mode in 0..4 {
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut admission = native_admission(&budget);
        let other = native_admission(&foreign);
        let root = Arc::clone(&admission.root);
        let baseline = budget.budget().used_bytes();
        let phase = admission.reserve_bootstrap_phase(
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 5).unwrap();
        let raw = decode_raw(if mode == 1 { &foreign } else { &budget }, b"a");
        let working_budget = if mode == 2 { &foreign } else { &budget };
        let working = working_budget.charge_external(
            reserved_codec_control_bytes().unwrap() - usize::from(mode == 3)).unwrap();
        let result = phase.quote_decoding(raw, working, decode_limits());
        if mode == 0 { drop(result.unwrap()); }
        else { assert_eq!(result.err(), Some(if mode == 3 { codec::Error::Capacity } else { codec::Error::Counter })); }
        let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert!(ledger.burned);
        assert_eq!(ledger.used, if mode == 0 { 6 + 131 } else { 6 });
        drop(ledger);
        assert!(!other.root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
        assert_eq!(other.root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 0);
        assert_eq!(budget.budget().used_bytes(), baseline);
        assert!(root.payload().configured.get().is_none());
    }
}

#[test]
fn native_reserved_decode_late_sibling_burn_cannot_be_cleared_by_construct() {
    use crate::json_bytes_codec::{self as codec, reserved_codec_control_bytes};
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut admission = native_admission(&budget);
    let root = Arc::clone(&admission.root);
    let baseline = budget.budget().used_bytes();
    let phase = admission.reserve_bootstrap_phase(
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
    let q = phase.quote_decoding(decode_raw(&budget, b"a"),
        budget.charge_external(reserved_codec_control_bytes().unwrap()).unwrap(), decode_limits()).unwrap();
    let sibling = begin_attempt(&root,
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
    drop(sibling);
    assert_eq!(q.construct().err(), Some(codec::Error::Source(Error::Burned)));
    assert!(root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    assert!(root.payload().configured.get().is_none());
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn native_reserved_decode_every_late_failure_burns_actual_root_retains_work_and_committed_control() {
    use crate::json_bytes_codec::{self as codec, ReplayRefusal, ReplayRefusalGuard};
    // 0 partial Resource; 1 partial panic; 2 factory panic; 3 malformed count;
    // 4 working growth; 5 pending inline admission; 6 injected Allocation.
    // Literal work: ab => 7*2+124 = 138; FF => 7*1+124 = 131.
    for mode in 0..7 {
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = budget.budget().used_bytes();
        let mut admission = native_admission(&budget);
        let root = Arc::clone(&admission.root);
        let committed = begin_attempt(&root,
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 3)
            .unwrap().commit().unwrap();
        let committed_bytes = committed.charged_bytes();
        let retained = budget.budget().used_bytes();
        assert_eq!(retained, baseline + root.charged_bytes() + committed_bytes);
        let phase = admission.reserve_bootstrap_phase(
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 7).unwrap();
        let prior = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
        assert_eq!(prior, 4 + 8);
        let bytes: &[u8] = if mode == 3 { &[0xff] } else { b"ab" };
        let q = phase.quote_decoding(decode_raw(&budget, bytes),
            budget.charge_external(codec::reserved_codec_control_bytes().unwrap()).unwrap(), decode_limits());
        if mode == 3 {
            assert_eq!(q.err(), Some(codec::Error::Decode(codec::DecodeError {
                kind: codec::DecodeFailure::InvalidStart, start: 0, end: 1, error_input_base: 0,
            })));
        } else {
            let q = q.unwrap();
            assert_eq!(q.resources().prepaid_work, 138);
            let ballast = if mode == 4 || mode == 5 {
                // Quote has control-floor working payload, two points => eight
                // growth bytes. Pending inline is its ACTUAL declared layout.
                let needed = if mode == 4 { 8 } else {
                    8 + size_of::<OwnedCharged<codec::PendingDecodedCustody>>()
                        - size_of::<OwnedExternalCharge>()
                };
                Some(budget.charge_external(MAX_ALLOCATION_BYTES - budget.budget().used_bytes()
                    - size_of::<OwnedExternalCharge>() - (needed - 1)).unwrap())
            } else { None };
            let hook = match mode {
                0 => Some(ReplayRefusalGuard::arm(ReplayRefusal::PartialResource)),
                1 => Some(ReplayRefusalGuard::arm(ReplayRefusal::PartialPanic)),
                2 => Some(ReplayRefusalGuard::arm(ReplayRefusal::FactoryPanic)),
                6 => Some(ReplayRefusalGuard::arm(ReplayRefusal::Allocation)),
                _ => None,
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| q.construct()));
            if mode == 1 || mode == 2 { assert!(result.is_err()); }
            else {
                assert_eq!(result.unwrap().err(), Some(codec::Error::Native(
                    if mode == 6 { NativeError::Allocation } else { NativeError::Resource })));
            }
            drop(hook); // thread-local flags restored after success/error/unwind
            drop(ballast);
        }
        let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert!(ledger.burned); // NOT SourceAdmission.reserved
        assert_eq!(ledger.used, prior + if mode == 3 { 131 } else { 138 });
        assert_eq!(ledger.limit, BOOTSTRAP_SOURCE_WORK);
        drop(ledger);
        assert!(root.payload().configured.get().is_none());
        assert_eq!(budget.budget().used_bytes(), retained);
        assert!(committed.is_for(budget.budget()));
        assert_eq!(committed.charged_bytes(), committed_bytes);
        assert!(matches!(begin_attempt(&root,
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0), Err(Error::Burned)));
        assert_eq!(budget.budget().used_bytes(), retained);
        drop(admission);
        drop(committed);
        drop(root);
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn native_reserved_decode_work_precedes_malformed_count_with_literal_paid_baseline() {
    use crate::json_bytes_codec::{self as codec, reserved_codec_control_bytes};
    for cumulative_short in [false, true] {
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = budget.budget().used_bytes();
        let mut admission = native_admission(&budget);
        let root = Arc::clone(&admission.root);
        let committed = begin_attempt(&root,
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 3)
            .unwrap().commit().unwrap();
        let retained = budget.budget().used_bytes();
        let phase = admission.reserve_bootstrap_phase(
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 7).unwrap();
        assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 12);
        let mut limits = decode_limits();
        if cumulative_short {
            phase.attempt.debit(BOOTSTRAP_SOURCE_WORK - 12 - 130).unwrap();
        } else {
            limits.max_work = 130; // local cap short of literal FF debit 131
        }
        let prior = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
        let result = phase.quote_decoding(decode_raw(&budget, &[0xff]),
            budget.charge_external(reserved_codec_control_bytes().unwrap()).unwrap(), limits);
        // InvalidStart would win if malformed count took precedence. This is
        // error-order coverage, not a detector counter or an inference from
        // replay factory counters; detection ordering also rests on source order.
        assert_eq!(result.err(), Some(if cumulative_short {
            codec::Error::Source(Error::Work)
        } else { codec::Error::Work }));
        let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert!(ledger.burned);
        assert_eq!(ledger.used, if cumulative_short { prior } else { prior + 131 });
        assert_eq!(ledger.limit, BOOTSTRAP_SOURCE_WORK);
        drop(ledger);
        assert!(root.payload().configured.get().is_none());
        assert!(committed.is_for(budget.budget()));
        assert_eq!(budget.budget().used_bytes(), retained);
        drop(admission);
        drop(committed);
        drop(root);
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn native_reserved_decode_distinct_backings_and_oversized_working_coverage_are_retained() {
    use crate::json_bytes_codec::{self as codec, reserved_codec_control_bytes};
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let mut admission = native_admission(&budget);
    let root = Arc::clone(&admission.root);
    let committed = begin_attempt(&root,
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 3)
        .unwrap().commit().unwrap();
    let retained = budget.budget().used_bytes();
    let phase = admission.reserve_bootstrap_phase(
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 7).unwrap();
    let bytes = [0xed, 0xa0, 0x80, 0xed, 0xb0, 0x80];
    let raw = budget.charge_external(37).unwrap().try_build(|_| {
        let mut raw = Vec::new();
        raw.try_reserve_exact(37).map_err(|_| NativeError::Allocation)?;
        if raw.capacity() != 37 { return Err(NativeError::Resource); }
        raw.extend_from_slice(&bytes);
        Ok(raw)
    }).unwrap();
    let raw_ptr = raw.payload().as_ptr();
    let raw_charge = raw.charged_bytes();
    let controls = reserved_codec_control_bytes().unwrap();
    let working = budget.charge_external(controls + 4096).unwrap();
    let working_charge = working.charged_bytes();
    let before_quote = budget.budget().used_bytes();
    let q = phase.quote_decoding(raw, working, decode_limits()).unwrap();
    assert_eq!(q.resources().prepaid_work, 166); // literal 7*6+124
    assert_eq!(q.resources().required_working_payload, controls + 8);
    assert_eq!(budget.budget().used_bytes(), before_quote); // no excess refund
    let inline = size_of::<OwnedCharged<codec::PendingDecodedCustody>>() - size_of::<OwnedExternalCharge>();
    assert_eq!(q.resources().working_charge_bytes, working_charge + inline);
    let pending = q.construct().unwrap();
    let decoded = pending.payload().decoded();
    assert_eq!(decoded.codepoints(), [0xd800, 0xdc00]);
    assert_eq!(decoded.original_owner().payload().as_ptr(), raw_ptr);
    assert_eq!(decoded.original_owner().payload().len(), 6);
    assert_eq!(decoded.original_owner().payload().capacity(), 37);
    assert_eq!(decoded.original_owner().payload().as_slice(), bytes);
    assert_eq!(decoded.original_owner().charged_bytes(), raw_charge);
    assert_ne!(decoded.codepoints().as_ptr().cast::<u8>(), raw_ptr);
    assert_eq!(decoded.resources().point_capacity_bytes, 8);
    assert_eq!(pending.charged_bytes(), working_charge + inline);
    assert_eq!(budget.budget().used_bytes(), before_quote + inline);
    assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 12 + 166);
    assert!(!root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    drop(pending);
    assert!(root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 12 + 166);
    assert_eq!(budget.budget().used_bytes(), retained);
    assert!(committed.is_for(budget.budget()));
    drop(admission);
    drop(committed);
    drop(root);
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn native_point_bridge_terminal_entry_refusal_has_no_fictitious_payment() {
    use crate::json_bytes_codec::reserved_codec_control_bytes;
    // Retained policy-review regression: the now-approved bounded terminal
    // refusal has no fictional payment. ANY positive debit fails in these states.
    // Probe/breaker are genuine separately charged sibling attempts, not bridge
    // controllers, replacement authority, new ledgers or a configured shortcut.
    for already_burned in [true, false] {
        let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = budget.budget().used_bytes();
        let mut admission = native_admission(&budget);
        let root = Arc::clone(&admission.root);
        let probe = begin_attempt(&root,
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
        let mut breaker = Some(begin_attempt(&root,
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap());
        let phase = admission.reserve_bootstrap_phase(
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
        let pending = phase.quote_decoding(decode_raw(&budget, b"null"),
            budget.charge_external(reserved_codec_control_bytes().unwrap()).unwrap(), decode_limits())
            .unwrap().construct().unwrap();
        assert_eq!(pending.payload().decoded().codepoints(), [110, 117, 108, 108]);
        let used = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
        assert_eq!(used, 1 + 1 + 1 + 152); // codec literal 7*4+124, three real fees
        assert!(Arc::ptr_eq(&probe.root, &root));
        if already_burned {
            drop(breaker.take());
        } else {
            probe.debit(BOOTSTRAP_SOURCE_WORK - used).unwrap();
        }
        let before = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
        let bytes = budget.budget().used_bytes();
        assert_eq!(probe.debit(1), Err(if already_burned { Error::Burned } else { Error::Work }));
        {
            let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
            assert!(ledger.burned);
            assert_eq!(ledger.used, before); // no entry payment was made
            assert_eq!(ledger.limit, BOOTSTRAP_SOURCE_WORK);
        }
        assert_eq!(probe.debit(1), Err(Error::Burned));
        assert_eq!(budget.budget().used_bytes(), bytes); // actual custody retained
        assert_eq!(pending.payload().decoded().original_owner().payload().as_slice(), b"null");
        assert!(root.payload().configured.get().is_none());
        drop(pending);
        drop(breaker);
        drop(probe);
        assert!(root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
        assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, before);
        assert_eq!(budget.budget().used_bytes(), baseline + root.charged_bytes());
        drop(admission);
        drop(root);
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

fn point_limits() -> crate::plan_input::json::PointLexicalLimits {
    use crate::plan_input::json::{PointLexicalLimits, PointWork};
    PointLexicalLimits { max_raw_capacity_bytes: 1 << 20, max_points: 1 << 20,
        max_nodes: 4096, max_depth: 64, max_string_points: 1 << 16,
        max_total_string_points: 1 << 18, max_number_points: 4096,
        max_payload_bytes: 8 << 20, max_work: 1_000_000,
        work: PointWork { quotation: 1_000_000, replay: 1_000_000, strings: 1_000_000,
            classify: 1_000_000, deduplicate: 1_000_000, cleanup: 1_000_000,
            transition: 1_000_000, total: 1_000_000 } }
}
fn point_pending(budget: &OwnedBudget, admission: &mut SourceAdmission, bytes: &[u8])
    -> OwnedCharged<crate::json_bytes_codec::PendingDecodedCustody> {
    let phase = admission.reserve_bootstrap_phase(
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
    phase.quote_decoding(decode_raw(budget, bytes),
        budget.charge_external(crate::json_bytes_codec::reserved_codec_control_bytes().unwrap()).unwrap(), decode_limits())
        .unwrap().construct().unwrap()
}

#[test]
fn native_point_entry_refusal_is_consuming_metadata_only_actual_root_burn() {
    use crate::plan_input::json::{quote_pending_lexical, point_lexical_control_bytes, point_test_counts, PointError};
    for mode in 0..6 {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = b.budget().used_bytes();
        let foreign = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut admission = native_admission(&b);
        let root = Arc::clone(&admission.root);
        let probe = begin_attempt(&root,
            b.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
        let pending = point_pending(&b, &mut admission, b"null");
        let mut limits = point_limits();
        let controls = point_lexical_control_bytes().unwrap();
        let working = if mode == 0 { foreign.charge_external(controls).unwrap() }
            else { b.charge_external(controls - usize::from(mode == 1)).unwrap() };
        // Mode 2 burn is produced by a GENUINE sibling's abandonment.
        if mode == 2 {
            drop(begin_attempt(&root, b.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap());
        }
        if mode == 3 {
            let used = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
            probe.debit(BOOTSTRAP_SOURCE_WORK - used).unwrap();
        }
        if mode == 4 { limits.work.quotation = 238; } // literal null entry 239
        if mode == 5 { limits.max_raw_capacity_bytes = 0; }
        let paid = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
        let counts = point_test_counts();
        let result = quote_pending_lexical(pending, working, limits);
        let expected = match mode {
            0 => PointError::Input(crate::plan_input::Error::Counter),
            1 => PointError::Input(crate::plan_input::Error::Bootstrap),
            2 => PointError::Source(Error::Burned), 3 => PointError::Source(Error::Work),
            4 => PointError::Input(crate::plan_input::Error::WorkLimit),
            _ => PointError::Input(crate::plan_input::Error::SourceLimit),
        };
        assert_eq!(result.err(), Some(expected));
        assert_eq!(point_test_counts(), counts); // ZERO scan/factory delta
        let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert!(ledger.burned); assert_eq!(ledger.used, paid); drop(ledger);
        assert_eq!(probe.debit(0), Err(Error::Burned));
        assert!(root.payload().configured.get().is_none());
        drop(probe); drop(admission);
        assert_eq!(b.budget().used_bytes(), baseline + root.charged_bytes());
        drop(root); assert_eq!(b.budget().used_bytes(), baseline);
    }
}

#[test]
fn native_point_literal_entry_and_complete_work_exact_one_short_same_ledger() {
    use crate::plan_input::json::{quote_pending_lexical, point_lexical_control_bytes, point_test_counts, PointError};
    for remaining in [238, 239, 415, 416] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut admission = native_admission(&b);
        let root = Arc::clone(&admission.root);
        let probe = begin_attempt(&root, b.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
        let pending = point_pending(&b, &mut admission, b"null");
        let used = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
        assert_eq!(used, 1 + 1 + 152); // probe fee, phase fee, unchanged decoder
        probe.debit(BOOTSTRAP_SOURCE_WORK - used - remaining).unwrap();
        let before = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
        let counts = point_test_counts();
        let q = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits());
        if remaining == 238 {
            assert_eq!(q.err(), Some(PointError::Source(Error::Work)));
            assert_eq!(point_test_counts(), counts);
            assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, before);
        } else {
            let q = q.unwrap();
            assert_eq!(q.resources().work.quotation, 239); // Q(4)=219 + E20
            assert_eq!(q.resources().work.replay, 144);
            assert_eq!(q.resources().work.strings, 10);
            assert_eq!(q.resources().work.classify, 1);
            assert_eq!(q.resources().work.deduplicate, 1);
            assert_eq!(q.resources().work.cleanup, 8);
            assert_eq!(q.resources().work.transition, 13);
            assert_eq!(q.resources().work.total, 416);
            assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, before + 239);
            if remaining == 239 { drop(q); }
            else {
                let output = q.construct();
                if remaining == 415 {
                    assert_eq!(output.err(), Some(PointError::Source(Error::Work)));
                    assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, before + 239);
                    assert_eq!(point_test_counts(), (counts.0 + 1, counts.1));
                } else {
                    let output = output.unwrap();
                    assert_eq!(output.payload().node(0).unwrap().point_range(), 0..4);
                    assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, before + 416);
                    assert!(root.payload().configured.get().is_none());
                    drop(output);
                }
            }
        }
        assert!(root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    }
    // An unrelated REAL sibling can burn an already funded quote. No second
    // scan/factory or future payment can reset that terminal root.
    let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut admission = native_admission(&b);
    let root = Arc::clone(&admission.root);
    let pending = point_pending(&b, &mut admission, b"null");
    let q = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits()).unwrap();
    drop(begin_attempt(&root, b.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap());
    let paid = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
    let counters = point_test_counts();
    assert_eq!(q.construct().err(), Some(PointError::Source(Error::Burned)));
    assert_eq!(point_test_counts(), counters);
    assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, paid);
}

#[test]
fn native_point_custody_distinct_backings_excess_charge_and_fact_retirement() {
    use crate::plan_input::json::{quote_pending_lexical, point_lexical_control_bytes, point_test_retirement};
    let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = b.budget().used_bytes();
    let mut admission = native_admission(&b);
    let root = Arc::clone(&admission.root);
    let pending = point_pending(&b, &mut admission, b"\"\xf0\x9f\xa6\x80\"");
    let point_ptr = pending.payload().decoded().codepoints().as_ptr();
    let point_capacity = pending.payload().point_capacity();
    let raw_ptr = pending.payload().decoded().original_owner().payload().as_ptr();
    let raw_capacity = pending.payload().decoded().original_owner().payload().capacity();
    assert_ne!(point_ptr.cast::<u8>(), raw_ptr);
    let extra = point_lexical_control_bytes().unwrap() + 4096;
    let q = quote_pending_lexical(pending, b.charge_external(extra).unwrap(), point_limits()).unwrap();
    assert_eq!(q.resources().work.quotation, 221); // P3, Q201 + E20
    assert_eq!(q.resources().work.replay, 143); // scan7 + 2 + 134
    assert_eq!(q.resources().work.total, 397);
    assert_eq!(q.resources().working_charge_bytes, extra + size_of::<OwnedCharged<crate::plan_input::json::PointLexicalCustody>>());
    let output = q.construct().unwrap();
    assert_eq!(output.charged_bytes(), extra + size_of::<OwnedCharged<crate::plan_input::json::PointLexicalCustody>>());
    assert_eq!(output.payload().points().as_ptr(), point_ptr);
    assert_eq!(output.payload().decoded_owner().payload().point_capacity(), point_capacity);
    assert_eq!(output.payload().raw_owner().payload().as_ptr(), raw_ptr);
    assert_eq!(output.payload().raw_owner().payload().capacity(), raw_capacity);
    assert_eq!(output.payload().node(0).unwrap().point_range(), 0..3);
    assert_eq!(output.payload().node(0).unwrap().string_points(), Some([0x1f980].as_slice()));
    let live = b.budget().used_bytes();
    drop(output);
    assert_eq!(point_test_retirement(), Some((std::ptr::from_ref(b.budget()) as usize, live)));
    assert!(root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    drop(admission); drop(root);
    assert_eq!(b.budget().used_bytes(), baseline);
}

#[test]
fn native_point_refusal_partial_unwind_sibling_survival_and_paid_cleanup() {
    use crate::plan_input::json::{quote_pending_lexical, point_lexical_control_bytes,
        point_test_counts, point_test_retirement, PointRefusal, PointError};
    for mode in 0..9 {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let baseline = b.budget().used_bytes();
        let mut admission = native_admission(&b);
        let root = Arc::clone(&admission.root);
        let committed = begin_attempt(&root, b.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 3).unwrap().commit().unwrap();
        let probe = begin_attempt(&root, b.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
        let held = b.budget().used_bytes();
        let pending = point_pending(&b, &mut admission, b"[\"a\",\"b\"]");
        assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 4 + 1 + 1 + 187);
        let q = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits()).unwrap();
        assert_eq!(q.resources().work.quotation, 335);
        assert_eq!(q.resources().work.total, 564);
        let growth = q.resources().required_working_payload - point_lexical_control_bytes().unwrap();
        let inline = size_of::<OwnedCharged<crate::plan_input::json::PointLexicalCustody>>() - size_of::<OwnedExternalCharge>();
        let ballast = if mode == 4 || mode == 5 {
            let room = if mode == 4 { growth - 1 } else { growth + inline - 1 };
            Some(b.charge_external(MAX_ALLOCATION_BYTES - b.budget().used_bytes() - room - size_of::<OwnedExternalCharge>()).unwrap())
        } else { None };
        let counters = point_test_counts();
        let covered = b.budget().used_bytes() + growth + inline;
        let _hook = PointRefusal::arm(match mode { 0..=3 => mode + 1, 6 => 5, 7 => 6, 8 => 7, _ => 0 });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| q.construct()));
        if mode == 2 || mode == 3 || mode == 7 { assert!(result.is_err()); }
        else {
            let error = result.unwrap().err().unwrap();
            assert_eq!(error, PointError::Input(if mode == 8 { crate::plan_input::Error::Factory }
                else { crate::plan_input::Error::Native(if mode == 0 { NativeError::Allocation } else { NativeError::Resource }) }));
        }
        if mode == 4 || mode == 5 { assert_eq!(point_test_counts(), counters); }
        if matches!(mode, 0 | 1 | 2 | 6 | 7) {
            assert_eq!(point_test_retirement(), Some((std::ptr::from_ref(b.budget()) as usize, covered)));
        }
        let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert!(ledger.burned); assert_eq!(ledger.used, 193 + 564); drop(ledger);
        assert_eq!(probe.debit(0), Err(Error::Burned));
        assert!(committed.is_for(b.budget()));
        drop(ballast); assert_eq!(b.budget().used_bytes(), held);
        drop(probe); drop(committed); drop(admission); drop(root);
        assert_eq!(b.budget().used_bytes(), baseline);
    }
}

#[test]
fn native_point_literal_strings_surrogates_encodings_numbers_and_duplicates() {
    use crate::plan_input::json::{quote_pending_lexical, point_lexical_control_bytes, PointError, LexicalNumberClass};
    for (raw, expected) in [
        (b"\"\\ud800\"".as_slice(), vec![0xd800]), (b"\"\\udc00\"".as_slice(), vec![0xdc00]),
        (b"\"\\ud800\\udc00\"".as_slice(), vec![0x10000]),
        (b"\"\\ud800\\ud800\\udc00\"".as_slice(), vec![0xd800, 0x10000]),
        (b"\"\\ud800X\"".as_slice(), vec![0xd800, 88]),
        (b"\"\\ud800\\u0041\"".as_slice(), vec![0xd800, 65]),
        (b"\"\\u0000\"".as_slice(), vec![0]),
        (b"\"\xed\xa0\x80\xed\xb0\x80\"".as_slice(), vec![0xd800, 0xdc00]),
        (b"\"\xf0\x9f\xa6\x80\"".as_slice(), vec![0x1f980]),
        (b"\xef\xbb\xbf\"A\"".as_slice(), vec![65]),
        (b"\xff\xfe\x22\x00\x41\x00\x22\x00".as_slice(), vec![65]),
        (b"\x00\x00\xfe\xff\x00\x00\x00\x22\x00\x00\x00\x41\x00\x00\x00\x22".as_slice(), vec![65]),
    ] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut admission = native_admission(&b);
        let root = Arc::clone(&admission.root);
        let pending = point_pending(&b, &mut admission, raw);
        let output = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits()).unwrap().construct().unwrap();
        assert_eq!(output.payload().node(0).unwrap().string_points(), Some(expected.as_slice()));
        assert!(root.payload().configured.get().is_none());
    }
    for raw in [b"{\"a\":0,\"\\u0061\":1}".as_slice(), b"{\"\\ud800\":0,\"\\uD800\":1}".as_slice()] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut admission = native_admission(&b);
        let root = Arc::clone(&admission.root);
        let pending = point_pending(&b, &mut admission, raw);
        let result = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits()).unwrap().construct();
        assert_eq!(result.err(), Some(PointError::Input(crate::plan_input::Error::DuplicateField)));
        assert!(root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    }
    // Canonically equivalent spellings are DISTINCT decoded point keys.
    {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut admission = native_admission(&b);
        let pending = point_pending(&b, &mut admission, b"{\"\\u00e9\":0,\"e\\u0301\":1}");
        let output = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits()).unwrap().construct().unwrap();
        assert_eq!(output.payload().node(0).unwrap().immediate_child_count(), 2);
    }
    let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut admission = native_admission(&b);
    let pending = point_pending(&b, &mut admission, b"[-0,1E+09,0.0]");
    let output = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits()).unwrap().construct().unwrap();
    assert_eq!(output.payload().node(0).unwrap().point_range(), 0..14);
    assert_eq!(output.payload().node(0).unwrap().subtree_end(), 4);
    assert_eq!(output.payload().node(0).unwrap().immediate_child_count(), 3);
    for (index, range, spelling, class) in [(1, 1..3, vec![45,48], LexicalNumberClass::Integer),
        (2, 4..9, vec![49,69,43,48,57], LexicalNumberClass::Float), (3, 10..13, vec![48,46,48], LexicalNumberClass::Float)] {
        let node = output.payload().node(index).unwrap();
        assert_eq!(node.point_range(), range);
        assert_eq!(node.number_points(), Some((spelling.as_slice(), class)));
    }
}

#[test]
fn native_point_empty_malformed_lookahead_nested_closing_and_depth_boundaries() {
    use crate::plan_input::json::{quote_pending_lexical, point_lexical_control_bytes, PointError};
    for raw in [b"".as_slice(), b"\xef\xbb\xbf".as_slice(), b"\xef\xbb\xbf\xef\xbb\xbfnull".as_slice(),
        b"\"\\ud800".as_slice(), b"\"\\ud800\\u".as_slice(), b"\"\\ud800\\u00".as_slice()] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut admission = native_admission(&b);
        let root = Arc::clone(&admission.root);
        let pending = point_pending(&b, &mut admission, raw);
        let empty = pending.payload().decoded().codepoints().is_empty();
        let paid = root.payload().ledger.as_ref().unwrap().lock().unwrap().used;
        let q = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits());
        assert_eq!(q.err(), Some(PointError::Input(crate::plan_input::Error::Syntax)));
        if empty { assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, paid + 167); }
        assert!(root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
    }
    for cap in [155, 154] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut admission = native_admission(&b);
        let pending = point_pending(&b, &mut admission, b"[[]]");
        let mut limits = point_limits(); limits.work.replay = cap;
        let q = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), limits);
        if cap == 154 { assert_eq!(q.err(), Some(PointError::Input(crate::plan_input::Error::WorkLimit))); }
        else {
            let q = q.unwrap(); assert_eq!(q.resources().work.replay, 155);
            assert_eq!(q.resources().work.quotation, 239); assert_eq!(q.resources().work.total, 435);
            let output = q.construct().unwrap();
            assert_eq!(output.payload().node(0).unwrap().point_range(), 0..4);
            assert_eq!(output.payload().node(1).unwrap().point_range(), 1..3);
        }
    }
    for containers in [65, 66] {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut admission = native_admission(&b);
        let mut raw = vec![b'['; containers]; raw.extend(vec![b']'; containers]);
        let pending = point_pending(&b, &mut admission, &raw);
        let q = quote_pending_lexical(pending, b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits());
        if containers == 65 { assert!(q.unwrap().construct().is_ok()); }
        else { assert_eq!(q.err(), Some(PointError::Input(crate::plan_input::Error::DepthLimit))); }
    }
}

#[test]
fn native_point_isolated_local_string_classify_dedup_exact_and_short_paid_meters() {
    use crate::plan_input::json::{quote_pending_lexical, point_lexical_control_bytes,
        point_test_counts, PointError, PointPass, PointPassLimit};
    // Independent literals, not point_work():
    // "a": P3 S1 D1 => 1 slot + 1 dispatch + 2 partial + 3 reads + 1 write + 2 = 10.
    // [12]: two span dispatches + two integer spelling reads = 4.
    // {"a":0,"b":1}: S5 + 2E=4 + ONE comparison*(L1+3)=4 => actual 13;
    // its separately paid m^2 dedup bound remains 5 + 4*4 + 4 = 25.
    for (pass, raw, exact, prior, quotation, funded, total) in [
        (PointPass::Strings, b"\"a\"".as_slice(), 10u128, 146u64, 221u128, 10u128, 397u64),
        (PointPass::Classify, b"[12]".as_slice(), 4, 153, 239, 4, 440),
        (PointPass::Deduplicate, b"{\"a\":0,\"b\":1}".as_slice(), 13, 216, 413, 25, 710),
    ] {
        for short in [false, true] {
            let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let baseline = b.budget().used_bytes();
            let mut admission = native_admission(&b);
            let root = Arc::clone(&admission.root);
            let pending = point_pending(&b, &mut admission, raw);
            assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, prior);
            let counts = point_test_counts();
            let q = quote_pending_lexical(pending,
                b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), point_limits()).unwrap();
            assert_eq!(q.resources().work.quotation, quotation);
            assert_eq!(q.resources().work.total, u128::from(total));
            assert_eq!(match pass { PointPass::Strings => q.resources().work.strings,
                PointPass::Classify => q.resources().work.classify,
                PointPass::Deduplicate => q.resources().work.deduplicate }, funded);
            assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, prior + quotation as u64);
            let _scope = PointPassLimit::arm(pass, exact - u128::from(short));
            let result = q.construct();
            if short { assert_eq!(result.err(), Some(PointError::Input(crate::plan_input::Error::WorkLimit))); }
            else { drop(result.unwrap()); }
            // These are LATE local paid-meter failures, not preflight refusals:
            // replay and factory legitimately occurred. FULL bound stayed paid,
            // including dedup's unused m^2 comparison allowance; no root override.
            assert_eq!(point_test_counts(), (counts.0 + 2, counts.1 + 1));
            let ledger = root.payload().ledger.as_ref().unwrap().lock().unwrap();
            assert!(ledger.burned); assert_eq!(ledger.used, prior + total); drop(ledger);
            assert!(root.payload().configured.get().is_none());
            drop(admission); drop(root);
            assert_eq!(b.budget().used_bytes(), baseline);
        }
    }
}

#[test]
fn native_point_isolated_cleanup_and_construct_transition_admission_caps() {
    use crate::plan_input::json::{quote_pending_lexical, point_lexical_control_bytes,
        point_test_counts, point_test_retirement, PointError};
    // Cleanup null: 1 string slot + 1 span + six control retirements = 8.
    // Construct: C01..C12 plus explicit C13 size+alignment group = 13.
    // These test REAL admission caps, NOT a fabricated failable Drop meter or
    // a measurement of compiled closure layout/CPU operations.
    for cleanup in [true, false] {
        for short in [false, true] {
            let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let baseline = b.budget().used_bytes();
            let mut admission = native_admission(&b);
            let root = Arc::clone(&admission.root);
            let pending = point_pending(&b, &mut admission, b"null");
            assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 153);
            let mut limits = point_limits();
            if cleanup { limits.work.cleanup = 8 - u128::from(short); }
            else { limits.work.transition = 13 - u128::from(short); }
            let counts = point_test_counts();
            let q = quote_pending_lexical(pending,
                b.charge_external(point_lexical_control_bytes().unwrap()).unwrap(), limits);
            if short {
                assert_eq!(q.err(), Some(PointError::Input(crate::plan_input::Error::WorkLimit)));
                assert_eq!(point_test_counts(), (counts.0 + 1, counts.1));
                assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 153 + 239);
            } else {
                let q = q.unwrap();
                assert_eq!(q.resources().work.cleanup, 8);
                assert_eq!(q.resources().work.transition, 13);
                assert_eq!(q.resources().work.total, 416);
                let output = q.construct().unwrap();
                let covering = b.budget().used_bytes();
                assert_eq!(root.payload().ledger.as_ref().unwrap().lock().unwrap().used, 153 + 416);
                drop(output);
                assert_eq!(point_test_retirement(), Some((std::ptr::from_ref(b.budget()) as usize, covering)));
                assert_eq!(point_test_counts(), (counts.0 + 2, counts.1 + 1));
            }
            assert!(root.payload().ledger.as_ref().unwrap().lock().unwrap().burned);
            drop(admission); drop(root);
            assert_eq!(b.budget().used_bytes(), baseline);
        }
    }
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

struct PreinstallSiblingBurnGuard;

impl PreinstallSiblingBurnGuard {
    fn arm(sibling: PrimitiveAttempt) -> Self {
        PREINSTALL_SIBLING_BURN.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none());
            *slot = Some(sibling);
        });
        Self
    }
}

impl Drop for PreinstallSiblingBurnGuard {
    fn drop(&mut self) {
        let sibling = PREINSTALL_SIBLING_BURN.with(|slot| slot.borrow_mut().take());
        drop(sibling);
    }
}

#[test]
fn native_bootstrap_postfactory_sibling_burn_prevents_install() {
    PREINSTALL_SIBLING_BURN.with(|slot| assert!(slot.borrow().is_none()));
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let mut limits = limits();
    limits.max_working_payload = MAX_ALLOCATION_BYTES;
    let mut admission = native_admission(&budget);
    let actual_root = Arc::clone(&admission.root);
    let prepared = native_prepared(&budget, &mut admission, RAW, 7);
    let preconfig = 8 + u64::try_from(bootstrap_json_limits().max_work).unwrap();
    let control_bytes = OriginalSourcePlan::primitive_attempt_control_bytes().unwrap();

    // An unconfigured root has no published plan. Commit a genuine primitive
    // control, not a plan, and retain it across the later sibling failure.
    let committed_attempt = begin_attempt(
        &actual_root,
        budget.charge_external(control_bytes).unwrap(),
        3,
    ).unwrap();
    assert!(Arc::ptr_eq(&committed_attempt.root, &actual_root));
    let committed = committed_attempt.commit().unwrap();
    let committed_charge = committed.charged_bytes();
    let sibling = begin_attempt(
        &actual_root,
        budget.charge_external(control_bytes).unwrap(),
        11,
    ).unwrap();
    assert!(Arc::ptr_eq(&sibling.root, &actual_root));
    assert!(sibling.controller.as_ref().unwrap().is_for(budget.budget()));
    let prepaid = preconfig + 4 + 12;
    {
        let ledger = actual_root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert_eq!(ledger.used, prepaid);
        assert!(!ledger.burned);
        assert_eq!(ledger.limit, BOOTSTRAP_SOURCE_WORK);
    }
    assert!(actual_root.payload().configured.get().is_none());

    // Quote against the actual remaining ledger after both primitive controls
    // have been charged and debited, before arming the interleave.
    let bytes = prepared.bootstrap_bytes(limits).unwrap();
    let working = budget.charge_external(bytes).unwrap();
    let before = attempts();
    let hook = PreinstallSiblingBurnGuard::arm(sibling);
    assert!(matches!(
        admission.configure(prepared, working, limits),
        Err(Error::Burned)
    ));
    assert_eq!(attempts(), before + 1);
    // Consumption is possible only after the real factory AND census returned
    // success, immediately before installation; the counter alone is not proof.
    PREINSTALL_SIBLING_BURN.with(|slot| assert!(slot.borrow().is_none()));
    assert!(actual_root.payload().configured.get().is_none());
    let used = {
        let ledger = actual_root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert!(ledger.burned);
        assert!(ledger.used > prepaid);
        ledger.used
    };
    assert!(matches!(
        begin_attempt(&actual_root, budget.charge_external(control_bytes).unwrap(), 0),
        Err(Error::Burned)
    ));
    {
        let ledger = actual_root.payload().ledger.as_ref().unwrap().lock().unwrap();
        assert!(ledger.burned);
        assert_eq!(ledger.used, used);
    }
    assert!(committed.is_for(budget.budget()));
    assert_eq!(committed.payload_bytes(), control_bytes);
    assert_eq!(committed.charged_bytes(), committed_charge);
    assert_eq!(
        budget.budget().used_bytes(),
        baseline + actual_root.charged_bytes() + committed_charge
    );
    drop(hook);
    drop(committed);
    assert_eq!(budget.budget().used_bytes(), baseline + actual_root.charged_bytes());
    drop(actual_root);
    assert_eq!(budget.budget().used_bytes(), baseline);
}

#[test]
fn native_bootstrap_exact_raw_cap_accepts_and_one_over_burns() {
    let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let baseline = budget.budget().used_bytes();
    let mut limits = limits();
    limits.max_working_payload = MAX_ALLOCATION_BYTES;
    assert_eq!(limits.max_work, BOOTSTRAP_SOURCE_WORK);
    assert_eq!(BOOTSTRAP_CONFIG_BYTES, 16384);
    assert_eq!(bootstrap_json_limits().max_depth, 16);

    for raw_bytes in [16384, 16385] {
        let mut admission = native_admission(&budget);
        let actual_root = Arc::clone(&admission.root);
        let phase = admission.reserve_bootstrap_phase(
            budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(),
            19,
        ).unwrap();
        let raw = budget.charge_external(raw_bytes).unwrap().try_build(|_| {
            let mut raw = Vec::with_capacity(raw_bytes);
            raw.extend_from_slice(RAW);
            raw.resize(raw_bytes, b' ');
            Ok(raw)
        }).unwrap();
        let raw_backing = raw.payload().as_ptr();
        assert!(std::ptr::eq(raw.budget(), actual_root.budget()));
        let working = budget.charge_external(json::bootstrap_bytes().unwrap()).unwrap();
        assert!(working.is_for(actual_root.budget()));
        let before = attempts();
        let prepared = phase.prepare_configuration(raw, working);
        if raw_bytes == 16385 {
            assert!(matches!(prepared, Err(Error::Capacity)));
            assert_eq!(attempts(), before);
            assert!(actual_root.payload().configured.get().is_none());
            {
                let ledger = actual_root.payload().ledger.as_ref().unwrap().lock().unwrap();
                assert_eq!(ledger.used, 20);
                assert!(ledger.burned);
                assert_eq!(ledger.limit, BOOTSTRAP_SOURCE_WORK);
            }
            assert!(matches!(admission.reserve_bootstrap_phase(
                budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(),
                0,
            ), Err(Error::Burned)));
            {
                let ledger = actual_root.payload().ledger.as_ref().unwrap().lock().unwrap();
                assert!(ledger.burned);
                assert_eq!(ledger.used, 20);
            }
            assert_eq!(attempts(), before);
            drop(admission);
        } else {
            let prepared = prepared.unwrap();
            let preconfig = 20 + u64::try_from(bootstrap_json_limits().max_work).unwrap();
            assert!(Arc::ptr_eq(&prepared.attempt.root, &actual_root));
            assert!(std::ptr::eq(prepared.source.budget(), actual_root.budget()));
            assert_eq!(prepared.source.payload().raw().as_ptr(), raw_backing);
            assert_eq!(actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().used, preconfig);
            let bytes = prepared.bootstrap_bytes(limits).unwrap();
            let source = admission.configure(
                prepared,
                budget.charge_external(bytes).unwrap(),
                limits,
            ).unwrap();
            assert_eq!(attempts(), before + 1);
            assert!(Arc::ptr_eq(source.root(), &actual_root));
            assert!(std::ptr::eq(source.budget(), budget.budget()));
            assert!(actual_root.payload().configured.get().is_some());
            assert_eq!(source.raw_configuration().len(), 16384);
            assert_eq!(source.raw_configuration().as_ptr(), raw_backing);
            assert_eq!(&source.raw_configuration()[..RAW.len()], RAW);
            assert!(source.raw_configuration()[RAW.len()..].iter().all(|byte| *byte == b' '));
            {
                let ledger = actual_root.payload().ledger.as_ref().unwrap().lock().unwrap();
                assert!(!ledger.burned);
                assert!(ledger.used > preconfig);
                assert_eq!(ledger.used, source.resources().prepaid_work);
                assert_eq!(ledger.limit, BOOTSTRAP_SOURCE_WORK);
            }
            drop(source);
        }
        drop(actual_root);
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
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
    let mut limits = limits();
    limits.max_working_payload = MAX_ALLOCATION_BYTES;
    let probe = prepared(&budget, RAW, 4, limits);
    let model_work = probe.prepared.work.total_work();
    drop(probe);
    let probe = prepared(&budget, RAW, 5, limits);
    let lower_work = probe.prepared.work.total_work();
    drop(probe);
    let mut admission = native_admission(&budget);
    let actual_root = Arc::clone(&admission.root);
    let prepared = native_prepared(&budget, &mut admission, RAW, 37);
    let raw_backing = prepared.source.payload().raw().as_ptr();
    prepared.prepay_policy(13).unwrap();
    let preconfig = 38 + u64::try_from(bootstrap_json_limits().max_work).unwrap() + 13;
    assert_eq!(actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().used, preconfig);
    let bytes = prepared.bootstrap_bytes(limits).unwrap();
    let before = attempts();
    let source = admission.configure(prepared, budget.charge_external(bytes).unwrap(), limits).unwrap();
    assert_eq!(attempts(), before + 1);
    assert!(Arc::ptr_eq(source.root(), &actual_root));
    // Observation of the retained allocation, NOT a source-authority predicate.
    assert_eq!(source.raw_configuration().as_ptr(), raw_backing);
    assert_eq!(source.resources().prepaid_work, preconfig + model_work);
    assert_eq!(actual_root.payload().ledger.as_ref().unwrap().lock().unwrap().used, preconfig + model_work);
    let sibling = source.lower(budget.charge_external(source.lower_bootstrap_bytes(limits).unwrap()).unwrap(), workload(5), limits).unwrap();
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
    let mut limits = limits();
    limits.max_working_payload = MAX_ALLOCATION_BYTES;
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
            let bytes = prepared.bootstrap_bytes(limits).unwrap();
            if failure == 1 {
                assert!(matches!(admission.configure(prepared, budget.charge_external(bytes).unwrap(), limits), Err(Error::Factory)));
            } else {
                PANIC_FACTORY.with(|flag| flag.set(true));
                let working = budget.charge_external(bytes).unwrap();
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| admission.configure(prepared, working, limits))).is_err());
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
    let mut limits = limits();
    limits.max_working_payload = MAX_ALLOCATION_BYTES;
    let mut admission = native_admission(&budget);
    let prepared = native_prepared(&budget, &mut admission, RAW, 0);
    let actual_root = Arc::clone(&admission.root);
    let bytes = prepared.bootstrap_bytes(limits).unwrap();
    let sibling = begin_attempt(&actual_root,
        budget.charge_external(OriginalSourcePlan::primitive_attempt_control_bytes().unwrap()).unwrap(), 0).unwrap();
    drop(sibling);
    let before = attempts();
    assert!(matches!(admission.configure(prepared, budget.charge_external(bytes).unwrap(), limits), Err(Error::Burned)));
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
    let mut limits = limits();
    limits.max_working_payload = MAX_ALLOCATION_BYTES;
    RETIREMENT_BUDGET.with(|slot| *slot.borrow_mut() = Some(budget.try_clone().unwrap()));
    let baseline = budget.budget().used_bytes();
    let mut admission = native_admission(&budget);
    let root_charge = admission.root.charged_bytes();
    let prepared = native_prepared(&budget, &mut admission, RAW, 0);
    let bytes = prepared.bootstrap_bytes(limits).unwrap();
    let source = admission.configure(prepared, budget.charge_external(bytes).unwrap(), limits).unwrap();
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
