use pllm_compiler::resolve_experiment;
use pllm_types::{canonical_bytes, configuration_digest_bytes, pipeline_digest};
use serde_json::{json, Value};

fn experiment() -> Value {
    json!({
        "schema": "pllm.experiment.v2",
        "name": "baseline",
        "pipeline": {
            "model": {"source": "model-a"},
            "components": {
                "inference": {"component": "pllm/inference", "params": {}},
                "kernels": {"component": "pllm/cpu", "params": {"threads": 4}},
                "linear": {"component": "pllm/masked-linear", "params": {}},
                "preparation": {
                    "component": "pllm/model-aware-corrections",
                    "params": {}
                }
            }
        },
        "deployment": {"kind": "local", "root": ".pllm/baseline"},
        "budget": {"requests": 1, "max_input_tokens": 8, "max_new_tokens": 4}
    })
}

#[test]
fn resolves_canonical_baseline_composition_and_identity() {
    let value = experiment();
    let bytes = canonical_bytes(&value);
    let resolved = resolve_experiment(&bytes).unwrap();

    assert_eq!(resolved.model(), "model-a");
    assert_eq!(
        resolved.canonical_composition(),
        canonical_bytes(&value["pipeline"])
    );
    assert_eq!(
        resolved.composition_digest(),
        &pipeline_digest(&value["pipeline"])
    );
    assert_eq!(
        resolved.configuration_digest(),
        &configuration_digest_bytes(&bytes)
    );
}

#[test]
fn rejects_noncanonical_or_unsupported_experiments() {
    let value = experiment();
    let pretty = serde_json::to_vec_pretty(&value).unwrap();
    assert!(resolve_experiment(&pretty)
        .unwrap_err()
        .contains("canonical compact sorted JSON"));

    let mut unsupported = value;
    unsupported["pipeline"]["profile"] = json!("legacy.selector");
    assert!(resolve_experiment(&canonical_bytes(&unsupported))
        .unwrap_err()
        .contains("unknown field `profile`"));
}

#[test]
fn requires_baseline_component_slots_and_identities() {
    for (slot, replacement, expected) in [
        ("preparation", Value::Null, "preparation component"),
        (
            "inference",
            json!({"component": "pllm/other", "params": {}}),
            "inference component pllm/inference",
        ),
    ] {
        let mut value = experiment();
        if replacement.is_null() {
            value["pipeline"]["components"]
                .as_object_mut()
                .unwrap()
                .remove(slot);
        } else {
            value["pipeline"]["components"][slot] = replacement;
        }
        assert!(resolve_experiment(&canonical_bytes(&value))
            .unwrap_err()
            .contains(expected));
    }
}

#[test]
fn validates_and_rejects_unexecuted_gated_component_selections() {
    let mut malformed = experiment();
    malformed["pipeline"]["components"]["nonlinear"] = json!({
        "component": "pllm/r03-crt/v1",
        "params": {"unexpected": "paper-name"}
    });
    assert!(resolve_experiment(&canonical_bytes(&malformed))
        .unwrap_err()
        .contains("does not accept parameters"));

    let mut unused = experiment();
    unused["pipeline"]["components"]["nonlinear"] = json!({
        "component": "pllm/r03-crt/v1",
        "params": {}
    });
    unused["pipeline"]["components"]["nonlinear_schedule"] = json!({
        "component": "pllm/independent-lanes/v1",
        "params": {"max_elements": 4}
    });
    assert!(resolve_experiment(&canonical_bytes(&unused))
        .unwrap_err()
        .contains("unsupported masked-linear component composition"));

    unused["pipeline"]["components"]["nonlinear_schedule"] = json!({
        "component": "pllm/chunked-independent-lanes/v1",
        "params": {"max_elements": 4096}
    });
    assert!(resolve_experiment(&canonical_bytes(&unused))
        .unwrap_err()
        .contains("unsupported masked-linear component composition"));

    unused["pipeline"]["components"]["nonlinear_schedule"]["params"]["max_elements"] =
        json!(4_000_001);
    assert!(resolve_experiment(&canonical_bytes(&unused))
        .unwrap_err()
        .contains("between 5 and 4000000"));

    let mut unknown = experiment();
    unknown["pipeline"]["components"]["unused"] = json!({
        "component": "example/unused",
        "params": {}
    });
    assert!(resolve_experiment(&canonical_bytes(&unknown))
        .unwrap_err()
        .contains("unsupported masked-linear component composition"));

    let mut duplicate = experiment();
    duplicate["pipeline"]["components"]["duplicate"] = json!({
        "component": "pllm/inference",
        "params": {}
    });
    assert!(resolve_experiment(&canonical_bytes(&duplicate))
        .unwrap_err()
        .contains("unsupported masked-linear component composition"));
}

#[test]
fn resolves_only_exact_verified_composition_with_bounded_freivalds_parameters() {
    let mut verified = experiment();
    verified["pipeline"]["components"]["verification"] = json!({
        "component": "pllm/freivalds-verify/v1",
        "params": {"target_failure_bits": 40}
    });
    assert!(resolve_experiment(&canonical_bytes(&verified)).is_ok());

    verified["pipeline"]["components"]["verification"]["params"]["target_failure_bits"] = json!(81);
    assert!(resolve_experiment(&canonical_bytes(&verified))
        .unwrap_err()
        .contains("target_failure_bits from 1 to 80"));
}

#[test]
fn quantization_is_a_bound_numeric_choice_not_an_unchecked_extra_slot() {
    let baseline = experiment();
    let baseline_digest = pipeline_digest(&baseline["pipeline"]);
    for (weights, activations) in [(4, 4), (4, 8), (8, 4), (8, 8)] {
        let mut selected = baseline.clone();
        selected["pipeline"]["components"]["quantization"] = json!({
            "component": "pllm/symmetric-per-row-quantization/v1",
            "params": {"weight_bits": weights, "activation_bits": activations}
        });
        let resolved = resolve_experiment(&canonical_bytes(&selected)).unwrap();
        assert_ne!(resolved.composition_digest(), &baseline_digest);
        assert_eq!(
            resolved.composition_digest(),
            &pipeline_digest(&selected["pipeline"])
        );
        selected["pipeline"]["components"]["verification"] = json!({
            "component": "pllm/freivalds-verify/v1",
            "params": {"target_failure_bits": 40}
        });
        assert!(resolve_experiment(&canonical_bytes(&selected)).is_ok());
    }

    for invalid in [
        json!({"component": "pllm/other", "params": {"weight_bits": 4, "activation_bits": 4}}),
        json!({"component": "pllm/symmetric-per-row-quantization/v1", "params": {"weight_bits": 4, "activation_bits": 3}}),
        json!({"component": "pllm/symmetric-per-row-quantization/v1", "params": {"weight_bits": 4, "activation_bits": 8, "unused": true}}),
    ] {
        let mut selected = baseline.clone();
        selected["pipeline"]["components"]["quantization"] = invalid;
        assert!(resolve_experiment(&canonical_bytes(&selected)).is_err());
    }
    let mut unused = baseline;
    unused["pipeline"]["components"]["quantization"] = json!({
        "component": "pllm/symmetric-per-row-quantization/v1",
        "params": {"weight_bits": 8, "activation_bits": 8}
    });
    unused["pipeline"]["components"]["unrecognized"] = json!({
        "component": "pllm/other", "params": {}
    });
    assert!(resolve_experiment(&canonical_bytes(&unused)).is_err());
}
