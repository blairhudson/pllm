use pllm_compiler::{resolve_experiment, BASELINE_EXPERIMENT_PROFILE};
use pllm_types::{canonical_bytes, configuration_digest_bytes};
use serde_json::{json, Value};

fn experiment() -> Value {
    json!({
        "schema": "pllm.experiment.v1",
        "name": "baseline",
        "pipeline": {
            "profile": BASELINE_EXPERIMENT_PROFILE,
            "model": {"source": "model-a"},
            "components": {
                "inference": {"component": "pllm/inference", "params": {}},
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
fn resolves_canonical_baseline_profile_and_identity() {
    let value = experiment();
    let bytes = canonical_bytes(&value);
    let resolved = resolve_experiment(&bytes).unwrap();

    assert_eq!(resolved.model(), "model-a");
    assert_eq!(
        resolved.canonical_profile(),
        canonical_bytes(&value["pipeline"])
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
    unsupported["pipeline"]["profile"] = json!("research.single_evaluator");
    assert!(resolve_experiment(&canonical_bytes(&unsupported))
        .unwrap_err()
        .contains("unsupported Experiment profile"));
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
