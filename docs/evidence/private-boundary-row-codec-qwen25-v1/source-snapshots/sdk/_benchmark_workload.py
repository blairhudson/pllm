"""Versioned public identity witnesses, not authentication or input disclosure.

The opaque HMAC value is comparable within a privately salted cohort. Public
readers cannot verify its preimage, prove equal prompts, or authenticate a worker.
"""
from pllm._benchmark_contract import digest

WORKLOAD_SCHEMA = "pllm.response_state_audit.workload_binding.v1"
MEASUREMENT_SCHEMA = "pllm.response_state_audit.workload_measurement.v1"
PRIVATE_INPUT_SCHEMA = "pllm.benchmark.private_inputs.v1"


def require_digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid workload binding digest")
    return value


def workload_binding(public_workload, private_inputs, prompt, fixture):
    return {"schema": WORKLOAD_SCHEMA,
            "public_workload_digest": require_digest(public_workload),
            "private_inputs_commitment": require_digest(private_inputs),
            "prompt_commitment": require_digest(prompt),
            "fixture_digest": require_digest(fixture)}


def measurement_binding(target_measurement):
    return {"schema": MEASUREMENT_SCHEMA, "workload_binding_schema": WORKLOAD_SCHEMA,
            "target_measurement_digest": require_digest(target_measurement)}


def binding_digest(binding):
    return digest(binding)
