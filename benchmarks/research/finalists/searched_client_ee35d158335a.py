"""Saved PLLM search candidate; contains public configuration only."""
import json

from pllm import Experiment


def _recorded_experiment():
    value = Experiment.from_spec(json.loads('{"budget":{"max_input_tokens":256,"max_new_tokens":8,"requests":1},"deployment":{"kind":"local","root":"local://research-scorecard"},"name":"searched-client-ee35d158335a","pipeline":{"components":{"kernels":{"component":"pllm/cpu","params":{"threads":4}},"linear":{"component":"pllm/cleartext-linear","params":{}},"quantization":{"component":"pllm/symmetric-per-row-quantization/v1","params":{"activation_bits":8,"causal_reduction":"prefix_f32","weight_bits":8}},"topology":{"component":"pllm/client-only/v1","params":{}}},"model":{"revision":"7ae557604adf67be50417f59c2c2f167def9a775","source":"Qwen/Qwen2.5-0.5B-Instruct"}},"schema":"pllm.experiment.v2"}'))
    return value.with_params(pipeline__profile='baseline.client_only_cpu')


# An explicit artifact directory changes local paths and configuration identity.
def experiment():
    from benchmarks.research.artifacts import relocate_public_artifacts
    return relocate_public_artifacts(_recorded_experiment())
