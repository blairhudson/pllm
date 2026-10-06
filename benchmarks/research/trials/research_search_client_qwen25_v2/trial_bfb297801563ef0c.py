"""Saved PLLM search candidate; contains public configuration only."""
import json

from pllm import Experiment


def experiment():
    return Experiment.from_spec(json.loads('{"budget":{"max_input_tokens":256,"max_new_tokens":8,"requests":1},"deployment":{"kind":"local","root":"local://research-scorecard"},"name":"qwen25-client-search","pipeline":{"components":{"kernels":{"component":"pllm/cpu","params":{"threads":1}},"linear":{"component":"pllm/cleartext-linear","params":{}},"quantization":{"component":"pllm/symmetric-per-row-quantization/v1","params":{"activation_bits":8,"causal_reduction":"prefix_f32","weight_bits":8}},"topology":{"component":"pllm/client-only/v1","params":{}}},"model":{"revision":"7ae557604adf67be50417f59c2c2f167def9a775","source":"Qwen/Qwen2.5-0.5B-Instruct"}},"schema":"pllm.experiment.v2"}'))
