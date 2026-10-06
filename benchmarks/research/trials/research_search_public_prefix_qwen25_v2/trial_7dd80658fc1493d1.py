"""Saved PLLM search candidate; contains public configuration only."""
import json

from pllm import Experiment


def experiment():
    return Experiment.from_spec(json.loads('{"budget":{"max_input_tokens":256,"max_new_tokens":8,"requests":1},"deployment":{"kind":"local","root":"local://research-scorecard"},"name":"qwen25-prepared-search","pipeline":{"components":{"cache":{"component":"pllm/client-prefix-reuse/v1","params":{"fixed_input_tokens":256,"max_bytes":33554432}},"inference":{"component":"pllm/inference","params":{}},"kernels":{"component":"pllm/cpu","params":{"threads":4}},"linear":{"component":"pllm/masked-linear","params":{}},"preparation":{"component":"pllm/model-aware-corrections","params":{}},"public_prefix":{"component":"pllm/public-prefix-capsule/v1","params":{"digest":"169c8d59966ab88263667983a180ee0d5e98dbb68acf676ddd834a502f7ea97a","path":"/private/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/research-scorecard-artifacts/prefix.msgpack","size_bytes":2967627}},"quantization":{"component":"pllm/symmetric-per-row-quantization/v1","params":{"activation_bits":8,"causal_reduction":"prefix_f32","weight_bits":8}},"tokenizer":{"component":"pllm/indexed-tokenizer/v1","params":{"digest":"48f6b4c9614323433845732098ed418d4bf7e9cb01bb07711ad72f3a9de12ffe","path":"/private/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/research-scorecard-artifacts/tokenizer"}}},"model":{"revision":"7ae557604adf67be50417f59c2c2f167def9a775","source":"Qwen/Qwen2.5-0.5B-Instruct"}},"schema":"pllm.experiment.v2"}'))
