"""Compare 0/1/2 client-owned decoder layers through the ordinary benchmark CLI.

Run `uv run pllm benchmark run --experiment examples/benchmarks/client_prefix_layers.py:baseline
--experiment examples/benchmarks/client_prefix_layers.py:prefix_one
--experiment examples/benchmarks/client_prefix_layers.py:prefix_two --max-output-tokens 1
--warmups 0 --repetitions 1 --trust-python`. Cold bundle delivery can outweigh online savings.
"""

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientPrefixLayers

_SOURCE = Model.hf(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
)
_PRECISION = SymmetricPerRow(weight_bits=8, activation_bits=8)
_BUDGET = ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=1)


def candidate(layers: int) -> Experiment:
    return Experiment(
        name=f"prepared-client-prefix-{layers}",
        pipeline=MaskedLinearCpu(
            _SOURCE, quantization=_PRECISION,
            placement=ClientPrefixLayers(layers) if layers else None,
        ),
        deployment=Deployment.local(root="local://layer-placement-comparison"),
        budget=_BUDGET,
    )


baseline = candidate(0)
prefix_one = candidate(1)
prefix_two = candidate(2)
assert len({baseline.pipeline.digest(), prefix_one.pipeline.digest(), prefix_two.pipeline.digest()}) == 3
