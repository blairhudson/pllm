"""Matched native two-worker transport choices for ordinary benchmark run."""

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.protocols import TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow

SOURCE = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")


def candidate(name, *, input_encoding="raw", output_encoding="raw"):
    return Experiment(
        name,
        TwoOnlineOffsetCpu(
            SOURCE,
            kernels=Cpu(threads=4),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            linear=TwoOnlineOffsetLinear(
                input_encoding=input_encoding, output_encoding=output_encoding
            ),
        ),
        Deployment.local(root="local://offset-transport"),
        ExecutionBudget(requests=4, max_input_tokens=64, max_new_tokens=32),
    )


raw = candidate("offset-raw")
seeded = candidate("offset-seeded", input_encoding="seeded")
row_residues = candidate("offset-row-residues", output_encoding="row_residues")
combined = candidate(
    "offset-seeded-row-residues", input_encoding="seeded", output_encoding="row_residues"
)
assert raw.pipeline.digest() != seeded.pipeline.digest()
assert seeded.resolve().client_runtime == "compiled_offset_v1"
