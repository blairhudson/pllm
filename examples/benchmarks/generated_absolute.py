"""Generated public ClientOnly W8A8 diagnostic, not paper/security qualification.

Run with the ordinary trusted Python factory and native benchmark backend::

    pllm benchmark run \
        --experiment examples/benchmarks/generated_absolute.py:candidate \
        --factory --trust-python --backend native --prompt ab \
        --max-output-tokens 2 --warmups 0 --repetitions 1 --temperature 0 \
        --capture-output-digest \
        --output target/gate1-generated-absolute-client-w8a8.json

The normal benchmark report records actual token counts: EOS may stop generation
before the two-token ceiling. This generated public checkpoint carries no privacy
or encrypted-protocol claim.
"""

from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.profiles import ClientOnlyCpu
from pllm.quantization import SymmetricPerRow
from pllm.sources.generated_absolute import tiny_checkpoint, write_checkpoint


def candidate() -> Experiment:
    """Materialize the deterministic source and declare its ordinary local run."""
    root = Path(__file__).resolve().parents[2] / "target" / "generated-absolute"
    config, tensors = tiny_checkpoint()
    checkpoint = write_checkpoint(root / "checkpoint", config, tensors)
    return Experiment(
        name="generated-absolute-client-w8a8",
        pipeline=ClientOnlyCpu(
            Model.path(str(checkpoint), model_id="tiny-absolute-position-source"),
            kernels=Cpu(threads=2),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        deployment=Deployment.local(root=str(root / "runtime")),
        budget=ExecutionBudget(requests=1, max_input_tokens=4, max_new_tokens=2),
    )
