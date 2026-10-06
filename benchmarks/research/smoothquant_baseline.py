"""Plain W8A8 control with the same causal arithmetic as channel equalization."""
from benchmarks.research.paper_baseline import experiment as control
from pllm.quantization import SymmetricPerRow


def experiment():
    return control().with_params(
        name="paper-qwen-w8a8", budget__requests=12,
        pipeline__quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
