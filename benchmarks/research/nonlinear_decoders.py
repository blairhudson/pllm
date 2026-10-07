"""Replayable CPU two-worker controls for the complete masked-wire decoder.

The fixed profile changes SiLU numerics. Only its fused/unfused variants share
a numeric comparison cohort; the clear-SiLU control retains separate quality.
"""
from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.nonlinear import MaskedWireGatedMultiplyQ12
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.protocols import ClientBundleTransport, TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def configuration(layout=None, *, composed=False, model=None):
    return Experiment(
        name=f"masked-wire-{layout or 'clear-silu'}{'-composed' if composed else ''}",
        pipeline=TwoOnlineOffsetCpu(
            model or Model.hf(MODEL, revision=REVISION),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            nonlinear=None if layout is None else MaskedWireGatedMultiplyQ12(layout=layout),
            linear=TwoOnlineOffsetLinear(**({"input_encoding": "seeded",
                "output_encoding": "row_residues", "dispatch": "seed_first"} if composed else {})),
            delivery=ClientBundleTransport("artifacts", compression="zlib", storage="paged", batch_objects=64),
        ),
        deployment=Deployment.local(root="."),
        budget=ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=1),
    )


def control():
    return configuration()


def unfused():
    return configuration("unfused")


def fused():
    return configuration("fused")


def unfused_composed():
    return configuration("unfused", composed=True)


def fused_composed():
    return configuration("fused", composed=True)
