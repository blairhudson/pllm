"""Broader public GKM source-model fidelity configurations, not campaign results.

Use both factories in one ordinary ``pllm benchmark quality`` invocation with
PROMPTS_FILE and DECODE_STEPS below. These constants document CLI inputs; the
Experiment does not override the quality command's options.

The installed quality adapter uses two workers and only finally accepted logits.
The independent reference is the same checkpoint's eager Transformers FP32
computation, optionally module-streamed. It is not an exact i16/Q11 oracle.
No W8A8 clear-kernel candidate, plaintext fallback, or alternate method is used.
"""

from examples.papers.gkm import experiment as _full
from examples.papers.gkm import seeded as _seeded


PROMPTS_FILE = "benchmarks/research/data/paper-quality-prompts.json"
DECODE_STEPS = 3


def experiment():
    """Pinned official checkpoint, approved full sharing and unchanged budgets."""
    return _full().with_params(name="gkm-quality-full")


def seeded():
    """Same numerical contract with the existing approved seeded-share profile."""
    return _seeded().with_params(name="gkm-quality-seeded")
