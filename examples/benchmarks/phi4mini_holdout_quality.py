"""Five additional public evaluation prompts; never used for equalization calibration."""

from __future__ import annotations

from examples.benchmarks.phi4mini_equalized_quality import equalized
from examples.benchmarks.phi4mini_reference_quality import w8a8


baseline = w8a8.with_params(budget__requests=5)
calibrated = equalized.with_params(budget__requests=5)
