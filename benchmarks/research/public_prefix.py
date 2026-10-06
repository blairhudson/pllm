"""Same exact stack with 96 explicitly public prefix tokens; workload-specific."""
import json

from benchmarks.research.common import artifact_directory
from benchmarks.research.sota import experiment as exact_stack
from pllm import Experiment
from pllm.state import ClientPrefixReuse, PublicPrefixCapsule


def experiment():
    root = artifact_directory()
    published = Experiment.from_spec(json.loads((root / "capsule.json").read_text()))
    control = exact_stack()
    return control.with_params(name="prepared-public-prefix", pipeline=control.pipeline.with_params(
        cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=256),
        public_prefix=PublicPrefixCapsule(**published.pipeline.components["public_prefix"].params)))
