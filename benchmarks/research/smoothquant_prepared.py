"""Plain W8A8 control with the paper study's exact codec/delivery composition."""
from benchmarks.research.paper_prepared import experiment as composed
from benchmarks.research.smoothquant_baseline import experiment as control


def experiment():
    parts = composed().pipeline.components
    return control().with_params(
        name="paper-qwen-w8a8-composed",
        pipeline__inventory=parts["inventory"],
        pipeline__linear=parts["linear"],
        pipeline__delivery=parts["delivery"],
    )
