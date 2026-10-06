"""Same Slalom-derived verifier with the exact codec/delivery composition."""
from benchmarks.research.paper_prepared import experiment as composed
from benchmarks.research.slalom_qwen import experiment as verified


def experiment():
    components = composed().pipeline.components
    return verified().with_params(
        name="paper-qwen-slalom-composed",
        pipeline__inventory=components["inventory"],
        pipeline__linear=components["linear"],
        pipeline__delivery=components["delivery"],
    )
