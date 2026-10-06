from benchmarks.research.combinations import build


def experiment():
    return build("search-cpu-prefix", capsule=True)
