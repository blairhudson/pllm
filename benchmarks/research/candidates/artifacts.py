from benchmarks.research.combinations import build


def experiment():
    return build("search-artifacts", delivery="artifacts", indexed=True)
