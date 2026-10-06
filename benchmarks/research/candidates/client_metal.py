from benchmarks.research.combinations import build


def experiment():
    return build("search-client-metal", topology="client", kernel="metal")
