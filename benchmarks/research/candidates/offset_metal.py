from benchmarks.research.combinations import build


def experiment():
    return build("search-offset-metal", topology="offset", kernel="metal")
