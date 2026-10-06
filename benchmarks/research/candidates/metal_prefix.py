from benchmarks.research.combinations import build


def experiment():
    return build("search-metal-prefix", kernel="metal", capsule=True)
