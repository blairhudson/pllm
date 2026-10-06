from benchmarks.research.combinations import build


def experiment():
    return build("search-metal-window", kernel="metal", window=4)
