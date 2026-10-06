from benchmarks.research.combinations import build


def experiment():
    return build("search-metal-duplex", kernel="metal", chunk=32, window=4)
