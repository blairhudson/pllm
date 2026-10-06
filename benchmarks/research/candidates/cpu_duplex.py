from benchmarks.research.combinations import build


def experiment():
    return build("search-cpu-duplex", chunk=4, window=4)
