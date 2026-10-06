from benchmarks.research.combinations import build


def experiment():
    return build("search-compressed", delivery="artifacts", compression="zlib", indexed=True)
