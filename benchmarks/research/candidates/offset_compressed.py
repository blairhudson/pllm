from benchmarks.research.combinations import build


def experiment():
    return build("search-offset-compressed", topology="offset", kernel="metal",
                 delivery="artifacts", compression="zlib")
