from benchmarks.research.combinations import build


def experiment():
    return build("search-cpu8", threads=8)
