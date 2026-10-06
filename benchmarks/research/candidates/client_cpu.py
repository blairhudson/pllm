from benchmarks.research.combinations import build


def experiment():
    return build("search-client-cpu", topology="client")
