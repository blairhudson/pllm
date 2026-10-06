from benchmarks.research.combinations import build


def experiment():
    return build("search-metal-mlp", kernel="metal", roles=("mlp_gate_up", "mlp_down"))
